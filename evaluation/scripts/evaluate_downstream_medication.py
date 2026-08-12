"""Medication recommendation scorer (LLM-as-a-judge).

Input: joined jsonl from prepare_downstream_evaluation.py:
  {patientId, visitId, response, targets}
  targets["药物"] = list[str]   (gold, ATC 第五级中文名称)

Gold is read directly from the preprocessed field "药物" (list[str]).
Prediction is parsed from response["药物推荐"]["推荐结果"]. The judge returns
a single 0-100 integer similarity score.

Changes vs the original script (field alignment to preprocessed):
  - gold source: answer["化疗药物"] (str) -> targets["药物"] (list[str])
  - response may now be a raw string (prepare output); clean_and_parse_response
    is applied so response["药物推荐"] can be read.
  - argparse moved under __main__ for testability.
"""
import json
import os
import re
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import argparse


API_URL = os.environ.get("COLOREASON_API_URL", "")
API_KEY = os.environ.get("COLOREASON_API_KEY", "")
API_MODEL = os.environ.get("COLOREASON_API_MODEL", "")
# ==================================


# ---------- response parsing (self-contained, shared shape) ----------

def strip_think_tags(s: str) -> str:
    """Run strip think tags."""
    if not isinstance(s, str):
        return ""
    s = s.replace("<|channel|>analysis<|message|>", "<think>")
    s = s.replace("<|end|><|start|>assistant<|channel|>final<|message|>", "</think>")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    return s.strip()


def remove_code_fences(s: str) -> str:
    """Run remove code fences."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    if (s.startswith("'''") and s.endswith("'''")) or (
        s.startswith('"""') and s.endswith('"""')
    ):
        s = s[3:-3].strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1].strip()
    return s


def clean_bad_fields(s: str) -> str:
    """Run clean bad fields."""
    lines = s.splitlines()
    cleaned_lines = []
    for ln in lines:
        if '"原文出处"' in ln or '\\"原文出处\\"' in ln:
            continue
        cleaned_lines.append(ln)
    fixed_lines = []
    for i, ln in enumerate(cleaned_lines):
        stripped = ln.strip()
        if '"理由"' in stripped and stripped.endswith(","):
            if i + 1 < len(cleaned_lines):
                next_line = cleaned_lines[i + 1].strip()
                if next_line.startswith("}") or next_line.startswith("]"):
                    ln = ln.rstrip(",")
        fixed_lines.append(ln)
    return "\n".join(fixed_lines)


def clean_and_parse_response(response_str: str):
    """Parse a raw model response string into a dict (or str fallback)."""
    s = strip_think_tags(response_str or "")
    s = remove_code_fences(s)
    s = clean_bad_fields(s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return candidate
    return s


# ---------- LLM-as-a-judge ----------

def ask_model(prompt: str) -> str:
    """Run ask model."""
    if not API_URL or not API_KEY or not API_MODEL:
        raise ValueError(
            "Downstream judge needs COLOREASON_API_URL, COLOREASON_API_KEY, "
            "and COLOREASON_API_MODEL environment variables to call the LLM."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "thinking": {"type": "enabled"},
    }

    resp = requests.post(API_URL, headers=headers, data=json.dumps(data), verify=False)
    print(resp.status_code)
    resp.raise_for_status()
    try:
        resp_json = resp.json()
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return resp.text


def build_chemo_prompt(gold_str: str, pred_list) -> str:
    """Run build chemo prompt. gold_str is a serialized reference list."""
    if pred_list is None:
        pred_list = []
    if not isinstance(pred_list, list):
        pred_list = [pred_list]

    pred_str = json.dumps(pred_list, ensure_ascii=False)
    gold_str_safe = gold_str if gold_str is not None else ""

    prompt = f"""
Act as a medical expert in pharmacotherapy. Compare a reference medication list
with a model-predicted medication list and assign an integer similarity score
from 0 to 100.

Scoring guidance:
- 80-100: nearly complete clinical match.
- 60-79: mostly correct with limited omissions or additions.
- 40-59: partial overlap.
- 20-39: weak overlap or substantial errors.
- 0-19: incorrect or not meaningfully comparable.

Treat generic names, brand names, minor spelling variants, and clinically
equivalent medication names as matches when appropriate. Do not invent or
modify medications. Output only one integer from 0 to 100.

<Input>
- Reference medications:
  {gold_str_safe}

- Predicted medications:
  {pred_str}
</Input>
"""
    return prompt.strip()


def parse_score_from_model_output(text: str):
    """Run parse score from model output."""
    if not isinstance(text, str):
        return None

    s = text.strip()

    try:
        v = int(s)
        if 0 <= v <= 100:
            return v
    except Exception:
        pass

    m = re.search(r"\d+", s)
    if not m:
        return None

    try:
        v = int(m.group())
        return max(0, min(100, v))
    except Exception:
        return None


def process_record(record):
    """Run process record. Gold from record['targets']['药物']; pred from response."""
    targets = record.get("targets", {}) or {}
    response = record.get("response", {})
    similarity = record.get("similarity", {}) or {}

    drug_gold = targets.get("药物", [])
    if not drug_gold:
        similarity["药物推荐"] = None
        record["similarity"] = similarity
        record["retry"] = False
        return record, False, False

    response_obj = clean_and_parse_response(response) if isinstance(response, str) else response
    if not isinstance(response_obj, dict):
        similarity["药物推荐"] = None
        record["similarity"] = similarity
        record["retry"] = False
        return record, False, False

    pred_list = (response_obj.get("药物推荐", {}) or {}).get("推荐结果", [])
    gold_str = json.dumps(drug_gold, ensure_ascii=False)
    prompt = build_chemo_prompt(gold_str, pred_list)

    try:
        model_out = ask_model(prompt)
    except Exception:
        similarity["药物推荐"] = None
        record["similarity"] = similarity
        record["retry"] = True
        return record, True, False

    score = parse_score_from_model_output(model_out)
    parse_fail = score is None

    similarity["药物推荐"] = score
    record["similarity"] = similarity

    record["retry"] = True if parse_fail else False

    return record, False, parse_fail


def process_jsonl_multithread(input_path: str, output_path: str, max_workers: int = 2):
    """Run process jsonl multithread."""

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    print(f"{input_path}")
    print(f"Loaded {total} records; starting multithreaded processing...")

    model_fail = 0
    parse_fail = 0
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_record, records[i]): i for i in range(total)
        }

        for future in tqdm(
            as_completed(future_map), total=total, desc="Pass1 Processing"
        ):
            idx = future_map[future]
            try:
                rec, mf, pf = future.result()
            except Exception:
                rec, mf, pf = records[idx], True, True

            if mf:
                model_fail += 1
            if pf:
                parse_fail += 1

            results[idx] = rec

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print("\n====== Pass 1 complete ======")
    print(f"Total records: {total}")
    print(f"Model call failures: {model_fail}")
    print(f"Score parse failures: {parse_fail}")
    print(f"Wrote JSONL file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file with model scoring.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file"
    )
    parser.add_argument("--max_workers", type=int, default=2, help="Thread pool size")
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    input_dir = os.path.dirname(INPUT_FILE)
    input_filename = os.path.basename(INPUT_FILE)
    filename_without_ext, file_ext = os.path.splitext(input_filename)
    new_filename = f"{filename_without_ext}_new{file_ext}"
    OUTPUT_FILE = os.path.join(input_dir, new_filename)

    process_jsonl_multithread(INPUT_FILE, OUTPUT_FILE, max_workers=args.max_workers)
