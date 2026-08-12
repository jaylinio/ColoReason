"""Diagnosis ICD scorer (LLM-as-a-judge).

Input: joined jsonl from prepare_downstream_evaluation.py:
  {patientId, visitId, response, targets}
  targets["诊断归一（编码）"] = list[{"code":..., "name":...}]   (gold)

Gold is read directly from preprocessed field "诊断归一（编码）" (code+name
pairs). Prediction is parsed from response["ICD预测"]["编码结果"]. The judge
returns a 0-100 score plus matched/missing/extra counts for failure analysis.
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


# ---------- response parsing (self-contained) ----------

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


# ---------- field extraction (aligned to preprocessed targets) ----------

def parse_gold_diagnosis(targets):
    """Read gold (name, code) pairs from targets["诊断归一（编码）"].

    preprocessed shape: list[{"code": ..., "name": ...}].
    Also tolerates a raw JSON string (provenance value) or legacy
    [{"诊断编码归一后":..,"诊断名称归一后":..}] dicts. Returns a de-duplicated,
    order-preserving list of (name, code) tuples.
    """
    if not isinstance(targets, dict):
        return []
    raw = targets.get("诊断归一（编码）")
    if raw in (None, "", "null", "None"):
        return []
    if isinstance(raw, str):
        try:
            items = json.loads(raw)
        except Exception:
            return []
    else:
        items = raw
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    pairs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = item.get("code") or item.get("诊断编码归一后") or ""
        name = item.get("name") or item.get("诊断名称归一后") or ""
        code = str(code).strip()
        if code:
            pairs.append((str(name).strip(), code))
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def parse_pred_codes(response_obj):
    """Parse predicted codes from response["ICD预测"]["编码结果"].

    Elements may be plain code strings or {"name","code"} dicts; unified to a
    list of (name_or_'', code) tuples.
    """
    if not isinstance(response_obj, dict):
        return []
    icd_pred = response_obj.get("ICD预测", {})
    if not isinstance(icd_pred, dict):
        return []
    raw_codes = icd_pred.get("编码结果", [])
    if not isinstance(raw_codes, list):
        raw_codes = [raw_codes]
    pairs = []
    for item in raw_codes:
        if isinstance(item, dict):
            code = item.get("code") or item.get("诊断编码归一后") or ""
            name = item.get("name") or item.get("诊断名称归一后") or ""
            code = str(code).strip()
            if code:
                pairs.append((str(name).strip(), code))
        else:
            code = str(item).strip()
            if code and code.lower() not in ("none", "null"):
                pairs.append(("", code))
    return pairs


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


def build_diagnosis_prompt(gold_pairs, pred_pairs) -> str:
    """Build the LLM-judge prompt. Gold carries name+code; pred is code-only typically."""
    gold_lines = [f"  {name} | {code}" if name else f"  {code}" for name, code in gold_pairs]
    pred_lines = [f"  {name} | {code}" if name else f"  {code}" for name, code in pred_pairs]
    gold_str = "\n".join(gold_lines) if gold_lines else "  (empty)"
    pred_str = "\n".join(pred_lines) if pred_lines else "  (empty)"

    prompt = f"""
Act as a medical coding expert (ICD-10). Compare a reference set of diagnoses
with a model-predicted set, and assign an overall integer similarity score
from 0 to 100.

Each item is given as "name | code" when a name is available, otherwise just the
code. Both sets are UNORDERED. There is no principal/secondary distinction.

Scoring guidance (overall score):
- 80-100: nearly complete clinical match; almost all codes matched.
- 60-79:  mostly matched, with limited omissions or additions.
- 40-59:  partial overlap.
- 20-39:  weak overlap or substantial errors.
- 0-19:   incorrect or not meaningfully comparable.

Matching rules:
- Match on CLINICAL EQUIVALENCE, not string equality. Treat codes that are
  parent/child or sibling within the same clinical category as a match
  (e.g., C18.7 and C18.x both = colon malignancy; M81400/3 and M8140/3 are the
  same morphology code in different notations).
- Use the Chinese name as the primary semantic signal and the code as a check.
- A code present in the reference but missing from the prediction is an
  OMISSION; a code in the prediction but absent from the reference is an
  ADDITION. Both lower the score, but omissions of clinically central codes
  weigh more.
- Do not invent codes. An empty prediction scores 0.

Output STRICTLY as a single-line JSON object, no prose, no code fences:
{{"matched": <int>, "missing": <int>, "extra": <int>, "rationale": "<one short sentence, <=20 words>", "score": <int 0-100>}}

Where matched = # reference codes matched in the prediction, missing = # reference
codes not matched, extra = # predicted codes not in the reference.

<Input>
- Reference (gold):
{gold_str}

- Predicted:
{pred_str}
</Input>
"""
    return prompt.strip()


def parse_judgement(text: str):
    """Parse the judge output. Prefer JSON; fall back to the first integer as score."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict) and "score" in obj:
                try:
                    score = max(0, min(100, int(obj.get("score"))))
                except Exception:
                    return None
                obj["score"] = score
                return obj
        except Exception:
            pass
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return {
            "score": max(0, min(100, int(m.group()))),
            "matched": None,
            "missing": None,
            "extra": None,
            "rationale": "",
        }
    except Exception:
        return None


def process_record(record):
    """Run process record. Gold from record['targets']; pred from response."""
    targets = record.get("targets", {}) or {}
    response = record.get("response", {})
    similarity = record.get("similarity", {}) or {}

    response_obj = clean_and_parse_response(response) if isinstance(response, str) else response

    gold_pairs = parse_gold_diagnosis(targets)
    pred_pairs = parse_pred_codes(response_obj if isinstance(response_obj, dict) else {})

    if not gold_pairs:
        similarity["ICD预测"] = None
        record["similarity"] = similarity
        record["judgement"] = None
        record["retry"] = False
        return record, False, False

    prompt = build_diagnosis_prompt(gold_pairs, pred_pairs)
    try:
        model_out = ask_model(prompt)
    except Exception:
        similarity["ICD预测"] = None
        record["similarity"] = similarity
        record["judgement"] = None
        record["retry"] = True
        return record, True, False

    judgement = parse_judgement(model_out)
    parse_fail = judgement is None
    similarity["ICD预测"] = judgement["score"] if judgement else None
    record["similarity"] = similarity
    record["judgement"] = judgement
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
    print(f"Loaded {total} records; starting multithreaded diagnosis judging...")

    model_fail = 0
    parse_fail = 0
    skipped_no_gold = 0
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_record, records[i]): i for i in range(total)
        }

        for future in tqdm(
            as_completed(future_map), total=total, desc="Diag Judging"
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
            if rec.get("similarity", {}).get("ICD预测") is None and not rec.get("retry"):
                skipped_no_gold += 1

            results[idx] = rec

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print("\n====== Diagnosis judging complete ======")
    print(f"Total records: {total}")
    print(f"Skipped (no gold diagnosis): {skipped_no_gold}")
    print(f"Model call failures: {model_fail}")
    print(f"Score parse failures: {parse_fail}")
    print(f"Wrote JSONL file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score diagnosis ICD predictions with an LLM-as-a-judge.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the joined input JSONL file"
    )
    parser.add_argument("--max_workers", type=int, default=2, help="Thread pool size")
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    input_dir = os.path.dirname(INPUT_FILE)
    input_filename = os.path.basename(INPUT_FILE)
    filename_without_ext, file_ext = os.path.splitext(input_filename)
    new_filename = f"{filename_without_ext}_diag_new{file_ext}"
    OUTPUT_FILE = os.path.join(input_dir, new_filename)

    process_jsonl_multithread(INPUT_FILE, OUTPUT_FILE, max_workers=args.max_workers)
