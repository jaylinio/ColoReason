"""30-day readmission scorer (deterministic rule).

Input: joined jsonl from prepare_downstream_evaluation.py:
  {patientId, visitId, response, targets}
  targets["30天再入院"] = dict with key "within_30_days" (bool)   (gold)

Gold label is read directly from the preprocessed field
targets["30天再入院"]["within_30_days"] — no timestamp recomputation needed.
Prediction is parsed from response["再入院概率"]["概率结果"] and thresholded.
Score is 100 on label match, 0 otherwise, None if either side is unknown.
"""
import json
import os
import re
import argparse


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


# ---------- readmission scoring (rule-based, deterministic) ----------

def to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def classify_readmission(prob, threshold=0.5):
    """Threshold the predicted readmission probability into a binary label."""
    pf = to_float_or_none(prob)
    if pf is None:
        return None
    return pf >= threshold


def gold_readmission_label(targets):
    """Read the gold 30-day readmission label from preprocessed targets.

    targets["30天再入院"]["within_30_days"] is the canonical boolean label
    materialized by build_inhouse_key_values.py. Returns True/False/None.
    """
    if not isinstance(targets, dict):
        return None
    rec = targets.get("30天再入院")
    if not isinstance(rec, dict):
        return None
    label = rec.get("within_30_days")
    if isinstance(label, bool):
        return label
    return None


def compare_readmission(response_obj, targets, threshold=0.5):
    """Compare predicted vs gold readmission label; 100 on match, 0 otherwise."""
    pred_label = None
    if isinstance(response_obj, dict):
        readmit = response_obj.get("再入院概率", {})
        pred_prob = readmit.get("概率结果", None) if isinstance(readmit, dict) else None
        pred_label = classify_readmission(pred_prob, threshold=threshold)
    gold_label = gold_readmission_label(targets)
    if pred_label is not None and gold_label is not None:
        return 100 if (pred_label == gold_label) else 0
    return None


def process_record(record, threshold=0.5):
    """Run process record. Gold from record['targets']; pred from response."""
    targets = record.get("targets", {}) or {}
    response = record.get("response", {})
    similarity = record.get("similarity", {}) or {}

    response_obj = clean_and_parse_response(response) if isinstance(response, str) else response
    similarity["再入院概率"] = compare_readmission(
        response_obj if isinstance(response_obj, dict) else {},
        targets,
        threshold=threshold,
    )
    record["similarity"] = similarity
    return record


def process_jsonl(input_path: str, output_path: str, threshold: float = 0.5):
    """Run process jsonl (single-pass, deterministic, no model calls)."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    valid = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            rec = process_record(rec, threshold=threshold)
            if rec.get("similarity", {}).get("再入院概率") is not None:
                valid += 1
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print("====== Readmission scoring complete ======")
    print(f"Total records: {total}")
    print(f"Valid comparisons (both labels known): {valid}")
    print(f"Wrote JSONL file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score 30-day readmission by deterministic rule.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the joined input JSONL file")
    parser.add_argument("--threshold", type=float, default=0.35, help="Probability threshold for the predicted label")
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    input_dir = os.path.dirname(INPUT_FILE)
    input_filename = os.path.basename(INPUT_FILE)
    filename_without_ext, file_ext = os.path.splitext(input_filename)
    OUTPUT_FILE = os.path.join(input_dir, f"{filename_without_ext}_readmit_new{file_ext}")

    process_jsonl(INPUT_FILE, OUTPUT_FILE, threshold=args.threshold)
