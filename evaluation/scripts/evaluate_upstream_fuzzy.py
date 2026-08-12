import json
import re
from fuzzywuzzy import fuzz


def clean_response(response_str):
    """Run clean response."""
    response_str = response_str.replace("<|channel|>analysis<|message|>", "<think>")
    response_str = response_str.replace(
        "<|end|><|start|>assistant<|channel|>final<|message|>", "</think>"
    )

    cleaned_response = re.sub(
        r"<think>.*?</think>", "", response_str, flags=re.DOTALL
    ).strip()

    match = re.search(r"\{.*\}", cleaned_response, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return cleaned_response


def compare_values(val1, val2):
    """Run compare values."""
    if val1 == "":
        return None
    if isinstance(val1, dict) and isinstance(val2, dict):
        nested_similarity = {}
        for key in val1:
            nested_similarity[key] = compare_values(val1[key], val2.get(key, ""))
        return nested_similarity
    return fuzz.ratio(str(val1), str(val2))


PLACEHOLDERS = {"未提及"}


def _normalize_tok(s: str) -> str:
    s = (s or "").strip().strip("\"'")

    if s in PLACEHOLDERS:
        return ""

    s = s.replace("－", "-").replace("–", "-").replace("—", "-")
    return s.strip()


def _to_list_keep_order(x):
    """Run  to list keep order."""
    if isinstance(x, list):
        items = [str(i) for i in x]
    elif isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                items = [str(i) for i in arr] if isinstance(arr, list) else [s]
            except Exception:
                items = re.split(r"[，,、;；/\n]+", s)
        else:
            items = re.split(r"[，,、;；/\n]+", s)
    else:
        items = [str(x)]

    seen, out = set(), []
    for it in items:
        t = _normalize_tok(it)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _pairwise_best_avg_one_direction(a_list, b_list):
    """Run  pairwise best avg one direction."""
    if not a_list or not b_list:
        return None
    scores = []
    for a in a_list:
        best = max((fuzz.ratio(a, b) for b in b_list), default=0)
        scores.append(best)
    return round(sum(scores) / len(scores), 2) if scores else None


# ============================================================


def process_jsonl(input_file):
    """Run process jsonl."""
    output_data = []
    SPECIAL_KEYS = {
        "免疫组化标志物",
        "免疫组化标志物_定量结果",
        "免疫组化标志物_定性结果",
    }

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line.strip())
            raw_response = data.get("response", "")
            raw_answer = data.get("answer", "")

            cleaned_response = clean_response(raw_response)

            similarity = {}
            if isinstance(cleaned_response, dict) and isinstance(raw_answer, dict):
                for key in cleaned_response:
                    if key in SPECIAL_KEYS:
                        resp_list = _to_list_keep_order(cleaned_response.get(key, ""))
                        ans_list = _to_list_keep_order(raw_answer.get(key, ""))

                        similarity[key] = _pairwise_best_avg_one_direction(
                            ans_list, resp_list
                        )
                    else:
                        similarity[key] = compare_values(
                            cleaned_response[key], raw_answer.get(key, "")
                        )

            output_data.append(
                {
                    "response": cleaned_response,
                    "answer": raw_answer,
                    "metadata": data.get("metadata", {}),
                    "similarity": similarity,
                }
            )

    output_file = input_file.replace(".jsonl", "_processed.jsonl")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in output_data:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write("\n")
    print(f"Processed file saved to: {output_file}")


if __name__ == "__main__":
    input_file = ""
    process_jsonl(input_file)
