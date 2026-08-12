"""Join preprocessed downstream gold (targets) with model inference responses.

Gold source: preprocessed/inhouse/downstream_test_key_values.jsonl
  each row = {patientId, visitId, registered_fields, targets, provenance}
  targets keys: "诊断归一（编码）" (list[{code,name}]),
                "30天再入院" (dict, incl. within_30_days bool),
                "药物" (list[str])

Responses source: a model inference jsonl, each row carrying (patientId,
visitId) and a response text (field "response" preferred; falls back to
"output"/"answer" or the last assistant message in "messages").

Join key: (patientId, visitId). Output one record per matched visit:
  {patientId, visitId, response, targets}
targets is passed through verbatim so downstream evaluators read the exact
preprocessed field names ("诊断归一（编码）" / "30天再入院" / "药物").
"""
import json
import os
import argparse


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_response(rec):
    """Pull the model response text from an inference record.

    Tries plain string fields first, then falls back to the last assistant
    message in a chat-style "messages" list.
    """
    for key in ("response", "output", "pred", "answer"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return ""


def normalize_id(v):
    return str(v).strip() if v is not None else ""


def main():
    parser = argparse.ArgumentParser(
        description="Join preprocessed downstream targets with model responses by (patientId, visitId)."
    )
    parser.add_argument(
        "--key-values",
        required=True,
        help="preprocessed/inhouse/downstream_test_key_values.jsonl (gold targets)",
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="model inference output jsonl (must carry patientId/visitId + response)",
    )
    parser.add_argument("--output", required=True, help="path to the joined output jsonl")
    args = parser.parse_args()

    gold = read_jsonl(args.key_values)
    responses = read_jsonl(args.responses)

    resp_map = {}
    for r in responses:
        pid = normalize_id(r.get("patientId"))
        vid = normalize_id(r.get("visitId", r.get("visit_id")))
        if pid and vid:
            resp_map[(pid, vid)] = extract_response(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    matched = 0
    missing = 0
    multi_target = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for g in gold:
            pid = normalize_id(g.get("patientId"))
            vid = normalize_id(g.get("visitId"))
            response = resp_map.get((pid, vid))
            if response is None:
                missing += 1
                continue
            targets = g.get("targets", {}) or {}
            if targets:
                multi_target += 1
            rec = {
                "patientId": pid,
                "visitId": vid,
                "response": response,
                "targets": targets,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            matched += 1

    print("====== prepare downstream complete ======")
    print(f"gold records: {len(gold)}")
    print(f"response records: {len(responses)} (unique (patientId,visitId): {len(resp_map)})")
    print(f"matched: {matched}")
    print(f"gold without response: {missing}")
    print(f"matched with non-empty targets: {multi_target}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
