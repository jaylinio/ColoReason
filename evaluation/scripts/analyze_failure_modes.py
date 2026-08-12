#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict


BASE = os.environ.get("FM_BASE", "")
OUT_DIR = os.environ.get("FM_OUT_DIR", os.path.join(BASE, "failure_mode_outputs"))
MODEL = os.environ.get("FM_MODEL", "")


def cfg_path(name, default):
    return os.environ.get(name, default)


UPSTREAM_RAW = cfg_path(
    "FM_UPSTREAM_RAW", os.path.join(BASE, "model_response_file", f"{MODEL}.jsonl")
)
UPSTREAM_FUZZY = cfg_path(
    "FM_UPSTREAM_FUZZY",
    os.path.join(BASE, "model_output_fuzzy2", MODEL, f"{MODEL}_processed.jsonl"),
)
UPSTREAM_SEM = cfg_path(
    "FM_UPSTREAM_SEM",
    os.path.join(
        BASE, "model_output_semantics2", MODEL, f"{MODEL}_processed_emb.jsonl"
    ),
)

DOWNSTREAM_RAW = cfg_path(
    "FM_DOWNSTREAM_RAW",
    os.path.join(BASE, "downstream_visit_response", f"{MODEL}.jsonl"),
)
DOWNSTREAM_PROC = cfg_path(
    "FM_DOWNSTREAM_PROC",
    os.path.join(
        BASE, "downstream_visit_result", MODEL, f"{MODEL}_processed_new.jsonl"
    ),
)
DOWNSTREAM_STRUCT_RAW = cfg_path(
    "FM_DOWNSTREAM_STRUCT_RAW",
    os.path.join(BASE, "downstream_visit_structure_response", f"{MODEL}.jsonl"),
)
DOWNSTREAM_STRUCT_PROC = cfg_path(
    "FM_DOWNSTREAM_STRUCT_PROC",
    os.path.join(
        BASE, "downstream_visit_structure_result", MODEL, f"{MODEL}_processed_new.jsonl"
    ),
)
SOURCE_NOTE = os.environ.get("FM_SOURCE_NOTE", "")

CSV_COLUMNS = [
    "case_id_hash",
    "entity_id_hash",
    "record_id_hash",
    "level",
    "field",
    "task",
    "question",
    "prediction",
    "reference",
    "is_correct",
    "model_reasoning",
    "source_evidence",
    "evidence_source_type",
    "evidence_time",
    "prediction_time",
    "reference_time",
    "failure_mode",
    "severity",
    "review_action",
    "model_error_or_data_limitation",
    "adjudicator_id",
    "needs_second_review",
    "note",
]

HIGH_IMPACT_FIELDS = {
    "TNM分期_T",
    "TNM分期_N",
    "TNM分期_M",
    "TNM分期_总分期",
    "诊断归一",
    "是否支持病理确诊原发结肠癌",
    "是否支持病理确诊原发直肠癌",
    "是否支持病理确诊原发直肠乙状结肠交界部癌",
    "有无脉管癌栓",
    "有无侵犯神经",
    "阳性淋巴结数目",
    "送检淋巴结是否阳性",
    "浸润深度",
    "ICD预测",
    "再入院概率",
    "药物推荐",
}

NORMALIZATION_FIELDS = {
    "诊断归一",
    "TNM分期_T",
    "TNM分期_N",
    "TNM分期_M",
    "TNM分期_总分期",
    "组织学分型信息",
    "免疫组化标志物",
    "免疫组化标志物_定量结果",
    "免疫组化标志物_定性结果",
    "ICD预测",
}

TEMPORAL_FIELDS = {"就诊日期", "再入院就诊日期", "再入院概率"}
UNKNOWN_VALUES = {
    "",
    "未提及",
    "未知",
    "不详",
    "nan",
    "None",
    "null",
    "YT_MISSING",
    "Nx",
    "Mx",
    "Tx",
}


def stable_hash(value):
    if value is None:
        value = ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def as_text(value, max_len=None):
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    if max_len and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def is_unknown(value):
    text = as_text(value).strip()
    return text in UNKNOWN_VALUES


def normalize_dates(text):
    text = re.sub(r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?", "[DATE]", text)
    text = re.sub(r"\d{4}[-/年.]\d{1,2}月?", "[MONTH]", text)
    text = re.sub(r"\d{13}\.0", "[TIMESTAMP_MS]", text)
    return text


def mask_sensitive(text):
    text = as_text(text)
    text = normalize_dates(text)
    text = re.sub(r"\b\d{7,}\b", "[NUM]", text)
    return text


def extract_between(text, start, end):
    if not text:
        return ""
    s = text.find(start)
    if s < 0:
        return ""
    s += len(start)
    e = text.find(end, s)
    if e < 0:
        return text[s:]
    return text[s:e]


def extract_final_json(response_text):
    text = response_text or ""
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.split(marker)[-1]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text.strip()


def extract_reasoning(response_text):
    text = response_text or ""
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.split(marker)[0]
    text = text.replace("<|channel|>analysis<|message|>", "")
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    return mask_sensitive(text)[:700]


def field_keywords(field, reference, prediction):
    words = [field]
    if field.startswith("TNM分期"):
        words += ["TNM", "分期", "pT", "T3", "N0", "M0"]
    if "诊断" in field or field == "ICD预测":
        words += [
            "诊断",
            "ICD",
            "编码",
            "ref_diagnose",
            "ref_diagnose_code",
            "admission_diagnosis",
            "出院诊断",
        ]
    if "化疗" in field or field == "药物推荐":
        words += ["化疗", "药物", "用药", "治疗", "医嘱"]
    if "呕吐" in field:
        words += ["呕吐", "恶心"]
    if "黑便" in field or "便血" in field:
        words += ["黑便", "便血", "血便", "大便带血", "血染"]
    if "腹胀" in field:
        words += ["腹胀"]
    if "体重" in field:
        words += ["体重"]
    if "溃疡" in field:
        words += ["溃疡"]
    if "息肉" in field:
        words += ["息肉"]
    if "大便困难" in field:
        words += ["大便困难", "便秘"]
    if "神经" in field:
        words += ["神经"]
    if "脉管" in field:
        words += ["脉管", "癌栓"]
    if "淋巴结" in field:
        words += ["淋巴结"]
    if field == "再入院概率":
        words += ["再入院", "入院", "出院", "手术", "风险", "就诊日期"]
    for value in [reference, prediction]:
        text = as_text(value)
        for token in re.findall(
            r"[A-Za-z]\d+[A-Za-z]?|[A-Z]\d{2}(?:\.\w+)?|[\u4e00-\u9fff]{2,8}", text
        ):
            if len(words) < 25:
                words.append(token)
    seen = []
    for w in words:
        if w and w not in seen:
            seen.append(w)
    return seen


def evidence_snippet(prompt, field, reference, prediction, structured=False):
    if structured:
        source = prompt
    else:
        source = extract_between(prompt, "<Text>", "</Text>") or prompt
    source = source.replace("\r", "\n")
    chunks = []
    for kw in field_keywords(field, reference, prediction):
        idx = source.find(kw)
        if idx >= 0:
            start = max(0, idx - 120)
            end = min(len(source), idx + len(kw) + 180)
            snippet = source[start:end]
            chunks.append(snippet)
        if len(chunks) >= 3:
            break
    if not chunks and structured:
        chunks = [source[-1200:]]
    if not chunks:
        return "source evidence unavailable in prompt after keyword search"
    joined = " ... ".join(chunks)
    return mask_sensitive(joined)[:1200]


def extract_question(prompt, level, field):
    if level == "record":
        field_block = extract_between(prompt, "<Field>", "</Field>")
        return f"Extract structured field `{field}` from one EHR record. Field set: {mask_sensitive(field_block)[:500]}"
    if level == "downstream":
        return f"Downstream visit task `{field}` from {'structured' if '结构化' in prompt[:100] else 'unstructured'} visit input."
    return f"Task field `{field}`."


def extract_time_from_answer(answer, key="就诊日期"):
    if isinstance(answer, dict):
        value = answer.get(key, "")
        if value:
            return "timestamp_ms_present"
    return ""


def failure_mode_for(field, pred, ref, sim, level, response_obj=None):
    if sim == "unparsable":
        return "format_parsing_error"
    pred_unknown = is_unknown(pred)
    ref_unknown = is_unknown(ref)
    if ref_unknown and not pred_unknown:
        return "unsupported_completion"
    if not ref_unknown and pred_unknown:
        return "missing_evidence"
    if field in TEMPORAL_FIELDS:
        return "temporal_misassignment"
    if field in NORMALIZATION_FIELDS:
        return "normalization_error"
    if level == "downstream" and field == "药物推荐":
        return "unsupported_completion"
    return (
        "schema_mismatch"
        if isinstance(pred, (dict, list)) and isinstance(ref, str)
        else "normalization_error"
    )


def review_action_for(mode):
    return {
        "missing_evidence": "retain_unknown",
        "conflicting_evidence": "retain_conflict",
        "temporal_misassignment": "review_timeline",
        "normalization_error": "correct_normalization",
        "unsupported_completion": "block_automatic_reuse",
        "schema_mismatch": "repair_schema_or_format",
        "format_parsing_error": "repair_schema_or_format",
        "reference_ambiguity": "adjudicate_reference",
    }.get(mode, "block_automatic_reuse")


def severity_for(field, mode, level):
    if mode in {"format_parsing_error"}:
        return "moderate"
    if field in HIGH_IMPACT_FIELDS or level == "downstream":
        return "high"
    if mode in {"unsupported_completion", "temporal_misassignment"}:
        return "high"
    return "moderate"


def is_bad_score(score, semantic_score=None):
    if score is None:
        return False
    if isinstance(score, (int, float)):
        if math.isnan(score):
            return False
        return score < 100
    return False


def correctness(score, pred, ref):
    if score == "unparsable":
        return "unparsable"
    if score is None:
        return "unknown"
    if isinstance(score, (int, float)):
        if score >= 100:
            return "true"
        if score > 0:
            return "partial"
        return "false"
    if as_text(pred) == as_text(ref):
        return "true"
    return "unknown"


def make_row(
    level,
    field,
    task,
    idx,
    processed,
    raw,
    sim_score,
    semantic_score=None,
    structured_prompt=False,
    forced_unknown=False,
):
    meta = processed.get("metadata") or {}
    patient = meta.get("patientId", "")
    visit = meta.get("visitId", "")
    pred = (
        (processed.get("response") or {}).get(field, "")
        if isinstance(processed.get("response"), dict)
        else processed.get("response")
    )
    ref = (
        (processed.get("answer") or {}).get(field, "")
        if isinstance(processed.get("answer"), dict)
        else processed.get("answer")
    )
    if level == "downstream":
        pred = (
            (processed.get("response") or {}).get(field, "")
            if isinstance(processed.get("response"), dict)
            else processed.get("response", "")
        )
        ref = processed.get("answer") or {}
    prompt = ""
    reasoning = ""
    if raw:
        msgs = raw.get("messages") or []
        prompt = msgs[0].get("content", "") if msgs else ""
        reasoning = extract_reasoning(raw.get("response", ""))
    if sim_score == "unparsable":
        pred = extract_final_json(raw.get("response", "")) if raw else pred
    mode = failure_mode_for(
        field, pred, ref, sim_score, level, processed.get("response")
    )
    sev = severity_for(field, mode, level)
    is_corr = correctness(sim_score, pred, ref)
    note_parts = []
    if semantic_score is not None:
        note_parts.append(f"semantic_similarity={semantic_score}")
    if sim_score is not None:
        note_parts.append(f"fuzzy_or_task_similarity={sim_score}")
    if forced_unknown:
        note_parts.append(
            "candidate retained because downstream metric was unavailable or LLM-scored; requires human review"
        )
    if field == "药物推荐":
        note_parts.append(
            "reference file contains source structured fields, not a direct gold medication list"
        )
    if mode == "missing_evidence" and is_unknown(pred):
        note_parts.append("prediction blank while reference populated")
    evidence = evidence_snippet(prompt, field, ref, pred, structured=structured_prompt)
    case_basis = f"{task}|{idx}|{patient}|{visit}|{field}"
    return {
        "case_id_hash": stable_hash(case_basis),
        "entity_id_hash": stable_hash(patient) if patient else "",
        "record_id_hash": stable_hash(visit) if visit else "",
        "level": level,
        "field": field,
        "task": f"{MODEL}:{task}",
        "question": extract_question(prompt, level, field),
        "prediction": mask_sensitive(pred)[:1500],
        "reference": mask_sensitive(ref)[:1500],
        "is_correct": "unknown" if forced_unknown else is_corr,
        "model_reasoning": reasoning,
        "source_evidence": evidence,
        "evidence_source_type": "structured table"
        if structured_prompt
        else (
            "source document"
            if level == "record"
            else "source document / evaluation file"
        ),
        "evidence_time": "relative visit/order available"
        if level == "downstream"
        else extract_time_from_answer(processed.get("answer")),
        "prediction_time": "timestamp/redacted if present in model output"
        if field in TEMPORAL_FIELDS
        else "",
        "reference_time": extract_time_from_answer(processed.get("answer"))
        if level in {"record", "downstream"}
        else "",
        "failure_mode": mode,
        "severity": sev,
        "review_action": review_action_for(mode),
        "model_error_or_data_limitation": "undetermined"
        if forced_unknown
        else (
            "model_error" if mode != "reference_ambiguity" else "reference_ambiguity"
        ),
        "adjudicator_id": "",
        "needs_second_review": "true" if sev in {"high", "critical"} else "false",
        "note": "; ".join(
            [p for p in note_parts + ([SOURCE_NOTE] if SOURCE_NOTE else []) if p]
        ),
    }


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_raw_by_index(path):
    return list(iter_jsonl(path))


def build_upstream_rows(max_per_field=8):
    rows = []
    raw_rows = load_raw_by_index(UPSTREAM_RAW)
    sem_iter = iter_jsonl(UPSTREAM_SEM)
    counters = Counter()
    total_candidates = 0
    for idx, proc in enumerate(iter_jsonl(UPSTREAM_FUZZY)):
        sem = next(sem_iter, None)
        raw = raw_rows[idx] if idx < len(raw_rows) else None
        sims = proc.get("similarity") or {}
        sem_sims = (sem or {}).get("similarity") or {}
        answer_obj = proc.get("answer") if isinstance(proc.get("answer"), dict) else {}
        response_obj = (
            proc.get("response") if isinstance(proc.get("response"), dict) else {}
        )
        fields = list(answer_obj.keys())
        if not fields and proc.get("metadata", {}).get("field"):
            fields = [proc.get("metadata", {}).get("field")]
        for field in fields:
            pred = response_obj.get(field, proc.get("response", ""))
            ref = answer_obj.get(field, proc.get("answer", ""))
            score = sims.get(field)
            sem_score = sem_sims.get(field)
            if not isinstance(proc.get("response"), dict):
                score = "unparsable"
            bad = is_bad_score(score, sem_score)
            unsafe = is_unknown(ref) and not is_unknown(pred)
            missing = (not is_unknown(ref)) and is_unknown(pred)
            low_sem = (
                isinstance(sem_score, (int, float))
                and sem_score < 80
                and not is_unknown(ref)
            )
            unparsable = score == "unparsable"
            if not (bad or unsafe or missing or low_sem or unparsable):
                continue
            total_candidates += 1
            key = field
            priority = 0
            if unparsable:
                priority += 5
            if field in HIGH_IMPACT_FIELDS:
                priority += 3
            if unsafe:
                priority += 3
            if missing:
                priority += 2
            if score == 0:
                priority += 2
            if counters[key] >= max_per_field and priority < 5:
                continue
            if counters[key] >= max_per_field + 4:
                continue
            row = make_row(
                "record",
                field,
                "upstream_field_extraction_fuzzy2",
                idx,
                proc,
                raw,
                score,
                semantic_score=sem_score,
                structured_prompt=False,
            )
            rows.append((priority, idx, row))
            counters[key] += 1
    rows.sort(key=lambda x: (-x[0], x[1], x[2]["field"]))
    return [r for _, _, r in rows], total_candidates


def build_downstream_rows(
    proc_path, raw_path, task, structured_prompt=False, max_per_field=20
):
    rows = []
    raw_rows = load_raw_by_index(raw_path)
    counters = Counter()
    total_candidates = 0
    for idx, proc in enumerate(iter_jsonl(proc_path)):
        raw = raw_rows[idx] if idx < len(raw_rows) else None
        sims = proc.get("similarity") or {}
        for field in ["ICD预测", "再入院概率", "药物推荐"]:
            score = sims.get(field)
            forced_unknown = False
            response_is_unparsed = not isinstance(proc.get("response"), dict)
            if response_is_unparsed:
                score = "unparsable"
                bad = True
            elif field == "药物推荐" and score is None:
                # Keep a bounded sample of medication recommendations because the available
                # processed file does not contain a direct gold list for automatic scoring.
                bad = counters[field] < 8
                forced_unknown = True
            else:
                bad = is_bad_score(score)
            if not bad:
                continue
            total_candidates += 1
            if counters[field] >= max_per_field:
                continue
            row = make_row(
                "downstream",
                field,
                task,
                idx,
                proc,
                raw,
                score,
                semantic_score=None,
                structured_prompt=structured_prompt,
                forced_unknown=forced_unknown,
            )
            rows.append(row)
            counters[field] += 1
    return rows, total_candidates


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_summary(rows, audit_denominator_note):
    total = len(rows)
    by_mode = Counter(r["failure_mode"] for r in rows)
    by_field = Counter(r["field"] for r in rows)
    by_level = Counter(r["level"] for r in rows)
    by_severity = Counter(r["severity"] for r in rows)

    mode_rows = []
    definitions = {
        "missing_evidence": (
            "Required information is absent from available records.",
            "retain_unknown",
        ),
        "conflicting_evidence": (
            "Source records disagree and cannot be reconciled.",
            "retain_conflict",
        ),
        "temporal_misassignment": (
            "Historical/current state assigned to the wrong time point.",
            "review_timeline",
        ),
        "normalization_error": (
            "Text meaning identified but controlled value/category is wrong.",
            "correct_normalization",
        ),
        "unsupported_completion": (
            "A value is filled without adequate source evidence.",
            "block_automatic_reuse",
        ),
        "schema_mismatch": (
            "Output does not match expected field, level, or schema.",
            "repair_schema_or_format",
        ),
        "format_parsing_error": (
            "Output cannot be parsed into expected structured format.",
            "repair_schema_or_format",
        ),
        "reference_ambiguity": (
            "Gold/reference answer is ambiguous or inconsistent.",
            "adjudicate_reference",
        ),
    }
    for mode, count in by_mode.most_common():
        definition, action = definitions.get(mode, ("", ""))
        mode_rows.append(
            {
                "failure_mode": mode,
                "definition": definition,
                "expected_review_action": action,
                "count": count,
                "percentage": f"{(count / total * 100 if total else 0):.2f}",
            }
        )
    write_csv(
        os.path.join(OUT_DIR, "failure_mode_summary_by_mode.csv"),
        mode_rows,
        ["failure_mode", "definition", "expected_review_action", "count", "percentage"],
    )
    write_csv(
        os.path.join(OUT_DIR, "failure_mode_summary_by_field.csv"),
        [
            {"field": k, "count": v, "percentage": f"{v / total * 100:.2f}"}
            for k, v in by_field.most_common()
        ],
        ["field", "count", "percentage"],
    )
    write_csv(
        os.path.join(OUT_DIR, "failure_mode_summary_by_level.csv"),
        [
            {"level": k, "count": v, "percentage": f"{v / total * 100:.2f}"}
            for k, v in by_level.most_common()
        ],
        ["level", "count", "percentage"],
    )
    write_csv(
        os.path.join(OUT_DIR, "failure_mode_summary_by_severity.csv"),
        [
            {"severity": k, "count": v, "percentage": f"{v / total * 100:.2f}"}
            for k, v in by_severity.most_common()
        ],
        ["severity", "count", "percentage"],
    )

    unsupported = by_mode.get("unsupported_completion", 0)
    unknown_filled = sum(
        1
        for r in rows
        if r["failure_mode"] == "unsupported_completion" and "reference" in r
    )
    conflict_collapsed = sum(
        1 for r in rows if r["failure_mode"] == "conflicting_evidence"
    )
    high_critical = sum(1 for r in rows if r["severity"] in {"high", "critical"})
    extra_rows = [
        {"metric": "audited_rows", "count": total, "note": audit_denominator_note},
        {"metric": "unsupported_completion", "count": unsupported, "note": ""},
        {
            "metric": "unknown_or_equivalent_states_incorrectly_filled",
            "count": unknown_filled,
            "note": "Heuristic: reference blank/unknown and prediction populated.",
        },
        {
            "metric": "conflict_states_incorrectly_collapsed",
            "count": conflict_collapsed,
            "note": "No explicit conflict labels were found in the processed model files.",
        },
        {"metric": "high_and_critical_cases", "count": high_critical, "note": ""},
    ]
    write_csv(
        os.path.join(OUT_DIR, "failure_mode_summary_metrics.csv"),
        extra_rows,
        ["metric", "count", "note"],
    )

    examples = []
    seen_modes = set()
    for row in rows:
        if row["failure_mode"] not in seen_modes or len(examples) < 8:
            examples.append(row)
            seen_modes.add(row["failure_mode"])
        if len(examples) >= 8 and len(seen_modes) >= min(len(by_mode), 5):
            break
    with open(
        os.path.join(OUT_DIR, "failure_mode_representative_examples.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Representative De-Identified Failure Examples\n\n")
        for i, row in enumerate(examples, 1):
            f.write(f"## Example {i}: {row['failure_mode']} / {row['field']}\n\n")
            f.write(f"- case_id_hash: `{row['case_id_hash']}`\n")
            f.write(f"- level/task: {row['level']} / {row['task']}\n")
            f.write(f"- prediction: {row['prediction'][:450]}\n")
            f.write(f"- reference: {row['reference'][:450]}\n")
            f.write(
                f"- is_correct: {row['is_correct']}; severity: {row['severity']}; review_action: {row['review_action']}\n"
            )
            f.write(f"- evidence: {row['source_evidence'][:650]}\n")
            f.write(f"- note: {row['note']}\n\n")

    with open(
        os.path.join(OUT_DIR, "failure_mode_report_blocks.md"), "w", encoding="utf-8"
    ) as f:
        top = ", ".join(
            f"{m} {c} ({c / total * 100:.1f}%)" for m, c in by_mode.most_common(5)
        )
        f.write("# Report-Ready Text Blocks\n\n")
        f.write("## Results\n")
        f.write(
            f"In the audited {MODEL} candidate subset, {total} bad or high-risk field/task rows were retained. "
            f"The most common pre-labeled failure modes were {top}. "
            f"High or critical cases accounted for {high_critical} rows. These percentages describe the audited subset only; {audit_denominator_note}\n\n"
        )
        f.write("## Methods\n")
        f.write(
            f"We searched {MODEL} upstream structured-field extraction records and downstream visit-level reuse records. "
            "For upstream records, raw prompts, model responses, references, fuzzy scores, and semantic scores were aligned by JSONL row index and split into field-level rows. "
            "For downstream records, unstructured-input and structured-input visit tasks were split into ICD prediction, medication recommendation, and readmission-probability rows. "
            "Rows were retained when automatic evaluation marked an error/partial match, when a reference-populated field was missing, when an unknown-equivalent reference was filled, or when downstream medication scoring was unavailable but required review. "
            "Taxonomy labels, severity, and review actions were pre-assigned heuristically for human adjudication; high-severity rows were marked for second review.\n\n"
        )
        if SOURCE_NOTE:
            f.write(f"Source mapping note: {SOURCE_NOTE}\n\n")
        f.write("## Taxonomy Summary\n\n")
        f.write(
            "| Failure mode | Definition | Expected review action | Count | Percentage |\n"
        )
        f.write("|---|---|---|---:|---:|\n")
        for row in mode_rows:
            f.write(
                f"| {row['failure_mode']} | {row['definition']} | {row['expected_review_action']} | {row['count']} | {row['percentage']}% |\n"
            )


def validate_rows(rows):
    errors = []
    for i, row in enumerate(rows, 2):
        if not row.get("failure_mode"):
            errors.append(f"row {i}: missing failure_mode")
        if (
            row.get("severity") in {"high", "critical"}
            and row.get("needs_second_review") != "true"
        ):
            errors.append(f"row {i}: high/critical without second review")
        if not row.get("source_evidence") and "unavailable" not in row.get("note", ""):
            errors.append(f"row {i}: no source evidence")
        for col in ["prediction", "reference", "is_correct"]:
            if row.get(col) is None or row.get(col) == "":
                errors.append(f"row {i}: empty {col}")
    return errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    upstream_rows, upstream_candidates = build_upstream_rows()
    downstream_rows, downstream_candidates = build_downstream_rows(
        DOWNSTREAM_PROC,
        DOWNSTREAM_RAW,
        "downstream_visit_unstructured_input",
        structured_prompt=False,
    )
    downstream_struct_rows, downstream_struct_candidates = build_downstream_rows(
        DOWNSTREAM_STRUCT_PROC,
        DOWNSTREAM_STRUCT_RAW,
        "downstream_visit_structured_reuse",
        structured_prompt=True,
    )
    rows = upstream_rows + downstream_rows + downstream_struct_rows
    # Fill display-safe placeholders for blank predictions/references after labeling.
    for row in rows:
        if row["prediction"] == "":
            row["prediction"] = "[EMPTY]"
        if row["reference"] == "":
            row["reference"] = "[EMPTY]"
        if row["source_evidence"] == "":
            row["source_evidence"] = "source evidence unavailable"
    out_csv = os.path.join(OUT_DIR, "failure_mode_analysis.csv")
    write_csv(out_csv, rows, CSV_COLUMNS)
    note = (
        f"candidate denominators before per-field caps: upstream={upstream_candidates}, "
        f"downstream_unstructured={downstream_candidates}, downstream_structured={downstream_struct_candidates}; "
        "sampling caps were applied per field/task, so this is not a full-cohort error rate."
    )
    if SOURCE_NOTE:
        note = note + " " + SOURCE_NOTE
    write_summary(rows, note)
    errors = validate_rows(rows)
    with open(os.path.join(OUT_DIR, "quality_check.txt"), "w", encoding="utf-8") as f:
        f.write(f"rows={len(rows)}\n")
        f.write(note + "\n")
        if errors:
            f.write("ERRORS\n")
            for err in errors:
                f.write(err + "\n")
        else:
            f.write("All automated quality checks passed.\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "upstream_rows": len(upstream_rows),
                "downstream_rows": len(downstream_rows),
                "downstream_structured_rows": len(downstream_struct_rows),
                "quality_errors": len(errors),
                "out_dir": OUT_DIR,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
