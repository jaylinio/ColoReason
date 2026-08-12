#!/usr/bin/env python3
"""Materialize split-aligned In-house structured key-values from raw CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FIELD_SPECS = {
    "TNM分期-M": ("结构化数据/就诊记录_TNM分期详情.csv", "TNM分期_M"),
    "TNM分期-N": ("结构化数据/就诊记录_TNM分期详情.csv", "TNM分期_N"),
    "TNM分期-T": ("结构化数据/就诊记录_TNM分期详情.csv", "TNM分期_T"),
    "TNM分期总分期": ("结构化数据/就诊记录_TNM分期详情.csv", "TNM分期_总分期"),
    "体重变化": ("结构化数据/一诉五史.csv", "体重变化"),
    "免疫组化标志物": ("结构化数据/病理报告_免疫组化.csv", "免疫组化标志物"),
    "免疫组化标志物定性结果": ("结构化数据/病理报告_免疫组化.csv", "免疫组化标志物_定性结果"),
    "免疫组化标志物定量结果": ("结构化数据/病理报告_免疫组化.csv", "免疫组化标志物_定量结果"),
    "大体分型": ("结构化数据/病理报告_肿瘤大体所见.csv", "大体分型"),
    "是否支持病理确诊原发直肠癌乙状结肠交界处癌": ("结构化数据/病理报告.csv", "是否支持病理确诊原发直肠乙状结肠交界部癌"),
    "是否支持病理确诊原发直肠癌": ("结构化数据/病理报告.csv", "是否支持病理确诊原发直肠癌"),
    "是否支持病理确诊原发结肠癌": ("结构化数据/病理报告.csv", "是否支持病理确诊原发结肠癌"),
    "有无侵犯神经": ("结构化数据/病理报告.csv", "有无侵犯神经"),
    "有无呕吐": ("结构化数据/一诉五史.csv", "有无呕吐"),
    "有无壁外血管侵犯（EMVI）": ("结构化数据/盆腔MR.csv", "有无壁外血管侵犯(EMVI)"),
    "有无大便困难": ("结构化数据/一诉五史.csv", "有无大便困难"),
    "有无消化道溃疡": ("结构化数据/一诉五史.csv", "有无消化道溃疡"),
    "有无淋巴结肿大": ("结构化数据/腹盆腔CT.csv", "有无淋巴结肿大"),
    "有无直肠系膜筋膜浸润（MRF）": ("结构化数据/盆腔MR.csv", "有无直肠系膜筋膜浸润(MRF)"),
    "有无肠息肉病史": ("结构化数据/一诉五史.csv", "有无肠息肉病史"),
    "有无脉管癌栓": ("结构化数据/病理报告.csv", "有无脉管癌栓"),
    "有无腹胀": ("结构化数据/一诉五史.csv", "有无腹胀"),
    "有无黑便或便血": ("结构化数据/一诉五史.csv", "有无黑便或便血"),
    "标本位置": ("结构化数据/病理报告_病理关键信息.csv", "标本位置"),
    "标本是否为转移灶": ("结构化数据/病理报告_病理关键信息.csv", "标本是否为转移灶"),
    "标本部位": ("结构化数据/病理报告_肿瘤大体所见.csv", "标本部位"),
    "浸润深度": ("结构化数据/病理报告_病理关键信息.csv", "浸润深度"),
    "淋巴结位置": ("结构化数据/病理报告_淋巴结详情.csv", "淋巴结位置"),
    "淋巴结肿大部位": ("结构化数据/腹盆腔CT.csv", "淋巴结肿大部位"),
    "直肠病灶与肛距离(cm)": ("结构化数据/盆腔MR.csv", "直肠病灶与肛门距离"),
    "组织学分型信息": ("结构化数据/病理报告_病理关键信息.csv", "组织学分型信息"),
    "肿瘤短径(cm)": ("结构化数据/病理报告_肿瘤大体所见.csv", "肿瘤短径"),
    "肿瘤长径(cm)": ("结构化数据/病理报告_肿瘤大体所见.csv", "肿瘤肠镜"),
    "诊断归一（名称）": ("结构化数据/就诊记录_诊断信息.csv", "诊断原文"),
    "送检淋巴结数目": ("结构化数据/病理报告_淋巴结详情.csv", "送检淋巴结数目"),
    "送检淋巴结是否阳性": ("结构化数据/病理报告_淋巴结详情.csv", "送检淋巴结是否阳性"),
    "阳性淋巴结数目": ("结构化数据/病理报告_淋巴结详情.csv", "阳性淋巴结数目"),
    "化疗疗效": ("结构化数据/化疗记录_化疗详情.csv", "化疗疗效"),
    "化疗目的": ("结构化数据/化疗记录_化疗详情.csv", "化疗目的"),
    "化疗药物": ("结构化数据/化疗记录_化疗详情.csv", "化疗药物"),
    "息肉所在部位": ("结构化数据/肠镜.csv", "息肉所在部位"),
    "有无息肉": ("结构化数据/肠镜.csv", "有无息肉"),
    "插镜能否顺利到达回盲部": ("结构化数据/肠镜.csv", "查镜能否顺利到达回盲部"),
}

PREPROCESSED = Path(__file__).resolve().parents[1]
CRCEHR_ROOT = PREPROCESSED.parent
DEFAULT_RAW_DIR = CRCEHR_ROOT / "rawdata" / "In-house"
DEFAULT_INHOUSE_DIR = PREPROCESSED / "inhouse"


def norm(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        prefix = text[:-2]
        if prefix.lstrip("-").isdigit():
            return prefix
    return text


def add_unique(items: list, value) -> None:
    """Append ``value`` to ``items`` if not already present.

    Accepts both hashable (str) and unhashable-but-comparable (dict) values:
    membership is tested with ``in`` (equality-based), which works for dicts.
    Used for str targets (upstream / 药物) and {code,name} dicts (诊断归一).
    """
    if value not in items:
        items.append(value)


def load_registration(path: Path) -> tuple[dict[str, set[tuple[str, str]]], set[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields: dict[str, set[tuple[str, str]]] = {}
    pairs: set[tuple[str, str]] = set()
    for field, rows in payload.items():
        entries = {(norm(row["patientId"]), norm(row["visitId"])) for row in rows}
        fields[field] = entries
        pairs.update(entries)
    return fields, pairs


def row_ids(row: dict[str, str]) -> tuple[str, str]:
    return norm(row.get("患者ID", row.get("patientId", ""))), norm(row.get("就诊ID", row.get("visitId", row.get("visit_id", ""))))


def provenance_ids(row: dict[str, str]) -> dict[str, str]:
    aliases = {
        "index": ("索引", "index"),
        "examinationId": ("检查ID", "examinationId"),
        "recordId": ("id",),
    }
    result = {}
    for output, names in aliases.items():
        value = next((norm(row.get(name, "")) for name in names if norm(row.get(name, ""))), "")
        if value:
            result[output] = value
    return result


def extract_upstream(raw: Path, registrations: dict[str, tuple[dict, set]]) -> dict[str, dict]:
    records = {split: {} for split in registrations}
    specs_by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for field, (source, column) in FIELD_SPECS.items():
        specs_by_file[source].append((field, column))

    for source, specs in specs_by_file.items():
        with (raw / source).open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [column for _, column in specs if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Missing columns in {source}: {missing}")
            for row in reader:
                pair = row_ids(row)
                for split, (field_pairs, _) in registrations.items():
                    matched = [(field, column) for field, column in specs if pair in field_pairs.get(field, set())]
                    if not matched:
                        continue
                    visit = records[split].setdefault(pair, {"patientId": pair[0], "visitId": pair[1], "registered_fields": [], "targets": defaultdict(list), "provenance": []})
                    ids = provenance_ids(row)
                    for field, column in matched:
                        add_unique(visit["registered_fields"], field)
                        value = "" if row.get(column) is None else str(row[column]).strip()
                        if value:
                            add_unique(visit["targets"][field], value)
                        visit["provenance"].append({"field": field, "value": value, "source_file": source, "source_column": column, **ids})
    return records


def timestamp_ms(value: str) -> float | None:
    try:
        number = float(value)
        return None if math.isnan(number) else number
    except (TypeError, ValueError):
        return None


def iso_time(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def extract_downstream(raw: Path, field_pairs: dict[str, set[tuple[str, str]]], all_pairs: set[tuple[str, str]]) -> dict:
    records = {pair: {"patientId": pair[0], "visitId": pair[1], "registered_fields": [], "targets": defaultdict(list), "provenance": []} for pair in all_pairs}

    diagnosis_file = "结构化数据/就诊记录_诊断信息.csv"
    with (raw / diagnosis_file).open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pair = row_ids(row)
            if pair not in field_pairs.get("诊断归一（编码）", set()):
                continue
            visit = records[pair]; add_unique(visit["registered_fields"], "诊断归一（编码）")
            raw_value = str(row.get("诊断归一", "")).strip()
            values = []
            if raw_value:
                try:
                    parsed = json.loads(raw_value)
                    # Downstream 诊断 target 同时保留归一编码与归一名称，供 LLM-as-a-judge
                    # 评估时按临床语义比较（编码做校验、名称做主语义信号）。
                    values = [
                        {
                            "code": str(item.get("诊断编码归一后", "")).strip(),
                            "name": str(item.get("诊断名称归一后", "")).strip(),
                        }
                        for item in parsed
                        if isinstance(item, dict) and str(item.get("诊断编码归一后", "")).strip()
                    ]
                except json.JSONDecodeError:
                    values = [{"code": raw_value, "name": ""}]
            for value in values:
                add_unique(visit["targets"]["诊断归一（编码）"], value)
            visit["provenance"].append({"field": "诊断归一（编码）", "value": raw_value, "source_file": diagnosis_file, "source_column": "诊断归一", **provenance_ids(row)})

    med_files = sorted((raw / "住院药物医嘱").glob("住院药物医嘱_医嘱详情_*.csv"))
    for path in med_files:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                pair = row_ids(row)
                if pair not in field_pairs.get("药物", set()):
                    continue
                visit = records[pair]; add_unique(visit["registered_fields"], "药物")
                value = str(row.get("ATC第五级分类_中文名称", "")).strip()
                if value:
                    add_unique(visit["targets"]["药物"], value)
                visit["provenance"].append({"field": "药物", "value": value, "source_file": f"住院药物医嘱/{path.name}", "source_column": "ATC第五级分类_中文名称", **provenance_ids(row)})

    visit_file = "结构化数据/就诊记录.csv"
    timeline: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    visit_rows = {}
    with (raw / visit_file).open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pair = row_ids(row); ts_raw = str(row.get("就诊日期", "")).strip(); ts = timestamp_ms(ts_raw)
            if ts is not None:
                timeline[pair[0]].append((ts, pair[1], ts_raw))
            if pair in field_pairs.get("30天再入院", set()):
                visit_rows[pair] = row
    for patient in timeline:
        timeline[patient].sort()
    for pair in field_pairs.get("30天再入院", set()):
        visit = records[pair]; add_unique(visit["registered_fields"], "30天再入院")
        row = visit_rows.get(pair, {}); current_raw = str(row.get("就诊日期", "")).strip(); current = timestamp_ms(current_raw)
        next_item = None
        if current is not None:
            next_item = next(((ts, visit_id, raw_value) for ts, visit_id, raw_value in timeline.get(pair[0], []) if ts > current), None)
        days = None if next_item is None else (next_item[0] - current) / 86400000
        derived = {
            "current_visit_date_raw": current_raw,
            "current_visit_date": iso_time(current),
            "next_visitId": None if next_item is None else next_item[1],
            "next_visit_date_raw": None if next_item is None else next_item[2],
            "next_visit_date": None if next_item is None else iso_time(next_item[0]),
            "days_to_next_visit": None if days is None else round(days, 6),
            "within_30_days": bool(days is not None and 0 < days <= 30),
        }
        visit["targets"]["30天再入院"] = derived
        visit["provenance"].append({"field": "30天再入院", "value": current_raw, "source_file": visit_file, "source_column": "就诊日期", "derivation": "next chronological visit for the same patient; 0 < delta_days <= 30", **provenance_ids(row)})
    return records


def serialize(path: Path, records: dict) -> dict:
    field_nonempty = Counter(); provenance_rows = 0
    with path.open("w", encoding="utf-8") as fh:
        for pair in sorted(records):
            row = records[pair]
            row["registered_fields"].sort()
            row["targets"] = dict(sorted(row["targets"].items()))
            for field, values in row["targets"].items():
                if isinstance(values, dict) or values:
                    field_nonempty[field] += 1
            provenance_rows += len(row["provenance"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"visits": len(records), "provenance_rows": provenance_rows, "nonempty_target_visits_by_field": dict(field_nonempty), "bytes": path.stat().st_size}


def write_readme(output: Path, manifest: dict) -> None:
    text = f"""# In-house 按划分整理的原始结构化键值

本目录将当前 ColoReason 划分对应的原始结构化键值物化到 `preprocessed`。训练和评估时不再需要扫描 `rawdata/In-house`。

## 文件

- `train_key_values.jsonl`：2,880 名患者训练集的上游结构化提取 target。
- `upstream_test_key_values.jsonl`：359 名患者上游独立测试集的结构化提取 target。
- `downstream_test_key_values.jsonl`：另一组 359 名患者下游独立测试集的 label。
- `field_schema.json`：目标字段到原始 CSV/列名的映射。
- `inhouse_key_values_manifest.json`：规模、文件大小、划分验证及生成信息。
- `../scripts/build_inhouse_key_values.py`：可复现生成脚本。
- 本目录内的 `split_manifest.json` 与 `*_registration.json` 是划分及样本索引的唯一依据。

## JSONL Schema

每行对应一个 `(patientId, visitId)`：

```json
{{
  "patientId": "...",
  "visitId": "...",
  "registered_fields": ["有无呕吐"],
  "targets": {{"有无呕吐": ["无"]}},
  "provenance": [
    {{"field": "有无呕吐", "value": "无", "source_file": "结构化数据/一诉五史.csv", "source_column": "有无呕吐"}}
  ]
}}
```

`targets` 保存去重后的非空原始值。`provenance` 保存所有匹配的原始行，包括空字符串，因此无需回到 raw 目录也能审计 missingness。重复检查或一对多记录使用列表表示，不互相覆盖。

## 划分语义

- 划分单位：`patientId`。
- 患者数量：train 2,880；upstream test 359；downstream test 359。
- 三个划分之间的患者交集均为 0。
- Train 和 upstream test 包含 43 个 `note -> structured key-value` 上游字段。
- Downstream test 只包含诊断编码、药物和 30 天再入院三个 target，不得合并进上游训练 target。

## Downstream 字段生成

- 诊断归一（编码）：解析原始 `诊断归一` JSON，每个条目同时保留 `诊断编码归一后` 与 `诊断名称归一后`，输出为 `{{"code": ..., "name": ...}}` 列表。编码用于规则校验、名称作为 LLM-as-a-judge 语义比较的主信号。字段名沿用 `诊断归一（编码）`，内容结构为编码-名称成对 dict。
- 药物：使用 `ATC第五级分类_中文名称`。
- 30 天再入院：唯一的派生字段。按同一患者的原始 `就诊日期` 排序，下一次就诊间隔位于 `(0, 30]` 天时，`within_30_days=true`。同时保留当前/下一次原始时间、规范日期、下一次 `visitId` 和间隔天数。

## 复现

在包含受控原始数据的环境中执行：

```bash
python3 scripts/build_inhouse_key_values.py \\
  --raw-dir /path/to/private/rawdata/In-house \\
  --preprocessed-dir /path/to/private/preprocessed/inhouse
```

当前产物：{manifest['outputs']['train']['visits']} 个 train visits，{manifest['outputs']['upstream_test']['visits']} 个 upstream-test visits，{manifest['outputs']['downstream_test']['visits']} 个 downstream-test visits。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _write_schema(output_dir: Path) -> None:
    """Write field_schema.json. 诊断条目 transform 同时反映编码与名称解析。"""
    schema = {field: {"source_file": source, "source_column": column} for field, (source, column) in FIELD_SPECS.items()}
    schema["诊断归一（编码）"] = {"source_file": "结构化数据/就诊记录_诊断信息.csv", "source_column": "诊断归一", "transform": "parse 诊断编码归一后 + 诊断名称归一后"}
    schema["药物"] = {"source_file": "住院药物医嘱/住院药物医嘱_医嘱详情_*.csv", "source_column": "ATC第五级分类_中文名称"}
    schema["30天再入院"] = {"source_file": "结构化数据/就诊记录.csv", "source_column": "就诊日期", "transform": "next chronological visit within (0, 30] days"}
    (output_dir / "field_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_downstream_only(args) -> dict:
    """增量模式：仅重建下游产物，不重跑上游抽取、不触碰 train/upstream_test jsonl。

    - 重写 downstream_test_key_values.jsonl；
    - 读旧 manifest，保留上游统计，仅更新 downstream_test 产物与时间戳；
    - 更新 field_schema.json（诊断 transform 描述）与 README。
    """
    output_dir = args.preprocessed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    downstream_fields, downstream_pairs = load_registration(args.preprocessed_dir / "downstream_test_registration.json")
    downstream = extract_downstream(args.raw_dir, downstream_fields, downstream_pairs)
    downstream_stat = serialize(output_dir / "downstream_test_key_values.jsonl", downstream)

    manifest_path = output_dir / "inhouse_key_values_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["last_incremental"] = {"scope": "downstream", "note": "only downstream_test_key_values.jsonl rebuilt; upstream outputs untouched"}
    manifest.setdefault("outputs", {})["downstream_test"] = downstream_stat
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_schema(output_dir)
    write_readme(output_dir, manifest)
    return {"mode": "downstream-only", "outputs": {"downstream_test": downstream_stat}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_INHOUSE_DIR)
    parser.add_argument(
        "--only",
        choices=["all", "downstream"],
        default="all",
        help="Incremental rebuild: 'downstream' rebuilds only downstream_test_key_values.jsonl "
        "(诊断/药物/30天再入院) and leaves train / upstream_test outputs untouched.",
    )
    args = parser.parse_args()

    if args.only == "downstream":
        print(json.dumps(_run_downstream_only(args), ensure_ascii=False, indent=2))
        return

    registrations = {}
    for split, name in (("train", "train_registration.json"), ("upstream_test", "upstream_test_registration.json")):
        registrations[split] = load_registration(args.preprocessed_dir / name)
    upstream = extract_upstream(args.raw_dir, registrations)
    downstream_fields, downstream_pairs = load_registration(args.preprocessed_dir / "downstream_test_registration.json")
    downstream = extract_downstream(args.raw_dir, downstream_fields, downstream_pairs)

    output_dir = args.preprocessed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": serialize(output_dir / "train_key_values.jsonl", upstream["train"]),
        "upstream_test": serialize(output_dir / "upstream_test_key_values.jsonl", upstream["upstream_test"]),
        "downstream_test": serialize(output_dir / "downstream_test_key_values.jsonl", downstream),
    }
    patient_sets = {split: {pair[0] for pair in records} for split, records in {**upstream, "downstream_test": downstream}.items()}
    overlap = {
        "train_upstream_test": len(patient_sets["train"] & patient_sets["upstream_test"]),
        "train_downstream_test": len(patient_sets["train"] & patient_sets["downstream_test"]),
        "upstream_downstream_test": len(patient_sets["upstream_test"] & patient_sets["downstream_test"]),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_authority": "split_manifest.json and *_registration.json in the same directory",
        "raw_data_root": str(args.raw_dir),
        "field_count": len(FIELD_SPECS),
        "outputs": outputs,
        "patient_counts": {key: len(value) for key, value in patient_sets.items()},
        "patient_overlaps": overlap,
    }
    if any(overlap.values()):
        raise ValueError(f"Patient leakage detected: {overlap}")
    _write_schema(output_dir)
    (output_dir / "inhouse_key_values_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
