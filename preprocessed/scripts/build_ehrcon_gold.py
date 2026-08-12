#!/usr/bin/env python3
"""Filter EHRCon consistency annotations into note-level extraction gold data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NOTE_TYPES = ("discharge", "physician", "nursing")
NULL_VALUES = {"", "n", "nan", "none", "null", "na", "n/a"}
PREPROCESSED = Path(__file__).resolve().parents[1]
CRCEHR_ROOT = PREPROCESSED.parent
DEFAULT_INPUT = CRCEHR_ROOT / "rawdata" / "EHRCon" / "ehrcon-1.0.1"
DEFAULT_OUTPUT = PREPROCESSED / "ehrcon"


def norm_id(value: object) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"(-?\d+)(?:\.0+)?", text)
    return match.group(1) if match else text


def load_split_ids(processed: Path, note_type: str) -> dict[str, str]:
    result: dict[str, str] = {}
    names = {"test": f"{note_type}_test.csv", "valid": f"{note_type}_val.csv"}
    for split, filename in names.items():
        path = processed / split / note_type / filename
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = {name.strip().upper(): name for name in (reader.fieldnames or [])}
            id_column = next((columns[name] for name in ("ROW_ID", "ROWID", "ROW-ID", "ID") if name in columns), None)
            if id_column is None:
                raise ValueError(f"Missing row ID column: {path}")
            for row in reader:
                result[norm_id(row[id_column])] = split
    return result


def clean_fields(fields: object) -> tuple[dict[str, str], list[str]]:
    if not isinstance(fields, dict):
        return {}, []
    cleaned: dict[str, str] = {}
    removed: list[str] = []
    for key, value in fields.items():
        normalized = str(value).strip() if value is not None else ""
        if normalized.lower() in NULL_VALUES:
            removed.append(str(key))
        else:
            cleaned[str(key)] = normalized
    return cleaned, removed


def append_unique(mapping: dict[str, list[str]], key: str, value: str) -> None:
    if value not in mapping[key]:
        mapping[key].append(value)


def build(input_root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    exports = input_root / "data" / "exports"
    processed = input_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "ehrcon_extraction_gold.jsonl"
    manifest_path = output_dir / "ehrcon_extraction_gold_manifest.json"
    stats_path = output_dir / "ehrcon_extraction_gold_field_stats.json"

    notes: dict[tuple[str, str], dict] = {}
    exclusion = Counter()
    removed_placeholders = Counter()
    source_counts = defaultdict(Counter)

    for note_type in NOTE_TYPES:
        split_ids = load_split_ids(processed, note_type)
        with (exports / f"{note_type}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                source_counts[note_type]["input_entities"] += 1
                entity_type = str(row.get("entity_type", "")).strip()
                if entity_type not in {"1", "2"}:
                    exclusion[f"entity_type_{entity_type or 'missing'}"] += 1
                    continue
                if str(row.get("label", "")).strip().lower() != "consistency":
                    exclusion["not_consistent"] += 1
                    continue
                if str(row.get("errors", "")).strip().lower() not in {"nan", "", "none", "null"}:
                    exclusion["nonempty_errors"] += 1
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    exclusion["empty_text"] += 1
                    continue
                fields, removed = clean_fields(row.get("fields"))
                for field in removed:
                    removed_placeholders[field] += 1
                if not fields:
                    exclusion["empty_fields_after_cleaning"] += 1
                    continue

                row_id = norm_id(row.get("row_id"))
                key = (note_type, row_id)
                note = notes.setdefault(
                    key,
                    {
                        "row_id": row_id,
                        "note_type": note_type,
                        "split": split_ids.get(row_id, "unknown"),
                        "text": text,
                        "target": defaultdict(list),
                        "annotations": [],
                    },
                )
                if note["text"] != text:
                    raise ValueError(f"Conflicting text for {note_type}:{row_id}")
                for field, value in fields.items():
                    append_unique(note["target"], field, value)
                note["annotations"].append(
                    {
                        "entity": str(row.get("entity", "")),
                        "entity_type": entity_type,
                        "position": str(row.get("position", "")),
                        "fields": fields,
                    }
                )
                source_counts[note_type]["gold_entities"] += 1

    ordered_notes = [notes[key] for key in sorted(notes)]
    with gold_path.open("w", encoding="utf-8") as fh:
        for note in ordered_notes:
            note["target"] = dict(sorted(note["target"].items()))
            fh.write(json.dumps(note, ensure_ascii=False) + "\n")

    field_stats: dict[str, dict] = {}
    for note in ordered_notes:
        note_type = note["note_type"]
        source_counts[note_type]["gold_notes"] += 1
        source_counts[note_type][f"{note['split']}_notes"] += 1
        for annotation in note["annotations"]:
            for field, value in annotation["fields"].items():
                stat = field_stats.setdefault(
                    f"{note_type}|{field}",
                    {
                        "note_type": note_type,
                        "field": field,
                        "gold_occurrences": 0,
                        "type1": 0,
                        "type2": 0,
                        "entities": set(),
                        "notes": set(),
                        "values": [],
                    },
                )
                stat["gold_occurrences"] += 1
                stat[f"type{annotation['entity_type']}"] += 1
                stat["entities"].add(annotation["entity"])
                stat["notes"].add(note["row_id"])
                if value not in stat["values"] and len(stat["values"]) < 5:
                    stat["values"].append(value)

    serial_stats = []
    for stat in field_stats.values():
        stat["unique_entities"] = len(stat.pop("entities"))
        stat["gold_notes"] = len(stat.pop("notes"))
        serial_stats.append(stat)
    serial_stats.sort(key=lambda item: (NOTE_TYPES.index(item["note_type"]), item["field"]))

    total_annotations = sum(len(note["annotations"]) for note in ordered_notes)
    manifest = {
        "dataset": "EHRCon 1.0.1",
        "task": "note-to-structured-key-value extraction",
        "gold_unit": "entity-level annotation aggregated by note",
        "criteria": [
            "entity_type in {1, 2}",
            "label == consistency",
            "errors is empty/NaN",
            "text is non-empty",
            "fields remain non-empty after placeholder removal",
        ],
        "placeholder_values_removed": sorted(NULL_VALUES),
        "notes": len(ordered_notes),
        "gold_entities": total_annotations,
        "field_mappings": len(serial_stats),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "excluded_entities": dict(exclusion),
        "removed_placeholder_fields": dict(removed_placeholders),
        "output": str(gold_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps({"manifest": manifest, "fields": serial_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gold_path, manifest_path, stats_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in build(args.input_root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
