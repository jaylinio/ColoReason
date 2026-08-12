import json
from collections import defaultdict
from typing import Dict, Any, Optional, Union

Number = Union[int, float]


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def aggregate_by_patient(
    input_file: str, output_file: str, on_conflict: str = "max"
) -> None:
    """Run aggregate by patient."""
    patients: Dict[str, Dict[str, Number]] = defaultdict(dict)

    with open(input_file, "r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata: Dict[str, Any] = data.get("metadata", {}) or {}
            similarity: Dict[str, Any] = data.get("similarity", {}) or {}

            patient_id: Optional[str] = metadata.get("patientId")
            field: Optional[str] = metadata.get("field")

            if not patient_id or not field:
                continue

            val = similarity.get(field, None)
            if not is_number(val):
                continue

            if field not in patients[patient_id]:
                patients[patient_id][field] = val
            else:
                if on_conflict == "max":
                    patients[patient_id][field] = max(patients[patient_id][field], val)  # type: ignore
                elif on_conflict == "last":
                    patients[patient_id][field] = val
                elif on_conflict == "first":
                    pass
                else:
                    patients[patient_id][field] = max(patients[patient_id][field], val)  # type: ignore

    with open(output_file, "w", encoding="utf-8") as fout:
        for pid in sorted(patients.keys()):
            out_obj = {"patientId": pid, "similarity": patients[pid]}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Wrote aggregated results to {output_file} ({len(patients)} patients)")


if __name__ == "__main__":
    input_path = ""
    output_path = input_path.replace(".jsonl", "_by_patient.jsonl")
    aggregate_by_patient(input_path, output_path, on_conflict="max")
