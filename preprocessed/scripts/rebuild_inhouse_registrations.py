#!/usr/bin/env python3
"""Canonicalize In-house registrations and create patient-level splits."""

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


PREPROCESSED_ROOT = Path(__file__).resolve().parents[1]
ROOT = PREPROCESSED_ROOT.parent
PREPROCESSED = PREPROCESSED_ROOT / "inhouse"
RAW = ROOT / "rawdata" / "In-house"
TEST_PATIENTS = 359

FIELD_RENAMES = {
    "TNM分期_M": "TNM分期-M",
    "TNM分期_N": "TNM分期-N",
    "TNM分期_T": "TNM分期-T",
    "TNM分期_总分期": "TNM分期总分期",
    "免疫组化标志物_定性结果": "免疫组化标志物定性结果",
    "免疫组化标志物_定量结果": "免疫组化标志物定量结果",
    "是否支持病理确诊原发直肠乙状结肠交界部癌": "是否支持病理确诊原发直肠癌乙状结肠交界处癌",
    "有无壁外血管侵犯(EMVI)": "有无壁外血管侵犯（EMVI）",
    "有无直肠系膜筋膜浸润(MRF)": "有无直肠系膜筋膜浸润（MRF）",
    "查镜能否顺利到达回盲部": "插镜能否顺利到达回盲部",
    "直肠病灶与肛门距离": "直肠病灶与肛距离(cm)",
    "肿瘤短径": "肿瘤短径(cm)",
    "肿瘤肠镜": "肿瘤长径(cm)",
    "诊断归一": "诊断归一（名称）",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_atomic(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def canonicalize(registration):
    result = {}
    for old_name, records in registration.items():
        new_name = FIELD_RENAMES.get(old_name, old_name)
        if new_name in result:
            raise ValueError(f"duplicate canonical field: {new_name}")
        result[new_name] = records
    return result


def record_pair(record):
    return str(record["patientId"]), str(record["visitId"])


def filter_registration(registration, patients):
    return {
        field: [record for record in records if str(record["patientId"]) in patients]
        for field, records in registration.items()
    }


def stable_patient_order(patients, namespace):
    def key(patient):
        value = f"coloreason:{namespace}:{patient}".encode()
        return hashlib.sha256(value).hexdigest()

    return sorted(patients, key=key)


def patient_visit_counts(registration, patients):
    visits = {patient: set() for patient in patients}
    for records in registration.values():
        for record in records:
            patient, visit = record_pair(record)
            if patient in visits:
                visits[patient].add(visit)

    return {patient: len(patient_visits) for patient, patient_visits in visits.items()}


def longitudinal_patient_order(registration, patients):
    visit_counts = patient_visit_counts(registration, patients)
    stable_ties = {
        patient: hashlib.sha256(f"coloreason:downstream-test:{patient}".encode()).hexdigest()
        for patient in patients
    }
    return sorted(patients, key=lambda patient: (-visit_counts[patient], stable_ties[patient]))


def read_pairs(path: Path, patient_column: str, visit_column: str):
    pairs = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row.get(patient_column, "").strip()
            visit = row.get(visit_column, "").strip()
            if patient and visit:
                pairs.add((patient, visit))
    return pairs


def medication_pairs():
    pairs = set()
    paths = sorted((RAW / "住院药物医嘱").glob("住院药物医嘱_医嘱详情_*.csv"))
    for path in paths:
        pairs.update(read_pairs(path, "patientId", "visitId"))
    return pairs


def pair_records(pairs, patients):
    return [
        {"patientId": patient, "visitId": visit}
        for patient, visit in sorted(pairs)
        if patient in patients
    ]


def patient_set(registration):
    return {
        str(record["patientId"])
        for records in registration.values()
        for record in records
    }


def validate_disjoint(*cohorts):
    for index, left in enumerate(cohorts):
        for right in cohorts[index + 1 :]:
            if left & right:
                raise ValueError("patient-level split leakage detected")


def main() -> None:
    source_path = PREPROCESSED / "registration.json"
    full = canonicalize(load_json(source_path))
    all_patients = patient_set(full)

    diagnosis = read_pairs(
        RAW / "结构化数据" / "就诊记录_诊断信息.csv", "患者ID", "就诊ID"
    )
    medication = medication_pairs()
    visits = read_pairs(RAW / "结构化数据" / "就诊记录.csv", "患者ID", "就诊ID")

    downstream_eligible = all_patients
    for pairs in (diagnosis, medication, visits):
        downstream_eligible &= {patient for patient, _ in pairs}
    if len(downstream_eligible) < TEST_PATIENTS:
        raise ValueError("not enough patients with all three downstream sources")

    downstream_test = set(
        longitudinal_patient_order(full, downstream_eligible)[:TEST_PATIENTS]
    )
    upstream_candidates = all_patients - downstream_test
    upstream_test = set(
        stable_patient_order(upstream_candidates, "upstream-test")[:TEST_PATIENTS]
    )
    train = all_patients - upstream_test - downstream_test
    validate_disjoint(train, upstream_test, downstream_test)

    train_registration = filter_registration(full, train)
    upstream_test_registration = filter_registration(full, upstream_test)
    downstream_test_registration = {
        "诊断归一（编码）": pair_records(diagnosis, downstream_test),
        "药物": pair_records(medication, downstream_test),
        "30天再入院": pair_records(visits, downstream_test),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = PREPROCESSED / f"registration_backup_{stamp}"
    backup.mkdir()
    for name in (
        "registration.json",
        "train_registration.json",
        "test_registration.json",
        "upstream_test_registration.json",
        "downstream_test_registration.json",
        "split_manifest.json",
    ):
        path = PREPROCESSED / name
        if path.exists():
            shutil.copy2(path, backup / name)

    dump_atomic(PREPROCESSED / "registration.json", full)
    dump_atomic(PREPROCESSED / "train_registration.json", train_registration)
    dump_atomic(
        PREPROCESSED / "upstream_test_registration.json", upstream_test_registration
    )
    dump_atomic(
        PREPROCESSED / "downstream_test_registration.json", downstream_test_registration
    )
    legacy_test = PREPROCESSED / "test_registration.json"
    if legacy_test.exists():
        legacy_test.unlink()

    downstream_visit_counts = sorted(
        patient_visit_counts(full, downstream_test).values()
    )

    manifest = {
        "schema_authority": "In-house canonical field names encoded in this script",
        "split_unit": "patientId",
        "split_protocol": {
            "train": "remaining patients after both held-out cohorts",
            "upstream_test": "upstream_test_registration.json",
            "downstream_test": "downstream_test_registration.json",
            "downstream_selection": "patients ranked by descending unique visit count",
        },
        "patient_counts": {
            "all": len(all_patients),
            "train": len(train),
            "upstream_test": len(upstream_test),
            "downstream_test": len(downstream_test),
        },
        "patient_overlaps": {
            "train_upstream_test": len(train & upstream_test),
            "train_downstream_test": len(train & downstream_test),
            "upstream_downstream_test": len(upstream_test & downstream_test),
        },
        "downstream_unique_visits_per_patient": {
            "min": downstream_visit_counts[0],
            "median": downstream_visit_counts[len(downstream_visit_counts) // 2],
            "max": downstream_visit_counts[-1],
        },
        "field_counts": {
            "upstream": len(full),
            "train": len(train_registration),
            "upstream_test": len(upstream_test_registration),
            "downstream": len(downstream_test_registration),
        },
        "entry_counts": {
            "registration": {field: len(records) for field, records in full.items()},
            "train": {field: len(records) for field, records in train_registration.items()},
            "upstream_test": {
                field: len(records) for field, records in upstream_test_registration.items()
            },
            "downstream_test": {
                field: len(records) for field, records in downstream_test_registration.items()
            },
        },
        "backup_directory": str(backup),
    }
    if set(train_registration) != set(upstream_test_registration):
        raise ValueError("train and upstream test fields differ")
    if any(not records for records in upstream_test_registration.values()):
        raise ValueError("upstream test contains an empty field")
    dump_atomic(PREPROCESSED / "split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
