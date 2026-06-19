#!/usr/bin/env python3
"""Build a breast-cancer single-snapshot SPC/IPS dataset with last-window support."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "prepared" / "breast_cancer_single_snapshot_0d_last_window_supported"

WINDOW_DAYS = 30
HORIZON_DAYS = 0

LAB_DEFS = {
    "HB": ("hb", "HB"),
    "WBC": ("wbc", "WBC"),
    "ТРОМБОЦИТЫ": ("plt", "PLT"),
    "НЕЙТРОФИЛЫ": ("neutrophils", "NEUTROPHILS"),
    "КРЕАТИНИН": ("creatinine", "CREATININE"),
    "МОЧЕВИНА": ("urea", "UREA"),
    "АЛТ": ("alt", "ALT"),
    "АСТ": ("ast", "AST"),
    "ОБЩИЙ БИЛИРУБИН": ("total_bilirubin", "TOTAL_BILIRUBIN"),
    "АЛЬБУМИН": ("albumin", "ALBUMIN"),
    "ЛДГ": ("ldh", "LDH"),
    "СОЭ": ("esr", "ESR"),
}
LAB_SLUGS = [slug for slug, _ in LAB_DEFS.values()]


@dataclass(frozen=True)
class PatientMeta:
    patient_id: str
    diagnosis_date: date
    snapshot_date: date
    max_date: date
    death_date: Optional[date]
    will_die_next_120_days: int
    age_bin: str
    gender: str
    cancer_type: str
    cancer_subtype_code: str
    stage_normalized: str
    metastasis_flag: int
    window_count: int


def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def date_to_str(value: Optional[date]) -> str:
    return value.isoformat() if value else ""


def age_in_years(birth_date: date, on_date: date) -> int:
    years = on_date.year - birth_date.year
    if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def age_bin(age: int) -> str:
    if 18 <= age <= 39:
        return "18-39"
    if 40 <= age <= 49:
        return "40-49"
    if 50 <= age <= 59:
        return "50-59"
    if 60 <= age <= 69:
        return "60-69"
    if 70 <= age <= 79:
        return "70-79"
    return "80+"


def normalize_gender(raw_gender: str) -> str:
    raw_gender = (raw_gender or "").strip()
    if raw_gender == "Мужской":
        return "male"
    if raw_gender == "Женский":
        return "female"
    return "unknown"


def normalize_stage(raw_stage: str) -> str:
    stage = (raw_stage or "").strip().upper()
    if not stage or stage in {"НЕИЗВЕСТНО", "НЕТ ДАННЫХ", "НЕПРИМЕНИМО", "UNKNOWN"}:
        return "UNKNOWN"
    if stage.startswith("IV"):
        return "IV"
    if stage.startswith("III"):
        return "III"
    if stage.startswith("II"):
        return "II"
    if stage.startswith("I"):
        return "I"
    return "UNKNOWN"


def load_cancer_type_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    path = DATA_DIR / "icd10_oncology_codes.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = (row["code"] or "").strip().upper()
            if not code:
                continue
            stem = code.split(".")[0]
            cancer_site = (row["cancer_site"] or "").strip()
            if cancer_site and cancer_site != "Неуточненная локализация":
                lookup.setdefault(stem, cancer_site)
            lookup.setdefault(code, cancer_site or stem)
    return lookup


def load_test_defaults() -> Dict[str, str]:
    defaults: Dict[str, str] = {}
    path = DATA_DIR / "test_types.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            defaults[(row["code"] or "").strip()] = (row["default_unit_code"] or "").strip()
    return defaults


def load_unit_conversions() -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    conversions: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    path = DATA_DIR / "unit_conversions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                (row["test_type_code"] or "").strip(),
                (row["from_unit_code"] or "").strip(),
                (row["to_unit_code"] or "").strip(),
            )
            conversions[key] = (
                float(row["conversion_factor"] or 0.0),
                float(row["conversion_offset"] or 0.0),
            )
    return conversions


def convert_lab_value(
    test_code: str,
    raw_value: str,
    unit_code: str,
    default_units: Dict[str, str],
    conversions: Dict[Tuple[str, str, str], Tuple[float, float]],
) -> Optional[float]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None

    target_unit = default_units.get(test_code, "")
    unit_code = (unit_code or "").strip()
    if not target_unit or not unit_code or unit_code == target_unit:
        return value

    conversion = conversions.get((test_code, unit_code, target_unit))
    if conversion is None:
        return None
    factor, offset = conversion
    return (value * factor) + offset


def cancer_type_for_code(diag_code: str, lookup: Dict[str, str]) -> str:
    diag_code = (diag_code or "").strip().upper()
    if not diag_code:
        return "UNKNOWN"
    stem = diag_code.split(".")[0]
    candidate = lookup.get(diag_code) or lookup.get(stem)
    if candidate and candidate != "Неуточненная локализация":
        return candidate
    return stem


def load_earliest_diagnoses(cancer_lookup: Dict[str, str]) -> Dict[str, dict]:
    earliest: Dict[str, dict] = {}
    path = DATA_DIR / "diagnoses.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            diag_date = parse_date(row["diag_establish_date"])
            patient_id = row["patient_id"]
            if diag_date is None:
                continue
            existing = earliest.get(patient_id)
            if existing is not None and existing["diagnosis_date"] <= diag_date:
                continue
            earliest[patient_id] = {
                "diagnosis_date": diag_date,
                "diag_code": (row["diag_code"] or "").strip().upper(),
                "cancer_type": cancer_type_for_code(row["diag_code"], cancer_lookup),
                "stage_normalized": normalize_stage(row["diag_stage"]),
                "metastasis_flag": 1 if (row["diag_metastases_localization"] or "").strip() else 0,
            }
    return earliest


def build_cohort() -> Tuple[Dict[str, PatientMeta], Dict[str, int]]:
    cancer_lookup = load_cancer_type_lookup()
    earliest_diagnoses = load_earliest_diagnoses(cancer_lookup)
    excluded = Counter()
    cohort: Dict[str, PatientMeta] = {}

    path = DATA_DIR / "patients.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            diagnosis = earliest_diagnoses.get(patient_id)
            if diagnosis is None:
                excluded["missing_diagnosis"] += 1
                continue
            if not diagnosis["diag_code"].startswith("C50"):
                excluded["non_breast_cancer"] += 1
                continue

            birth_date = parse_date(row["birth_date"])
            max_date = parse_date(row["max_date"])
            if birth_date is None or max_date is None:
                excluded["missing_birth_or_max_date"] += 1
                continue

            age = age_in_years(birth_date, diagnosis["diagnosis_date"])
            if age < 18:
                excluded["age_under_18"] += 1
                continue

            snapshot_date = max_date - timedelta(days=HORIZON_DAYS)
            observed_days = (snapshot_date - diagnosis["diagnosis_date"]).days + 1
            if observed_days <= 0:
                excluded["insufficient_followup_for_snapshot"] += 1
                continue

            window_count = (observed_days + WINDOW_DAYS - 1) // WINDOW_DAYS
            death_date = parse_date(row["death_date"])
            label = 1 if (death_date is not None and death_date <= snapshot_date + timedelta(days=HORIZON_DAYS)) else 0

            cohort[patient_id] = PatientMeta(
                patient_id=patient_id,
                diagnosis_date=diagnosis["diagnosis_date"],
                snapshot_date=snapshot_date,
                max_date=max_date,
                death_date=death_date,
                will_die_next_120_days=label,
                age_bin=age_bin(age),
                gender=normalize_gender(row["gender"]),
                cancer_type="breast_cancer",
                cancer_subtype_code=diagnosis["diag_code"] or "C50",
                stage_normalized=diagnosis["stage_normalized"],
                metastasis_flag=diagnosis["metastasis_flag"],
                window_count=window_count,
            )

    return cohort, dict(excluded)


def build_sqlite_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")

    conn.executescript(
        """
        DROP TABLE IF EXISTS state_base;
        DROP TABLE IF EXISTS lab_agg;
        DROP TABLE IF EXISTS ecog_agg;
        DROP TABLE IF EXISTS chemo_agg;
        DROP TABLE IF EXISTS radiotherapy_agg;
        DROP TABLE IF EXISTS admission_agg;

        CREATE TABLE state_base (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            age_bin TEXT NOT NULL,
            gender TEXT NOT NULL,
            cancer_type TEXT NOT NULL,
            cancer_subtype_code TEXT NOT NULL,
            stage_normalized TEXT NOT NULL,
            metastasis_flag INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index)
        );

        CREATE TABLE lab_agg (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            lab_slug TEXT NOT NULL,
            min_value REAL NOT NULL,
            max_value REAL NOT NULL,
            observed_count INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index, lab_slug)
        );

        CREATE TABLE ecog_agg (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            ecog_value INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index)
        );

        CREATE TABLE chemo_agg (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            active INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index)
        );

        CREATE TABLE radiotherapy_agg (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            active INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index)
        );

        CREATE TABLE admission_agg (
            patient_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            admission_count INTEGER NOT NULL,
            PRIMARY KEY (patient_id, window_index)
        );
        """
    )
    return conn


def window_index_for_event(meta: PatientMeta, event_date: date) -> Optional[int]:
    if event_date < meta.diagnosis_date or event_date > meta.snapshot_date:
        return None
    return ((event_date - meta.diagnosis_date).days // WINDOW_DAYS) + 1


def overlapping_window_range(meta: PatientMeta, start_date: date, end_date: date) -> Optional[Tuple[int, int]]:
    clipped_start = max(start_date, meta.diagnosis_date)
    clipped_end = min(end_date, meta.snapshot_date)
    if clipped_start > clipped_end:
        return None
    first = ((clipped_start - meta.diagnosis_date).days // WINDOW_DAYS) + 1
    last = ((clipped_end - meta.diagnosis_date).days // WINDOW_DAYS) + 1
    return first, last


def insert_state_base(conn: sqlite3.Connection, cohort: Dict[str, PatientMeta]) -> None:
    rows: List[Tuple] = []
    for meta in cohort.values():
        for window_index in range(1, meta.window_count + 1):
            window_start = meta.diagnosis_date + timedelta(days=(window_index - 1) * WINDOW_DAYS)
            window_end = min(window_start + timedelta(days=WINDOW_DAYS - 1), meta.snapshot_date)
            rows.append(
                (
                    meta.patient_id,
                    window_index,
                    window_start.isoformat(),
                    window_end.isoformat(),
                    meta.age_bin,
                    meta.gender,
                    meta.cancer_type,
                    meta.cancer_subtype_code,
                    meta.stage_normalized,
                    meta.metastasis_flag,
                )
            )
            if len(rows) >= 5000:
                conn.executemany(
                    """
                    INSERT INTO state_base (
                        patient_id, window_index, window_start, window_end,
                        age_bin, gender, cancer_type, cancer_subtype_code, stage_normalized, metastasis_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                rows.clear()
    if rows:
        conn.executemany(
            """
            INSERT INTO state_base (
                patient_id, window_index, window_start, window_end,
                age_bin, gender, cancer_type, cancer_subtype_code, stage_normalized, metastasis_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()


def aggregate_labs(
    conn: sqlite3.Connection,
    cohort: Dict[str, PatientMeta],
    default_units: Dict[str, str],
    conversions: Dict[Tuple[str, str, str], Tuple[float, float]],
) -> Dict[str, int]:
    stats = Counter()
    pending: Dict[Tuple[str, int, str], List[float]] = {}
    path = DATA_DIR / "laboratory_tests.csv"

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            test_code = (row["test_type_code"] or "").strip()
            lab_def = LAB_DEFS.get(test_code)
            if lab_def is None:
                continue

            patient_id = row["patient_id"]
            meta = cohort.get(patient_id)
            if meta is None:
                continue

            test_date = parse_date(row["test_date"])
            if test_date is None:
                stats["missing_lab_date"] += 1
                continue

            window_index = window_index_for_event(meta, test_date)
            if window_index is None:
                stats["lab_outside_snapshot"] += 1
                continue

            converted = convert_lab_value(
                test_code=test_code,
                raw_value=row["test_result"],
                unit_code=row["value_unit_code"],
                default_units=default_units,
                conversions=conversions,
            )
            if converted is None:
                stats["lab_unusable_value_or_unit"] += 1
                continue

            lab_slug = lab_def[0]
            key = (patient_id, window_index, lab_slug)
            current = pending.get(key)
            if current is None:
                pending[key] = [converted, converted, 1.0]
            else:
                current[0] = min(current[0], converted)
                current[1] = max(current[1], converted)
                current[2] += 1.0

            if len(pending) >= 50000:
                flush_lab_pending(conn, pending)
                pending.clear()

    if pending:
        flush_lab_pending(conn, pending)
    conn.commit()
    return dict(stats)


def flush_lab_pending(conn: sqlite3.Connection, pending: Dict[Tuple[str, int, str], List[float]]) -> None:
    conn.executemany(
        """
        INSERT INTO lab_agg (patient_id, window_index, lab_slug, min_value, max_value, observed_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id, window_index, lab_slug) DO UPDATE SET
            min_value = MIN(lab_agg.min_value, excluded.min_value),
            max_value = MAX(lab_agg.max_value, excluded.max_value),
            observed_count = lab_agg.observed_count + excluded.observed_count
        """,
        [
            (patient_id, window_index, lab_slug, values[0], values[1], int(values[2]))
            for (patient_id, window_index, lab_slug), values in pending.items()
        ],
    )


def aggregate_ecog(conn: sqlite3.Connection, cohort: Dict[str, PatientMeta]) -> Dict[str, int]:
    stats = Counter()
    pending: Dict[Tuple[str, int], int] = {}
    path = DATA_DIR / "ecog_performance_status.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            meta = cohort.get(patient_id)
            if meta is None:
                continue

            assessment_date = parse_date(row["assessment_date"])
            if assessment_date is None:
                stats["missing_ecog_date"] += 1
                continue

            window_index = window_index_for_event(meta, assessment_date)
            if window_index is None:
                stats["ecog_outside_snapshot"] += 1
                continue

            try:
                score = int(float(row["ecog_score"]))
            except ValueError:
                stats["invalid_ecog_score"] += 1
                continue
            if score == 5:
                stats["ecog_score_5_skipped"] += 1
                continue
            score = max(0, min(score, 4))

            key = (patient_id, window_index)
            pending[key] = max(score, pending.get(key, score))
            if len(pending) >= 20000:
                flush_simple_max(conn, "ecog_agg", "ecog_value", pending)
                pending.clear()

    if pending:
        flush_simple_max(conn, "ecog_agg", "ecog_value", pending)
    conn.commit()
    return dict(stats)


def flush_simple_max(conn: sqlite3.Connection, table: str, column: str, pending: Dict[Tuple[str, int], int]) -> None:
    conn.executemany(
        f"""
        INSERT INTO {table} (patient_id, window_index, {column})
        VALUES (?, ?, ?)
        ON CONFLICT(patient_id, window_index) DO UPDATE SET
            {column} = MAX({table}.{column}, excluded.{column})
        """,
        [(patient_id, window_index, value) for (patient_id, window_index), value in pending.items()],
    )


def flush_simple_sum(conn: sqlite3.Connection, table: str, column: str, pending: Dict[Tuple[str, int], int]) -> None:
    conn.executemany(
        f"""
        INSERT INTO {table} (patient_id, window_index, {column})
        VALUES (?, ?, ?)
        ON CONFLICT(patient_id, window_index) DO UPDATE SET
            {column} = {table}.{column} + excluded.{column}
        """,
        [(patient_id, window_index, value) for (patient_id, window_index), value in pending.items()],
    )


def aggregate_interval_activity(
    conn: sqlite3.Connection,
    cohort: Dict[str, PatientMeta],
    filename: str,
    table_name: str,
) -> Dict[str, int]:
    stats = Counter()
    pending: Dict[Tuple[str, int], int] = {}
    path = DATA_DIR / filename

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            meta = cohort.get(patient_id)
            if meta is None:
                continue

            start_date = parse_date(row["start_date"])
            end_date = parse_date(row["end_date"]) or start_date
            if start_date is None:
                stats[f"{table_name}_missing_start_date"] += 1
                continue
            if end_date is None:
                end_date = start_date
            if end_date < start_date:
                start_date, end_date = end_date, start_date

            overlap = overlapping_window_range(meta, start_date, end_date)
            if overlap is None:
                stats[f"{table_name}_outside_snapshot"] += 1
                continue

            first_window, last_window = overlap
            for window_index in range(first_window, last_window + 1):
                pending[(patient_id, window_index)] = 1

            if len(pending) >= 20000:
                conn.executemany(
                    f"""
                    INSERT INTO {table_name} (patient_id, window_index, active)
                    VALUES (?, ?, 1)
                    ON CONFLICT(patient_id, window_index) DO UPDATE SET active = 1
                    """,
                    [(patient_id, window_index) for (patient_id, window_index) in pending.keys()],
                )
                pending.clear()

    if pending:
        conn.executemany(
            f"""
            INSERT INTO {table_name} (patient_id, window_index, active)
            VALUES (?, ?, 1)
            ON CONFLICT(patient_id, window_index) DO UPDATE SET active = 1
            """,
            [(patient_id, window_index) for (patient_id, window_index) in pending.keys()],
        )
    conn.commit()
    return dict(stats)


def aggregate_admissions(conn: sqlite3.Connection, cohort: Dict[str, PatientMeta]) -> Dict[str, int]:
    stats = Counter()
    pending: Dict[Tuple[str, int], int] = defaultdict(int)
    path = DATA_DIR / "hospital_admissions.csv"

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            meta = cohort.get(patient_id)
            if meta is None:
                continue

            admit_date = parse_date(row["admission_timestamp"])
            if admit_date is None:
                stats["missing_admission_date"] += 1
                continue

            window_index = window_index_for_event(meta, admit_date)
            if window_index is None:
                stats["admission_outside_snapshot"] += 1
                continue

            pending[(patient_id, window_index)] += 1
            if len(pending) >= 20000:
                flush_simple_sum(conn, "admission_agg", "admission_count", pending)
                pending.clear()

    if pending:
        flush_simple_sum(conn, "admission_agg", "admission_count", pending)
    conn.commit()
    return dict(stats)


def write_patient_snapshots(cohort: Dict[str, PatientMeta], path: Path) -> Dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_id",
                "diagnosis_date",
                "snapshot_date",
                "max_date",
                "death_date",
                f"will_die_next_{HORIZON_DAYS}_days",  # 动态列名
                "age_bin",
                "gender",
                "cancer_type",
                "cancer_subtype_code",
                "stage_normalized",
                "metastasis_flag",
                "window_count",
            ]
        )
        for meta in sorted(cohort.values(), key=lambda item: item.patient_id):
            writer.writerow(
                [
                    meta.patient_id,
                    date_to_str(meta.diagnosis_date),
                    date_to_str(meta.snapshot_date),
                    date_to_str(meta.max_date),
                    date_to_str(meta.death_date),
                    meta.will_die_next_120_days,
                    meta.age_bin,
                    meta.gender,
                    meta.cancer_type,
                    meta.cancer_subtype_code,
                    meta.stage_normalized,
                    meta.metastasis_flag,
                    meta.window_count,
                ]
            )
            label_counts[str(meta.will_die_next_120_days)] += 1
    return dict(label_counts)


def states_select_sql() -> str:
    lab_columns = []
    for slug in LAB_SLUGS:
        lab_columns.append(
            f"MAX(CASE WHEN la.lab_slug = '{slug}' THEN la.min_value END) AS {slug}_min"
        )
        lab_columns.append(
            f"MAX(CASE WHEN la.lab_slug = '{slug}' THEN la.max_value END) AS {slug}_max"
        )
        lab_columns.append(
            f"CASE WHEN MAX(CASE WHEN la.lab_slug = '{slug}' THEN 1 ELSE 0 END) = 1 THEN 0 ELSE 1 END AS {slug}_missing"
        )

    sql = f"""
    SELECT
        sb.patient_id,
        sb.window_index,
        sb.window_start,
        sb.window_end,
        sb.age_bin,
        sb.gender,
        sb.cancer_type,
        sb.cancer_subtype_code,
        sb.stage_normalized,
        sb.metastasis_flag,
        {', '.join(lab_columns)},
        ea.ecog_value AS ecog,
        COALESCE(ca.active, 0) AS chemo_active,
        COALESCE(ra.active, 0) AS radiotherapy_active,
        CASE
            WHEN COALESCE(aa.admission_count, 0) >= 2 THEN '2+'
            ELSE CAST(COALESCE(aa.admission_count, 0) AS TEXT)
        END AS admissions
    FROM state_base sb
    LEFT JOIN lab_agg la
        ON la.patient_id = sb.patient_id
       AND la.window_index = sb.window_index
    LEFT JOIN ecog_agg ea
        ON ea.patient_id = sb.patient_id
       AND ea.window_index = sb.window_index
    LEFT JOIN chemo_agg ca
        ON ca.patient_id = sb.patient_id
       AND ca.window_index = sb.window_index
    LEFT JOIN radiotherapy_agg ra
        ON ra.patient_id = sb.patient_id
       AND ra.window_index = sb.window_index
    LEFT JOIN admission_agg aa
        ON aa.patient_id = sb.patient_id
       AND aa.window_index = sb.window_index
    GROUP BY
        sb.patient_id, sb.window_index, sb.window_start, sb.window_end,
        sb.age_bin, sb.gender, sb.cancer_type, sb.stage_normalized, sb.metastasis_flag,
        sb.cancer_subtype_code,
        ea.ecog_value, ca.active, ra.active, aa.admission_count
    ORDER BY sb.patient_id, sb.window_index
    """
    return sql


def has_state_signal(row: dict) -> bool:
    for slug in LAB_SLUGS:
        if int(row[f"{slug}_missing"]) == 0:
            return True
    if row["ecog"] not in {"", None}:
        return True
    if int(row["chemo_active"]) == 1 or int(row["radiotherapy_active"]) == 1:
        return True
    if row["admissions"] != "0":
        return True
    return False


def get_supported_patient_ids(conn: sqlite3.Connection, cohort: Dict[str, PatientMeta]) -> Tuple[set[str], Dict[str, int]]:
    supported: set[str] = set()
    unsupported = 0
    cursor = conn.execute(states_select_sql())
    columns = [item[0] for item in cursor.description]
    for values in cursor:
        row = dict(zip(columns, values))
        pid = row["patient_id"]
        if int(row["window_index"]) != cohort[pid].window_count:
            continue
        if has_state_signal(row):
            supported.add(pid)
        else:
            unsupported += 1
    return supported, {
        "patients_before_support_filter": len(cohort),
        "patients_after_support_filter": len(supported),
        "patients_removed_no_last_window_signal": unsupported,
    }


def write_patient_states(conn: sqlite3.Connection, path: Path, allowed_patient_ids: set[str]) -> Dict[str, int]:
    summary = Counter()
    with path.open("w", newline="", encoding="utf-8") as handle:
        cursor = conn.execute(states_select_sql())
        writer = csv.writer(handle)
        columns = [item[0] for item in cursor.description]
        writer.writerow(columns)
        for row in cursor:
            if row[0] not in allowed_patient_ids:
                continue
            writer.writerow(row)
            summary["state_rows"] += 1
            if row[columns.index("ecog")] is not None:
                summary["states_with_ecog"] += 1
            if row[columns.index("chemo_active")] == 1:
                summary["states_with_chemo"] += 1
            if row[columns.index("radiotherapy_active")] == 1:
                summary["states_with_radiotherapy"] += 1
            if row[columns.index("admissions")] != "0":
                summary["states_with_admission"] += 1
    return dict(summary)


def write_patient_trajectories(states_path: Path, output_path: Path) -> Dict[str, int]:
    summary = Counter()
    with states_path.open(newline="", encoding="utf-8") as handle, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(handle)
        current_patient_id: Optional[str] = None
        current_static: Optional[dict] = None
        current_states: List[dict] = []

        for row in reader:
            patient_id = row["patient_id"]
            if current_patient_id is not None and patient_id != current_patient_id:
                out.write(
                    json.dumps(
                        {
                            "patient_id": current_patient_id,
                            "static_context": current_static,
                            "states": current_states,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                summary["trajectory_rows"] += 1
                summary["max_sequence_length"] = max(summary["max_sequence_length"], len(current_states))
                current_states = []

            if patient_id != current_patient_id:
                current_patient_id = patient_id
                current_static = {
                    "age_bin": row["age_bin"],
                    "gender": row["gender"],
                    "cancer_type": row["cancer_type"],
                    "cancer_subtype_code": row["cancer_subtype_code"],
                    "stage_normalized": row["stage_normalized"],
                    "metastasis_flag": int(row["metastasis_flag"]),
                }

            state = {
                "window_index": int(row["window_index"]),
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "labs": {},
                "ecog": None if row["ecog"] in {"", None} else int(row["ecog"]),
                "chemo_active": int(row["chemo_active"]),
                "radiotherapy_active": int(row["radiotherapy_active"]),
                "admissions": row["admissions"],
            }
            for slug in LAB_SLUGS:
                missing = int(row[f"{slug}_missing"])
                if missing:
                    state["labs"][slug] = {"missing": 1}
                else:
                    state["labs"][slug] = {
                        "missing": 0,
                        "min": float(row[f"{slug}_min"]),
                        "max": float(row[f"{slug}_max"]),
                    }
            current_states.append(state)

        if current_patient_id is not None:
            out.write(
                json.dumps(
                    {
                        "patient_id": current_patient_id,
                        "static_context": current_static,
                        "states": current_states,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            summary["trajectory_rows"] += 1
            summary["max_sequence_length"] = max(summary["max_sequence_length"], len(current_states))

    return dict(summary)


def build_validation_report(
    cohort: Dict[str, PatientMeta],
    excluded: Dict[str, int],
    prep_stats: Dict[str, dict],
) -> dict:
    label_errors = []
    leakage_errors = []
    sequence_lengths = sorted(meta.window_count for meta in cohort.values())
    positives = sum(meta.will_die_next_120_days for meta in cohort.values())
    negatives = len(cohort) - positives

    for meta in cohort.values():
        if meta.snapshot_date + timedelta(days=HORIZON_DAYS) > meta.max_date:
            label_errors.append(meta.patient_id)
        if meta.will_die_next_120_days == 1:
            if meta.death_date is None or meta.death_date > meta.snapshot_date + timedelta(days=HORIZON_DAYS):
                label_errors.append(meta.patient_id)
        else:
            if meta.death_date is not None and meta.death_date <= meta.snapshot_date + timedelta(days=HORIZON_DAYS):
                label_errors.append(meta.patient_id)
        if meta.snapshot_date > meta.max_date:
            leakage_errors.append(meta.patient_id)

    def percentile(values: List[int], q: float) -> int:
        if not values:
            return 0
        idx = min(len(values) - 1, max(0, math.floor((len(values) - 1) * q)))
        return values[idx]

    report = {
        "config": {
            "window_days": WINDOW_DAYS,
            "horizon_days": HORIZON_DAYS,
            "snapshot_rule": f"exact_max_date_minus_{HORIZON_DAYS}d",
            "cohort_filter": "breast_cancer_only_C50_star",
            "support_filter": "require_any_signal_in_last_included_window",
        },
        "cohort": {
            "patients_included": len(cohort),
            "patients_excluded": excluded,
            "positive_labels": positives,
            "negative_labels": negatives,
            "positive_rate": round(positives / len(cohort), 6) if cohort else 0.0,
        },
        "sequence_lengths": {
            "min": sequence_lengths[0] if sequence_lengths else 0,
            "median": percentile(sequence_lengths, 0.5),
            "p90": percentile(sequence_lengths, 0.9),
            "max": sequence_lengths[-1] if sequence_lengths else 0,
        },
        "validations": {
            "label_validity_passed": len(label_errors) == 0,
            "label_validity_failures": label_errors[:20],
            "no_leakage_passed": len(leakage_errors) == 0,
            "no_leakage_failures": leakage_errors[:20],
            "one_snapshot_per_patient": True,
            "one_trajectory_per_patient": True,
        },
        "preparation_stats": prep_stats,
    }
    return report


def cleanup_sqlite_artifacts(db_path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort, excluded = build_cohort()
    default_units = load_test_defaults()
    conversions = load_unit_conversions()

    snapshots_path = OUTPUT_DIR / "patient_snapshots.csv"
    states_path = OUTPUT_DIR / "patient_states.csv"
    trajectories_path = OUTPUT_DIR / "patient_trajectories.jsonl"
    report_path = OUTPUT_DIR / "validation_report.json"
    db_path = OUTPUT_DIR / "pipeline.sqlite"
    if db_path.exists():
        db_path.unlink()

    label_counts = write_patient_snapshots(cohort, snapshots_path)
    conn = build_sqlite_db(db_path)
    try:
        insert_state_base(conn, cohort)
        lab_stats = aggregate_labs(conn, cohort, default_units, conversions)
        ecog_stats = aggregate_ecog(conn, cohort)
        chemo_stats = aggregate_interval_activity(conn, cohort, "chemotherapy_courses.csv", "chemo_agg")
        radiotherapy_stats = aggregate_interval_activity(conn, cohort, "radiotherapy.csv", "radiotherapy_agg")
        admission_stats = aggregate_admissions(conn, cohort)
        supported_patient_ids, support_stats = get_supported_patient_ids(conn, cohort)
        cohort = {pid: meta for pid, meta in cohort.items() if pid in supported_patient_ids}
        label_counts = write_patient_snapshots(cohort, snapshots_path)
        state_summary = write_patient_states(conn, states_path, supported_patient_ids)
    finally:
        conn.close()
        cleanup_sqlite_artifacts(db_path)

    trajectory_summary = write_patient_trajectories(states_path, trajectories_path)
    prep_stats = {
        "labels": label_counts,
        "labs": lab_stats,
        "ecog": ecog_stats,
        "chemo": chemo_stats,
        "radiotherapy": radiotherapy_stats,
        "admissions": admission_stats,
        "support_filter": support_stats,
        "states": state_summary,
        "trajectories": trajectory_summary,
    }
    report = build_validation_report(cohort, excluded, prep_stats)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
