#!/usr/bin/env python3
"""
独立脚本：从原始数据表计算数值特征的全局顶区间（最小值、最大值），
仅考虑符合条件的乳腺癌患者（诊断代码 C50.*，年龄≥18岁，有诊断日期等）。
"""

import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import date, datetime

DATA_DIR = Path(r"D:\PythonProject\sequence_pattern_structures - 副本\data")
OUTPUT_CSV = Path(r"D:\PythonProject\sequence_pattern_structures - 副本\find_top_intervals\top_intervals.csv")

# 需要计算顶区间的数值特征列表
NUMERIC_FEATURES = ["hb", "wbc", "plt", "neutrophils", "creatinine", "urea", "alt", "ast", "total_bilirubin", "albumin", "ldh", "esr", "ecog"]

def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()

def age_in_years(birth_date: date, on_date: date) -> int:
    years = on_date.year - birth_date.year
    if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years

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

def build_cohort_patient_set() -> set:
    """返回符合 breast cancer cohort 且年龄>=18的患者ID集合"""
    cancer_lookup = load_cancer_type_lookup()
    earliest_diagnoses = {}
    # 加载最早诊断
    with (DATA_DIR / "diagnoses.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            diag_date = parse_date(row["diag_establish_date"])
            patient_id = row["patient_id"]
            if diag_date is None:
                continue
            existing = earliest_diagnoses.get(patient_id)
            if existing is not None and existing["diagnosis_date"] <= diag_date:
                continue
            earliest_diagnoses[patient_id] = {
                "diagnosis_date": diag_date,
                "diag_code": (row["diag_code"] or "").strip().upper(),
                "stage_normalized": normalize_stage(row["diag_stage"]),
                "metastasis_flag": 1 if (row["diag_metastases_localization"] or "").strip() else 0,
            }

    cohort_ids = set()
    with (DATA_DIR / "patients.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            diagnosis = earliest_diagnoses.get(patient_id)
            if diagnosis is None:
                continue
            if not diagnosis["diag_code"].startswith("C50"):
                continue
            birth_date = parse_date(row["birth_date"])
            if birth_date is None:
                continue
            age = age_in_years(birth_date, diagnosis["diagnosis_date"])
            if age < 18:
                continue
            cohort_ids.add(patient_id)
    return cohort_ids

def compute_top_intervals() -> Dict[str, Tuple[float, float]]:
    """主函数：计算所有数值特征的全局最小值和最大值"""
    print("正在收集符合条件的乳腺癌患者...")
    cohort_patients = build_cohort_patient_set()
    print(f"符合条件的患者数: {len(cohort_patients)}")

    default_units = load_test_defaults()
    conversions = load_unit_conversions()

    # 实验室测试代码到特征名的映射
    lab_defs = {
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

    intervals = {feat: [] for feat in NUMERIC_FEATURES}

    # 处理实验室数据
    print("处理实验室数据...")
    with (DATA_DIR / "laboratory_tests.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patient_id"]
            if patient_id not in cohort_patients:
                continue
            test_code = (row["test_type_code"] or "").strip()
            lab_def = lab_defs.get(test_code)
            if lab_def is None:
                continue
            slug = lab_def[0]
            converted = convert_lab_value(
                test_code=test_code,
                raw_value=row["test_result"],
                unit_code=row["value_unit_code"],
                default_units=default_units,
                conversions=conversions,
            )
            if converted is not None:
                intervals[slug].append(converted)

    # 处理 ECOG
    print("处理 ECOG 数据...")
    with (DATA_DIR / "ecog_performance_status.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patient_id"]
            if patient_id not in cohort_patients:
                continue
            try:
                score = int(float(row["ecog_score"]))
            except (ValueError, TypeError):
                continue
            if score == 5:
                continue
            score = max(0, min(score, 4))
            intervals["ecog"].append(score)

    # 计算全局最小和最大
    top_intervals = {}
    for feat, values in intervals.items():
        if values:
            top_intervals[feat] = (min(values), max(values))
        else:
            # 如果没有数据，则使用默认范围（通常不会发生）
            if feat == "ecog":
                top_intervals[feat] = (0.0, 4.0)
            else:
                top_intervals[feat] = (0.0, 1.0)
            print(f"警告: 特征 {feat} 没有观测值，使用默认顶区间 {top_intervals[feat]}")
    return top_intervals

def save_top_intervals_to_csv(top_intervals: Dict[str, Tuple[float, float]], output_path: Path):
    """将顶区间保存为 CSV 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "min", "max"])
        for feat, (minv, maxv) in top_intervals.items():
            writer.writerow([feat, minv, maxv])
    print(f"顶区间已保存到: {output_path}")

def main():
    print("开始计算顶区间...")
    top_intervals = compute_top_intervals()
    print("\n计算得到的顶区间:")
    for feat, (minv, maxv) in top_intervals.items():
        print(f"  {feat}: [{minv:.4f}, {maxv:.4f}]")
    save_top_intervals_to_csv(top_intervals, OUTPUT_CSV)

if __name__ == "__main__":
    main()