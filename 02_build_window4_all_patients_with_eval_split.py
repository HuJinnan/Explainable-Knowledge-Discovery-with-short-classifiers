from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


WINDOW_LEN = 4
LABEL_BOUNDARIES: List[Tuple[int, int, float]] = [
    (0, 0, 120),
    (1, 121, float("inf")),
]
USE_ALL_PATIENT = 1
CUT_DAYS = 0

STATES_CSV = Path(
    r"D:\PythonProject\sequence_pattern_structures\prepared\breast_cancer_single_snapshot_0d_last_window_supported\patient_states.csv"
)
SNAPSHOTS_CSV = Path(
    r"D:\PythonProject\sequence_pattern_structures\prepared\breast_cancer_single_snapshot_0d_last_window_supported\patient_snapshots.csv"
)

ROOT_DIR = Path(r"D:\PythonProject\paper\window4_cut_0_days_all_patients")
UNSPLIT_DIR = ROOT_DIR / "raw_data"
SPLIT_DIR = ROOT_DIR / "split_data"

RANDOM_SEED = 42
FULL_TRAIN_RATIO = 0.8

MINING_UNSPLIT_IPS = UNSPLIT_DIR / "subseq_ips_window4_cut_0_days_all_patients.jsonl"
MINING_UNSPLIT_CSV = UNSPLIT_DIR / "subseq_flattened_window4_cut_0_days_all_patients.csv"
EVAL_UNSPLIT_IPS = UNSPLIT_DIR / "nearest_window_subseq_ips_window4_cut_0_days_all_patients.jsonl"
EVAL_UNSPLIT_CSV = UNSPLIT_DIR / "nearest_window_subseq_flattened_window4_cut_0_days_all_patients.csv"

MINING_TRAIN_IPS = SPLIT_DIR / "mining_train_subseq_ips.jsonl"
MINING_TRAIN_CSV = SPLIT_DIR / "mining_train_subseq_flattened.csv"
MINING_TEST_IPS = SPLIT_DIR / "mining_test_subseq_ips.jsonl"
MINING_TEST_CSV = SPLIT_DIR / "mining_test_subseq_flattened.csv"
EVAL_TRAIN_IPS = SPLIT_DIR / "eval_train_subseq_ips.jsonl"
EVAL_TRAIN_CSV = SPLIT_DIR / "eval_train_subseq_flattened.csv"
EVAL_TEST_IPS = SPLIT_DIR / "eval_test_subseq_ips.jsonl"
EVAL_TEST_CSV = SPLIT_DIR / "eval_test_subseq_flattened.csv"


def parse_date(date_str: str) -> Optional[date]:
    if not date_str or date_str.strip() == "":
        return None
    date_str = date_str.strip().replace("/", "-")
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def days_between(date1: date, date2: date) -> int:
    return (date2 - date1).days


def get_label_by_days(days: int) -> int:
    for label, low, high in LABEL_BOUNDARIES:
        if low <= days <= high:
            return label
    return -1


def states_are_continuous(states: List[dict]) -> bool:
    if len(states) < 2:
        return True
    idxs = [s.get("window_index") for s in states]
    if any(i is None for i in idxs):
        return False
    return all(idxs[i] == idxs[0] + i for i in range(len(idxs)))


def flatten_state(state: dict, pos: int) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    labs = state.get("labs", {})
    for lab_name, lab_data in labs.items():
        if isinstance(lab_data, dict):
            missing = lab_data.get("missing", 1)
            flat[f"{lab_name}_min_{pos}"] = lab_data.get("min") if not missing else None
            flat[f"{lab_name}_max_{pos}"] = lab_data.get("max") if not missing else None
            flat[f"{lab_name}_missing_{pos}"] = missing
    flat[f"ecog_{pos}"] = state.get("ecog")
    flat[f"chemo_active_{pos}"] = state.get("chemo_active")
    flat[f"radiotherapy_active_{pos}"] = state.get("radiotherapy_active")
    flat[f"admissions_{pos}"] = state.get("admissions")
    return flat


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_flattened_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label", "source_patient_id"])
        return
    all_cols = set()
    for row in rows:
        all_cols.update(row.keys())
    base = ["sample_id", "label", "source_patient_id"]
    others = sorted([c for c in all_cols if c not in base])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base + others)
        writer.writeheader()
        for row in rows:
            full = {col: row.get(col, None) for col in base + others}
            writer.writerow(full)


def load_snapshot_info() -> Tuple[Dict[str, date], Dict[str, date]]:
    patient_death_date: Dict[str, date] = {}
    patient_snapshot_date: Dict[str, date] = {}
    with SNAPSHOTS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            death = parse_date(row.get("death_date"))
            if death:
                patient_death_date[pid] = death
            snap = parse_date(row.get("snapshot_date"))
            if snap:
                patient_snapshot_date[pid] = snap
    return patient_death_date, patient_snapshot_date


def load_patient_states(patient_death_date: Dict[str, date]) -> Dict[str, List[dict]]:
    patient_states: Dict[str, List[dict]] = defaultdict(list)
    with STATES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            if not USE_ALL_PATIENT and pid not in patient_death_date:
                continue

            state = {
                "window_index": int(row["window_index"]) if row["window_index"] else None,
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "labs": {},
                "ecog": None,
                "chemo_active": int(row["chemo_active"]) if row["chemo_active"] else 0,
                "radiotherapy_active": int(row["radiotherapy_active"]) if row["radiotherapy_active"] else 0,
                "admissions": row["admissions"],
            }
            lab_slugs = [
                "hb",
                "wbc",
                "plt",
                "neutrophils",
                "creatinine",
                "urea",
                "alt",
                "ast",
                "total_bilirubin",
                "albumin",
                "ldh",
                "esr",
            ]
            for slug in lab_slugs:
                min_key = f"{slug}_min"
                max_key = f"{slug}_max"
                miss_key = f"{slug}_missing"
                if miss_key in row:
                    missing = int(row[miss_key]) if row[miss_key] else 1
                    if missing:
                        state["labs"][slug] = {"missing": 1}
                    else:
                        try:
                            min_val = float(row[min_key]) if row[min_key] else None
                            max_val = float(row[max_key]) if row[max_key] else None
                            state["labs"][slug] = {"missing": 0, "min": min_val, "max": max_val}
                        except Exception:
                            state["labs"][slug] = {"missing": 1}
            ecog_val = row.get("ecog")
            if ecog_val and ecog_val.strip():
                try:
                    state["ecog"] = int(ecog_val)
                except Exception:
                    state["ecog"] = None
            patient_states[pid].append(state)

    for pid in patient_states:
        patient_states[pid].sort(key=lambda x: x.get("window_index", 0))
    return patient_states


def build_effective_reference_dates(
    patient_states: Dict[str, List[dict]],
    patient_death_date: Dict[str, date],
    patient_snapshot_date: Dict[str, date],
) -> Dict[str, date]:
    effective_dates: Dict[str, date] = {}
    for pid, states in patient_states.items():
        if pid in patient_snapshot_date:
            effective_dates[pid] = patient_snapshot_date[pid]
        elif pid in patient_death_date:
            effective_dates[pid] = patient_death_date[pid]
        elif USE_ALL_PATIENT:
            last_state = states[-1]
            last_end = parse_date(last_state.get("window_end"))
            if last_end is not None:
                effective_dates[pid] = last_end
    return effective_dates


def dedup_samples(samples_ips: List[dict], samples_flat: List[dict]) -> Tuple[List[dict], List[dict], int]:
    seen = set()
    dedup_ips: List[dict] = []
    dedup_flat: List[dict] = []
    for ips, flat in zip(samples_ips, samples_flat):
        states_key = json.dumps(ips["states"], sort_keys=True, ensure_ascii=False)
        key = (states_key, ips["label"])
        if key in seen:
            continue
        seen.add(key)
        dedup_ips.append(ips)
        dedup_flat.append(flat)
    removed = len(samples_ips) - len(dedup_ips)
    return dedup_ips, dedup_flat, removed


def generate_unsplit_datasets() -> Tuple[List[dict], List[dict], List[dict], List[dict], dict]:
    patient_death_date, patient_snapshot_date = load_snapshot_info()
    patient_states = load_patient_states(patient_death_date)
    effective_dates = build_effective_reference_dates(patient_states, patient_death_date, patient_snapshot_date)

    last_label = LABEL_BOUNDARIES[-1][0]
    mining_ips: List[dict] = []
    mining_flat: List[dict] = []
    eval_ips: List[dict] = []
    eval_flat: List[dict] = []
    sample_counter = 0

    for pid, states in patient_states.items():
        if pid not in effective_dates:
            continue
        reference_date = effective_dates[pid]
        n = len(states)
        if n < WINDOW_LEN:
            continue

        is_alive = (pid not in patient_death_date) and USE_ALL_PATIENT

        days_list = []
        for state in states:
            end = parse_date(state.get("window_end"))
            if end:
                days_list.append(days_between(end, reference_date))
            else:
                days_list.append(None)
        labels = [get_label_by_days(days) if days is not None else -1 for days in days_list]

        patient_samples_ips: List[dict] = []
        patient_samples_flat: List[dict] = []

        i = 0
        while i < n:
            if labels[i] == -1:
                i += 1
                continue
            start = i
            current_label = labels[i]
            j = i + 1
            while j < n and labels[j] == current_label and states_are_continuous(states[start : j + 1]):
                j += 1
            seg_len = j - start
            if seg_len >= WINDOW_LEN:
                num_windows = seg_len - WINDOW_LEN + 1
                for offset in range(num_windows):
                    win_start = start + offset
                    win_end = win_start + WINDOW_LEN - 1
                    window_states = states[win_start : win_end + 1]
                    if not states_are_continuous(window_states):
                        continue
                    sample_id = f"sample_{sample_counter}"
                    ips_obj = {
                        "sample_id": sample_id,
                        "source_patient_id": pid,
                        "label": current_label,
                        "states": window_states,
                    }
                    flat_row = {
                        "sample_id": sample_id,
                        "label": current_label,
                        "source_patient_id": pid,
                    }
                    for pos, state in enumerate(window_states, start=1):
                        flat_row.update(flatten_state(state, pos))
                    patient_samples_ips.append(ips_obj)
                    patient_samples_flat.append(flat_row)
                    sample_counter += 1
            i = j

        if is_alive:
            patient_samples_ips = [sample for sample in patient_samples_ips if sample["label"] == last_label]
            patient_samples_flat = [sample for sample in patient_samples_flat if sample["label"] == last_label]

        mining_ips.extend(patient_samples_ips)
        mining_flat.extend(patient_samples_flat)

        if patient_samples_ips:
            eval_ips.append(patient_samples_ips[-1])
            eval_flat.append(patient_samples_flat[-1])

    dedup_mining_ips, dedup_mining_flat, removed = dedup_samples(mining_ips, mining_flat)

    info = {
        "raw_mining_sample_count": len(mining_ips),
        "dedup_mining_sample_count": len(dedup_mining_ips),
        "dedup_removed_count": removed,
        "eval_single_window_sample_count": len(eval_ips),
        "dedup_mining_label_dist": dict(Counter(obj["label"] for obj in dedup_mining_ips)),
        "eval_single_window_label_dist": dict(Counter(obj["label"] for obj in eval_ips)),
    }
    return dedup_mining_ips, dedup_mining_flat, eval_ips, eval_flat, info


def derive_patient_split_from_mining_data(
    ips_rows: List[dict],
    seed: int,
    train_ratio: float,
) -> Tuple[List[str], List[str], List[int], List[int], dict]:
    random.seed(seed)

    patient_ids = [row["source_patient_id"] for row in ips_rows]
    labels = [row["label"] for row in ips_rows]

    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_label: Dict[str, int] = {}
    for idx, pid in enumerate(patient_ids):
        patient_to_indices[pid].append(idx)
        if pid not in patient_label:
            patient_label[pid] = labels[idx]

    unique_patients = list(patient_to_indices.keys())
    label_to_patients: Dict[int, List[str]] = defaultdict(list)
    for pid in unique_patients:
        label_to_patients[patient_label[pid]].append(pid)

    train_patients: List[str] = []
    test_patients: List[str] = []
    for label, patients in label_to_patients.items():
        random.shuffle(patients)
        split_point = int(len(patients) * train_ratio)
        train_patients.extend(patients[:split_point])
        test_patients.extend(patients[split_point:])

    random.shuffle(train_patients)
    random.shuffle(test_patients)

    train_indices: List[int] = []
    test_indices: List[int] = []
    for pid in train_patients:
        train_indices.extend(patient_to_indices[pid])
    for pid in test_patients:
        test_indices.extend(patient_to_indices[pid])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    split_info = {
        "train_ratio": train_ratio,
        "train_patient_count": len(train_patients),
        "test_patient_count": len(test_patients),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_label_dist": dict(Counter(labels[i] for i in train_indices)),
        "test_label_dist": dict(Counter(labels[i] for i in test_indices)),
    }
    return train_patients, test_patients, train_indices, test_indices, split_info


def split_by_patient_lists(
    ips_rows: List[dict],
    flat_rows: List[dict],
    train_patients: Set[str],
    test_patients: Set[str],
    shuffle_indices: bool,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], int]:
    train_indices: List[int] = []
    test_indices: List[int] = []
    dropped = 0

    for idx, ips in enumerate(ips_rows):
        pid = ips["source_patient_id"]
        if pid in train_patients:
            train_indices.append(idx)
        elif pid in test_patients:
            test_indices.append(idx)
        else:
            dropped += 1

    if shuffle_indices:
        rng = random.Random(seed)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    train_ips = [ips_rows[idx] for idx in train_indices]
    train_flat = [flat_rows[idx] for idx in train_indices]
    test_ips = [ips_rows[idx] for idx in test_indices]
    test_flat = [flat_rows[idx] for idx in test_indices]
    return train_ips, train_flat, test_ips, test_flat, dropped


def main() -> None:
    print("Generating unsplit mining/evaluation datasets...")
    mining_ips, mining_flat, eval_ips, eval_flat, generation_info = generate_unsplit_datasets()
    print(f"Mining unsplit samples after dedup: {len(mining_ips)}")
    print(f"Evaluation unsplit samples: {len(eval_ips)}")

    UNSPLIT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(MINING_UNSPLIT_IPS, mining_ips)
    write_flattened_csv(MINING_UNSPLIT_CSV, mining_flat)
    write_jsonl(EVAL_UNSPLIT_IPS, eval_ips)
    write_flattened_csv(EVAL_UNSPLIT_CSV, eval_flat)

    print("Deriving train/test patient split from all-windows unsplit data...")
    train_patients_list, test_patients_list, mining_train_indices, mining_test_indices, split_info = (
        derive_patient_split_from_mining_data(
            mining_ips,
            seed=RANDOM_SEED,
            train_ratio=FULL_TRAIN_RATIO,
        )
    )
    train_patients = set(train_patients_list)
    test_patients = set(test_patients_list)
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")

    mining_train_ips = [mining_ips[idx] for idx in mining_train_indices]
    mining_train_flat = [mining_flat[idx] for idx in mining_train_indices]
    mining_test_ips = [mining_ips[idx] for idx in mining_test_indices]
    mining_test_flat = [mining_flat[idx] for idx in mining_test_indices]
    mining_dropped = 0

    print("Applying derived patient split to evaluation dataset...")
    eval_train_ips, eval_train_flat, eval_test_ips, eval_test_flat, eval_dropped = split_by_patient_lists(
        eval_ips,
        eval_flat,
        train_patients,
        test_patients,
        shuffle_indices=False,
        seed=RANDOM_SEED,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(MINING_TRAIN_IPS, mining_train_ips)
    write_flattened_csv(MINING_TRAIN_CSV, mining_train_flat)
    write_jsonl(MINING_TEST_IPS, mining_test_ips)
    write_flattened_csv(MINING_TEST_CSV, mining_test_flat)
    write_jsonl(EVAL_TRAIN_IPS, eval_train_ips)
    write_flattened_csv(EVAL_TRAIN_CSV, eval_train_flat)
    write_jsonl(EVAL_TEST_IPS, eval_test_ips)
    write_flattened_csv(EVAL_TEST_CSV, eval_test_flat)

    split_info.update(
        {
            "random_seed": RANDOM_SEED,
            "mining_train_samples": len(mining_train_ips),
            "mining_train_label_dist": dict(Counter(obj["label"] for obj in mining_train_ips)),
            "mining_test_samples": len(mining_test_ips),
            "mining_test_label_dist": dict(Counter(obj["label"] for obj in mining_test_ips)),
            "eval_train_samples": len(eval_train_ips),
            "eval_train_label_dist": dict(Counter(obj["label"] for obj in eval_train_ips)),
            "eval_test_samples": len(eval_test_ips),
            "eval_test_label_dist": dict(Counter(obj["label"] for obj in eval_test_ips)),
            "mining_unmatched_samples": mining_dropped,
            "eval_unmatched_samples": eval_dropped,
        }
    )

    (UNSPLIT_DIR / "generation_info.json").write_text(
        json.dumps(generation_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (SPLIT_DIR / "split_info.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (SPLIT_DIR / "train_patient_ids.json").write_text(
        json.dumps(train_patients_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (SPLIT_DIR / "test_patient_ids.json").write_text(
        json.dumps(test_patients_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Unsplit outputs saved under: {UNSPLIT_DIR}")
    print(f"Split outputs saved under: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
