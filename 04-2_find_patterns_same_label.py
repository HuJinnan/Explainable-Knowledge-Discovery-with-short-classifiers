import csv
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit


MINING_TRAIN_IPS = Path(r"D:\PythonProject\paper\window4_cut_0_days_all_patients\split_data\mining_train_subseq_ips.jsonl")
SUPPORT_TRAIN_IPS = Path(r"D:\PythonProject\paper\window4_cut_0_days_all_patients\split_data\eval_train_subseq_ips.jsonl")
TOP_INTERVALS_CSV = Path(r"D:\PythonProject\sequence_pattern_structures\find_top_intervals\top_intervals.csv")
OUTPUT_DIR = Path(r"D:\PythonProject\paper\patterns_same_label_11features_all_patient_cut_0_days")

NUMERIC_FEATURES = ["plt","neutrophils","creatinine","urea","ast","total_bilirubin","albumin","ldh","ecog",]
CATEGORICAL_FEATURES = ["admissions", "chemo_active"]

K = 10
ONLY_LABEL0_SEEDS = 0

INTERVAL_WIDTH_THRESHOLD_RATIO = 0.6
MISSING_NUMERIC_PENALTY = 0.25
NUM_WORKERS = 20

NUM_NUMERIC = len(NUMERIC_FEATURES)
NUM_CATEGORICAL = len(CATEGORICAL_FEATURES)
NUMERIC_DIM = NUM_NUMERIC * 3
STATE_DIM = NUMERIC_DIM + NUM_CATEGORICAL


def parse_admissions(val: Any) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if val == "2+":
            return 2
        if val.isdigit():
            return int(val)
    return 0


def parse_bool(val: Any) -> int:
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return 1 if val.lower() in ("1", "true", "yes") else 0
    return 0


def finite_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def sample_id_for(sample: dict, index: int) -> str:
    for key in ("sample_id", "block_id", "id"):
        if sample.get(key):
            return str(sample[key])
    patient_id = sample.get("source_patient_id") or sample.get("patient_id") or ""
    block_start = sample.get("block_start") or sample.get("block_start_date") or ""
    block_end = sample.get("block_end") or sample.get("block_end_date") or ""
    if patient_id or block_start or block_end:
        return f"{patient_id}::{block_start}-{block_end}"
    return f"sample_{index}"


def load_samples(ips_path: Path) -> List[dict]:
    samples = []
    with ips_path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_top_intervals_from_csv(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    top_intervals = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row["feature"].strip()
            minv = float(row["min"])
            maxv = float(row["max"])
            if not np.isfinite(minv) or not np.isfinite(maxv):
                raise ValueError(f"Top interval for {feature} contains non-finite values: {(minv, maxv)}")
            if minv > maxv:
                raise ValueError(f"Top interval for {feature} has min > max: {(minv, maxv)}")
            top_intervals[feature] = (minv, maxv)

    missing = set(NUMERIC_FEATURES) - set(top_intervals)
    if missing:
        raise ValueError(f"Top interval CSV is missing numeric features: {missing}")
    return top_intervals


def state_to_array(state: dict, top_intervals: Dict[str, Tuple[float, float]]) -> np.ndarray:
    vec = []
    for feat in NUMERIC_FEATURES:
        top_min, top_max = top_intervals[feat]
        if feat == "ecog":
            val = state.get("ecog")
            if isinstance(val, dict):
                raw_min = finite_float_or_none(val.get("min"))
                raw_max = finite_float_or_none(val.get("max"))
                is_missing = val.get("missing") == 1
            else:
                raw_min = finite_float_or_none(val)
                raw_max = raw_min
                is_missing = val is None
        else:
            value = state.get("labs", {}).get(feat, {"missing": 1})
            raw_min = finite_float_or_none(value.get("min"))
            raw_max = finite_float_or_none(value.get("max"))
            is_missing = value.get("missing") == 1

        if is_missing or raw_min is None or raw_max is None or raw_min > raw_max:
            minv, maxv = top_min, top_max
            missing = 1.0
        else:
            minv, maxv = raw_min, raw_max
            missing = 0.0
        vec.extend([minv, maxv, missing])

    for feat in CATEGORICAL_FEATURES:
        if feat == "admissions":
            val = parse_admissions(state.get("admissions", 0))
        else:
            val = parse_bool(state.get(feat, 0))
        vec.append(float(val))
    return np.array(vec, dtype=np.float32)


def sequence_to_array(seq: List[dict], top_intervals: Dict[str, Tuple[float, float]]) -> np.ndarray:
    return np.array([state_to_array(s, top_intervals) for s in seq], dtype=np.float32)


def prepare_data(
    samples: List[dict],
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    sequences = []
    labels = []
    patient_ids = []
    sample_ids = []
    for idx, sample in enumerate(samples):
        sequences.append(sequence_to_array(sample["states"], top_intervals))
        labels.append(int(sample["label"]))
        patient_ids.append(str(sample.get("source_patient_id", sample.get("patient_id", ""))))
        sample_ids.append(sample_id_for(sample, idx))
    return np.stack(sequences, axis=0), np.array(labels, dtype=int), patient_ids, sample_ids


def precompute_dataset_stats(labels: np.ndarray, patient_ids: List[str]) -> Tuple[int, int, int, int]:
    total_samples_0 = int(np.sum(labels == 0))
    total_samples_1 = int(np.sum(labels == 1))
    patients_0 = {pid for pid, label in zip(patient_ids, labels) if label == 0}
    patients_1 = {pid for pid, label in zip(patient_ids, labels) if label == 1}
    return total_samples_0, total_samples_1, len(patients_0), len(patients_1)


@njit
def numeric_distance_numba(min1, max1, miss1, min2, max2, miss2, top_width) -> float:
    if not (np.isfinite(min1) and np.isfinite(max1) and np.isfinite(min2) and np.isfinite(max2)):
        return MISSING_NUMERIC_PENALTY
    if miss1 == 1.0 and miss2 == 1.0:
        return 0.0
    if miss1 == 1.0 or miss2 == 1.0:
        return MISSING_NUMERIC_PENALTY
    mid1 = (min1 + max1) * 0.5
    mid2 = (min2 + max2) * 0.5
    w1 = max1 - min1
    w2 = max2 - min2
    return (abs(mid1 - mid2) / top_width) + (abs(w1 - w2) / top_width)


@njit
def state_distance_numba(state1, state2, numeric_top_widths) -> float:
    total = 0.0
    for i in range(NUM_NUMERIC):
        base = i * 3
        min1, max1, miss1 = state1[base], state1[base + 1], state1[base + 2]
        min2, max2, miss2 = state2[base], state2[base + 1], state2[base + 2]
        total += numeric_distance_numba(min1, max1, miss1, min2, max2, miss2, numeric_top_widths[i])

    """
    cat_start = NUM_NUMERIC * 3
    for i in range(NUM_CATEGORICAL):
        total += abs(state1[cat_start + i] - state2[cat_start + i])
    return total / (NUM_NUMERIC + NUM_CATEGORICAL)
    """

    # 01 Knn logic
    cat_start = NUM_NUMERIC * 3
    for i in range(NUM_CATEGORICAL):
        diff = abs(state1[cat_start + i] - state2[cat_start + i])
        if i == 0:  # admissions
            total += min(diff, 3.0) / 3.0
        else:  # chemo_active
            total += diff
    return total / (NUM_NUMERIC + NUM_CATEGORICAL)


@njit
def subsequence_distance_numba(seq1, seq2, numeric_top_widths) -> float:
    total = 0.0
    for t in range(seq1.shape[0]):
        total += state_distance_numba(seq1[t], seq2[t], numeric_top_widths)
    return total / seq1.shape[0]


@njit
def meet_numeric_numba(values, top_min, top_max):
    minv = np.inf
    maxv = -np.inf
    has_missing = False
    for i in range(values.shape[0]):
        if values[i, 2] == 1.0:
            has_missing = True
        if not (np.isfinite(values[i, 0]) and np.isfinite(values[i, 1])):
            has_missing = True
            if top_min < minv:
                minv = top_min
            if top_max > maxv:
                maxv = top_max
            continue
        if values[i, 0] < minv:
            minv = values[i, 0]
        if values[i, 1] > maxv:
            maxv = values[i, 1]

    if not (np.isfinite(minv) and np.isfinite(maxv)):
        minv = top_min
        maxv = top_max
    return minv, maxv, has_missing


@njit
def generalize_pattern_numba(neighborhood, top_intervals_min, top_intervals_max, length, k):
    pattern = np.zeros((length, STATE_DIM), dtype=np.float32)
    for pos in range(length):
        for i in range(NUM_NUMERIC):
            base = i * 3
            vals = np.zeros((k, 3), dtype=np.float32)
            for m in range(k):
                vals[m, 0] = neighborhood[m, pos, base]
                vals[m, 1] = neighborhood[m, pos, base + 1]
                vals[m, 2] = neighborhood[m, pos, base + 2]
            minv, maxv, has_missing = meet_numeric_numba(vals, top_intervals_min[i], top_intervals_max[i])
            pattern[pos, base] = minv
            pattern[pos, base + 1] = maxv
            pattern[pos, base + 2] = 1.0 if has_missing else 0.0

    cat_start = NUM_NUMERIC * 3
    for pos in range(length):
        for i in range(NUM_CATEGORICAL):
            idx = cat_start + i
            vals = np.zeros(k, dtype=np.float32)
            for m in range(k):
                vals[m] = neighborhood[m, pos, idx]
            pattern[pos, idx] = np.min(vals)
    return pattern


@njit
def pattern_matches_subseq_numba(pattern, subseq):
    for pos in range(pattern.shape[0]):
        p_state = pattern[pos]
        s_state = subseq[pos]
        for i in range(NUM_NUMERIC):
            base = i * 3
            p_min = p_state[base]
            p_max = p_state[base + 1]
            s_min = s_state[base]
            s_max = s_state[base + 1]
            if not (np.isfinite(p_min) and np.isfinite(p_max) and np.isfinite(s_min) and np.isfinite(s_max)):
                return False
            if not (p_min <= s_min and p_max >= s_max):
                return False

        cat_start = NUM_NUMERIC * 3
        for i in range(NUM_CATEGORICAL):
            if not (p_state[cat_start + i] <= s_state[cat_start + i]):
                return False
    return True


@njit
def passes_interval_usefulness_numba(pattern, top_intervals_min, top_intervals_max):
    threshold = INTERVAL_WIDTH_THRESHOLD_RATIO
    for pos in range(pattern.shape[0]):
        for i in range(NUM_NUMERIC):
            base = i * 3
            minv = pattern[pos, base]
            maxv = pattern[pos, base + 1]
            if not (np.isfinite(minv) and np.isfinite(maxv)):
                continue
            top_width = max(top_intervals_max[i] - top_intervals_min[i], 1e-12)
            if maxv - minv < threshold * top_width:
                return True
    return False


def process_seed(
    seed_idx: int,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    train_patient_ids: List[str],
    train_sample_ids: List[str],
    numeric_top_widths: np.ndarray,
    top_intervals_min: np.ndarray,
    top_intervals_max: np.ndarray,
    k: int,
) -> Optional[dict]:
    seed_seq = train_data[seed_idx]
    seed_patient = train_patient_ids[seed_idx]
    seed_label = train_labels[seed_idx]

    distances = np.empty(train_data.shape[0], dtype=np.float32)
    for i in range(train_data.shape[0]):
        distances[i] = subsequence_distance_numba(seed_seq, train_data[i], numeric_top_widths)

    same_label_indices = [i for i in range(train_data.shape[0]) if train_labels[i] == seed_label]
    sorted_same_label = sorted(same_label_indices, key=lambda i: distances[i])

    source_indices = [seed_idx]
    seen_patients = {seed_patient}
    for idx in sorted_same_label:
        if idx == seed_idx:
            continue
        patient_id = train_patient_ids[idx]
        if patient_id in seen_patients:
            continue
        seen_patients.add(patient_id)
        source_indices.append(idx)
        if len(source_indices) == k:
            break

    neighborhood = train_data[source_indices]
    pattern = generalize_pattern_numba(
        neighborhood,
        top_intervals_min,
        top_intervals_max,
        train_data.shape[1],
        len(source_indices),
    )
    return {
        "pattern": pattern,
        "source_indices": source_indices,
        "source_patient_ids": [train_patient_ids[i] for i in source_indices],
        "source_sample_ids": [train_sample_ids[i] for i in source_indices],
    }


def merge_source_lists(target: dict, source: dict) -> None:
    known = set(zip(target["source_patient_ids"], target["source_sample_ids"]))
    for patient_id, sample_id in zip(source["source_patient_ids"], source["source_sample_ids"]):
        item = (patient_id, sample_id)
        if item in known:
            continue
        known.add(item)
        target["source_patient_ids"].append(patient_id)
        target["source_sample_ids"].append(sample_id)


def mine_patterns_mixed(
    train_data,
    train_labels,
    train_patient_ids,
    train_sample_ids,
    numeric_top_widths,
    top_intervals_min,
    top_intervals_max,
    num_workers,
    k,
):
    if ONLY_LABEL0_SEEDS == 1:
        all_indices = [i for i in range(train_data.shape[0]) if train_labels[i] == 0]
    else:
        all_indices = list(range(train_data.shape[0]))
    if len(all_indices) < 2:
        return []

    print(f"Mining same-label patterns with {num_workers} workers over {len(all_indices)} seeds")
    worker = partial(
        process_seed,
        train_data=train_data,
        train_labels=train_labels,
        train_patient_ids=train_patient_ids,
        train_sample_ids=train_sample_ids,
        numeric_top_widths=numeric_top_widths,
        top_intervals_min=top_intervals_min,
        top_intervals_max=top_intervals_max,
        k=k,
    )
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker, all_indices)

    unique_patterns = {}
    for rec in results:
        if rec is None:
            continue
        key = rec["pattern"].tobytes()
        if key not in unique_patterns:
            unique_patterns[key] = {
                "pattern": rec["pattern"],
                "source_patient_ids": list(rec["source_patient_ids"]),
                "source_sample_ids": list(rec["source_sample_ids"]),
            }
        else:
            merge_source_lists(unique_patterns[key], rec)

    candidate_records = []
    for rec in unique_patterns.values():
        if not passes_interval_usefulness_numba(rec["pattern"], top_intervals_min, top_intervals_max):
            continue
        candidate_records.append(rec)
    print(f"Patterns after interval usefulness filter: {len(candidate_records)}")
    return candidate_records


def compute_dedup_metrics_excluding_sources(
    pattern: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[str],
    source_patient_ids: List[str],
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[int, int, float, float, float, float]:
    source_patients = set(source_patient_ids)
    matched_patients_0 = set()
    matched_patients_1 = set()
    for idx in range(len(data)):
        patient_id = patient_ids[idx]
        if patient_id in source_patients:
            continue
        if not pattern_matches_subseq_numba(pattern, data[idx]):
            continue
        if labels[idx] == 0:
            matched_patients_0.add(patient_id)
        else:
            matched_patients_1.add(patient_id)

    support0 = len(matched_patients_0)
    support1 = len(matched_patients_1)
    total = support0 + support1
    precision0 = support0 / total if total else 0.0
    precision1 = support1 / total if total else 0.0
    recall0 = support0 / total_patients_0 if total_patients_0 else 0.0
    recall1 = support1 / total_patients_1 if total_patients_1 else 0.0
    return support0, support1, precision0, precision1, recall0, recall1


def pattern_array_to_dict(pattern: np.ndarray) -> List[dict]:
    state_dicts = []
    for pos in range(pattern.shape[0]):
        state = pattern[pos]
        state_dict = {}
        for i, feat in enumerate(NUMERIC_FEATURES):
            base = i * 3
            state_dict[feat] = {
                "min": float(state[base]),
                "max": float(state[base + 1]),
                "has_missing_source": bool(state[base + 2] == 1.0),
            }
        cat_start = NUM_NUMERIC * 3
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            val = int(state[cat_start + i])
            state_dict[feat] = val
        state_dicts.append(state_dict)
    return state_dicts


def pattern_dict_to_array(states: List[dict]) -> np.ndarray:
    pattern = np.zeros((len(states), STATE_DIM), dtype=np.float32)
    for pos, state in enumerate(states):
        for i, feat in enumerate(NUMERIC_FEATURES):
            base = i * 3
            values = state[feat]
            pattern[pos, base] = float(values["min"])
            pattern[pos, base + 1] = float(values["max"])
            pattern[pos, base + 2] = 1.0 if values.get("has_missing_source") else 0.0

        cat_start = NUM_NUMERIC * 3
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            pattern[pos, cat_start + i] = float(state[feat])
    return pattern


def load_existing_pattern_records(patterns_jsonl: Path) -> List[dict]:
    records: List[dict] = []
    with patterns_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(
                {
                    "pattern_id": obj["pattern_id"],
                    "pattern": pattern_dict_to_array(obj["states"]),
                    "source_patient_ids": list(obj.get("source_patient_ids", [])),
                    "source_sample_ids": list(obj.get("source_sample_ids", [])),
                }
            )
    return records


def build_pattern_records_from_candidate_records(candidate_records: List[dict]) -> List[dict]:
    records: List[dict] = []
    for idx, rec in enumerate(candidate_records):
        records.append(
            {
                "pattern_id": f"pattern_{idx}",
                "pattern": rec["pattern"],
                "source_patient_ids": list(rec["source_patient_ids"]),
                "source_sample_ids": list(rec["source_sample_ids"]),
            }
        )
    return records


def compute_metrics_rows(
    pattern_records: List[dict],
    support_data: np.ndarray,
    support_labels: np.ndarray,
    support_patient_ids: List[str],
    support_patients_0: int,
    support_patients_1: int,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for rec in pattern_records:
        metrics = compute_dedup_metrics_excluding_sources(
            rec["pattern"],
            support_data,
            support_labels,
            support_patient_ids,
            rec["source_patient_ids"],
            support_patients_0,
            support_patients_1,
        )
        support0, support1, _, _, _, _ = metrics
        if support0 + support1 == 0:
            continue
        rows.append([rec["pattern_id"], *metrics])
    return rows


def write_pattern_outputs(
    pattern_records: List[dict],
    metrics_rows: List[List[Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "patterns.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in pattern_records:
            f.write(
                json.dumps(
                    {
                        "pattern_id": rec["pattern_id"],
                        "states": pattern_array_to_dict(rec["pattern"]),
                        "source_patient_ids": rec["source_patient_ids"],
                        "source_sample_ids": rec["source_sample_ids"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"JSONL saved: {jsonl_path}")

    metrics_path = output_dir / "pattern_metrics.csv"
    metrics_header = [
        "pattern_id",
        "train_dedup_support0_excluding_sources",
        "train_dedup_support1_excluding_sources",
        "train_dedup_precision0_excluding_sources",
        "train_dedup_precision1_excluding_sources",
        "train_dedup_recall0_excluding_sources",
        "train_dedup_recall1_excluding_sources",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics_header)
        writer.writerows(metrics_rows)
    print(f"CSV saved: {metrics_path}")

    source_path = output_dir / "pattern_sources.csv"
    source_csv_rows: List[List[Any]] = []
    for rec in pattern_records:
        for source_order, (patient_id, sample_id) in enumerate(
            zip(rec["source_patient_ids"], rec["source_sample_ids"]),
            start=1,
        ):
            source_csv_rows.append([rec["pattern_id"], source_order, patient_id, sample_id])
    with source_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pattern_id", "source_order", "source_patient_id", "source_sample_id"])
        writer.writerows(source_csv_rows)
    print(f"Pattern sources CSV saved: {source_path}")


def main() -> None:
    start_time = time.time()
    print("Loading mining-train data...")
    mining_train_samples = load_samples(MINING_TRAIN_IPS)
    print(f"Mining-train samples: {len(mining_train_samples)}")

    print("Loading support-train data...")
    support_train_samples = load_samples(SUPPORT_TRAIN_IPS)
    print(f"Support-train samples: {len(support_train_samples)}")

    print(f"Loading top intervals from {TOP_INTERVALS_CSV}")
    top_intervals = load_top_intervals_from_csv(TOP_INTERVALS_CSV)
    mining_train_data, mining_train_labels, mining_train_patient_ids, mining_train_sample_ids = prepare_data(
        mining_train_samples, top_intervals
    )
    support_train_data, support_train_labels, support_train_patient_ids, _ = prepare_data(
        support_train_samples, top_intervals
    )
    print(f"Mining-train data shape: {mining_train_data.shape}")
    print(f"Mining-train label distribution: {np.unique(mining_train_labels, return_counts=True)}")
    print(f"Support-train data shape: {support_train_data.shape}")
    print(f"Support-train label distribution: {np.unique(support_train_labels, return_counts=True)}")

    numeric_top_widths = np.array(
        [max(top_intervals[feat][1] - top_intervals[feat][0], 1e-12) for feat in NUMERIC_FEATURES],
        dtype=np.float32,
    )
    top_intervals_min = np.array([top_intervals[feat][0] for feat in NUMERIC_FEATURES], dtype=np.float32)
    top_intervals_max = np.array([top_intervals[feat][1] for feat in NUMERIC_FEATURES], dtype=np.float32)

    _, _, support_train_patients_0, support_train_patients_1 = precompute_dataset_stats(
        support_train_labels, support_train_patient_ids
    )

    candidate_records = mine_patterns_mixed(
        mining_train_data,
        mining_train_labels,
        mining_train_patient_ids,
        mining_train_sample_ids,
        numeric_top_widths,
        top_intervals_min,
        top_intervals_max,
        NUM_WORKERS,
        K,
    )

    pattern_records = build_pattern_records_from_candidate_records(candidate_records)
    metrics_rows = compute_metrics_rows(
        pattern_records,
        support_train_data,
        support_train_labels,
        support_train_patient_ids,
        support_train_patients_0,
        support_train_patients_1,
    )
    kept_pattern_ids = {row[0] for row in metrics_rows}
    pattern_records = [rec for rec in pattern_records if rec["pattern_id"] in kept_pattern_ids]
    write_pattern_outputs(pattern_records, metrics_rows, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    mp.freeze_support()
    main()
