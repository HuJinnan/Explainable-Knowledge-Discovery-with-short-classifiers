from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit


PATTERNS_JSONL = Path(
    r"D:\PythonProject\paper\patterns_same_label_11features_all_patient_cut_0_days\patterns.jsonl"
)
TRAIN_IPS = Path(
    r"D:\PythonProject\paper\window4_cut_0_days_all_patients\split_data\eval_train_subseq_ips.jsonl"
)
TOP_INTERVALS_CSV = Path(
    r"D:\PythonProject\sequence_pattern_structures\find_top_intervals\top_intervals.csv"
)
OUTPUT_DIR = Path(
    r"D:\PythonProject\paper\single_feature_atoms_initial"
)
BEST_BALANCED_PRECISION_THRESHOLD = 0.8


@dataclass
class AtomRecord:
    atom_id: str
    feature: str
    feature_type: str
    states: List[dict]
    source_pattern_id: str
    source_patient_ids: List[str]
    source_sample_ids: List[str]


@dataclass
class EvaluatedAtom:
    atom: AtomRecord
    support0: int
    support1: int
    precision0: float
    precision1: float
    recall0: float
    recall1: float
    balanced_precision0: float
    balanced_precision1: float
    best_balanced_precision: float
    support_bits0: int
    support_bits1: int


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
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_top_intervals_from_csv(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    top_intervals = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row["feature"].strip()
            top_intervals[feature] = (float(row["min"]), float(row["max"]))
    return top_intervals


def discover_features_from_pattern(pattern_obj: dict) -> Tuple[List[str], List[str]]:
    numeric_features = []
    categorical_features = []
    first_state = pattern_obj["states"][0]
    for feature, value in first_state.items():
        if isinstance(value, dict):
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    return numeric_features, categorical_features


def normalize_numeric_state(state: dict) -> dict:
    return {
        "min": float(state["min"]),
        "max": float(state["max"]),
        "has_missing_source": bool(state.get("has_missing_source", False)),
    }


def normalize_categorical_state(value: Any) -> int:
    return int(value)


def build_atom_signature(feature: str, feature_type: str, states: List[dict]) -> str:
    payload = {
        "feature": feature,
        "feature_type": feature_type,
        "states": states,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def is_top_numeric_atom(states: List[dict], top_min: float, top_max: float) -> bool:
    for state in states:
        if not (
            float(state["min"]) == float(top_min)
            and float(state["max"]) == float(top_max)
        ):
            return False
    return True


def is_top_categorical_atom(states: List[dict]) -> bool:
    return all(int(state["value"]) == 0 for state in states)


def extract_atoms_from_patterns(
    patterns_jsonl: Path,
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[List[AtomRecord], List[str], List[str]]:
    atoms: List[AtomRecord] = []
    seen_signatures = set()
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    with patterns_jsonl.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if line_index == 0:
                numeric_features, categorical_features = discover_features_from_pattern(obj)

            pattern_id = obj["pattern_id"]
            source_patient_ids = list(obj.get("source_patient_ids", []))
            source_sample_ids = list(obj.get("source_sample_ids", []))
            states = obj["states"]

            for feature in numeric_features:
                feature_states = [normalize_numeric_state(state[feature]) for state in states]
                top_min, top_max = top_intervals[feature]
                if is_top_numeric_atom(feature_states, top_min, top_max):
                    continue
                signature = build_atom_signature(feature, "numeric", feature_states)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                atoms.append(
                    AtomRecord(
                        atom_id=f"atom_{len(atoms)}",
                        feature=feature,
                        feature_type="numeric",
                        states=feature_states,
                        source_pattern_id=pattern_id,
                        source_patient_ids=source_patient_ids,
                        source_sample_ids=source_sample_ids,
                    )
                )

            for feature in categorical_features:
                feature_states = [{"value": normalize_categorical_state(state[feature])} for state in states]
                if is_top_categorical_atom(feature_states):
                    continue
                signature = build_atom_signature(feature, "categorical", feature_states)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                atoms.append(
                    AtomRecord(
                        atom_id=f"atom_{len(atoms)}",
                        feature=feature,
                        feature_type="categorical",
                        states=feature_states,
                        source_pattern_id=pattern_id,
                        source_patient_ids=source_patient_ids,
                        source_sample_ids=source_sample_ids,
                    )
                )

    return atoms, numeric_features, categorical_features


def feature_state_to_numeric_vector(
    feature: str,
    state: dict,
    top_intervals: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    top_min, top_max = top_intervals[feature]
    if feature == "ecog":
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
        value = state.get("labs", {}).get(feature, {"missing": 1})
        raw_min = finite_float_or_none(value.get("min"))
        raw_max = finite_float_or_none(value.get("max"))
        is_missing = value.get("missing") == 1

    if is_missing or raw_min is None or raw_max is None or raw_min > raw_max:
        return np.array([top_min, top_max, 1.0], dtype=np.float32)
    return np.array([raw_min, raw_max, 0.0], dtype=np.float32)


def feature_state_to_categorical_value(feature: str, state: dict) -> int:
    if feature == "admissions":
        return parse_admissions(state.get("admissions", 0))
    return parse_bool(state.get(feature, 0))


def prepare_train_feature_data(
    train_samples: List[dict],
    numeric_features: List[str],
    categorical_features: List[str],
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    List[str],
    List[str],
]:
    numeric_data = {feature: [] for feature in numeric_features}
    categorical_data = {feature: [] for feature in categorical_features}
    labels = []
    patient_ids = []
    sample_ids = []

    for idx, sample in enumerate(train_samples):
        states = sample["states"]
        labels.append(int(sample["label"]))
        patient_ids.append(str(sample.get("source_patient_id", sample.get("patient_id", ""))))
        sample_ids.append(sample_id_for(sample, idx))

        for feature in numeric_features:
            seq = [feature_state_to_numeric_vector(feature, state, top_intervals) for state in states]
            numeric_data[feature].append(np.array(seq, dtype=np.float32))
        for feature in categorical_features:
            seq = [feature_state_to_categorical_value(feature, state) for state in states]
            categorical_data[feature].append(np.array(seq, dtype=np.int16))

    numeric_arrays = {
        feature: np.stack(rows, axis=0).astype(np.float32)
        for feature, rows in numeric_data.items()
    }
    categorical_arrays = {
        feature: np.stack(rows, axis=0).astype(np.int16)
        for feature, rows in categorical_data.items()
    }

    return numeric_arrays, categorical_arrays, np.array(labels, dtype=np.int16), patient_ids, sample_ids


def build_patient_index_maps(
    labels: np.ndarray,
    patient_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], int, int]:
    patient_to_global: Dict[str, int] = {}
    sample_patient_global_idx = np.empty(len(patient_ids), dtype=np.int32)
    label0_patient_to_idx: Dict[str, int] = {}
    label1_patient_to_idx: Dict[str, int] = {}
    sample_patient_label0_idx = np.full(len(patient_ids), -1, dtype=np.int32)
    sample_patient_label1_idx = np.full(len(patient_ids), -1, dtype=np.int32)

    for sample_idx, patient_id in enumerate(patient_ids):
        if patient_id not in patient_to_global:
            patient_to_global[patient_id] = len(patient_to_global)
        sample_patient_global_idx[sample_idx] = patient_to_global[patient_id]
        if int(labels[sample_idx]) == 0:
            if patient_id not in label0_patient_to_idx:
                label0_patient_to_idx[patient_id] = len(label0_patient_to_idx)
            sample_patient_label0_idx[sample_idx] = label0_patient_to_idx[patient_id]
        else:
            if patient_id not in label1_patient_to_idx:
                label1_patient_to_idx[patient_id] = len(label1_patient_to_idx)
            sample_patient_label1_idx[sample_idx] = label1_patient_to_idx[patient_id]

    return (
        sample_patient_global_idx,
        sample_patient_label0_idx,
        sample_patient_label1_idx,
        patient_to_global,
        len(label0_patient_to_idx),
        len(label1_patient_to_idx),
    )


@njit
def contains_int(values: np.ndarray, target: int) -> bool:
    for i in range(values.shape[0]):
        if values[i] == target:
            return True
    return False


@njit
def numeric_atom_matches_sample(atom: np.ndarray, sample: np.ndarray) -> bool:
    for pos in range(atom.shape[0]):
        atom_min = atom[pos, 0]
        atom_max = atom[pos, 1]
        sample_min = sample[pos, 0]
        sample_max = sample[pos, 1]
        if not (np.isfinite(atom_min) and np.isfinite(atom_max) and np.isfinite(sample_min) and np.isfinite(sample_max)):
            return False
        if not (atom_min <= sample_min and atom_max >= sample_max):
            return False
    return True


@njit
def categorical_atom_matches_sample(atom: np.ndarray, sample: np.ndarray) -> bool:
    for pos in range(atom.shape[0]):
        if not (atom[pos] <= sample[pos]):
            return False
    return True


@njit
def compute_numeric_atom_metrics_numba(
    atom: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sample_patient_global_idx: np.ndarray,
    sample_patient_label0_idx: np.ndarray,
    sample_patient_label1_idx: np.ndarray,
    source_patient_global_idx: np.ndarray,
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[int, int, float, float, float, float]:
    matched_patients_0 = np.zeros(total_patients_0, dtype=np.uint8)
    matched_patients_1 = np.zeros(total_patients_1, dtype=np.uint8)

    for sample_idx in range(data.shape[0]):
        patient_global_idx = sample_patient_global_idx[sample_idx]
        if contains_int(source_patient_global_idx, patient_global_idx):
            continue
        if not numeric_atom_matches_sample(atom, data[sample_idx]):
            continue
        if labels[sample_idx] == 0:
            matched_patients_0[sample_patient_label0_idx[sample_idx]] = 1
        else:
            matched_patients_1[sample_patient_label1_idx[sample_idx]] = 1

    support0 = int(np.sum(matched_patients_0))
    support1 = int(np.sum(matched_patients_1))
    total = support0 + support1
    precision0 = support0 / total if total else 0.0
    precision1 = support1 / total if total else 0.0
    recall0 = support0 / total_patients_0 if total_patients_0 else 0.0
    recall1 = support1 / total_patients_1 if total_patients_1 else 0.0
    return support0, support1, precision0, precision1, recall0, recall1


@njit
def compute_numeric_atom_support_arrays_numba(
    atom: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sample_patient_global_idx: np.ndarray,
    sample_patient_label0_idx: np.ndarray,
    sample_patient_label1_idx: np.ndarray,
    source_patient_global_idx: np.ndarray,
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    matched_patients_0 = np.zeros(total_patients_0, dtype=np.uint8)
    matched_patients_1 = np.zeros(total_patients_1, dtype=np.uint8)

    for sample_idx in range(data.shape[0]):
        patient_global_idx = sample_patient_global_idx[sample_idx]
        if contains_int(source_patient_global_idx, patient_global_idx):
            continue
        if not numeric_atom_matches_sample(atom, data[sample_idx]):
            continue
        if labels[sample_idx] == 0:
            matched_patients_0[sample_patient_label0_idx[sample_idx]] = 1
        else:
            matched_patients_1[sample_patient_label1_idx[sample_idx]] = 1

    return matched_patients_0, matched_patients_1


@njit
def compute_categorical_atom_metrics_numba(
    atom: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sample_patient_global_idx: np.ndarray,
    sample_patient_label0_idx: np.ndarray,
    sample_patient_label1_idx: np.ndarray,
    source_patient_global_idx: np.ndarray,
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[int, int, float, float, float, float]:
    matched_patients_0 = np.zeros(total_patients_0, dtype=np.uint8)
    matched_patients_1 = np.zeros(total_patients_1, dtype=np.uint8)

    for sample_idx in range(data.shape[0]):
        patient_global_idx = sample_patient_global_idx[sample_idx]
        if contains_int(source_patient_global_idx, patient_global_idx):
            continue
        if not categorical_atom_matches_sample(atom, data[sample_idx]):
            continue
        if labels[sample_idx] == 0:
            matched_patients_0[sample_patient_label0_idx[sample_idx]] = 1
        else:
            matched_patients_1[sample_patient_label1_idx[sample_idx]] = 1

    support0 = int(np.sum(matched_patients_0))
    support1 = int(np.sum(matched_patients_1))
    total = support0 + support1
    precision0 = support0 / total if total else 0.0
    precision1 = support1 / total if total else 0.0
    recall0 = support0 / total_patients_0 if total_patients_0 else 0.0
    recall1 = support1 / total_patients_1 if total_patients_1 else 0.0
    return support0, support1, precision0, precision1, recall0, recall1


@njit
def compute_categorical_atom_support_arrays_numba(
    atom: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sample_patient_global_idx: np.ndarray,
    sample_patient_label0_idx: np.ndarray,
    sample_patient_label1_idx: np.ndarray,
    source_patient_global_idx: np.ndarray,
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    matched_patients_0 = np.zeros(total_patients_0, dtype=np.uint8)
    matched_patients_1 = np.zeros(total_patients_1, dtype=np.uint8)

    for sample_idx in range(data.shape[0]):
        patient_global_idx = sample_patient_global_idx[sample_idx]
        if contains_int(source_patient_global_idx, patient_global_idx):
            continue
        if not categorical_atom_matches_sample(atom, data[sample_idx]):
            continue
        if labels[sample_idx] == 0:
            matched_patients_0[sample_patient_label0_idx[sample_idx]] = 1
        else:
            matched_patients_1[sample_patient_label1_idx[sample_idx]] = 1

    return matched_patients_0, matched_patients_1


def atom_numeric_states_to_array(states: List[dict]) -> np.ndarray:
    rows = []
    for state in states:
        rows.append(
            [
                float(state["min"]),
                float(state["max"]),
                1.0 if bool(state.get("has_missing_source", False)) else 0.0,
            ]
        )
    return np.array(rows, dtype=np.float32)


def atom_categorical_states_to_array(states: List[dict]) -> np.ndarray:
    return np.array([int(state["value"]) for state in states], dtype=np.int16)


def compute_balanced_precisions(
    support0: int,
    support1: int,
    total_patients_0: int,
    total_patients_1: int,
) -> Tuple[float, float, float]:
    hit_rate0 = support0 / total_patients_0 if total_patients_0 else 0.0
    hit_rate1 = support1 / total_patients_1 if total_patients_1 else 0.0
    denom = hit_rate0 + hit_rate1
    if denom == 0.0:
        return 0.0, 0.0, 0.0
    balanced_precision0 = hit_rate0 / denom
    balanced_precision1 = hit_rate1 / denom
    return (
        balanced_precision0,
        balanced_precision1,
        max(balanced_precision0, balanced_precision1),
    )


def patient_indicator_to_bitset(indicator: np.ndarray) -> int:
    packed = np.packbits(indicator.astype(np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def atom_dominates_for_label0(atom_a: EvaluatedAtom, atom_b: EvaluatedAtom) -> bool:
    return (
        (atom_a.support_bits0 | atom_b.support_bits0) == atom_a.support_bits0
        and (atom_a.support_bits1 | atom_b.support_bits1) == atom_b.support_bits1
    )


def atom_id_sort_key(atom_id: str) -> Tuple[int, str]:
    try:
        return int(atom_id.split("_")[-1]), atom_id
    except ValueError:
        return 10**18, atom_id


def deduplicate_atoms_by_support_containment(
    evaluated_atoms: List[EvaluatedAtom],
    total_patients_0: int,
) -> List[EvaluatedAtom]:
    unique_by_support_pair: Dict[Tuple[int, int], EvaluatedAtom] = {}
    for atom in evaluated_atoms:
        key = (atom.support_bits0, atom.support_bits1)
        existing = unique_by_support_pair.get(key)
        if existing is None:
            unique_by_support_pair[key] = atom
            continue
        if (
            atom.precision0,
            atom.recall0,
            atom.support0,
            -atom.support1,
            -atom_id_sort_key(atom.atom.atom_id)[0],
        ) > (
            existing.precision0,
            existing.recall0,
            existing.support0,
            -existing.support1,
            -atom_id_sort_key(existing.atom.atom_id)[0],
        ):
            unique_by_support_pair[key] = atom

    unique_atoms = list(unique_by_support_pair.values())
    print(
        f"Atoms kept after exact support-set dedup: {len(unique_atoms)} "
        f"(from {len(evaluated_atoms)})"
    )

    sorted_atoms = sorted(
        unique_atoms,
        key=lambda item: (
            item.support1,
            -item.support0,
            -item.precision0,
            -item.recall0,
            atom_id_sort_key(item.atom.atom_id),
        ),
    )
    frontier_by_support0: Dict[int, List[EvaluatedAtom]] = {}
    kept_atoms: List[EvaluatedAtom] = []

    for atom_idx, atom in enumerate(sorted_atoms, start=1):
        if atom_idx % 1000 == 0:
            print(
                f"Containment dedup progress: {atom_idx}/{len(sorted_atoms)}",
                flush=True,
            )
        dominated = False
        for support0_count in range(total_patients_0, atom.support0 - 1, -1):
            for candidate in frontier_by_support0.get(support0_count, []):
                if atom_dominates_for_label0(candidate, atom):
                    dominated = True
                    break
            if dominated:
                break

        if dominated:
            continue

        kept_atoms.append(atom)
        frontier_by_support0.setdefault(atom.support0, []).append(atom)

    kept_atom_ids = {item.atom.atom_id for item in kept_atoms}
    ordered_kept_atoms = [
        atom for atom in unique_atoms
        if atom.atom.atom_id in kept_atom_ids
    ]
    return ordered_kept_atoms


def build_metric_row(atom_eval: EvaluatedAtom) -> List[Any]:
    return [
        atom_eval.atom.atom_id,
        atom_eval.atom.feature,
        atom_eval.atom.feature_type,
        atom_eval.atom.source_pattern_id,
        atom_eval.support0,
        atom_eval.support1,
        atom_eval.precision0,
        atom_eval.precision1,
        atom_eval.recall0,
        atom_eval.recall1,
        atom_eval.balanced_precision0,
        atom_eval.balanced_precision1,
        atom_eval.best_balanced_precision,
    ]


def write_atoms_jsonl(path: Path, atoms: List[AtomRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for atom in atoms:
            payload = {
                "atom_id": atom.atom_id,
                "feature": atom.feature,
                "feature_type": atom.feature_type,
                "states": atom.states,
                "source_pattern_id": atom.source_pattern_id,
                "source_patient_ids": atom.source_patient_ids,
                "source_sample_ids": atom.source_sample_ids,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    start_time = time.time()
    print("Loading top intervals...")
    top_intervals = load_top_intervals_from_csv(TOP_INTERVALS_CSV)

    print("Extracting single-feature atoms from existing patterns...")
    atoms, numeric_features, categorical_features = extract_atoms_from_patterns(PATTERNS_JSONL, top_intervals)
    print(f"Atoms kept after trivial-top removal and exact dedup: {len(atoms)}")

    print("Loading eval-train subsequences...")
    train_samples = load_samples(TRAIN_IPS)
    print(f"Eval-train samples: {len(train_samples)}")

    print("Preparing per-feature eval-train data...")
    numeric_data, categorical_data, labels, patient_ids, _ = prepare_train_feature_data(
        train_samples,
        numeric_features,
        categorical_features,
        top_intervals,
    )
    (
        sample_patient_global_idx,
        sample_patient_label0_idx,
        sample_patient_label1_idx,
        patient_to_global,
        total_patients_0,
        total_patients_1,
    ) = build_patient_index_maps(labels, patient_ids)
    print(f"Unique eval-train patients: label0={total_patients_0}, label1={total_patients_1}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atoms_jsonl_path = OUTPUT_DIR / "atoms.jsonl"
    atom_metrics_csv_path = OUTPUT_DIR / "atom_metrics.csv"
    evaluated_atoms: List[EvaluatedAtom] = []

    print("Computing eval-train metrics and concrete support patient sets for each atom...")
    for atom_idx, atom in enumerate(atoms, start=1):
        if atom_idx % 1000 == 0:
            print(f"Processed {atom_idx}/{len(atoms)} atoms", flush=True)

        source_patient_global_idx = np.array(
            [patient_to_global[pid] for pid in atom.source_patient_ids if pid in patient_to_global],
            dtype=np.int32,
        )

        if atom.feature_type == "numeric":
            atom_array = atom_numeric_states_to_array(atom.states)
            matched_patients_0, matched_patients_1 = compute_numeric_atom_support_arrays_numba(
                atom_array,
                numeric_data[atom.feature],
                labels,
                sample_patient_global_idx,
                sample_patient_label0_idx,
                sample_patient_label1_idx,
                source_patient_global_idx,
                total_patients_0,
                total_patients_1,
            )
        else:
            atom_array = atom_categorical_states_to_array(atom.states)
            matched_patients_0, matched_patients_1 = compute_categorical_atom_support_arrays_numba(
                atom_array,
                categorical_data[atom.feature],
                labels,
                sample_patient_global_idx,
                sample_patient_label0_idx,
                sample_patient_label1_idx,
                source_patient_global_idx,
                total_patients_0,
                total_patients_1,
            )

        support0 = int(np.sum(matched_patients_0))
        support1 = int(np.sum(matched_patients_1))
        total_support = support0 + support1
        precision0 = support0 / total_support if total_support else 0.0
        precision1 = support1 / total_support if total_support else 0.0
        recall0 = support0 / total_patients_0 if total_patients_0 else 0.0
        recall1 = support1 / total_patients_1 if total_patients_1 else 0.0
        balanced_precision0, balanced_precision1, best_balanced_precision = compute_balanced_precisions(
            support0,
            support1,
            total_patients_0,
            total_patients_1,
        )

        evaluated_atoms.append(
            EvaluatedAtom(
                atom=atom,
                support0=support0,
                support1=support1,
                precision0=precision0,
                precision1=precision1,
                recall0=recall0,
                recall1=recall1,
                balanced_precision0=balanced_precision0,
                balanced_precision1=balanced_precision1,
                best_balanced_precision=best_balanced_precision,
                support_bits0=patient_indicator_to_bitset(matched_patients_0),
                support_bits1=patient_indicator_to_bitset(matched_patients_1),
            )
        )

    print("Removing atoms dominated by stronger label0-support containment...")
    deduplicated_atoms = deduplicate_atoms_by_support_containment(
        evaluated_atoms,
        total_patients_0,
    )
    print(f"Atoms kept after support-set containment dedup: {len(deduplicated_atoms)}")

    filtered_atoms = [
        atom_eval for atom_eval in deduplicated_atoms
        if atom_eval.best_balanced_precision >= BEST_BALANCED_PRECISION_THRESHOLD
    ]
    print(
        f"Atoms kept after best_balanced_precision >= {BEST_BALANCED_PRECISION_THRESHOLD}: "
        f"{len(filtered_atoms)}"
    )

    kept_atoms = [atom_eval.atom for atom_eval in filtered_atoms]
    kept_metric_rows = [build_metric_row(atom_eval) for atom_eval in filtered_atoms]

    print("Writing filtered atom JSONL...")
    write_atoms_jsonl(atoms_jsonl_path, kept_atoms)

    print("Writing filtered atom metrics CSV...")
    with atom_metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "atom_id",
                "feature",
                "feature_type",
                "source_pattern_id",
                "train_dedup_support0_excluding_sources",
                "train_dedup_support1_excluding_sources",
                "train_dedup_precision0_excluding_sources",
                "train_dedup_precision1_excluding_sources",
                "train_dedup_recall0_excluding_sources",
                "train_dedup_recall1_excluding_sources",
                "balanced_precision0",
                "balanced_precision1",
                "best_balanced_precision",
            ]
        )
        writer.writerows(kept_metric_rows)

    elapsed = time.time() - start_time
    print(f"Atoms JSONL saved: {atoms_jsonl_path}")
    print(f"Atom metrics CSV saved: {atom_metrics_csv_path}")
    print(f"Elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
