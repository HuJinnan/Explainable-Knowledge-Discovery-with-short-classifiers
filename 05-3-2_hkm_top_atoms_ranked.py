from __future__ import annotations

import csv
import heapq
import json
import math
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass
from itertools import combinations, islice, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


ATOMS_JSONL = Path(r"D:\PythonProject\paper\single_feature_atoms_initial\atoms.jsonl")
ATOM_METRICS_CSV = Path(r"D:\PythonProject\paper\single_feature_atoms_initial\atom_metrics.csv")
TRAIN_IPS = Path(r"D:\PythonProject\paper\window4_cut_0_days_all_patients\split_data\eval_train_subseq_ips.jsonl")
TEST_IPS = Path(r"D:\PythonProject\paper\window4_cut_0_days_all_patients\split_data\eval_test_subseq_ips.jsonl")
TOP_INTERVALS_CSV = Path(r"D:\PythonProject\sequence_pattern_structures\find_top_intervals\top_intervals.csv")
HKM_TEMPLATE_DIR = Path(r"D:\PythonProject\paper\hkm_dedup_output")
DEFAULT_OUTPUT_ROOT = Path(r"D:\PythonProject\paper\hkm_top_atoms_ranked_outputs")

LENGTHS = (2, 3, 4)
DEFAULT_TOP_ATOMS_PER_FEATURE = 40
DEFAULT_TOP_HKMS_PER_LENGTH = 5000
DEFAULT_CURVE_TOP_N: Optional[int] = None
DEFAULT_NUM_WORKERS = 28
DEFAULT_CHECKPOINT_EVERY_HKMS = 6000
LOGIC_PRECISION0 = "precision0"
LOGIC_PRECISION0_PLUS_RECALL0 = "precision0_plus_recall0"
LOGIC_NAMES = (LOGIC_PRECISION0, LOGIC_PRECISION0_PLUS_RECALL0)


@dataclass(frozen=True)
class Atom:
    atom_id: str
    feature: str
    feature_type: str
    states: List[dict]
    source_pattern_id: str
    source_patient_ids: List[str]
    source_sample_ids: List[str]
    train_support0: int
    train_support1: int
    train_precision0: float
    train_precision1: float
    train_recall0: float
    train_recall1: float
    balanced_precision0: float
    balanced_precision1: float
    best_balanced_precision: float


@dataclass(frozen=True)
class Template:
    formula_id: str
    length: int
    formula: str
    compiled_expr: object


@dataclass(frozen=True)
class AtomMatchInfo:
    atom: Atom
    train_match_bits0: int
    train_match_bits1: int
    test_match_bits0: int
    test_match_bits1: int
    train_source_bits0: int
    train_source_bits1: int


@dataclass
class HKMRecord:
    hkm_id: str
    formula_id: str
    length: int
    formula: str
    atom_ids: List[str]
    features: List[str]
    source_pattern_ids: List[str]
    train_support0: int
    train_support1: int
    train_precision0: float
    train_precision1: float
    train_recall0: float
    train_recall1: float
    test_support0: int
    test_support1: int
    test_precision0: float
    test_precision1: float
    test_recall0: float
    test_recall1: float
    train_matched_bits0: int
    train_matched_bits1: int
    test_matched_bits0: int
    test_matched_bits1: int


@dataclass(frozen=True)
class WorkItem:
    length: int
    feature_subset: Tuple[str, ...]


WORKER_ATOM_INFOS_BY_FEATURE: Dict[str, List[AtomMatchInfo]] = {}
WORKER_TEMPLATES_BY_LENGTH: Dict[int, List[Template]] = {}
WORKER_TRAIN_TOTALS: Tuple[int, int] = (0, 0)
WORKER_TEST_TOTALS: Tuple[int, int] = (0, 0)
WORKER_TOP_HKMS_PER_LENGTH: int = 0
WORKER_OUTPUT_DIR: Optional[Path] = None
WORKER_CHECKPOINT_EVERY_HKMS: int = DEFAULT_CHECKPOINT_EVERY_HKMS


def finite_float_or_none(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def parse_admissions(val: object) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if val == "2+":
            return 2
        if val.isdigit():
            return int(val)
    return 0


def parse_bool(val: object) -> int:
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return 1 if val.lower() in ("1", "true", "yes") else 0
    return 0


def load_samples(ips_path: Path) -> List[dict]:
    samples: List[dict] = []
    with ips_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_top_intervals_from_csv(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    top_intervals: Dict[str, Tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            top_intervals[row["feature"].strip()] = (float(row["min"]), float(row["max"]))
    return top_intervals


def count_label_patients(samples: Sequence[dict]) -> Tuple[int, int]:
    patients0 = set()
    patients1 = set()
    for sample in samples:
        patient_id = str(sample.get("source_patient_id", sample.get("patient_id", "")))
        label = int(sample["label"])
        if label == 0:
            patients0.add(patient_id)
        else:
            patients1.add(patient_id)
    return len(patients0), len(patients1)


def load_atoms_by_id(atoms_jsonl: Path) -> Dict[str, dict]:
    atoms: Dict[str, dict] = {}
    with atoms_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            atoms[obj["atom_id"]] = obj
    return atoms


def load_selected_atoms(
    atoms_jsonl: Path,
    atom_metrics_csv: Path,
    top_atoms_per_feature: int,
) -> List[Atom]:
    atoms_by_id = load_atoms_by_id(atoms_jsonl)
    grouped_rows: Dict[str, List[dict]] = {}

    with atom_metrics_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            atom_id = row["atom_id"]
            atom_obj = atoms_by_id.get(atom_id)
            if atom_obj is None:
                continue
            feature = atom_obj["feature"]
            enriched = dict(row)
            enriched["_feature"] = feature
            grouped_rows.setdefault(feature, []).append(enriched)

    selected_atoms: List[Atom] = []
    for feature, rows in grouped_rows.items():
        rows.sort(
            key=lambda row: (
                -float(row["best_balanced_precision"]),
                -float(row["train_dedup_precision0_excluding_sources"]),
                -float(row["train_dedup_support0_excluding_sources"]),
                row["atom_id"],
            )
        )
        for row in rows[:top_atoms_per_feature]:
            atom_obj = atoms_by_id[row["atom_id"]]
            selected_atoms.append(
                Atom(
                    atom_id=row["atom_id"],
                    feature=atom_obj["feature"],
                    feature_type=atom_obj["feature_type"],
                    states=atom_obj["states"],
                    source_pattern_id=atom_obj["source_pattern_id"],
                    source_patient_ids=list(atom_obj.get("source_patient_ids", [])),
                    source_sample_ids=list(atom_obj.get("source_sample_ids", [])),
                    train_support0=int(float(row["train_dedup_support0_excluding_sources"])),
                    train_support1=int(float(row["train_dedup_support1_excluding_sources"])),
                    train_precision0=float(row["train_dedup_precision0_excluding_sources"]),
                    train_precision1=float(row["train_dedup_precision1_excluding_sources"]),
                    train_recall0=float(row["train_dedup_recall0_excluding_sources"]),
                    train_recall1=float(row["train_dedup_recall1_excluding_sources"]),
                    balanced_precision0=float(row["balanced_precision0"]),
                    balanced_precision1=float(row["balanced_precision1"]),
                    best_balanced_precision=float(row["best_balanced_precision"]),
                )
            )
    selected_atoms.sort(key=lambda atom: (atom.feature, atom.atom_id))
    return selected_atoms


def discover_needed_features(atoms: Sequence[Atom]) -> Tuple[List[str], List[str]]:
    numeric = sorted({atom.feature for atom in atoms if atom.feature_type == "numeric"})
    categorical = sorted({atom.feature for atom in atoms if atom.feature_type == "categorical"})
    return numeric, categorical


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


def prepare_feature_data(
    samples: Sequence[dict],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    numeric_data: Dict[str, List[np.ndarray]] = {feature: [] for feature in numeric_features}
    categorical_data: Dict[str, List[np.ndarray]] = {feature: [] for feature in categorical_features}

    for sample in samples:
        states = sample["states"]
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
    return numeric_arrays, categorical_arrays


def build_patient_maps(
    samples: Sequence[dict],
) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    patient_label0_idx: Dict[str, int] = {}
    patient_label1_idx: Dict[str, int] = {}
    sample_patient_label0_idx: List[int] = []
    sample_patient_label1_idx: List[int] = []
    sample_labels: List[int] = []

    for sample in samples:
        patient_id = str(sample.get("source_patient_id", sample.get("patient_id", "")))
        label = int(sample["label"])
        sample_labels.append(label)
        if label == 0:
            if patient_id not in patient_label0_idx:
                patient_label0_idx[patient_id] = len(patient_label0_idx)
            sample_patient_label0_idx.append(patient_label0_idx[patient_id])
            sample_patient_label1_idx.append(-1)
        else:
            if patient_id not in patient_label1_idx:
                patient_label1_idx[patient_id] = len(patient_label1_idx)
            sample_patient_label0_idx.append(-1)
            sample_patient_label1_idx.append(patient_label1_idx[patient_id])

    return (
        patient_label0_idx,
        patient_label1_idx,
        np.array(sample_patient_label0_idx, dtype=np.int32),
        np.array(sample_patient_label1_idx, dtype=np.int32),
        np.array(sample_labels, dtype=np.int16),
    )


def atom_numeric_states_to_array(states: Sequence[dict]) -> np.ndarray:
    return np.array(
        [
            [
                float(state["min"]),
                float(state["max"]),
                1.0 if bool(state.get("has_missing_source", False)) else 0.0,
            ]
            for state in states
        ],
        dtype=np.float32,
    )


def atom_categorical_states_to_array(states: Sequence[dict]) -> np.ndarray:
    return np.array([int(state["value"]) for state in states], dtype=np.int16)


@njit(parallel=True)
def compute_numeric_match_matrix(atoms: np.ndarray, data: np.ndarray) -> np.ndarray:
    out = np.zeros((atoms.shape[0], data.shape[0]), dtype=np.uint8)
    for atom_idx in prange(atoms.shape[0]):
        for sample_idx in range(data.shape[0]):
            ok = True
            for pos in range(atoms.shape[1]):
                atom_min = atoms[atom_idx, pos, 0]
                atom_max = atoms[atom_idx, pos, 1]
                sample_min = data[sample_idx, pos, 0]
                sample_max = data[sample_idx, pos, 1]
                if not (
                    np.isfinite(atom_min)
                    and np.isfinite(atom_max)
                    and np.isfinite(sample_min)
                    and np.isfinite(sample_max)
                ):
                    ok = False
                    break
                if not (atom_min <= sample_min and atom_max >= sample_max):
                    ok = False
                    break
            if ok:
                out[atom_idx, sample_idx] = 1
    return out


@njit(parallel=True)
def compute_categorical_match_matrix(atoms: np.ndarray, data: np.ndarray) -> np.ndarray:
    out = np.zeros((atoms.shape[0], data.shape[0]), dtype=np.uint8)
    for atom_idx in prange(atoms.shape[0]):
        for sample_idx in range(data.shape[0]):
            ok = True
            for pos in range(atoms.shape[1]):
                if not (atoms[atom_idx, pos] <= data[sample_idx, pos]):
                    ok = False
                    break
            if ok:
                out[atom_idx, sample_idx] = 1
    return out


def make_bitset(indices: np.ndarray) -> int:
    bits = 0
    for idx in indices:
        bits |= 1 << int(idx)
    return bits


def build_match_bits_for_split(
    atoms: Sequence[Atom],
    samples: Sequence[dict],
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int], Dict[str, int]]:
    patient_label0_idx, patient_label1_idx, sample_patient_label0_idx, sample_patient_label1_idx, sample_labels = (
        build_patient_maps(samples)
    )
    numeric_features, categorical_features = discover_needed_features(atoms)
    numeric_data, categorical_data = prepare_feature_data(samples, numeric_features, categorical_features, top_intervals)

    atoms_by_feature: Dict[str, List[Atom]] = {}
    for atom in atoms:
        atoms_by_feature.setdefault(atom.feature, []).append(atom)

    bits_by_atom_id: Dict[str, Tuple[int, int]] = {}

    for feature in numeric_features:
        feature_atoms = atoms_by_feature.get(feature, [])
        if not feature_atoms:
            continue
        atom_arrays = np.stack([atom_numeric_states_to_array(atom.states) for atom in feature_atoms], axis=0)
        match_matrix = compute_numeric_match_matrix(atom_arrays, numeric_data[feature])
        for atom_idx, atom in enumerate(feature_atoms):
            matched_sample_indices = np.flatnonzero(match_matrix[atom_idx])
            label0_patient_indices = np.unique(
                sample_patient_label0_idx[matched_sample_indices[sample_labels[matched_sample_indices] == 0]]
            )
            label1_patient_indices = np.unique(
                sample_patient_label1_idx[matched_sample_indices[sample_labels[matched_sample_indices] == 1]]
            )
            label0_patient_indices = label0_patient_indices[label0_patient_indices >= 0]
            label1_patient_indices = label1_patient_indices[label1_patient_indices >= 0]
            bits_by_atom_id[atom.atom_id] = (
                make_bitset(label0_patient_indices),
                make_bitset(label1_patient_indices),
            )

    for feature in categorical_features:
        feature_atoms = atoms_by_feature.get(feature, [])
        if not feature_atoms:
            continue
        atom_arrays = np.stack([atom_categorical_states_to_array(atom.states) for atom in feature_atoms], axis=0)
        match_matrix = compute_categorical_match_matrix(atom_arrays, categorical_data[feature])
        for atom_idx, atom in enumerate(feature_atoms):
            matched_sample_indices = np.flatnonzero(match_matrix[atom_idx])
            label0_patient_indices = np.unique(
                sample_patient_label0_idx[matched_sample_indices[sample_labels[matched_sample_indices] == 0]]
            )
            label1_patient_indices = np.unique(
                sample_patient_label1_idx[matched_sample_indices[sample_labels[matched_sample_indices] == 1]]
            )
            label0_patient_indices = label0_patient_indices[label0_patient_indices >= 0]
            label1_patient_indices = label1_patient_indices[label1_patient_indices >= 0]
            bits_by_atom_id[atom.atom_id] = (
                make_bitset(label0_patient_indices),
                make_bitset(label1_patient_indices),
            )

    return bits_by_atom_id, patient_label0_idx, patient_label1_idx


def build_atom_match_infos(
    atoms: Sequence[Atom],
    train_samples: Sequence[dict],
    test_samples: Sequence[dict],
    top_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[List[AtomMatchInfo], Tuple[int, int], Tuple[int, int]]:
    train_bits_by_atom_id, train_label0_idx, train_label1_idx = build_match_bits_for_split(
        atoms, train_samples, top_intervals
    )
    test_bits_by_atom_id, test_label0_idx, test_label1_idx = build_match_bits_for_split(
        atoms, test_samples, top_intervals
    )

    atom_infos: List[AtomMatchInfo] = []
    for atom in atoms:
        train_source_bits0 = 0
        train_source_bits1 = 0
        for patient_id in atom.source_patient_ids:
            if patient_id in train_label0_idx:
                train_source_bits0 |= 1 << train_label0_idx[patient_id]
            if patient_id in train_label1_idx:
                train_source_bits1 |= 1 << train_label1_idx[patient_id]

        train_bits0, train_bits1 = train_bits_by_atom_id[atom.atom_id]
        test_bits0, test_bits1 = test_bits_by_atom_id[atom.atom_id]
        atom_infos.append(
            AtomMatchInfo(
                atom=atom,
                train_match_bits0=train_bits0,
                train_match_bits1=train_bits1,
                test_match_bits0=test_bits0,
                test_match_bits1=test_bits1,
                train_source_bits0=train_source_bits0,
                train_source_bits1=train_source_bits1,
            )
        )

    return atom_infos, (len(train_label0_idx), len(train_label1_idx)), (len(test_label0_idx), len(test_label1_idx))


def load_templates(template_dir: Path, lengths: Iterable[int]) -> Dict[int, List[Template]]:
    templates_by_length: Dict[int, List[Template]] = {}
    for length in lengths:
        path = template_dir / f"hkm_len{length}_dedup.jsonl"
        templates: List[Template] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                expr = obj["formula"]
                expr = expr.replace("NOT x1", "(mask ^ x1)")
                expr = expr.replace("NOT x2", "(mask ^ x2)")
                expr = expr.replace("NOT x3", "(mask ^ x3)")
                expr = expr.replace("NOT x4", "(mask ^ x4)")
                expr = expr.replace(" AND ", " & ")
                expr = expr.replace(" OR ", " | ")
                templates.append(
                    Template(
                        formula_id=obj["formula_id"],
                        length=length,
                        formula=obj["formula"],
                        compiled_expr=compile(expr, f"<{obj['formula_id']}>", "eval"),
                    )
                )
        templates_by_length[length] = templates
    return templates_by_length


def evaluate_template(compiled_expr: object, mask: int, values: Sequence[int]) -> int:
    env = {
        "mask": mask,
        "x1": values[0] if len(values) > 0 else 0,
        "x2": values[1] if len(values) > 1 else 0,
        "x3": values[2] if len(values) > 2 else 0,
        "x4": values[3] if len(values) > 3 else 0,
    }
    return eval(compiled_expr, {}, env)


def compute_precision_recall(
    support0: int,
    support1: int,
    total0: int,
    total1: int,
) -> Tuple[float, float, float, float]:
    total = support0 + support1
    precision0 = support0 / total if total else 0.0
    precision1 = support1 / total if total else 0.0
    recall0 = support0 / total0 if total0 else 0.0
    recall1 = support1 / total1 if total1 else 0.0
    return precision0, precision1, recall0, recall1


def support_signature(record: HKMRecord) -> Tuple[int, int]:
    return record.train_matched_bits0, record.train_matched_bits1


def precision0_recall0_mean(record: HKMRecord) -> float:
    return (record.train_precision0 + record.train_recall0) / 2.0


def logic_rank_key(record: HKMRecord, logic_name: str) -> Tuple[float, float, float, int, int]:
    if logic_name == LOGIC_PRECISION0:
        return (
            record.train_precision0,
            record.train_recall0,
            record.train_precision0 + record.train_recall0,
            record.train_support0,
            -record.train_support1,
        )
    if logic_name == LOGIC_PRECISION0_PLUS_RECALL0:
        return (
            precision0_recall0_mean(record),
            record.train_precision0,
            record.train_recall0,
            record.train_support0,
            -record.train_support1,
        )
    raise ValueError(f"Unknown logic_name: {logic_name}")


def canonical_record_key(record: HKMRecord) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
    return (
        record.formula_id,
        record.hkm_id,
        tuple(record.atom_ids),
        tuple(record.features),
    )


def record_to_dict(record: HKMRecord) -> dict:
    return {
        "hkm_id": record.hkm_id,
        "formula_id": record.formula_id,
        "length": record.length,
        "formula": record.formula,
        "atom_ids": record.atom_ids,
        "features": record.features,
        "source_pattern_ids": record.source_pattern_ids,
        "train_support0": record.train_support0,
        "train_support1": record.train_support1,
        "train_precision0": record.train_precision0,
        "train_precision1": record.train_precision1,
        "train_recall0": record.train_recall0,
        "train_recall1": record.train_recall1,
        "test_support0": record.test_support0,
        "test_support1": record.test_support1,
        "test_precision0": record.test_precision0,
        "test_precision1": record.test_precision1,
        "test_recall0": record.test_recall0,
        "test_recall1": record.test_recall1,
        "train_matched_bits0_hex": format(record.train_matched_bits0, "x"),
        "train_matched_bits1_hex": format(record.train_matched_bits1, "x"),
        "test_matched_bits0_hex": format(record.test_matched_bits0, "x"),
        "test_matched_bits1_hex": format(record.test_matched_bits1, "x"),
    }


def record_from_dict(payload: dict) -> HKMRecord:
    return HKMRecord(
        hkm_id=payload["hkm_id"],
        formula_id=payload["formula_id"],
        length=int(payload["length"]),
        formula=payload["formula"],
        atom_ids=list(payload["atom_ids"]),
        features=list(payload["features"]),
        source_pattern_ids=list(payload["source_pattern_ids"]),
        train_support0=int(payload["train_support0"]),
        train_support1=int(payload["train_support1"]),
        train_precision0=float(payload["train_precision0"]),
        train_precision1=float(payload["train_precision1"]),
        train_recall0=float(payload["train_recall0"]),
        train_recall1=float(payload["train_recall1"]),
        test_support0=int(payload["test_support0"]),
        test_support1=int(payload["test_support1"]),
        test_precision0=float(payload["test_precision0"]),
        test_precision1=float(payload["test_precision1"]),
        test_recall0=float(payload["test_recall0"]),
        test_recall1=float(payload["test_recall1"]),
        train_matched_bits0=int(payload["train_matched_bits0_hex"], 16),
        train_matched_bits1=int(payload["train_matched_bits1_hex"], 16),
        test_matched_bits0=int(payload["test_matched_bits0_hex"], 16),
        test_matched_bits1=int(payload["test_matched_bits1_hex"], 16),
    )


def choose_better_equal_support_record(
    current: HKMRecord,
    candidate: HKMRecord,
    logic_name: str,
) -> HKMRecord:
    current_rank = logic_rank_key(current, logic_name)
    candidate_rank = logic_rank_key(candidate, logic_name)
    if candidate_rank > current_rank:
        return candidate
    if candidate_rank < current_rank:
        return current
    if canonical_record_key(candidate) < canonical_record_key(current):
        return candidate
    return current


def prune_logic_heap(
    heap: List[Tuple[Tuple[float, float, float, int, int], Tuple[int, int], str]],
    records_by_support: Dict[Tuple[int, int], HKMRecord],
) -> None:
    while heap:
        _, support_key, record_id = heap[0]
        current = records_by_support.get(support_key)
        if current is not None and current.hkm_id == record_id:
            return
        heapq.heappop(heap)


def update_logic_pool(
    records_by_support: Dict[Tuple[int, int], HKMRecord],
    heap: List[Tuple[Tuple[float, float, float, int, int], Tuple[int, int], str]],
    record: HKMRecord,
    logic_name: str,
    max_size: int,
) -> None:
    support_key = support_signature(record)
    current = records_by_support.get(support_key)
    if current is not None:
        chosen = choose_better_equal_support_record(current, record, logic_name)
        if chosen is not current:
            records_by_support[support_key] = chosen
            heapq.heappush(heap, (logic_rank_key(chosen, logic_name), support_key, chosen.hkm_id))
        return

    if len(records_by_support) < max_size:
        records_by_support[support_key] = record
        heapq.heappush(heap, (logic_rank_key(record, logic_name), support_key, record.hkm_id))
        return

    prune_logic_heap(heap, records_by_support)
    if not heap:
        records_by_support[support_key] = record
        heapq.heappush(heap, (logic_rank_key(record, logic_name), support_key, record.hkm_id))
        return

    worst_rank, worst_support_key, _ = heap[0]
    candidate_rank = logic_rank_key(record, logic_name)
    if candidate_rank > worst_rank:
        del records_by_support[worst_support_key]
        records_by_support[support_key] = record
        heapq.heapreplace(heap, (candidate_rank, support_key, record.hkm_id))


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    last_error: Optional[BaseException] = None
    for _ in range(20):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Atomic write failed unexpectedly for {path}")


def atomic_write_json(path: Path, payload: object, indent: Optional[int] = None) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def build_corrupt_path(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}.corrupt_{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.corrupt_{timestamp}_{counter}")
        counter += 1
    return candidate


def quarantine_corrupted_checkpoint(path: Path, reason: str) -> None:
    corrupt_path = build_corrupt_path(path)
    try:
        path.replace(corrupt_path)
        print(
            f"Checkpoint read failed for {path.name} ({reason}). "
            f"Moved corrupted file to {corrupt_path.name}.",
            flush=True,
        )
    except OSError as exc:
        print(
            f"Checkpoint read failed for {path.name} ({reason}), and moving it aside also failed: {exc}",
            flush=True,
        )


def try_load_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        quarantine_corrupted_checkpoint(path, str(exc))
        return None


def estimate_total_hkm_count(
    feature_count: int,
    top_atoms_per_feature: int,
    templates_by_length: Dict[int, List[Template]],
    lengths: Iterable[int],
) -> int:
    total = 0
    for length in lengths:
        total += math.comb(feature_count, length) * (top_atoms_per_feature ** length) * len(templates_by_length[length])
    return total


def chunk_key_for_feature_subset(length: int, feature_subset: Sequence[str]) -> str:
    return f"len{length}__{'__'.join(feature_subset)}"


def checkpoint_dir_for_output(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def checkpoint_path_for_work_item(output_dir: Path, work_item: WorkItem) -> Path:
    checkpoint_dir = checkpoint_dir_for_output(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{chunk_key_for_feature_subset(work_item.length, work_item.feature_subset)}.json"


def partial_checkpoint_path_for_work_item(output_dir: Path, work_item: WorkItem) -> Path:
    checkpoint_dir = checkpoint_dir_for_output(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{chunk_key_for_feature_subset(work_item.length, work_item.feature_subset)}__partial.json"


def save_progress_summary(
    output_dir: Path,
    total_tasks: int,
    completed_tasks: int,
    skipped_tasks: int,
) -> None:
    payload = {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "skipped_tasks": skipped_tasks,
        "pending_tasks": total_tasks - completed_tasks - skipped_tasks,
    }
    atomic_write_json(output_dir / "checkpoint_progress.json", payload, indent=2)


def build_work_items(
    atom_infos: Sequence[AtomMatchInfo],
    lengths: Iterable[int],
) -> List[WorkItem]:
    feature_names = sorted({info.atom.feature for info in atom_infos})
    work_items: List[WorkItem] = []
    for length in lengths:
        for feature_subset in combinations(feature_names, length):
            work_items.append(WorkItem(length=length, feature_subset=tuple(feature_subset)))
    return work_items


def save_partial_checkpoint_result(
    path: Path,
    work_item: WorkItem,
    processed_count: int,
    next_combo_idx: int,
    next_template_idx: int,
    top_records_by_logic: Dict[str, Sequence[HKMRecord]],
) -> None:
    payload = {
        "length": work_item.length,
        "feature_subset": list(work_item.feature_subset),
        "processed_count": processed_count,
        "next_combo_idx": next_combo_idx,
        "next_template_idx": next_template_idx,
        "top_records_by_logic": {
            logic_name: [record_to_dict(record) for record in records]
            for logic_name, records in top_records_by_logic.items()
        },
    }
    atomic_write_json(path, payload)


def estimate_total_hkm_count_from_work_items(
    atom_infos: Sequence[AtomMatchInfo],
    templates_by_length: Dict[int, List[Template]],
    lengths: Iterable[int],
) -> int:
    counts_by_feature: Dict[str, int] = {}
    for info in atom_infos:
        counts_by_feature[info.atom.feature] = counts_by_feature.get(info.atom.feature, 0) + 1
    total = 0
    for length in lengths:
        for feature_subset in combinations(sorted(counts_by_feature), length):
            prod_count = 1
            for feature in feature_subset:
                prod_count *= counts_by_feature[feature]
            total += prod_count * len(templates_by_length[length])
    return total


def init_worker(
    atom_infos_by_feature: Dict[str, List[AtomMatchInfo]],
    train_totals: Tuple[int, int],
    test_totals: Tuple[int, int],
    top_hkms_per_length: int,
    template_dir: Path,
    lengths: Sequence[int],
    output_dir: Path,
    checkpoint_every_hkms: int,
) -> None:
    global WORKER_ATOM_INFOS_BY_FEATURE
    global WORKER_TEMPLATES_BY_LENGTH
    global WORKER_TRAIN_TOTALS
    global WORKER_TEST_TOTALS
    global WORKER_TOP_HKMS_PER_LENGTH
    global WORKER_OUTPUT_DIR
    global WORKER_CHECKPOINT_EVERY_HKMS
    WORKER_ATOM_INFOS_BY_FEATURE = atom_infos_by_feature
    WORKER_TEMPLATES_BY_LENGTH = load_templates(template_dir, lengths)
    WORKER_TRAIN_TOTALS = train_totals
    WORKER_TEST_TOTALS = test_totals
    WORKER_TOP_HKMS_PER_LENGTH = top_hkms_per_length
    WORKER_OUTPUT_DIR = output_dir
    WORKER_CHECKPOINT_EVERY_HKMS = checkpoint_every_hkms


def process_work_item(work_item: WorkItem) -> dict:
    length = work_item.length
    feature_subset = work_item.feature_subset
    templates = WORKER_TEMPLATES_BY_LENGTH[length]
    atom_lists = [WORKER_ATOM_INFOS_BY_FEATURE[feature] for feature in feature_subset]
    train_total0, train_total1 = WORKER_TRAIN_TOTALS
    test_total0, test_total1 = WORKER_TEST_TOTALS
    train_mask0 = (1 << train_total0) - 1
    train_mask1 = (1 << train_total1) - 1
    test_mask0 = (1 << test_total0) - 1
    test_mask1 = (1 << test_total1) - 1

    local_records_by_logic: Dict[str, Dict[Tuple[int, int], HKMRecord]] = {
        logic_name: {} for logic_name in LOGIC_NAMES
    }
    local_heaps_by_logic: Dict[str, List[Tuple[Tuple[float, float, float, int, int], Tuple[int, int], str]]] = {
        logic_name: [] for logic_name in LOGIC_NAMES
    }
    processed_count = 0
    subset_key = chunk_key_for_feature_subset(length, feature_subset)
    if WORKER_OUTPUT_DIR is None:
        raise ValueError("WORKER_OUTPUT_DIR is not initialized.")
    final_checkpoint_path = checkpoint_path_for_work_item(WORKER_OUTPUT_DIR, work_item)
    partial_checkpoint_path = partial_checkpoint_path_for_work_item(WORKER_OUTPUT_DIR, work_item)
    resume_combo_idx = 0
    resume_template_idx = 0

    if partial_checkpoint_path.exists():
        partial_payload = load_checkpoint_result(partial_checkpoint_path)
        if partial_payload is not None:
            processed_count = int(partial_payload.get("processed_count", 0))
            resume_combo_idx = int(partial_payload.get("next_combo_idx", 0))
            resume_template_idx = int(partial_payload.get("next_template_idx", 0))
            payload_by_logic = partial_payload.get("top_records_by_logic", {})
            for logic_name in LOGIC_NAMES:
                for record_payload in payload_by_logic.get(logic_name, []):
                    record = record_from_dict(record_payload)
                    update_logic_pool(
                        local_records_by_logic[logic_name],
                        local_heaps_by_logic[logic_name],
                        record,
                        logic_name,
                        WORKER_TOP_HKMS_PER_LENGTH,
                    )

    product_iter = islice(product(*atom_lists), resume_combo_idx, None)
    for combo_idx, atom_tuple in enumerate(product_iter, start=resume_combo_idx):
        train_values0 = [info.train_match_bits0 for info in atom_tuple]
        train_values1 = [info.train_match_bits1 for info in atom_tuple]
        test_values0 = [info.test_match_bits0 for info in atom_tuple]
        test_values1 = [info.test_match_bits1 for info in atom_tuple]

        source_union0 = 0
        source_union1 = 0
        atom_ids = [info.atom.atom_id for info in atom_tuple]
        features = [info.atom.feature for info in atom_tuple]
        source_pattern_ids = [info.atom.source_pattern_id for info in atom_tuple]

        for info in atom_tuple:
            source_union0 |= info.train_source_bits0
            source_union1 |= info.train_source_bits1

        current_template_start = resume_template_idx if combo_idx == resume_combo_idx else 0
        for template_idx in range(current_template_start, len(templates)):
            template = templates[template_idx]
            train_matched_bits0 = evaluate_template(template.compiled_expr, train_mask0, train_values0)
            train_matched_bits1 = evaluate_template(template.compiled_expr, train_mask1, train_values1)
            train_matched_bits0 &= train_mask0 ^ source_union0
            train_matched_bits1 &= train_mask1 ^ source_union1

            train_support0 = train_matched_bits0.bit_count()
            train_support1 = train_matched_bits1.bit_count()
            train_precision0, train_precision1, train_recall0, train_recall1 = compute_precision_recall(
                train_support0,
                train_support1,
                train_total0,
                train_total1,
            )

            test_matched_bits0 = evaluate_template(template.compiled_expr, test_mask0, test_values0)
            test_matched_bits1 = evaluate_template(template.compiled_expr, test_mask1, test_values1)
            test_support0 = test_matched_bits0.bit_count()
            test_support1 = test_matched_bits1.bit_count()
            test_precision0, test_precision1, test_recall0, test_recall1 = compute_precision_recall(
                test_support0,
                test_support1,
                test_total0,
                test_total1,
            )

            record = HKMRecord(
                hkm_id=f"{subset_key}__combo{combo_idx}__tmpl{template_idx}",
                formula_id=template.formula_id,
                length=length,
                formula=template.formula,
                atom_ids=atom_ids,
                features=features,
                source_pattern_ids=source_pattern_ids,
                train_support0=train_support0,
                train_support1=train_support1,
                train_precision0=train_precision0,
                train_precision1=train_precision1,
                train_recall0=train_recall0,
                train_recall1=train_recall1,
                test_support0=test_support0,
                test_support1=test_support1,
                test_precision0=test_precision0,
                test_precision1=test_precision1,
                test_recall0=test_recall0,
                test_recall1=test_recall1,
                train_matched_bits0=train_matched_bits0,
                train_matched_bits1=train_matched_bits1,
                test_matched_bits0=test_matched_bits0,
                test_matched_bits1=test_matched_bits1,
            )
            for logic_name in LOGIC_NAMES:
                update_logic_pool(
                    local_records_by_logic[logic_name],
                    local_heaps_by_logic[logic_name],
                    record,
                    logic_name,
                    WORKER_TOP_HKMS_PER_LENGTH,
                )
            processed_count += 1

            if WORKER_CHECKPOINT_EVERY_HKMS > 0 and processed_count % WORKER_CHECKPOINT_EVERY_HKMS == 0:
                next_combo_idx = combo_idx
                next_template_idx = template_idx + 1
                if next_template_idx >= len(templates):
                    next_combo_idx = combo_idx + 1
                    next_template_idx = 0
                save_partial_checkpoint_result(
                    partial_checkpoint_path,
                    work_item,
                    processed_count,
                    next_combo_idx,
                    next_template_idx,
                    {
                        logic_name: sorted(
                            local_records_by_logic[logic_name].values(),
                            key=lambda item: logic_rank_key(item, logic_name),
                            reverse=True,
                        )
                        for logic_name in LOGIC_NAMES
                    },
                )

        resume_template_idx = 0

    top_records_by_logic = {
        logic_name: sorted(
            local_records_by_logic[logic_name].values(),
            key=lambda item: logic_rank_key(item, logic_name),
            reverse=True,
        )
        for logic_name in LOGIC_NAMES
    }
    result = {
        "length": length,
        "feature_subset": list(feature_subset),
        "processed_count": processed_count,
        "top_records_by_logic": {
            logic_name: [record_to_dict(record) for record in records]
            for logic_name, records in top_records_by_logic.items()
        },
    }
    save_checkpoint_result(final_checkpoint_path, result)
    if partial_checkpoint_path.exists():
        partial_checkpoint_path.unlink()
    return result


def save_checkpoint_result(path: Path, result: dict) -> None:
    atomic_write_json(path, result)


def load_checkpoint_result(path: Path) -> Optional[dict]:
    return try_load_json_file(path)


def merge_checkpoint_results(
    checkpoint_results: Sequence[dict],
    lengths: Iterable[int],
    top_hkms_per_length: int,
) -> Tuple[Dict[str, Dict[int, List[HKMRecord]]], Dict[int, int]]:
    records_by_logic_by_length: Dict[str, Dict[int, Dict[Tuple[int, int], HKMRecord]]] = {
        logic_name: {length: {} for length in lengths}
        for logic_name in LOGIC_NAMES
    }
    heaps_by_logic_by_length: Dict[str, Dict[int, List[Tuple[Tuple[float, float, float, int, int], Tuple[int, int], str]]]] = {
        logic_name: {length: [] for length in lengths}
        for logic_name in LOGIC_NAMES
    }
    processed_counts: Dict[int, int] = {length: 0 for length in lengths}

    for result in checkpoint_results:
        length = int(result["length"])
        processed_counts[length] += int(result["processed_count"])
        payload_by_logic = result.get("top_records_by_logic", {})
        for logic_name in LOGIC_NAMES:
            for record_payload in payload_by_logic.get(logic_name, []):
                record = record_from_dict(record_payload)
                update_logic_pool(
                    records_by_logic_by_length[logic_name][length],
                    heaps_by_logic_by_length[logic_name][length],
                    record,
                    logic_name,
                    top_hkms_per_length,
                )

    merged: Dict[str, Dict[int, List[HKMRecord]]] = {logic_name: {} for logic_name in LOGIC_NAMES}
    for logic_name in LOGIC_NAMES:
        for length in lengths:
            records = list(records_by_logic_by_length[logic_name][length].values())
            records.sort(key=lambda item: logic_rank_key(item, logic_name), reverse=True)
            merged[logic_name][length] = records
    return merged, processed_counts


def enumerate_top_hkms_by_length(
    atom_infos: Sequence[AtomMatchInfo],
    templates_by_length: Dict[int, List[Template]],
    top_hkms_per_length: int,
    lengths: Sequence[int],
    train_totals: Tuple[int, int],
    test_totals: Tuple[int, int],
    output_dir: Path,
    num_workers: int,
    checkpoint_every_hkms: int,
) -> Tuple[Dict[str, Dict[int, List[HKMRecord]]], Dict[int, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atom_infos_by_feature: Dict[str, List[AtomMatchInfo]] = {}
    for info in atom_infos:
        atom_infos_by_feature.setdefault(info.atom.feature, []).append(info)

    checkpoint_results: List[dict] = []
    all_work_items = build_work_items(atom_infos, lengths)
    total_task_count = len(all_work_items)
    completed_task_count = 0

    for length in lengths:
        length_work_items = [item for item in all_work_items if item.length == length]
        pending_work_items: List[WorkItem] = []
        skipped_for_length = 0

        for work_item in length_work_items:
            checkpoint_path = checkpoint_path_for_work_item(output_dir, work_item)
            if checkpoint_path.exists():
                checkpoint_payload = load_checkpoint_result(checkpoint_path)
                if checkpoint_payload is not None:
                    checkpoint_results.append(checkpoint_payload)
                    completed_task_count += 1
                    skipped_for_length += 1
                else:
                    pending_work_items.append(work_item)
            else:
                pending_work_items.append(work_item)

        save_progress_summary(output_dir, total_task_count, completed_task_count, 0)

        if pending_work_items:
            print(
                f"Processing length {length} with {len(pending_work_items)} pending chunks "
                f"and {skipped_for_length} completed chunks using {num_workers} workers",
                flush=True,
            )
            with mp.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(
                    atom_infos_by_feature,
                    train_totals,
                    test_totals,
                    top_hkms_per_length,
                    HKM_TEMPLATE_DIR,
                    (length,),
                    output_dir,
                    checkpoint_every_hkms,
                ),
            ) as pool:
                for length_completed_idx, result in enumerate(pool.imap_unordered(process_work_item, pending_work_items), start=1):
                    checkpoint_results.append(result)
                    completed_task_count += 1
                    save_progress_summary(output_dir, total_task_count, completed_task_count, 0)
                    work_item = WorkItem(length=int(result["length"]), feature_subset=tuple(result["feature_subset"]))
                    print(
                        f"Completed length {length} chunk {length_completed_idx}/{len(pending_work_items)}: "
                        f"{chunk_key_for_feature_subset(work_item.length, work_item.feature_subset)} "
                        f"(processed {result['processed_count']} HKMs)",
                        flush=True,
                    )
        else:
            print(f"All length {length} chunks already exist. Loading saved results only.", flush=True)

    return merge_checkpoint_results(checkpoint_results, lengths, top_hkms_per_length)


def write_final_hkm_outputs(
    output_dir: Path,
    top_records_by_logic_by_length: Dict[str, Dict[int, List[HKMRecord]]],
    processed_counts: Dict[int, int],
    top_atoms_per_feature: int,
    top_hkms_per_length: int,
    curve_top_n: Optional[int],
    train_totals: Tuple[int, int],
    test_totals: Tuple[int, int],
    num_workers: int,
    lengths: Sequence[int],
    checkpoint_every_hkms: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "top_atoms_per_feature": top_atoms_per_feature,
        "top_hkms_per_length": top_hkms_per_length,
        "curve_top_n": curve_top_n,
        "num_workers": num_workers,
        "lengths": list(lengths),
        "logic_names": list(LOGIC_NAMES),
        "checkpoint_every_hkms": checkpoint_every_hkms,
        "processed_hkm_counts_by_length": processed_counts,
        "retained_hkm_counts_by_logic_and_length": {
            logic_name: {
                length: len(top_records_by_logic_by_length[logic_name].get(length, []))
                for length in lengths
            }
            for logic_name in LOGIC_NAMES
        },
        "train_totals": {"label0": train_totals[0], "label1": train_totals[1]},
        "test_totals": {"label0": test_totals[0], "label1": test_totals[1]},
    }
    atomic_write_json(output_dir / "run_summary.json", summary, indent=2)

    for logic_name in LOGIC_NAMES:
        for length in lengths:
            records = top_records_by_logic_by_length[logic_name].get(length, [])
            jsonl_path = output_dir / f"hkms_{logic_name}_len{length}_top{top_hkms_per_length}.jsonl"
            csv_path = output_dir / f"hkm_metrics_{logic_name}_len{length}_top{top_hkms_per_length}.csv"

            with jsonl_path.open("w", encoding="utf-8") as f:
                for rank, record in enumerate(records, start=1):
                    payload = {"rank": rank, **record_to_dict(record)}
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rank",
                        "hkm_id",
                        "formula_id",
                        "length",
                        "formula",
                        "atom_ids",
                        "features",
                        "source_pattern_ids",
                        "train_dedup_support0_excluding_sources",
                        "train_dedup_support1_excluding_sources",
                        "train_dedup_precision0_excluding_sources",
                        "train_dedup_precision1_excluding_sources",
                        "train_dedup_recall0_excluding_sources",
                        "train_dedup_recall1_excluding_sources",
                        "train_precision0_recall0_mean",
                        "test_support0",
                        "test_support1",
                        "test_precision0",
                        "test_precision1",
                        "test_recall0",
                        "test_recall1",
                    ]
                )
                for rank, record in enumerate(records, start=1):
                    writer.writerow(
                        [
                            rank,
                            record.hkm_id,
                            record.formula_id,
                            record.length,
                            record.formula,
                            json.dumps(record.atom_ids, ensure_ascii=False),
                            json.dumps(record.features, ensure_ascii=False),
                            json.dumps(record.source_pattern_ids, ensure_ascii=False),
                            record.train_support0,
                            record.train_support1,
                            record.train_precision0,
                            record.train_precision1,
                            record.train_recall0,
                            record.train_recall1,
                            precision0_recall0_mean(record),
                            record.test_support0,
                            record.test_support1,
                            record.test_precision0,
                            record.test_precision1,
                            record.test_recall0,
                            record.test_recall1,
                        ]
                    )


def plot_saved_hkm_curves(
    output_dir: Path,
    curve_top_n: Optional[int] = None,
    plot_config: Optional[dict] = None,
) -> None:
    plot_config = plot_config or {}
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    lengths = [int(length) for length in summary.get("lengths", list(LENGTHS))]
    logic_names = list(summary.get("logic_names", list(LOGIC_NAMES)))
    train_totals = (
        int(summary["train_totals"]["label0"]),
        int(summary["train_totals"]["label1"]),
    )
    test_totals = (
        int(summary["test_totals"]["label0"]),
        int(summary["test_totals"]["label1"]),
    )
    top_hkms_per_length = int(summary["top_hkms_per_length"])

    for logic_name in logic_names:
        for length in lengths:
            jsonl_path = output_dir / f"hkms_{logic_name}_len{length}_top{top_hkms_per_length}.jsonl"
            if not jsonl_path.exists():
                continue
            curve_suffix = "all" if curve_top_n is None else f"top{curve_top_n}"
            curve_csv_path = output_dir / f"hkm_curve_{logic_name}_len{length}_{curve_suffix}.csv"
            plot_path = output_dir / f"hkm_curve_{logic_name}_len{length}_{curve_suffix}.png"

            records: List[HKMRecord] = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    records.append(record_from_dict(payload))

            effective_count = len(records) if curve_top_n is None else min(curve_top_n, len(records))
            curve_records = records[:effective_count]
            curve_rows: List[List[float]] = []
            cumulative_train0 = 0
            cumulative_train1 = 0
            cumulative_test0 = 0
            cumulative_test1 = 0
            x_values: List[int] = []
            train0_values: List[float] = []
            train1_values: List[float] = []
            test0_values: List[float] = []
            test1_values: List[float] = []

            for idx, record in enumerate(curve_records, start=1):
                cumulative_train0 |= record.train_matched_bits0
                cumulative_train1 |= record.train_matched_bits1
                cumulative_test0 |= record.test_matched_bits0
                cumulative_test1 |= record.test_matched_bits1

                train0_ratio = cumulative_train0.bit_count() / train_totals[0] if train_totals[0] else 0.0
                train1_ratio = cumulative_train1.bit_count() / train_totals[1] if train_totals[1] else 0.0
                test0_ratio = cumulative_test0.bit_count() / test_totals[0] if test_totals[0] else 0.0
                test1_ratio = cumulative_test1.bit_count() / test_totals[1] if test_totals[1] else 0.0

                x_values.append(idx)
                train0_values.append(train0_ratio)
                train1_values.append(train1_ratio)
                test0_values.append(test0_ratio)
                test1_values.append(test1_ratio)
                curve_rows.append([idx, train0_ratio, train1_ratio, test0_ratio, test1_ratio])

            with curve_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "top_n_hkms",
                        "train_label0_ratio",
                        "train_label1_ratio",
                        "test_label0_ratio",
                        "test_label1_ratio",
                    ]
                )
                writer.writerows(curve_rows)

            figsize = tuple(plot_config.get("figsize", (8, 5)))
            dpi = int(plot_config.get("dpi", 150))
            train0_color = plot_config.get("train_label0_color", "tab:blue")
            train1_color = plot_config.get("train_label1_color", train0_color)
            test0_color = plot_config.get("test_label0_color", "tab:orange")
            test1_color = plot_config.get("test_label1_color", test0_color)
            train0_linestyle = plot_config.get("train_label0_linestyle", "-")
            train1_linestyle = plot_config.get("train_label1_linestyle", "--")
            test0_linestyle = plot_config.get("test_label0_linestyle", "-")
            test1_linestyle = plot_config.get("test_label1_linestyle", "--")
            train0_label = plot_config.get("train_label0_label", "train label0")
            train1_label = plot_config.get("train_label1_label", "train label1")
            test0_label = plot_config.get("test_label0_label", "test label0")
            test1_label = plot_config.get("test_label1_label", "test label1")
            xlabel = plot_config.get("xlabel", "Top N HKMs")
            ylabel = plot_config.get("ylabel", "Covered patient ratio")
            title_template = plot_config.get("title_template", "{logic_name} - HKM Length {length} Coverage Curve")
            ylim = plot_config.get("ylim", (0.0, 1.0))
            grid = bool(plot_config.get("grid", True))
            grid_alpha = float(plot_config.get("grid_alpha", 0.3))
            legend = bool(plot_config.get("legend", True))

            plt.figure(figsize=figsize)
            plt.plot(x_values, train0_values, color=train0_color, linestyle=train0_linestyle, label=train0_label)
            plt.plot(x_values, train1_values, color=train1_color, linestyle=train1_linestyle, label=train1_label)
            plt.plot(x_values, test0_values, color=test0_color, linestyle=test0_linestyle, label=test0_label)
            plt.plot(x_values, test1_values, color=test1_color, linestyle=test1_linestyle, label=test1_label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title_template.format(length=length, top_n=effective_count, logic_name=logic_name))
            if ylim is not None:
                plt.ylim(*ylim)
            if grid:
                plt.grid(True, alpha=grid_alpha)
            if legend:
                plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=dpi)
            plt.close()


def build_output_dir(
    top_atoms_per_feature: int,
    top_hkms_per_length: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    return output_root / (
        f"atoms_per_feature_{top_atoms_per_feature}"
        f"_top_hkms_{top_hkms_per_length}"
    )
