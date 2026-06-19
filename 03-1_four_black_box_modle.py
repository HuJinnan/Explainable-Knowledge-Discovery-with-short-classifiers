from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_BUILDER_PATH = SCRIPT_DIR / "02_build_window4_all_patients_with_eval_split.py"
DIRECT_DATA_DIR = SCRIPT_DIR / "breast_cancer_single_snapshot_0d_last_window_supported"
DIRECT_STATES_CSV = DIRECT_DATA_DIR / "patient_states.csv"
DIRECT_SNAPSHOTS_CSV = DIRECT_DATA_DIR / "patient_snapshots.csv"
OUTPUT_ROOT = SCRIPT_DIR / "black_box_results" / "new_all_windows"
DATASETS_ROOT = OUTPUT_ROOT / "datasets"
RESULTS_XLSX = OUTPUT_ROOT / "results.xlsx"
RESULTS_CSV = OUTPUT_ROOT / "results_long.csv"
MANIFEST_JSON = OUTPUT_ROOT / "experiment_manifest.json"
DATASET_MANIFEST_JSON = OUTPUT_ROOT / "dataset_manifest.json"

RANDOM_STATE = 42
TRAIN_RATIO = 0.8


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    section: str
    window_len: int
    label_boundaries: Sequence[Tuple[int, int, float]]
    info_window_text: str
    window_size_cell: int
    model_prefix: str = ""


def build_experiment_specs() -> List[ExperimentSpec]:
    three_class_short = ((0, 0, 120), (1, 121, 240), (2, 241, inf))
    three_class_long = ((0, 0, 180), (1, 181, 360), (2, 361, inf))
    three_class_alt_1 = ((0, 0, 120), (1, 121, 720), (2, 721, inf))
    three_class_alt_2 = ((0, 0, 120), (1, 121, 1200), (2, 1201, inf))
    two_class_short = ((0, 0, 120), (1, 121, inf))
    two_class_long = ((0, 0, 180), (1, 181, inf))

    specs: List[ExperimentSpec] = [
        ExperimentSpec("three_class_w1", "three_class", 1, three_class_short, "1", 1),
        ExperimentSpec("three_class_w2", "three_class", 2, three_class_short, "2", 2),
        ExperimentSpec("three_class_w3", "three_class", 3, three_class_short, "3", 3),
        ExperimentSpec("three_class_w4", "three_class", 4, three_class_short, "4", 4),
        ExperimentSpec("three_class_w5", "three_class", 5, three_class_long, "5", 5),
        ExperimentSpec("three_class_w6", "three_class", 6, three_class_long, "6", 6),
        ExperimentSpec("three_class_w4_alt1", "three_class", 4, three_class_alt_1, "4", 4, "(1) "),
        ExperimentSpec("three_class_w4_alt2", "three_class", 4, three_class_alt_2, "4", 4, "(2) "),
        ExperimentSpec("two_class_w1", "two_class", 1, two_class_short, "1", 1, "(2) "),
        ExperimentSpec("two_class_w2", "two_class", 2, two_class_short, "2", 2, "(2) "),
        ExperimentSpec("two_class_w3", "two_class", 3, two_class_short, "3", 3, "(2) "),
        ExperimentSpec("two_class_w4", "two_class", 4, two_class_short, "4", 4, "(3) "),
        ExperimentSpec("two_class_w5", "two_class", 5, two_class_long, "5", 5, "(2) "),
        ExperimentSpec("two_class_w6", "two_class", 6, two_class_long, "6-2", 6, "(2) "),
    ]
    return specs


def load_split_builder():
    spec = importlib.util.spec_from_file_location("split_builder", SPLIT_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load split builder from {SPLIT_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_boundaries(boundaries: Sequence[Tuple[int, int, float]]) -> str:
    parts = []
    for label, low, high in boundaries:
        high_text = "inf" if high == inf else str(int(high))
        parts.append(f"({label}, {int(low)}, {high_text})")
    return "[" + ", ".join(parts) + "]"


def configure_split_builder(module: Any, spec: ExperimentSpec) -> None:
    module.STATES_CSV = DIRECT_STATES_CSV
    module.SNAPSHOTS_CSV = DIRECT_SNAPSHOTS_CSV
    module.WINDOW_LEN = spec.window_len
    module.LABEL_BOUNDARIES = list(spec.label_boundaries)
    module.USE_ALL_PATIENT = 1
    module.CUT_DAYS = 0
    module.RANDOM_SEED = RANDOM_STATE
    module.FULL_TRAIN_RATIO = TRAIN_RATIO


def configure_split_builder_data_paths(module: Any) -> None:
    module.STATES_CSV = DIRECT_STATES_CSV
    module.SNAPSHOTS_CSV = DIRECT_SNAPSHOTS_CSV


def label_counter_dict(labels: Iterable[int]) -> Dict[int, int]:
    series = pd.Series(list(labels), dtype="int64")
    if series.empty:
        return {}
    counts = series.value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def generate_unsplit_datasets_from_loaded(
    module: Any,
    patient_states: Dict[str, List[dict]],
    patient_death_date: Dict[str, Any],
    effective_dates: Dict[str, Any],
) -> Tuple[List[dict], List[dict], List[dict], List[dict], Dict[str, Any]]:
    last_label = module.LABEL_BOUNDARIES[-1][0]
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
        if n < module.WINDOW_LEN:
            continue

        is_alive = (pid not in patient_death_date) and module.USE_ALL_PATIENT

        days_list = []
        for state in states:
            end = module.parse_date(state.get("window_end"))
            if end:
                days_list.append(module.days_between(end, reference_date))
            else:
                days_list.append(None)
        labels = [module.get_label_by_days(days) if days is not None else -1 for days in days_list]

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
            while j < n and labels[j] == current_label and module.states_are_continuous(states[start : j + 1]):
                j += 1
            seg_len = j - start
            if seg_len >= module.WINDOW_LEN:
                num_windows = seg_len - module.WINDOW_LEN + 1
                for offset in range(num_windows):
                    win_start = start + offset
                    win_end = win_start + module.WINDOW_LEN - 1
                    window_states = states[win_start : win_end + 1]
                    if not module.states_are_continuous(window_states):
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
                        flat_row.update(module.flatten_state(state, pos))
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

    dedup_mining_ips, dedup_mining_flat, removed = module.dedup_samples(mining_ips, mining_flat)
    info = {
        "raw_mining_sample_count": len(mining_ips),
        "dedup_mining_sample_count": len(dedup_mining_ips),
        "dedup_removed_count": removed,
        "eval_single_window_sample_count": len(eval_ips),
        "dedup_mining_label_dist": label_counter_dict(obj["label"] for obj in dedup_mining_ips),
        "eval_single_window_label_dist": label_counter_dict(obj["label"] for obj in eval_ips),
    }
    return dedup_mining_ips, dedup_mining_flat, eval_ips, eval_flat, info


def write_mining_split_outputs(
    module: Any,
    output_dir: Path,
    train_ips: List[dict],
    train_flat: List[dict],
    test_ips: List[dict],
    test_flat: List[dict],
    split_info: Dict[str, Any],
    generation_info: Dict[str, Any],
    train_patient_ids: List[str],
    test_patient_ids: List[str],
) -> Tuple[Path, Path]:
    split_dir = output_dir / "split_data"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_ips_path = split_dir / "mining_train_subseq_ips.jsonl"
    train_csv_path = split_dir / "mining_train_subseq_flattened.csv"
    test_ips_path = split_dir / "mining_test_subseq_ips.jsonl"
    test_csv_path = split_dir / "mining_test_subseq_flattened.csv"

    module.write_jsonl(train_ips_path, train_ips)
    module.write_flattened_csv(train_csv_path, train_flat)
    module.write_jsonl(test_ips_path, test_ips)
    module.write_flattened_csv(test_csv_path, test_flat)

    (split_dir / "split_info.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (split_dir / "generation_info.json").write_text(
        json.dumps(generation_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (split_dir / "train_patient_ids.json").write_text(
        json.dumps(train_patient_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (split_dir / "test_patient_ids.json").write_text(
        json.dumps(test_patient_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_csv_path, test_csv_path


def dataset_output_dir(spec: ExperimentSpec) -> Path:
    return DATASETS_ROOT / spec.experiment_id


def dataset_split_paths(spec: ExperimentSpec) -> Dict[str, Path]:
    split_dir = dataset_output_dir(spec) / "split_data"
    return {
        "split_dir": split_dir,
        "train_csv": split_dir / "mining_train_subseq_flattened.csv",
        "test_csv": split_dir / "mining_test_subseq_flattened.csv",
        "train_ips": split_dir / "mining_train_subseq_ips.jsonl",
        "test_ips": split_dir / "mining_test_subseq_ips.jsonl",
        "split_info": split_dir / "split_info.json",
        "generation_info": split_dir / "generation_info.json",
        "train_patients": split_dir / "train_patient_ids.json",
        "test_patients": split_dir / "test_patient_ids.json",
    }


def dataset_is_ready(spec: ExperimentSpec) -> bool:
    paths = dataset_split_paths(spec)
    required_keys = [
        "train_csv",
        "test_csv",
        "train_ips",
        "test_ips",
        "split_info",
        "generation_info",
        "train_patients",
        "test_patients",
    ]
    return all(paths[key].exists() for key in required_keys)


def build_mining_dataset_for_experiment(
    module: Any,
    spec: ExperimentSpec,
    patient_states: Dict[str, List[dict]],
    patient_death_date: Dict[str, Any],
    effective_dates: Dict[str, Any],
) -> Dict[str, Any]:
    configure_split_builder(module, spec)
    output_dir = DATASETS_ROOT / spec.experiment_id

    mining_ips, mining_flat, eval_ips, eval_flat, generation_info = generate_unsplit_datasets_from_loaded(
        module,
        patient_states,
        patient_death_date,
        effective_dates,
    )
    train_patients, test_patients, mining_train_idx, mining_test_idx, split_info = (
        module.derive_patient_split_from_mining_data(
            mining_ips,
            seed=RANDOM_STATE,
            train_ratio=TRAIN_RATIO,
        )
    )
    mining_train_ips = [mining_ips[idx] for idx in mining_train_idx]
    mining_train_flat = [mining_flat[idx] for idx in mining_train_idx]
    mining_test_ips = [mining_ips[idx] for idx in mining_test_idx]
    mining_test_flat = [mining_flat[idx] for idx in mining_test_idx]

    split_info = dict(split_info)
    split_info.update(
        {
            "random_seed": RANDOM_STATE,
            "window_len": spec.window_len,
            "label_boundaries": [list(item) for item in spec.label_boundaries],
            "mining_train_samples": len(mining_train_idx),
            "mining_test_samples": len(mining_test_idx),
            "mining_train_label_dist": label_counter_dict(row["label"] for row in mining_train_ips),
            "mining_test_label_dist": label_counter_dict(row["label"] for row in mining_test_ips),
        }
    )

    train_csv, test_csv = write_mining_split_outputs(
        module,
        output_dir,
        mining_train_ips,
        mining_train_flat,
        mining_test_ips,
        mining_test_flat,
        split_info,
        generation_info,
        train_patients,
        test_patients,
    )
    return {
        "output_dir": output_dir,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "split_info": split_info,
        "generation_info": generation_info,
    }


def load_cached_dataset_info(spec: ExperimentSpec) -> Dict[str, Any]:
    paths = dataset_split_paths(spec)
    split_info = json.loads(paths["split_info"].read_text(encoding="utf-8"))
    generation_info = json.loads(paths["generation_info"].read_text(encoding="utf-8"))
    return {
        "output_dir": dataset_output_dir(spec),
        "train_csv": paths["train_csv"],
        "test_csv": paths["test_csv"],
        "split_info": split_info,
        "generation_info": generation_info,
    }


def prepare_all_datasets(
    split_builder: Any,
    experiment_specs: Sequence[ExperimentSpec],
) -> Dict[str, Dict[str, Any]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    configure_split_builder_data_paths(split_builder)
    print("Loading base patient snapshot metadata...", flush=True)
    patient_death_date, patient_snapshot_date = split_builder.load_snapshot_info()
    print("Loading patient states once for all experiments...", flush=True)
    patient_states = split_builder.load_patient_states(patient_death_date)
    effective_dates = split_builder.build_effective_reference_dates(
        patient_states,
        patient_death_date,
        patient_snapshot_date,
    )

    dataset_infos: Dict[str, Dict[str, Any]] = {}
    dataset_manifest_rows: List[Dict[str, Any]] = []
    for spec in experiment_specs:
        if dataset_is_ready(spec):
            print(f"Reusing cached dataset {spec.experiment_id}...", flush=True)
            dataset_info = load_cached_dataset_info(spec)
        else:
            print(f"Preparing dataset {spec.experiment_id}...", flush=True)
            dataset_info = build_mining_dataset_for_experiment(
                split_builder,
                spec,
                patient_states,
                patient_death_date,
                effective_dates,
            )
        dataset_infos[spec.experiment_id] = dataset_info
        dataset_manifest_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "section": spec.section,
                "window_len": spec.window_len,
                "label_boundaries": format_boundaries(spec.label_boundaries),
                "output_dir": str(dataset_info["output_dir"]),
                "train_csv": str(dataset_info["train_csv"]),
                "test_csv": str(dataset_info["test_csv"]),
                "split_info_path": str(dataset_split_paths(spec)["split_info"]),
                "generation_info_path": str(dataset_split_paths(spec)["generation_info"]),
            }
        )

    DATASET_MANIFEST_JSON.write_text(
        json.dumps(dataset_manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset_infos


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    label_col = "label"
    feature_cols = [c for c in train_df.columns if c not in [label_col, "sample_id", "source_patient_id"]]

    x_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].copy()

    for col in x_train.columns:
        x_train[col] = x_train[col].replace({"2+": 2})
        x_test[col] = x_test[col].replace({"2+": 2})
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
        median_val = x_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        x_train[col] = x_train[col].fillna(median_val)
        x_test[col] = x_test[col].fillna(median_val)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test


def build_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE, gamma="scale"),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
    return models


def evaluate_models(train_csv: Path, test_csv: Path, class_labels: Sequence[int]) -> List[Dict[str, Any]]:
    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)
    x_train_scaled, x_test_scaled, y_train, y_test = prepare_features(train_df, test_df)

    results: List[Dict[str, Any]] = []
    for model_name, model in build_models().items():
        print(f"  Training {model_name} on {train_csv.parent.parent.name}...", flush=True)
        if model_name == "XGBoost":
            train_labels = sorted(int(label) for label in pd.Series(y_train).dropna().unique())
            label_to_encoded = {label: idx for idx, label in enumerate(train_labels)}
            encoded_to_label = {idx: label for label, idx in label_to_encoded.items()}
            y_train_fit = pd.Series(y_train).map(label_to_encoded).astype(int)
            model.fit(x_train_scaled, y_train_fit)
            encoded_pred = model.predict(x_test_scaled)
            y_pred = np.array([encoded_to_label[int(label)] for label in encoded_pred])
        else:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=list(class_labels),
            zero_division=0,
        )
        record: Dict[str, Any] = {
            "Model": model_name,
            "Accuracy": acc,
            "Macro_F1": float(f1.mean()),
        }
        for idx, label in enumerate(class_labels):
            record[f"Class{label}_Precision"] = float(precision[idx])
            record[f"Class{label}_Recall"] = float(recall[idx])
            record[f"Class{label}_F1"] = float(f1[idx])
            record[f"Support{label}"] = int(support[idx])
        results.append(record)
    return results


def build_summary_rows(spec: ExperimentSpec, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    info_text = f"window size: {spec.info_window_text}\nlabel: {format_boundaries(spec.label_boundaries)}"
    for idx, metric in enumerate(metrics):
        row = {
            "information": info_text if idx == 0 else None,
            "Window Size": spec.window_size_cell if idx == 0 else None,
            "Model": f"{spec.model_prefix}{metric['Model']}",
            "Class0_Precision": metric.get("Class0_Precision"),
            "Class0_Recall": metric.get("Class0_Recall"),
            "Class1_Precision": metric.get("Class1_Precision"),
            "Class1_Recall": metric.get("Class1_Recall"),
            "Class2_Precision": metric.get("Class2_Precision"),
            "Class2_Recall": metric.get("Class2_Recall"),
            "Macro_F1": metric.get("Macro_F1"),
            "Support0": metric.get("Support0"),
            "Support1": metric.get("Support1"),
            "Support2": metric.get("Support2"),
        }
        rows.append(row)
    return rows


def write_summary_outputs(summary_rows: List[Dict[str, Any]], manifest_rows: List[Dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    columns = [
        "information",
        "Window Size",
        "Model",
        "Class0_Precision",
        "Class0_Recall",
        "Class1_Precision",
        "Class1_Recall",
        "Class2_Precision",
        "Class2_Recall",
        "Macro_F1",
        "Support0",
        "Support1",
        "Support2",
    ]
    df = pd.DataFrame(summary_rows, columns=columns)
    df.to_excel(RESULTS_XLSX, sheet_name="results", index=False)
    pd.DataFrame(manifest_rows).to_csv(RESULTS_CSV, index=False)
    MANIFEST_JSON.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_section_separator(rows: List[Dict[str, Any]]) -> None:
    rows.append({key: None for key in [
        "information",
        "Window Size",
        "Model",
        "Class0_Precision",
        "Class0_Recall",
        "Class1_Precision",
        "Class1_Recall",
        "Class2_Precision",
        "Class2_Recall",
        "Macro_F1",
        "Support0",
        "Support1",
        "Support2",
    ]})


def run_all_models_from_datasets(
    experiment_specs: Sequence[ExperimentSpec],
    dataset_infos: Dict[str, Dict[str, Any]],
) -> None:
    summary_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    grouped_specs = {
        "three_class": [spec for spec in experiment_specs if spec.section == "three_class"],
        "two_class": [spec for spec in experiment_specs if spec.section == "two_class"],
    }

    for section_idx, section_name in enumerate(("three_class", "two_class"), start=1):
        for spec in grouped_specs[section_name]:
            print(f"Running models for {spec.experiment_id}...", flush=True)
            dataset_info = dataset_infos[spec.experiment_id]
            class_labels = [label for label, _, _ in spec.label_boundaries]
            metrics = evaluate_models(
                dataset_info["train_csv"],
                dataset_info["test_csv"],
                class_labels=class_labels,
            )
            summary_rows.extend(build_summary_rows(spec, metrics))

            for metric in metrics:
                manifest_rows.append(
                    {
                        "experiment_id": spec.experiment_id,
                        "section": spec.section,
                        "window_len": spec.window_len,
                        "label_boundaries": format_boundaries(spec.label_boundaries),
                        "model": metric["Model"],
                        "train_csv": str(dataset_info["train_csv"]),
                        "test_csv": str(dataset_info["test_csv"]),
                        "output_dir": str(dataset_info["output_dir"]),
                        **metric,
                    }
                )

        if section_idx == 1:
            append_section_separator(summary_rows)
            append_section_separator(summary_rows)

    summary_rows.append({key: None for key in summary_rows[0].keys()})
    summary_rows.append({"information": "comment"})
    summary_rows.append(
        {
            "information": (
                "Updated results generated with all-window mining splits and "
                "supported breast-cancer patients including surviving patients."
            )
        }
    )

    write_summary_outputs(summary_rows, manifest_rows)
    print(f"Summary workbook saved to: {RESULTS_XLSX}", flush=True)
    print(f"Long-form metrics saved to: {RESULTS_CSV}", flush=True)


def main() -> None:
    split_builder = load_split_builder()
    experiment_specs = build_experiment_specs()
    dataset_infos = prepare_all_datasets(split_builder, experiment_specs)
    run_all_models_from_datasets(experiment_specs, dataset_infos)


if __name__ == "__main__":
    main()
