from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
TRAIN_CSV = SCRIPT_DIR / "window4_cut_0_days_all_patients" / "split_data" / "eval_train_subseq_flattened.csv"
TEST_CSV = SCRIPT_DIR / "window4_cut_0_days_all_patients" / "split_data" / "eval_test_subseq_flattened.csv"

OUTPUT_ROOT = SCRIPT_DIR / "black_box_results" / "baseline"
RESULTS_XLSX = OUTPUT_ROOT / "results_mean.xlsx"
RESULTS_CSV = OUTPUT_ROOT / "results_mean.csv"
PER_REPEAT_CSV = OUTPUT_ROOT / "results_per_repeat.csv"
MANIFEST_JSON = OUTPUT_ROOT / "baseline_manifest.json"

CLASS_LABELS = [0, 1]
NUM_REPEATS = 100
BASE_RANDOM_STATE = 42


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
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


def build_models(seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVM (RBF)": SVC(kernel="rbf", random_state=seed, gamma="scale"),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=seed,
            eval_metric="mlogloss",
        )
    return models


def evaluate_once(
    train_csv: Path,
    test_csv: Path,
    class_labels: Sequence[int],
    seed: int,
) -> List[Dict[str, Any]]:
    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)
    x_train_scaled, x_test_scaled, y_train, y_test = prepare_features(train_df, test_df)

    results: List[Dict[str, Any]] = []
    for model_name, model in build_models(seed).items():
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
            "repeat_index": seed - BASE_RANDOM_STATE,
            "seed": seed,
            "Model": model_name,
            "Accuracy": float(acc),
            "Macro_F1": float(f1.mean()),
        }
        for idx, label in enumerate(class_labels):
            record[f"Class{label}_Precision"] = float(precision[idx])
            record[f"Class{label}_Recall"] = float(recall[idx])
            record[f"Class{label}_F1"] = float(f1[idx])
            record[f"Support{label}"] = int(support[idx])
        results.append(record)
    return results


def summarize_results(per_repeat_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Accuracy",
        "Macro_F1",
        "Class0_Precision",
        "Class0_Recall",
        "Class0_F1",
        "Class1_Precision",
        "Class1_Recall",
        "Class1_F1",
    ]
    support_cols = ["Support0", "Support1"]

    agg_spec = {col: "mean" for col in metric_cols}
    agg_spec.update({col: "first" for col in support_cols})
    summary = per_repeat_df.groupby("Model", as_index=False).agg(agg_spec)
    summary.insert(0, "information", "window size: 4\nlabel: [(0, 0, 120), (1, 121, inf)]")
    summary.insert(1, "Window Size", 4)
    summary.insert(2, "Repeat_Count", NUM_REPEATS)
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for repeat_idx in range(NUM_REPEATS):
        seed = BASE_RANDOM_STATE + repeat_idx
        all_rows.extend(evaluate_once(TRAIN_CSV, TEST_CSV, CLASS_LABELS, seed))

    per_repeat_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(per_repeat_df)

    per_repeat_df.to_csv(PER_REPEAT_CSV, index=False)
    summary_df.to_csv(RESULTS_CSV, index=False)
    summary_df.to_excel(RESULTS_XLSX, sheet_name="baseline_mean", index=False)

    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "train_csv": str(TRAIN_CSV),
                "test_csv": str(TEST_CSV),
                "class_labels": CLASS_LABELS,
                "window_len": 4,
                "num_repeats": NUM_REPEATS,
                "base_random_state": BASE_RANDOM_STATE,
                "output_files": {
                    "results_xlsx": str(RESULTS_XLSX),
                    "results_csv": str(RESULTS_CSV),
                    "per_repeat_csv": str(PER_REPEAT_CSV),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
