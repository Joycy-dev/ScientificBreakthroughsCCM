"""Leakage-free five-fold training for the AI-domain cusp indicator.

All sample-dependent quantities are estimated from each training fold only:
citation-function weights, the scale-calibration coefficient beta, and the
F1-optimal decision threshold. The corresponding training-fold quantities are
then applied unchanged to the held-out fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import fisher_exact
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


REQUIRED_COLUMNS = [
    "idx",
    "journal",
    "year",
    "award",
    "title",
    "num_all",
    "num_d",
    "num_m",
    "num_s",
    "num_refs",
    "mean_similarity",
    "max_similarity",
    "min_similarity",
    "std_similarity",
    "m",
]

NUMERIC_COLUMNS = [
    "year",
    "award",
    "num_all",
    "num_d",
    "num_m",
    "num_s",
    "num_refs",
    "mean_similarity",
    "max_similarity",
    "min_similarity",
    "std_similarity",
    "m",
]

COUNT_COLUMNS = ["num_all", "num_d", "num_m", "num_s", "num_refs"]

BOOTSTRAP_METRICS = [
    "RD",
    "RR",
    "OR",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "Lift_at_20pct",
    "Recall_at_20pct",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Leakage-free five-fold training for the AI dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "AI_data.xlsx",
        help="Input XLSX file. Defaults to ../AI_data.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Output directory. Defaults to the script directory.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260612)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_excel(path)
    data.columns = [str(column).strip() for column in data.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise ValueError(f"Input validation failed: {name}. {details}")

    record_check(
        "required_values_complete",
        not data[REQUIRED_COLUMNS].isna().any().any(),
        f"missing_cells={int(data[REQUIRED_COLUMNS].isna().sum().sum())}",
    )
    record_check(
        "idx_unique",
        bool(data["idx"].is_unique),
        f"duplicate_idx={int(data['idx'].duplicated().sum())}",
    )
    labels = set(data["award"].astype(int).unique())
    record_check(
        "binary_award_labels",
        labels == {0, 1},
        f"labels={sorted(labels)}",
    )

    finite_numeric = np.isfinite(data[NUMERIC_COLUMNS].to_numpy(dtype=float)).all()
    record_check("numeric_values_finite", bool(finite_numeric), "all numeric inputs finite")

    integer_like = np.isclose(
        data[COUNT_COLUMNS].to_numpy(dtype=float),
        np.round(data[COUNT_COLUMNS].to_numpy(dtype=float)),
    ).all()
    nonnegative_counts = (data[COUNT_COLUMNS] >= 0).all().all()
    record_check(
        "counts_nonnegative_integers",
        bool(integer_like and nonnegative_counts),
        "num_all/num_d/num_m/num_s/num_refs checked",
    )
    record_check(
        "num_all_positive",
        bool((data["num_all"] > 0).all()),
        f"zero_or_negative={int((data['num_all'] <= 0).sum())}",
    )
    count_difference = (
        data["num_d"] + data["num_m"] + data["num_s"] - data["num_all"]
    )
    record_check(
        "citation_counts_reconcile",
        bool((count_difference == 0).all()),
        f"mismatched_rows={int((count_difference != 0).sum())}",
    )

    expected_m = data["max_similarity"] - data["min_similarity"]
    maximum_m_error = float((data["m"] - expected_m).abs().max())
    record_check(
        "m_matches_similarity_range",
        maximum_m_error <= 1e-8,
        f"max_abs_error={maximum_m_error:.12g}",
    )

    data["award"] = data["award"].astype(int)
    for column in COUNT_COLUMNS + ["year"]:
        data[column] = data[column].astype(int)

    # These are paper-level quantities and therefore safe to calculate before CV.
    data["P_d"] = data["num_d"] / data["num_all"]
    data["P_m"] = data["num_m"] / data["num_all"]
    data["P_s"] = data["num_s"] / data["num_all"]

    proportion_error = (data[["P_d", "P_m", "P_s"]].sum(axis=1) - 1.0).abs()
    max_proportion_error = float(proportion_error.max())
    record_check(
        "citation_proportions_sum_to_one",
        max_proportion_error <= 1e-12,
        f"max_abs_error={max_proportion_error:.12g}",
    )

    class_counts = data["award"].value_counts().sort_index()
    record_check(
        "enough_cases_for_five_folds",
        int(class_counts.min()) >= 5,
        f"class_counts={class_counts.to_dict()}",
    )

    validation = pd.DataFrame(checks)
    return data, validation


def calculate_citation_weights(train: pd.DataFrame) -> dict[str, float]:
    n_samples = len(train)
    if n_samples < 2:
        raise ValueError("At least two training samples are required for weights.")

    importance: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    for suffix in ("d", "m", "s"):
        values = train[f"P_{suffix}"].to_numpy(dtype=float)
        total = float(values.sum())
        if total <= 0:
            raise ValueError(f"Training-fold P_{suffix} has a non-positive sum.")
        q = values / total
        q_positive = q[q > 0]
        entropy = float(-np.sum(q_positive * np.log(q_positive)))
        scarcity = 1.0 - float(values.mean())
        concentration = 1.0 - entropy / math.log(n_samples)
        importance_value = (scarcity + concentration) / 2.0
        importance[suffix] = importance_value
        diagnostics[f"scarcity_{suffix}"] = scarcity
        diagnostics[f"concentration_{suffix}"] = concentration
        diagnostics[f"importance_{suffix}"] = importance_value

    denominator = sum(importance.values())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Citation-function importance values cannot be normalized.")

    weights = {f"weight_{suffix}": importance[suffix] / denominator for suffix in importance}
    weights.update(diagnostics)
    return weights


def calculate_relevance(data: pd.DataFrame, parameters: dict[str, float]) -> np.ndarray:
    return (
        parameters["weight_d"] * data["P_d"].to_numpy(dtype=float)
        + parameters["weight_m"] * data["P_m"].to_numpy(dtype=float)
        + parameters["weight_s"] * data["P_s"].to_numpy(dtype=float)
    )


def calculate_beta(train_m: np.ndarray, train_n: np.ndarray) -> tuple[float, float, float]:
    sigma_m_cubed = float(np.std(np.asarray(train_m, dtype=float) ** 3, ddof=1))
    sigma_n_squared = float(np.std(np.asarray(train_n, dtype=float) ** 2, ddof=1))
    if (
        not np.isfinite(sigma_m_cubed)
        or not np.isfinite(sigma_n_squared)
        or sigma_m_cubed <= 0
        or sigma_n_squared <= 0
    ):
        raise ValueError("Training-fold dispersion is invalid; beta cannot be calculated.")
    beta = math.sqrt((8.0 * sigma_m_cubed) / (27.0 * sigma_n_squared))
    return beta, sigma_m_cubed, sigma_n_squared


def calculate_discriminant(
    m: np.ndarray, n: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_std = -np.asarray(m, dtype=float)
    n_std = beta * np.asarray(n, dtype=float)
    delta = 8.0 * m_std**3 + 27.0 * n_std**2
    return m_std, n_std, delta


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=float))
    unique = unique[np.isfinite(unique)]
    if len(unique) == 0:
        raise ValueError("No finite discriminant values are available.")
    if len(unique) == 1:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    epsilon = max(float(np.ptp(unique)), 1.0) * 1e-9
    return np.concatenate(([unique[0] - epsilon], midpoints, [unique[-1] + epsilon]))


def learn_f1_optimal_threshold(delta: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    best = {
        "threshold": float(np.median(delta)),
        "F1": -math.inf,
        "Precision": -math.inf,
        "Recall": -math.inf,
        "Accuracy": -math.inf,
    }
    for threshold in candidate_thresholds(delta):
        predicted = (delta <= threshold).astype(int)
        precision = float(precision_score(labels, predicted, zero_division=0))
        recall = float(recall_score(labels, predicted, zero_division=0))
        f1 = float(f1_score(labels, predicted, zero_division=0))
        accuracy = float(accuracy_score(labels, predicted))
        key = (f1, recall, precision, -float(threshold))
        best_key = (best["F1"], best["Recall"], best["Precision"], -best["threshold"])
        if key > best_key:
            best = {
                "threshold": float(threshold),
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
            }
    return best


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.inf
    return numerator / denominator


def evaluate_predictions(
    labels: np.ndarray,
    predicted: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    score = np.asarray(score, dtype=float)

    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))

    selected = predicted == 1
    unselected = predicted == 0
    selected_rate = float(labels[selected].mean()) if selected.any() else math.nan
    unselected_rate = float(labels[unselected].mean()) if unselected.any() else math.nan
    rd = selected_rate - unselected_rate
    rr = safe_ratio(selected_rate, unselected_rate)
    odds_ratio, fisher_p = fisher_exact([[tp, fp], [fn, tn]])

    top_n = max(1, int(math.ceil(0.20 * len(labels))))
    top_indices = np.argsort(-score, kind="stable")[:top_n]
    top_awards = int(labels[top_indices].sum())
    overall_rate = float(labels.mean())
    top_rate = float(labels[top_indices].mean())

    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, score))
        pr_auc = float(average_precision_score(labels, score))
    else:
        roc_auc = math.nan
        pr_auc = math.nan

    return {
        "N": int(len(labels)),
        "award_n": int(labels.sum()),
        "control_n": int((labels == 0).sum()),
        "award_rate": overall_rate,
        "predicted_breakthrough_n": int(selected.sum()),
        "predicted_control_n": int(unselected.sum()),
        "award_rate_delta_below_threshold": selected_rate,
        "award_rate_delta_above_threshold": unselected_rate,
        "RD": float(rd),
        "RR": float(rr),
        "OR": float(odds_ratio),
        "Fisher_p": float(fisher_p),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Lift_at_20pct": float(safe_ratio(top_rate, overall_rate)),
        "Recall_at_20pct": float(safe_ratio(top_awards, int(labels.sum()))),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def run_fivefold_training(
    data: pd.DataFrame,
    folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = data["award"].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    n_rows = len(data)
    oof_fold = np.zeros(n_rows, dtype=int)
    oof_prediction = np.zeros(n_rows, dtype=int)
    oof_score = np.full(n_rows, np.nan, dtype=float)
    oof_n = np.full(n_rows, np.nan, dtype=float)
    oof_m_std = np.full(n_rows, np.nan, dtype=float)
    oof_n_std = np.full(n_rows, np.nan, dtype=float)
    oof_delta = np.full(n_rows, np.nan, dtype=float)
    oof_threshold = np.full(n_rows, np.nan, dtype=float)
    oof_beta = np.full(n_rows, np.nan, dtype=float)
    oof_weight_d = np.full(n_rows, np.nan, dtype=float)
    oof_weight_m = np.full(n_rows, np.nan, dtype=float)
    oof_weight_s = np.full(n_rows, np.nan, dtype=float)

    fold_parameter_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, labels), start=1):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        parameters = calculate_citation_weights(train)
        train_n = calculate_relevance(train, parameters)
        test_n = calculate_relevance(test, parameters)

        beta, sigma_m_cubed, sigma_n_squared = calculate_beta(
            train["m"].to_numpy(dtype=float), train_n
        )
        train_m_std, train_n_std, train_delta = calculate_discriminant(
            train["m"].to_numpy(dtype=float), train_n, beta
        )
        test_m_std, test_n_std, test_delta = calculate_discriminant(
            test["m"].to_numpy(dtype=float), test_n, beta
        )

        learned = learn_f1_optimal_threshold(train_delta, train_labels)
        threshold = learned["threshold"]
        test_prediction = (test_delta <= threshold).astype(int)
        test_score = threshold - test_delta
        test_metrics = evaluate_predictions(test_labels, test_prediction, test_score)

        oof_fold[test_idx] = fold
        oof_prediction[test_idx] = test_prediction
        oof_score[test_idx] = test_score
        oof_n[test_idx] = test_n
        oof_m_std[test_idx] = test_m_std
        oof_n_std[test_idx] = test_n_std
        oof_delta[test_idx] = test_delta
        oof_threshold[test_idx] = threshold
        oof_beta[test_idx] = beta
        oof_weight_d[test_idx] = parameters["weight_d"]
        oof_weight_m[test_idx] = parameters["weight_m"]
        oof_weight_s[test_idx] = parameters["weight_s"]

        fold_parameter_rows.append(
            {
                "field": "AI",
                "fold": fold,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_award_n": int(train_labels.sum()),
                "train_control_n": int((train_labels == 0).sum()),
                "test_award_n": int(test_labels.sum()),
                "test_control_n": int((test_labels == 0).sum()),
                **parameters,
                "weights_sum": float(
                    parameters["weight_d"]
                    + parameters["weight_m"]
                    + parameters["weight_s"]
                ),
                "weight_order_d_gt_m_gt_s": bool(
                    parameters["weight_d"]
                    > parameters["weight_m"]
                    > parameters["weight_s"]
                ),
                "sigma_train_m_cubed": sigma_m_cubed,
                "sigma_train_n_squared": sigma_n_squared,
                "beta": beta,
                "threshold": threshold,
                "train_accuracy_at_threshold": learned["Accuracy"],
                "train_precision_at_threshold": learned["Precision"],
                "train_recall_at_threshold": learned["Recall"],
                "train_f1_at_threshold": learned["F1"],
            }
        )
        fold_metric_rows.append(
            {
                "field": "AI",
                "fold": fold,
                "threshold": threshold,
                **test_metrics,
            }
        )

        # Explicitly retain these arrays to make the train/test calculation path clear.
        _ = train_m_std, train_n_std

    if not (
        np.all(oof_fold > 0)
        and np.isfinite(oof_score).all()
        and np.isfinite(oof_delta).all()
    ):
        raise RuntimeError("OOF predictions are incomplete.")

    oof = data.copy()
    oof["fold"] = oof_fold
    oof["P_d"] = data["P_d"]
    oof["P_m"] = data["P_m"]
    oof["P_s"] = data["P_s"]
    oof["weight_d_used"] = oof_weight_d
    oof["weight_m_used"] = oof_weight_m
    oof["weight_s_used"] = oof_weight_s
    oof["n"] = oof_n
    oof["beta_used"] = oof_beta
    oof["m_std"] = oof_m_std
    oof["n_std"] = oof_n_std
    oof["delta"] = oof_delta
    oof["threshold_used"] = oof_threshold
    oof["prediction"] = oof_prediction
    oof["breakthrough_score"] = oof_score
    oof = oof.sort_values("idx").reset_index(drop=True)

    fold_parameters = pd.DataFrame(fold_parameter_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    return fold_parameters, fold_metrics, oof


def stratified_bootstrap_confidence_intervals(
    oof: pd.DataFrame,
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    labels = oof["award"].to_numpy(dtype=int)
    predicted = oof["prediction"].to_numpy(dtype=int)
    score = oof["breakthrough_score"].to_numpy(dtype=float)
    point = evaluate_predictions(labels, predicted, score)

    if repetitions <= 0:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower_95": math.nan,
                    "ci_upper_95": math.nan,
                    "valid_bootstrap_repetitions": 0,
                }
                for metric in BOOTSTRAP_METRICS
            ]
        )

    rng = np.random.default_rng(random_state)
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(repetitions):
        bootstrap_idx = np.concatenate(
            [
                rng.choice(class_zero, size=len(class_zero), replace=True),
                rng.choice(class_one, size=len(class_one), replace=True),
            ]
        )
        rng.shuffle(bootstrap_idx)
        result = evaluate_predictions(
            labels[bootstrap_idx], predicted[bootstrap_idx], score[bootstrap_idx]
        )
        for metric in BOOTSTRAP_METRICS:
            value = float(result[metric])
            if np.isfinite(value):
                samples[metric].append(value)

    rows = []
    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(samples[metric], dtype=float)
        lower, upper = (
            np.quantile(values, [0.025, 0.975]) if len(values) else (math.nan, math.nan)
        )
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "valid_bootstrap_repetitions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def summarize_fold_parameters(fold_parameters: pd.DataFrame) -> pd.DataFrame:
    variables = ["weight_d", "weight_m", "weight_s", "beta", "threshold"]
    rows = []
    for variable in variables:
        values = fold_parameters[variable].astype(float)
        rows.append(
            {
                "field": "AI",
                "parameter": variable,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def make_table5(fold_parameters: pd.DataFrame) -> pd.DataFrame:
    thresholds = fold_parameters.sort_values("fold")["threshold"].to_numpy(dtype=float)
    row: dict[str, object] = {"Field": "AI"}
    for index, value in enumerate(thresholds, start=1):
        row[f"Fold {index}"] = float(value)
    row["Mean"] = float(np.mean(thresholds))
    row["SD"] = float(np.std(thresholds, ddof=1))
    return pd.DataFrame([row])


def make_table6(
    overall_metrics: dict[str, float | int],
    confidence_intervals: pd.DataFrame,
) -> pd.DataFrame:
    ci_lookup = confidence_intervals.set_index("metric")
    rows = []
    specification = [
        ("Discriminative strength", "RD", "RD"),
        ("Discriminative strength", "RR", "RR"),
        ("Discriminative strength", "OR", "OR"),
        ("Classification performance", "Accuracy", "Accuracy"),
        ("Classification performance", "Precision", "Precision"),
        ("Classification performance", "Recall", "Recall"),
        ("Classification performance", "F1-score", "F1"),
        ("Threshold-independent performance", "ROC-AUC", "ROC_AUC"),
        ("Threshold-independent performance", "PR-AUC", "PR_AUC"),
        ("Ranking capability", "Lift@20%", "Lift_at_20pct"),
        ("Ranking capability", "Recall@20%", "Recall_at_20pct"),
    ]
    for category, display_name, key in specification:
        rows.append(
            {
                "Category": category,
                "Metric": display_name,
                "AI": overall_metrics[key],
                "CI_lower_95": ci_lookup.loc[key, "ci_lower_95"],
                "CI_upper_95": ci_lookup.loc[key, "ci_upper_95"],
            }
        )
    return pd.DataFrame(rows)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, validation = load_and_validate_data(input_path)
    fold_parameters, fold_metrics, oof = run_fivefold_training(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    overall_metrics = evaluate_predictions(
        oof["award"].to_numpy(dtype=int),
        oof["prediction"].to_numpy(dtype=int),
        oof["breakthrough_score"].to_numpy(dtype=float),
    )
    overall = pd.DataFrame([{"field": "AI", "evaluation": "OOF pooled", **overall_metrics}])
    confidence_intervals = stratified_bootstrap_confidence_intervals(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 1,
    )
    parameter_summary = summarize_fold_parameters(fold_parameters)
    table5 = make_table5(fold_parameters)
    table6 = make_table6(overall_metrics, confidence_intervals)

    outputs = {
        "input_validation.csv": validation,
        "fold_parameters.csv": fold_parameters,
        "fold_test_metrics.csv": fold_metrics,
        "fold_parameter_summary.csv": parameter_summary,
        "oof_predictions.csv": oof,
        "overall_oof_metrics.csv": overall,
        "metric_confidence_intervals.csv": confidence_intervals,
        "table5_ai_thresholds.csv": table5,
        "table6_ai_metrics.csv": table6,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_directory": str(output_dir),
        "field": "AI",
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "proportion_definitions": {
            "P_d": "num_d / num_all",
            "P_m": "num_m / num_all",
            "P_s": "num_s / num_all",
        },
        "fold_only_parameters": [
            "weight_d",
            "weight_m",
            "weight_s",
            "beta",
            "F1-optimal threshold",
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "output_files": list(outputs),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("AI five-fold training completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(table5.to_string(index=False))
    print(table6.to_string(index=False))


if __name__ == "__main__":
    main()
