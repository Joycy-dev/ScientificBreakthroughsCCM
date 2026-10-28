"""Fair five-fold comparison of four reviewer-requested baselines and the cusp model.

Models:
1. baseline_1_heterogeneity_only: unpenalized logistic regression using m
2. baseline_2_relevance_only: unpenalized logistic regression using n
3. baseline_3_reference_count_only: unpenalized logistic regression using
   log(1 + num_refs)
4. baseline_4_simple_logistic_m_n: unpenalized logistic regression using m + n
5. proposed_cusp: Delta = -8*m^3 + 27*beta^2*n^2

All models use the same outer folds. Citation-function weights, beta, feature
standardization, model coefficients, and F1-optimal empirical thresholds are
estimated from each training fold only and applied unchanged to its held-out
fold. Reference count is log-transformed before training-fold standardization,
matching the earlier reference-count robustness analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent
BASE_TRAINING_SCRIPT = AI_DIR / "AI_train" / "ai_fivefold_training.py"
MAIN_OOF_PATH = AI_DIR / "AI_train" / "oof_predictions.csv"
REFERENCE_COUNT_OOF_PATH = (
    AI_DIR / "AI_robustness" / "reference_count_oof_predictions.csv"
)

LOGISTIC_MODELS = {
    "baseline_1_heterogeneity_only": ["m"],
    "baseline_2_relevance_only": ["n"],
    "baseline_3_reference_count_only": ["log_num_refs"],
    "baseline_4_simple_logistic_m_n": ["m", "n"],
}
BASELINE_MODELS = list(LOGISTIC_MODELS)
PROPOSED_MODEL = "proposed_cusp"
ALL_MODELS = [*BASELINE_MODELS, PROPOSED_MODEL]
CORE_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "Lift_at_20pct",
    "Recall_at_20pct",
]


def load_base_training_module():
    spec = importlib.util.spec_from_file_location(
        "ai_fivefold_training_base_for_comparison", BASE_TRAINING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_training_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four requested baselines with the AI-domain cusp model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=AI_DIR / "AI_data.xlsx",
        help="Input XLSX file. Defaults to ../AI_data.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory. Defaults to this script directory.",
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


def learn_high_score_f1_threshold(
    score: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Learn score >= threshold using cumulative confusion counts."""
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    total_positive = int(sorted_labels.sum())
    total_n = len(sorted_labels)
    group_ends = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    epsilon = max(float(np.ptp(sorted_score)), 1.0) * 1e-9

    best = {
        "score_threshold": float(sorted_score[0] + epsilon),
        "F1": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "Accuracy": float((total_n - total_positive) / total_n),
    }
    for end in group_ends:
        predicted_positive = int(end + 1)
        tp = int(cumulative_tp[end])
        fp = predicted_positive - tp
        fn = total_positive - tp
        tn = total_n - tp - fp - fn
        precision = float(tp / predicted_positive) if predicted_positive else 0.0
        recall = float(tp / total_positive) if total_positive else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall > 0
            else 0.0
        )
        accuracy = float((tp + tn) / total_n)
        threshold = (
            float((sorted_score[end] + sorted_score[end + 1]) / 2.0)
            if end + 1 < total_n
            else float(sorted_score[end] - epsilon)
        )
        key = (f1, recall, precision, threshold)
        best_key = (
            best["F1"],
            best["Recall"],
            best["Precision"],
            best["score_threshold"],
        )
        if key > best_key:
            best = {
                "score_threshold": threshold,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
            }
    return best


def run_comparison(
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
    oof_n = np.full(n_rows, np.nan, dtype=float)
    oof_probability = {
        model: np.full(n_rows, np.nan, dtype=float) for model in LOGISTIC_MODELS
    }
    oof_prediction = {
        model: np.zeros(n_rows, dtype=int) for model in LOGISTIC_MODELS
    }
    oof_threshold = {
        model: np.full(n_rows, np.nan, dtype=float) for model in LOGISTIC_MODELS
    }
    oof_cusp_delta = np.full(n_rows, np.nan, dtype=float)
    oof_cusp_score = np.full(n_rows, np.nan, dtype=float)
    oof_cusp_prediction = np.zeros(n_rows, dtype=int)
    oof_cusp_threshold = np.full(n_rows, np.nan, dtype=float)
    oof_cusp_beta = np.full(n_rows, np.nan, dtype=float)

    parameter_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data, labels), start=1
    ):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        citation_parameters = base.calculate_citation_weights(train)
        train_n = base.calculate_relevance(train, citation_parameters)
        test_n = base.calculate_relevance(test, citation_parameters)
        train_features = pd.DataFrame(
            {
                "m": train["m"].to_numpy(dtype=float),
                "n": train_n,
                "log_num_refs": np.log1p(
                    train["num_refs"].to_numpy(dtype=float)
                ),
            },
            index=train.index,
        )
        test_features = pd.DataFrame(
            {
                "m": test["m"].to_numpy(dtype=float),
                "n": test_n,
                "log_num_refs": np.log1p(
                    test["num_refs"].to_numpy(dtype=float)
                ),
            },
            index=test.index,
        )

        scaler = StandardScaler()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index,
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_features),
            columns=test_features.columns,
            index=test_features.index,
        )

        oof_fold[test_idx] = fold
        oof_n[test_idx] = test_n

        for model_name, feature_names in LOGISTIC_MODELS.items():
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=10000,
                random_state=random_state,
            )
            model.fit(train_scaled[feature_names], train_labels)
            train_probability = model.predict_proba(train_scaled[feature_names])[:, 1]
            test_probability = model.predict_proba(test_scaled[feature_names])[:, 1]
            learned = learn_high_score_f1_threshold(train_probability, train_labels)
            threshold = learned["score_threshold"]
            test_prediction = (test_probability >= threshold).astype(int)
            test_metrics = base.evaluate_predictions(
                test_labels, test_prediction, test_probability
            )

            oof_probability[model_name][test_idx] = test_probability
            oof_prediction[model_name][test_idx] = test_prediction
            oof_threshold[model_name][test_idx] = threshold

            coefficients = dict(zip(feature_names, model.coef_[0], strict=True))
            parameter_rows.append(
                {
                    "field": "AI",
                    "fold": fold,
                    "model": model_name,
                    "features": "+".join(feature_names),
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "weight_d": citation_parameters["weight_d"],
                    "weight_m": citation_parameters["weight_m"],
                    "weight_s": citation_parameters["weight_s"],
                    "scaler_mean_m": float(scaler.mean_[0]),
                    "scaler_sd_m": float(scaler.scale_[0]),
                    "scaler_mean_n": float(scaler.mean_[1]),
                    "scaler_sd_n": float(scaler.scale_[1]),
                    "scaler_mean_log_num_refs": float(scaler.mean_[2]),
                    "scaler_sd_log_num_refs": float(scaler.scale_[2]),
                    "intercept": float(model.intercept_[0]),
                    "coefficient_m": float(coefficients.get("m", math.nan)),
                    "coefficient_n": float(coefficients.get("n", math.nan)),
                    "coefficient_log_num_refs": float(
                        coefficients.get("log_num_refs", math.nan)
                    ),
                    "beta": math.nan,
                    "empirical_threshold": threshold,
                    "threshold_scale": "probability",
                    "train_accuracy_at_threshold": learned["Accuracy"],
                    "train_precision_at_threshold": learned["Precision"],
                    "train_recall_at_threshold": learned["Recall"],
                    "train_f1_at_threshold": learned["F1"],
                }
            )
            metric_rows.append(
                {
                    "field": "AI",
                    "fold": fold,
                    "model": model_name,
                    "features": "+".join(feature_names),
                    "empirical_threshold": threshold,
                    "threshold_scale": "probability",
                    **test_metrics,
                }
            )

        train_m = train["m"].to_numpy(dtype=float)
        test_m = test["m"].to_numpy(dtype=float)
        beta, sigma_m_cubed, sigma_n_squared = base.calculate_beta(train_m, train_n)
        _, _, train_delta = base.calculate_discriminant(train_m, train_n, beta)
        _, _, test_delta = base.calculate_discriminant(test_m, test_n, beta)
        learned_cusp = learn_high_score_f1_threshold(-train_delta, train_labels)
        cusp_score_threshold = learned_cusp["score_threshold"]
        cusp_delta_threshold = -cusp_score_threshold
        cusp_prediction = ((-test_delta) >= cusp_score_threshold).astype(int)
        cusp_score = cusp_delta_threshold - test_delta
        cusp_metrics = base.evaluate_predictions(
            test_labels, cusp_prediction, cusp_score
        )

        oof_cusp_delta[test_idx] = test_delta
        oof_cusp_score[test_idx] = cusp_score
        oof_cusp_prediction[test_idx] = cusp_prediction
        oof_cusp_threshold[test_idx] = cusp_delta_threshold
        oof_cusp_beta[test_idx] = beta

        parameter_rows.append(
            {
                "field": "AI",
                "fold": fold,
                "model": PROPOSED_MODEL,
                "features": "m+n",
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "weight_d": citation_parameters["weight_d"],
                "weight_m": citation_parameters["weight_m"],
                "weight_s": citation_parameters["weight_s"],
                "scaler_mean_m": math.nan,
                "scaler_sd_m": math.nan,
                "scaler_mean_n": math.nan,
                "scaler_sd_n": math.nan,
                "scaler_mean_log_num_refs": math.nan,
                "scaler_sd_log_num_refs": math.nan,
                "intercept": math.nan,
                "coefficient_m": math.nan,
                "coefficient_n": math.nan,
                "coefficient_log_num_refs": math.nan,
                "beta": beta,
                "sigma_train_m_cubed": sigma_m_cubed,
                "sigma_train_n_squared": sigma_n_squared,
                "empirical_threshold": cusp_delta_threshold,
                "threshold_scale": "Delta",
                "train_accuracy_at_threshold": learned_cusp["Accuracy"],
                "train_precision_at_threshold": learned_cusp["Precision"],
                "train_recall_at_threshold": learned_cusp["Recall"],
                "train_f1_at_threshold": learned_cusp["F1"],
            }
        )
        metric_rows.append(
            {
                "field": "AI",
                "fold": fold,
                "model": PROPOSED_MODEL,
                "features": "m+n",
                "empirical_threshold": cusp_delta_threshold,
                "threshold_scale": "Delta",
                **cusp_metrics,
            }
        )

    if not (np.all(oof_fold > 0) and np.isfinite(oof_n).all()):
        raise RuntimeError("OOF fold assignments or relevance values are incomplete.")
    for model_name in LOGISTIC_MODELS:
        if not np.isfinite(oof_probability[model_name]).all():
            raise RuntimeError(f"Incomplete probabilities for {model_name}.")
    if not (
        np.isfinite(oof_cusp_delta).all()
        and np.isfinite(oof_cusp_score).all()
        and np.isfinite(oof_cusp_beta).all()
    ):
        raise RuntimeError("Incomplete cusp OOF results.")

    oof = data[["idx", "award", "num_refs", "m"]].copy()
    oof["fold"] = oof_fold
    oof["n"] = oof_n
    oof["log_num_refs"] = np.log1p(oof["num_refs"].to_numpy(dtype=float))
    for model_name in LOGISTIC_MODELS:
        oof[f"score_{model_name}"] = oof_probability[model_name]
        oof[f"prediction_{model_name}"] = oof_prediction[model_name]
        oof[f"threshold_{model_name}"] = oof_threshold[model_name]
    oof[f"delta_{PROPOSED_MODEL}"] = oof_cusp_delta
    oof[f"score_{PROPOSED_MODEL}"] = oof_cusp_score
    oof[f"prediction_{PROPOSED_MODEL}"] = oof_cusp_prediction
    oof[f"threshold_{PROPOSED_MODEL}"] = oof_cusp_threshold
    oof[f"beta_{PROPOSED_MODEL}"] = oof_cusp_beta
    oof = oof.sort_values("idx").reset_index(drop=True)

    return pd.DataFrame(parameter_rows), pd.DataFrame(metric_rows), oof


def overall_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    labels = oof["award"].to_numpy(dtype=int)
    rows = []
    for model_name in ALL_MODELS:
        result = base.evaluate_predictions(
            labels,
            oof[f"prediction_{model_name}"].to_numpy(dtype=int),
            oof[f"score_{model_name}"].to_numpy(dtype=float),
        )
        features = (
            "+".join(LOGISTIC_MODELS[model_name])
            if model_name in LOGISTIC_MODELS
            else "m+n"
        )
        rows.append(
            {
                "field": "AI",
                "evaluation": "OOF pooled",
                "model": model_name,
                "features": features,
                **result,
            }
        )
    return pd.DataFrame(rows)


def make_display_table(overall: pd.DataFrame) -> pd.DataFrame:
    display_names = {
        "baseline_1_heterogeneity_only": "B1: Heterogeneity only (m)",
        "baseline_2_relevance_only": "B2: Relevance only (n)",
        "baseline_3_reference_count_only": (
            "B3: Reference count only [log(1+num_refs)]"
        ),
        "baseline_4_simple_logistic_m_n": "B4: Simple logistic (m+n)",
        PROPOSED_MODEL: "Proposed cusp model (m,n)",
    }
    rows = []
    for _, row in overall.iterrows():
        rows.append(
            {
                "Model": display_names[row["model"]],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "ROC_AUC": row["ROC_AUC"],
                "PR_AUC": row["PR_AUC"],
                "Lift_at_20pct": row["Lift_at_20pct"],
                "Recall_at_20pct": row["Recall_at_20pct"],
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_cusp_minus_each_baseline(
    oof: pd.DataFrame,
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    labels = oof["award"].to_numpy(dtype=int)
    cusp_prediction = oof[f"prediction_{PROPOSED_MODEL}"].to_numpy(dtype=int)
    cusp_score = oof[f"score_{PROPOSED_MODEL}"].to_numpy(dtype=float)
    point_cusp = base.evaluate_predictions(labels, cusp_prediction, cusp_score)
    baseline_arrays = {
        model_name: (
            oof[f"prediction_{model_name}"].to_numpy(dtype=int),
            oof[f"score_{model_name}"].to_numpy(dtype=float),
        )
        for model_name in BASELINE_MODELS
    }
    point_baselines = {
        model_name: base.evaluate_predictions(labels, prediction, score)
        for model_name, (prediction, score) in baseline_arrays.items()
    }
    differences: dict[str, dict[str, list[float]]] = {
        model_name: {metric: [] for metric in CORE_METRICS}
        for model_name in BASELINE_MODELS
    }

    rng = np.random.default_rng(random_state)
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    for _ in range(repetitions):
        sample_idx = np.concatenate(
            [
                rng.choice(class_zero, size=len(class_zero), replace=True),
                rng.choice(class_one, size=len(class_one), replace=True),
            ]
        )
        rng.shuffle(sample_idx)
        sample_cusp = base.evaluate_predictions(
            labels[sample_idx],
            cusp_prediction[sample_idx],
            cusp_score[sample_idx],
        )
        for model_name, (baseline_prediction, baseline_score) in baseline_arrays.items():
            sample_baseline = base.evaluate_predictions(
                labels[sample_idx],
                baseline_prediction[sample_idx],
                baseline_score[sample_idx],
            )
            for metric in CORE_METRICS:
                difference = float(sample_cusp[metric]) - float(
                    sample_baseline[metric]
                )
                if np.isfinite(difference):
                    differences[model_name][metric].append(difference)

    rows = []
    for model_name in BASELINE_MODELS:
        for metric in CORE_METRICS:
            values = np.asarray(differences[model_name][metric], dtype=float)
            lower, upper = (
                np.quantile(values, [0.025, 0.975])
                if len(values)
                else (math.nan, math.nan)
            )
            rows.append(
                {
                    "baseline_model": model_name,
                    "metric": metric,
                    "baseline_value": float(point_baselines[model_name][metric]),
                    "cusp_value": float(point_cusp[metric]),
                    "cusp_minus_baseline": float(point_cusp[metric])
                    - float(point_baselines[model_name][metric]),
                    "difference_ci_lower_95": float(lower),
                    "difference_ci_upper_95": float(upper),
                    "valid_bootstrap_repetitions": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def design_checks(oof: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Model-comparison design check failed: {name}. {details}")

    add(
        "all_oof_rows_present",
        bool(oof["idx"].is_unique and len(oof) > 0),
        f"rows={len(oof)}, unique_idx={oof['idx'].nunique()}",
    )
    score_columns = [f"score_{model}" for model in ALL_MODELS]
    add(
        "all_model_scores_finite",
        bool(np.isfinite(oof[score_columns].to_numpy(dtype=float)).all()),
        "all five OOF score columns checked",
    )

    if MAIN_OOF_PATH.exists():
        main_oof = pd.read_csv(MAIN_OOF_PATH).sort_values("idx").reset_index(drop=True)
        same_idx = np.array_equal(oof["idx"].to_numpy(), main_oof["idx"].to_numpy())
        same_fold = same_idx and np.array_equal(
            oof["fold"].to_numpy(dtype=int), main_oof["fold"].to_numpy(dtype=int)
        )
        max_n_difference = (
            float(
                np.max(
                    np.abs(
                        oof["n"].to_numpy(dtype=float)
                        - main_oof["n"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        max_delta_difference = (
            float(
                np.max(
                    np.abs(
                        oof[f"delta_{PROPOSED_MODEL}"].to_numpy(dtype=float)
                        - main_oof["delta"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        max_score_difference = (
            float(
                np.max(
                    np.abs(
                        oof[f"score_{PROPOSED_MODEL}"].to_numpy(dtype=float)
                        - main_oof["breakthrough_score"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        same_prediction = same_idx and np.array_equal(
            oof[f"prediction_{PROPOSED_MODEL}"].to_numpy(dtype=int),
            main_oof["prediction"].to_numpy(dtype=int),
        )
        add("same_fold_as_main_model", bool(same_fold), "compared by idx")
        add(
            "same_relevance_as_main_model",
            max_n_difference <= 1e-12,
            f"max_abs_difference={max_n_difference:.12g}",
        )
        add(
            "cusp_delta_reproduces_main_model",
            max_delta_difference <= 1e-12,
            f"max_abs_difference={max_delta_difference:.12g}",
        )
        add(
            "cusp_score_reproduces_main_model",
            max_score_difference <= 1e-12,
            f"max_abs_difference={max_score_difference:.12g}",
        )
        add(
            "cusp_predictions_reproduce_main_model",
            bool(same_prediction),
            "OOF predictions compared by idx",
        )

    if REFERENCE_COUNT_OOF_PATH.exists():
        prior_count = (
            pd.read_csv(REFERENCE_COUNT_OOF_PATH)
            .sort_values("idx")
            .reset_index(drop=True)
        )
        same_idx = np.array_equal(
            oof["idx"].to_numpy(dtype=int),
            prior_count["idx"].to_numpy(dtype=int),
        )
        count_model = "baseline_3_reference_count_only"
        max_probability_difference = (
            float(
                np.max(
                    np.abs(
                        oof[f"score_{count_model}"].to_numpy(dtype=float)
                        - prior_count["probability_M0_count"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        max_threshold_difference = (
            float(
                np.max(
                    np.abs(
                        oof[f"threshold_{count_model}"].to_numpy(dtype=float)
                        - prior_count["threshold_M0_count"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        same_prediction = same_idx and np.array_equal(
            oof[f"prediction_{count_model}"].to_numpy(dtype=int),
            prior_count["prediction_M0_count"].to_numpy(dtype=int),
        )
        add(
            "reference_count_probability_reproduces_prior_M0",
            max_probability_difference <= 1e-12,
            f"max_abs_difference={max_probability_difference:.12g}",
        )
        add(
            "reference_count_threshold_reproduces_prior_M0",
            max_threshold_difference <= 1e-12,
            f"max_abs_difference={max_threshold_difference:.12g}",
        )
        add(
            "reference_count_predictions_reproduce_prior_M0",
            bool(same_prediction),
            "OOF predictions compared by idx",
        )

    return pd.DataFrame(checks)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, validation = base.load_and_validate_data(input_path)
    fold_parameters, fold_metrics, oof = run_comparison(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    checks = design_checks(oof)
    overall = overall_metrics(oof)
    display_table = make_display_table(overall)
    paired_difference = paired_bootstrap_cusp_minus_each_baseline(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 30,
    )
    baseline_overall = overall[overall["model"].isin(BASELINE_MODELS)].copy()

    outputs = {
        "four_baselines_input_validation.csv": validation,
        "four_baselines_vs_cusp_design_checks.csv": checks,
        "four_baselines_fold_parameters.csv": fold_parameters[
            fold_parameters["model"].isin(BASELINE_MODELS)
        ].copy(),
        "four_baselines_fold_test_metrics.csv": fold_metrics[
            fold_metrics["model"].isin(BASELINE_MODELS)
        ].copy(),
        "four_baselines_overall_metrics.csv": baseline_overall,
        "five_models_fold_parameters.csv": fold_parameters,
        "five_models_fold_test_metrics.csv": fold_metrics,
        "five_models_oof_predictions.csv": oof,
        "five_models_overall_comparison.csv": overall,
        "five_models_display_table.csv": display_table,
        "cusp_vs_each_of_four_baselines_paired_bootstrap.csv": paired_difference,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "base_training_script": str(BASE_TRAINING_SCRIPT),
        "base_training_script_sha256": file_sha256(BASE_TRAINING_SCRIPT),
        "main_oof_file": str(MAIN_OOF_PATH),
        "prior_reference_count_oof_file": str(REFERENCE_COUNT_OOF_PATH),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "models": {
            "baseline_1_heterogeneity_only": (
                "unpenalized logistic regression using m"
            ),
            "baseline_2_relevance_only": (
                "unpenalized logistic regression using n"
            ),
            "baseline_3_reference_count_only": (
                "unpenalized logistic regression using log(1 + num_refs)"
            ),
            "baseline_4_simple_logistic_m_n": (
                "unpenalized logistic regression using m + n"
            ),
            PROPOSED_MODEL: "Delta = -8*m^3 + 27*beta^2*n^2",
        },
        "feature_scaling": (
            "m, n, and log_num_refs standardized using training-fold mean and "
            "standard deviation for logistic models only; training-fold parameters "
            "applied unchanged to the held-out fold."
        ),
        "threshold_selection": (
            "F1-optimal empirical threshold learned independently in each training "
            "fold for every model."
        ),
        "reference_count_transform": "log(1 + num_refs)",
        "reference_count_used_only_in": "baseline_3_reference_count_only",
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
    with (output_dir / "four_baselines_vs_cusp_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("AI four-baseline versus cusp comparison completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF model comparison:")
    print(display_table.to_string(index=False))
    print("\nPaired bootstrap: proposed cusp minus each baseline")
    print(paired_difference.to_string(index=False))


if __name__ == "__main__":
    main()
