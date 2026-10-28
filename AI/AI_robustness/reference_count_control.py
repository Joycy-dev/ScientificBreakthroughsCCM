"""Five-fold auxiliary analysis controlling for reference count.

This script does not modify the cusp model. It fits three nested logistic
regression models on the same folds used by the main AI-domain experiment:

M0: award ~ log(1 + num_refs)
M1: award ~ n + log(1 + num_refs)
M2: award ~ n + m + log(1 + num_refs)

Citation-function weights and all standardization parameters are estimated
from each training fold only. F1-optimal probability thresholds are also
learned on the training fold and applied unchanged to the held-out fold.
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
import statsmodels.api as sm
import statsmodels
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent
BASE_TRAINING_SCRIPT = AI_DIR / "AI_train" / "ai_fivefold_training.py"
MAIN_OOF_PATH = AI_DIR / "AI_train" / "oof_predictions.csv"

MODEL_FEATURES = {
    "M0_count": ["log_num_refs"],
    "M1_relevance_count": ["n", "log_num_refs"],
    "M2_relevance_heterogeneity_count": ["n", "m", "log_num_refs"],
}

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
        "ai_fivefold_training_base_for_count_control", BASE_TRAINING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_training_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-fold reference-count control analysis for the AI dataset."
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


def probability_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(probability, dtype=float))
    unique = unique[np.isfinite(unique)]
    if len(unique) == 0:
        raise ValueError("No finite training probabilities are available.")
    if len(unique) == 1:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    epsilon = max(float(np.ptp(unique)), 1.0) * 1e-9
    return np.concatenate(([unique[0] - epsilon], midpoints, [unique[-1] + epsilon]))


def learn_f1_probability_threshold(
    probability: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    total_positive = int(sorted_labels.sum())
    total_n = len(sorted_labels)
    group_ends = np.flatnonzero(
        np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    )
    epsilon = max(float(np.ptp(sorted_probability)), 1.0) * 1e-9

    best = {
        "threshold": float(sorted_probability[0] + epsilon),
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
            float((sorted_probability[end] + sorted_probability[end + 1]) / 2.0)
            if end + 1 < total_n
            else float(sorted_probability[end] - epsilon)
        )
        key = (f1, recall, precision, float(threshold))
        best_key = (
            best["F1"],
            best["Recall"],
            best["Precision"],
            best["threshold"],
        )
        if key > best_key:
            best = {
                "threshold": float(threshold),
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
            }
    return best


def run_fivefold_control_models(
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
    oof_log_num_refs = np.log1p(data["num_refs"].to_numpy(dtype=float))
    oof_probability = {
        model: np.full(n_rows, np.nan, dtype=float) for model in MODEL_FEATURES
    }
    oof_prediction = {
        model: np.zeros(n_rows, dtype=int) for model in MODEL_FEATURES
    }
    oof_threshold = {
        model: np.full(n_rows, np.nan, dtype=float) for model in MODEL_FEATURES
    }

    fold_parameter_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []

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
                "n": train_n,
                "m": train["m"].to_numpy(dtype=float),
                "log_num_refs": np.log1p(train["num_refs"].to_numpy(dtype=float)),
            },
            index=train.index,
        )
        test_features = pd.DataFrame(
            {
                "n": test_n,
                "m": test["m"].to_numpy(dtype=float),
                "log_num_refs": np.log1p(test["num_refs"].to_numpy(dtype=float)),
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

        for model_name, feature_names in MODEL_FEATURES.items():
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=10000,
                random_state=random_state,
            )
            model.fit(train_scaled[feature_names], train_labels)
            train_probability = model.predict_proba(train_scaled[feature_names])[:, 1]
            test_probability = model.predict_proba(test_scaled[feature_names])[:, 1]

            learned = learn_f1_probability_threshold(train_probability, train_labels)
            threshold = learned["threshold"]
            test_prediction = (test_probability >= threshold).astype(int)
            test_metrics = base.evaluate_predictions(
                test_labels, test_prediction, test_probability
            )

            oof_probability[model_name][test_idx] = test_probability
            oof_prediction[model_name][test_idx] = test_prediction
            oof_threshold[model_name][test_idx] = threshold

            coefficient_lookup = dict(zip(feature_names, model.coef_[0], strict=True))
            fold_parameter_rows.append(
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
                    "scaler_mean_n": float(scaler.mean_[0]),
                    "scaler_sd_n": float(scaler.scale_[0]),
                    "scaler_mean_m": float(scaler.mean_[1]),
                    "scaler_sd_m": float(scaler.scale_[1]),
                    "scaler_mean_log_num_refs": float(scaler.mean_[2]),
                    "scaler_sd_log_num_refs": float(scaler.scale_[2]),
                    "intercept": float(model.intercept_[0]),
                    "coefficient_n": float(coefficient_lookup.get("n", math.nan)),
                    "coefficient_m": float(coefficient_lookup.get("m", math.nan)),
                    "coefficient_log_num_refs": float(
                        coefficient_lookup.get("log_num_refs", math.nan)
                    ),
                    "probability_threshold": threshold,
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
                    "model": model_name,
                    "probability_threshold": threshold,
                    **test_metrics,
                }
            )

    if not np.all(oof_fold > 0) or not np.isfinite(oof_n).all():
        raise RuntimeError("OOF fold assignments or relevance values are incomplete.")
    for model_name in MODEL_FEATURES:
        if not np.isfinite(oof_probability[model_name]).all():
            raise RuntimeError(f"OOF probabilities are incomplete for {model_name}.")

    oof = data[["idx", "award", "num_refs", "m"]].copy()
    oof["fold"] = oof_fold
    oof["n"] = oof_n
    oof["log_num_refs"] = oof_log_num_refs
    for model_name in MODEL_FEATURES:
        oof[f"probability_{model_name}"] = oof_probability[model_name]
        oof[f"prediction_{model_name}"] = oof_prediction[model_name]
        oof[f"threshold_{model_name}"] = oof_threshold[model_name]
    oof = oof.sort_values("idx").reset_index(drop=True)

    return pd.DataFrame(fold_parameter_rows), pd.DataFrame(fold_metric_rows), oof


def evaluate_oof_models(oof: pd.DataFrame) -> pd.DataFrame:
    labels = oof["award"].to_numpy(dtype=int)
    rows = []
    for model_name in MODEL_FEATURES:
        result = base.evaluate_predictions(
            labels,
            oof[f"prediction_{model_name}"].to_numpy(dtype=int),
            oof[f"probability_{model_name}"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "field": "AI",
                "evaluation": "OOF pooled",
                "model": model_name,
                "features": "+".join(MODEL_FEATURES[model_name]),
                **result,
            }
        )
    return pd.DataFrame(rows)


def make_metric_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    lookup = overall.set_index("model")
    rows = []
    for metric in CORE_METRICS:
        m0 = float(lookup.loc["M0_count", metric])
        m1 = float(lookup.loc["M1_relevance_count", metric])
        m2 = float(lookup.loc["M2_relevance_heterogeneity_count", metric])
        rows.append(
            {
                "metric": metric,
                "M0_count": m0,
                "M1_relevance_count": m1,
                "M2_relevance_heterogeneity_count": m2,
                "M2_minus_M1": m2 - m1,
                "M1_minus_M0": m1 - m0,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_m2_minus_m1(
    oof: pd.DataFrame,
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    labels = oof["award"].to_numpy(dtype=int)
    m1_prediction = oof["prediction_M1_relevance_count"].to_numpy(dtype=int)
    m1_probability = oof["probability_M1_relevance_count"].to_numpy(dtype=float)
    m2_prediction = oof[
        "prediction_M2_relevance_heterogeneity_count"
    ].to_numpy(dtype=int)
    m2_probability = oof[
        "probability_M2_relevance_heterogeneity_count"
    ].to_numpy(dtype=float)

    point_m1 = base.evaluate_predictions(labels, m1_prediction, m1_probability)
    point_m2 = base.evaluate_predictions(labels, m2_prediction, m2_probability)
    differences: dict[str, list[float]] = {metric: [] for metric in CORE_METRICS}

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
        sample_m1 = base.evaluate_predictions(
            labels[sample_idx],
            m1_prediction[sample_idx],
            m1_probability[sample_idx],
        )
        sample_m2 = base.evaluate_predictions(
            labels[sample_idx],
            m2_prediction[sample_idx],
            m2_probability[sample_idx],
        )
        for metric in CORE_METRICS:
            difference = float(sample_m2[metric]) - float(sample_m1[metric])
            if np.isfinite(difference):
                differences[metric].append(difference)

    rows = []
    for metric in CORE_METRICS:
        values = np.asarray(differences[metric], dtype=float)
        lower, upper = (
            np.quantile(values, [0.025, 0.975])
            if len(values)
            else (math.nan, math.nan)
        )
        rows.append(
            {
                "metric": metric,
                "M1_relevance_count": float(point_m1[metric]),
                "M2_relevance_heterogeneity_count": float(point_m2[metric]),
                "M2_minus_M1": float(point_m2[metric]) - float(point_m1[metric]),
                "difference_ci_lower_95": float(lower),
                "difference_ci_upper_95": float(upper),
                "valid_bootstrap_repetitions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def adjusted_inference(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = ["n", "m", "log_num_refs"]
    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(oof[feature_names]),
        columns=feature_names,
        index=oof.index,
    )
    labels = oof["award"].to_numpy(dtype=int)

    m1_features = ["n", "log_num_refs"]
    m2_features = ["n", "m", "log_num_refs"]
    m1_fit = sm.Logit(labels, sm.add_constant(standardized[m1_features])).fit(
        disp=False, cov_type="HC3"
    )
    m2_fit = sm.Logit(labels, sm.add_constant(standardized[m2_features])).fit(
        disp=False, cov_type="HC3"
    )

    rows = []
    for model_name, fitted, features in (
        ("M1_relevance_count", m1_fit, m1_features),
        ("M2_relevance_heterogeneity_count", m2_fit, m2_features),
    ):
        conf = fitted.conf_int(alpha=0.05)
        for term in ["const", *features]:
            coefficient = float(fitted.params[term])
            lower = float(conf.loc[term, 0])
            upper = float(conf.loc[term, 1])
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coefficient": coefficient,
                    "robust_se_HC3": float(fitted.bse[term]),
                    "z": float(fitted.tvalues[term]),
                    "p_value": float(fitted.pvalues[term]),
                    "coefficient_ci_lower_95": lower,
                    "coefficient_ci_upper_95": upper,
                    "odds_ratio": float(np.exp(coefficient)),
                    "odds_ratio_ci_lower_95": float(np.exp(lower)),
                    "odds_ratio_ci_upper_95": float(np.exp(upper)),
                    "unit": "one pooled-sample standard deviation",
                }
            )

    likelihood_ratio = 2.0 * (m2_fit.llf - m1_fit.llf)
    degrees_freedom = int(m2_fit.df_model - m1_fit.df_model)
    comparison = pd.DataFrame(
        [
            {
                "comparison": "M2 versus M1",
                "added_term": "m",
                "M1_log_likelihood": float(m1_fit.llf),
                "M2_log_likelihood": float(m2_fit.llf),
                "likelihood_ratio_chi2": float(likelihood_ratio),
                "degrees_freedom": degrees_freedom,
                "likelihood_ratio_p_value": float(
                    chi2.sf(likelihood_ratio, degrees_freedom)
                ),
                "M1_AIC": float(m1_fit.aic),
                "M2_AIC": float(m2_fit.aic),
                "M1_BIC": float(m1_fit.bic),
                "M2_BIC": float(m2_fit.bic),
            }
        ]
    )
    return pd.DataFrame(rows), comparison


def make_design_checks(
    oof: pd.DataFrame,
    fold_parameters: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Design check failed: {name}. {details}")

    add(
        "all_oof_rows_present",
        bool(oof["idx"].is_unique and len(oof) > 0),
        f"rows={len(oof)}, unique_idx={oof['idx'].nunique()}",
    )
    add(
        "all_probabilities_finite",
        bool(
            np.isfinite(
                oof[
                    [f"probability_{model}" for model in MODEL_FEATURES]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all three OOF probability columns checked",
    )
    add(
        "positive_training_scaler_sd",
        bool(
            (
                fold_parameters[
                    ["scaler_sd_n", "scaler_sd_m", "scaler_sd_log_num_refs"]
                ]
                > 0
            )
            .all()
            .all()
        ),
        "n, m, and log_num_refs scaler SDs checked",
    )

    if MAIN_OOF_PATH.exists():
        main_oof = pd.read_csv(MAIN_OOF_PATH).sort_values("idx").reset_index(drop=True)
        same_idx = oof["idx"].equals(main_oof["idx"])
        same_fold = same_idx and np.array_equal(
            oof["fold"].to_numpy(dtype=int),
            main_oof["fold"].to_numpy(dtype=int),
        )
        max_n_difference = (
            float(np.max(np.abs(oof["n"] - main_oof["n"]))) if same_idx else math.inf
        )
        add("same_fold_as_main_model", bool(same_fold), "compared by idx")
        add(
            "same_relevance_as_main_model",
            max_n_difference <= 1e-12,
            f"max_abs_difference={max_n_difference:.12g}",
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
    fold_parameters, fold_metrics, oof = run_fivefold_control_models(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    checks = make_design_checks(oof, fold_parameters)
    overall = evaluate_oof_models(oof)
    metric_comparison = make_metric_comparison(overall)
    paired_differences = paired_bootstrap_m2_minus_m1(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 20,
    )
    adjusted_coefficients, adjusted_model_comparison = adjusted_inference(oof)

    outputs = {
        "reference_count_input_validation.csv": validation,
        "reference_count_design_checks.csv": checks,
        "reference_count_fold_parameters.csv": fold_parameters,
        "reference_count_fold_test_metrics.csv": fold_metrics,
        "reference_count_oof_predictions.csv": oof,
        "reference_count_overall_metrics.csv": overall,
        "reference_count_metric_comparison.csv": metric_comparison,
        "reference_count_paired_bootstrap_m2_vs_m1.csv": paired_differences,
        "reference_count_adjusted_coefficients.csv": adjusted_coefficients,
        "reference_count_adjusted_model_comparison.csv": adjusted_model_comparison,
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
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "models": MODEL_FEATURES,
        "reference_count_transform": "log(1 + num_refs)",
        "standardization": (
            "StandardScaler fit on n, m, and log_num_refs in each training fold; "
            "training-fold parameters applied unchanged to the held-out fold."
        ),
        "estimator": (
            "Unpenalized binary logistic regression; F1-optimal probability "
            "threshold learned on each training fold."
        ),
        "inference": (
            "Pooled auxiliary Logit using OOF n and standardized predictors, "
            "with HC3 robust standard errors."
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "output_files": list(outputs),
    }
    with (output_dir / "reference_count_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Reference-count control analysis completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF metric comparison:")
    print(metric_comparison.to_string(index=False))
    print("\nAdjusted M2 heterogeneity coefficient:")
    print(
        adjusted_coefficients[
            (adjusted_coefficients["model"] == "M2_relevance_heterogeneity_count")
            & (adjusted_coefficients["term"] == "m")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
