"""Sensitivity analysis for the AI-domain cusp scale coefficient beta.

For every outer training fold, the original dispersion-calibrated beta is
multiplied by 0.50, 0.75, 1.00, 1.25, and 1.50. The F1-optimal empirical
classification threshold is re-learned on the training fold for each multiplier
and applied unchanged to its held-out fold. The theoretical cusp boundary at
Delta = 0 is evaluated separately for the baseline multiplier (1.00).
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
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent
BASE_TRAINING_SCRIPT = AI_DIR / "AI_train" / "ai_fivefold_training.py"
MAIN_OOF_PATH = AI_DIR / "AI_train" / "oof_predictions.csv"

DEFAULT_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
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
        "ai_fivefold_training_base_for_sensitivity", BASE_TRAINING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_training_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-domain sensitivity analysis for the cusp beta coefficient."
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
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_MULTIPLIERS),
    )
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
    """Learn score >= threshold using exact cumulative confusion counts."""
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


def run_sensitivity(
    data: pd.DataFrame,
    folds: int,
    random_state: int,
    multipliers: list[float],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    labels = data["award"].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    multiplier_values = sorted(set(float(value) for value in multipliers))
    if any(value <= 0 or not np.isfinite(value) for value in multiplier_values):
        raise ValueError("All beta multipliers must be positive finite values.")
    if 1.0 not in multiplier_values:
        raise ValueError("Sensitivity multipliers must include the baseline 1.0.")

    n_rows = len(data)
    oof = {
        multiplier: {
            "fold": np.zeros(n_rows, dtype=int),
            "n": np.full(n_rows, np.nan, dtype=float),
            "beta_base": np.full(n_rows, np.nan, dtype=float),
            "beta_used": np.full(n_rows, np.nan, dtype=float),
            "delta": np.full(n_rows, np.nan, dtype=float),
            "score": np.full(n_rows, np.nan, dtype=float),
            "delta_threshold": np.full(n_rows, np.nan, dtype=float),
            "prediction": np.zeros(n_rows, dtype=int),
        }
        for multiplier in multiplier_values
    }
    zero_boundary_prediction = np.zeros(n_rows, dtype=int)
    zero_boundary_score = np.full(n_rows, np.nan, dtype=float)

    fold_parameter_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data, labels), start=1
    ):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        parameters = base.calculate_citation_weights(train)
        train_n = base.calculate_relevance(train, parameters)
        test_n = base.calculate_relevance(test, parameters)
        train_m = train["m"].to_numpy(dtype=float)
        test_m = test["m"].to_numpy(dtype=float)
        beta_base, sigma_m_cubed, sigma_n_squared = base.calculate_beta(
            train_m, train_n
        )

        for multiplier in multiplier_values:
            beta_used = beta_base * multiplier
            _, _, train_delta = base.calculate_discriminant(
                train_m, train_n, beta_used
            )
            _, _, test_delta = base.calculate_discriminant(
                test_m, test_n, beta_used
            )
            train_score = -train_delta
            test_raw_score = -test_delta
            learned = learn_high_score_f1_threshold(train_score, train_labels)
            score_threshold = learned["score_threshold"]
            delta_threshold = -score_threshold
            test_prediction = (test_raw_score >= score_threshold).astype(int)
            # Center each held-out fold on its training-learned threshold, as in
            # the main model, so pooled OOF ranking scores share one origin.
            test_score = test_raw_score - score_threshold
            test_metrics = base.evaluate_predictions(
                test_labels, test_prediction, test_score
            )

            values = oof[multiplier]
            values["fold"][test_idx] = fold
            values["n"][test_idx] = test_n
            values["beta_base"][test_idx] = beta_base
            values["beta_used"][test_idx] = beta_used
            values["delta"][test_idx] = test_delta
            values["score"][test_idx] = test_score
            values["delta_threshold"][test_idx] = delta_threshold
            values["prediction"][test_idx] = test_prediction

            fold_parameter_rows.append(
                {
                    "field": "AI",
                    "fold": fold,
                    "beta_multiplier": multiplier,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "weight_d": parameters["weight_d"],
                    "weight_m": parameters["weight_m"],
                    "weight_s": parameters["weight_s"],
                    "sigma_train_m_cubed": sigma_m_cubed,
                    "sigma_train_n_squared": sigma_n_squared,
                    "beta_base": beta_base,
                    "beta_used": beta_used,
                    "empirical_delta_threshold": delta_threshold,
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
                    "beta_multiplier": multiplier,
                    "empirical_delta_threshold": delta_threshold,
                    **test_metrics,
                }
            )

            if multiplier == 1.0:
                zero_boundary_prediction[test_idx] = (test_delta <= 0.0).astype(int)
                zero_boundary_score[test_idx] = -test_delta

    long_oof_frames = []
    overall_rows = []
    for multiplier in multiplier_values:
        values = oof[multiplier]
        if not (
            np.all(values["fold"] > 0)
            and np.isfinite(values["score"]).all()
            and np.isfinite(values["delta"]).all()
        ):
            raise RuntimeError(f"Incomplete OOF results for multiplier {multiplier}.")
        frame = data[["idx", "award", "m"]].copy()
        frame["beta_multiplier"] = multiplier
        frame["fold"] = values["fold"]
        frame["n"] = values["n"]
        frame["beta_base"] = values["beta_base"]
        frame["beta_used"] = values["beta_used"]
        frame["delta"] = values["delta"]
        frame["breakthrough_score"] = values["score"]
        frame["empirical_delta_threshold"] = values["delta_threshold"]
        frame["prediction"] = values["prediction"]
        long_oof_frames.append(frame)

        metrics = base.evaluate_predictions(
            labels,
            values["prediction"],
            values["score"],
        )
        overall_rows.append(
            {
                "field": "AI",
                "evaluation": "OOF pooled",
                "threshold_rule": "training-fold F1-optimal empirical threshold",
                "beta_multiplier": multiplier,
                **metrics,
            }
        )

    zero_metrics = base.evaluate_predictions(
        labels,
        zero_boundary_prediction,
        zero_boundary_score,
    )
    zero_boundary = pd.DataFrame(
        [
            {
                "field": "AI",
                "evaluation": "OOF pooled",
                "beta_multiplier": 1.0,
                "threshold_rule": "theoretical cusp boundary Delta = 0",
                "delta_threshold": 0.0,
                **zero_metrics,
            }
        ]
    )
    return (
        pd.DataFrame(fold_parameter_rows),
        pd.DataFrame(fold_metric_rows),
        pd.concat(long_oof_frames, ignore_index=True).sort_values(
            ["beta_multiplier", "idx"]
        ),
        pd.DataFrame(overall_rows),
        zero_boundary,
    )


def make_metric_sensitivity_summary(overall: pd.DataFrame) -> pd.DataFrame:
    baseline = overall.loc[overall["beta_multiplier"] == 1.0].iloc[0]
    rows = []
    for metric in CORE_METRICS:
        values = overall[metric].astype(float)
        baseline_value = float(baseline[metric])
        minimum = float(values.min())
        maximum = float(values.max())
        max_abs_change = float(np.max(np.abs(values - baseline_value)))
        rows.append(
            {
                "metric": metric,
                "baseline_at_1.00": baseline_value,
                "minimum_across_multipliers": minimum,
                "maximum_across_multipliers": maximum,
                "range": maximum - minimum,
                "maximum_absolute_change_from_baseline": max_abs_change,
                "maximum_relative_change_from_baseline": (
                    max_abs_change / abs(baseline_value)
                    if baseline_value != 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_design_checks(
    overall: pd.DataFrame,
    sensitivity_oof: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Sensitivity design check failed: {name}. {details}")

    baseline_oof = sensitivity_oof[
        sensitivity_oof["beta_multiplier"] == 1.0
    ].sort_values("idx").reset_index(drop=True)
    add(
        "baseline_oof_complete",
        bool(len(baseline_oof) > 0 and baseline_oof["idx"].is_unique),
        f"rows={len(baseline_oof)}, unique_idx={baseline_oof['idx'].nunique()}",
    )
    add(
        "all_sensitivity_values_finite",
        bool(
            np.isfinite(
                sensitivity_oof[
                    ["n", "beta_base", "beta_used", "delta", "breakthrough_score"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "n, beta, Delta, and scores checked",
    )

    if MAIN_OOF_PATH.exists():
        main_oof = pd.read_csv(MAIN_OOF_PATH).sort_values("idx").reset_index(drop=True)
        same_idx = np.array_equal(
            baseline_oof["idx"].to_numpy(), main_oof["idx"].to_numpy()
        )
        same_fold = same_idx and np.array_equal(
            baseline_oof["fold"].to_numpy(dtype=int),
            main_oof["fold"].to_numpy(dtype=int),
        )
        max_delta_difference = (
            float(
                np.max(
                    np.abs(
                        baseline_oof["delta"].to_numpy(dtype=float)
                        - main_oof["delta"].to_numpy(dtype=float)
                    )
                )
            )
            if same_idx
            else math.inf
        )
        prediction_match = same_idx and np.array_equal(
            baseline_oof["prediction"].to_numpy(dtype=int),
            main_oof["prediction"].to_numpy(dtype=int),
        )
        add("same_fold_as_main_model", bool(same_fold), "compared by idx")
        add(
            "baseline_delta_reproduces_main_model",
            max_delta_difference <= 1e-12,
            f"max_abs_difference={max_delta_difference:.12g}",
        )
        add(
            "baseline_predictions_reproduce_main_model",
            bool(prediction_match),
            "OOF predictions compared by idx",
        )

        main_overall = base.evaluate_predictions(
            main_oof["award"].to_numpy(dtype=int),
            main_oof["prediction"].to_numpy(dtype=int),
            main_oof["breakthrough_score"].to_numpy(dtype=float),
        )
        baseline_row = overall.loc[overall["beta_multiplier"] == 1.0].iloc[0]
        max_metric_difference = max(
            abs(float(baseline_row[metric]) - float(main_overall[metric]))
            for metric in CORE_METRICS
        )
        add(
            "baseline_metrics_reproduce_main_model",
            max_metric_difference <= 1e-12,
            f"max_abs_difference={max_metric_difference:.12g}",
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
    (
        fold_parameters,
        fold_metrics,
        sensitivity_oof,
        overall,
        zero_boundary,
    ) = run_sensitivity(
        data,
        folds=args.folds,
        random_state=args.random_state,
        multipliers=args.multipliers,
    )
    summary = make_metric_sensitivity_summary(overall)
    checks = make_design_checks(overall, sensitivity_oof)

    outputs = {
        "sensitivity_input_validation.csv": validation,
        "sensitivity_design_checks.csv": checks,
        "sensitivity_fold_parameters.csv": fold_parameters,
        "sensitivity_fold_test_metrics.csv": fold_metrics,
        "sensitivity_oof_predictions.csv": sensitivity_oof,
        "sensitivity_overall_metrics.csv": overall,
        "sensitivity_metric_range_summary.csv": summary,
        "theoretical_zero_boundary_metrics.csv": zero_boundary,
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
        "beta_multipliers": sorted(set(float(value) for value in args.multipliers)),
        "empirical_threshold": (
            "F1-optimal Delta threshold learned independently within each training "
            "fold for every beta multiplier."
        ),
        "theoretical_boundary": (
            "Delta = 0 evaluated separately at beta multiplier 1.0; it is not "
            "treated as the empirically learned classification threshold."
        ),
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
    with (output_dir / "sensitivity_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("AI beta sensitivity analysis completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF metrics across beta multipliers:")
    print(
        overall[["beta_multiplier", *CORE_METRICS]].to_string(index=False)
    )
    print("\nTheoretical zero-boundary metrics at baseline beta:")
    print(zero_boundary[["delta_threshold", *CORE_METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
