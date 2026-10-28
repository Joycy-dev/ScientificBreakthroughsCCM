"""Robustness check for the AI-domain knowledge-heterogeneity indicator.

The script compares two otherwise identical leakage-free five-fold cusp models:

1. range: max_similarity - min_similarity (the main specification)
2. std_similarity: the population standard deviation of reference similarities

Both variants use exactly the same folds. Citation-function weights, beta, and
the F1-optimal threshold are estimated from each training fold only and then
applied unchanged to its held-out fold.
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


def load_base_training_module():
    spec = importlib.util.spec_from_file_location(
        "ai_fivefold_training_base", BASE_TRAINING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_training_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare range- and standard-deviation heterogeneity indicators."
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


def run_variant(
    data: pd.DataFrame,
    heterogeneity_column: str,
    variant: str,
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
    arrays = {
        "fold": np.zeros(n_rows, dtype=int),
        "prediction": np.zeros(n_rows, dtype=int),
        "score": np.full(n_rows, np.nan, dtype=float),
        "n": np.full(n_rows, np.nan, dtype=float),
        "cusp_m": np.full(n_rows, np.nan, dtype=float),
        "n_std": np.full(n_rows, np.nan, dtype=float),
        "delta": np.full(n_rows, np.nan, dtype=float),
        "threshold": np.full(n_rows, np.nan, dtype=float),
        "beta": np.full(n_rows, np.nan, dtype=float),
        "weight_d": np.full(n_rows, np.nan, dtype=float),
        "weight_m": np.full(n_rows, np.nan, dtype=float),
        "weight_s": np.full(n_rows, np.nan, dtype=float),
    }

    parameter_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

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

        train_h = train[heterogeneity_column].to_numpy(dtype=float)
        test_h = test[heterogeneity_column].to_numpy(dtype=float)
        beta, sigma_h_cubed, sigma_n_squared = base.calculate_beta(train_h, train_n)
        train_cusp_m, train_n_std, train_delta = base.calculate_discriminant(
            train_h, train_n, beta
        )
        test_cusp_m, test_n_std, test_delta = base.calculate_discriminant(
            test_h, test_n, beta
        )

        learned = base.learn_f1_optimal_threshold(train_delta, train_labels)
        threshold = learned["threshold"]
        test_prediction = (test_delta <= threshold).astype(int)
        test_score = threshold - test_delta
        test_metrics = base.evaluate_predictions(
            test_labels, test_prediction, test_score
        )

        arrays["fold"][test_idx] = fold
        arrays["prediction"][test_idx] = test_prediction
        arrays["score"][test_idx] = test_score
        arrays["n"][test_idx] = test_n
        arrays["cusp_m"][test_idx] = test_cusp_m
        arrays["n_std"][test_idx] = test_n_std
        arrays["delta"][test_idx] = test_delta
        arrays["threshold"][test_idx] = threshold
        arrays["beta"][test_idx] = beta
        arrays["weight_d"][test_idx] = parameters["weight_d"]
        arrays["weight_m"][test_idx] = parameters["weight_m"]
        arrays["weight_s"][test_idx] = parameters["weight_s"]

        parameter_rows.append(
            {
                "field": "AI",
                "variant": variant,
                "heterogeneity_column": heterogeneity_column,
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
                "sigma_train_heterogeneity_cubed": sigma_h_cubed,
                "sigma_train_n_squared": sigma_n_squared,
                "beta": beta,
                "threshold": threshold,
                "train_accuracy_at_threshold": learned["Accuracy"],
                "train_precision_at_threshold": learned["Precision"],
                "train_recall_at_threshold": learned["Recall"],
                "train_f1_at_threshold": learned["F1"],
            }
        )
        metric_rows.append(
            {
                "field": "AI",
                "variant": variant,
                "heterogeneity_column": heterogeneity_column,
                "fold": fold,
                "threshold": threshold,
                **test_metrics,
            }
        )

        # Keep the training calculations explicit for auditability.
        _ = train_cusp_m, train_n_std

    if not (
        np.all(arrays["fold"] > 0)
        and np.isfinite(arrays["score"]).all()
        and np.isfinite(arrays["delta"]).all()
    ):
        raise RuntimeError(f"Incomplete OOF predictions for variant: {variant}")

    oof = data.copy()
    oof["variant"] = variant
    oof["heterogeneity_column"] = heterogeneity_column
    oof["heterogeneity_raw"] = data[heterogeneity_column]
    oof["fold"] = arrays["fold"]
    oof["P_d"] = data["P_d"]
    oof["P_m"] = data["P_m"]
    oof["P_s"] = data["P_s"]
    oof["weight_d_used"] = arrays["weight_d"]
    oof["weight_m_used"] = arrays["weight_m"]
    oof["weight_s_used"] = arrays["weight_s"]
    oof["n"] = arrays["n"]
    oof["beta_used"] = arrays["beta"]
    oof["cusp_m"] = arrays["cusp_m"]
    oof["n_std"] = arrays["n_std"]
    oof["delta"] = arrays["delta"]
    oof["threshold_used"] = arrays["threshold"]
    oof["prediction"] = arrays["prediction"]
    oof["breakthrough_score"] = arrays["score"]
    oof = oof.sort_values("idx").reset_index(drop=True)

    return pd.DataFrame(parameter_rows), pd.DataFrame(metric_rows), oof


def overall_metrics(oof: pd.DataFrame) -> dict[str, float | int]:
    return base.evaluate_predictions(
        oof["award"].to_numpy(dtype=int),
        oof["prediction"].to_numpy(dtype=int),
        oof["breakthrough_score"].to_numpy(dtype=float),
    )


def make_metric_comparison(
    range_metrics: dict[str, float | int],
    std_metrics: dict[str, float | int],
) -> pd.DataFrame:
    rows = []
    for metric in base.BOOTSTRAP_METRICS:
        range_value = float(range_metrics[metric])
        std_value = float(std_metrics[metric])
        difference = std_value - range_value
        relative_change = (
            difference / abs(range_value)
            if np.isfinite(range_value) and range_value != 0
            else math.nan
        )
        rows.append(
            {
                "metric": metric,
                "range_main": range_value,
                "std_alternative": std_value,
                "std_minus_range": difference,
                "relative_change": relative_change,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_differences(
    range_oof: pd.DataFrame,
    std_oof: pd.DataFrame,
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    if not range_oof["idx"].equals(std_oof["idx"]):
        raise ValueError("OOF rows are not aligned by idx.")
    if not range_oof["award"].equals(std_oof["award"]):
        raise ValueError("OOF labels differ between variants.")

    labels = range_oof["award"].to_numpy(dtype=int)
    range_prediction = range_oof["prediction"].to_numpy(dtype=int)
    range_score = range_oof["breakthrough_score"].to_numpy(dtype=float)
    std_prediction = std_oof["prediction"].to_numpy(dtype=int)
    std_score = std_oof["breakthrough_score"].to_numpy(dtype=float)

    point_range = base.evaluate_predictions(labels, range_prediction, range_score)
    point_std = base.evaluate_predictions(labels, std_prediction, std_score)
    differences: dict[str, list[float]] = {
        metric: [] for metric in base.BOOTSTRAP_METRICS
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
        sample_range = base.evaluate_predictions(
            labels[sample_idx],
            range_prediction[sample_idx],
            range_score[sample_idx],
        )
        sample_std = base.evaluate_predictions(
            labels[sample_idx],
            std_prediction[sample_idx],
            std_score[sample_idx],
        )
        for metric in base.BOOTSTRAP_METRICS:
            difference = float(sample_std[metric]) - float(sample_range[metric])
            if np.isfinite(difference):
                differences[metric].append(difference)

    rows = []
    for metric in base.BOOTSTRAP_METRICS:
        values = np.asarray(differences[metric], dtype=float)
        lower, upper = (
            np.quantile(values, [0.025, 0.975])
            if len(values)
            else (math.nan, math.nan)
        )
        rows.append(
            {
                "metric": metric,
                "range_main": float(point_range[metric]),
                "std_alternative": float(point_std[metric]),
                "std_minus_range": float(point_std[metric])
                - float(point_range[metric]),
                "difference_ci_lower_95": float(lower),
                "difference_ci_upper_95": float(upper),
                "valid_bootstrap_repetitions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def indicator_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["num_refs", "m", "std_similarity"]
    descriptive = data[columns].describe().T.reset_index(names="variable")
    correlation_rows = []
    for method in ("pearson", "spearman"):
        matrix = data[columns].corr(method=method)
        for left, right in (
            ("num_refs", "m"),
            ("num_refs", "std_similarity"),
            ("m", "std_similarity"),
        ):
            correlation_rows.append(
                {
                    "method": method,
                    "variable_1": left,
                    "variable_2": right,
                    "correlation": float(matrix.loc[left, right]),
                }
            )
    return descriptive, pd.DataFrame(correlation_rows)


def design_checks(
    range_parameters: pd.DataFrame,
    std_parameters: pd.DataFrame,
    range_oof: pd.DataFrame,
    std_oof: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Robustness design check failed: {name}. {details}")

    add(
        "same_oof_row_order",
        bool(range_oof["idx"].equals(std_oof["idx"])),
        "idx sequences compared",
    )
    add(
        "same_fold_assignment",
        bool(range_oof["fold"].equals(std_oof["fold"])),
        "fold assignments compared for every sample",
    )
    for weight in ("weight_d", "weight_m", "weight_s"):
        max_difference = float(
            np.max(
                np.abs(
                    range_parameters[weight].to_numpy(dtype=float)
                    - std_parameters[weight].to_numpy(dtype=float)
                )
            )
        )
        add(
            f"same_{weight}_by_fold",
            max_difference <= 1e-12,
            f"max_abs_difference={max_difference:.12g}",
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
    descriptive, correlations = indicator_diagnostics(data)

    range_parameters, range_fold_metrics, range_oof = run_variant(
        data,
        heterogeneity_column="m",
        variant="range_main",
        folds=args.folds,
        random_state=args.random_state,
    )
    std_parameters, std_fold_metrics, std_oof = run_variant(
        data,
        heterogeneity_column="std_similarity",
        variant="std_alternative",
        folds=args.folds,
        random_state=args.random_state,
    )

    checks = design_checks(range_parameters, std_parameters, range_oof, std_oof)
    range_overall = overall_metrics(range_oof)
    std_overall = overall_metrics(std_oof)
    comparison = make_metric_comparison(range_overall, std_overall)
    paired_differences = paired_bootstrap_differences(
        range_oof,
        std_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 10,
    )
    range_ci = base.stratified_bootstrap_confidence_intervals(
        range_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 1,
    )
    std_ci = base.stratified_bootstrap_confidence_intervals(
        std_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 2,
    )

    parameter_comparison = range_parameters.merge(
        std_parameters,
        on=["field", "fold"],
        suffixes=("_range", "_std"),
        validate="one_to_one",
    )

    outputs = {
        "input_validation.csv": validation,
        "indicator_descriptive_statistics.csv": descriptive,
        "indicator_correlations.csv": correlations,
        "design_checks.csv": checks,
        "fold_parameters_range.csv": range_parameters,
        "fold_parameters_std.csv": std_parameters,
        "fold_parameter_comparison.csv": parameter_comparison,
        "fold_test_metrics_range.csv": range_fold_metrics,
        "fold_test_metrics_std.csv": std_fold_metrics,
        "oof_predictions_range.csv": range_oof,
        "oof_predictions_std.csv": std_oof,
        "metric_confidence_intervals_range.csv": range_ci,
        "metric_confidence_intervals_std.csv": std_ci,
        "overall_metrics_comparison.csv": comparison,
        "paired_bootstrap_metric_differences.csv": paired_differences,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "base_training_script": str(BASE_TRAINING_SCRIPT),
        "base_training_script_sha256": file_sha256(BASE_TRAINING_SCRIPT),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "variants": {
            "range_main": {
                "heterogeneity_column": "m",
                "definition": "max_similarity - min_similarity",
            },
            "std_alternative": {
                "heterogeneity_column": "std_similarity",
                "definition": "numpy population standard deviation (ddof=0)",
            },
        },
        "leakage_control": (
            "Citation weights, beta, and F1-optimal threshold are estimated "
            "within each training fold and applied unchanged to its test fold."
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
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("AI heterogeneity robustness check completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOverall OOF metric comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
