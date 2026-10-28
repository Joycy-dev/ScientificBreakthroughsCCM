"""Physics-domain robustness check for the knowledge-heterogeneity indicator.

The analysis compares the main range indicator with std_similarity using the
same stratified folds. All fold-dependent weights, beta values, and thresholds
are learned from the training portion only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


FIELD = "Physics"
SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_DIR = SCRIPT_DIR.parent
PHYSICS_TRAINING_SCRIPT = (
    PHYSICS_DIR / "physics_train" / "physics_fivefold_training.py"
)
AI_ROBUSTNESS_SCRIPT = (
    PHYSICS_DIR.parent
    / "AI"
    / "AI_robustness"
    / "ai_heterogeneity_robustness.py"
)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


physics_training = load_module("physics_training_for_robustness", PHYSICS_TRAINING_SCRIPT)
robustness = load_module("ai_heterogeneity_robustness_core", AI_ROBUSTNESS_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics robustness comparison of range and SD heterogeneity."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PHYSICS_DIR / "physics_data.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
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


def replace_field_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "field" in result.columns:
        result["field"] = FIELD
    if "Field" in result.columns:
        result["Field"] = FIELD
    if "AI" in result.columns:
        result = result.rename(columns={"AI": FIELD})
    return result


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, validation = physics_training.load_and_validate_data(input_path)
    descriptive, correlations = robustness.indicator_diagnostics(data)

    range_parameters, range_fold_metrics, range_oof = robustness.run_variant(
        data,
        heterogeneity_column="m",
        variant="range_main",
        folds=args.folds,
        random_state=args.random_state,
    )
    std_parameters, std_fold_metrics, std_oof = robustness.run_variant(
        data,
        heterogeneity_column="std_similarity",
        variant="std_alternative",
        folds=args.folds,
        random_state=args.random_state,
    )

    checks = robustness.design_checks(
        range_parameters, std_parameters, range_oof, std_oof
    )
    range_overall = robustness.overall_metrics(range_oof)
    std_overall = robustness.overall_metrics(std_oof)
    comparison = robustness.make_metric_comparison(range_overall, std_overall)
    paired_differences = robustness.paired_bootstrap_differences(
        range_oof,
        std_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 10,
    )
    range_ci = robustness.base.stratified_bootstrap_confidence_intervals(
        range_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 1,
    )
    std_ci = robustness.base.stratified_bootstrap_confidence_intervals(
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

    range_parameters = replace_field_labels(range_parameters)
    std_parameters = replace_field_labels(std_parameters)
    range_fold_metrics = replace_field_labels(range_fold_metrics)
    std_fold_metrics = replace_field_labels(std_fold_metrics)
    parameter_comparison = replace_field_labels(parameter_comparison)

    outputs = {
        "physics_indicator_input_validation.csv": validation,
        "physics_indicator_descriptive_statistics.csv": descriptive,
        "physics_indicator_correlations.csv": correlations,
        "physics_indicator_design_checks.csv": checks,
        "physics_indicator_fold_parameters_range.csv": range_parameters,
        "physics_indicator_fold_parameters_std.csv": std_parameters,
        "physics_indicator_fold_parameter_comparison.csv": parameter_comparison,
        "physics_indicator_fold_test_metrics_range.csv": range_fold_metrics,
        "physics_indicator_fold_test_metrics_std.csv": std_fold_metrics,
        "physics_indicator_oof_predictions_range.csv": range_oof,
        "physics_indicator_oof_predictions_std.csv": std_oof,
        "physics_indicator_ci_range.csv": range_ci,
        "physics_indicator_ci_std.csv": std_ci,
        "physics_indicator_metrics_comparison.csv": comparison,
        "physics_indicator_paired_bootstrap.csv": paired_differences,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "field": FIELD,
        "physics_training_script": str(PHYSICS_TRAINING_SCRIPT),
        "physics_training_script_sha256": file_sha256(PHYSICS_TRAINING_SCRIPT),
        "robustness_core_script": str(AI_ROBUSTNESS_SCRIPT),
        "robustness_core_script_sha256": file_sha256(AI_ROBUSTNESS_SCRIPT),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "variants": {
            "range_main": "max_similarity - min_similarity",
            "std_alternative": "std_similarity",
        },
        "leakage_control": (
            "Citation weights, beta, and F1-optimal threshold are learned "
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
    with (output_dir / "physics_indicator_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Physics heterogeneity-indicator robustness check completed.")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOverall OOF metric comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
