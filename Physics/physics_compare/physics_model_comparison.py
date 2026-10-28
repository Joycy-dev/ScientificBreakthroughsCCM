"""Fair Physics-domain comparison of four requested baselines and the cusp model.

All models use the same stratified five folds. Citation weights, feature
standardization, model coefficients, beta, and F1-optimal thresholds are
estimated on each training fold and applied unchanged to its held-out fold.
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
PHYSICS_MAIN_OOF = PHYSICS_DIR / "physics_train" / "oof_predictions.csv"
PHYSICS_REFERENCE_COUNT_OOF = (
    PHYSICS_DIR
    / "physics_robustness"
    / "physics_reference_count_oof_predictions.csv"
)
AI_COMPARISON_SCRIPT = (
    PHYSICS_DIR.parent / "AI" / "AI_compare" / "ai_model_comparison.py"
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


physics_training = load_module(
    "physics_training_for_model_comparison", PHYSICS_TRAINING_SCRIPT
)
comparison = load_module("ai_model_comparison_core", AI_COMPARISON_SCRIPT)
comparison.MAIN_OOF_PATH = PHYSICS_MAIN_OOF
comparison.REFERENCE_COUNT_OOF_PATH = PHYSICS_REFERENCE_COUNT_OOF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four requested baselines with the Physics cusp model."
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
    fold_parameters, fold_metrics, oof = comparison.run_comparison(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    oof["idx"] = oof["idx"].astype("int64")
    checks = comparison.design_checks(oof)
    overall = comparison.overall_metrics(oof)
    display_table = comparison.make_display_table(overall)
    paired_difference = comparison.paired_bootstrap_cusp_minus_each_baseline(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 30,
    )

    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)
    overall = replace_field_labels(overall)
    baseline_overall = overall[
        overall["model"].isin(comparison.BASELINE_MODELS)
    ].copy()

    outputs = {
        "physics_four_baselines_input_validation.csv": validation,
        "physics_four_baselines_vs_cusp_design_checks.csv": checks,
        "physics_four_baselines_fold_parameters.csv": fold_parameters[
            fold_parameters["model"].isin(comparison.BASELINE_MODELS)
        ].copy(),
        "physics_four_baselines_fold_test_metrics.csv": fold_metrics[
            fold_metrics["model"].isin(comparison.BASELINE_MODELS)
        ].copy(),
        "physics_four_baselines_overall_metrics.csv": baseline_overall,
        "physics_five_models_fold_parameters.csv": fold_parameters,
        "physics_five_models_fold_test_metrics.csv": fold_metrics,
        "physics_five_models_oof_predictions.csv": oof,
        "physics_five_models_overall_comparison.csv": overall,
        "physics_five_models_display_table.csv": display_table,
        "physics_cusp_vs_each_of_four_baselines_paired_bootstrap.csv": (
            paired_difference
        ),
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
        "comparison_core_script": str(AI_COMPARISON_SCRIPT),
        "comparison_core_script_sha256": file_sha256(AI_COMPARISON_SCRIPT),
        "main_oof_file": str(PHYSICS_MAIN_OOF),
        "main_oof_sha256": file_sha256(PHYSICS_MAIN_OOF),
        "prior_reference_count_oof_file": str(PHYSICS_REFERENCE_COUNT_OOF),
        "prior_reference_count_oof_sha256": file_sha256(
            PHYSICS_REFERENCE_COUNT_OOF
        ),
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
            comparison.PROPOSED_MODEL: "Delta = -8*m^3 + 27*beta^2*n^2",
        },
        "feature_scaling": (
            "m, n, and log_num_refs standardized using training-fold mean and "
            "standard deviation for logistic models only; training-fold "
            "parameters applied unchanged to the held-out fold."
        ),
        "threshold_selection": (
            "F1-optimal empirical threshold learned independently in each "
            "training fold for every model."
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
    with (output_dir / "physics_four_baselines_vs_cusp_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Physics four-baseline versus cusp comparison completed.")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF model comparison:")
    print(display_table.to_string(index=False))
    print("\nPaired bootstrap: proposed cusp minus each baseline")
    print(paired_difference.to_string(index=False))


if __name__ == "__main__":
    main()
