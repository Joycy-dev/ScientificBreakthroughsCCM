"""Physics-domain auxiliary analysis controlling for reference count.

Three nested logistic models are evaluated on the same five folds as the main
Physics cusp experiment. All citation weights, standardization parameters, and
probability thresholds are learned from training data only.
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
import statsmodels


FIELD = "Physics"
SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_DIR = SCRIPT_DIR.parent
PHYSICS_TRAINING_SCRIPT = (
    PHYSICS_DIR / "physics_train" / "physics_fivefold_training.py"
)
PHYSICS_MAIN_OOF = PHYSICS_DIR / "physics_train" / "oof_predictions.csv"
AI_CONTROL_SCRIPT = (
    PHYSICS_DIR.parent
    / "AI"
    / "AI_robustness"
    / "reference_count_control.py"
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


physics_training = load_module("physics_training_for_count_control", PHYSICS_TRAINING_SCRIPT)
control = load_module("ai_reference_count_control_core", AI_CONTROL_SCRIPT)
control.MAIN_OOF_PATH = PHYSICS_MAIN_OOF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics five-fold reference-count control analysis."
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
    fold_parameters, fold_metrics, oof = control.run_fivefold_control_models(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    # pandas infers the persisted main-model idx as int64, whereas the in-memory
    # Windows integer cast can be int32. Normalize the identifier dtype before
    # the strict Series.equals audit; the identifier values are unchanged.
    oof["idx"] = oof["idx"].astype("int64")
    checks = control.make_design_checks(oof, fold_parameters)
    overall = control.evaluate_oof_models(oof)
    metric_comparison = control.make_metric_comparison(overall)
    paired_differences = control.paired_bootstrap_m2_minus_m1(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 20,
    )
    adjusted_coefficients, adjusted_model_comparison = control.adjusted_inference(oof)

    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)
    overall = replace_field_labels(overall)

    outputs = {
        "physics_reference_count_input_validation.csv": validation,
        "physics_reference_count_design_checks.csv": checks,
        "physics_reference_count_fold_parameters.csv": fold_parameters,
        "physics_reference_count_fold_test_metrics.csv": fold_metrics,
        "physics_reference_count_oof_predictions.csv": oof,
        "physics_reference_count_overall_metrics.csv": overall,
        "physics_reference_count_metric_comparison.csv": metric_comparison,
        "physics_reference_count_paired_bootstrap_m2_vs_m1.csv": paired_differences,
        "physics_reference_count_adjusted_coefficients.csv": adjusted_coefficients,
        "physics_reference_count_adjusted_model_comparison.csv": adjusted_model_comparison,
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
        "control_core_script": str(AI_CONTROL_SCRIPT),
        "control_core_script_sha256": file_sha256(AI_CONTROL_SCRIPT),
        "main_oof_file": str(PHYSICS_MAIN_OOF),
        "main_oof_sha256": file_sha256(PHYSICS_MAIN_OOF),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "models": control.MODEL_FEATURES,
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
    with (output_dir / "physics_reference_count_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    heterogeneity_row = adjusted_coefficients[
        (adjusted_coefficients["model"] == "M2_relevance_heterogeneity_count")
        & (adjusted_coefficients["term"] == "m")
    ]
    print("Physics reference-count control analysis completed.")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF metric comparison:")
    print(metric_comparison.to_string(index=False))
    print("\nAdjusted M2 heterogeneity coefficient:")
    print(heterogeneity_row.to_string(index=False))


if __name__ == "__main__":
    main()
