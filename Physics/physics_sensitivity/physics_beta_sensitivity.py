"""Physics-domain sensitivity analysis for the cusp scale coefficient beta.

For each training fold, the dispersion-calibrated beta is multiplied by 0.50,
0.75, 1.00, 1.25, and 1.50. A separate F1-optimal empirical threshold is
learned for every fold and multiplier and then applied unchanged to the held-out
fold. The theoretical boundary Delta = 0 is evaluated separately at 1.00 beta.
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
AI_SENSITIVITY_SCRIPT = (
    PHYSICS_DIR.parent
    / "AI"
    / "AI_sensitivity"
    / "ai_beta_sensitivity.py"
)
DEFAULT_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)


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
    "physics_training_for_beta_sensitivity", PHYSICS_TRAINING_SCRIPT
)
sensitivity = load_module("ai_beta_sensitivity_core", AI_SENSITIVITY_SCRIPT)
sensitivity.MAIN_OOF_PATH = PHYSICS_MAIN_OOF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-domain sensitivity analysis for beta."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PHYSICS_DIR / "physics_data.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
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
    (
        fold_parameters,
        fold_metrics,
        sensitivity_oof,
        overall,
        zero_boundary,
    ) = sensitivity.run_sensitivity(
        data,
        folds=args.folds,
        random_state=args.random_state,
        multipliers=args.multipliers,
    )

    # Normalize persisted identifiers before comparison with the main-model OOF.
    sensitivity_oof["idx"] = sensitivity_oof["idx"].astype("int64")
    summary = sensitivity.make_metric_sensitivity_summary(overall)
    checks = sensitivity.make_design_checks(overall, sensitivity_oof)

    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)
    overall = replace_field_labels(overall)
    zero_boundary = replace_field_labels(zero_boundary)

    outputs = {
        "physics_sensitivity_input_validation.csv": validation,
        "physics_sensitivity_design_checks.csv": checks,
        "physics_sensitivity_fold_parameters.csv": fold_parameters,
        "physics_sensitivity_fold_test_metrics.csv": fold_metrics,
        "physics_sensitivity_oof_predictions.csv": sensitivity_oof,
        "physics_sensitivity_overall_metrics.csv": overall,
        "physics_sensitivity_metric_range_summary.csv": summary,
        "physics_sensitivity_theoretical_zero_boundary_metrics.csv": zero_boundary,
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
        "sensitivity_core_script": str(AI_SENSITIVITY_SCRIPT),
        "sensitivity_core_script_sha256": file_sha256(AI_SENSITIVITY_SCRIPT),
        "main_oof_file": str(PHYSICS_MAIN_OOF),
        "main_oof_sha256": file_sha256(PHYSICS_MAIN_OOF),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "random_state": int(args.random_state),
        "beta_multipliers": sorted(set(float(value) for value in args.multipliers)),
        "empirical_threshold": (
            "F1-optimal Delta threshold learned independently within each "
            "training fold for every beta multiplier."
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
    with (output_dir / "physics_sensitivity_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Physics beta sensitivity analysis completed.")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF metrics across beta multipliers:")
    print(
        overall[["beta_multiplier", *sensitivity.CORE_METRICS]].to_string(
            index=False
        )
    )
    print("\nTheoretical zero-boundary metrics at baseline beta:")
    print(
        zero_boundary[["delta_threshold", *sensitivity.CORE_METRICS]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
