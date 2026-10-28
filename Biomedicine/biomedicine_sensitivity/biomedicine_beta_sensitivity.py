"""Sensitivity analysis for the Biomedicine-domain cusp scale coefficient beta.

The analysis reuses the established sensitivity core while adapting record
identity handling to the combined Lasker/Nobel dataset. ``id`` is the unique
audit key; the source-local ``idx`` is retained for traceability but is never
used as a feature or unique key. For every training fold, the fold-specific
beta is multiplied by 0.50, 0.75, 1.00, 1.25, and 1.50, and the empirical F1
threshold is learned on that training fold only.
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


FIELD = "Biomedicine"
SCRIPT_DIR = Path(__file__).resolve().parent
BIOMEDICINE_DIR = SCRIPT_DIR.parent
BIOMEDICINE_TRAINING_SCRIPT = (
    BIOMEDICINE_DIR / "biomedicine_train" / "biomedicine_fivefold_training.py"
)
AI_SENSITIVITY_SCRIPT = (
    BIOMEDICINE_DIR.parent / "AI" / "AI_sensitivity" / "ai_beta_sensitivity.py"
)
MAIN_OOF_PATH = BIOMEDICINE_DIR / "biomedicine_train" / "oof_predictions.csv"
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


biomedicine = load_module(
    "biomedicine_training_for_sensitivity", BIOMEDICINE_TRAINING_SCRIPT
)
sensitivity = load_module(
    "ai_beta_sensitivity_core_for_biomedicine", AI_SENSITIVITY_SCRIPT
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Biomedicine-domain sensitivity analysis for cusp beta."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BIOMEDICINE_DIR / "biomedicine_data.csv",
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


def replace_field(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "field" in result.columns:
        result["field"] = FIELD
    if "Field" in result.columns:
        result["Field"] = FIELD
    return result


def restore_record_identity(
    sensitivity_oof: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    restored = sensitivity_oof.merge(
        identity,
        on="idx",
        how="left",
        validate="many_to_one",
    )
    if restored["id"].isna().any():
        raise RuntimeError("Failed to restore one or more Biomedicine record ids.")
    restored = restored.rename(
        columns={"idx": "row_id_internal", "local_idx": "idx"}
    )
    preferred = [
        "id",
        "idx",
        "row_id_internal",
        "award",
        "m",
        "beta_multiplier",
        "fold",
        "n",
        "beta_base",
        "beta_used",
        "delta",
        "breakthrough_score",
        "empirical_delta_threshold",
        "prediction",
    ]
    return restored[preferred].sort_values(
        ["beta_multiplier", "id"], kind="stable"
    ).reset_index(drop=True)


def make_biomedicine_design_checks(
    generic_checks: pd.DataFrame,
    restored_oof: pd.DataFrame,
    overall: pd.DataFrame,
    main_oof_path: Path,
) -> pd.DataFrame:
    rows = generic_checks.to_dict("records")

    def add(name: str, passed: bool, details: str) -> None:
        rows.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Sensitivity design check failed: {name}. {details}")

    expected_multipliers = sorted(restored_oof["beta_multiplier"].unique())
    expected_n = int(restored_oof["id"].nunique())
    group_counts = restored_oof.groupby("beta_multiplier")["id"].size()
    unique_counts = restored_oof.groupby("beta_multiplier")["id"].nunique()
    add(
        "id_complete_and_unique_within_multiplier",
        bool(
            (group_counts == expected_n).all()
            and (unique_counts == expected_n).all()
        ),
        f"multipliers={expected_multipliers}, rows_per_multiplier={group_counts.to_dict()}",
    )
    add(
        "local_idx_retained_not_used_as_key",
        True,
        (
            f"duplicate_local_idx_per_multiplier="
            f"{restored_oof[restored_oof['beta_multiplier'] == 1.0]['idx'].duplicated().sum()}; "
            "unique audit key is id"
        ),
    )
    beta_error = float(
        np.max(
            np.abs(
                restored_oof["beta_used"].to_numpy(dtype=float)
                - restored_oof["beta_base"].to_numpy(dtype=float)
                * restored_oof["beta_multiplier"].to_numpy(dtype=float)
            )
        )
    )
    add(
        "beta_multiplier_applied_exactly",
        beta_error <= 1e-12,
        f"max_abs_error={beta_error:.12g}",
    )

    main_oof = pd.read_csv(main_oof_path, encoding="utf-8-sig").sort_values(
        "id", kind="stable"
    ).reset_index(drop=True)
    baseline = restored_oof.loc[
        restored_oof["beta_multiplier"] == 1.0
    ].sort_values("id", kind="stable").reset_index(drop=True)
    same_id = np.array_equal(baseline["id"].to_numpy(), main_oof["id"].to_numpy())
    add("baseline_ids_match_main_model", bool(same_id), "compared by unique id")

    same_fold = same_id and np.array_equal(
        baseline["fold"].to_numpy(dtype=int),
        main_oof["fold"].to_numpy(dtype=int),
    )
    add("baseline_folds_match_main_model", bool(same_fold), "compared by id")

    for check_name, sensitivity_column, main_column in [
        ("baseline_n_reproduces_main_model", "n", "n"),
        ("baseline_beta_reproduces_main_model", "beta_used", "beta_used"),
        ("baseline_delta_reproduces_main_model", "delta", "delta"),
        (
            "baseline_threshold_reproduces_main_model",
            "empirical_delta_threshold",
            "threshold_used",
        ),
        (
            "baseline_score_reproduces_main_model",
            "breakthrough_score",
            "breakthrough_score",
        ),
    ]:
        maximum_difference = (
            float(
                np.max(
                    np.abs(
                        baseline[sensitivity_column].to_numpy(dtype=float)
                        - main_oof[main_column].to_numpy(dtype=float)
                    )
                )
            )
            if same_id
            else math.inf
        )
        add(
            check_name,
            maximum_difference <= 1e-12,
            f"max_abs_difference={maximum_difference:.12g}",
        )

    prediction_match = same_id and np.array_equal(
        baseline["prediction"].to_numpy(dtype=int),
        main_oof["prediction"].to_numpy(dtype=int),
    )
    add(
        "baseline_predictions_reproduce_main_model",
        bool(prediction_match),
        "OOF predictions compared by id",
    )

    main_metrics = sensitivity.base.evaluate_predictions(
        main_oof["award"].to_numpy(dtype=int),
        main_oof["prediction"].to_numpy(dtype=int),
        main_oof["breakthrough_score"].to_numpy(dtype=float),
    )
    baseline_metrics = overall.loc[overall["beta_multiplier"] == 1.0].iloc[0]
    maximum_metric_difference = max(
        abs(float(baseline_metrics[metric]) - float(main_metrics[metric]))
        for metric in sensitivity.CORE_METRICS
    )
    add(
        "baseline_metrics_reproduce_main_model",
        maximum_metric_difference <= 1e-12,
        f"max_abs_difference={maximum_metric_difference:.12g}",
    )
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, validation = biomedicine.load_and_validate_data(input_path)
    internal = data.rename(columns={"idx": "local_idx"}).copy()
    internal["idx"] = np.arange(len(internal), dtype=np.int64)
    identity = internal[["idx", "id", "local_idx"]].copy()

    (
        fold_parameters,
        fold_metrics,
        sensitivity_oof_internal,
        overall,
        zero_boundary,
    ) = sensitivity.run_sensitivity(
        internal,
        folds=args.folds,
        random_state=args.random_state,
        multipliers=args.multipliers,
    )
    fold_parameters = replace_field(fold_parameters)
    fold_metrics = replace_field(fold_metrics)
    overall = replace_field(overall)
    zero_boundary = replace_field(zero_boundary)
    summary = sensitivity.make_metric_sensitivity_summary(overall)

    # The generic check uses an AI-specific main-OOF path. Disable that branch,
    # then perform stricter Biomedicine checks below using the unique ``id``.
    sensitivity.MAIN_OOF_PATH = output_dir / "__no_generic_main_oof__.csv"
    generic_checks = sensitivity.make_design_checks(
        overall, sensitivity_oof_internal
    )
    restored_oof = restore_record_identity(sensitivity_oof_internal, identity)
    checks = make_biomedicine_design_checks(
        generic_checks, restored_oof, overall, MAIN_OOF_PATH
    )

    outputs = {
        "biomedicine_sensitivity_input_validation.csv": validation,
        "biomedicine_sensitivity_design_checks.csv": checks,
        "biomedicine_sensitivity_fold_parameters.csv": fold_parameters,
        "biomedicine_sensitivity_fold_test_metrics.csv": fold_metrics,
        "biomedicine_sensitivity_oof_predictions.csv": restored_oof,
        "biomedicine_sensitivity_overall_metrics.csv": overall,
        "biomedicine_sensitivity_metric_range_summary.csv": summary,
        "biomedicine_sensitivity_theoretical_zero_boundary_metrics.csv": zero_boundary,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_directory": str(output_dir),
        "field": FIELD,
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "unique_record_key": "id",
        "local_source_index": "idx (retained but not used as a key or feature)",
        "temporary_internal_row_key": (
            "row_id_internal (0-based row number used only because source-local idx repeats)"
        ),
        "folds": int(args.folds),
        "stratification_label": "award",
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
        "biomedicine_training_script": str(BIOMEDICINE_TRAINING_SCRIPT),
        "biomedicine_training_script_sha256": file_sha256(
            BIOMEDICINE_TRAINING_SCRIPT
        ),
        "ai_sensitivity_core": str(AI_SENSITIVITY_SCRIPT),
        "ai_sensitivity_core_sha256": file_sha256(AI_SENSITIVITY_SCRIPT),
        "main_oof_file": str(MAIN_OOF_PATH),
        "main_oof_sha256": file_sha256(MAIN_OOF_PATH),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
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
    manifest_path = output_dir / "biomedicine_sensitivity_run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Biomedicine beta sensitivity analysis completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF metrics across beta multipliers:")
    print(overall[["beta_multiplier", *sensitivity.CORE_METRICS]].to_string(index=False))
    print("\nTheoretical zero-boundary metrics at baseline beta:")
    print(
        zero_boundary[["delta_threshold", *sensitivity.CORE_METRICS]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
