"""Biomedicine-domain robustness analyses for knowledge heterogeneity.

The script performs both reviewer-requested checks:

1. replace the similarity range with ``std_similarity``;
2. control for log(1 + reference count) in nested logistic models.

The combined Lasker/Nobel data are aligned by the unique ``id``. A temporary
integer row key is used only to call the shared computational routines because
the source-local ``idx`` values legitimately repeat across the two awards.
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


FIELD = "Biomedicine"
SCRIPT_DIR = Path(__file__).resolve().parent
BIOMEDICINE_DIR = SCRIPT_DIR.parent
BIOMEDICINE_TRAINING_SCRIPT = (
    BIOMEDICINE_DIR
    / "biomedicine_train"
    / "biomedicine_fivefold_training.py"
)
BIOMEDICINE_MAIN_OOF = (
    BIOMEDICINE_DIR / "biomedicine_train" / "oof_predictions.csv"
)
AI_INDICATOR_SCRIPT = (
    BIOMEDICINE_DIR.parent
    / "AI"
    / "AI_robustness"
    / "ai_heterogeneity_robustness.py"
)
AI_REFERENCE_COUNT_SCRIPT = (
    BIOMEDICINE_DIR.parent
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


training = load_module(
    "biomedicine_training_for_robustness", BIOMEDICINE_TRAINING_SCRIPT
)
indicator = load_module("ai_indicator_robustness_core_for_biomedicine", AI_INDICATOR_SCRIPT)
reference = load_module(
    "ai_reference_count_core_for_biomedicine", AI_REFERENCE_COUNT_SCRIPT
)
# Main-model agreement is audited by unique id below, not by repeated local idx.
reference.MAIN_OOF_PATH = SCRIPT_DIR / "__id_based_audit_used_instead__.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Biomedicine heterogeneity robustness analyses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BIOMEDICINE_DIR / "biomedicine_data.csv",
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


def prepare_internal_data(data: pd.DataFrame) -> pd.DataFrame:
    internal = data.rename(columns={"idx": "local_idx"}).copy()
    internal["idx"] = np.arange(len(internal), dtype=np.int64)
    return internal


def restore_full_oof(frame: pd.DataFrame) -> pd.DataFrame:
    restored = frame.rename(
        columns={"idx": "row_id_internal", "local_idx": "idx"}
    ).copy()
    restored = restored.drop(columns="row_id_internal")
    return restored.sort_values("id", kind="stable").reset_index(drop=True)


def restore_reference_oof(
    frame: pd.DataFrame, internal_data: pd.DataFrame
) -> pd.DataFrame:
    mapping = internal_data[["idx", "id", "local_idx"]].rename(
        columns={"idx": "row_id_internal", "local_idx": "idx"}
    )
    restored = frame.rename(columns={"idx": "row_id_internal"}).merge(
        mapping,
        on="row_id_internal",
        how="left",
        validate="one_to_one",
    )
    restored = restored.drop(columns="row_id_internal")
    first = ["id", "idx", "award", "num_refs", "m", "fold", "n", "log_num_refs"]
    remaining = [column for column in restored.columns if column not in first]
    return restored[first + remaining].sort_values("id", kind="stable").reset_index(
        drop=True
    )


def replace_field_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "field" in result.columns:
        result["field"] = FIELD
    if "Field" in result.columns:
        result["Field"] = FIELD
    if "AI" in result.columns:
        result = result.rename(columns={"AI": FIELD})
    return result


def append_check(
    checks: pd.DataFrame, name: str, passed: bool, details: str
) -> pd.DataFrame:
    if not passed:
        raise RuntimeError(f"Biomedicine robustness audit failed: {name}. {details}")
    row = pd.DataFrame([{"check": name, "status": "PASS", "details": details}])
    return pd.concat([checks, row], ignore_index=True)


def audit_against_main(
    checks: pd.DataFrame,
    candidate: pd.DataFrame,
    compare_delta: bool,
) -> pd.DataFrame:
    main = pd.read_csv(BIOMEDICINE_MAIN_OOF, encoding="utf-8-sig").sort_values(
        "id", kind="stable"
    ).reset_index(drop=True)
    candidate = candidate.sort_values("id", kind="stable").reset_index(drop=True)
    same_id = np.array_equal(
        candidate["id"].astype(str).to_numpy(),
        main["id"].astype(str).to_numpy(),
    )
    checks = append_check(checks, "same_id_as_main_model", bool(same_id), "compared by id")
    same_fold = same_id and np.array_equal(
        candidate["fold"].to_numpy(dtype=int), main["fold"].to_numpy(dtype=int)
    )
    checks = append_check(checks, "same_fold_as_main_model", bool(same_fold), "compared by id")
    max_n_difference = (
        float(np.max(np.abs(candidate["n"].to_numpy() - main["n"].to_numpy())))
        if same_id
        else np.inf
    )
    checks = append_check(
        checks,
        "same_relevance_as_main_model",
        max_n_difference <= 1e-12,
        f"max_abs_difference={max_n_difference:.12g}",
    )
    if compare_delta:
        max_delta_difference = (
            float(
                np.max(
                    np.abs(
                        candidate["delta"].to_numpy(dtype=float)
                        - main["delta"].to_numpy(dtype=float)
                    )
                )
            )
            if same_id
            else np.inf
        )
        checks = append_check(
            checks,
            "range_delta_reproduces_main_model",
            max_delta_difference <= 1e-12,
            f"max_abs_difference={max_delta_difference:.12g}",
        )
        prediction_match = same_id and np.array_equal(
            candidate["prediction"].to_numpy(dtype=int),
            main["prediction"].to_numpy(dtype=int),
        )
        checks = append_check(
            checks,
            "range_predictions_reproduce_main_model",
            bool(prediction_match),
            "OOF predictions compared by id",
        )
    return checks


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_indicator_analysis(
    data: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    folds: int,
    random_state: int,
    bootstrap_reps: int,
    input_path: Path,
) -> None:
    internal = prepare_internal_data(data)
    descriptive, correlations = indicator.indicator_diagnostics(internal)
    range_parameters, range_fold_metrics, range_oof_internal = indicator.run_variant(
        internal, "m", "range_main", folds, random_state
    )
    std_parameters, std_fold_metrics, std_oof_internal = indicator.run_variant(
        internal, "std_similarity", "std_alternative", folds, random_state
    )
    checks = indicator.design_checks(
        range_parameters, std_parameters, range_oof_internal, std_oof_internal
    )
    range_overall = indicator.overall_metrics(range_oof_internal)
    std_overall = indicator.overall_metrics(std_oof_internal)
    comparison = indicator.make_metric_comparison(range_overall, std_overall)
    paired = indicator.paired_bootstrap_differences(
        range_oof_internal,
        std_oof_internal,
        repetitions=bootstrap_reps,
        random_state=random_state + 10,
    )
    range_ci = indicator.base.stratified_bootstrap_confidence_intervals(
        range_oof_internal, bootstrap_reps, random_state + 1
    )
    std_ci = indicator.base.stratified_bootstrap_confidence_intervals(
        std_oof_internal, bootstrap_reps, random_state + 2
    )
    parameter_comparison = range_parameters.merge(
        std_parameters,
        on=["field", "fold"],
        suffixes=("_range", "_std"),
        validate="one_to_one",
    )

    range_oof = restore_full_oof(range_oof_internal)
    std_oof = restore_full_oof(std_oof_internal)
    checks = audit_against_main(checks, range_oof, compare_delta=True)
    range_parameters = replace_field_labels(range_parameters)
    std_parameters = replace_field_labels(std_parameters)
    range_fold_metrics = replace_field_labels(range_fold_metrics)
    std_fold_metrics = replace_field_labels(std_fold_metrics)
    parameter_comparison = replace_field_labels(parameter_comparison)

    outputs = {
        "biomedicine_indicator_input_validation.csv": validation,
        "biomedicine_indicator_descriptive_statistics.csv": descriptive,
        "biomedicine_indicator_correlations.csv": correlations,
        "biomedicine_indicator_design_checks.csv": checks,
        "biomedicine_indicator_fold_parameters_range.csv": range_parameters,
        "biomedicine_indicator_fold_parameters_std.csv": std_parameters,
        "biomedicine_indicator_fold_parameter_comparison.csv": parameter_comparison,
        "biomedicine_indicator_fold_test_metrics_range.csv": range_fold_metrics,
        "biomedicine_indicator_fold_test_metrics_std.csv": std_fold_metrics,
        "biomedicine_indicator_oof_predictions_range.csv": range_oof,
        "biomedicine_indicator_oof_predictions_std.csv": std_oof,
        "biomedicine_indicator_ci_range.csv": range_ci,
        "biomedicine_indicator_ci_std.csv": std_ci,
        "biomedicine_indicator_metrics_comparison.csv": comparison,
        "biomedicine_indicator_paired_bootstrap.csv": paired,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "field": FIELD,
        "unique_record_key": "id",
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(folds),
        "random_state": int(random_state),
        "bootstrap_repetitions": int(bootstrap_reps),
        "training_script": str(BIOMEDICINE_TRAINING_SCRIPT),
        "indicator_core_script": str(AI_INDICATOR_SCRIPT),
        "variants": {
            "range_main": "max_similarity - min_similarity",
            "std_alternative": "std_similarity",
        },
        "output_files": list(outputs),
    }
    with (output_dir / "biomedicine_indicator_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Biomedicine indicator robustness completed.")
    print(checks.to_string(index=False))
    print(comparison.to_string(index=False))


def run_reference_count_analysis(
    data: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    folds: int,
    random_state: int,
    bootstrap_reps: int,
    input_path: Path,
) -> None:
    internal = prepare_internal_data(data)
    fold_parameters, fold_metrics, oof_internal = reference.run_fivefold_control_models(
        internal, folds, random_state
    )
    checks = reference.make_design_checks(oof_internal, fold_parameters)
    overall = reference.evaluate_oof_models(oof_internal)
    metric_comparison = reference.make_metric_comparison(overall)
    paired = reference.paired_bootstrap_m2_minus_m1(
        oof_internal, bootstrap_reps, random_state + 20
    )
    adjusted_coefficients, adjusted_model_comparison = reference.adjusted_inference(
        oof_internal
    )
    oof = restore_reference_oof(oof_internal, internal)
    checks = audit_against_main(checks, oof, compare_delta=False)
    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)
    overall = replace_field_labels(overall)

    outputs = {
        "biomedicine_reference_count_input_validation.csv": validation,
        "biomedicine_reference_count_design_checks.csv": checks,
        "biomedicine_reference_count_fold_parameters.csv": fold_parameters,
        "biomedicine_reference_count_fold_test_metrics.csv": fold_metrics,
        "biomedicine_reference_count_oof_predictions.csv": oof,
        "biomedicine_reference_count_overall_metrics.csv": overall,
        "biomedicine_reference_count_metric_comparison.csv": metric_comparison,
        "biomedicine_reference_count_paired_bootstrap_m2_vs_m1.csv": paired,
        "biomedicine_reference_count_adjusted_coefficients.csv": adjusted_coefficients,
        "biomedicine_reference_count_adjusted_model_comparison.csv": (
            adjusted_model_comparison
        ),
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "field": FIELD,
        "unique_record_key": "id",
        "main_oof_file": str(BIOMEDICINE_MAIN_OOF),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(folds),
        "random_state": int(random_state),
        "bootstrap_repetitions": int(bootstrap_reps),
        "models": reference.MODEL_FEATURES,
        "reference_count_transform": "log(1 + num_refs)",
        "standardization": "training-fold StandardScaler; unchanged on test fold",
        "training_script": str(BIOMEDICINE_TRAINING_SCRIPT),
        "reference_core_script": str(AI_REFERENCE_COUNT_SCRIPT),
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
    with (output_dir / "biomedicine_reference_count_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    target = adjusted_coefficients[
        (adjusted_coefficients["model"] == "M2_relevance_heterogeneity_count")
        & (adjusted_coefficients["term"] == "m")
    ]
    print("Biomedicine reference-count control completed.")
    print(checks.to_string(index=False))
    print(metric_comparison.to_string(index=False))
    print(target.to_string(index=False))


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data, validation = training.load_and_validate_data(input_path)
    run_indicator_analysis(
        data,
        validation,
        output_dir,
        args.folds,
        args.random_state,
        args.bootstrap_reps,
        input_path,
    )
    run_reference_count_analysis(
        data,
        validation,
        output_dir,
        args.folds,
        args.random_state,
        args.bootstrap_reps,
        input_path,
    )


if __name__ == "__main__":
    main()
