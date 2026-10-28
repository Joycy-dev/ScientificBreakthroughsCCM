"""Fair Biomedicine-domain comparison of four baselines and the cusp model.

The combined Lasker/Nobel dataset uses ``id`` as its unique record key. The
source-local ``idx`` is retained only for traceability. Every model uses the
same stratified five folds, and all fold-dependent transformations, model
parameters, and F1-optimal classification thresholds are learned from the
training fold only.
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
BIOMEDICINE_MAIN_OOF = BIOMEDICINE_DIR / "biomedicine_train" / "oof_predictions.csv"
BIOMEDICINE_REFERENCE_COUNT_OOF = (
    BIOMEDICINE_DIR
    / "biomedicine_robustness"
    / "biomedicine_reference_count_oof_predictions.csv"
)
AI_COMPARISON_SCRIPT = (
    BIOMEDICINE_DIR.parent / "AI" / "AI_compare" / "ai_model_comparison.py"
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


biomedicine_training = load_module(
    "biomedicine_training_for_model_comparison", BIOMEDICINE_TRAINING_SCRIPT
)
comparison = load_module(
    "ai_model_comparison_core_for_biomedicine", AI_COMPARISON_SCRIPT
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four requested baselines with the Biomedicine cusp model."
    )
    parser.add_argument(
        "--input", type=Path, default=BIOMEDICINE_DIR / "biomedicine_data.csv"
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


def restore_record_identity(oof: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    restored = oof.merge(identity, on="idx", how="left", validate="one_to_one")
    if restored["id"].isna().any():
        raise RuntimeError("Failed to restore one or more Biomedicine record ids.")
    restored = restored.rename(
        columns={"idx": "row_id_internal", "local_idx": "idx"}
    )
    first_columns = [
        "id",
        "idx",
        "row_id_internal",
        "award",
        "num_refs",
        "m",
        "fold",
        "n",
        "log_num_refs",
    ]
    remaining = [column for column in restored.columns if column not in first_columns]
    return restored[first_columns + remaining].sort_values(
        "id", kind="stable"
    ).reset_index(drop=True)


def make_biomedicine_design_checks(
    generic_checks: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    rows = generic_checks.to_dict("records")

    def add(name: str, passed: bool, details: str) -> None:
        rows.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise RuntimeError(f"Model-comparison design check failed: {name}. {details}")

    add(
        "all_ids_present_and_unique",
        bool(oof["id"].notna().all() and oof["id"].is_unique),
        f"rows={len(oof)}, unique_id={oof['id'].nunique()}",
    )
    add(
        "local_idx_retained_not_used_as_key",
        True,
        f"duplicate_local_idx={int(oof['idx'].duplicated().sum())}; unique audit key is id",
    )

    main_oof = pd.read_csv(
        BIOMEDICINE_MAIN_OOF, encoding="utf-8-sig"
    ).sort_values("id", kind="stable").reset_index(drop=True)
    same_id = np.array_equal(oof["id"].to_numpy(), main_oof["id"].to_numpy())
    add("cusp_ids_match_main_model", bool(same_id), "compared by unique id")
    same_fold = same_id and np.array_equal(
        oof["fold"].to_numpy(dtype=int), main_oof["fold"].to_numpy(dtype=int)
    )
    add("same_fold_as_main_model", bool(same_fold), "compared by id")

    cusp_columns = [
        ("same_relevance_as_main_model", "n", "n"),
        (
            "cusp_delta_reproduces_main_model",
            f"delta_{comparison.PROPOSED_MODEL}",
            "delta",
        ),
        (
            "cusp_score_reproduces_main_model",
            f"score_{comparison.PROPOSED_MODEL}",
            "breakthrough_score",
        ),
        (
            "cusp_threshold_reproduces_main_model",
            f"threshold_{comparison.PROPOSED_MODEL}",
            "threshold_used",
        ),
        (
            "cusp_beta_reproduces_main_model",
            f"beta_{comparison.PROPOSED_MODEL}",
            "beta_used",
        ),
    ]
    for check_name, comparison_column, main_column in cusp_columns:
        maximum_difference = (
            float(
                np.max(
                    np.abs(
                        oof[comparison_column].to_numpy(dtype=float)
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

    cusp_prediction_match = same_id and np.array_equal(
        oof[f"prediction_{comparison.PROPOSED_MODEL}"].to_numpy(dtype=int),
        main_oof["prediction"].to_numpy(dtype=int),
    )
    add(
        "cusp_predictions_reproduce_main_model",
        bool(cusp_prediction_match),
        "OOF predictions compared by id",
    )

    main_metrics = comparison.base.evaluate_predictions(
        main_oof["award"].to_numpy(dtype=int),
        main_oof["prediction"].to_numpy(dtype=int),
        main_oof["breakthrough_score"].to_numpy(dtype=float),
    )
    cusp_metrics = overall.loc[
        overall["model"] == comparison.PROPOSED_MODEL
    ].iloc[0]
    maximum_metric_difference = max(
        abs(float(cusp_metrics[metric]) - float(main_metrics[metric]))
        for metric in comparison.CORE_METRICS
    )
    add(
        "cusp_metrics_reproduce_main_model",
        maximum_metric_difference <= 1e-12,
        f"max_abs_difference={maximum_metric_difference:.12g}",
    )

    prior_count = pd.read_csv(
        BIOMEDICINE_REFERENCE_COUNT_OOF, encoding="utf-8-sig"
    ).sort_values("id", kind="stable").reset_index(drop=True)
    same_count_id = np.array_equal(oof["id"].to_numpy(), prior_count["id"].to_numpy())
    add(
        "reference_count_ids_match_prior_M0",
        bool(same_count_id),
        "compared by unique id",
    )
    count_model = "baseline_3_reference_count_only"
    for check_name, comparison_column, prior_column in [
        (
            "reference_count_probability_reproduces_prior_M0",
            f"score_{count_model}",
            "probability_M0_count",
        ),
        (
            "reference_count_threshold_reproduces_prior_M0",
            f"threshold_{count_model}",
            "threshold_M0_count",
        ),
    ]:
        maximum_difference = (
            float(
                np.max(
                    np.abs(
                        oof[comparison_column].to_numpy(dtype=float)
                        - prior_count[prior_column].to_numpy(dtype=float)
                    )
                )
            )
            if same_count_id
            else math.inf
        )
        add(
            check_name,
            maximum_difference <= 1e-12,
            f"max_abs_difference={maximum_difference:.12g}",
        )
    count_prediction_match = same_count_id and np.array_equal(
        oof[f"prediction_{count_model}"].to_numpy(dtype=int),
        prior_count["prediction_M0_count"].to_numpy(dtype=int),
    )
    add(
        "reference_count_predictions_reproduce_prior_M0",
        bool(count_prediction_match),
        "OOF predictions compared by id",
    )
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, validation = biomedicine_training.load_and_validate_data(input_path)
    internal = data.rename(columns={"idx": "local_idx"}).copy()
    internal["idx"] = np.arange(len(internal), dtype=np.int64)
    identity = internal[["idx", "id", "local_idx"]].copy()

    fold_parameters, fold_metrics, internal_oof = comparison.run_comparison(
        internal,
        folds=args.folds,
        random_state=args.random_state,
    )
    overall = comparison.overall_metrics(internal_oof)
    display_table = comparison.make_display_table(overall)
    paired_difference = comparison.paired_bootstrap_cusp_minus_each_baseline(
        internal_oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 30,
    )

    # The reusable core's external-file checks assume that ``idx`` is unique.
    # Run its generic checks on the temporary row key, then audit external
    # Biomedicine results below using the true unique ``id``.
    comparison.MAIN_OOF_PATH = output_dir / "__no_generic_main_oof__.csv"
    comparison.REFERENCE_COUNT_OOF_PATH = output_dir / "__no_generic_count_oof__.csv"
    generic_checks = comparison.design_checks(internal_oof)
    oof = restore_record_identity(internal_oof, identity)

    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)
    overall = replace_field_labels(overall)
    baseline_overall = overall[
        overall["model"].isin(comparison.BASELINE_MODELS)
    ].copy()
    checks = make_biomedicine_design_checks(generic_checks, oof, overall)

    outputs = {
        "biomedicine_four_baselines_input_validation.csv": validation,
        "biomedicine_four_baselines_vs_cusp_design_checks.csv": checks,
        "biomedicine_four_baselines_fold_parameters.csv": fold_parameters[
            fold_parameters["model"].isin(comparison.BASELINE_MODELS)
        ].copy(),
        "biomedicine_four_baselines_fold_test_metrics.csv": fold_metrics[
            fold_metrics["model"].isin(comparison.BASELINE_MODELS)
        ].copy(),
        "biomedicine_four_baselines_overall_metrics.csv": baseline_overall,
        "biomedicine_five_models_fold_parameters.csv": fold_parameters,
        "biomedicine_five_models_fold_test_metrics.csv": fold_metrics,
        "biomedicine_five_models_oof_predictions.csv": oof,
        "biomedicine_five_models_overall_comparison.csv": overall,
        "biomedicine_five_models_display_table.csv": display_table,
        "biomedicine_cusp_vs_each_of_four_baselines_paired_bootstrap.csv": (
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
        "unique_record_key": "id",
        "local_source_index": "idx (retained but not used as a key or feature)",
        "temporary_internal_row_key": (
            "row_id_internal (0-based row number used only because source-local idx repeats)"
        ),
        "biomedicine_training_script": str(BIOMEDICINE_TRAINING_SCRIPT),
        "biomedicine_training_script_sha256": file_sha256(
            BIOMEDICINE_TRAINING_SCRIPT
        ),
        "comparison_core_script": str(AI_COMPARISON_SCRIPT),
        "comparison_core_script_sha256": file_sha256(AI_COMPARISON_SCRIPT),
        "main_oof_file": str(BIOMEDICINE_MAIN_OOF),
        "main_oof_sha256": file_sha256(BIOMEDICINE_MAIN_OOF),
        "prior_reference_count_oof_file": str(BIOMEDICINE_REFERENCE_COUNT_OOF),
        "prior_reference_count_oof_sha256": file_sha256(
            BIOMEDICINE_REFERENCE_COUNT_OOF
        ),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "output_directory": str(output_dir),
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "folds": int(args.folds),
        "stratification_label": "award",
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
    manifest_path = output_dir / "biomedicine_four_baselines_vs_cusp_run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("Biomedicine four-baseline versus cusp comparison completed.")
    print("\nDesign checks:")
    print(checks.to_string(index=False))
    print("\nOOF model comparison:")
    print(display_table.to_string(index=False))
    print("\nPaired bootstrap: proposed cusp minus each baseline")
    print(paired_difference.to_string(index=False))


if __name__ == "__main__":
    main()
