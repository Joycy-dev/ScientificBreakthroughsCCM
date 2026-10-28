"""Leakage-free five-fold training for the Biomedicine-domain cusp indicator.

The combined Lasker/Nobel dataset uses ``id`` as its unique record key. The
source-local ``idx`` may repeat and is retained only for traceability. All
sample-dependent quantities are estimated from each training fold only and
then applied unchanged to the corresponding held-out fold.
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


FIELD = "Biomedicine"
REQUIRED_COLUMNS = [
    "id",
    "idx",
    "title",
    "award",
    "num_all",
    "num_d",
    "num_m",
    "num_s",
    "num_refs",
    "mean_similarity",
    "max_similarity",
    "min_similarity",
    "std_similarity",
    "m",
]
NUMERIC_COLUMNS = [
    "idx",
    "award",
    "num_all",
    "num_d",
    "num_m",
    "num_s",
    "num_refs",
    "mean_similarity",
    "max_similarity",
    "min_similarity",
    "std_similarity",
    "m",
]
COUNT_COLUMNS = ["num_all", "num_d", "num_m", "num_s", "num_refs"]


def load_training_core(script_dir: Path):
    core_path = (
        script_dir.parent.parent
        / "AI"
        / "AI_train"
        / "ai_fivefold_training.py"
    )
    if not core_path.exists():
        raise FileNotFoundError(f"AI training core not found: {core_path}")
    specification = importlib.util.spec_from_file_location(
        "ai_training_core_for_biomedicine", core_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load AI training core: {core_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, core_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Leakage-free five-fold training for the Biomedicine dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "biomedicine_data.csv",
        help="Input CSV file. Defaults to ../biomedicine_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Output directory. Defaults to the script directory.",
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


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = [str(column).strip() for column in data.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()
    data["id"] = data["id"].astype("string").str.strip()
    data["title"] = data["title"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "details": details}
        )
        if not passed:
            raise ValueError(f"Input validation failed: {name}. {details}")

    record_check(
        "required_values_complete",
        not data[REQUIRED_COLUMNS].isna().any().any(),
        f"missing_cells={int(data[REQUIRED_COLUMNS].isna().sum().sum())}",
    )
    record_check(
        "id_unique",
        bool(data["id"].is_unique),
        f"duplicate_id={int(data['id'].duplicated().sum())}",
    )
    local_idx_duplicates = int(data["idx"].duplicated().sum())
    record_check(
        "local_idx_retained_not_used_as_key",
        True,
        f"duplicate_idx={local_idx_duplicates}; unique key is id",
    )
    sources = data["id"].str.split("_", n=1).str[0]
    source_counts = sources.value_counts().sort_index().to_dict()
    record_check(
        "expected_award_sources_present",
        set(source_counts) == {"lasker", "nobel"},
        f"source_counts={source_counts}",
    )

    labels = set(data["award"].dropna().astype(int).unique())
    record_check(
        "binary_award_labels",
        labels == {0, 1},
        f"labels={sorted(labels)}",
    )
    record_check(
        "numeric_values_finite",
        bool(np.isfinite(data[NUMERIC_COLUMNS].to_numpy(dtype=float)).all()),
        "all numeric inputs finite",
    )

    count_values = data[COUNT_COLUMNS].to_numpy(dtype=float)
    integer_like = np.isclose(count_values, np.round(count_values)).all()
    nonnegative_counts = (data[COUNT_COLUMNS] >= 0).all().all()
    record_check(
        "counts_nonnegative_integers",
        bool(integer_like and nonnegative_counts),
        "num_all/num_d/num_m/num_s/num_refs checked",
    )
    record_check(
        "num_all_positive",
        bool((data["num_all"] > 0).all()),
        f"zero_or_negative={int((data['num_all'] <= 0).sum())}",
    )
    count_difference = data["num_d"] + data["num_m"] + data["num_s"] - data["num_all"]
    record_check(
        "citation_counts_reconcile",
        bool((count_difference == 0).all()),
        f"mismatched_rows={int((count_difference != 0).sum())}",
    )

    expected_m = data["max_similarity"] - data["min_similarity"]
    maximum_m_error = float((data["m"] - expected_m).abs().max())
    record_check(
        "m_matches_similarity_range",
        maximum_m_error <= 1e-8,
        f"max_abs_error={maximum_m_error:.12g}",
    )

    data["idx"] = data["idx"].astype(int)
    data["award"] = data["award"].astype(int)
    for column in COUNT_COLUMNS:
        data[column] = data[column].astype(int)

    # Row-level proportions use only each paper's own citation counts.
    data["P_d"] = data["num_d"] / data["num_all"]
    data["P_m"] = data["num_m"] / data["num_all"]
    data["P_s"] = data["num_s"] / data["num_all"]
    proportion_error = (data[["P_d", "P_m", "P_s"]].sum(axis=1) - 1.0).abs()
    maximum_proportion_error = float(proportion_error.max())
    record_check(
        "citation_proportions_sum_to_one",
        maximum_proportion_error <= 1e-12,
        f"max_abs_error={maximum_proportion_error:.12g}",
    )

    class_counts = data["award"].value_counts().sort_index()
    record_check(
        "enough_cases_for_five_folds",
        int(class_counts.min()) >= 5,
        f"class_counts={class_counts.to_dict()}",
    )

    source_label_counts = (
        pd.DataFrame({"source": sources, "award": data["award"]})
        .groupby(["source", "award"])
        .size()
        .to_dict()
    )
    record_check(
        "source_label_counts_documented",
        True,
        f"source_label_counts={source_label_counts}",
    )
    return data, pd.DataFrame(checks)


def replace_field_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "field" in result.columns:
        result["field"] = FIELD
    if "Field" in result.columns:
        result["Field"] = FIELD
    if "AI" in result.columns:
        result = result.rename(columns={"AI": FIELD})
    return result


def save_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core, core_path = load_training_core(script_dir)
    data, validation = load_and_validate_data(input_path)
    fold_parameters, fold_metrics, oof = core.run_fivefold_training(
        data,
        folds=args.folds,
        random_state=args.random_state,
    )
    oof = oof.sort_values("id", kind="stable").reset_index(drop=True)
    fold_parameters = replace_field_labels(fold_parameters)
    fold_metrics = replace_field_labels(fold_metrics)

    overall_metrics = core.evaluate_predictions(
        oof["award"].to_numpy(dtype=int),
        oof["prediction"].to_numpy(dtype=int),
        oof["breakthrough_score"].to_numpy(dtype=float),
    )
    overall = pd.DataFrame(
        [{"field": FIELD, "evaluation": "OOF pooled", **overall_metrics}]
    )
    confidence_intervals = core.stratified_bootstrap_confidence_intervals(
        oof,
        repetitions=args.bootstrap_reps,
        random_state=args.random_state + 1,
    )
    parameter_summary = replace_field_labels(
        core.summarize_fold_parameters(fold_parameters)
    )
    table5 = replace_field_labels(core.make_table5(fold_parameters))
    table6 = replace_field_labels(core.make_table6(overall_metrics, confidence_intervals))

    outputs = {
        "input_validation.csv": validation,
        "fold_parameters.csv": fold_parameters,
        "fold_test_metrics.csv": fold_metrics,
        "fold_parameter_summary.csv": parameter_summary,
        "oof_predictions.csv": oof,
        "overall_oof_metrics.csv": overall,
        "metric_confidence_intervals.csv": confidence_intervals,
        "table5_biomedicine_thresholds.csv": table5,
        "table6_biomedicine_metrics.csv": table6,
    }
    for filename, frame in outputs.items():
        save_csv(frame, output_dir / filename)

    source_counts = data["id"].str.split("_", n=1).str[0].value_counts().to_dict()
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_directory": str(output_dir),
        "field": FIELD,
        "sample_n": int(len(data)),
        "award_n": int(data["award"].sum()),
        "control_n": int((data["award"] == 0).sum()),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "unique_record_key": "id",
        "local_source_index": "idx (retained but not used as a unique key or feature)",
        "folds": int(args.folds),
        "stratification_label": "award",
        "random_state": int(args.random_state),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "training_core": str(core_path),
        "training_core_sha256": file_sha256(core_path),
        "proportion_definitions": {
            "P_d": "num_d / num_all",
            "P_m": "num_m / num_all",
            "P_s": "num_s / num_all",
        },
        "fold_only_parameters": [
            "weight_d",
            "weight_m",
            "weight_s",
            "beta",
            "F1-optimal threshold",
        ],
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

    print("Biomedicine five-fold training completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(table5.to_string(index=False))
    print(table6.to_string(index=False))


if __name__ == "__main__":
    main()
