from __future__ import annotations

import argparse
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover - allows running without scipy
    fisher_exact = None


ALIASES = {
    "award": ["award", "label", "breakthrough", "target", "class", "y"],
    "m": ["m", "M", "heterogeneity", "knowledge_heterogeneity", "raw_m"],
    "m_std": ["m_std", "M_std", "scaled_m", "standardized_m", "m_scaled"],
    "n": ["n", "N", "relevance", "knowledge_relevance", "raw_n"],
    "n_std": ["n_std", "N_std", "scaled_n", "standardized_n", "n_scaled"],
    "delta": ["delta", "Delta", "Δ", "discriminant", "discriminant_result"],
    "Pd": ["Pd", "P_d", "deep_ratio", "deep_citation_ratio"],
    "Pm": ["Pm", "P_m", "medium_ratio", "moderate_citation_ratio"],
    "Ps": ["Ps", "P_s", "P_l", "shallow_ratio", "shallow_citation_ratio"],
}


def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("δ", "delta").replace("Δ", "delta")
    return re.sub(r"[^a-z0-9]+", "", text)


def fuzzy_score(column: str, alias: str) -> float:
    col_norm = normalize_name(column)
    alias_norm = normalize_name(alias)
    if col_norm == alias_norm:
        return 1.0
    if alias_norm and (alias_norm in col_norm or col_norm in alias_norm):
        return 0.92
    return SequenceMatcher(None, col_norm, alias_norm).ratio()


def identify_columns(columns: list[str]) -> tuple[dict[str, str | None], pd.DataFrame]:
    mapping = {}
    rows = []
    for canonical, aliases in ALIASES.items():
        candidates = []
        for column in columns:
            score = max(fuzzy_score(column, alias) for alias in aliases)
            candidates.append((score, column))
        candidates.sort(reverse=True)
        best_score, best_column = candidates[0]
        mapping[canonical] = best_column if best_score >= 0.60 else None
        for rank, (score, column) in enumerate(candidates[:3], start=1):
            rows.append(
                {
                    "canonical": canonical,
                    "rank": rank,
                    "candidate": column,
                    "score": score,
                    "selected": rank == 1 and best_score >= 0.60,
                }
            )
    return mapping, pd.DataFrame(rows)


def load_table(path: Path, sheet: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if sheet is None:
            sheet = 0
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def prepare_data(path: Path, sheet: str | int | None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = load_table(path, sheet)
    raw.columns = [str(column).strip() for column in raw.columns]
    mapping, matches = identify_columns(list(raw.columns))

    required = ["award"]
    missing = [name for name in required if mapping[name] is None]
    if missing:
        raise ValueError(f"Could not identify required field(s): {missing}")

    data = pd.DataFrame(index=raw.index)
    for canonical, source in mapping.items():
        if source is not None:
            data[canonical] = pd.to_numeric(raw[source], errors="coerce")

    if "delta" not in data or data["delta"].isna().all():
        if mapping.get("m_std") is None and mapping.get("m") is not None:
            data["m_std"] = -data["m"]
            mapping["m_std"] = "derived: -m"
        if "m_std" not in data or "n_std" not in data:
            raise ValueError(
                "Delta/discriminant is absent and m_std/n_std are insufficient "
                "to recompute Delta."
            )
        data["delta"] = 8.0 * data["m_std"] ** 3 + 27.0 * data["n_std"] ** 2
        mapping["delta"] = "derived: 8*m_std^3 + 27*n_std^2"

    data["award"] = pd.to_numeric(data["award"], errors="coerce").round()
    before = len(data)
    data = data.dropna(subset=["award", "delta"]).copy()
    dropped = before - len(data)
    data["award"] = data["award"].astype(int)
    if not set(data["award"].unique()).issubset({0, 1}):
        raise ValueError("The award/label field must contain only 0 and 1.")

    if "m_std" in data and "n_std" in data:
        data["delta_recomputed"] = 8.0 * data["m_std"] ** 3 + 27.0 * data["n_std"] ** 2
        data["delta_formula_error"] = data["delta"] - data["delta_recomputed"]

    data.attrs["dropped_incomplete_rows"] = dropped
    return data, mapping, matches


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) == 0:
        raise ValueError("No finite Delta values are available.")
    if len(unique) == 1:
        return unique
    mids = (unique[:-1] + unique[1:]) / 2.0
    epsilon = max(np.ptp(unique), 1.0) * 1e-9
    return np.concatenate(([unique[0] - epsilon], mids, [unique[-1] + epsilon]))


def learn_f1_optimal_threshold(delta: np.ndarray, labels: np.ndarray) -> dict:
    best = {
        "threshold": float(np.median(delta)),
        "F1": -np.inf,
        "Precision": -np.inf,
        "Recall": -np.inf,
        "Accuracy": -np.inf,
    }
    for threshold in candidate_thresholds(delta):
        predicted = (delta <= threshold).astype(int)
        precision = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        f1 = f1_score(labels, predicted, zero_division=0)
        accuracy = accuracy_score(labels, predicted)
        key = (f1, recall, precision, -threshold)
        best_key = (best["F1"], best["Recall"], best["Precision"], -best["threshold"])
        if key > best_key:
            best = {
                "threshold": float(threshold),
                "F1": float(f1),
                "Precision": float(precision),
                "Recall": float(recall),
                "Accuracy": float(accuracy),
            }
    return best


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.inf
    return numerator / denominator


def evaluate_predictions(
    labels: np.ndarray,
    predicted: np.ndarray,
    score: np.ndarray,
    threshold: float | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    score = np.asarray(score, dtype=float)

    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))

    selected = predicted == 1
    unselected = predicted == 0
    award_rate_below = float(labels[selected].mean()) if selected.any() else np.nan
    award_rate_above = float(labels[unselected].mean()) if unselected.any() else np.nan
    rd = award_rate_below - award_rate_above
    rr = safe_ratio(award_rate_below, award_rate_above)
    odds_ratio = safe_ratio(tp / fp if fp else math.inf, fn / tn if tn else math.inf)
    fisher_p = np.nan
    if fisher_exact is not None:
        odds_ratio, fisher_p = fisher_exact([[tp, fp], [fn, tn]])

    top_n = max(1, int(np.ceil(0.20 * len(labels))))
    top_indices = np.argsort(-score, kind="stable")[:top_n]
    top_awards = int(labels[top_indices].sum())
    overall_award_rate = float(labels.mean())
    top_award_rate = float(labels[top_indices].mean())
    lift20 = safe_ratio(top_award_rate, overall_award_rate)
    recall20 = safe_ratio(top_awards, int(labels.sum()))

    return {
        "threshold": threshold,
        "N": len(labels),
        "award_n": int(labels.sum()),
        "control_n": int((labels == 0).sum()),
        "award_rate": overall_award_rate,
        "predicted_breakthrough_n": int(selected.sum()),
        "predicted_control_n": int(unselected.sum()),
        "award_rate_delta_below_threshold": award_rate_below,
        "award_rate_delta_above_threshold": award_rate_above,
        "RD": rd,
        "RR": rr,
        "OR": odds_ratio,
        "Fisher_p": fisher_p,
        "Accuracy": accuracy_score(labels, predicted),
        "Precision": precision_score(labels, predicted, zero_division=0),
        "Recall": recall_score(labels, predicted, zero_division=0),
        "F1": f1_score(labels, predicted, zero_division=0),
        "Lift_at_20pct": lift20,
        "Recall_at_20pct": recall20,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def run_cross_validation(
    data: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = data["award"].to_numpy(dtype=int)
    delta = data["delta"].to_numpy(dtype=float)

    min_class_count = int(pd.Series(labels).value_counts().min())
    if min_class_count < 2:
        raise ValueError("Each class must contain at least two samples for CV.")
    actual_splits = min(n_splits, min_class_count)
    if actual_splits != n_splits:
        print(
            f"Warning: requested {n_splits} folds, but the minority class has "
            f"only {min_class_count} samples. Using {actual_splits} folds."
        )

    splitter = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_predicted = np.zeros(len(data), dtype=int)
    oof_score = np.zeros(len(data), dtype=float)
    fold_rows = []
    sample_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(delta, labels), start=1):
        train_delta = delta[train_idx]
        train_labels = labels[train_idx]
        test_delta = delta[test_idx]
        test_labels = labels[test_idx]

        learned = learn_f1_optimal_threshold(train_delta, train_labels)
        threshold = learned["threshold"]
        test_predicted = (test_delta <= threshold).astype(int)
        test_score = threshold - test_delta

        oof_predicted[test_idx] = test_predicted
        oof_score[test_idx] = test_score

        fold_metrics = evaluate_predictions(
            test_labels,
            test_predicted,
            test_score,
            threshold=threshold,
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "train_award_n": int(train_labels.sum()),
                "test_award_n": int(test_labels.sum()),
                "train_F1_at_selected_threshold": learned["F1"],
                "train_Precision_at_selected_threshold": learned["Precision"],
                "train_Recall_at_selected_threshold": learned["Recall"],
                **fold_metrics,
            }
        )

        for idx, pred, score in zip(test_idx, test_predicted, test_score):
            sample_rows.append(
                {
                    "row_index": int(idx),
                    "fold": fold,
                    "award": int(labels[idx]),
                    "delta": float(delta[idx]),
                    "threshold_used": threshold,
                    "prediction": int(pred),
                    "breakthrough_score": float(score),
                }
            )

    overall_oof = evaluate_predictions(labels, oof_predicted, oof_score, threshold=np.nan)
    overall = pd.DataFrame([{"evaluation": "OOF pooled", **overall_oof}])
    folds = pd.DataFrame(fold_rows)
    samples = pd.DataFrame(sample_rows).sort_values("row_index")
    return overall, folds, samples


def summarize_thresholds(folds: pd.DataFrame, full_threshold: float) -> pd.DataFrame:
    thresholds = folds["threshold"].astype(float)
    return pd.DataFrame(
        [
            {
                "full_data_threshold": full_threshold,
                "cv_threshold_mean": thresholds.mean(),
                "cv_threshold_std": thresholds.std(ddof=1),
                "cv_threshold_min": thresholds.min(),
                "cv_threshold_median": thresholds.median(),
                "cv_threshold_max": thresholds.max(),
            }
        ]
    )


def describe_core_variables(data: pd.DataFrame) -> pd.DataFrame:
    available = [
        variable
        for variable in ["m", "m_std", "n", "n_std", "delta", "Pd", "Pm", "Ps"]
        if variable in data.columns
    ]
    rows = []
    for group_name, group in [
        ("Overall", data),
        ("Breakthrough_award_1", data[data["award"] == 1]),
        ("Control_award_0", data[data["award"] == 0]),
    ]:
        for variable in available:
            series = group[variable].dropna()
            rows.append(
                {
                    "group": group_name,
                    "variable": variable,
                    "N": int(series.count()),
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(ddof=1),
                    "min": series.min(),
                    "max": series.max(),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-domain five-fold threshold experiment for breakthrough-paper "
            "identification using Delta <= threshold."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV/TSV/XLSX file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <input_stem>_threshold_results.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name or zero-based sheet index. Defaults to first sheet.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=20260612,
        help="Random seed for stratified CV splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.with_name(f"{input_path.stem}_threshold_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data, mapping, matches = prepare_data(input_path, sheet)
    labels = data["award"].to_numpy(dtype=int)
    delta = data["delta"].to_numpy(dtype=float)

    full_learned = learn_f1_optimal_threshold(delta, labels)
    full_threshold = full_learned["threshold"]
    full_predicted = (delta <= full_threshold).astype(int)
    full_score = full_threshold - delta
    full_metrics = evaluate_predictions(
        labels,
        full_predicted,
        full_score,
        threshold=full_threshold,
    )
    full_metrics = pd.DataFrame([{"evaluation": "Full-data fit", **full_metrics}])

    oof_metrics, fold_metrics, sample_predictions = run_cross_validation(
        data,
        n_splits=args.folds,
        random_state=args.random_state,
    )
    threshold_summary = summarize_thresholds(fold_metrics, full_threshold)
    descriptive = describe_core_variables(data)

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "field_mapping": mapping,
        "rows_after_cleaning": len(data),
        "dropped_incomplete_rows": data.attrs.get("dropped_incomplete_rows", 0),
        "award_counts": data["award"].value_counts().sort_index().to_dict(),
        "prediction_rule": "award_hat = 1 if Delta <= threshold else 0",
        "threshold_learning": (
            "Exhaustive one-dimensional search over candidate thresholds; "
            "candidate thresholds are adjacent Delta midpoints; selected threshold "
            "maximizes training F1, with ties broken by Recall, Precision, and "
            "smaller threshold."
        ),
        "cv": {
            "requested_folds": args.folds,
            "random_state": args.random_state,
            "splitter": "StratifiedKFold(shuffle=True)",
        },
    }

    matches.to_csv(output_dir / "field_fuzzy_matches.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(output_dir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([oof_metrics, full_metrics], ignore_index=True).to_csv(
        output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig"
    )
    threshold_summary.to_csv(
        output_dir / "threshold_summary.csv", index=False, encoding="utf-8-sig"
    )
    sample_predictions.to_csv(
        output_dir / "sample_level_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nField mapping")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    print("\nThreshold summary")
    print(threshold_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOverall metrics")
    print(
        pd.concat([oof_metrics, full_metrics], ignore_index=True).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
