"""Read-only paired project-cluster stability diagnostics for round-two models.

The temporal holdout has already been used in model selection. These percentile
bootstrap intervals are conditional diagnostics for these frozen predictions
and observed project clusters, not independent Kaggle/test-set guarantees.
No model is trained and no existing artifact is changed.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "round2" / "stability"
SEED = 42
REPLICATES = 500
TARGET_UNSEEN_SHARE = 0.631615
MODELS = ("v1", "anchor", "trend")
INPUTS = {
    "v1": ROOT / "outputs/experiments/selected_temporal_valid.npy",
    "anchor": ROOT / "outputs/round2/neighborhood/neighborhood_hedonic_anchor_temporal_predictions.npy",
    "trend": ROOT / "outputs/round2/neighborhood/neighborhood_hedonic_trend_temporal_predictions.npy",
    "v1_indices": ROOT / "outputs/experiments/selected_temporal_indices.npy",
    "neighborhood_indices": ROOT / "outputs/round2/neighborhood/temporal_indices.npy",
}
CAUTION = (
    "本時間 holdout 已用於候選選型；這是對固定預測與已觀察建案的條件診斷，"
    "不是獨立 Kaggle 成績，也不是完全未觸及的驗證。"
    "區間不修正先前模型選型的樂觀偏差，不包含未來市場漂移、跨建案地區/時間相關性、"
    "重新訓練的不確定性或未見建案比例本身的不確定性。"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_strings(series):
    values = series.astype("string").fillna("__MISSING__").str.strip()
    return values.mask(values.isin(["", "nan", "NaN", "None", "null", "<NA>"]), "__MISSING__").astype(str)


def main():
    started = time.perf_counter()
    # Reproduce the established exact-row deduplication with read-only cache
    # operations, avoiding loader helpers that could rebuild cache artifacts.
    raw = pd.read_pickle(ROOT / "data_cache/train.pkl")
    features = pd.read_pickle(ROOT / "data_cache/train_features.pkl")
    schema = json.loads((ROOT / "reports/feature_schema.json").read_text(encoding="utf-8"))
    if schema["feature_source_sha256"] != sha256(ROOT / "src/house_features.py"):
        raise ValueError("Feature schema source hash is stale")
    if len(raw) != len(features) or list(features.columns) != schema["columns"]:
        raise ValueError("Raw/feature row or column alignment differs")
    keep = ~raw.duplicated(subset=[column for column in raw if column != "編號"]).to_numpy()
    y = raw["總價元"].to_numpy(dtype="float64")[keep]
    X = features.loc[keep].reset_index(drop=True)
    if len(X) != 455255:
        raise ValueError("Unexpected deduplicated training count")
    days = X["trade_day_index"].to_numpy()
    cutoff_days = (pd.Timestamp("2024-04-01") - pd.Timestamp("2000-01-01")).days
    end_days = (pd.Timestamp("2024-08-10") - pd.Timestamp("2000-01-01")).days
    fit_indices = np.flatnonzero(np.isfinite(days) & (days < cutoff_days))
    expected_indices = np.flatnonzero(np.isfinite(days) & (days >= cutoff_days) & (days <= end_days))
    indices = np.load(INPUTS["v1_indices"], allow_pickle=False)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("V1 prediction indices do not match the fixed temporal split")
    if not np.array_equal(indices, np.load(INPUTS["neighborhood_indices"], allow_pickle=False)):
        raise ValueError("Candidate temporal prediction indices differ from V1")
    if len(indices) != 58206:
        raise ValueError("Expected 58,206 temporal holdout records")

    holdout = X.iloc[indices].reset_index(drop=True)
    target = y[indices]
    predictions = np.column_stack([np.load(INPUTS[name], allow_pickle=False) for name in MODELS])
    if predictions.shape != (len(target), 3) or not np.isfinite(predictions).all():
        raise ValueError("Prediction shape or finite-value check failed")
    errors = np.abs(target[:, None] - predictions)
    train_projects = set(clean_strings(X.iloc[fit_indices]["project_name"])) - {"__MISSING__"}
    project_names = clean_strings(holdout["project_name"])
    missing_project = project_names.eq("__MISSING__").to_numpy()
    seen = project_names.isin(train_projects).to_numpy()
    if int(seen.sum()) != 32227:
        raise ValueError("Seen/unseen definition differs from existing temporal evaluation")

    cities = clean_strings(holdout["address_city"])
    districts = clean_strings(holdout["address_district"])
    roads = clean_strings(holdout["address_road"])
    cluster_keys = [
        json.dumps(["missing_project_geo", city, district, road], ensure_ascii=False)
        if missing else json.dumps(["project", project], ensure_ascii=False)
        for project, missing, city, district, road
        in zip(project_names, missing_project, cities, districts, roads)
    ]
    codes, unique_keys = pd.factorize(pd.Series(cluster_keys, dtype="string"), sort=True)
    group_count = np.bincount(codes).astype("int64")
    group_seen_count = np.bincount(codes, weights=seen.astype("int64"))
    if not np.all((group_seen_count == 0) | (group_seen_count == group_count)):
        raise ValueError("A project cluster unexpectedly spans both seen/unseen strata")
    group_seen = group_seen_count > 0
    group_error_sums = np.column_stack([
        np.bincount(codes, weights=errors[:, column], minlength=len(group_count))
        for column in range(3)
    ])
    group_missing = np.bincount(codes, weights=missing_project.astype("int64")) > 0
    strata = {"seen": np.flatnonzero(group_seen), "unseen": np.flatnonzero(~group_seen)}
    if any(len(groups) == 0 for groups in strata.values()):
        raise ValueError("Both project strata must have at least one cluster")

    overall_rows = []
    point_regular = errors.mean(axis=0)
    point_seen = errors[seen].mean(axis=0)
    point_unseen = errors[~seen].mean(axis=0)
    point_weighted = (1 - TARGET_UNSEEN_SHARE) * point_seen + TARGET_UNSEEN_SHARE * point_unseen
    for column, name in enumerate(MODELS):
        overall_rows.append({
            "model": name, "rows": len(target), "mae": float(point_regular[column]),
            "seen_project_rows": int(seen.sum()), "seen_project_mae": float(point_seen[column]),
            "unseen_project_rows": int((~seen).sum()), "unseen_project_mae": float(point_unseen[column]),
            "target_unseen_share": TARGET_UNSEEN_SHARE, "reweighted_mae": float(point_weighted[column]),
        })

    dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(holdout["trade_day_index"], unit="D")
    months = dates.dt.strftime("%Y-%m").to_numpy()
    monthly_rows = []
    for month in ("2024-04", "2024-05", "2024-06", "2024-07", "2024-08"):
        month_mask = months == month
        for segment, mask in (("all", month_mask), ("seen", month_mask & seen),
                              ("unseen", month_mask & ~seen)):
            for column, name in enumerate(MODELS):
                monthly_rows.append({
                    "month": month, "segment": segment, "model": name, "rows": int(mask.sum()),
                    "mae": float(errors[mask, column].mean()) if mask.any() else None,
                    "delta_vs_v1": float((errors[mask, 0] - errors[mask, column]).mean()) if mask.any() else None,
                })

    # Pair all predictions by using the very same sampled project clusters
    # for every model and every contrast in each bootstrap replicate.
    rng = np.random.default_rng(SEED)
    bootstrap_regular = np.empty((REPLICATES, 3), dtype="float64")
    bootstrap_weighted = np.empty_like(bootstrap_regular)
    for replicate in range(REPLICATES):
        stratum_error_sums = {}
        stratum_row_counts = {}
        for label, groups in strata.items():
            sampled_groups = groups[rng.integers(0, len(groups), size=len(groups))]
            stratum_error_sums[label] = group_error_sums[sampled_groups].sum(axis=0)
            stratum_row_counts[label] = int(group_count[sampled_groups].sum())
        total_errors = stratum_error_sums["seen"] + stratum_error_sums["unseen"]
        total_rows = stratum_row_counts["seen"] + stratum_row_counts["unseen"]
        bootstrap_regular[replicate] = total_errors / total_rows
        bootstrap_weighted[replicate] = (
            (1 - TARGET_UNSEEN_SHARE) * stratum_error_sums["seen"] / stratum_row_counts["seen"]
            + TARGET_UNSEEN_SHARE * stratum_error_sums["unseen"] / stratum_row_counts["unseen"]
        )

    summary_rows = []
    replicate_rows = {"replicate": np.arange(1, REPLICATES + 1)}
    contrasts = (("v1_vs_anchor", 0, 1), ("v1_vs_trend", 0, 2), ("anchor_vs_trend", 1, 2))
    for contrast, left, right in contrasts:
        for metric, boot, point in (
            ("mae_improvement", bootstrap_regular, point_regular),
            ("target_mix_mae_improvement", bootstrap_weighted, point_weighted),
        ):
            values = boot[:, left] - boot[:, right]
            lower, upper = np.quantile(values, [0.025, 0.975])
            row = {
                "contrast": contrast, "baseline": MODELS[left], "candidate": MODELS[right],
                "metric": metric, "point_delta": float(point[left] - point[right]),
                "bootstrap_mean_delta": float(values.mean()),
                "ci95_lower": float(lower), "ci95_upper": float(upper),
                "positive_replicate_fraction": float((values > 0).mean()),
                "replicates": REPLICATES, "seed": SEED,
            }
            summary_rows.append(row)
            replicate_rows[f"{contrast}_{metric}"] = values

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(OUT / "overall_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(OUT / "monthly_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "bootstrap_summary.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(OUT / "bootstrap_replicates.csv", index=False)
    cluster_frame = pd.DataFrame({
        "cluster_key": unique_keys, "rows": group_count, "seen_project": group_seen,
        "missing_project_geo_fallback": group_missing,
        **{f"{name}_absolute_error_sum": group_error_sums[:, i] for i, name in enumerate(MODELS)},
    })
    cluster_frame.to_csv(OUT / "cluster_error_sums.csv", index=False, encoding="utf-8-sig")
    report = {
        "purpose": "Conditional stability diagnostics on the model-selection temporal holdout",
        "caution": CAUTION,
        "temporal_cutoff": "2024-04-01", "temporal_end_inclusive": "2024-08-10",
        "august_is_partial_month": True,
        "fit_rows": len(fit_indices), "holdout_rows": len(target),
        "observed_unseen_row_share": float((~seen).mean()),
        "target_unseen_row_share": TARGET_UNSEEN_SHARE,
        "target_share_basis": "User-specified rounded 63.1615%; held fixed in each bootstrap replicate",
        "cluster_definition": "project_name; missing project_name uses (address_city, address_district, address_road)",
        "bootstrap_method": "Paired cluster bootstrap, stratified by seen/unseen; sample the original number of groups with replacement within each stratum; aggregate group error sums and row counts",
        "ordinary_mae_denominator": "Total resampled row count across both strata",
        "reweighted_mae": "(1 - 0.631615) * seen_MAE + 0.631615 * unseen_MAE",
        "interval_method": "2.5th and 97.5th percentiles of 500 paired cluster bootstrap deltas",
        "delta_direction": "baseline absolute error minus candidate absolute error; positive means candidate improvement",
        "seed": SEED, "replicates": REPLICATES,
        "clusters": {
            "total": len(group_count), "seen": len(strata["seen"]), "unseen": len(strata["unseen"]),
            "missing_project_rows": int(missing_project.sum()),
            "missing_project_geo_clusters": int(group_missing.sum()),
            "all_three_geo_fields_missing_rows": int((missing_project & cities.eq("__MISSING__").to_numpy()
                    & districts.eq("__MISSING__").to_numpy() & roads.eq("__MISSING__").to_numpy()).sum()),
            "largest_cluster_rows": int(group_count.max()),
            "largest_cluster_row_share": float(group_count.max() / len(target)),
        },
        "overall_metrics": overall_rows, "bootstrap_summary": summary_rows,
        "monthly_metrics": monthly_rows,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in INPUTS.values()},
        "script_sha256": sha256(Path(__file__)),
        "feature_source_sha256": schema["feature_source_sha256"],
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "model_training_performed": False, "external_data_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "stability_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"clusters": report["clusters"], "elapsed_seconds": report["elapsed_seconds"],
                      "bootstrap_summary": summary_rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


