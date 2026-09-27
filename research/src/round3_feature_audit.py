"""Read-only round-three feature/reference audit; does not train prediction models.

Writes only this task's outputs/round3/audit/feature_* artifacts. Descriptive
within-project regressions assess regional time-coefficient stability; they are
never used to alter a prediction, select a submission, or infer test labels.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "3"

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_common import load_data, full_training_data, temporal_indices
from prepare_data import ROOT, ID, sha256
from round2_neighborhood import MISSING, PARAMETERS, group_keys, matrices, _predict_reference_all

OUT = ROOT / "outputs" / "round3" / "audit"
EPOCH = pd.Timestamp("2000-01-01")
ACTIONS = []


def action(name, **details):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "action": name, **details}
    ACTIONS.append(record)
    print(json.dumps(record, ensure_ascii=True), flush=True)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def csv(frame, name):
    assert name.startswith("feature_") and name.endswith(".csv")
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return str((OUT / name).relative_to(ROOT)).replace("\\", "/")


def summary(values):
    a = np.asarray(values, dtype="float64")
    a = a[np.isfinite(a)]
    if not len(a):
        return {"rows": 0}
    return {"rows": len(a), "mean": np.mean(a), "min": np.min(a), "p10": np.quantile(a, .1),
            "p25": np.quantile(a, .25), "median": np.median(a), "p75": np.quantile(a, .75),
            "p90": np.quantile(a, .9), "p95": np.quantile(a, .95), "p99": np.quantile(a, .99),
            "max": np.max(a)}


def date_strings(x):
    return pd.to_datetime(x["trade_day_index"], unit="D", origin="2000-01-01", errors="coerce")


def joined(x, columns):
    return x[columns].astype(str).agg("|".join, axis=1)


def seen_masks(fit, query):
    answer = {}
    for label, columns in {
        "name": ["project_name"],
        "city_name": ["address_city", "project_name"],
        "region_name": ["address_city", "address_district", "project_name"],
        "region_style_name": ["address_city", "address_district", "building_type", "main_use", "project_name"],
    }.items():
        known = set(joined(fit.loc[fit.project_name != MISSING], columns))
        answer[label] = ((query.project_name != MISSING) & joined(query, columns).isin(known)).to_numpy()
    return answer


def distribution(fit, test, column):
    a = fit[column].value_counts().rename("fit_rows")
    b = test[column].value_counts().rename("test_rows")
    result = pd.concat([a, b], axis=1).fillna(0).reset_index().rename(columns={"index": "category"})
    if "category" not in result:
        result = result.rename(columns={result.columns[0]: "category"})
    result["fit_share"] = result.fit_rows / len(fit)
    result["test_share"] = result.test_rows / len(test)
    result["share_delta"] = result.test_share - result.fit_share
    result["test_to_fit_share_ratio"] = result.test_share / result.fit_share.replace(0, np.nan)
    p, q = result.fit_share.to_numpy(), result.test_share.to_numpy()
    mid = (p + q) / 2
    js = .5 * np.sum(p[p > 0] * np.log(p[p > 0] / mid[p > 0]))
    js += .5 * np.sum(q[q > 0] * np.log(q[q > 0] / mid[q > 0]))
    result["feature"] = column
    return result, {"total_variation_distance": float(np.abs(p-q).sum()/2),
                    "jensen_shannon_divergence_nats": float(js),
                    "test_unseen_category_rows": int(result.loc[result.fit_rows == 0, "test_rows"].sum())}


def descriptive_hedonic(frame, target, model, end_day, years):
    """Fixed-form within-project robust regression for audit, not a predictor."""
    dates = frame.trade_day_index.to_numpy(dtype="float64")
    area = frame.building_area.to_numpy(dtype="float64")
    up = np.divide(target, area, out=np.full(len(frame), np.nan), where=area > 0)
    lo, hi = model["fit_report"]["reference_unit_price_bounds"]
    eligible = (np.isfinite(dates) & (dates <= end_day) & (dates >= end_day-years*365.25)
                & (area >= 15) & (area <= 1500) & (up >= lo) & (up <= hi)
                & (frame.project_name.to_numpy() != MISSING))
    ix = np.flatnonzero(eligible)
    _, h, _ = matrices(frame, model["fill_values"])
    work = pd.DataFrame(h[ix], columns=["log_area", "parking", "floor", "year"])
    work["target"] = np.log(up[ix])
    work["group"] = group_keys(frame)["project"].iloc[ix].to_numpy()
    work["city"] = frame.address_city.iloc[ix].to_numpy()
    work["day"] = dates[ix]
    counts = work.groupby("group").target.transform("size")
    work = work.loc[counts >= PARAMETERS["hedonic_project_minimum"]].copy()
    names = ["log_area", "parking", "floor", "year", "target"]
    centered = work[names] - work.groupby("group")[names].transform("mean")
    records = []
    for city in ["ALL_REGIONS", *sorted(work.city.unique())]:
        mask = np.ones(len(work), dtype=bool) if city == "ALL_REGIONS" else work.city.eq(city).to_numpy()
        sub = work.loc[mask]
        n, groups = len(sub), sub.group.nunique()
        rec = {"city": city, "lookback_years": years, "end_date": str((EPOCH+pd.Timedelta(days=end_day)).date()),
               "rows": n, "project_groups": groups, "status": "insufficient_support"}
        if n < 300 or groups < 20:
            records.append(rec)
            continue
        x = centered.loc[mask, names[:4]].to_numpy(dtype="float64")
        y = centered.loc[mask, "target"].to_numpy(dtype="float64")
        weights = np.ones(n)
        beta = np.zeros(4)
        for _ in range(6):
            gram = x.T @ (weights[:, None] * x) + np.eye(4) * 1e-5
            beta = np.linalg.solve(gram, x.T @ (weights*y))
            residual = y-x@beta
            weights = np.minimum(1., .10 / np.maximum(np.abs(residual), 1e-8))
        # Cluster sandwich approximation; selection and future market uncertainty are not covered.
        bread = np.linalg.inv(x.T @ (weights[:, None]*x) + np.eye(4)*1e-5)
        scores = pd.DataFrame(x * (weights*residual)[:, None]).groupby(
            sub.group.to_numpy(), sort=False).sum().to_numpy()
        cov = bread @ (scores.T @ scores) @ bread * groups/(groups-1) * (n-1)/(n-4)
        se = float(np.sqrt(max(cov[3, 3], 0.)))
        day_span = sub.groupby("group").day.agg(["min", "max"])
        condition = float(np.linalg.cond(gram))
        rec.update({
            "status": "estimated", "annual_log_coefficient": float(beta[3]),
            "annual_multiplier_change": float(np.expm1(beta[3])) if abs(beta[3]) < 5 else None,
            "approx_project_cluster_se": se, "approx_ci95_lower": float(beta[3]-1.96*se),
            "approx_ci95_upper": float(beta[3]+1.96*se), "condition_number": condition,
            "median_project_day_span": float((day_span["max"]-day_span["min"]).median()),
            "log_area_coefficient": float(beta[0]), "parking_coefficient": float(beta[1]),
            "floor_per10_coefficient": float(beta[2]),
            "global_coefficient_outside_approx_ci": bool(
                model["beta"][3] < beta[3]-1.96*se or model["beta"][3] > beta[3]+1.96*se),
            "warning": "high_condition_number" if condition > 1e8 else "",
        })
        records.append(rec)
    return records


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed_paths = [
        "src/house_features.py", "src/round2_neighborhood.py",
        "outputs/round2/neighborhood/final_reference.pkl",
        "outputs/round2/neighborhood/final_reference.json",
        "models/round2/release_config.json",
        "outputs/submission_v2.csv", "outputs/submission_v2_anchor.csv", "outputs/submission_v2_mix.csv",
    ]
    before = {p: sha256(ROOT/p) for p in fixed_paths}
    action("start_read_only_audit", no_training=True, no_external_data=True,
           no_public_score_optimization=True)
    d = load_data()
    full, target = full_training_data(d)
    test = d["X_test"]
    metadata = json.loads((ROOT/"outputs/round2/neighborhood/final_reference.json").read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == d["metadata"]["source_sha256"]
    assert metadata["feature_source_sha256"] == before["src/house_features.py"]
    assert metadata["script_sha256"] == before["src/round2_neighborhood.py"]
    assert metadata["pickle_sha256"] == before["outputs/round2/neighborhood/final_reference.pkl"]
    with (ROOT/metadata["pickle_file"]).open("rb") as handle:
        model = pickle.load(handle)
    assert len(full) == model["fit_report"]["fitting_rows"] == 467221
    end_day = model["fit_report"]["hedonic"]["maximum_fit_day_index"]
    result = {
        "audit_type": "descriptive_and_saved_model_diagnostics_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "source_sha256": d["metadata"]["source_sha256"],
        "reference_sha256": metadata["pickle_sha256"], "saved_global_annual_log_coefficient": model["beta"][3],
        "saved_global_annual_multiplier_change": np.expm1(model["beta"][3]),
        "constraints": {"no_external_property_data": True, "no_government_answer_lookup": True,
                        "no_test_labels": True, "no_public_score_tuning": True,
                        "no_predictive_model_training": True, "no_v1_v2_modification": True},
        "files": {}, "actions": ACTIONS,
    }

    dates_rows, monthly_rows, missing_rows, address_examples = [], [], [], []
    bug_rows = []
    for split in ("train", "valid", "test"):
        x = d[f"X_{split}"]
        dates = date_strings(x)
        valid_dates = dates.dropna()
        days = x.trade_day_index.to_numpy(dtype="float64")
        rec = {"split": split, "rows": len(x), "date_min": str(valid_dates.min().date()),
               "date_max": str(valid_dates.max().date()), "invalid_dates": int(dates.isna().sum()),
               "after_final_fit_rows": int(np.sum(days > end_day)),
               "after_final_fit_fraction": float(np.mean(days > end_day)),
               "days_beyond_fit": summary(np.maximum(days-end_day, 0))}
        dates_rows.append(rec)
        counts = dates.dt.to_period("M").astype(str).value_counts().sort_index()
        monthly_rows.extend({"split": split, "month": month, "rows": int(count)}
                            for month, count in counts.items())
        for column in ("project_name", "address_raw", "address_city", "address_district", "address_road",
                       "address_section", "address_locality", "district", "building_type", "main_use"):
            missing_rows.append({"split": split, "feature": column,
                                 "missing_rows": int(x[column].eq(MISSING).sum()),
                                 "missing_rate": float(x[column].eq(MISSING).mean()),
                                 "distinct_nonmissing": int(x.loc[x[column] != MISSING, column].nunique())})
        raw_d = x.district.astype(str).str.replace("台", "臺", regex=False).str.strip()
        parsed = x.address_district.astype(str).str.replace("台", "臺", regex=False).str.strip()
        mismatch = (raw_d != MISSING) & (parsed != MISSING) & raw_d.ne(parsed)
        truncated = mismatch & pd.Series([a.startswith(b) and len(a) > len(b)
                                          for a, b in zip(raw_d, parsed)], index=x.index)
        known_early_suffix = truncated & parsed.str.endswith(("市", "鎮", "鄉")) & raw_d.str.endswith("區")
        bug_rows.append({
            "split": split, "district_disagreement_rows": int(mismatch.sum()),
            "parsed_prefix_truncation_rows": int(truncated.sum()),
            "early_suffix_truncation_rows": int(known_early_suffix.sum()),
            "early_suffix_truncation_rate": float(known_early_suffix.mean()),
            "land_parcel_rows": int(x.address_is_land_parcel.eq(1).sum()),
            "road_starts_with_leftover_district_suffix_rows": int(x.address_road.str.startswith("區").sum()),
        })
        group = pd.DataFrame({"raw_district": raw_d[mismatch], "parsed_district": parsed[mismatch],
                              "city": x.loc[mismatch, "address_city"]}).value_counts().reset_index(name="rows")
        group.insert(0, "split", split)
        if len(group):
            address_examples.append(group)
    result["date_span"] = dates_rows
    result["address_parsing"] = {
        "concrete_bug": "Lazy district regex stops at an internal 市/鎮 suffix (新市區/前鎮區/平鎮區), leaving 區 in road.",
        "counts": bug_rows,
        "interpretation": "Semantic extraction bug, generally systematic in train/test; by itself not evidence of label leakage or the public-score cause.",
        "missing_city_risk": "Reference region keys accept missing city/district; missing coverage is quantified separately.",
    }
    result["files"]["monthly_counts"] = csv(pd.DataFrame(monthly_rows), "feature_monthly_counts.csv")
    result["files"]["missingness"] = csv(pd.DataFrame(missing_rows), "feature_missingness.csv")
    result["files"]["district_disagreements"] = csv(pd.concat(address_examples, ignore_index=True)
        if address_examples else pd.DataFrame(), "feature_district_disagreements.csv")
    action("date_and_address_audit_complete", address_bug_counts=bug_rows)

    distribution_frames = []
    result["distribution_shift"] = {}
    for column in ("address_city", "address_district", "building_type", "main_use", "transaction_type"):
        table, stats = distribution(full, test, column)
        distribution_frames.append(table)
        result["distribution_shift"][column] = stats
    result["files"]["category_distribution"] = csv(pd.concat(distribution_frames, ignore_index=True),
                                                  "feature_category_distribution.csv")
    fit_idx, temporal_idx = temporal_indices(d, "2024-04-01", end="2024-08-10")
    comparisons = [
        ("original_valid", d["X_train"], d["X_valid"]),
        ("temporal_valid", d["X_train"].iloc[fit_idx], d["X_train"].iloc[temporal_idx]),
        ("test", full, test),
    ]
    result["project_seen_definitions"] = []
    test_seen = None
    for label, fit, query in comparisons:
        seen = seen_masks(fit, query)
        if label == "test":
            test_seen = seen
        result["project_seen_definitions"].append({
             "split": label, "query_rows": len(query),
            "seen_project_rows": int(seen["name"].sum()),
            "missing_project_rows": int(query.project_name.eq(MISSING).sum()),
            "nonmissing_new_project_rows": int((~seen["name"] & query.project_name.ne(MISSING).to_numpy()).sum()),
            "seen_project_rate": float(seen["name"].mean()),
            "missing_project_rate": float(query.project_name.eq(MISSING).mean()),
            "nonmissing_new_project_rate": float((~seen["name"] & query.project_name.ne(MISSING).to_numpy()).mean()),
            **{f"unseen_{name}_rows": int((~mask).sum()) for name, mask in seen.items()},
            **{f"unseen_{name}_rate": float((~mask).mean()) for name, mask in seen.items()},
            "seen_name_but_unseen_city_name_rows": int((seen["name"] & ~seen["city_name"]).sum()),
            "seen_name_but_unseen_region_name_rows": int((seen["name"] & ~seen["region_name"]).sum()),
            "seen_name_but_unseen_region_style_name_rows": int((seen["name"] & ~seen["region_style_name"]).sum()),
        })
    valid_projects = full.loc[full.project_name != MISSING,
                              ["project_name", "address_city", "address_district"]].copy()
    valid_projects["region"] = joined(valid_projects, ["address_city", "address_district"])
    collisions = valid_projects.groupby("project_name").agg(
        fit_rows=("address_city", "size"), cities=("address_city", "nunique"),
        regions=("region", "nunique"))
    city_labels = valid_projects.groupby("project_name").address_city.agg(lambda s: "|".join(sorted(set(s))))
    collisions["city_names"] = city_labels
    collisions["test_rows"] = test.project_name.value_counts().reindex(collisions.index, fill_value=0)
    collisions = collisions.loc[(collisions.cities > 1) | (collisions.regions > 1)].reset_index()
    result["project_name_collisions"] = {
        "cross_city_names": int((collisions.cities > 1).sum()),
        "cross_region_names": len(collisions),
        "test_rows_with_cross_city_name": int(collisions.loc[collisions.cities > 1, "test_rows"].sum()),
        "test_rows_with_cross_region_name": int(collisions.test_rows.sum()),
        "reference_guard": "Neighborhood project key already contains city, district, building type and main use, so identical names across cities are not directly pooled.",
        "metric_grouping_issue": "Current seen/unseen scoring uses name alone; it can label a new geographic project as seen.",
    }
    result["files"]["project_collisions"] = csv(collisions.sort_values("test_rows", ascending=False),
                                              "feature_project_collisions.csv")
    action("distribution_and_project_identity_complete",
           seen_definitions=result["project_seen_definitions"])

    cache = ROOT / "outputs" / "round2" / "final_predictions"
    manifest = json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == before["models/round2/release_config.json"]
    assert manifest["test_source_sha256"] == d["metadata"]["source_sha256"]["test"]
    baseline = .75*np.load(cache/"catboost_raw_test.npy") + .25*np.load(cache/"lightgbm_l1_unit63_test.npy")
    assert baseline.shape == (len(test),) and np.isfinite(baseline).all() and (baseline >= 0).all()
    with threadpool_limits(limits=3):
        predictions, diagnostics, diagnostic_report = _predict_reference_all(model, test, baseline)
    for name, values in predictions.items():
        saved = np.load(cache/f"{name}_test.npy")
        if not np.array_equal(values, saved):
            raise AssertionError(f"Saved model replay differs: {name}, max={np.max(np.abs(values-saved))}")
    anchor, trend = predictions["neighborhood_hedonic_anchor"], predictions["neighborhood_hedonic_trend"]
    confidence = diagnostics.confidence.to_numpy()
    days_future = np.maximum(test.trade_day_index.to_numpy(dtype="float64")-end_day, 0)
    trend_delta = trend-anchor
    trend_pct = trend_delta/np.maximum(anchor, 1)
    multiplier = np.exp(model["beta"][3]*np.minimum(days_future, 366)/365.25)
    exact_delta = (1-confidence)*baseline*(multiplier-1)
    assert np.allclose(trend_delta, exact_delta, atol=1e-7, rtol=1e-12)
    row_diag = diagnostics.copy()
    row_diag.insert(0, ID, d["ids_test"])
    for column in ("address_city", "address_district", "project_name", "building_type", "main_use",
                   "building_area", "trade_day_index"):
        row_diag[column] = test[column].to_numpy()
    row_diag["month"] = date_strings(test).dt.strftime("%Y-%m")
    row_diag["seen_name"] = test_seen["name"]
    row_diag["project_group"] = np.where(test.project_name.eq(MISSING), "missing_project",
        np.where(test_seen["name"], "seen_project", "nonmissing_new_project"))
    row_diag["seen_region_project"] = test_seen["region_name"]
    row_diag["baseline"] = baseline
    row_diag["anchor"] = anchor
    row_diag["trend"] = trend
    row_diag["trend_delta_TWD"] = trend_delta
    row_diag["trend_delta_fraction_of_anchor"] = trend_pct
    row_diag["future_days"] = days_future
    row_diag["missing_region"] = (test.address_city.eq(MISSING)|test.address_district.eq(MISSING)).to_numpy()
    result["files"]["test_reference_diagnostics"] = csv(row_diag, "feature_test_reference_diagnostics.csv")
    coverage = []
    for grouping in ("source", "seen_name", "project_group", "address_city", "month"):
        for label, indices in row_diag.groupby(grouping, dropna=False).groups.items():
            sub = row_diag.loc[indices]
            coverage.append({
                "grouping": grouping, "group": str(label), "rows": len(sub),
                "mean_confidence": sub.confidence.mean(), "median_confidence": sub.confidence.median(),
                "confidence_below_0_1_rate": sub.confidence.lt(.1).mean(),
                "confidence_above_0_5_rate": sub.confidence.gt(.5).mean(),
                "base_fallback_rate": sub.source.eq("base_fallback").mean(),
                "mean_future_days": sub.future_days.mean(), "mean_trend_delta_TWD": sub.trend_delta_TWD.mean(),
                "median_trend_delta_TWD": sub.trend_delta_TWD.median(),
                "mean_trend_delta_fraction": sub.trend_delta_fraction_of_anchor.mean(),
                "p90_trend_delta_fraction": sub.trend_delta_fraction_of_anchor.quantile(.9),
                "mean_abs_anchor_correction_TWD": np.abs(sub.anchor-sub.baseline).mean(),
            })
    result["files"]["coverage_by_group"] = csv(pd.DataFrame(coverage), "feature_reference_coverage.csv")
    result["reference_coverage"] = {
        "saved_predictions_reproduced_exactly": True,
         "source_counts": diagnostics.source.value_counts().to_dict(),
        "three_project_groups": [r for r in coverage if r["grouping"] == "project_group"],
        "confidence": summary(confidence),
        "confidence_below_0_1_rows": int((confidence < .1).sum()),
        "confidence_above_0_5_rows": int((confidence > .5).sum()),
        "missing_region_query_rows": int(row_diag.missing_region.sum()),
        "missing_region_with_reference_rows": int((row_diag.missing_region & row_diag.source.ne("base_fallback")).sum()),
        "nearest_distance": summary(diagnostics.nearest_distance),
        "reference_dispersion": summary(diagnostics.reference_dispersion),
        "final_fit_in_sample_coverage_not_compared": "Self-neighbors would make confidence optimistically incomparable to unseen test rows.",
    }
    result["trend_effect"] = {
        "formula": "(1-confidence) * baseline * (exp(beta_time * min(future_days,366)/365.25)-1)",
        "extra_trend_added_to_baseline_share_only": True,
        "days_beyond_final_fit": summary(days_future), "base_multiplier": summary(multiplier),
        "delta_TWD": summary(trend_delta), "delta_fraction_of_anchor": summary(trend_pct),
        "anchor_minus_baseline_TWD": summary(anchor-baseline),
        "trend_minus_baseline_TWD": summary(trend-baseline),
        "note": "Anchor already has local date normalization. This measures only the additional baseline future-trend effect, not all date dependence.",
    }
    action("reference_replay_and_confidence_complete", confidence=summary(confidence),
           delta_TWD=summary(trend_delta))

    region_rows = []
    for end, years, window in [
        (end_day, 4., "final_fit_last4y"), (end_day, 2., "final_fit_last2y"),
        ((pd.Timestamp("2024-03-31")-EPOCH).days, 4., "pre_time_holdout_last4y"),
    ]:
        with threadpool_limits(limits=3):
            estimates = descriptive_hedonic(full, target, model, end, years)
        for record in estimates:
            record["window"] = window
            record["test_rows"] = len(test) if record["city"] == "ALL_REGIONS" else int(test.address_city.eq(record["city"]).sum())
            region_rows.append(record)
        action("descriptive_regional_time_audit_complete", window=window,
               supported_regions=sum(r["status"]=="estimated" for r in estimates))
    region_table = pd.DataFrame(region_rows)
    result["files"]["regional_time_coefficients"] = csv(region_table, "feature_regional_time_coefficients.csv")
    final = region_table.loc[(region_table.window == "final_fit_last4y") & region_table.city.ne("ALL_REGIONS")
                              & region_table.status.eq("estimated")]
    two = region_table.loc[region_table.window.eq("final_fit_last2y") & region_table.status.eq("estimated")]
    stability = final[["city", "annual_log_coefficient", "test_rows"]].merge(
        two[["city", "annual_log_coefficient"]], on="city", suffixes=("_4y", "_2y"))
    stability["absolute_window_difference"] = np.abs(stability.annual_log_coefficient_4y-stability.annual_log_coefficient_2y)
    stability["sign_changes"] = np.sign(stability.annual_log_coefficient_4y) != np.sign(stability.annual_log_coefficient_2y)
    result["files"]["regional_time_stability"] = csv(stability.sort_values("test_rows", ascending=False),
                                                    "feature_regional_time_stability.csv")
    global_check = region_table.loc[(region_table.window=="final_fit_last4y") & region_table.city.eq("ALL_REGIONS")].iloc[0]
    assert abs(global_check.annual_log_coefficient-model["beta"][3]) < 1e-9
    result["regional_time_audit"] = {
        "descriptive_only_no_prediction_changes": True,
        "method": "Same within-project demeaning and 6 Huber IRLS updates; fixed 4-year and 2-year windows. Regional audit coefficients are not clipped.",
        "support_rule": "At least 300 rows and 20 composite projects; each included project at least 8 observations.",
        "uncertainty": "Approximate project-cluster sandwich CI; conditional descriptive diagnostic, not an honest forecast interval or post-selection guarantee.",
        "global_coefficient_reproduced": global_check.annual_log_coefficient,
        "supported_final_regions": len(final),
        "supported_region_test_coverage": float(final.test_rows.sum()/len(test)),
        "regional_annual_log_coefficient": summary(final.annual_log_coefficient),
        "test_weighted_regional_annual_log_coefficient": float(np.average(
            final.annual_log_coefficient, weights=final.test_rows)) if final.test_rows.sum() else None,
        "regions_global_outside_approx_ci": int(final.global_coefficient_outside_approx_ci.sum()),
        "window_absolute_difference": summary(stability.absolute_window_difference),
        "window_sign_changes": int(stability.sign_changes.sum()),
        "fixed_training_quantile_bounds_note": "All regional windows use the frozen final-fit unit-price bounds solely for a common descriptive comparison; no holdout MAE is claimed for these regressions.",
    }
    result["structural_improvements"] = [
        {
            "proposal": "Correct administrative parsing and use composite geographic project identity for seen/unseen evaluation and group folds.",
            "details": "Prefer supplied district when it exactly matches the address prefix after a recognized city; otherwise use a longest valid district suffix match. Preserve raw names; do not collapse same-name projects across cities.",
            "validation": "Before any model changes, deterministic address fixtures plus target-free split counts. Then compare unchanged fixed model predictions under name-only versus city/district/project grouping; future candidate validation must hold out composite projects.",
            "overfit_control": "Parsing correction derives from string semantics, not price residuals or Public score. Freeze the corrected parser and split rules before evaluating model changes.",
        },
        {
            "proposal": "Validate future extrapolation separately from local reference normalization, with a conservative hierarchical regional time effect only if it is stable.",
            "details": "Do not assume the 4-month temporal holdout establishes an 11.7-month trend. Keep anchor as fixed zero-extra-trend control; compare a predeclared global effect and strongly shrunk regional effect, with minimum support and fallback to global/zero. Do not directly deploy the descriptive regional coefficients.",
            "validation": "Expanding-window rolling origins with 1-4, 5-8, and 9-12 month horizon bands; group by composite project. Estimate trends only before each origin, select on earlier origins and reserve the latest eligible long-horizon origin once. Report both seen/unseen and city-weighted MAE, project bootstrap, and high-price tails.",
            "overfit_control": "At most one hierarchical alternative plus fixed anchor/global controls; preregister shrinkage/support rules, use no test labels or Public-score search, and require consistent improvement across dates and cities.",
        },
    ]
    after = {p: sha256(ROOT/p) for p in fixed_paths}
    if before != after:
        raise RuntimeError("A protected v1/v2 source or artifact changed during the audit")
    result["protected_artifacts_unchanged"] = before
    result["elapsed_seconds"] = time.perf_counter()-started
    action("audit_complete", seconds=result["elapsed_seconds"], protected_artifacts_unchanged=True)
    result["actions"] = ACTIONS
    result["files"]["action_log"] = csv(pd.DataFrame(ACTIONS), "feature_actions.csv")
    (OUT/"feature_audit.json").write_text(
        json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(plain({"audit": str(OUT/"feature_audit.json"), "date_span": result["date_span"],
                           "reference_coverage": result["reference_coverage"],
                           "regional_time_audit": result["regional_time_audit"]}), ensure_ascii=True), flush=True)


def correction_audit():
    """Compare group partitions and replay only when correction is pure relabeling."""
    import hashlib
    from round3_features import correct_address_features, self_check, ADDRESS_COLUMNS
    report_path = OUT / "feature_audit.json"
    if not report_path.is_file():
        raise FileNotFoundError("Run the primary audit first.")
    result = json.loads(report_path.read_text(encoding="utf-8"))
    d = load_data()
    full, _ = full_training_data(d)
    raw_full = pd.concat([d["raw_train"], d["raw_valid"]], ignore_index=True)
    keep = ~raw_full.duplicated(subset=[c for c in raw_full if c != ID]).to_numpy()
    raw_full = raw_full.loc[keep].reset_index(drop=True)
    correction_check = self_check()
    corrected = {}
    change_rows = []
    for split in ("train", "valid", "test"):
        old = d[f"X_{split}"]
        new = correct_address_features(d[f"raw_{split}"], old)
        corrected[split] = new
        changed = pd.DataFrame({c: old[c].ne(new[c]) for c in ADDRESS_COLUMNS})
        for column in ADDRESS_COLUMNS:
            change_rows.append({"split": split, "feature": column,
                                "changed_rows": int(changed[column].sum())})
        change_rows.append({"split": split, "feature": "any_address_feature",
                            "changed_rows": int(changed.any(axis=1).sum())})
        numeric_cols = old.select_dtypes(include="number").columns
        pd.testing.assert_frame_equal(old[numeric_cols], new[numeric_cols])
    corrected_full = correct_address_features(raw_full, full)
    with (ROOT / "outputs/round2/neighborhood/final_reference.pkl").open("rb") as handle:
        reference = pickle.load(handle)
    old_fit, new_fit = group_keys(full), group_keys(corrected_full)
    old_test, new_test = group_keys(d["X_test"]), group_keys(corrected["test"])
    partitions = []
    remapped_groups = {}
    all_bijective = True
    old_first = pd.Series("__NO_REFERENCE__", index=d["X_test"].index)
    new_first = old_first.copy()
    old_first_level = pd.Series("base_fallback", index=d["X_test"].index)
    new_first_level = old_first_level.copy()
    query_eligible = d["X_test"].building_area.between(15, 1500).fillna(False)
    missing_key = "__AUDIT_NULL_KEY__"
    for level in reference["groups"]:
        old_joint = pd.concat([old_fit[level], old_test[level]], ignore_index=True).fillna(missing_key)
        new_joint = pd.concat([new_fit[level], new_test[level]], ignore_index=True).fillna(missing_key)
        pair = pd.DataFrame({"old": old_joint, "new": new_joint})
        old_sizes = pair.groupby("old")["new"].nunique()
        new_sizes = pair.groupby("new")["old"].nunique()
        split_keys = set(old_sizes[old_sizes > 1].index)
        merge_keys = set(new_sizes[new_sizes > 1].index)
        bijective = not split_keys and not merge_keys
        all_bijective &= bijective
        old_to_new = pair.drop_duplicates("old").set_index("old")["new"].to_dict()
        if bijective:
            remapped_groups[level] = {old_to_new[key]: indices
                                      for key, indices in reference["groups"][level].items()}
        # A hash of sorted member indices tests actual reference membership, not text labels.
        valid_index = np.flatnonzero(np.isfinite(reference["reference_residual"]))
        min_rows = {"project":6, "address":8, "street_section":20,
                    "locality":20, "street":20, "district_style":80}[level]
        new_members = pd.Series(valid_index, index=new_fit[level].iloc[valid_index].to_numpy()).groupby(level=0).apply(
            lambda s: s.to_numpy(dtype="int64")).to_dict()
        new_members = {k:v for k,v in new_members.items() if len(v) >= min_rows}
        def fingerprints(groups):
            return {k: hashlib.sha256(np.sort(v).astype("int64").tobytes()).hexdigest()
                    for k,v in groups.items()}
        old_fp, new_fp = fingerprints(reference["groups"][level]), fingerprints(new_members)
        old_query_fp = old_test[level].map(old_fp).fillna("__NO_REFERENCE__")
        new_query_fp = new_test[level].map(new_fp).fillna("__NO_REFERENCE__")
        old_select = query_eligible & old_first.eq("__NO_REFERENCE__") & old_query_fp.ne("__NO_REFERENCE__")
        new_select = query_eligible & new_first.eq("__NO_REFERENCE__") & new_query_fp.ne("__NO_REFERENCE__")
        old_first.loc[old_select] = old_query_fp.loc[old_select]
        new_first.loc[new_select] = new_query_fp.loc[new_select]
        old_first_level.loc[old_select] = level
        new_first_level.loc[new_select] = level
        partitions.append({
            "level": level, "partition_is_bijective_relabeling": bijective,
            "old_keys_split": len(split_keys), "new_keys_merged": len(merge_keys),
            "joint_rows_in_split_keys": int(old_joint.isin(split_keys).sum()),
            "joint_rows_in_merge_keys": int(new_joint.isin(merge_keys).sum()),
            "fit_keys_changed_rows": int(old_fit[level].fillna(missing_key).ne(new_fit[level].fillna(missing_key)).sum()),
            "test_keys_changed_rows": int(old_test[level].fillna(missing_key).ne(new_test[level].fillna(missing_key)).sum()),
            "test_reference_member_set_changed_rows": int(old_query_fp.ne(new_query_fp).sum()),
        })
    section = {
        "api": "round3_features.correct_address_features(raw, X=None) or corrected_features(raw)",
        "source_sha256": sha256(ROOT/"src/round3_features.py"), "minimal_assertions": correction_check,
        "numeric_features_unchanged": True, "changes": change_rows, "partitions": partitions,
        "all_levels_bijective_relabeling": all_bijective,
        "no_coefficients_refit": True,
        "actually_selected_reference_member_changes": int(old_first.ne(new_first).sum()),
        "actually_selected_reference_level_changes": int(old_first_level.ne(new_first_level).sum()),
        "affected_test_ids": d["ids_test"][old_first.ne(new_first).to_numpy()].tolist(),
    }
    if all_bijective:
        remapped = dict(reference)
        remapped["groups"] = remapped_groups
        cache = ROOT/"outputs/round2/final_predictions"
        baseline = .75*np.load(cache/"catboost_raw_test.npy")+.25*np.load(cache/"lightgbm_l1_unit63_test.npy")
        with threadpool_limits(limits=3):
            p, diag, _ = _predict_reference_all(remapped, corrected["test"], baseline)
        comparisons = {}
        for name, values in p.items():
            old = np.load(cache/f"{name}_test.npy")
            comparisons[name] = {"exactly_equal": bool(np.array_equal(values, old)),
                                 "max_absolute_difference_TWD": float(np.max(np.abs(values-old)))}
        section["replay"] = comparisons
        section["candidate_implication"] = (
            "Data-quality correction only: pure relabeling preserves neighborhood membership and predictions; do not spend a model candidate on it."
            if all(v["exactly_equal"] for v in comparisons.values()) else
            "Partitions are equivalent but numeric replay differs; investigate before using.")
    else:
        section["replay"] = None
        section["candidate_implication"] = "Some partitions changed; membership differences quantified. No altered-reference prediction or fitting performed."
    result["address_correction_partition_audit"] = section
    result["files"]["address_correction_changes"] = csv(pd.DataFrame(change_rows), "feature_address_correction_changes.csv")
    result["files"]["address_correction_partitions"] = csv(pd.DataFrame(partitions), "feature_address_correction_partitions.csv")
    result["primary_audit_script_sha256"] = result.get("primary_audit_script_sha256", result["script_sha256"])
    result["script_sha256"] = sha256(Path(__file__))
    result["actions"].append({"utc": datetime.now(timezone.utc).isoformat(),
        "action": "address_correction_partition_audit_complete", "no_training": True,
        "all_levels_bijective_relabeling": all_bijective})
    result["files"]["action_log"] = csv(pd.DataFrame(result["actions"]), "feature_actions.csv")
    report_path.write_text(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(plain(section), ensure_ascii=True), flush=True)

if __name__ == "__main__":
    import sys
    if "--correction-only" not in sys.argv:
        main()
    correction_audit()
