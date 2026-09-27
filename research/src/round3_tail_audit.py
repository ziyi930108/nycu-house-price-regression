"""Audit prediction tails and area-sensitive failure modes using frozen local data.

No models are trained, no test labels are available or used, no submissions are
written, and no existing artifact is changed. Two fixed diagnostic routing
rules use an existing 2,000 m2 area flag and transaction semantics, not public
leaderboard feedback or test-label tuning.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round3/audit"
ID, TARGET = "編號", "總價元"
AREA_KNOT = 2000.0
MISSING = "__MISSING__"
FILES = []
BASE_MODELS = ["cat_raw", "lgbm_unit", "lgbm_spatial", "v1", "anchor", "v2", "mix"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy(relative, rows):
    path = ROOT / relative
    FILES.append(path)
    array = np.load(path, allow_pickle=False).astype("float64")
    if array.shape != (rows,) or not np.isfinite(array).all():
        raise ValueError(f"Invalid frozen predictions: {relative}")
    return array


def finite_profile(values):
    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    result = {"rows": len(values), "finite": len(finite), "missing_or_nonfinite": int((~np.isfinite(values)).sum())}
    if not len(finite):
        return result
    result.update({"minimum": float(finite.min()), "median": float(np.median(finite)),
                   "p90": float(np.quantile(finite, .9)), "p99": float(np.quantile(finite, .99)),
                   "p99_9": float(np.quantile(finite, .999)), "maximum": float(finite.max()),
                   "zero": int((finite == 0).sum()), "negative": int((finite < 0).sum()),
                   "above_100million": int((finite > 1e8).sum()),
                   "above_1billion": int((finite > 1e9).sum())})
    return result


def effective_area(X):
    building = X["building_area"].to_numpy(dtype="float64")
    parking = X["parking_area"].to_numpy(dtype="float64")
    return np.where(np.isfinite(building) & (building > 0), building,
                    np.where(np.isfinite(parking) & (parking > 0), parking, 1.0))


def project_set(X):
    return set(X["project_name"].astype(str)) - {MISSING}


def flags(X, seen):
    n = len(X)
    area = X["building_area"].to_numpy(dtype="float64")
    land = X["land_area"].to_numpy(dtype="float64")
    building_count = X["building_count"].to_numpy(dtype="float64")
    typ = X["transaction_type"].fillna(MISSING).astype(str)
    land_only = typ.str.contains("土地", regex=False).to_numpy() & ~typ.str.contains("房地|建物", regex=True).to_numpy()
    parking_only = typ.str.contains("車位", regex=False).to_numpy() & ~typ.str.contains("房地|建物|土地", regex=True).to_numpy()
    bulk = np.isfinite(building_count) & (building_count > 1)
    masks = {
        "all": np.ones(n, dtype=bool),
        "seen_project": seen, "unseen_project": ~seen,
        "building_area_zero": area == 0, "building_area_missing": ~np.isfinite(area),
        "building_area_negative": area < 0, "building_area_0_to_100": (area > 0) & (area <= 100),
        "building_area_100_to_300": (area > 100) & (area <= 300),
        "building_area_300_to_1000": (area > 300) & (area <= 1000),
        "building_area_1000_to_2000": (area > 1000) & (area <= 2000),
        "building_area_2000_to_10000": (area > 2000) & (area <= 10000),
        "building_area_gt_10000": area > 10000,
        "building_area_gt_2000": area > AREA_KNOT,
        "land_area_gt_10000": land > 10000,
        "bulk_building_count_gt1": bulk, "land_only": land_only,
        "parking_only": parking_only, "bulk_or_land": bulk | land_only,
        "entire_building": X["is_entire_building"].to_numpy() > 0,
        "unusual_layout": X["unusual_layout"].to_numpy() > 0,
        "trade_date_invalid": X["trade_date_invalid"].to_numpy() > 0,
        "trade_date_out_of_range": X["trade_date_out_of_range"].to_numpy() > 0,
    }
    for column in ("land_area_negative", "parking_area_negative", "remark_related_party",
                   "remark_partial_share", "remark_special_sale", "remark_land_only"):
        masks[column] = X[column].to_numpy() > 0
    return masks


def key_ids(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and np.equal(numeric, np.floor(numeric)).all():
        return numeric.astype("int64").astype(str)
    return series.astype(str)


def csv_test_prediction(relative, raw_test):
    path = ROOT / relative
    FILES.append(path)
    frame = pd.read_csv(path)
    ids = key_ids(frame[ID])
    if ids.duplicated().any():
        raise ValueError(f"Duplicate IDs in {relative}")
    lookup = pd.Series(frame[TARGET].to_numpy(dtype="float64"), index=ids)
    return lookup.loc[key_ids(raw_test[ID])].to_numpy()


def load_inputs():
    raw, X = {}, {}
    for split in ("train", "valid", "test"):
        raw[split] = pd.read_pickle(ROOT / f"data_cache/{split}.pkl")
        X[split] = pd.read_pickle(ROOT / f"data_cache/{split}_features.pkl")
        if len(raw[split]) != len(X[split]):
            raise ValueError(f"Unaligned cache {split}")
    keep = ~raw["train"].duplicated(subset=[c for c in raw["train"] if c != ID]).to_numpy()
    raw["train"] = raw["train"].loc[keep].reset_index(drop=True)
    X["train"] = X["train"].loc[keep].reset_index(drop=True)
    days = X["train"]["trade_day_index"].to_numpy()
    cut = (pd.Timestamp("2024-04-01") - pd.Timestamp("2000-01-01")).days
    end = (pd.Timestamp("2024-08-10") - pd.Timestamp("2000-01-01")).days
    fit_idx = np.flatnonzero(np.isfinite(days) & (days < cut))
    time_idx = np.flatnonzero(np.isfinite(days) & (days >= cut) & (days <= end))
    old_idx = read_npy("outputs/experiments/selected_temporal_indices.npy", len(time_idx)).astype("int64")
    if not np.array_equal(old_idx, time_idx):
        raise ValueError("Temporal indices mismatch")
    for name, indices in (("temporal_fit", fit_idx), ("temporal", time_idx)):
        raw[name] = raw["train"].iloc[indices].reset_index(drop=True)
        X[name] = X["train"].iloc[indices].reset_index(drop=True)
    y = {split: frame[TARGET].to_numpy(dtype="float64") for split, frame in raw.items() if TARGET in frame}
    train_projects = project_set(X["train"])
    time_projects = project_set(X["temporal_fit"])
    full_projects = train_projects | project_set(X["valid"])
    seen = {
        "train": X["train"]["project_name"].isin(train_projects).to_numpy(),
        "valid": X["valid"]["project_name"].isin(train_projects).to_numpy(),
        "test": X["test"]["project_name"].isin(full_projects).to_numpy(),
        "temporal": X["temporal"]["project_name"].isin(time_projects).to_numpy(),
        "temporal_fit": X["temporal_fit"]["project_name"].isin(time_projects).to_numpy(),
    }
    preds = {}
    for split, suffix in (("valid", "valid"), ("temporal", "temporal_valid")):
        n = len(X[split])
        preds[split] = {
            "cat_raw": read_npy(f"outputs/experiments/catboost_raw_{suffix}.npy", n),
            "lgbm_unit": read_npy(f"outputs/experiments/lightgbm_l1_unit63_{suffix}.npy", n),
            "lgbm_spatial": read_npy(f"outputs/round2/lightgbm/lightgbm_unit_spatial63_{suffix}.npy", n),
            "v1": read_npy(f"outputs/experiments/selected_{suffix}.npy", n),
            "anchor": read_npy(f"outputs/round2/neighborhood/neighborhood_hedonic_anchor_{split}_predictions.npy", n),
            "v2": read_npy(f"outputs/round2/neighborhood/neighborhood_hedonic_trend_{split}_predictions.npy", n),
        }
        preds[split]["mix"] = .75 * preds[split]["v2"] + .25 * preds[split]["lgbm_spatial"]
        expected_base = .75 * preds[split]["cat_raw"] + .25 * preds[split]["lgbm_unit"]
        if not np.allclose(expected_base, preds[split]["v1"], rtol=1e-10, atol=1e-5):
            raise ValueError(f"Frozen v1 composition differs in {split}")
    n = len(X["test"])
    test_paths = {
        "cat_raw": "catboost_raw", "lgbm_unit": "lightgbm_l1_unit63",
        "lgbm_spatial": "lightgbm_unit_spatial63",
        "anchor": "neighborhood_hedonic_anchor", "v2": "neighborhood_hedonic_trend",
    }
    preds["test"] = {name: read_npy(f"outputs/round2/final_predictions/{stem}_test.npy", n)
                     for name, stem in test_paths.items()}
    preds["test"]["v1"] = .75 * preds["test"]["cat_raw"] + .25 * preds["test"]["lgbm_unit"]
    preds["test"]["mix"] = .75 * preds["test"]["v2"] + .25 * preds["test"]["lgbm_spatial"]
    csv_deltas = {}
    for name, filename in (("v1", "submission.csv"), ("v2", "submission_v2.csv"),
                           ("anchor", "submission_v2_anchor.csv"), ("mix", "submission_v2_mix.csv")):
        saved = csv_test_prediction(f"outputs/{filename}", raw["test"])
        difference = float(np.max(np.abs(saved - preds["test"][name])))
        csv_deltas[name] = difference
        if difference > .5001:
            raise ValueError(f"Frozen {name} array does not reproduce submitted CSV within rounding: {difference}")
    return raw, X, y, seen, preds, csv_deltas


def record_rows(split, reason, indices, raw, X, y, seen, preds):
    if not len(indices):
        return []
    frame = pd.DataFrame({
        "split": split, "reason": reason, "row_index": indices,
        "id": raw[split].iloc[indices][ID].astype(str).to_numpy(),
        "seen_project": seen[split][indices],
    })
    columns = ["project_name", "address_city", "address_district", "transaction_type",
               "building_type", "main_use", "land_area", "building_area", "parking_area",
               "building_count", "parking_count", "rooms", "total_floors",
               "remark_related_party", "remark_partial_share", "remark_special_sale"]
    for column in columns:
        frame[column] = X[split].iloc[indices][column].to_numpy()
    area = effective_area(X[split])[indices]
    frame["effective_area"] = area
    if split in y:
        frame["observed_price"] = y[split][indices]
    if split in preds:
        p = preds[split]
        for name in BASE_MODELS:
            frame[f"prediction_{name}"] = p[name][indices]
            if split in y:
                frame[f"absolute_error_{name}"] = np.abs(p[name][indices] - y[split][indices])
        frame["implied_lgbm_unit_price_per_effective_m2"] = p["lgbm_unit"][indices] / area
        frame["implied_spatial_price_per_effective_m2"] = p["lgbm_spatial"][indices] / area
        frame["unit_component_share_of_v1"] = .25 * p["lgbm_unit"][indices] / np.maximum(p["v1"][indices], 1)
        frame["unit_increment_over_raw"] = p["v1"][indices] - p["cat_raw"][indices]
        frame["anchor_increment_over_v1"] = p["anchor"][indices] - p["v1"][indices]
        frame["trend_increment_over_anchor"] = p["v2"][indices] - p["anchor"][indices]
        frame["spatial_increment_in_mix"] = p["mix"][indices] - p["v2"][indices]
    return frame.to_dict("records")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    started = time.perf_counter()
    raw, X, y, seen, preds, csv_deltas = load_inputs()
    profile_rows, flag_rows, type_rows, error_rows, tail_rows, top_rows, diagnostic_rows = [], [], [], [], [], [], []
    masks = {split: flags(frame, seen[split]) for split, frame in X.items()}
    train_area_p99 = float(np.nanquantile(X["train"]["building_area"], .99))
    train_area_p999 = float(np.nanquantile(X["train"]["building_area"], .999))
    fit_area_max = {split: float(np.nanmax(X["temporal_fit" if split == "temporal" else "train"]["building_area"]))
                    for split in ("valid", "temporal", "test")}
    for split, frame in X.items():
        for column in ("building_area", "land_area", "parking_area", "indoor_area",
                       "building_count", "rooms", "bathrooms"):
            profile_rows.append({"split": split, "variable": column, **finite_profile(frame[column])})
        if split in y:
            profile_rows.append({"split": split, "variable": "observed_price", **finite_profile(y[split])})
        for label, mask in masks[split].items():
            flag_rows.append({"split": split, "flag": label, "rows": int(mask.sum()), "share": float(mask.mean())})
        area = frame["building_area"].to_numpy(dtype="float64")
        for typ, indices in frame.groupby("transaction_type", dropna=False).indices.items():
            indices = np.asarray(indices)
            record = {"split": split, "transaction_type": str(typ), "rows": len(indices),
                      "share": len(indices) / len(frame),
                      "building_area_median": float(np.nanmedian(area[indices])),
                      "building_area_p99": float(np.nanquantile(area[indices], .99))}
            if split in y:
                record.update({"observed_price_median": float(np.median(y[split][indices])),
                               "observed_price_max": float(y[split][indices].max())})
            if split in preds:
                record["v2_prediction_max"] = float(preds[split]["v2"][indices].max())
            type_rows.append(record)
        top_area_idx = np.argsort(np.nan_to_num(area, nan=-np.inf))[-30:][::-1]
        top_rows += record_rows(split, "highest_building_area", top_area_idx, raw, X, y, seen, preds)

    # Rules come from the existing area flag and transaction semantics.
    # They are not optimized against public feedback or selected as submissions.
    for split, p in preds.items():
        area = effective_area(X[split])
        taper = np.minimum(1.0, AREA_KNOT / area)
        bulk_land = masks[split]["bulk_or_land"]
        diagnostics = {
            "v1_unit_taper_2000m2": p["cat_raw"] + taper * (p["v1"] - p["cat_raw"]),
            "v2_area_taper_to_raw_2000m2": p["cat_raw"] + taper * (p["v2"] - p["cat_raw"]),
            "mix_area_taper_to_raw_2000m2": p["cat_raw"] + taper * (p["mix"] - p["cat_raw"]),
            "v2_bulk_land_route_raw": np.where(bulk_land, p["cat_raw"], p["v2"]),
            "mix_bulk_land_route_raw": np.where(bulk_land, p["cat_raw"], p["mix"]),
        }
        p.update(diagnostics)
        masks[split]["building_area_above_training_p99"] = X[split]["building_area"].to_numpy() > train_area_p99
        masks[split]["building_area_above_training_p999"] = X[split]["building_area"].to_numpy() > train_area_p999
        masks[split]["building_area_above_fit_max"] = X[split]["building_area"].to_numpy() > fit_area_max[split]
        masks[split]["v2_prediction_gt100million"] = p["v2"] > 1e8
        masks[split]["v2_prediction_gt1billion"] = p["v2"] > 1e9
        masks[split]["unit_gt_2x_raw"] = p["lgbm_unit"] > 2 * np.maximum(p["cat_raw"], 1)
        masks[split]["spatial_gt_2x_raw"] = p["lgbm_spatial"] > 2 * np.maximum(p["cat_raw"], 1)
        if split in y:
            masks[split]["observed_price_gt100million"] = y[split] > 1e8
            masks[split]["observed_price_gt1billion"] = y[split] > 1e9
            base_error = np.abs(y[split] - p["v2"])
            top_error_mask = np.zeros(len(base_error), dtype=bool)
            top_error_mask[np.argsort(base_error)[-max(1, int(np.ceil(len(base_error) * .01))):]] = True
            masks[split]["top_1pct_v2_error_descriptive_only"] = top_error_mask
            for name in ("v2", "mix", "lgbm_unit", "lgbm_spatial"):
                ranking = np.argsort(np.abs(y[split] - p[name]))[-30:][::-1]
                top_rows += record_rows(split, f"highest_absolute_error_{name}", ranking, raw, X, y, seen, preds)
        for name, prediction in p.items():
            profile_rows.append({"split": split, "variable": f"prediction_{name}", **finite_profile(prediction)})
            if name in ("lgbm_unit", "lgbm_spatial"):
                positive_building = X[split]["building_area"].to_numpy() > 0
                profile_rows.append({"split": split, "variable": f"{name}_implied_unit_price_positive_building_area",
                                     **finite_profile((prediction / area)[positive_building])})
            for threshold in (1e8, 1e9):
                large = prediction > threshold
                record = {"split": split, "model": name, "threshold": threshold,
                          "rows": int(large.sum()), "share": float(large.mean()),
                          "sum_predictions": float(prediction[large].sum()),
                          "maximum_prediction": float(prediction.max()),
                          "large_building_area_rows": int((large & masks[split]["building_area_gt_2000"]).sum()),
                          "bulk_or_land_rows": int((large & masks[split]["bulk_or_land"]).sum()),
                          "unseen_project_rows": int((large & ~seen[split]).sum())}
                if split in y:
                    record.update({"mae": float(np.abs(prediction[large] - y[split][large]).mean()) if large.any() else None,
                                   "observed_above_threshold_rows": int((large & (y[split] > threshold)).sum())})
                tail_rows.append(record)
            if split in y:
                error = np.abs(prediction - y[split])
                for label, mask in masks[split].items():
                    if not mask.any():
                        continue
                    error_rows.append({
                        "split": split, "segment": label, "model": name, "rows": int(mask.sum()),
                        "distinct_project_names_including_missing": int(X[split].loc[mask, "project_name"].nunique(dropna=False)),
                        "mae": float(error[mask].mean()),
                        "mean_signed_error_pred_minus_actual": float((prediction[mask] - y[split][mask]).mean()),
                        "error_p99": float(np.quantile(error[mask], .99)),
                        "absolute_error_sum": float(error[mask].sum()),
                        "contribution_to_overall_mae": float(error[mask].sum() / len(error)),
                        "share_of_model_absolute_error": float(error[mask].sum() / error.sum()),
                        "overprediction_fraction": float((prediction[mask] > y[split][mask]).mean()),
                    })
        if split == "test":
            for name in BASE_MODELS:
                ranking = np.argsort(p[name])[-30:][::-1]
                top_rows += record_rows(split, f"highest_prediction_{name}", ranking, raw, X, y, seen, preds)
        for name, prediction in diagnostics.items():
            baseline_name = "mix" if name.startswith("mix") else "v1" if name.startswith("v1") else "v2"
            changed = ~np.isclose(prediction, p[baseline_name], rtol=0, atol=1e-6)
            record = {"split": split, "diagnostic": name, "baseline": baseline_name,
                      "changed_rows": int(changed.sum()), "maximum_prediction": float(prediction.max()),
                      "prediction_gt100million_rows": int((prediction > 1e8).sum()),
                      "prediction_gt1billion_rows": int((prediction > 1e9).sum()),
                      "sum_absolute_prediction_change": float(np.abs(prediction - p[baseline_name]).sum()),
                      "full_split_mae_change_upper_bound_without_labels": float(np.abs(prediction - p[baseline_name]).mean())}
            if split in y:
                original_error = np.abs(p[baseline_name] - y[split])
                new_error = np.abs(prediction - y[split])
                record.update({"mae": float(new_error.mean()), "baseline_mae": float(original_error.mean()),
                               "mae_improvement": float(original_error.mean() - new_error.mean()),
                               "changed_rows_mae_before": float(original_error[changed].mean()) if changed.any() else None,
                               "changed_rows_mae_after": float(new_error[changed].mean()) if changed.any() else None,
                               "improved_rows": int((new_error < original_error - 1e-6).sum()),
                               "worsened_rows": int((new_error > original_error + 1e-6).sum())})
            diagnostic_rows.append(record)

    # Rebuild counts after adding the descriptive tail masks.
    flag_rows = [{"split": split, "flag": label, "rows": int(mask.sum()), "share": float(mask.mean())}
                 for split, split_masks in masks.items() for label, mask in split_masks.items()]
    def segment_result(split, segment, model="v2"):
        return next(r for r in error_rows if r["split"] == split and r["segment"] == segment and r["model"] == model)
    largest_test = int(np.argmax(preds["test"]["v2"]))
    largest_test_record = record_rows("test", "largest_v2_prediction", np.array([largest_test]), raw, X, y, seen, preds)[0]
    findings = {
        "largest_test_case": largest_test_record,
        "largest_test_area_ratio_to_training_max": float(X["test"]["building_area"].max() / X["train"]["building_area"].max()),
        "temporal_gt10000m2": {name: segment_result("temporal", "building_area_gt_10000", name)
                               for name in ("cat_raw", "lgbm_unit", "lgbm_spatial", "v2", "mix")},
        "v2_temporal_seen": segment_result("temporal", "seen_project"),
        "v2_temporal_unseen": segment_result("temporal", "unseen_project"),
        "v2_temporal_gt2000m2": segment_result("temporal", "building_area_gt_2000"),
        "conclusions": [
            "The largest test prediction is mechanically driven by unit-price predictions multiplied by an area 3.17 times the training maximum. Its neighborhood anchor increment is zero; the trend increment is much smaller than the unit component.",
            "Large predictions are exposure, not evidence of wrong test labels. The two temporal observations above 10,000 m2 belong to one project and each have actual price 2.99 billion; both unit models are much closer than raw CatBoost. Evidence from this rare subgroup is highly concentrated.",
            "Both fixed correction families worsen original-valid and temporal MAE. This audit therefore does not recommend deploying an area cap, area taper to raw, or bulk/land routing to raw.",
            "Prioritize the already planned fixed recency/generalization experiment: unseen projects account for 66.33 percent of v2 temporal absolute error, whereas area above 2,000 m2 accounts for 4.58 percent. Assess signed error and seen/unseen strata on a new year-long development split before freezing confirmation.",
            "Retain explicit unit-model versus raw-model disagreement and commercial/industrial/parking-count subgroup reporting in future evaluation; building_count alone misses the largest commercial building because its count is one. No additional correction threshold is selected here."
        ],
        "full_test_change_bound_note": "By the triangle inequality, each diagnostic's sum absolute prediction change divided by all 47,073 rows bounds its possible whole-test MAE change regardless of unknown labels. It does not bound an unknown public subset's MAE.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "numeric_profiles": profile_rows, "segment_counts": flag_rows, "transaction_types": type_rows,
        "segment_errors": error_rows, "prediction_tail_counts": tail_rows,
        "tail_rows": top_rows, "global_rule_diagnostics": diagnostic_rows,
    }
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(OUT / f"tail_{name}.csv", index=False, encoding="utf-8-sig")
    report = {
        "purpose": "Local tail-risk diagnosis of frozen v1/v2/mix predictions conditional on previously selected holdouts",
        "findings": findings,
        "constraints": {"test_labels_used": False, "public_score_used_for_threshold_selection": False,
                        "model_training_performed": False, "submission_written": False,
                        "existing_artifacts_modified": False, "external_property_data_used": False},
        "temporal_cutoff": "2024-04-01", "temporal_end": "2024-08-10",
        "rows": {name: len(frame) for name, frame in X.items()},
        "test_unseen_project_share_relative_to_train_plus_valid": float((~seen["test"]).mean()),
        "csv_alignment_max_rounding_difference": csv_deltas,
        "building_area_train_p99": train_area_p99, "building_area_train_p999": train_area_p999,
        "building_area_fit_max": fit_area_max,
        "rules": {
            "area_taper": {
                "knot_m2": AREA_KNOT,
                "source": "Existing house_features large_area building-area threshold; not chosen by public score",
                "formula": "cat_raw + min(1, 2000/effective_area) * (original_prediction - cat_raw)",
                "interpretation": "Smoothly reduce extrapolating area-sensitive prediction weight above 2,000 m2; identical below the knot",
            },
            "bulk_land_route_raw": {
                "formula": "Use cat_raw only when building_count > 1 or transaction_type is land-only; otherwise retain original",
                "interpretation": "Fixed transaction-semantic diagnostic, not a searched cutoff",
            },
        },
        "diagnostic_caution": "These are fixed sensitivity analyses on previously selected holdouts, not validated new submissions. Test tail sizes are exposure, not known test errors. The V2 routing rule is a postprocessing diagnostic and does not rerun the neighborhood reference model.",
        "project_membership_note": "Missing project names are counted as unseen, including train profiling. A seen name alone does not guarantee observed support for a transaction size or type.",
        "metric_note": "Segment masks overlap; their error shares must not be added. Top-error masks depend on labels and are descriptive only, never a routing rule.",
        "global_rule_diagnostics": diagnostic_rows,
        "numeric_profiles": profile_rows,
        "prediction_tail_counts": tail_rows,
        "source_sha256": {s: json.loads((ROOT / f"data_cache/{s}.json").read_text(encoding="utf-8"))["source_sha256"]
                          for s in ("train", "valid", "test")},
        "input_prediction_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(FILES))},
        "script_sha256": sha(Path(__file__)), "elapsed_seconds": time.perf_counter() - started,
        "csv_files": {name: f"outputs/round3/audit/tail_{name}.csv" for name in tables},
    }
    (OUT / "tail_audit.json").write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(safe_json({"rows": report["rows"], "area_p99": train_area_p99, "area_p999": train_area_p999,
                              "csv_alignment": csv_deltas, "diagnostics": diagnostic_rows,
                              "elapsed_seconds": report["elapsed_seconds"]}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

