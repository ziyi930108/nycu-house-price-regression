"""Two local-neighborhood price-reference candidates; no external data.

Run --stage temporal first, inspect its holdout metrics, then --stage valid.
All price references and hedonic coefficients use the fitting side only.
This script never fits on train+valid or predicts the competition test prices.
"""
from __future__ import annotations
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "3"

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from model_common import ROOT, load_data, temporal_indices, mae
from prepare_data import sha256

OUT = ROOT / "outputs" / "round2" / "neighborhood"
MISSING = "__MISSING__"
NAMES = ("neighborhood_hedonic_anchor", "neighborhood_hedonic_trend")
BASE_COMPONENTS = (("catboost_raw", 0.75), ("lightgbm_l1_unit63", 0.25))
PARAMETERS = {
    "reference_building_area_bounds": [15.0, 1500.0],
    "reference_unit_price_quantile_bounds": [0.001, 0.999],
    "minimum_project_rows": 6,
    "minimum_address_rows": 8,
    "minimum_street_rows": 20,
    "minimum_district_rows": 80,
    "neighbors_project": 16,
    "neighbors_other": 40,
    "distance_features": {
        "log_building_area": 0.35, "parking_fraction": 0.15, "floor_mean": 12.0,
        "total_floors": 15.0, "rooms": 2.0, "land_building_ratio": 0.5,
        "transaction_years": 0.75, "parking_count": 2.0,
    },
    "hedonic_features": ["log_building_area", "parking_fraction", "floor_mean_per10", "transaction_years"],
    "hedonic_recent_years": 4,
    "hedonic_project_minimum": 8,
    "hedonic_huber_log_threshold": 0.10,
    "hedonic_irls_iterations": 6,
    "hedonic_coefficient_lower": [-0.7, -2.0, -0.05, -0.08],
    "hedonic_coefficient_upper": [0.3, 0.5, 0.15, 0.20],
    "neighborhood_base_weights": {
        "project": 0.75, "address": 0.65, "street_section": 0.40,
        "locality": 0.35, "street": 0.30, "district_style": 0.12,
    },
    "local_to_base_price_ratio_bounds": [0.65, 1.5],
    "distance_confidence_scale": 2.0,
    "dispersion_log_scale": 0.18,
    "trend_future_day_cap": 366,
    "cpu_threads": 3,
}


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def numeric(frame, name, lower=None, upper=None):
    values = frame[name].to_numpy(dtype="float64")
    if lower is not None:
        values = np.maximum(values, lower)
    if upper is not None:
        values = np.minimum(values, upper)
    return values


def matrices(frame, fill_values=None):
    area = numeric(frame, "building_area", 1.0)
    park = numeric(frame, "parking_fraction", 0.0, 0.8)
    floor = numeric(frame, "floor_mean", -3.0, 60.0)
    total = numeric(frame, "total_floors", 1.0, 70.0)
    rooms = numeric(frame, "rooms", 0.0, 10.0)
    land = numeric(frame, "land_building_ratio", 0.0, 4.0)
    dates = numeric(frame, "trade_day_index")
    parking_count = numeric(frame, "parking_count", 0.0, 6.0)
    raw = np.column_stack([
        np.log(area), park, floor, total, rooms, land,
        (dates - 8000.0) / 365.25, parking_count,
    ])
    if fill_values is None:
        fill_values = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        fill_values = np.nan_to_num(fill_values)
    raw = np.where(np.isfinite(raw), raw, fill_values[None, :])
    distance = raw / np.array([0.35, 0.15, 12.0, 15.0, 2.0, 0.5, 0.75, 2.0])
    hedonic = raw[:, [0, 1, 2, 6]].copy()
    hedonic[:, 2] /= 10.0
    return distance, hedonic, fill_values


def group_keys(frame):
    city = frame["address_city"].astype(str)
    district = frame["address_district"].astype(str)
    region = city + "|" + district
    style = frame["building_type"].astype(str) + "|" + frame["main_use"].astype(str)
    prefix = region + "|" + style + "|"
    project = frame["project_name"].astype(str)
    road = frame["address_road"].astype(str)
    locality = frame["address_locality"].astype(str)
    address = frame["address_raw"].astype(str)
    keys = {
        "project": (prefix + project).where(project != MISSING),
        "address": (prefix + address).where(address != MISSING),
        "street_section": (prefix + road + "|" + frame["address_section"].astype(str)).where(road != MISSING),
        "locality": (prefix + locality).where(locality != MISSING),
        "street": (prefix + road).where(road != MISSING),
        "district_style": region + "|" + style,
    }
    return keys


def fit_hedonic(frame, target, area, hedonic, reference_mask, project_keys):
    dates = numeric(frame, "trade_day_index")
    maximum_day = float(np.nanmax(dates))
    selected = (
        reference_mask & np.isfinite(dates) &
        (dates >= maximum_day - PARAMETERS["hedonic_recent_years"] * 365.25) &
        (frame["project_name"].to_numpy() != MISSING)
    )
    index = np.flatnonzero(selected)
    work = pd.DataFrame(hedonic[index], columns=["log_area", "parking", "floor", "year"])
    work["target"] = np.log(target[index] / area[index])
    work["group"] = project_keys.iloc[index].to_numpy()
    counts = work.groupby("group")["target"].transform("size")
    work = work.loc[counts >= PARAMETERS["hedonic_project_minimum"]].copy()
    names = ["log_area", "parking", "floor", "year", "target"]
    centered = work[names] - work.groupby("group")[names].transform("mean")
    design = centered.iloc[:, :4].to_numpy(dtype="float64")
    response = centered["target"].to_numpy(dtype="float64")
    if len(response) < 100:
        raise ValueError("Insufficient within-project observations for hedonic normalization")
    weights = np.ones(len(response), dtype="float64")
    beta = np.zeros(4)
    for _ in range(PARAMETERS["hedonic_irls_iterations"]):
        gram = design.T @ (weights[:, None] * design) + np.eye(4) * 1e-5
        beta = np.linalg.solve(gram, design.T @ (weights * response))
        residual = response - design @ beta
        weights = np.minimum(1.0, PARAMETERS["hedonic_huber_log_threshold"] / np.maximum(np.abs(residual), 1e-8))
    beta = np.clip(beta, PARAMETERS["hedonic_coefficient_lower"], PARAMETERS["hedonic_coefficient_upper"])
    return beta, {
        "fit_rows": int(len(response)), "project_groups": int(work["group"].nunique()),
        "maximum_fit_day_index": maximum_day,
        "coefficients": {name: float(value) for name, value in zip(PARAMETERS["hedonic_features"], beta)},
    }


def weighted_median(values, weights):
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    threshold = sorted_weights.sum(axis=1) * 0.5
    selected = (np.cumsum(sorted_weights, axis=1) >= threshold[:, None]).argmax(axis=1)
    return sorted_values[np.arange(len(values)), selected]


def fit_final_reference(X, y, *, parameters=None):
    """Fit a serializable reference model using only the supplied labeled rows.

    Returns a plain dict containing numeric reference arrays, grouping indices,
    robust within-project coefficients and imputation values. It can be saved
    with joblib.dump(..., compress=3) or pickle. No query/test rows are used.

    The present version implements exactly PARAMETERS; pass that dictionary or
    omit parameters. Callers choose which labeled split to supply. The function
    does not itself concatenate train/valid or read any files.
    """
    if parameters is not None and parameters != PARAMETERS:
        raise ValueError("This model version requires the recorded PARAMETERS without overrides")
    required = [
        "building_area", "parking_fraction", "floor_mean", "total_floors",
        "rooms", "land_building_ratio", "trade_day_index", "parking_count",
        "address_city", "address_district", "building_type", "main_use",
        "project_name", "address_road", "address_locality", "address_raw",
        "address_section",
    ]
    missing = [name for name in required if name not in X]
    if missing:
        raise ValueError(f"Missing reference feature columns: {missing}")
    started = time.perf_counter()
    fit = X[required].reset_index(drop=True)
    target = np.asarray(y, dtype="float64")
    if target.shape != (len(fit),) or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Reference prices must be aligned, finite and positive")
    area = numeric(fit, "building_area")
    lower, upper = PARAMETERS["reference_building_area_bounds"]
    reference = np.isfinite(area) & (area >= lower) & (area <= upper)
    if int(reference.sum()) < 100:
        raise ValueError("Too few usable neighborhood reference transactions")
    unit_price = np.divide(target, area, out=np.full(len(area), np.nan), where=area > 0)
    low_unit, high_unit = np.quantile(unit_price[reference], PARAMETERS["reference_unit_price_quantile_bounds"])
    reference &= (unit_price >= low_unit) & (unit_price <= high_unit)
    fit_distance, fit_hedonic_x, fill_values = matrices(fit)
    fit_keys = group_keys(fit)
    with threadpool_limits(limits=3):
        beta, hedonic_report = fit_hedonic(
            fit, target, area, fit_hedonic_x, reference, fit_keys["project"]
        )
    residual = np.full(len(fit), np.nan)
    residual[reference] = np.log(unit_price[reference]) - fit_hedonic_x[reference] @ beta
    minimums = {
        "project": 6, "address": 8, "street_section": 20,
        "locality": 20, "street": 20, "district_style": 80,
    }
    reference_index = np.flatnonzero(reference)
    groups = {}
    for level, minimum in minimums.items():
        fit_key = fit_keys[level].iloc[reference_index]
        level_groups = pd.Series(reference_index, index=fit_key.to_numpy()).groupby(level=0).apply(
            lambda series: series.to_numpy(dtype="int64")
        ).to_dict()
        groups[level] = {key: value for key, value in level_groups.items() if len(value) >= minimum}
    return {
        "format_version": 1,
        "parameters": json.loads(json.dumps(PARAMETERS)),
        "required_feature_columns": required,
        "reference_distance": fit_distance,
        "reference_residual": residual,
        "fill_values": fill_values,
        "beta": beta,
        "groups": groups,
        "fit_report": {
            "reference_rows": int(reference.sum()), "fitting_rows": len(fit),
            "reference_unit_price_bounds": [float(low_unit), float(high_unit)],
            "hedonic": hedonic_report,
            "fit_seconds": time.perf_counter() - started,
            "validation_targets_used_for_references": False,
        },
    }


def _predict_reference_all(model, X_new, baseline_predictions):
    if model.get("format_version") != 1 or model.get("parameters") != PARAMETERS:
        raise ValueError("Reference model format/parameters do not match this implementation")
    missing = [name for name in model["required_feature_columns"] if name not in X_new]
    if missing:
        raise ValueError(f"Missing inference feature columns: {missing}")
    started = time.perf_counter()
    query = X_new.reset_index(drop=True)
    baseline = np.asarray(baseline_predictions, dtype="float64")
    if baseline.shape != (len(query),) or not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Base predictions must be aligned, finite, nonnegative prices in TWD")
    lower, upper = PARAMETERS["reference_building_area_bounds"]
    query_area = numeric(query, "building_area")
    eligible_query = np.isfinite(query_area) & (query_area >= lower) & (query_area <= upper)
    query_distance, query_hedonic_x, _ = matrices(query, model["fill_values"])
    query_keys = group_keys(query)
    beta = model["beta"]
    local_log_price = np.full(len(query), np.nan)
    confidence = np.zeros(len(query))
    source = np.full(len(query), "base_fallback", dtype=object)
    nearest_distance = np.full(len(query), np.nan)
    dispersion = np.full(len(query), np.nan)
    for level, groups in model["groups"].items():
        unresolved = eligible_query & np.isnan(local_log_price)
        query_index = np.flatnonzero(unresolved & query_keys[level].isin(groups).to_numpy())
        assignments = pd.Series(query_index, index=query_keys[level].iloc[query_index].to_numpy())
        covered = 0
        for key, indices in assignments.groupby(level=0):
            qi = indices.to_numpy(dtype="int64")
            ri = groups[key]
            neighbor_count = min(
                PARAMETERS["neighbors_project"] if level == "project" else PARAMETERS["neighbors_other"],
                len(ri),
            )
            tree = cKDTree(model["reference_distance"][ri], leafsize=32)
            distance, positions = tree.query(query_distance[qi], k=neighbor_count, workers=3)
            if neighbor_count == 1:
                distance, positions = distance[:, None], positions[:, None]
            reference_values = model["reference_residual"][ri[positions]]
            weights = 1.0 / (0.05 + distance ** 2)
            center = weighted_median(reference_values, weights)
            mad = weighted_median(np.abs(reference_values - center[:, None]), weights)
            local_log_price[qi] = center + query_hedonic_x[qi] @ beta + np.log(query_area[qi])
            nearest_distance[qi] = distance[:, 0]
            dispersion[qi] = mad
            confidence[qi] = (
                PARAMETERS["neighborhood_base_weights"][level] /
                (1.0 + (distance[:, 0] / PARAMETERS["distance_confidence_scale"]) ** 2) *
                np.exp(-mad / PARAMETERS["dispersion_log_scale"])
            )
            source[qi] = level
            covered += len(qi)
        print(json.dumps({"level": level, "assigned_rows": covered, "reference_groups": len(groups)}), flush=True)
    log_base = np.log(np.maximum(baseline, 1.0))
    correction = np.nan_to_num(local_log_price - log_base, nan=0.0)
    bounds = np.log(PARAMETERS["local_to_base_price_ratio_bounds"])
    local_price = np.maximum(baseline, 1.0) * np.exp(np.clip(correction, bounds[0], bounds[1]))
    anchor = (1.0 - confidence) * baseline + confidence * local_price
    future_days = np.maximum(
        np.nan_to_num(
            numeric(query, "trade_day_index") -
            model["fit_report"]["hedonic"]["maximum_fit_day_index"], nan=0.0
        ), 0.0
    )
    future_days = np.minimum(future_days, PARAMETERS["trend_future_day_cap"])
    trend_base = baseline * np.exp(beta[3] * future_days / 365.25)
    trend = (1.0 - confidence) * trend_base + confidence * local_price
    predictions = {NAMES[0]: np.maximum(anchor, 0.0), NAMES[1]: np.maximum(trend, 0.0)}
    for name, values in predictions.items():
        if values.shape != (len(query),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid candidate predictions: {name}")
    diagnostics = pd.DataFrame({
        "source": source, "confidence": confidence,
        "nearest_distance": nearest_distance, "reference_dispersion": dispersion,
    })
    report = {
        **model["fit_report"],
        "query_rows": len(query), "eligible_query_rows": int(eligible_query.sum()),
        "source_counts": pd.Series(source).value_counts().to_dict(),
        "mean_reference_weight": float(confidence.mean()),
        "seconds": time.perf_counter() - started,
    }
    return predictions, diagnostics, report


def predict_reference(model, X_new, baseline_predictions, *, variant="trend",
                      return_diagnostics=False):
    """Apply a saved model to fresh features and the v1 base float predictions.

    baseline_predictions must equal 0.75 * CatBoost-raw + 0.25 * old
    LightGBM-unit (each model's prediction clipped at zero before mixing).
    Accepts variant='anchor'/'trend' or either complete candidate name.
    Returns a one-dimensional float64 vector in TWD; no rounding is applied.
    Optional return_diagnostics=True returns (prediction, diagnostics, report).
    """
    aliases = {"anchor": NAMES[0], "trend": NAMES[1], NAMES[0]: NAMES[0], NAMES[1]: NAMES[1]}
    if variant not in aliases:
        raise ValueError(f"Unknown neighborhood variant: {variant}")
    with threadpool_limits(limits=3):
        predictions, diagnostics, report = _predict_reference_all(model, X_new, baseline_predictions)
    prediction = predictions[aliases[variant]]
    return (prediction, diagnostics, report) if return_diagnostics else prediction


apply_reference = predict_reference


def local_predictions(fit, target, query, baseline):
    """Fit on the supplied fitting-side labels and predict both candidates."""
    started = time.perf_counter()
    model = fit_final_reference(fit, target)
    predictions, diagnostics, report = _predict_reference_all(model, query, baseline)
    report["seconds"] = time.perf_counter() - started
    return predictions, diagnostics, report


def base_predictions(stage, expected_length, expected_indices=None):
    combined = np.zeros(expected_length, dtype="float64")
    inputs = {}
    for name, weight in BASE_COMPONENTS:
        path = ROOT / "outputs" / "experiments" / (
            f"{name}_temporal_valid.npy" if stage == "temporal" else f"{name}_valid.npy"
        )
        prediction = np.load(path, allow_pickle=False)
        if prediction.shape != (expected_length,) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid archived base predictions: {path}")
        if stage == "temporal":
            index_path = path.with_name(f"{name}_temporal_indices.npy")
            if not np.array_equal(np.load(index_path, allow_pickle=False), expected_indices):
                raise ValueError("Archived base temporal indices are not aligned")
            inputs[index_path.relative_to(ROOT).as_posix()] = sha256(index_path)
        combined += weight * prediction
        inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    return combined, inputs


def score(target, prediction, seen, test_unseen_rate):
    seen_mae = mae(target[seen], prediction[seen]) if seen.any() else None
    unseen_mae = mae(target[~seen], prediction[~seen]) if (~seen).any() else None
    return {
        "mae": mae(target, prediction),
        "seen_project_mae": seen_mae, "unseen_project_mae": unseen_mae,
        "seen_project_rows": int(seen.sum()), "unseen_project_rows": int((~seen).sum()),
        "reweighted_temporal_mae": (
            (1-test_unseen_rate)*seen_mae + test_unseen_rate*unseen_mae
            if seen_mae is not None and unseen_mae is not None else None
        ),
    }


def update_manifest():
    entries = []
    for name in NAMES:
        path = OUT / f"{name}.json"
        if not path.exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        stages = metadata.get("stages", {})
        entries.append({
            "name": name,
            "valid_predictions": stages.get("valid", {}).get("predictions_path"),
            "temporal_predictions": stages.get("temporal", {}).get("predictions_path"),
            "temporal_indices": stages.get("temporal", {}).get("indices_path"),
            "metadata_path": path.relative_to(ROOT).as_posix(),
        })
    json_write(OUT / "candidates.json", entries)


def run_stage(data, stage, candidate_names):
    if stage == "temporal":
        fitting_indices, query_indices = temporal_indices(data, "2024-04-01", end="2024-08-10")
        fit = data["X_train"].iloc[fitting_indices]
        target_fit = data["y_train"][fitting_indices]
        query = data["X_train"].iloc[query_indices]
        target_query = data["y_train"][query_indices]
    else:
        fitting_indices = np.arange(len(data["X_train"]))
        query_indices = np.arange(len(data["X_valid"]))
        fit, target_fit = data["X_train"], data["y_train"]
        query, target_query = data["X_valid"], data["y_valid"]
    baseline, base_hashes = base_predictions(stage, len(query), query_indices if stage == "temporal" else None)
    known = set(fit["project_name"]) - {MISSING}
    seen = query["project_name"].isin(known).to_numpy()
    final_known = (set(data["X_train"]["project_name"]) | set(data["X_valid"]["project_name"])) - {MISSING}
    test_unseen_rate = float((~data["X_test"]["project_name"].isin(final_known)).mean())
    print(json.dumps({"stage": stage, "fit_rows": len(fit), "query_rows": len(query),
                      "base_metrics": score(target_query, baseline, seen, test_unseen_rate)}), flush=True)
    with threadpool_limits(limits=3):
        predictions, diagnostics, reference_report = local_predictions(fit, target_fit, query, baseline)

    indices_path = OUT / f"{stage}_indices.npy"
    np.save(indices_path, query_indices)
    diagnostics.to_csv(OUT / f"{stage}_reference_diagnostics.csv", index=False)
    json_write(OUT / f"{stage}_reference_fit.json", reference_report)
    script_hash = sha256(Path(__file__))
    for name in candidate_names:
        prediction = predictions[name]
        path = OUT / f"{name}_{stage}_predictions.npy"
        np.save(path, prediction)
        metrics = score(target_query, prediction, seen, test_unseen_rate)
        if stage == "valid":
            clean = data["valid_clean_mask"]
            metrics["clean_mae"] = mae(target_query[clean], prediction[clean])
        metrics.update({
            "rows": len(query),
            "predictions_path": path.relative_to(ROOT).as_posix(),
            "indices_path": indices_path.relative_to(ROOT).as_posix(),
            "base_metrics": score(target_query, baseline, seen, test_unseen_rate),
            "reference_fit_path": (OUT / f"{stage}_reference_fit.json").relative_to(ROOT).as_posix(),
            "fit_rows": len(fit),
            "base_artifact_sha256": base_hashes,
        })
        metadata_path = OUT / f"{name}.json"
        old = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        stages = old.get("stages", {}) if old.get("script_sha256") == script_hash else {}
        stages[stage] = metrics
        metadata = {
            "name": name, "family": "local_hedonic_reference",
            "parameters": PARAMETERS,
            "trend_extrapolation": name == NAMES[1],
            "base_components": [{"name": n, "weight": w} for n, w in BASE_COMPONENTS],
            "stages": stages,
            "test_unseen_project_rate": test_unseen_rate,
            "source_sha256": data["metadata"]["source_sha256"],
            "feature_source_sha256": sha256(ROOT / "src" / "house_features.py"),
            "script_sha256": script_hash,
            "constraints": {
                "no_external_property_data": True, "no_government_answer_lookup": True,
                "reference_labels_from_fit_split_only": True, "no_test_price_prediction": True,
                "no_train_plus_valid_refit": True,
            },
            "reproduce": ".venv/Scripts/python.exe src/round2_neighborhood.py --stage " + stage,
        }
        json_write(metadata_path, metadata)
        print(json.dumps({"name": name, "stage": stage, **metrics}), flush=True)
    update_manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("temporal", "valid", "all"), default="temporal")
    parser.add_argument("--names", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for stage in (("temporal", "valid") if args.stage == "all" else (args.stage,)):
        run_stage(data, stage, args.names)


if __name__ == "__main__":
    main()
