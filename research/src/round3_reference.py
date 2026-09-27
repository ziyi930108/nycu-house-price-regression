"""Fixed, strongly shrunk city-specific FUTURE-baseline trend for round three.

No files containing data are read by the public fit/predict APIs. The frozen
v2 reference, its local normalization, nearest neighbors and confidence remain
unchanged. Only the extra future trend of the remaining baseline share changes.
All constants were fixed before development/confirmation scoring.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "3"

import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from round2_neighborhood import (
    PARAMETERS as V2_PARAMETERS, MISSING, group_keys, matrices, numeric, predict_reference,
)

REGIONAL_PARAMETERS = {
    "lookback_years": 4.0,
    "minimum_project_observations": 8,
    "minimum_city_projects": 30,
    "minimum_city_rows": 1000,
    "shrinkage_project_prior": 200.0,
    "maximum_city_weight": 0.5,
    "huber_log_threshold": 0.10,
    "irls_iterations": 6,
    "gram_ridge": 1e-5,
    "future_day_cap": 366.0,
    "hedonic_features": ["log_building_area", "parking_fraction", "floor_mean_per10", "transaction_years"],
    "coefficient_lower": [-0.7, -2.0, -0.05, -0.08],
    "coefficient_upper": [0.3, 0.5, 0.15, 0.20],
    "cpu_threads": 3,
}


def _reference_identity(model):
    # Fingerprints the supplied in-memory reference, including arrays and group members.
    return hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()


def _source_hashes():
    directory = Path(__file__).resolve().parent
    names = ("round3_reference.py", "round2_neighborhood.py", "house_features.py")
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}


def _frame_fingerprint(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def fit_regional_trend(Xfit: pd.DataFrame, yfit, reference_model: dict) -> dict:
    """Fit the one predeclared regional extension from fitting-side inputs only.

    Xfit/yfit must be the exact aligned rows used for reference_model. Returns
    a JSON/pickle serializable plain dictionary. It never reads load_data,
    test/validation labels, a file of labels, or any public leaderboard score.
    """
    if reference_model.get("format_version") != 1 or reference_model.get("parameters") != V2_PARAMETERS:
        raise ValueError("Expected the frozen v2 reference format and parameters.")
    if "編號" in Xfit or "總價元" in Xfit:
        raise ValueError("Xfit must contain model features, without ID/target.")
    fit = Xfit.reset_index(drop=True)
    y = np.asarray(yfit, dtype="float64")
    if y.shape != (len(fit),) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Fitting labels must be aligned, finite and positive.")
    if len(fit) != reference_model["fit_report"]["fitting_rows"]:
        raise ValueError("Fitting rows differ from the supplied reference.")
    dates = numeric(fit, "trade_day_index")
    end_day = float(np.nanmax(dates))
    expected_end = reference_model["fit_report"]["hedonic"]["maximum_fit_day_index"]
    if end_day != expected_end:
        raise ValueError("Fitting dates differ from the supplied reference.")
    global_beta = float(reference_model["beta"][3])
    area = numeric(fit, "building_area")
    unit = np.divide(y, area, out=np.full(len(y), np.nan), where=area > 0)
    low, high = reference_model["fit_report"]["reference_unit_price_bounds"]
    area_low, area_high = V2_PARAMETERS["reference_building_area_bounds"]
    eligible = (
        np.isfinite(dates) & (dates <= end_day)
        & (dates >= end_day-REGIONAL_PARAMETERS["lookback_years"]*365.25)
        & (area >= area_low) & (area <= area_high) & (unit >= low) & (unit <= high)
        & fit.project_name.ne(MISSING).to_numpy()
    )
    ix = np.flatnonzero(eligible)
    _, hedonic, _ = matrices(fit, reference_model["fill_values"])
    work = pd.DataFrame(hedonic[ix], columns=["log_area", "parking", "floor", "year"])
    work["target"] = np.log(unit[ix])
    work["group"] = group_keys(fit)["project"].iloc[ix].to_numpy()
    work["city"] = fit.address_city.iloc[ix].to_numpy()
    counts = work.groupby("group").target.transform("size")
    work = work.loc[counts >= REGIONAL_PARAMETERS["minimum_project_observations"]].copy()
    names = ["log_area", "parking", "floor", "year", "target"]
    centered = work[names] - work.groupby("group")[names].transform("mean")
    city_records = {}
    with threadpool_limits(limits=REGIONAL_PARAMETERS["cpu_threads"]):
        for city in sorted(fit.address_city.astype(str).unique()):
            mask = work.city.eq(city).to_numpy()
            rows = int(mask.sum())
            groups = int(work.loc[mask, "group"].nunique())
            record = {
                "rows": rows, "projects": groups, "supported": False,
                "fallback_reason": None, "raw_coefficients": None, "clipped_coefficients": None,
                "city_time_beta": global_beta, "shrinkage_weight": 0.0,
                "effective_time_beta": global_beta,
            }
            if city == MISSING:
                record["fallback_reason"] = "missing_city"
            elif rows < REGIONAL_PARAMETERS["minimum_city_rows"]:
                record["fallback_reason"] = "insufficient_rows"
            elif groups < REGIONAL_PARAMETERS["minimum_city_projects"]:
                record["fallback_reason"] = "insufficient_projects"
            else:
                design = centered.loc[mask, names[:4]].to_numpy(dtype="float64")
                response = centered.loc[mask, "target"].to_numpy(dtype="float64")
                weights = np.ones(rows, dtype="float64")
                beta = np.zeros(4, dtype="float64")
                for _ in range(REGIONAL_PARAMETERS["irls_iterations"]):
                    gram = design.T @ (weights[:, None]*design) + np.eye(4)*REGIONAL_PARAMETERS["gram_ridge"]
                    beta = np.linalg.solve(gram, design.T @ (weights*response))
                    residual = response-design@beta
                    weights = np.minimum(1., REGIONAL_PARAMETERS["huber_log_threshold"] /
                                         np.maximum(np.abs(residual), 1e-8))
                clipped = np.clip(beta, REGIONAL_PARAMETERS["coefficient_lower"],
                                  REGIONAL_PARAMETERS["coefficient_upper"])
                weight = min(REGIONAL_PARAMETERS["maximum_city_weight"],
                             groups/(groups+REGIONAL_PARAMETERS["shrinkage_project_prior"]))
                record.update({
                    "supported": True, "raw_coefficients": [float(v) for v in beta],
                    "clipped_coefficients": [float(v) for v in clipped],
                    "city_time_beta": float(clipped[3]), "shrinkage_weight": float(weight),
                    "effective_time_beta": float((1-weight)*global_beta + weight*clipped[3]),
                    "condition_number": float(np.linalg.cond(gram)),
                })
            city_records[str(city)] = record
    result = {
        "format_version": 1, "family": "strongly_shrunk_city_future_trend",
        "parameters": json.loads(json.dumps(REGIONAL_PARAMETERS)),
        "global_beta": global_beta, "maximum_fit_day_index": end_day,
        "training_rows": len(fit), "eligible_rows_before_project_minimum": int(eligible.sum()),
        "eligible_rows_after_project_minimum": len(work),
        "eligible_projects": int(work.group.nunique()),
        "reference_unit_price_bounds": [float(low), float(high)],
        "source_sha256": _source_hashes(),
        "fit_features_sha256": _frame_fingerprint(fit),
        "fit_targets_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "reference_identity_sha256": _reference_identity(reference_model),
        "cities": city_records,
        "constraints": {"fit_inputs_only": True, "query_labels_used": False,
                        "public_score_used": False, "v2_local_reference_unchanged": True,
                        "thresholds_selected_on_validation": False},
    }
    json.dumps(result, allow_nan=False)
    return result


def predict_regional_trend(reference_model: dict, regional_model: dict,
                           Xquery: pd.DataFrame, baseline) -> np.ndarray:
    """Return nonnegative float TWD predictions without fitting or rounding.

    baseline must be the nonnegative, unrounded base from the predeclared
    candidate (v1 or recency blend) of this split.
    The result equals anchor + the extra future trend on the non-reference
    share. Missing/unsupported/unseen cities fall back to the v2 global beta.
    """
    if regional_model.get("format_version") != 1 or regional_model.get("parameters") != REGIONAL_PARAMETERS:
        raise ValueError("Regional model format/parameters differ from the fixed candidate.")
    if regional_model["reference_identity_sha256"] != _reference_identity(reference_model):
        raise ValueError("Regional model was fitted for a different reference.")
    if regional_model["source_sha256"] != _source_hashes():
        raise ValueError("Regional inference source changed after fitting.")
    base = np.asarray(baseline, dtype="float64")
    if base.shape != (len(Xquery),) or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Baseline must be aligned, finite, nonnegative float prices.")
    anchor, diagnostics, _ = predict_reference(
        reference_model, Xquery, base, variant="anchor", return_diagnostics=True)
    mapping = {city: record["effective_time_beta"] for city, record in regional_model["cities"].items()}
    beta = Xquery.address_city.astype(str).map(mapping).fillna(regional_model["global_beta"]).to_numpy(dtype="float64")
    future = np.maximum(np.nan_to_num(
        numeric(Xquery, "trade_day_index")-regional_model["maximum_fit_day_index"], nan=0.), 0.)
    future = np.minimum(future, REGIONAL_PARAMETERS["future_day_cap"])
    prediction = anchor + (1-diagnostics.confidence.to_numpy())*base*np.expm1(beta*future/365.25)
    prediction = np.maximum(np.asarray(prediction, dtype="float64"), 0.)
    if prediction.shape != base.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid regional predictions.")
    return prediction


def self_check():
    """Synthetic-only smoke checks; no real validation or test data are read."""
    from round2_neighborhood import fit_final_reference
    rng = np.random.default_rng(42)
    n = 2400
    project = np.arange(n) % 60
    city_a = project < 30
    dates = rng.integers(7000, 8450, size=n).astype("float64")
    area = rng.uniform(60, 220, size=n)
    parking = rng.uniform(.05, .35, size=n)
    floor = rng.integers(2, 25, size=n).astype("float64")
    x = pd.DataFrame({
        "building_area": area, "parking_fraction": parking, "floor_mean": floor,
        "total_floors": np.full(n, 25.), "rooms": rng.integers(2, 5, size=n).astype(float),
        "land_building_ratio": np.full(n, .15), "trade_day_index": dates,
        "parking_count": np.ones(n), "address_city": np.where(city_a, "測試甲市", "測試乙市"),
        "address_district": np.full(n, "測試區"), "building_type": np.full(n, "住宅大樓"),
        "main_use": np.full(n, "住家用"), "project_name": [f"synthetic_{p:02d}" for p in project],
        "address_road": np.full(n, "測試路"), "address_locality": np.full(n, MISSING),
        "address_raw": [f"測試路{p}號" for p in project], "address_section": np.full(n, "一段"),
    })
    y = area*np.exp(11.6 + project*.003 - .03*np.log(area) - .5*parking
                    + .04*floor/10 + np.where(city_a, .035, .105)*(dates-8000)/365.25)
    reference = fit_final_reference(x, y)
    before = _reference_identity(reference)
    model = fit_regional_trend(x, y, reference)
    assert _reference_identity(reference) == before
    assert all(r["supported"] for r in model["cities"].values())
    for record in model["cities"].values():
        expected = min(.5, record["projects"]/(record["projects"]+200.))
        assert record["shrinkage_weight"] == expected
        assert np.isclose(record["effective_time_beta"],
                          (1-expected)*model["global_beta"]+expected*record["city_time_beta"])
    small_x, small_y = x.iloc[:500].copy(), y[:500].copy()
    small_ref = fit_final_reference(small_x, small_y)
    fallback = fit_regional_trend(small_x, small_y, small_ref)
    assert all(r["shrinkage_weight"] == 0 for r in fallback["cities"].values())
    query = small_x.iloc[:12].copy().reset_index(drop=True)
    query["trade_day_index"] = np.nanmax(small_x.trade_day_index)+np.tile([1., 60., 180., 355.], 3)
    query.loc[0, "address_city"] = "未見城市"
    base = small_y[:12]
    old = predict_reference(small_ref, query, base, variant="trend")
    new = predict_regional_trend(small_ref, fallback, query, base)
    max_diff = float(np.max(np.abs(old-new)))
    assert np.allclose(old, new, rtol=1e-12, atol=1e-7)
    assert np.isfinite(predict_regional_trend(reference, model, query, base)).all()
    json.dumps(model, allow_nan=False)
    return {"status": "passed", "synthetic_rows": n, "supported_city_rule_verified": True,
            "fallback_max_abs_difference_TWD": max_diff, "real_data_used": False,
            "predictive_candidate_evaluation_performed": False}


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False))
