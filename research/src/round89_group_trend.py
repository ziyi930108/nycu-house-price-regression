"""R89 fit-only type/use time slopes with coherent reference/spatial views.

Only this wrapper is a complete model. Template global beta is a fallback,
not the coefficient to use for supported groups. No query labels or file data
are accepted by fit/predict; source files are read only for identity checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd

import round2_neighborhood as reference_api
import round8_spatial_reference as spatial_api

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "type_use_shrunk_time_coherent_reference_spatial"
PARAMETERS = {
    "format_version": 1, "group_columns": ["building_type", "main_use"],
    "lookback_years": 4.0, "days_per_year": 365.25,
    "minimum_project_rows": 8, "minimum_project_span_days": 180.0,
    "minimum_group_rows": 1000, "minimum_independent_projects": 30,
    "minimum_group_span_days": 730.0, "irls_iterations": 6,
    "huber_log_threshold": .10, "gram_ridge": 1e-5,
    "initial_row_weight": "1/reference_project_support_rows",
    "each_irls_weight": "initial_row_weight * Huber_factor",
    "response": "centered_log_unit - centered_hedonic @ global_beta",
    "predictor": "centered_transaction_years", "fit_parameter": "time_delta_only",
    "time_coefficient_bounds": [-.08, .20], "shrinkage_project_prior": 200.0,
    "maximum_group_weight": .5, "future_day_cap": 366.0,
    "local_hedonic_date": "original_unclipped_query_date_with_original_fit_fill",
    "fallback": "original_full_query_global_reference_and_spatial_bitwise",
    "residual_update_scope": "all_original_finite_reference_rows",
    "query_labels_used": False,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def parameters():
    return copy.deepcopy(PARAMETERS)


def source_hashes():
    return {"src/" + name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in ("round89_group_trend.py", "round2_neighborhood.py",
                         "round8_spatial_reference.py", "round6_geo.py", "house_features.py")}


def values_sha256(a):
    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()


def state_identity(value):
    """Value identity independent of pickle memoization and ndarray strides."""
    digest = hashlib.sha256()

    def part(tag, data=b""):
        digest.update(tag); digest.update(struct.pack("<Q", len(data))); digest.update(data)

    def visit(x):
        if x is None:
            part(b"N")
        elif isinstance(x, (bool, np.bool_)):
            part(b"B", bytes([int(x)]))
        elif isinstance(x, (int, np.integer)):
            part(b"I", str(int(x)).encode())
        elif isinstance(x, (float, np.floating)):
            part(b"F", struct.pack("<d", float(x)))
        elif isinstance(x, str):
            part(b"S", x.encode("utf-8"))
        elif isinstance(x, np.ndarray):
            part(b"A", str(x.dtype).encode()); visit(tuple(x.shape))
            if x.dtype.hasobject:
                for v in x.flat: visit(v)
            else:
                part(b"V", np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            part(b"D", str(len(x)).encode())
            for k in sorted(x, key=lambda z: (type(z).__name__, repr(z))):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            part(b"T" if isinstance(x, tuple) else b"L", str(len(x)).encode())
            for v in x: visit(v)
        else:
            raise TypeError("Unsupported state value: " + type(x).__name__)
    visit(value)
    return digest.hexdigest()


def _style_keys(X):
    columns = [X[c].fillna(reference_api.MISSING).astype(str).to_numpy()
               for c in PARAMETERS["group_columns"]]
    keys = [tuple(v) for v in zip(*columns)]
    return keys


def _known_group(key):
    return all(v.strip() not in ("", reference_api.MISSING, "nan", "None", "<NA>", "NaT") for v in key)


def _check_group_purity(reference, spatial, style_keys):
    for level, groups in reference["groups"].items():
        for key, rows in groups.items():
            need(len({style_keys[int(i)] for i in rows}) == 1,
                 "Reference group mixes type/use: " + level + " " + str(key))
    for key, group in spatial["groups"].items():
        keys = {style_keys[int(i)] for rows in group["row_positions"] for i in rows}
        need(len(keys) == 1, "Spatial group mixes raw type/use (possible normalization collision): " + str(key))


def _fit_delta(predictor, response, project):
    counts = pd.Series(project).groupby(pd.Series(project), sort=False).transform("size").to_numpy(dtype="float64")
    initial = 1.0 / counts
    weights = initial.copy()
    delta = 0.0
    information = float(np.sum(initial * predictor * predictor))
    need(np.isfinite(information) and information > 0, "Insufficient within-project time information")
    for _ in range(PARAMETERS["irls_iterations"]):
        numerator = float(np.sum(weights * predictor * response))
        denominator = float(np.sum(weights * predictor * predictor)) + PARAMETERS["gram_ridge"]
        delta = numerator / denominator
        residual = response - predictor * delta
        huber = np.minimum(1., PARAMETERS["huber_log_threshold"] / np.maximum(np.abs(residual), 1e-8))
        weights = initial * huber
    totals = pd.Series(initial).groupby(pd.Series(project), sort=False).sum().to_numpy()
    need(np.allclose(totals, 1., rtol=0, atol=1e-12), "Initial project weights do not sum to one")
    return float(delta), {"initial_project_weight_min": float(totals.min()),
                         "initial_project_weight_max": float(totals.max()),
                         "initial_weight_sum": float(initial.sum()),
                         "final_weight_sum": float(weights.sum()),
                         "initial_time_information": information}


def _centered_work(work, beta):
    names = ["log_area", "parking", "floor", "time", "log_unit"]
    centered = work[names] - work.groupby("project")[names].transform("mean")
    design = centered[names[:4]].to_numpy(dtype="float64")
    response = centered["log_unit"].to_numpy(dtype="float64") - design @ beta
    return design, response


def _content_identity(model):
    return state_identity({k: v for k, v in model.items() if k != "state_content_sha256"})


def validate_model(model):
    need(model.get("format_version") == 1 and model.get("family") == FAMILY, "Not an R89 wrapper")
    need(model.get("parameters") == PARAMETERS, "R89 fixed parameters changed")
    need(model.get("source_sha256") == source_hashes(), "R89 inference source changed")
    for name in ("global_reference", "global_spatial", "reference_template", "spatial_template"):
        need(state_identity(model[name]) == model["component_identities"][name], "Component state changed: " + name)
    need(model.get("state_content_sha256") == _content_identity(model), "Wrapper content changed")
    global_beta = np.asarray(model["global_reference"]["beta"], dtype="float64")
    need(global_beta.shape == (4,) and np.isfinite(global_beta).all(), "Invalid global beta")
    need(np.array_equal(model["reference_template"]["beta"], global_beta), "Template beta must be explicit global fallback")
    need(np.array_equal(model["spatial_template"]["beta"], global_beta), "Spatial template beta must be explicit global fallback")
    need(values_sha256(model["reference_template"]["reference_residual"]) ==
         model["spatial_template"]["state_summary"]["existing_reference_residual_sha256"], "New R/S residual identity differs")
    for key, record in model["group_trends"].items():
        need(isinstance(key, tuple) and len(key) == 2, "Invalid type/use key")
        if record["supported"]:
            alpha = min(.5, record["independent_projects"] / (record["independent_projects"] + 200.))
            effective = (1-alpha) * global_beta[3] + alpha * record["clipped_gamma"]
            need(record["shrinkage_weight"] == alpha and record["effective_gamma"] == effective, "Group shrinkage changed")
            need(-.08 <= record["clipped_gamma"] <= .20, "Clipped time coefficient out of bounds")
        else:
            need(record["effective_gamma"] == global_beta[3] and record["effective_delta"] == 0., "Unsupported group did not fall back")
        need(record["effective_delta"] == record["effective_gamma"] - global_beta[3], "Inconsistent effective delta")
    return True


def fit_model(X, y, G, reference, spatial_state):
    """Estimate time deltas from fit inputs and rebuild coherent spatial state."""
    started = time.monotonic()
    schema = spatial_api._schema(X, G)
    need(reference.get("format_version") == 1 and reference.get("parameters") == reference_api.PARAMETERS, "Unexpected global reference")
    need(spatial_state.get("format_version") == 1 and spatial_state.get("parameters") == spatial_api.parameters(), "Unexpected global spatial state")
    need(spatial_state["schema"] == schema and reference["fit_report"]["fitting_rows"] == len(X), "Fit schema or row count differs")
    target = np.asarray(y, dtype="float64")
    need(target.shape == (len(X),) and np.isfinite(target).all() and (target > 0).all(), "Invalid fit targets")
    before = {"X": spatial_api.frame_sha(X), "G": spatial_api.frame_sha(G),
              "y": values_sha256(target), "reference": state_identity(reference), "spatial": state_identity(spatial_state)}
    summary = spatial_state["state_summary"]
    need(summary["fit_X_sha256"] == before["X"] and summary["fit_geo_sha256"] == before["G"], "Global spatial fit frames differ")
    old_residual = np.asarray(reference["reference_residual"], dtype="float64")
    beta = np.asarray(reference["beta"], dtype="float64")
    need(values_sha256(old_residual) == summary["existing_reference_residual_sha256"], "Global R/S residual source mismatch")
    need(np.array_equal(beta, spatial_state["beta"]) and np.array_equal(reference["fill_values"], spatial_state["fill_values"]), "Global R/S normalization differs")
    dates = reference_api.numeric(X, "trade_day_index")
    area = reference_api.numeric(X, "building_area")
    anchor = float(reference["fit_report"]["hedonic"]["maximum_fit_day_index"])
    need(anchor == spatial_state["anchor_day_index"] == float(np.nanmax(dates)), "Fit anchor differs")
    _, H, _ = reference_api.matrices(X, reference["fill_values"])
    finite = np.isfinite(old_residual)
    need(old_residual.shape == (len(X),) and beta.shape == (4,) and np.isfinite(beta).all(), "Global array schema differs")
    need(np.isfinite(area[finite]).all() and (area[finite] > 0).all(), "Invalid reference area")
    log_unit = np.full(len(X), np.nan)
    log_unit[finite] = np.log(target[finite] / area[finite])
    need(np.allclose(log_unit[finite] - (H @ beta)[finite], old_residual[finite], rtol=0, atol=1e-10), "Fit targets disagree with global reference residuals")
    keys = _style_keys(X)
    _check_group_purity(reference, spatial_state, keys)
    project = reference_api.group_keys(X)["project"]
    independent, named = spatial_api._project_keys(X)
    eligible = finite & np.isfinite(dates) & (dates >= anchor - 4*365.25) & (dates <= anchor) & named & project.notna().to_numpy()
    positions = np.flatnonzero(eligible)
    work = pd.DataFrame(H[positions], columns=["log_area", "parking", "floor", "time"])
    work["log_unit"] = log_unit[positions]
    work["project"] = project.iloc[positions].to_numpy()
    work["independent_project"] = independent[positions]
    work["type_use"] = [keys[int(i)] for i in positions]
    work["date"] = dates[positions]
    counts = work.groupby("project").date.transform("size")
    spans = work.groupby("project").date.transform("max") - work.groupby("project").date.transform("min")
    work = work.loc[(counts >= 8) & (spans >= 180)].copy()
    records = {}
    for key in sorted(set(keys)):
        mask = work.type_use.map(lambda v: v == key).to_numpy(dtype=bool)
        sub = work.loc[mask]
        rows = len(sub); projects = int(sub.independent_project.nunique())
        span = float(sub.date.max() - sub.date.min()) if rows else 0.
        record = {"rows": rows, "reference_projects": int(sub.project.nunique()), "independent_projects": projects,
                  "span_days": span, "supported": False, "fallback_reason": None,
                  "fitted_delta": None, "raw_gamma": None, "clipped_gamma": None,
                  "shrinkage_weight": 0., "effective_gamma": float(beta[3]), "effective_delta": 0.}
        if not _known_group(key): record["fallback_reason"] = "missing_type_or_use"
        elif rows < 1000: record["fallback_reason"] = "insufficient_supported_rows"
        elif projects < 30: record["fallback_reason"] = "insufficient_independent_projects"
        elif span < 730: record["fallback_reason"] = "insufficient_group_date_span"
        else:
            design, response = _centered_work(sub, beta)
            if not np.isfinite(design).all() or not np.isfinite(response).all() or np.sum(design[:, 3] ** 2) <= 0:
                record["fallback_reason"] = "invalid_or_zero_time_information"
            else:
                delta, weight_report = _fit_delta(design[:, 3], response, sub.project.to_numpy())
                raw = float(beta[3] + delta); clipped = float(np.clip(raw, -.08, .20))
                alpha = float(min(.5, projects / (projects + 200.)))
                gamma = float((1-alpha)*beta[3] + alpha*clipped)
                record.update(supported=True, fitted_delta=delta, raw_gamma=raw, clipped_gamma=clipped,
                              shrinkage_weight=alpha, effective_gamma=gamma, effective_delta=float(gamma-beta[3]),
                              weight_report=weight_report)
        records[key] = record
    delta = np.asarray([records[k]["effective_delta"] for k in keys], dtype="float64")
    new_residual = old_residual.copy()
    changed = finite & (delta != 0.)
    new_residual[changed] = old_residual[changed] - H[changed, 3] * delta[changed]
    need(np.isfinite(new_residual).tolist() == finite.tolist(), "Finite reference mask changed")
    template = reference.copy()
    template["beta"] = beta.copy()
    template["reference_residual"] = new_residual
    new_spatial, _ = spatial_api.build(template, X, G)
    _check_group_purity(template, new_spatial, keys)
    need(new_spatial["state_summary"]["selected_fit_row_positions_sha256"] == summary["selected_fit_row_positions_sha256"], "Spatial eligible rows changed")
    components = {"global_reference": reference, "global_spatial": spatial_state,
                  "reference_template": template, "spatial_template": new_spatial}
    metadata = {"family": FAMILY, "format_version": 1, "fit_rows": len(X), "schema": schema,
                "input_sha256": before, "anchor_day_index": anchor, "global_beta": beta.tolist(),
                "eligible_rows_before_project_support": int(eligible.sum()), "rows_after_project_support": len(work),
                "supported_group_count": sum(v["supported"] for v in records.values()), "total_groups": len(records),
                "supported_groups": [{"building_type": k[0], "main_use": k[1],
                    "support": {"rows": v["rows"], "independent_project_keys": v["independent_projects"],
                                "combined_date_span_days": v["span_days"], "r2_project_groups": v["reference_projects"]},
                    "trend": copy.deepcopy(v)} for k, v in records.items() if v["supported"]],
                "all_group_records": [{"building_type": k[0], "main_use": k[1], **copy.deepcopy(v)} for k, v in records.items()],
                "coefficient_fits": sum(v["supported"] for v in records.values()),
                "new_tree_or_NN_fits": 0, "new_time_wrapper_fits": 1, "new_spatial_builds": 1,
                "all_reference_and_spatial_groups_single_type_use": True, "query_used_in_fit": False,
                "reference_finite_rows": int(finite.sum()), "reference_residual_changed_rows": int(changed.sum()),
                "old_residual_sha256": values_sha256(old_residual), "new_residual_sha256": values_sha256(new_residual),
                "template_beta_role": "global_fallback_only_not_a_standalone_model",
                "parameters": parameters(), "source_sha256": source_hashes(), "fit_seconds": time.monotonic()-started}
    model = {"family": FAMILY, "format_version": 1, "parameters": parameters(), "source_sha256": source_hashes(),
             "schema": schema, "group_trends": records, "metadata": metadata, **components,
             "component_identities": {k: state_identity(v) for k, v in components.items()}}
    model["state_content_sha256"] = _content_identity(model)
    validate_model(model)
    need(before == {"X": spatial_api.frame_sha(X), "G": spatial_api.frame_sha(G), "y": values_sha256(target),
                    "reference": state_identity(reference), "spatial": state_identity(spatial_state)}, "fit inputs mutated")
    return model, copy.deepcopy(metadata)


def predict_model(model, X, G, base):
    """Unrounded TWD and diagnostics; no query target argument or fitted query state."""
    validate_model(model)
    need(spatial_api._schema(X, G) == model["schema"], "Query schema differs from fit")
    base = np.asarray(base, dtype="float64")
    need(base.shape == (len(X),) and np.isfinite(base).all() and (base >= 0).all(), "Invalid query base")
    X_before, G_before, base_before = spatial_api.frame_sha(X), spatial_api.frame_sha(G), values_sha256(base)
    global_r, global_rd, _ = reference_api.predict_reference(model["global_reference"], X, base, variant="trend", return_diagnostics=True)
    global_rd.index = X.index.copy()
    global_s, global_sd, _ = spatial_api.predict(model["global_spatial"], X, G, base, global_r, global_rd)
    result_r, result_s = global_r.copy(), global_s.copy()
    result_rd, result_sd = global_rd.copy(deep=True), global_sd.copy(deep=True)
    keys = _style_keys(X)
    positions = {}
    for i, key in enumerate(keys): positions.setdefault(key, []).append(i)
    supported_mask = np.asarray([bool(model["group_trends"].get(k, {}).get("supported", False)) for k in keys], dtype=bool)
    affected = np.zeros(len(X), dtype=bool)
    processed = []
    for key, rows in positions.items():
        record = model["group_trends"].get(key)
        if record is None or not record["supported"] or record["effective_delta"] == 0.: continue
        indices = np.asarray(rows, dtype="int64")
        affected[indices] = True
        q = X.iloc[indices].reset_index(drop=True); geo = G.iloc[indices].reset_index(drop=True)
        r_view = model["reference_template"].copy(); s_view = model["spatial_template"].copy()
        r_view["beta"] = r_view["beta"].copy(); s_view["beta"] = s_view["beta"].copy()
        r_view["beta"][3] = s_view["beta"][3] = record["effective_gamma"]
        pr, dr, _ = reference_api.predict_reference(r_view, q, base[indices], variant="trend", return_diagnostics=True)
        ps, ds, _ = spatial_api.predict(s_view, q, geo, base[indices], pr, dr)
        result_r[indices], result_s[indices] = pr, ps
        need(list(dr.columns) == list(result_rd.columns) and list(ds.columns) == list(result_sd.columns), "Diagnostic schema changed")
        for col in result_rd: result_rd.iloc[indices, result_rd.columns.get_loc(col)] = dr[col].to_numpy()
        for col in result_sd: result_sd.iloc[indices, result_sd.columns.get_loc(col)] = ds[col].to_numpy()
        processed.append({"type_use": list(key), "query_rows": len(rows), "effective_gamma": record["effective_gamma"],
                          "effective_delta": record["effective_delta"]})
    need(np.array_equal(result_r[~affected], global_r[~affected]) and np.array_equal(result_s[~affected], global_s[~affected]), "Fallback vectors changed")
    need(result_rd.loc[~affected].equals(global_rd.loc[~affected]) and result_sd.loc[~affected].equals(global_sd.loc[~affected]), "Fallback diagnostics changed")
    need(np.isfinite(result_r).all() and np.isfinite(result_s).all() and (result_r >= 0).all() and (result_s >= 0).all(), "Invalid R89 prices")
    need((X_before, G_before, base_before) == (spatial_api.frame_sha(X), spatial_api.frame_sha(G), values_sha256(base)), "Query inputs mutated")
    need(model["state_content_sha256"] == _content_identity(model), "Inference mutated wrapper state")
    return {"reference": result_r, "spatial": result_s,
            "reference_diagnostics": result_rd, "spatial_diagnostics": result_sd,
            "global_reference_prediction": global_r, "global_spatial_prediction": global_s,
            "global_reference_diagnostics": global_rd, "global_spatial_diagnostics": global_sd,
            "affected_mask": affected, "fit_supported_mask": supported_mask,
            "report": {"family": FAMILY, "query_rows": len(X), "affected_rows": int(affected.sum()),
                       "fit_supported_query_rows": int(supported_mask.sum()),
                       "fallback_rows": int((~affected).sum()), "processed_groups": processed,
                       "global_reference_calls": 1, "global_spatial_calls": 1,
                       "group_reference_calls": len(processed), "group_spatial_calls": len(processed),
                       "fallback_vectors_and_diagnostics_bitwise": True, "query_labels_used": False,
                       "query_fit_performed": False, "input_frames_mutated": False,
                       "state_content_sha256": model["state_content_sha256"]}}


def self_check():
    """Small fabricated-reference checks; no course files or candidate MAE."""
    import contextlib
    import inspect
    import io
    import pickle
    from house_features import build_features
    from round6_geo import encode_geo_features
    checks = []

    def passed(name, condition):
        need(condition, "Synthetic check failed: " + name)
        checks.append(name)

    def frames():
        rows = []
        cases = [("positive", 40, 30, .08, 1100), ("negative", 40, 30, -.04, 1100),
                 ("neutral", 40, 30, 0., 1100), ("few", 20, 60, .08, 1100),
                 ("short", 40, 30, .08, 90), ("short_group", 40, 30, .08, 400),
                 ("boundary_project", 40, 30, .08, 180), ("boundary_group", 40, 30, .08, 730),
                 (reference_api.MISSING, 40, 30, .08, 1100)]
        for typ, count, n, delta, span in cases:
            for project in range(count):
                nrow = 3*n if typ == "positive" and project == 0 else n
                end = 9100 - 920*project/(count-1) if typ == "boundary_project" else 9100
                for day in np.linspace(end-span, end, nrow):
                    rows.append((typ, f"synthetic_{typ}_{project}", day, delta, project))
        spec = pd.DataFrame(rows, columns=["type", "project", "day", "delta", "project_number"])
        n = len(spec)
        raw = pd.DataFrame({"鄉鎮市區": ["大安區"]*n, "土地位置建物門牌": ["臺北市大安區合成路1號"]*n,
            "建案名稱": spec.project, "建物型態": ["住宅大樓"]*n, "主要用途": ["住家用"]*n,
            "建物移轉總面積平方公尺": [100.]*n, "土地移轉總面積平方公尺": [20.]*n,
            "車位移轉總面積平方公尺": [0.]*n, "交易筆棟數": ["土地1建物1車位0"]*n,
            "建物現況格局-房": [2]*n, "建物現況格局-廳": [1]*n, "建物現況格局-衛": [1]*n,
            "移轉層次": ["五層"]*n, "總樓層數": [10]*n, "交易年月日": ["1130101"]*n})
        X = build_features(raw)
        X["building_type"] = spec.type.to_numpy()
        X["trade_day_index"] = spec.day.to_numpy(dtype="float32")
        G = encode_geo_features(pd.DataFrame({"geo_latitude": [25.]*n, "geo_longitude": [121.]*n,
            "geo_road_length_m": [100.]*n, "geo_extent_diagonal_m": [0.]*n,
            "geo_nearby_fragment_grouping": [0]*n, "geo_match_status": ["road_level"]*n}))
        return X, G, spec

    def fabricated_reference(X, y, beta):
        # Use the legacy builder only to obtain legal neighbors/fill/filtering,
        # then explicitly install the known synthetic global coefficient.
        ref = reference_api.fit_final_reference(X, y)
        _, h, _ = reference_api.matrices(X, ref["fill_values"])
        valid = np.isfinite(ref["reference_residual"])
        ref["beta"] = beta.copy()
        ref["reference_residual"][valid] = (np.log(y / X.building_area.to_numpy(dtype="float64")) - h @ beta)[valid]
        ref["fit_report"]["hedonic"]["coefficients"] = dict(zip(reference_api.PARAMETERS["hedonic_features"], beta.tolist()))
        return ref

    X, G, spec = frames()
    beta = np.array([0., 0., 0., .06])
    _, H, _ = reference_api.matrices(X)
    y = X.building_area.to_numpy(dtype="float64") * np.exp(11. + .002*spec.project_number.to_numpy() +
                                                        H[:, 3]*(beta[3] + spec.delta.to_numpy()))
    ref = fabricated_reference(X, y, beta)
    old_s, _ = spatial_api.build(ref, X, G)
    before = (spatial_api.frame_sha(X), spatial_api.frame_sha(G), values_sha256(y), state_identity(ref), state_identity(old_s))
    model, meta = fit_model(X, y, G, ref, old_s)
    pos = model["group_trends"][("positive", "住家用")]
    neg = model["group_trends"][("negative", "住家用")]
    passed("known_positive_and_negative_time_deltas", pos["supported"] and neg["supported"] and
           abs(pos["fitted_delta"]-.08) < 1e-6 and abs(neg["fitted_delta"]+.04) < 1e-6)
    passed("fixed_shrinkage_and_coefficient_bounds", all(not r["supported"] or
           (r["shrinkage_weight"] == min(.5, r["independent_projects"]/(r["independent_projects"]+200.)) and
            -.08 <= r["clipped_gamma"] <= .20) for r in model["group_trends"].values()))
    passed("few_projects_short_project_short_group_and_missing_fallback", all(
           not model["group_trends"][(k, "住家用")]["supported"] for k in ("few", "short", "short_group", reference_api.MISSING)))
    passed("inclusive_180_day_project_and_730_day_group_boundaries", model["group_trends"][("boundary_project", "住家用")]["supported"] and
           model["group_trends"][("boundary_group", "住家用")]["supported"])
    passed("per_project_equal_initial_total_weight", pos["weight_report"]["initial_project_weight_min"] > 1-1e-12 and
           pos["weight_report"]["initial_project_weight_max"] < 1+1e-12)
    # Exercise clip on slopes beyond both boundaries without another real fit.
    dp, _ = _fit_delta(np.tile([-1., 0., 1.], 2), np.tile([-.5, 0., .5], 2), np.repeat(["a", "b"], 3))
    dm, _ = _fit_delta(np.tile([-1., 0., 1.], 2), np.tile([.5, 0., -.5], 2), np.repeat(["a", "b"], 3))
    passed("known_extreme_slopes_clip_to_both_bounds", float(np.clip(.06+dp, -.08, .20)) == .20 and
           float(np.clip(.06+dm, -.08, .20)) == -.08)
    equal_t = np.concatenate([[-1., 0., 1.], np.tile([-1., 0., 1.], 3)])
    equal_y = np.concatenate([.08*np.array([-1., 0., 1.]), -.08*np.tile([-1., 0., 1.], 3)])
    equal_project = np.array(["small"]*3 + ["large"]*9)
    equal_delta, _ = _fit_delta(equal_t, equal_y, equal_project)
    passed("repeated_large_project_does_not_receive_extra_initial_vote", abs(equal_delta) < 1e-14)
    cw = pd.DataFrame({"log_area": np.zeros(12), "parking": np.zeros(12), "floor": np.zeros(12),
                       "time": equal_t, "log_unit": equal_y, "project": equal_project})
    design_a, response_a = _centered_work(cw, np.zeros(4))
    cw.loc[cw.project.eq("large"), "log_unit"] += 2.
    design_b, response_b = _centered_work(cw, np.zeros(4))
    delta_a, _ = _fit_delta(design_a[:, 3], response_a, equal_project)
    delta_b, _ = _fit_delta(design_b[:, 3], response_b, equal_project)
    passed("project_level_shift_removed_before_time_fit", abs(delta_a-delta_b) < 1e-14)
    delta = np.asarray([model["group_trends"][k]["effective_delta"] for k in _style_keys(X)])
    finite = np.isfinite(ref["reference_residual"])
    expected = ref["reference_residual"].copy(); mask = finite & (delta != 0.)
    expected[mask] -= H[mask, 3] * delta[mask]
    passed("all_finite_reference_residuals_use_same_group_delta", np.array_equal(expected, model["reference_template"]["reference_residual"], equal_nan=True))
    passed("new_spatial_contains_new_reference_residual_identity", values_sha256(expected) == model["spatial_template"]["state_summary"]["existing_reference_residual_sha256"])
    passed("fit_inputs_not_mutated", before == (spatial_api.frame_sha(X), spatial_api.frame_sha(G), values_sha256(y), state_identity(ref), state_identity(old_s)))
    identity = model["state_content_sha256"]
    meta["driver_added_field"] = "detached"
    passed("returned_metadata_is_detached", "driver_added_field" not in model["metadata"] and model["state_content_sha256"] == identity)
    chosen = [int(np.flatnonzero(spec.type.eq(k))[0]) for k in ("positive", "negative", "neutral", "few", "short", reference_api.MISSING)]
    q = X.iloc[chosen + chosen[:2]].reset_index(drop=True).copy(); g = G.iloc[chosen + chosen[:2]].reset_index(drop=True).copy()
    q["project_name"] = ["query_new"]*len(q)
    for col in ("address_raw", "address_road", "address_locality"):
        q[col] = reference_api.MISSING
    q["trade_day_index"] = np.array([9280., 9400., 9280., 9500., np.nan, 9300., 9600., 10000.], dtype="float32")
    q.loc[2, "building_type"] = "unseen_type"
    # Pandas may infer StringDtype from synthetic scalar/list assignment;
    # synthetic query construction must retain the original feature schema.
    q = q.astype(X.dtypes.to_dict())
    q.index = g.index = pd.Index([10, 11, 12, 13, 14, 15, 16, 17])
    base = np.full(len(q), 1e7)
    blob = __import__("pickle").dumps(model, protocol=5)
    with contextlib.redirect_stdout(io.StringIO()):
        out = predict_model(model, q, g, base)
    passed("supported_mask_is_model_and_features_only", out["affected_mask"].tolist() == [True, True, False, False, False, False, True, True])
    sd = out["spatial_diagnostics"]
    passed("nonzero_group_delta_reaches_actual_spatial_applied_path", bool(sd.applied.iloc[0]) and bool(sd.applied.iloc[6]))
    qbeta = beta.copy(); qbeta[3] = pos["effective_gamma"]
    _, qh, _ = reference_api.matrices(q, ref["fill_values"])
    expected_future = base[6]*np.exp(qbeta[3]*366/365.25)
    log_ratio = np.log(float(q.building_area.iloc[6])) + sd.residual_center.iloc[6] + float(qh[6] @ qbeta) - np.log(expected_future)
    expected_g = expected_future*np.exp(np.clip(log_ratio, np.log(.65), np.log(1.5)))
    passed("spatial_G_and_future_base_share_gamma_and_only_base_caps_days", np.isclose(sd.base_future.iloc[6], expected_future, rtol=1e-14) and
           np.isclose(sd.geographic_price.iloc[6], expected_g, rtol=1e-14) and qh[6, 3] == (float(q.trade_day_index.iloc[6])-8000)/365.25)
    passed("fallback_and_missing_and_unseen_vectors_bitwise", np.array_equal(out["reference"][~out["affected_mask"]], out["global_reference_prediction"][~out["affected_mask"]]) and
           np.array_equal(out["spatial"][~out["affected_mask"]], out["global_spatial_prediction"][~out["affected_mask"]]))
    passed("query_state_not_mutated", pickle.dumps(model, protocol=5) == blob)
    order = np.array([6, 2, 5, 1, 7, 0, 4, 3]); inverse = np.argsort(order)
    with contextlib.redirect_stdout(io.StringIO()):
        perm = predict_model(model, q.iloc[order], g.iloc[order], base[order])
    passed("query_permutation_scattered_to_original_positions", np.array_equal(perm["reference"][inverse], out["reference"]) and np.array_equal(perm["spatial"][inverse], out["spatial"]))
    reloaded = pickle.loads(blob)
    passed("pickle_roundtrip_value_identity", state_identity(reloaded) == state_identity(model))
    with contextlib.redirect_stdout(io.StringIO()):
        replay = predict_model(reloaded, q, g, base)
    passed("pickle_reload_all_synthetic_predictions_and_diagnostics_exact", all(
           (out[k].equals(replay[k]) if isinstance(out[k], pd.DataFrame) else np.array_equal(out[k], replay[k]))
           for k in ("reference", "spatial", "reference_diagnostics", "spatial_diagnostics", "affected_mask")))
    passed("query_api_has_no_labels", list(inspect.signature(predict_model).parameters) == ["model", "X", "G", "base"])
    # A constant synthetic unit price with all-zero global beta identifies an
    # exactly neutral delta; this checks the supported-but-zero policy route.
    z_beta = np.zeros(4); z_y = X.building_area.to_numpy(dtype="float64").copy()
    z_ref = fabricated_reference(X, z_y, z_beta); z_sp, _ = spatial_api.build(z_ref, X, G)
    z_model, _ = fit_model(X, z_y, G, z_ref, z_sp)
    passed("exact_zero_delta_identified_from_neutral_synthetic_prices", all(r["effective_delta"] == 0. for r in z_model["group_trends"].values()))
    with contextlib.redirect_stdout(io.StringIO()):
        z_out = predict_model(z_model, q, g, base)
    passed("supported_zero_delta_preserves_full_global_vectors", not z_out["affected_mask"].any() and
           np.array_equal(z_out["reference"], z_out["global_reference_prediction"]) and np.array_equal(z_out["spatial"], z_out["global_spatial_prediction"]))
    return {"status": "passed", "checks": checks, "synthetic_fit_rows": len(X), "synthetic_query_rows": len(q),
            "course_data_read": False, "real_candidate_scores_computed": False, "new_tree_or_NN_fits": 0,
            "synthetic_wrapper_fits": 2, "parameters": parameters(), "source_sha256": source_hashes()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    receipt = self_check(); receipt["elapsed_seconds"] = time.monotonic()-started
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))
