"""R96 fixed reference-confidence contraction; no fitting or real-data loading.

The original R2/R8 sources remain unchanged. The caller must replay and verify
the original reference prediction/diagnostics before applying this policy, then
pass BOTH returned values to original R8 spatial.predict with its own raw base.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import round2_neighborhood as original

ROOT = Path(__file__).resolve().parents[1]
PLAN = "reports/round96_plan.md"
PLAN_SHA = "7178be401e3927f4ec26164159d2b515aef97e3a2f0158c3e19ee50d1c08a071"
DEPENDENCIES = {
    PLAN: PLAN_SHA,
    "src/round2_neighborhood.py": "cae244c40e6616d6c72f3aff7c3423a50a5bfbfbf35df33a7d044186eb491ae9",
    "src/round8_spatial_reference.py": "de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32",
    "src/house_features.py": "108e42d4429961a10a704f85bd66791599d904e6e96381c4c2f6dd4275316388",
    "src/round6_geo.py": "df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0",
}
PARAMETERS = {
    "format_version": 1,
    "candidate": "R96",
    "rho": 0.5,
    "reference_formula": "Bfuture + 0.5 * (original_reference - Bfuture), only where c > 0",
    "zero_confidence": "copy original reference exactly",
    "confidence_formula": "0.5 * original confidence",
    "future_base": "original raw branch base * exp(original beta[3] * original capped future_days / 365.25)",
    "spatial_contract": "original spatial.predict receives new reference and new confidence; source unchanged",
    "branches": ["nn", "log", "unit"],
    "final": "median(candidate nn spatial, candidate log spatial, candidate unit spatial)",
    "new_fits": 0,
    "query_labels_used": False,
    "new_reference_or_spatial_state": False,
}
_FIXED_PARAMETERS = json.dumps(PARAMETERS, sort_keys=True)
_FIXED_ORIGINAL_PARAMETERS = json.dumps(original.PARAMETERS, sort_keys=True)
DIAGNOSTIC_COLUMNS = ("source", "confidence", "nearest_distance", "reference_dispersion")
SOURCES = ("base_fallback", "project", "address", "street_section", "locality", "street", "district_style")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [*DEPENDENCIES, Path(__file__).resolve().relative_to(ROOT).as_posix()]
    return {name: _sha(ROOT / name) for name in paths}


def check_sources():
    values = source_hashes()
    for name, expected in DEPENDENCIES.items():
        need(values[name] == expected, "Source or plan changed: " + name)
    need(json.dumps(PARAMETERS, sort_keys=True) == _FIXED_PARAMETERS, "R96 parameters changed")
    need(json.dumps(original.PARAMETERS, sort_keys=True) == _FIXED_ORIGINAL_PARAMETERS,
         "Original reference parameters mutated")
    return values


def parameters():
    return copy.deepcopy(PARAMETERS)


def values_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _prices(value, n, label):
    v = np.asarray(value)
    need(v.dtype == np.dtype("float64") and v.shape == (n,), label + " dtype/shape differs")
    need(np.isfinite(v).all() and (v >= 0).all(), label + " must be finite and nonnegative")
    return v


def future_base(reference_model, Xquery, base_TWD):
    """Use R2's exact order; no max(B,1), imputation refit, or date eligibility gate."""
    need(isinstance(Xquery, pd.DataFrame) and "trade_day_index" in Xquery, "Missing query date frame")
    n = len(Xquery)
    baseline = _prices(base_TWD, n, "base_TWD")
    need(reference_model.get("format_version") == 1 and
         reference_model.get("parameters") == original.PARAMETERS, "Original reference contract differs")
    beta = np.asarray(reference_model["beta"], dtype="float64")
    anchor = float(reference_model["fit_report"]["hedonic"]["maximum_fit_day_index"])
    need(beta.shape == (4,) and np.isfinite(beta).all() and np.isfinite(anchor), "Invalid fit beta/anchor")
    future_days = np.maximum(np.nan_to_num(
        Xquery["trade_day_index"].to_numpy(dtype="float64") - anchor, nan=0.0), 0.0)
    future_days = np.minimum(future_days, original.PARAMETERS["trend_future_day_cap"])
    with np.errstate(over="ignore", invalid="ignore"):
        result = baseline * np.exp(beta[3] * future_days / 365.25)
    need(result.shape == (n,) and np.isfinite(result).all() and (result >= 0).all(), "Invalid future_base")
    return result


def _contract_values(Bfuture, original_P, confidence, rho):
    """Internal synthetic endpoint path; public inference fixes rho=0.5."""
    need(np.isfinite(rho) and 0 <= rho <= 1, "Invalid internal rho")
    output = original_P.copy()
    mask = confidence > 0
    if rho != 1.0:
        output[mask] = Bfuture[mask] + rho * (original_P[mask] - Bfuture[mask])
    new_confidence = rho * confidence
    need(np.isfinite(output).all() and (output >= 0).all(), "Invalid contracted prices")
    return output, new_confidence


def shrink_reference(reference_model, Xquery, base_TWD, original_reference_TWD,
                     original_reference_diagnostics):
    """Return (new_reference_float64, new_diagnostics, JSON_report); no query y."""
    sources_sha = check_sources()
    n = len(Xquery)
    base = _prices(base_TWD, n, "base_TWD")
    prediction = _prices(original_reference_TWD, n, "original_reference_TWD")
    diag = original_reference_diagnostics
    need(isinstance(diag, pd.DataFrame) and diag.shape == (n, 4), "Reference diagnostics shape differs")
    need(tuple(diag.columns) == DIAGNOSTIC_COLUMNS and diag.index.equals(Xquery.index),
         "Reference diagnostic columns/index differ")
    need(not diag["source"].isna().any() and
         diag["source"].isin(SOURCES).all(), "Invalid reference source")
    for name in DIAGNOSTIC_COLUMNS[1:]:
        need(diag[name].dtype == np.dtype("float64"), "Diagnostic dtype differs: " + name)
    confidence = diag["confidence"].to_numpy(copy=False)
    need(np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all(),
         "Invalid confidence")
    source = diag["source"].to_numpy(copy=False)
    need((confidence[source == "base_fallback"] == 0).all(), "base_fallback has nonzero confidence")
    future = future_base(reference_model, Xquery, base)
    zero = confidence == 0
    need(np.array_equal(prediction[zero], future[zero]), "Zero-confidence original P differs from future_base")
    result, new_c = _contract_values(future, prediction, confidence, PARAMETERS["rho"])
    new_diag = diag.copy(deep=True)
    new_diag["confidence"] = new_c
    need(result[zero].tobytes() == prediction[zero].tobytes(), "Zero-confidence values changed")
    report = {
        "format_version": 1, "candidate": "R96", "parameters": parameters(),
        "plan_sha256": PLAN_SHA, "source_hashes": sources_sha,
        "source_sha256": sources_sha[Path(__file__).resolve().relative_to(ROOT).as_posix()],
        "query_rows": n, "supported_reference_rows": int((confidence > 0).sum()),
        "zero_confidence_rows": int(zero.sum()), "base_fallback_rows": int((source == "base_fallback").sum()),
        "changed_reference_rows": int(np.count_nonzero(result != prediction)),
        "mean_original_confidence": float(confidence.mean()) if n else None,
        "mean_candidate_confidence": float(new_c.mean()) if n else None,
        "source_counts": {str(k): int(v) for k, v in diag["source"].value_counts().items()},
        "base_TWD_sha256": values_sha(base), "original_reference_sha256": values_sha(prediction),
        "future_base_sha256": values_sha(future), "candidate_reference_sha256": values_sha(result),
        "original_confidence_sha256": values_sha(confidence), "candidate_confidence_sha256": values_sha(new_c),
        "zero_confidence_reference_bitwise_preserved": True,
        "diagnostics_only_confidence_changed": True, "input_state_and_values_mutated": False,
        "query_labels_used": False, "new_fits": 0, "new_reference_or_spatial_state": False,
        "caller_must_pass_new_reference_and_confidence_to_original_spatial": True,
    }
    return result, new_diag, report


def self_check():
    """Small artificial states only: exercise real R2/R8 APIs without any fit."""
    import contextlib
    import io
    import inspect
    import round8_spatial_reference as spatial
    from house_features import build_features
    from round6_geo import encode_geo_features

    started = time.perf_counter()
    checks = []

    def passed(name, condition):
        need(condition, "Synthetic failed: " + name)
        checks.append(name)

    def rejected(name, call):
        try:
            call()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError("Synthetic invalid input accepted: " + name)

    ref = {"format_version": 1, "parameters": copy.deepcopy(original.PARAMETERS),
           "beta": np.array([0., 0., 0., .07]),
           "fit_report": {"hedonic": {"maximum_fit_day_index": 9000.}}}
    dates = np.array([8999., 9000., 9001., 9366., 9367., np.nan, np.inf, -np.inf])
    X = pd.DataFrame({"trade_day_index": dates}, index=pd.Index(range(10, 18)))
    b = np.array([0., .5, 1., 1e6, 2e6, 3e6, 4e6, 5e6])
    f = future_base(ref, X, b)
    expected = b * np.exp(.07 * np.array([0., 0., 1., 366., 366., 0., 366., 0.]) / 365.25)
    passed("future_base_R2_order_zero_subunit_nan_infinity_cap_exact", f.tobytes() == expected.tobytes())
    c = np.array([0., 1., .5, 0., .75, .12, .3, 0.])
    p = f.copy(); p[c > 0] += np.array([.25, .1, 1e5, 2e5, 3e5])
    d = pd.DataFrame({"source": np.array(["base_fallback", "project", "address", "street", "project", "district_style", "street", "base_fallback"], dtype=object),
                      "confidence": c, "nearest_distance": np.zeros(8), "reference_dispersion": np.zeros(8)}, index=X.index)
    before = pickle.dumps((ref, X, b, p, d), protocol=5)
    new_p, new_d, meta = shrink_reference(ref, X, b, p, d)
    expected = p.copy(); expected[c > 0] = f[c > 0] + .5 * (p[c > 0] - f[c > 0])
    passed("fixed_half_formula_exact", new_p.tobytes() == expected.tobytes())
    passed("zero_confidence_including_named_underflow_preserved", new_p[c == 0].tobytes() == p[c == 0].tobytes())
    passed("confidence_halved_exact", new_d.confidence.to_numpy().tobytes() == (.5*c).tobytes())
    pd.testing.assert_frame_equal(new_d.drop(columns="confidence"), d.drop(columns="confidence"), check_exact=True)
    passed("diagnostic_source_distance_dispersion_index_preserved", True)
    passed("all_input_bytes_unchanged", before == pickle.dumps((ref, X, b, p, d), protocol=5))
    passed("new_outputs_not_aliases", not np.shares_memory(new_p,p) and not np.shares_memory(new_d.confidence.to_numpy(),c))
    p1,c1 = _contract_values(f,p,c,1.0)
    passed("internal_rho_one_exact_control", p1.tobytes()==p.tobytes() and c1.tobytes()==c.tobytes())
    p0,_ = _contract_values(f,p,c,0.0)
    passed("internal_rho_zero_future_limit", p0.tobytes()==f.tobytes())
    order = np.array([7,4,2,0,6,1,3,5])
    np2,nd2,_ = shrink_reference(ref,X.iloc[order],b[order],p[order],d.iloc[order])
    passed("row_permutation_equivariance", np2.tobytes()==new_p[order].tobytes())
    pd.testing.assert_frame_equal(nd2,new_d.iloc[order],check_exact=True)
    passed("row_permutation_diagnostics_exact",True)
    loaded=pickle.loads(pickle.dumps((ref,X,b,p,d),protocol=5))
    rp,rd,rm=shrink_reference(*loaded)
    pd.testing.assert_frame_equal(rd,new_d,check_exact=True)
    passed("pickle_reload_all_prices_diagnostics_metadata_exact",rp.tobytes()==new_p.tobytes() and rm==meta)
    passed("public_API_no_query_label_or_rho_argument",list(inspect.signature(shrink_reference).parameters)==[
        "reference_model","Xquery","base_TWD","original_reference_TWD","original_reference_diagnostics"])
    rejected("reject_float32_base",lambda:shrink_reference(ref,X,b.astype('float32'),p,d))
    rejected("reject_shape_mismatch",lambda:shrink_reference(ref,X,b,p[:-1],d))
    bad=d.copy();bad.index=pd.RangeIndex(8)
    rejected("reject_index_mismatch",lambda:shrink_reference(ref,X,b,p,bad))
    for label,value in [("nan",np.nan),("negative",-.1),("above_one",1.1)]:
        bad=d.copy();bad.loc[11,"confidence"]=value
        rejected("reject_confidence_"+label,lambda bad=bad:shrink_reference(ref,X,b,p,bad))
    bad=d.copy();bad.loc[10,"confidence"]=.1
    rejected("reject_fallback_nonzero_confidence",lambda:shrink_reference(ref,X,b,p,bad))
    bad=p.copy();bad[0]=1.
    rejected("reject_inconsistent_original_fallback",lambda:shrink_reference(ref,X,b,bad,d))
    bad=b.copy();bad[0]=-1.
    rejected("reject_negative_base",lambda:shrink_reference(ref,X,bad,p,d))
    bad=p.copy();bad[2]=np.nan
    rejected("reject_nonfinite_reference",lambda:shrink_reference(ref,X,b,bad,d))
    bad=copy.deepcopy(ref);bad["format_version"]=2
    rejected("reject_changed_reference_contract",lambda:shrink_reference(bad,X,b,p,d))
    bad=copy.deepcopy(ref);bad["beta"][3]=np.nan
    rejected("reject_nonfinite_beta",lambda:shrink_reference(bad,X,b,p,d))

    def features(projects):
        n=len(projects)
        raw=pd.DataFrame({"鄉鎮市區":["大安區"]*n,"土地位置建物門牌":["臺北市大安區合成路1號"]*n,
            "建案名稱":projects,"建物型態":["住宅大樓"]*n,"主要用途":["住家用"]*n,
            "建物移轉總面積平方公尺":[100.]*n,"土地移轉總面積平方公尺":[20.]*n,
            "車位移轉總面積平方公尺":[0.]*n,"交易筆棟數":["土地1建物1車位0"]*n,
            "建物現況格局-房":[2]*n,"建物現況格局-廳":[1]*n,"建物現況格局-衛":[1]*n,
            "移轉層次":["五層"]*n,"總樓層數":[10]*n,"交易年月日":["1130101"]*n})
        out=build_features(raw);out["trade_day_index"]=np.full(n,9000.,dtype="float32")
        return out

    def geography(longitudes):
        n=len(longitudes)
        return encode_geo_features(pd.DataFrame({"geo_latitude":[25.]*n,"geo_longitude":longitudes,
            "geo_road_length_m":[100.]*n,"geo_extent_diagonal_m":[0.]*n,
            "geo_nearby_fragment_grouping":[0]*n,"geo_match_status":["road_level"]*n}))

    fit=features([name for name in "ABCDEF" for _ in range(8)])
    distance,hedonic,fill=original.matrices(fit)
    state=copy.deepcopy(ref);state["fit_report"]["fitting_rows"]=len(fit)
    state["required_feature_columns"]=list(fit.columns)
    state["fill_values"]=fill;state["reference_distance"]=distance
    state["reference_residual"]=np.log(10000.)+np.repeat([.1,.12,.15,.17,.19,.2],8)-hedonic@state["beta"]
    keys=original.group_keys(fit)
    state["groups"]={name:{str(k):np.flatnonzero(keys[name].to_numpy()==k).astype('int64')
        for k in keys[name].dropna().unique()} for name in ("project","district_style")}
    geo=geography([121.]*len(fit));sp,_=spatial.build(state,fit,geo)
    q=features(["Q","A","Q2","Q3","Q4","Q5","Q6"])
    q["trade_day_index"]=np.array([9100,9100,9100,9100,np.nan,9500,9200],dtype='float32')
    q.loc[2,"address_district"]="另一區";q.loc[3,"building_area"]=2000.
    qgeo=geography([121.,121.,121.,121.,121.,122.,121.])
    base=np.array([1e6,2e6,3e6,4e6,5e6,6e6,0.])
    frozen=pickle.dumps((state,sp,q,qgeo,base),protocol=5)
    with contextlib.redirect_stdout(io.StringIO()):
        old_p,old_d,_=original.predict_reference(state,q,base,variant="trend",return_diagnostics=True)
    old_u,old_sd,_=spatial.predict(sp,q,qgeo,base,old_p,old_d)
    new_p,new_d,_=shrink_reference(state,q,base,old_p,old_d)
    new_u,new_sd,_=spatial.predict(sp,q,qgeo,base,new_p,new_d)
    passed("real_R2_R8_synthetic_paths_present", old_d.source.tolist()==[
        "district_style","project","base_fallback","base_fallback","district_style","district_style","district_style"]
        and old_sd.applied.tolist()==[True,False,True,False,False,False,False])
    zero=old_d.confidence.to_numpy()==0
    passed("fallback_with_and_without_spatial_full_bitwise_unchanged",new_u[zero].tobytes()==old_u[zero].tobytes())
    for name in ("source","nearest_distance","reference_dispersion"):
        pd.testing.assert_series_equal(new_d[name],old_d[name],check_exact=True)
    for name in ("v2_source","eligible","applied","reason","geographic_price","base_future","spatial_weight"):
        pd.testing.assert_series_equal(new_sd[name],old_sd[name],check_exact=True)
    passed("spatial_source_eligibility_G_trend_alpha_unchanged",True)
    passed("spatial_received_halved_confidence_exact",new_sd.v2_confidence.to_numpy().tobytes()==(.5*old_d.confidence.to_numpy()).tobytes())
    expected=(1-new_d.confidence.to_numpy())*new_sd.spatial_weight.to_numpy()
    passed("spatial_effective_remaining_weight_uses_new_confidence",new_sd.effective_remaining_weight.to_numpy().tobytes()==expected.tobytes())
    active=new_sd.applied.to_numpy()
    expected=new_p[active]+new_sd.effective_remaining_weight.to_numpy()[active]*(new_sd.geographic_price.to_numpy()[active]-new_sd.base_future.to_numpy()[active])
    passed("spatial_price_formula_exact",new_u[active].tobytes()==expected.tobytes())
    passed("unsupported_spatial_retains_new_reference",new_u[~active].tobytes()==new_p[~active].tobytes())
    passed("invalid_date_keeps_R2_local_but_no_spatial",old_d.confidence.iloc[4]>0 and not new_sd.applied.iloc[4])
    passed("zero_base_not_replaced_with_one",future_base(state,q,base)[6]==0. and new_p[6]==.5*old_p[6])
    equal_p=old_p.copy();equal_p[0]=future_base(state,q,base)[0]
    equal_new,equal_d,_=shrink_reference(state,q,base,equal_p,old_d)
    equal_old_u,_,_=spatial.predict(sp,q,qgeo,base,equal_p,old_d)
    equal_new_u,_,_=spatial.predict(sp,q,qgeo,base,equal_new,equal_d)
    passed("positive_c_equal_future_reference_price_still_halves_confidence",
           equal_new[0]==equal_p[0] and equal_d.confidence.iloc[0]==.5*old_d.confidence.iloc[0])
    passed("equal_reference_price_does_not_suppress_changed_spatial_weight",
           equal_new_u[0]!=equal_old_u[0])
    endpoint,endpoint_c=_contract_values(future_base(state,q,base),old_p,old_d.confidence.to_numpy(),1.)
    endpoint_d=old_d.copy(deep=True);endpoint_d['confidence']=endpoint_c
    endpoint_u,endpoint_sd,_=spatial.predict(sp,q,qgeo,base,endpoint,endpoint_d)
    pd.testing.assert_frame_equal(endpoint_sd,old_sd,check_exact=True)
    passed("rho_one_full_R2_S_control_bitwise",endpoint_u.tobytes()==old_u.tobytes())
    passed("synthetic_reference_spatial_query_inputs_immutable",frozen==pickle.dumps((state,sp,q,qgeo,base),protocol=5))
    loaded=pickle.loads(pickle.dumps((state,sp,q,qgeo,base,old_p,old_d),protocol=5))
    lr,ls,lq,lg,lb,lp,ld=loaded
    rp,rd,_=shrink_reference(lr,lq,lb,lp,ld);ru,rsd,_=spatial.predict(ls,lq,lg,lb,rp,rd)
    pd.testing.assert_frame_equal(rsd,new_sd,check_exact=True)
    passed("full_synthetic_saved_reload_exact",ru.tobytes()==new_u.tobytes())
    return {"status":"passed","exit_code":0,"plan_sha256":PLAN_SHA,"parameters":parameters(),
        "source_sha256":_sha(__file__),"source_hashes":check_sources(),"checks":checks,"check_count":len(checks),
        "seconds":time.perf_counter()-started,"real_data_loaded":False,"query_labels_read":False,
        "new_real_native_fits":0,"synthetic_model_fits":0,"GPU_calls":0,
        "synthetic_reference_predict_calls":1,"synthetic_spatial_build_calls":1,"synthetic_spatial_predict_calls":6,
        "runtime":{"numpy":np.__version__,"pandas":pd.__version__}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-check",action="store_true",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    need(not args.out.exists(),"Refuse to overwrite a selfcheck receipt")
    try:
        report=self_check()
    except Exception:
        report={"status":"failed","exit_code":1,"plan_sha256":PLAN_SHA,"source_sha256":_sha(__file__),
            "source_hashes":source_hashes(),"real_data_loaded":False,"new_real_native_fits":0,
            "traceback":traceback.format_exc()}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        raise
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","check_count","seconds","source_sha256")},ensure_ascii=False))


if __name__=="__main__":
    main()
