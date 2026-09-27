"""One fixed round-five project-block identity mask; no training on import.

The feature state uses only fit X. Whole named projects share a stable SHA256
mask; new named query projects and missing project names are masked. The v2
reference must continue to receive the unchanged original feature frame.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MISSING = "__MISSING__"
MASK_TOKEN = "__R5_PROJECT_MASK__"
IDENTIFIER_COLUMNS = ("project_name", "address_raw", "building_and_unit", "building_block")
CONFIG_PATH = ROOT / "runs/v1/models/final_config.json"
MASK_RULES = {
    "format_version": 1,
    "feature_count": 87,
    "categorical_feature_count": 21,
    "mask_columns": list(IDENTIFIER_COLUMNS),
    "mask_token": MASK_TOKEN,
    "missing_token": MISSING,
    "project_key": "address_city|address_district|complete_project_name",
    "key_normalization": "fill missing; convert to string; strip surrounding whitespace; normalize missing sentinels only",
    "hash_prefix": "round5|42|",
    "hash_function": "SHA256, first 8 bytes as unsigned big-endian integer",
    "mask_if_integer_less_than": 2**63,
    "mask_probability_per_named_project": 0.5,
    "fit_named_project_mask": "same deterministic mask for every row of the geographic project",
    "query_seen_project_mask": "reuse fitted flag",
    "query_unseen_named_project_mask": True,
    "fit_and_query_missing_project_mask": True,
    "missing_name_is_not_a_physical_project": True,
    "fit_state_uses_y": False,
    "fit_state_uses_query": False,
    "other_feature_columns_unchanged": True,
    "reference_uses_original_features": True,
    "raw_target_divisor": 1000000.0,
    "sample_weight": "uniform",
    "iterations": 2999,
    "random_seed": 42,
    "threads": 4,
    "task_type": "GPU",
    "no_eval_set": True,
    "no_early_stopping": True,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",", ":"),allow_nan=False).encode("utf-8")).hexdigest()


def _normal(series):
    value=series.fillna(MISSING).astype(str).str.strip()
    return value.mask(value.isin(["", "nan", "None", "<NA>", "NaT"]),MISSING)


def _check_X(X):
    if not isinstance(X,pd.DataFrame) or len(X.columns)!=87 or not X.columns.is_unique:
        raise ValueError("Expected the unchanged original 87-column feature frame")
    if any(name in X for name in ("編號", "總價元")):
        raise ValueError("Identifiers and target prices cannot be model features")
    if not set((*IDENTIFIER_COLUMNS,"address_city","address_district")).issubset(X.columns):
        raise ValueError("Required project/location fields are missing")
    if X[list(IDENTIFIER_COLUMNS)].eq(MASK_TOKEN).any().any():
        raise ValueError("Pass original features; reserved mask token found in input")


def project_keys(X):
    """Return row-aligned geographic keys plus missing-name flags, without y."""
    _check_X(X)
    project=_normal(X.project_name)
    return _normal(X.address_city)+"|"+_normal(X.address_district)+"|"+project,project.eq(MISSING)


def _project_mask(key):
    digest=hashlib.sha256((MASK_RULES["hash_prefix"]+key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8],"big",signed=False)<MASK_RULES["mask_if_integer_less_than"]


def fit_feature_state(X):
    """Fit a JSON state from X only; do not create one group for missing names."""
    keys,missing=project_keys(X)
    flags={key:bool(_project_mask(key)) for key in sorted(set(keys.loc[~missing]))}
    mask=missing.to_numpy() | keys.map(flags).fillna(True).to_numpy(dtype=bool)
    state={"format_version":1,"mask_rules":json.loads(json.dumps(MASK_RULES)),
           "source_sha256":_sha(Path(__file__)),
           "feature_columns":list(X.columns),"feature_dtypes":[str(t) for t in X.dtypes],
           "fit_feature_values_sha256":hashlib.sha256(pd.util.hash_pandas_object(X,index=False).to_numpy(dtype="uint64").tobytes()).hexdigest(),
           "fit_rows":len(X),"project_mask_flags":flags,
           "fit_summary":{"named_projects":len(flags),"masked_named_projects":sum(flags.values()),
                          "missing_name_rows":int(missing.sum()),"masked_rows":int(mask.sum()),
                          "masked_row_fraction":float(mask.mean()) if len(mask) else None},
           "target_used":False,"query_rows_used":False}
    state["state_content_sha256"]=_content_sha(state)
    return state


def _check_state(state,X):
    _check_X(X)
    if state.get("mask_rules")!=MASK_RULES or state.get("source_sha256")!=_sha(Path(__file__)):
        raise ValueError("Frozen project-mask rules/source differ")
    if state.get("feature_columns")!=list(X.columns) or state.get("feature_dtypes")!=[str(t) for t in X.dtypes]:
        raise ValueError("Original feature order/dtypes differ from fitted state")
    payload={k:v for k,v in state.items() if k!="state_content_sha256"}
    if state.get("state_content_sha256")!=_content_sha(payload):
        raise ValueError("Project-mask state content changed")
    if any(type(flag) is not bool for flag in state["project_mask_flags"].values()):
        raise ValueError("Project-mask flags must be JSON booleans")


def query_mask(X,state):
    """Known projects reuse fit state; new or missing projects always mask."""
    _check_state(state,X)
    keys,missing=project_keys(X)
    flags=keys.map(state["project_mask_flags"])
    return missing.to_numpy() | flags.fillna(True).to_numpy(dtype=bool)


def transform_features(X,state):
    """Return a copy of X with exactly four string columns masked where required."""
    mask=query_mask(X,state)
    result=X.copy(deep=True)
    result.loc[mask,list(IDENTIFIER_COLUMNS)]=MASK_TOKEN
    other=[name for name in X if name not in IDENTIFIER_COLUMNS]
    if not result[other].equals(X[other]) or not result.dtypes.equals(X.dtypes):
        raise AssertionError("The mask changed non-identifier values or feature types")
    if not result.loc[mask,list(IDENTIFIER_COLUMNS)].eq(MASK_TOKEN).all().all():
        raise AssertionError("Incomplete identity block masking")
    return result


def parameters():
    """Reuse frozen v1 raw CatBoost settings with the prespecified fixed overrides."""
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    raw=[c for c in config["components"] if c["family"]=="catboost" and c["target_mode"]=="raw"]
    if len(raw)!=1:
        raise ValueError("Expected exactly one frozen v1 raw CatBoost component")
    params={**raw[0]["parameters"],"iterations":2999,"thread_count":4,"verbose":300}
    if params.get("random_seed")!=42 or params.get("task_type")!="GPU" or params.get("loss_function")!="MAE":
        raise ValueError("The predeclared seed, device, or objective changed")
    return params


def _categories(X,supplied):
    actual=[name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
    if list(supplied)!=actual or len(actual)!=21:
        raise ValueError("Expected the original 21 categorical columns in order")
    return actual


def fit_model(X,y,categorical_columns):
    """Return (CatBoost model, JSON feature state, metadata); no eval/query labels."""
    from catboost import CatBoostRegressor,Pool
    categories=_categories(X,categorical_columns)
    target=np.asarray(y,dtype="float64")
    if target.shape!=(len(X),) or not np.isfinite(target).all() or (target<0).any():
        raise ValueError("Fit targets must be finite, nonnegative, and row-aligned")
    began=time.perf_counter()
    state=fit_feature_state(X)
    masked=transform_features(X,state)
    config=parameters()
    model=CatBoostRegressor(**config)
    model.fit(Pool(masked,target/1e6,cat_features=categories,thread_count=4))
    if int(model.tree_count_)!=2999:
        raise ValueError("Model did not contain the fixed 2999 trees")
    if state["source_sha256"]!=_sha(Path(__file__)):
        raise RuntimeError("Project-mask source changed during training")
    metadata={"family":"catboost","component":"project_mask_raw","target_mode":"raw",
              "parameters":config,"iterations":2999,"training_rows":len(X),
              "feature_columns":list(X.columns),"categorical_columns":categories,
              "feature_count":87,"categorical_feature_count":21,
              "target_formula":"y/1e6","prediction_formula":"max(model_output*1e6,0)",
              "sample_weight":"uniform","mask_rules":json.loads(json.dumps(MASK_RULES)),
              "feature_state_content_sha256":state["state_content_sha256"],"mask_summary":state["fit_summary"],
              "script_sha256":_sha(Path(__file__)),"base_configuration_sha256":_sha(CONFIG_PATH),
              "fit_target_values_sha256":hashlib.sha256(target.tobytes()).hexdigest(),
              "fit_uses_eval_set":False,"early_stopping_used":False,"query_labels_used":False,
              "target_used_for_mask_state":False,"reference_features_changed":False,
              "elapsed_seconds":time.perf_counter()-began,"external_property_data_used":False,
              "limitations":["A geographic name key is not a guaranteed physical-project identifier.",
                             "Other structural/location features can still partially identify a project.",
                             "Fixed masking is not a proven improvement and does not itself extrapolate future price trends."]}
    return model,state,metadata


def predict_model(model,state,X,categorical_columns):
    """Return unrounded float64 TWD, using immutable fit-only identity decisions."""
    from catboost import Pool
    categories=_categories(X,categorical_columns)
    masked=transform_features(X,state)
    if list(model.feature_names_)!=list(masked.columns):
        raise ValueError("Model feature schema differs from mask state")
    result=np.maximum(np.asarray(model.predict(Pool(masked,cat_features=categories,thread_count=4),thread_count=4),dtype="float64")*1e6,0.0)
    if result.shape!=(len(X),) or not np.isfinite(result).all():
        raise ValueError("Invalid masked raw price predictions")
    return result


def self_check():
    """Synthetic, feature-only checks; never call fit_model or read real prices."""
    from house_features import build_features
    prefix="臺南市|新市|"  # Frozen legacy parser; this candidate does not correct original geography.
    names=[f"測試建案{i}" for i in range(100)]
    masked_name=next(name for name in names if _project_mask(prefix+name))
    visible_name=next(name for name in names if not _project_mask(prefix+name))
    raw=[]
    for name in (masked_name,visible_name):
        for floor in (3,7,12):
            raw.append({"鄉鎮市區":"新市區","土地位置建物門牌":"臺南市新市區中正路10號",
                        "建案名稱":name,"棟及號":f"A棟{floor}F","移轉層次":str(floor),"總樓層數":15,
                        "建物型態":"住宅大樓","主要用途":"住家用","建物移轉總面積平方公尺":100+floor,
                        "編號":floor,"總價元":1234567})
    for name in (None,""):
        raw.append({**raw[0],"建案名稱":name})
    frame=pd.DataFrame(raw)
    X=build_features(frame)
    X.index=pd.Index([10,30,50,70,90,110,130,150])
    original=X.copy(deep=True)
    state=fit_feature_state(X)
    frozen=json.dumps(state,ensure_ascii=False,sort_keys=True)
    reloaded=json.loads(json.dumps(state,ensure_ascii=False))
    actual=query_mask(X,reloaded)
    assert actual.tolist()==[True,True,True,False,False,False,True,True]
    assert len(state["project_mask_flags"])==2 and state["fit_summary"]["missing_name_rows"]==2
    assert not any(key.endswith("|"+MISSING) for key in state["project_mask_flags"])
    out=transform_features(X,reloaded)
    pd.testing.assert_frame_equal(X,original)
    pd.testing.assert_frame_equal(out.loc[~actual],X.loc[~actual])
    other=[name for name in X if name not in IDENTIFIER_COLUMNS]
    pd.testing.assert_frame_equal(out[other],X[other])
    assert out.loc[actual,list(IDENTIFIER_COLUMNS)].eq(MASK_TOKEN).all().all()
    query=X.iloc[[0,3,6]].copy()
    query.iloc[0,query.columns.get_loc("project_name")]="完全全新的案名"
    qout=transform_features(query,state)
    assert query_mask(query,state).tolist()==[True,False,True]
    assert qout.iloc[0][list(IDENTIFIER_COLUMNS)].eq(MASK_TOKEN).all()
    assert json.dumps(state,ensure_ascii=False,sort_keys=True)==frozen
    changed=frame.copy();changed["總價元"]=999999999;changed["編號"]=999
    independently=build_features(changed);independently.index=X.index
    assert fit_feature_state(independently)==state
    assert len(out.columns)==87 and out.dtypes.equals(X.dtypes)
    return {"status":"passed","synthetic_rows":len(X),"same_project_all_rows_consistent":True,
            "known_project_flag_reused":True,"unseen_named_project_masked":True,"missing_names_masked_separately":True,
            "all_four_identifiers_removed":True,"other_83_columns_unchanged":True,"input_unchanged":True,
            "json_roundtrip":True,"state_unchanged_by_query":True,"target_and_id_independent":True,
            "model_training_performed":False,"model_scoring_performed":False,"columns":87,"categorical_columns":21}


if __name__=="__main__":
    print(json.dumps(self_check(),ensure_ascii=True),flush=True)