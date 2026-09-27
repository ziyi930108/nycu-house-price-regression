"""Frozen pure-X location/name augmentation for one round-four raw CatBoost.

No training runs on import. Feature dictionaries use only the X passed to fit.
The original 87 features are retained exactly; v2 reference inputs are unchanged.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import numpy as np
import pandas as pd

MISSING = "__MISSING__"
ROOT = Path(__file__).resolve().parents[1]
FEATURE_RULES = {
    "format_version": 1,
    "original_feature_count": 87,
    "added_categorical_columns": ["r4_region", "r4_primary_road", "r4_secondary_road", "r4_road_pair", "r4_project_prefix2"],
    "added_numeric_columns": ["r4_multiple_roads", "r4_secondary_log_rows", "r4_secondary_log_projects", "r4_prefix_log_projects", "r4_repeated_city_prefix"],
    "road_minimum_rows": 20,
    "road_minimum_geo_projects": 3,
    "road_pair_minimum_rows": 20,
    "road_pair_minimum_geo_projects": 3,
    "maximum_roads_per_row": 2,
    "road_support_counts_first_and_second_positions_together": True,
    "each_fit_row_counted_once_per_unique_road": True,
    "road_pair_unordered": True,
    "road_key": "normalized_city|normalized_district|building_type|main_use|road",
    "project_key": "normalized_city|normalized_district|full_project_name",
    "prefix_characters": 2,
    "prefix_minimum_name_characters": 4,
    "prefix_minimum_geo_projects": 5,
    "prefix_minimum_regions": 2,
    "prefix_is_not_verified_developer_identity": True,
    "unknown_category": MISSING,
    "unknown_support": 0,
    "fit_only_vocabularies": True,
    "raw_target_divisor": 1000000.0,
    "iterations": 2999,
    "threads": 4,
    "no_eval_set": True,
    "no_early_stopping": True,
}
ADDED_CATEGORICAL = FEATURE_RULES["added_categorical_columns"]
ADDED_NUMERIC = FEATURE_RULES["added_numeric_columns"]
ROAD = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{1,22}?(?:大道|路|街)")
DIGIT = "零〇一二三四五六七八九十百千兩"


def _source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _normal(value):
    return unicodedata.normalize("NFKC", str(value)).replace("台", "臺").strip()


def _address_parts(address, supplied_district, city, old_district):
    text, c = _normal(address), _normal(city)
    supplied = _normal(supplied_district)
    body, repetitions = text, 0
    if c != MISSING:
        while body.startswith(c) and repetitions < 3:
            body = body[len(c):]
            repetitions += 1
    if supplied != MISSING and body.startswith(supplied):
        district = supplied
        body = body[len(supplied):]
    else:
        match = re.match(r"^(.{1,5}?[區鄉鎮市])", body)
        district = match[1] if match else _normal(old_district)
        if match:
            body = body[len(match[1]):]
    roads = []
    for token in ROAD.findall(body):
        token = re.split(r"與|及|交叉|交會|交界|位於|對面", token)[-1]
        token = re.sub(r"^(?:[0-9"+DIGIT+r"]+段)?(?:[0-9]+巷)?(?:[0-9]+弄)?", "", token)
        token = re.sub(r"^.{1,5}?[里村](?=.{2,}(?:路|街)$)", "", token)
        if 2 <= len(token) <= 22 and token not in roads:
            roads.append(token)
    if len(roads) > 3:
        roads = roads[:1]
    return c + "|" + district, tuple(roads[:2]), repetitions


def _check_X(X):
    if not isinstance(X, pd.DataFrame) or len(X.columns) != FEATURE_RULES["original_feature_count"]:
        raise ValueError("Expected the original 87-column feature DataFrame")
    if any(name in X for name in ("編號", "總價元")):
        raise ValueError("IDs and targets must not be passed as model features")
    needed = {"address_raw", "district", "address_city", "address_district", "building_type", "main_use", "project_name"}
    if not needed.issubset(X.columns) or not X.columns.is_unique:
        raise ValueError("Original feature schema is missing required columns")
    if any(name in X for name in ADDED_CATEGORICAL + ADDED_NUMERIC):
        raise ValueError("Pass original features; do not augment twice")


def _parse_rows(X):
    columns = ["address_raw", "district", "address_city", "address_district", "building_type", "main_use", "project_name"]
    address_cache = {}
    rows = []
    for address, supplied, city, old_district, building, use, project in X[columns].itertuples(index=False, name=None):
        key = (address, supplied, city, old_district)
        if key not in address_cache:
            address_cache[key] = _address_parts(*key)
        region, roads, repetitions = address_cache[key]
        style = str(building) + "|" + str(use)
        road_keys = tuple(region + "|" + style + "|" + road for road in roads)
        pair = region + "|" + style + "|" + "&".join(sorted(roads)) if len(roads) == 2 else MISSING
        missing_project = str(project) == MISSING
        geo_project = region + "|" + str(project) if not missing_project else None
        clean = re.sub(r"[\s\W_]+", "", _normal(project))
        prefix = clean[:2] if not missing_project and len(clean) >= 4 else MISSING
        rows.append((region, road_keys, pair, prefix, geo_project, repetitions))
    return rows


def fit_feature_state(X):
    """Build a JSON-serializable state using X only; no labels or globals are fitted."""
    _check_X(X)
    rows = _parse_rows(X)
    roads, pairs = Counter(), Counter()
    road_projects, pair_projects = defaultdict(set), defaultdict(set)
    prefix_projects, prefix_regions = defaultdict(set), defaultdict(set)
    regions = set()
    for region, road_keys, pair, prefix, project, repetitions in rows:
        regions.add(region)
        for key in set(road_keys):
            roads[key] += 1
            if project is not None:
                road_projects[key].add(project)
        if pair != MISSING:
            pairs[pair] += 1
            if project is not None:
                pair_projects[pair].add(project)
        if prefix != MISSING and project is not None:
            prefix_projects[prefix].add(project)
            prefix_regions[prefix].add(region)
    road_map = {key: {"rows": int(n), "projects": len(road_projects[key])} for key,n in sorted(roads.items())
                if n >= FEATURE_RULES["road_minimum_rows"] and len(road_projects[key]) >= FEATURE_RULES["road_minimum_geo_projects"]}
    pair_map = {key: {"rows": int(n), "projects": len(pair_projects[key])} for key,n in sorted(pairs.items())
                if n >= FEATURE_RULES["road_pair_minimum_rows"] and len(pair_projects[key]) >= FEATURE_RULES["road_pair_minimum_geo_projects"]}
    prefix_map = {key: {"projects":len(value), "regions":len(prefix_regions[key])} for key,value in sorted(prefix_projects.items())
                  if len(value) >= FEATURE_RULES["prefix_minimum_geo_projects"] and len(prefix_regions[key]) >= FEATURE_RULES["prefix_minimum_regions"]}
    state = {"format_version": 1, "feature_rules": json.loads(json.dumps(FEATURE_RULES)),
             "source_sha256": _source_sha(), "original_columns": list(X.columns),
             "original_dtypes": [str(t) for t in X.dtypes], "fit_rows": len(X),
             "regions": sorted(regions), "roads": road_map, "road_pairs": pair_map, "prefixes": prefix_map,
             "fit_feature_values_sha256": hashlib.sha256(pd.util.hash_pandas_object(X,index=False).to_numpy(dtype="uint64").tobytes()).hexdigest(),
             "support_summary": {"regions":len(regions), "supported_roads":len(road_map), "supported_road_pairs":len(pair_map), "supported_prefixes":len(prefix_map)},
             "uses_prices": False, "query_vocabulary_used": False}
    state["state_sha256"] = _canonical_hash(state)
    return state


def _check_state(state, X):
    _check_X(X)
    if state.get("feature_rules") != FEATURE_RULES or state.get("source_sha256") != _source_sha():
        raise ValueError("Frozen feature rules/source mismatch")
    if list(X.columns) != state["original_columns"] or [str(t) for t in X.dtypes] != state["original_dtypes"]:
        raise ValueError("Original feature schema or dtypes differ from fitted state")
    if state.get("state_sha256") != _canonical_hash({k:v for k,v in state.items() if k != "state_sha256"}):
        raise ValueError("Feature state content changed")


def transform_features(X, state):
    """Return an independent 97-column frame; never add query words to state."""
    _check_state(state, X)
    regions, road_map, pair_map, prefix_map = set(state["regions"]), state["roads"], state["road_pairs"], state["prefixes"]
    added = {name: [] for name in ADDED_CATEGORICAL + ADDED_NUMERIC}
    for region, roads, pair, prefix, project, repetitions in _parse_rows(X):
        first = roads[0] if roads else MISSING
        second = roads[1] if len(roads) >= 2 else MISSING
        secondary = road_map.get(second, {})
        prefix_stats = prefix_map.get(prefix, {})
        added["r4_region"].append(region if region in regions else MISSING)
        added["r4_primary_road"].append(first if first in road_map else MISSING)
        added["r4_secondary_road"].append(second if second in road_map else MISSING)
        added["r4_road_pair"].append(pair if pair in pair_map else MISSING)
        added["r4_project_prefix2"].append(prefix if prefix in prefix_map else MISSING)
        added["r4_multiple_roads"].append(float(len(roads) >= 2))
        added["r4_secondary_log_rows"].append(np.log1p(secondary.get("rows", 0)))
        added["r4_secondary_log_projects"].append(np.log1p(secondary.get("projects", 0)))
        added["r4_prefix_log_projects"].append(np.log1p(prefix_stats.get("projects", 0)))
        added["r4_repeated_city_prefix"].append(float(repetitions >= 2))
    result = X.copy(deep=True)
    for name in ADDED_CATEGORICAL:
        result[name] = pd.Series(added[name], index=X.index, dtype=object)
    for name in ADDED_NUMERIC:
        result[name] = np.asarray(added[name], dtype="float32")
    if not result[X.columns].equals(X) or len(result.columns) != 97:
        raise AssertionError("Original features changed")
    return result


def _categorical_columns(X, supplied):
    actual = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    if list(supplied) != actual:
        raise ValueError("Original categorical column order differs")
    return actual + ADDED_CATEGORICAL


def fit_augmented_raw(X, y, categorical_columns):
    """Fit the single predeclared raw-MAE model, with no query/evaluation input."""
    from catboost import CatBoostRegressor, Pool
    from round4_time_normalized import parameters
    categorical = _categorical_columns(X,categorical_columns)
    y = np.asarray(y, dtype="float64")
    if y.shape != (len(X),) or not np.isfinite(y).all() or (y<=0).any():
        raise ValueError("Expected aligned finite positive TWD training targets")
    started=time.perf_counter()
    state=fit_feature_state(X)
    augmented=transform_features(X,state)
    configuration = parameters()
    if configuration["iterations"] != 2999 or configuration["thread_count"] != 4 or configuration["verbose"] != 300:
        raise ValueError("The predeclared raw CatBoost configuration changed")
    model=CatBoostRegressor(**configuration)
    model.fit(Pool(augmented, y/1e6, cat_features=categorical,thread_count=4))
    metadata={"family":"catboost","target_mode":"raw","iterations":2999,"parameters":configuration,
              "fit_rows":len(X),"feature_columns":list(augmented),"categorical_columns":categorical,
              "feature_state_sha256":state["state_sha256"],"feature_rules":FEATURE_RULES,
              "script_sha256":_source_sha(),"fit_uses_eval_set":False,"early_stopping_used":False,
              "fit_target_values_sha256":hashlib.sha256(y.tobytes()).hexdigest(),
              "target_divisor":1e6,"support_summary":state["support_summary"],
              "elapsed_seconds":time.perf_counter()-started,"external_property_data":False,
              "reference_features_changed":False}
    return model,state,metadata


def predict_augmented_raw(model, feature_state, X, categorical_columns):
    """Return nonnegative, unrounded float64 TWD predictions from frozen state."""
    from catboost import Pool
    categorical=_categorical_columns(X,categorical_columns)
    augmented=transform_features(X,feature_state)
    if list(model.feature_names_) != list(augmented):
        raise ValueError("Model and augmented feature schema differ")
    prediction=np.asarray(model.predict(Pool(augmented,cat_features=categorical,thread_count=4),thread_count=4),dtype="float64")*1e6
    if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid augmented predictions")
    return np.maximum(prediction,0.0)


def self_check():
    """Small target-free assertions: support, reload, order and unknown vocabulary."""
    from house_features import build_features
    raw=[]
    for district in ("新市區","前鎮區"):
        city="臺南市" if district=="新市區" else "高雄市"
        for project in range(3):
            for row in range(8):
                raw.append({"鄉鎮市區":district,"土地位置建物門牌":city+district+"中正路與民生路交叉口",
                            "建案名稱":f"國際新案{district}{project}","建物型態":"住宅大樓","主要用途":"住家用"})
    X=build_features(pd.DataFrame(raw))
    original=X.copy(deep=True)
    state=fit_feature_state(X)
    state_before=json.dumps(state,ensure_ascii=False,sort_keys=True)
    restored=json.loads(json.dumps(state,ensure_ascii=False))
    Z=transform_features(X,restored)
    assert len(Z.columns)==97 and Z[ADDED_CATEGORICAL].dtypes.eq(object).all()
    assert Z[ADDED_NUMERIC].dtypes.eq(np.dtype("float32")).all()
    assert Z.r4_secondary_road.ne(MISSING).all() and Z.r4_road_pair.ne(MISSING).all()
    assert Z.r4_project_prefix2.eq("國際").all()
    pd.testing.assert_frame_equal(X,original)
    Q=X.iloc[[0]].copy()
    Q.loc[:,"address_raw"]="臺南市新市區民生路與中正路交叉口"
    reverse=transform_features(Q,state)
    assert reverse.r4_road_pair.iloc[0]==Z.r4_road_pair.iloc[0]
    assert reverse.r4_secondary_road.iloc[0]==Z.r4_primary_road.iloc[0]
    Q.loc[:,"address_raw"]="臺南市新市區未知路與陌生路交叉口"
    Q.loc[:,"project_name"]="從未見過新建案"
    unseen=transform_features(Q,state)
    assert unseen.r4_primary_road.iloc[0]==MISSING and unseen.r4_secondary_road.iloc[0]==MISSING
    assert unseen.r4_project_prefix2.iloc[0]==MISSING
    assert unseen.r4_secondary_log_rows.iloc[0]==0 and unseen.r4_prefix_log_projects.iloc[0]==0
    assert json.dumps(state,ensure_ascii=False,sort_keys=True)==state_before
    assert _address_parts("新竹市新竹市北區金雅路與金竹路", "新竹市", "新竹市", "新竹市")[0]=="新竹市|北區"
    assert _address_parts("桃園市平鎮區中豐路與民生路", "平鎮區", "桃園市", "平鎮")[0]=="桃園市|平鎮區"
    assert _address_parts("臺北市大安區和平東路二段", "大安區", "臺北市", "大安區")[1]==("和平東路",)
    return {"status":"passed","fit_rows":len(X),"columns":len(Z.columns),"added_categorical":5,"added_numeric":5,
            "unknown_support_zero":True,"original_X_unchanged":True,"state_roundtrip":True,
            "query_does_not_change_state":True,"road_positions_share_support":True,
            "road_pair_unordered":True,"new_model_training":False}


if __name__=="__main__":
    print(json.dumps(self_check(),ensure_ascii=True),flush=True)