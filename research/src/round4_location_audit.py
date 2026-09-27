"""Round-four location opportunity audit: feature-only; no fitting or new scoring.

Run: .venv/Scripts/python.exe src/round4_location_audit.py
Only feature caches, frozen source/configuration, and existing development metrics
are read. Raw price-bearing data, prediction arrays, confirmation results, and
external data are never opened.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round4/audit"
REPORT = ROOT / "reports/round4_location_opportunity.md"
MISSING = "__MISSING__"
READS = []


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    p = ROOT / path
    READS.append(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def normal(value):
    return unicodedata.normalize("NFKC", str(value)).replace("台", "臺").strip()


def qstats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {"rows": int(len(x)), "min": float(x.min()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "p90": float(np.quantile(x, .9)) if len(x) else None,
            "max": float(x.max()) if len(x) else None}


def rate_count(mask, group):
    n = int(np.sum(group))
    c = int(np.sum(np.asarray(mask, dtype=bool) & group))
    return {"rows": c, "denominator": n, "rate": c/n if n else None}


ROAD = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{1,22}?(?:大道|路|街)")
DIGIT = "零〇一二三四五六七八九十百千兩"


def address_parts(address, supplied_district, city, old_district):
    """Conservative descriptive parser; this is not a model feature replacement."""
    text = normal(address)
    c = normal(city)
    supplied = normal(supplied_district)
    body = text
    city_repetitions = 0
    if c != MISSING:
        while body.startswith(c) and city_repetitions < 3:
            body = body[len(c):]
            city_repetitions += 1
    if supplied != MISSING and body.startswith(supplied):
        district = supplied
        body = body[len(supplied):]
    else:
        m = re.match(r"^(.{1,5}?[區鄉鎮市])", body)
        district = m[1] if m else normal(old_district)
        if m:
            body = body[len(m[1]):]
    raw_tokens = ROAD.findall(body)
    roads = []
    for token in raw_tokens:
        token = re.split(r"與|及|交叉|交會|交界|位於|對面", token)[-1]
        token = re.sub(r"^(?:[0-9"+DIGIT+r"]+段)?(?:[0-9]+巷)?(?:[0-9]+弄)?", "", token)
        token = re.sub(r"^.{1,5}?[里村](?=.{2,}(?:路|街)$)", "", token)
        if 2 <= len(token) <= 22 and token not in roads:
            roads.append(token)
    # More than three road-like pieces is an ambiguous description, not a road list.
    if len(roads) > 3:
        roads = roads[:1]
    return c, district, tuple(roads[:2]), city_repetitions, body


def enrich(frame):
    cache = {}
    rows = []
    for values in frame[["address_raw", "district", "address_city", "address_district"]].itertuples(index=False, name=None):
        if values not in cache:
            cache[values] = address_parts(*values)
        rows.append(cache[values])
    derived = pd.DataFrame(rows, columns=["city", "district", "roads", "city_repetitions", "body"])
    derived["region"] = derived.city + "|" + derived.district
    derived["project"] = derived.region + "|" + frame.project_name.astype(str).to_numpy()
    derived["legacy_region"] = frame.address_city.astype(str) + "|" + frame.address_district.astype(str)
    derived["legacy_project"] = derived.legacy_region + "|" + frame.project_name.astype(str)
    derived["style"] = frame.building_type.astype(str) + "|" + frame.main_use.astype(str)
    derived["project_missing"] = frame.project_name.eq(MISSING).to_numpy()
    derived["first_road"] = derived.roads.map(lambda x: x[0] if x else MISSING)
    return derived


def keys(frame, derived):
    prefix = derived.legacy_region + "|" + derived["style"] + "|"
    ret = {"city_district_style": prefix,
           "project": (prefix + frame.project_name).where(frame.project_name.ne(MISSING)),
           "address": (prefix + frame.address_raw).where(frame.address_raw.ne(MISSING)),
           "road": (prefix + frame.address_road).where(frame.address_road.ne(MISSING)),
           "road_section": (prefix + frame.address_road + "|" + frame.address_section).where(frame.address_road.ne(MISSING)),
           "locality": (prefix + frame.address_locality).where(frame.address_locality.ne(MISSING))}
    return ret


def coverage(fit, fd, query, qd, scope):
    known = set(fd.loc[~fd.project_missing, "legacy_project"])
    group = np.where(qd.project_missing, "missing", np.where(qd.legacy_project.isin(known), "seen", "new_nonmissing"))
    fitkeys, querykeys = keys(fit, fd), keys(query, qd)
    results = []
    for level in fitkeys:
        fk, qk = fitkeys[level], querykeys[level]
        support = fk.value_counts()
        work = pd.DataFrame({"key": fk, "project": fd.legacy_project.where(~fd.project_missing)})
        project_support = work.dropna().drop_duplicates().groupby("key").size()
        nr = qk.map(support).fillna(0).to_numpy()
        npj = qk.map(project_support).fillna(0).to_numpy()
        for g in ("all", "seen", "new_nonmissing", "missing"):
            mask = np.ones(len(query), dtype=bool) if g == "all" else group == g
            results.append({"scope": scope, "level": level, "group": g,
                            "rows": int(mask.sum()), "nonmissing_key": rate_count(qk.notna(), mask),
                            "seen_key": rate_count(nr > 0, mask),
                            "at_least_20_rows": rate_count(nr >= 20, mask),
                            "at_least_20_rows_3_projects": rate_count((nr >= 20) & (npj >= 3), mask),
                            "reference_rows": qstats(nr[mask]), "distinct_reference_projects": qstats(npj[mask])})
    return group, results


def road_opportunity(fit, fd, query, qd, labels, scope):
    row_count, projects, old_projects = Counter(), defaultdict(set), defaultdict(set)
    fit_roads, query_roads = [], []
    # Count each row once per road; no price filter or labels are used.
    for roads,region,style,missing,project,legacy_region,legacy_project,old in zip(fd.roads,fd.region,fd["style"],fd.project_missing,fd.project,fd.legacy_region,fd.legacy_project,fit.address_road):
        prefix = region + "|" + style + "|"
        for road in roads:
            key = prefix + road
            row_count[key] += 1
            if not missing:
                projects[key].add(project)
        if old != MISSING and not missing:
            old_key = legacy_region + "|" + style + "|" + old
            old_projects[old_key].add(legacy_project)
    multi, added, alt_support, first_support, both_support = [], [], [], [], []
    legacy_support = []
    for roads,region,style,legacy_region,old in zip(qd.roads,qd.region,qd["style"],qd.legacy_region,query.address_road):
        prefix = region + "|" + style + "|"
        supported = [row_count[prefix+r] >= 20 and len(projects[prefix+r]) >= 3 for r in roads]
        old_key = legacy_region + "|" + style + "|" + old
        legacy_support.append(len(old_projects[old_key]) >= 3)
        multi.append(len(roads) >= 2)
        added.append(any(r != old for r in roads))
        first_support.append(bool(supported and supported[0]))
        alt_support.append(any(supported[1:]))
        both_support.append(sum(supported) >= 2)
    result = []
    for g in ("all", "seen", "new_nonmissing", "missing"):
        mask = np.ones(len(query), dtype=bool) if g == "all" else labels == g
        result.append({"scope": scope, "group": g, "rows": int(mask.sum()),
            "multiple_road_tokens": rate_count(multi, mask),
            "road_token_beyond_legacy_first": rate_count(added, mask),
            "second_road_supported_by_20_rows_3_projects": rate_count(alt_support, mask),
            "both_roads_supported_by_20_rows_3_projects": rate_count(both_support, mask),
            "second_road_rescues_unsupported_first": rate_count(np.asarray(alt_support) & ~np.asarray(first_support), mask),
            "multiple_roads_without_legacy_3_project_support": rate_count(np.asarray(multi) & ~np.asarray(legacy_support), mask)})
    return result


def project_prefixes(fit, fd, query, qd, labels, scope):
    # Prefix vocabulary support is computed from fit only, across distinct geo-projects.
    cleaned = fit.project_name.map(lambda s: re.sub(r"[\s\W_]+", "", normal(s)))
    query_cleaned = query.project_name.map(lambda s: re.sub(r"[\s\W_]+", "", normal(s)))
    results = []
    for width in (2, 3, 4):
        eligible = fit.project_name.ne(MISSING) & cleaned.str.len().ge(width+2)
        vocab = pd.DataFrame({"prefix": cleaned.str[:width], "project": fd.legacy_project,
                              "region": fd.legacy_region})[eligible].drop_duplicates()
        counts = vocab.groupby("prefix").agg(projects=("project", "nunique"), regions=("region", "nunique"))
        qprefix = query_cleaned.str[:width]
        p = qprefix.map(counts.projects).fillna(0).to_numpy()
        r = qprefix.map(counts.regions).fillna(0).to_numpy()
        valid = query.project_name.ne(MISSING).to_numpy() & query_cleaned.str.len().ge(width+2).to_numpy()
        for g in ("all", "seen", "new_nonmissing", "missing"):
            mask = np.ones(len(query), dtype=bool) if g == "all" else labels == g
            results.append({"scope": scope, "prefix_width": width, "group": g,
                "supported_by_5_projects_2_regions": rate_count(valid & (p>=5) & (r>=2), mask),
                "supported_by_10_projects_3_regions": rate_count(valid & (p>=10) & (r>=3), mask),
                "distinct_fit_prefixes_with_5_projects_2_regions": int(((counts.projects>=5)&(counts.regions>=2)).sum())})
    return results


def diagnostics(frame, derived, split):
    features = ["address_city", "address_district", "address_road", "address_section", "address_locality",
                "building_block", "unit_floor", "floor_mean", "total_floors", "relative_floor", "project_name"]
    missing = {}
    for column in features:
        values = frame[column]
        mask = values.isna() if pd.api.types.is_numeric_dtype(values) else values.eq(MISSING)
        missing[column] = {"rows": int(mask.sum()), "rate": float(mask.mean())}
    floors = frame.floor_mean.to_numpy()
    total = frame.total_floors.to_numpy()
    unit = frame.unit_floor.to_numpy()
    return {"split": split, "rows": len(frame), "missing": missing,
        "repeated_city_prefix": int((derived.city_repetitions >= 2).sum()),
        "normalization_changes_district": int(derived.district.ne(frame.address_district).sum()),
        "multiple_road_tokens": int(derived.roads.map(len).ge(2).sum()),
        "floor_mean_missing_but_unit_floor_available": int((~np.isfinite(floors)&np.isfinite(unit)).sum()),
        "parsed_floor_above_total": int((np.isfinite(floors)&np.isfinite(total)&(floors>total)&(total>0)).sum()),
        "raw_numeric_floor_available": int(np.isfinite(floors).sum()),
        "total_floors": qstats(total)}


def main():
    source_paths = ["src/house_features.py", "src/round2_neighborhood.py", "src/round2_lightgbm.py",
                    "src/round2_catboost.py", "src/round3_features.py", "src/round3_reference.py",
                    "src/round4_augmented.py", "src/round4_location_audit.py"]
    for p in source_paths[:-1]:
        (ROOT/p).read_text(encoding="utf-8-sig")
        READS.append(p)
    schema = read_json("reports/feature_schema.json")
    if schema["feature_source_sha256"] != digest(ROOT/"src/house_features.py"):
        raise ValueError("Frozen feature source hash differs from cache schema")
    frames, derived = {}, {}
    cache_paths = []
    for split in ("train", "valid", "test"):
        path = f"data_cache/{split}_features.pkl"
        READS.append(path)
        cache_paths.append(path)
        frames[split] = pd.read_pickle(ROOT/path).reset_index(drop=True)
        if any(c in frames[split] for c in ("總價元", "編號")):
            raise ValueError("Price/ID column found in feature-only input")
        if list(frames[split]) != schema["columns"]:
            raise ValueError("Feature schema mismatch")
        derived[split] = enrich(frames[split])
    histories = {}
    for family in ("catboost", "lightgbm", "neighborhood"):
        histories[family] = read_json(f"outputs/round2/{family}/candidates.json")
    # Read development only; never open confirmation or its derived summary report.
    dev = read_json("outputs/round3/evaluation/development/results.json")
    history_dev = {name: {"overall_mae": m["overall_mae"], "weighted_mae": m["weighted_mae"],
                         "eligible": None if name == "v2_control" else dev["selection"]["candidate_checks"][name]["eligible"]}
                   for name,m in dev["metrics"].items()}
    full = pd.concat([frames["train"],frames["valid"]],ignore_index=True)
    full_d = pd.concat([derived["train"],derived["valid"]],ignore_index=True)
    scopes = [(frames["train"], derived["train"], frames["valid"], derived["valid"], "train_to_original_valid"),
              (frames["train"], derived["train"], frames["test"], derived["test"], "train_to_test"),
              (full, full_d, frames["test"], derived["test"], "train_valid_features_to_test")]
    support, roads, prefixes, groups = [], [], [], []
    for fit,fd,query,qd,name in scopes:
        label, cov = coverage(fit,fd,query,qd,name)
        support.extend(cov)
        roads.extend(road_opportunity(fit,fd,query,qd,label,name))
        prefixes.extend(project_prefixes(fit,fd,query,qd,label,name))
        groups.append({"scope":name,"groups":{g:int((label==g).sum()) for g in ("seen","new_nonmissing","missing")}})
    diagnostics_rows = [diagnostics(frames[s],derived[s],s) for s in frames]
    # A feature-only historical origin mirrors the development dates; cache duplicate
    # rows remain because reconstructing price-aware dedup would require raw labels.
    day = frames["train"].trade_day_index.to_numpy()
    start = (pd.Timestamp("2022-08-11")-pd.Timestamp("2000-01-01")).days
    end = (pd.Timestamp("2023-08-10")-pd.Timestamp("2000-01-01")).days
    fm, qm = np.isfinite(day)&(day<start), np.isfinite(day)&(day>=start)&(day<=end)
    fit,fd = frames["train"].loc[fm].reset_index(drop=True),derived["train"].loc[fm].reset_index(drop=True)
    query,qd = frames["train"].loc[qm].reset_index(drop=True),derived["train"].loc[qm].reset_index(drop=True)
    label,cov=coverage(fit,fd,query,qd,"development_feature_only_origin")
    support.extend(cov)
    roads.extend(road_opportunity(fit,fd,query,qd,label,"development_feature_only_origin"))
    prefixes.extend(project_prefixes(fit,fd,query,qd,label,"development_feature_only_origin"))
    groups.append({"scope":"development_feature_only_origin","fit_rows":len(fit),"query_rows":len(query),
                   "groups":{g:int((label==g).sum()) for g in ("seen","new_nonmissing","missing")}})
    from round4_augmented import FEATURE_RULES,fit_feature_state,transform_features
    bundle_state=fit_feature_state(full)
    augmented=transform_features(frames["test"],bundle_state)
    known=set(full_d.loc[~full_d.project_missing,"legacy_project"])
    new_mask=(~derived["test"].project_missing & ~derived["test"].legacy_project.isin(known)).to_numpy()
    bundle_coverage={name:rate_count(augmented[name].ne(MISSING),new_mask)
                     for name in FEATURE_RULES["added_categorical_columns"]}
    bundle={"feature_rules":FEATURE_RULES,"fit_rows":len(full),"fit_state_saved":False,
            "support_summary":bundle_state["support_summary"],
            "new_test_project_category_coverage":bundle_coverage,
            "original_columns_unchanged":augmented[frames["test"].columns].equals(frames["test"]),
            "total_columns":len(augmented.columns),"categorical_columns":26,
            "no_price_model_fit":True,
            "prefix_count_note":"Actual bundle uses normalized geographic projects and regions; descriptive prefix table uses legacy geography for continuity."}
    report={"audit_type":"feature_only_distribution_and_existing_development_history",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":{p:digest(ROOT/p) for p in source_paths},
        "feature_cache_sha256":{p:digest(ROOT/p) for p in cache_paths},
        "constraints":{"no_external_property_data":True,"no_government_answer_lookup":True,
                       "no_price_bearing_raw_files_read":True,"no_test_prices":True,
                       "no_confirmation_errors_read":True,"no_new_model_training":True,"no_new_candidate_scoring":True,
                       "old_artifacts_unchanged":True},
        "data_policy":{"feature_rows_include_529_price_aware_train_duplicates":True,
                       "train_rows":len(frames["train"]),"valid_rows":len(frames["valid"]),"test_rows":len(frames["test"]),
                       "reason":"Only prebuilt feature caches are opened. Price-aware original row deduplication is deliberately not reconstructed; supports are descriptive, not model fit counts.",
                       "test_features_used_for_descriptive_coverage_only":True,
                       "support_and_prefix_vocabulary_learned_from_fit_side_only":True,
                       "support_does_not_apply_price_quantile_filter":"Actual eligible reference support may be lower."},
        "group_definition":"legacy address_city|address_district|project_name; missing names kept separate",
        "groups":groups,"feature_diagnostics":diagnostics_rows,"location_support":support,
        "intersection_road_opportunity":roads,"project_prefix_opportunity":prefixes,
        "existing_round2_candidates":histories,"existing_round3_development":history_dev,
        "development_frozen_choice":dev["selection"]["selected_candidate"],
        "fixed_augmented_bundle":bundle,
        "read_artifacts":READS,
        "recommendations":[
            {"name":"intersection_aware_shared_road_features", "status":"untrained proposal",
             "mechanism":"Use supplied address text to retain up to two explicit road names; normalize repeated city prefixes conservatively and generate city/district/style/road and road-pair keys. All maps and supports are fit-only. No external coordinates or fabricated distances.",
             "complements_existing":"Legacy base features and v2 reference retain the first road only. A second road or road-pair can identify locality for a previously unseen project.",
             "bounded_design":"One combined augmented raw CatBoost candidate with the project-prefix component below: fixed original 2999 iterations, 87 unchanged inputs plus 10 X-only columns. Road and unordered road-pair support minimum 20 fit rows from 3 geo-projects. The v2 reference remains unchanged.",
             "validation":"Unit-check explicit intersection examples; group by geo-project and use the newly precommitted annual development split. Confirm once only if preset overall, new-project, horizon-quarter and large-city gates pass. Do not choose parser/support thresholds from test outcomes."},
            {"name":"fit_only_project_subword_transfer", "status":"untrained proposal",
             "mechanism":"Add a fixed 2-character project-name prefix category plus train-only support counts; keep prefix only with at least 5 distinct geo-projects across 2 regions, otherwise missing. Preserve the full existing feature set; do not interpret a prefix as a verified developer brand.",
             "complements_existing":"Trees currently treat whole project names as unrelated categories; unseen whole names may share a learned prefix. No earlier candidate has added project subwords.",
             "bounded_design":"Same single combined augmented raw CatBoost candidate as the road component; fixed depth/round count and loss, no prefix-length or blend-weight search. Any target encoding must be chronological out-of-fold within fit only, or rely on CatBoost ordered categorical handling.",
             "validation":"Predeclare only this 2-character representation before scores. Check new-nonmissing projects and time quarters separately; require improvement on the same strict planned split. Audit generic-word collisions and fall back if support is weak."}
        ],
        "not_new_opportunities":["Total floors, floor mean, rooms, land/building ratio, parking and date already enter v2 nearest-neighbor distance.",
            "Project/address/unit removal and stronger categorical smoothing already tried in round2 spatial63 and smooth31.",
            "Address prefix correction already exists; it was not used in v3 pricing and its selected test neighbor membership did not change in the prior audit.",
            "Regional future trend and recency blends already failed the predeclared round3 development gate; they are not fresh feature ideas."]}
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"location_opportunity.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_markdown(report)
    print(json.dumps({"json":str(path.relative_to(ROOT)),"report":str(REPORT.relative_to(ROOT)),
                      "rows":report["data_policy"],"test_groups":groups[2],"bundle":bundle,
                      "road_test_new":[r for r in roads if r["scope"]=="train_valid_features_to_test" and r["group"]=="new_nonmissing"],
                      "prefix_test_new":[r for r in prefixes if r["scope"]=="train_valid_features_to_test" and r["group"]=="new_nonmissing"]},ensure_ascii=True),flush=True)


def write_markdown(r):
    scope="train_valid_features_to_test"
    take=lambda items,**filters:next(x for x in items if all(x[k]==v for k,v in filters.items()))
    g=take(r["groups"],scope=scope)["groups"]
    road=take(r["intersection_road_opportunity"],scope=scope,group="new_nonmissing")
    prefix=take(r["project_prefix_opportunity"],scope=scope,group="new_nonmissing",prefix_width=2)
    fmt=lambda v:f"{v['rows']:,}（{100*v['rate']:.2f}%）" if v['rate'] is not None else "無樣本"
    lines=["# 第四輪地點特徵機會稽核","",
      "本文件只檢查特徵分布與已存在的開發期結果；沒有訓練或評分新候選，也沒有讀取確認期誤差、原始價格欄或外部成交資料。以下覆蓋率不代表 MAE 改善。","",
      "## 資料口徑","",
      f"直接讀取凍結的 87 欄特徵快取：train {r['data_policy']['train_rows']:,}、valid 11,994、test 47,073 列。為避免讀入價格，未重建含價格的去重規則，因此 train 計數仍含既有 529 筆完全重複列；支持數是描述性統計，不是模型的實際訓練列數。",
      "支持字典及建案子詞詞彙只由每次對照的 fit 側建立；test 僅提供未標記的查詢特徵。另列原 train→valid、train→test 及 2022-08-11 起一年的 feature-only 歷史切分。歷史切分也保留上述重複列，不能冒充模型回測樣本數。","",
      f"依舊版城市＋行政區＋建案識別，test 已見 {g['seen']:,}、真正新建案 {g['new_nonmissing']:,}、建案缺失 {g['missing']:,} 列。下列覆蓋率以 train＋valid 的無價格特徵為參考，分母為新建案 {g['new_nonmissing']:,} 列。","",
      "## 已有能力與沒有漏掉的項目","",
      "87 欄已包含城市、行政區、道路、段、里／村／地段、完整地址、建案與棟號，以及樓層、總樓高、相對樓層、距頂樓、面積及格局。v2 局部參考的近鄰距離也已使用總樓高、樓層、房數、車位數與土地／建物比，不能把『新增樓高配對』當成尚未試過的補強。","",
      "| 地點粒度（含建物型態／用途） | 新建案列中有至少 20 筆參考 | 同時來自至少 3 建案 |","|---|---:|---:|"]
    for key,title in (("road_section","路段"),("road","道路"),("locality","里／村／地段"),("city_district_style","行政區")):
        c=take(r["location_support"],scope=scope,group="new_nonmissing",level=key)
        lines.append(f"| {title} | {fmt(c['at_least_20_rows'])} | {fmt(c['at_least_20_rows_3_projects'])} |")
    lines += ["","這些數量沒有套用價格分位數篩選；真實 reference 的可用樣本可能更少。既有行政區層級也已是備援，單純新增同名行政區欄位不會提供新資訊。","",
      "## 最多兩個具體建議","",
      "### 1. 保留路口的第二條道路，建立可跨新建案共用的地點特徵","",
      f"新建案 test 中可從地址拆出多條道路者 {fmt(road['multiple_road_tokens'])}；第二條路在 fit 側有至少 20 筆、3 個建案支持者 {fmt(road['second_road_supported_by_20_rows_3_projects'])}；第一條路不夠支持、第二條能補上的列為 {fmt(road['second_road_rescues_unsupported_first'])}。",
      "現有特徵及 reference 主要只保留第一條路，完整地址雖保留為類別，仍無法讓不同地址共享第二條路。建議只增加一個固定版本：同城市、行政區及建物型態／用途下，拆出明確的兩路交會；新增道路對或第二道路支持。先以人工可判讀案例檢查解析，不把字面路名當座標或實際距離。已收斂為一個固定 raw CatBoost 增強套件：與下列前綴特徵合併，原 87 欄完全保留，再加 5 類別＋5 數值欄（總計 97 欄、26 類別）。不修改 v2 局部參考。訓練固定 2,999 輪及原參數，無 eval_set；尚未訓練或評分。","",
      "### 2. 以 fit-only 支持門檻保留建案名稱前綴","",
      f"新建案 test 名稱雖未完整出現，其前 2 字能在 fit 中找到至少 5 建案、2 行政區者 {fmt(prefix['supported_by_5_projects_2_regions'])}；較嚴格的 10 建案、3 行政區支持為 {fmt(prefix['supported_by_10_projects_3_regions'])}。",
      "既有樹模型把完整案名當類別，尚未試過案名子詞；可固定增加前 2 字類別與無價格支持數，支持不足填缺失，其餘參數與舊特徵固定。前綴可能是泛用詞，不能直接宣稱為建商品牌。若使用價格編碼，必須僅在 fit 內採時間 OOF；較簡單的版本可沿用 CatBoost 的 ordered 類別處理。本稽核也列 3／4 字覆蓋作描述，不依 MAE 選長度；第一候選只預先指定 2 字。","",
      "## 已經試過，避免重複包裝","",
      "| 類似方向 | 既有動作 | 本輪判斷 |","|---|---|---|",
      "| 減少建案記憶 | 第一輪 CatBoost structural；第二輪 LightGBM spatial63 移除建案、完整地址、棟號與棟別 | 不是新增地理資訊 |",
      "| 強化正則化 | LightGBM smooth31／spatial63 的類別平滑、較大葉節點樣本；CatBoost depth9 | 已有同類候選 |",
      "| 局部交易參考 | v2 anchor／trend：建案、地址、路段、里／村、道路、行政區階層 | 本提案只補第二道路或可共用文字片段 |",
      "| 地址截斷修正 | round3_features 已實作；先前稽核顯示實際選中的 test 參考成員未變 | 已有資料品質修復，不當成保證改善 |",
      "| 近期與地域趨勢 | 第三輪固定三個新候選 | 地域候選小幅改善，但未達原訂 1% 門檻；不能說全部都退步 |","",
      "## 固定增強套件與純 X 檢查","",
      "最終 10 欄固定為：r4_region、r4_primary_road、r4_secondary_road、r4_road_pair、r4_project_prefix2 五類別；r4_multiple_roads、r4_secondary_log_rows、r4_secondary_log_projects、r4_prefix_log_projects、r4_repeated_city_prefix 五個 float32 數值。道路把第一／第二位置一併計數，同列同路只計一次；道路對無順序且另建字典。道路／道路對門檻 20 列且 3 地理建案，前綴固定 2 字、至少 5 地理建案且 2 行政區。未知詞回缺失、支持數回 0，query 不更新 fit 字典。",
      f"實際純 X 套件以 train＋valid 特徵建字典，新建案 test 的第二道路可用 {fmt(r['fixed_augmented_bundle']['new_test_project_category_coverage']['r4_secondary_road'])}、道路對可用 {fmt(r['fixed_augmented_bundle']['new_test_project_category_coverage']['r4_road_pair'])}、前綴可用 {fmt(r['fixed_augmented_bundle']['new_test_project_category_coverage']['r4_project_prefix2'])}。這次只測轉換與覆蓋，沒有訓練價格模型；支持字典未保存成模型。",
      "覆蓋率表為延續先前口徑，建案前綴支持採舊地理鍵；實際增強套件使用正規化地理鍵，兩者的少量差異在 JSON 分別列出，不混同為完全一致。其他原始 87 欄及 v2 reference 完全保留。",
      "", "## 驗證與重現","",
      "兩個資訊來源合併為單一固定候選，都只是機理假設，尚無新候選分數。實作前固定解析規則、支持門檻及候選數；使用主流程另外預先封存的年度開發規約，分開看已見／真正新建案／缺失三組與每個預測季度、大城市。確認期只在選型凍結後使用一次，不能用本輪稽核查看其誤差來挑特徵；不以 Public 分數選字串規則或係數。","",
      "```text",".venv\\Scripts\\python.exe src\\round4_location_audit.py","```","",
      "- [完整機器可讀稽核](../outputs/round4/audit/location_opportunity.json)",
      "- [可重現程式](../src/round4_location_audit.py)",
      "- [固定增強套件 API](../src/round4_augmented.py)",
      "- [現有特徵](../src/house_features.py)",
      "- [v2 局部參考](../src/round2_neighborhood.py)",
      "- [第三輪既有開發期結果](../outputs/round3/evaluation/development/results.json)",""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()