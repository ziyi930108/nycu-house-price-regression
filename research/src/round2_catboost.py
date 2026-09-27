"""Bounded local CatBoost round: two temporal trials, then the best on original valid."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import json
import time
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import catboost
from model_common import ROOT, load_data, temporal_indices, mae, evaluate
from train_catboost import params, transform
from prepare_data import sha256

OUT = ROOT / "outputs" / "round2" / "catboost"
CONFIGS = [
    {"name": "catboost_recency2y_d8", "depth": 8, "iterations": 2800,
     "learning_rate": 0.065, "half_life_days": 730.0, "minimum_weight": 0.1},
    {"name": "catboost_depth9", "depth": 9, "iterations": 2600,
     "learning_rate": 0.065, "half_life_days": None, "minimum_weight": 1.0},
]

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def rel(path):
    return path.relative_to(ROOT).as_posix()

def guard_shared_cache():
    schema = json.loads((ROOT / "reports" / "feature_schema.json").read_text(encoding="utf-8"))
    signature = {s: schema["splits"][s]["input_source_sha256"] for s in ("train", "valid")}
    existing = ROOT / "data_cache" / "valid_clean_mask.json"
    if not existing.is_file() or not (existing.parent / "valid_clean_mask.npy").is_file():
        raise RuntimeError("Existing clean-mask cache is required; round2 cannot write shared caches.")
    if json.loads(existing.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("Shared clean-mask cache is stale.")
    return schema

def recency_weights(X, config):
    if config["half_life_days"] is None:
        return np.ones(len(X), dtype="float64"), {"method": "uniform"}
    days = X["trade_day_index"].to_numpy(dtype="float64")
    finite = np.isfinite(days)
    anchor = float(days[finite].max())
    weights = np.full(len(X), config["minimum_weight"], dtype="float64")
    weights[finite] = np.maximum(config["minimum_weight"],
                                np.exp2(-(anchor - days[finite]) / config["half_life_days"]))
    return weights, {"method": "training-date-only exponential recency",
                     "anchor_day_index_from_training_only": anchor,
                     "half_life_days": config["half_life_days"],
                     "minimum_weight": config["minimum_weight"],
                     "actual_min": float(weights.min()), "actual_max": float(weights.max()),
                     "actual_mean": float(weights.mean())}

def predict_raw(model, X, categorical):
    return np.maximum(model.predict(Pool(X, cat_features=categorical, thread_count=5), thread_count=5) * 1e6, 0.0)

def fit_final_model(X, y, categorical_columns, candidate="catboost_recency2y_d8", iterations=None):
    """Explicit final-refit API; returns (model, metadata), never writes shared files.

    The caller supplies its deduplicated final training table and saves the model.
    Recency weights use dates from X only. No evaluation or test labels are read.
    Call predict_raw(model, X_test, categorical_columns) for total-price predictions.
    This helper consumes the GPU only when explicitly called.
    """
    matches = [dict(c) for c in CONFIGS if c["name"] == candidate]
    if not matches:
        raise ValueError("Unknown bounded-round candidate: " + str(candidate))
    config = matches[0]
    if iterations is not None:
        if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
            raise ValueError("iterations must be a positive integer.")
        config["iterations"] = int(iterations)
    y = np.asarray(y, dtype="float64")
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("Final training labels must be aligned and finite.")
    args = argparse.Namespace(iterations=config["iterations"], depth=config["depth"],
                              learning_rate=config["learning_rate"], target_mode="raw",
                              device="GPU", structural=False)
    configuration = {k:v for k,v in params(args).items() if v is not None}
    configuration.update(thread_count=5, verbose=300)
    target, base_weights, _ = transform(X, y, "raw")
    recency, recency_info = recency_weights(X, config)
    training = Pool(X, target, cat_features=categorical_columns,
                    weight=base_weights * recency, thread_count=5)
    model = CatBoostRegressor(**configuration)
    model.fit(training)
    metadata = {"name":candidate, "family":"catboost", "target_mode":"raw",
                "parameters":configuration, "recency":recency_info,
                "iterations":int(model.tree_count_), "training_rows":len(X),
                "external_property_data_used":False}
    return model, metadata

def fit_trial(config, mode, data, fit_idx, time_idx, unseen_rate):
    start = time.monotonic()
    if mode == "temporal":
        X, y = data["X_train"].iloc[fit_idx], data["y_train"][fit_idx]
        Xv, yv = data["X_train"].iloc[time_idx], data["y_train"][time_idx]
    else:
        X, y, Xv, yv = data["X_train"], data["y_train"], data["X_valid"], data["y_valid"]
    args = argparse.Namespace(iterations=config["iterations"], depth=config["depth"],
                              learning_rate=config["learning_rate"], target_mode="raw",
                              device="GPU", structural=False)
    configuration = {k:v for k,v in params(args).items() if v is not None}
    configuration.update(thread_count=5, verbose=300)
    target, base_weights, _ = transform(X, y, "raw")
    recency, recency_info = recency_weights(X, config)
    train_pool = Pool(X, target, cat_features=data["categorical_columns"], weight=base_weights * recency, thread_count=5)
    valid_pool = Pool(Xv, np.asarray(yv) / 1e6, cat_features=data["categorical_columns"], thread_count=5)
    model = CatBoostRegressor(**configuration)
    print(json.dumps({"event":"fit_started","name":config["name"],"mode":mode,
                      "train_rows":len(X),"valid_rows":len(Xv),"parameters":configuration,
                      "recency":recency_info}), flush=True)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=180)
    pred = predict_raw(model, Xv, data["categorical_columns"])
    assert len(pred) == (58206 if mode == "temporal" else 11994)
    model_path = OUT / (config["name"] + "_" + mode + ".cbm")
    prediction_path = OUT / (config["name"] + ("_temporal_valid.npy" if mode=="temporal" else "_valid.npy"))
    model.save_model(str(model_path))
    np.save(prediction_path, pred)
    result = {"name":config["name"],"mode":mode,"model_family":"CatBoost","version":catboost.__version__,
              "target_mode":"raw","parameters":configuration,"recency":recency_info,
              "best_iteration":int(model.tree_count_),"training_rows":len(X),"validation_rows":len(Xv),
              "model_path":rel(model_path),"model_sha256":sha256(model_path),
              "predictions_path":rel(prediction_path),"mae":mae(yv,pred),
              "elapsed_seconds":round(time.monotonic()-start,2),
              "external_property_data_used":False}
    if mode == "temporal":
        seen = Xv["project_name"].isin(set(X["project_name"]) - {"__MISSING__"}).to_numpy()
        result.update(cutoff="2024-04-01",end="2024-08-10",seen_project_rows=int(seen.sum()),
                      unseen_project_rows=int((~seen).sum()),seen_project_mae=mae(yv[seen],pred[seen]),
                      unseen_project_mae=mae(yv[~seen],pred[~seen]),test_unseen_project_rate=unseen_rate)
        result["reweighted_temporal_mae"] = ((1-unseen_rate)*result["seen_project_mae"] +
                                            unseen_rate*result["unseen_project_mae"])
        index_path = OUT / (config["name"] + "_temporal_indices.npy")
        np.save(index_path,time_idx)
        result["temporal_indices_path"] = rel(index_path)
        result["temporal_indices_sha256"] = sha256(index_path)
        np.save(OUT / (config["name"] + "_temporal_seen_project.npy"), seen)
    else:
        result.update(evaluate(data,pred))
    write_json(OUT / (config["name"] + "_" + mode + ".json"),result)
    print(json.dumps({"event":"fit_completed",**{k:result[k] for k in
          ("name","mode","best_iteration","mae","elapsed_seconds")},
          "reweighted_temporal_mae":result.get("reweighted_temporal_mae")}), flush=True)
    del model,train_pool,valid_pool,X,Xv
    gc.collect()
    return result

def update_manifest(records, schema):
    manifest=[]
    for config in CONFIGS:
        name=config["name"]
        if name not in records or "temporal" not in records[name]:
            continue
        info=records[name]
        metadata={"name":name,"family":"catboost","target_mode":"raw",
                  "feature_source_sha256":schema["feature_source_sha256"],
                  "configuration":config,"temporal":info["temporal"],
                  "validation":info.get("validation")}
        metadata_path=OUT/(name+".json")
        write_json(metadata_path,metadata)
        manifest.append({"name":name,
                         "valid_predictions":info.get("validation",{}).get("predictions_path"),
                         "temporal_predictions":info["temporal"]["predictions_path"],
                         "temporal_indices":info["temporal"]["temporal_indices_path"],
                         "metadata_path":rel(metadata_path)})
    write_json(OUT/"candidates.json",manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["all","temporal","validate"],default="all")
    parser.add_argument("--candidate",choices=[x["name"] for x in CONFIGS])
    parser.add_argument("--soft-budget-seconds",type=int,default=840)
    args=parser.parse_args()
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    schema=guard_shared_cache()
    data=load_data()
    fit_idx,time_idx=temporal_indices(data,"2024-04-01","2024-08-10")
    assert np.array_equal(time_idx,np.load(ROOT/"outputs"/"experiments"/"selected_temporal_indices.npy",allow_pickle=False))
    assert len(fit_idx)==397041 and len(time_idx)==58206
    old_results=json.loads((ROOT/"reports"/"results.json").read_text(encoding="utf-8"))
    unseen_rate=float(old_results["selection"]["test_unseen_project_rate"])
    baseline_pred=np.load(ROOT/"outputs"/"experiments"/"catboost_raw_temporal_valid.npy",allow_pickle=False)
    Xt=data["X_train"].iloc[time_idx];yt=data["y_train"][time_idx]
    seen=Xt["project_name"].isin(set(data["X_train"].iloc[fit_idx]["project_name"])-{"__MISSING__"}).to_numpy()
    month=(Xt["trade_year"].astype(int)*100+Xt["trade_month"].astype(int)).to_numpy()
    baseline={"name":"catboost_raw","mae":mae(yt,baseline_pred),
              "seen_project_mae":mae(yt[seen],baseline_pred[seen]),
              "unseen_project_mae":mae(yt[~seen],baseline_pred[~seen]),
              "month_mae":{str(m):mae(yt[month==m],baseline_pred[month==m]) for m in np.unique(month)},
              "test_unseen_project_rate":unseen_rate}
    baseline["reweighted_temporal_mae"]=(1-unseen_rate)*baseline["seen_project_mae"]+unseen_rate*baseline["unseen_project_mae"]
    write_json(OUT/"baseline_diagnostics.json",baseline)
    print(json.dumps({"event":"baseline_diagnostics",**baseline}),flush=True)
    records={}
    for config in CONFIGS:
        name=config["name"]
        for mode in ("temporal","validation"):
            path=OUT/(name+"_"+mode+".json")
            if path.exists(): records.setdefault(name,{})[mode]=json.loads(path.read_text(encoding="utf-8"))
    chosen=[c for c in CONFIGS if args.candidate is None or c["name"]==args.candidate]
    if args.stage in ("all","temporal"):
        for config in chosen:
            if time.monotonic()-started>args.soft_budget_seconds-240 and records:
                print("Soft budget: skipping further temporal candidates.",flush=True)
                break
            result=fit_trial(config,"temporal",data,fit_idx,time_idx,unseen_rate)
            records.setdefault(config["name"],{})["temporal"]=result
            update_manifest(records,schema)
    available=[c for c in chosen if "temporal" in records.get(c["name"],{})]
    if not available:raise RuntimeError("No temporal candidate available.")
    best=min(available,key=lambda c:records[c["name"]]["temporal"]["reweighted_temporal_mae"])
    if args.stage in ("all","validate"):
        result=fit_trial(best,"validation",data,fit_idx,time_idx,unseen_rate)
        records[best["name"]]["validation"]=result
        update_manifest(records,schema)
    summary={"baseline":baseline,"best_temporal_candidate":best["name"],
             "candidate_results":records,"elapsed_seconds":round(time.monotonic()-started,2),
             "no_final_fit":True,"no_test_predictions":True}
    write_json(OUT/"round_summary.json",summary)
    print(json.dumps({"event":"round_completed","best":best["name"],
                      "elapsed_seconds":summary["elapsed_seconds"],
                      "manifest":rel(OUT/"candidates.json")}),flush=True)

if __name__=="__main__":
    main()
