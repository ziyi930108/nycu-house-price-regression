"""Train CatBoost candidates, a chronological check, or a final refit."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import catboost
from model_common import (ROOT, MODEL_DIR, EXPERIMENT_DIR, load_data, save_experiment,
                          full_training_data, temporal_indices, mae)

def transform(features, prices, target_mode):
    if target_mode == "raw":
        factor = np.full(len(features), 1e6)
        weights = np.ones(len(features))
    else:
        area = features["building_area"].to_numpy(dtype="float64")
        parking = features["parking_area"].to_numpy(dtype="float64")
        area = np.where(np.isfinite(area) & (area > 0), area,
                        np.where(np.isfinite(parking) & (parking > 0), parking, 1.0))
        factor, weights = area * 1e4, area / 100.0
    return (None if prices is None else np.asarray(prices) / factor), weights, factor

def params(args):
    configuration = dict(iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
                loss_function="MAE", eval_metric="MAE", l2_leaf_reg=6.0, random_seed=42,
                task_type=args.device, devices="0" if args.device == "GPU" else None,
                thread_count=6, boosting_type="Plain", bootstrap_type="Bernoulli",
                subsample=.85, random_strength=.5, one_hot_max_size=8,
                max_ctr_complexity=2, border_count=128, allow_writing_files=False,
                gpu_ram_part=.75 if args.device == "GPU" else None,
                verbose=200)
    if getattr(args, "structural", False):
        configuration["ignored_features"] = ["project_name", "address_raw", "building_and_unit"]
    return configuration

def fit_model(X, y, Xv, yv, categorical, args):
    target, weights, _ = transform(X, y, args.target_mode)
    train_pool = Pool(X, target, cat_features=categorical, weight=weights)
    configuration = {k:v for k,v in params(args).items() if v is not None}
    model = CatBoostRegressor(**configuration)
    eval_pool = None
    if Xv is not None:
        target_v, weights_v, _ = transform(Xv, yv, args.target_mode)
        eval_pool = Pool(Xv, target_v, cat_features=categorical, weight=weights_v)
    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=200 if eval_pool is not None else None,
              use_best_model=eval_pool is not None)
    return model, configuration

def predict(model, features, categorical, mode):
    _, _, factor = transform(features, None, mode)
    prediction = model.predict(Pool(features, cat_features=categorical), thread_count=6) * factor
    return np.maximum(prediction, 0.0)

def run(args):
    started = time.time()
    data = load_data()
    if args.mode == "validate":
        X, y, Xv, yv = data["X_train"], data["y_train"], data["X_valid"], data["y_valid"]
    elif args.mode == "temporal":
        fit_idx, valid_idx = temporal_indices(data, args.cutoff)
        X, y = data["X_train"].iloc[fit_idx], data["y_train"][fit_idx]
        Xv, yv = data["X_train"].iloc[valid_idx], data["y_train"][valid_idx]
    else:
        X, y = full_training_data(data)
        Xv, yv = None, None
    print(json.dumps({"mode":args.mode,"name":args.name,"train_rows":len(X),
                      "valid_rows":0 if Xv is None else len(Xv),"target_mode":args.target_mode}), flush=True)
    model, configuration = fit_model(X, y, Xv, yv, data["categorical_columns"], args)
    suffix = "" if args.mode == "validate" else "_" + args.mode
    model_path = MODEL_DIR / f"{args.name}{suffix}.cbm"
    model.save_model(str(model_path))
    metadata = {"model_family":"CatBoost", "version":catboost.__version__,
                "target_mode":args.target_mode,"parameters":configuration,
                "best_iteration":int(model.tree_count_),"training_rows":len(X),
                "model_file":str(model_path.relative_to(ROOT)),
                "duration_seconds":round(time.time()-started,2)}
    pd.DataFrame({"feature":X.columns,"importance":model.get_feature_importance()}).sort_values(
        "importance",ascending=False).to_csv(EXPERIMENT_DIR / f"{args.name}{suffix}_importance.csv",index=False)
    if args.mode == "validate":
        pv = predict(model, Xv, data["categorical_columns"], args.target_mode)
        pt = predict(model, data["X_test"], data["categorical_columns"], args.target_mode)
        save_experiment(args.name, data, pv, pt, metadata)
    elif args.mode == "temporal":
        pv = predict(model, Xv, data["categorical_columns"], args.target_mode)
        seen = Xv["project_name"].isin(set(X["project_name"]) - {"__MISSING__"}).to_numpy()
        result = {**metadata,"name":args.name,"cutoff":args.cutoff,"end":"2024-08-10",
                  "mae":mae(yv,pv),"validation_rows":len(yv),"seen_project_rows":int(seen.sum()),
                  "unseen_project_rows":int((~seen).sum()),"seen_project_mae":mae(yv[seen],pv[seen]),
                  "unseen_project_mae":mae(yv[~seen],pv[~seen]),
                  "median_baseline_mae":mae(yv,np.full(len(yv),np.median(y)))}
        np.save(EXPERIMENT_DIR / f"{args.name}_temporal_valid.npy",pv)
        np.save(EXPERIMENT_DIR / f"{args.name}_temporal_indices.npy",valid_idx)
        (EXPERIMENT_DIR / f"{args.name}_temporal.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result),flush=True)
    else:
        pt = predict(model,data["X_test"],data["categorical_columns"],args.target_mode)
        np.save(EXPERIMENT_DIR / f"{args.name}_final_test.npy",pt)
        (EXPERIMENT_DIR / f"{args.name}_final.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        print(json.dumps(metadata),flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["validate","temporal","final"],default="validate")
    parser.add_argument("--target-mode",choices=["raw","unit"],default="raw")
    parser.add_argument("--name",default="catboost_raw")
    parser.add_argument("--iterations",type=int,default=3000)
    parser.add_argument("--depth",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=.065)
    parser.add_argument("--device",choices=["GPU","CPU"],default="GPU")
    parser.add_argument("--cutoff",default="2024-04-01")
    parser.add_argument("--structural", action="store_true", help="Ignore three high-cardinality property identity categories.")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
