"""Refit the selected configuration on all supplied labeled data."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from model_common import ROOT, EXPERIMENT_DIR, load_data, full_training_data
from prepare_data import sha256
from train_catboost import transform
from train_lightgbm import encode_features, effective_area, price_factor

def run(config_path, device):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    current_hash = sha256(ROOT / "src" / "house_features.py")
    if config["feature_source_sha256"] != current_hash:
        raise RuntimeError("The selected configuration refers to different feature code.")
    data = load_data()
    X, y = full_training_data(data)
    training_log = []
    for component in config["components"]:
        started = time.time()
        name, mode = component["name"], component["target_mode"]
        destination = ROOT / component["model_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        iterations = int(component["iterations"])
        print(f"Final fit: {name}, {len(X):,} labeled rows, {iterations} iterations", flush=True)
        if component["family"] == "catboost":
            params = component["parameters"].copy()
            params["iterations"] = iterations
            params["task_type"] = device
            if device == "CPU":
                params.pop("devices", None)
                params.pop("gpu_ram_part", None)
            else:
                params["devices"] = "0"
                params["gpu_ram_part"] = .75
            target, weights, _ = transform(X, y, mode)
            model = CatBoostRegressor(**params)
            model.fit(Pool(X, target, cat_features=data["categorical_columns"], weight=weights))
            model.save_model(str(destination))
        elif component["family"] == "lightgbm":
            encoded, maps, safe_names, cats = encode_features(
                {"train":X}, data["categorical_columns"])
            params = component["parameters"].copy()
            params["n_estimators"] = iterations
            model = lgb.LGBMRegressor(**params)
            factor = price_factor(X, mode)
            weights = effective_area(X) / 100.0 if mode == "unit" else None
            model.fit(encoded["train"], y / factor, sample_weight=weights,
                      categorical_feature=cats)
            model.booster_.save_model(str(destination), num_iteration=iterations)
            mapping_path = ROOT / component["category_maps_path"]
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(json.dumps(maps, ensure_ascii=False), encoding="utf-8")
        else:
            raise ValueError(f"Unknown model family: {component['family']}")
        record = {"name":name, "rows":len(X), "iterations":iterations,
                  "seconds":round(time.time()-started,2),
                  "model_sha256":sha256(destination)}
        if component["family"] == "lightgbm":
            record["category_maps_sha256"] = sha256(ROOT / component["category_maps_path"])
        training_log.append(record)
        print(json.dumps(record), flush=True)
    result = {"training_rows":len(X), "sources":["train","valid"],
              "config_sha256":sha256(config_path), "device":device,
              "feature_source_sha256":current_hash, "components":training_log}
    (ROOT / "reports" / "final_training.json").write_text(json.dumps(result,indent=2),encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"models"/"final_config.json")
    parser.add_argument("--device",choices=["GPU","CPU"],default="GPU")
    args = parser.parse_args()
    run(args.config,args.device)

if __name__ == "__main__":
    main()
