"""Generate a checked submission using only the test workbook and saved models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from prepare_data import ROOT, DATA_DIR, SAMPLE_FILENAME, ID, TARGET, sha256
from house_features import build_features
from train_catboost import transform
from train_lightgbm import encode_features, price_factor

def predict_file(input_path, output_path, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(ROOT/"src"/"house_features.py") != config["feature_source_sha256"]:
        raise RuntimeError("Feature implementation does not match the saved models.")
    training_path = ROOT / "reports" / "final_training.json"
    if not training_path.is_file():
        raise RuntimeError("Run train_final.py for this configuration before prediction.")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("config_sha256") != sha256(config_path):
        raise RuntimeError("Configuration changed after final training; rerun train_final.py.")
    records = {record["name"]: record for record in training["components"]}
    if set(records) != {component["name"] for component in config["components"]}:
        raise RuntimeError("Saved model membership differs from the trained configuration.")
    for component in config["components"]:
        record = records[component["name"]]
        if sha256(ROOT / component["model_path"]) != record["model_sha256"]:
            raise RuntimeError(f"Saved model changed: {component['name']}")
        if "category_maps_path" in component:
            if sha256(ROOT / component["category_maps_path"]) != record["category_maps_sha256"]:
                raise RuntimeError(f"Saved category mappings changed: {component['name']}")
    raw = pd.read_excel(input_path,engine="openpyxl",dtype=object)
    if ID not in raw or TARGET in raw:
        raise ValueError("Expected a test workbook with IDs and without prices.")
    if raw[ID].isna().any() or not raw[ID].is_unique:
        raise ValueError("Prediction IDs must be present and unique.")
    features = build_features(raw)
    if list(features.columns) != config["feature_columns"]:
        raise ValueError("Feature schema does not match the saved models.")
    prediction = np.zeros(len(features),dtype="float64")
    if not np.isclose(sum(c["weight"] for c in config["components"]),1.0):
        raise ValueError("Ensemble weights must sum to one.")
    components = {}
    for component in config["components"]:
        path = ROOT/component["model_path"]
        if component["family"] == "catboost":
            model = CatBoostRegressor()
            model.load_model(str(path))
            _, _, factor = transform(features,None,component["target_mode"])
            values = model.predict(Pool(features,cat_features=config["categorical_columns"]),thread_count=6)*factor
        elif component["family"] == "lightgbm":
            maps = json.loads((ROOT/component["category_maps_path"]).read_text(encoding="utf-8"))
            encoded, _, _, _ = encode_features({"train":features},config["categorical_columns"],category_maps=maps)
            model = lgb.Booster(model_file=str(path))
            values = model.predict(encoded["train"],num_iteration=component["iterations"],num_threads=4)*price_factor(features,component["target_mode"])
        else:
            raise ValueError("Unknown model family")
        values = np.maximum(np.asarray(values,dtype="float64"),0)
        if values.shape != prediction.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions from {component['name']}")
        prediction += component["weight"]*values
        components[component["name"]] = {"weight":component["weight"],"minimum":float(values.min()),"maximum":float(values.max())}
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite final predictions")
    rounded = np.maximum(np.rint(prediction),1).astype("int64")
    submission = pd.DataFrame({ID:raw[ID].to_numpy(),TARGET:rounded})
    sample_path = DATA_DIR/SAMPLE_FILENAME
    if input_path.resolve() == (DATA_DIR/"test-reindex-test-v2.2.xlsx").resolve() and sample_path.exists():
        sample = pd.read_csv(sample_path,encoding="utf-8-sig")
        if submission[ID].astype("string").tolist() != sample[ID].astype("string").tolist():
            raise ValueError("Submission IDs/order differ from the supplied template.")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    submission.to_csv(output_path,index=False,encoding="utf-8-sig")
    check = pd.read_csv(output_path,encoding="utf-8-sig")
    if list(check.columns) != [ID,TARGET] or len(check) != len(raw) or not check[ID].is_unique:
        raise ValueError("Saved CSV failed shape/header/ID checks")
    if check[ID].astype("string").tolist() != raw[ID].astype("string").tolist() or not np.array_equal(check[TARGET],rounded):
        raise ValueError("Saved CSV failed prediction/row-order round-trip check")
    audit = {"path":str(output_path),"rows":len(check),"columns":list(check.columns),
             "id_order_verified":True,"finite_predictions":True,"positive_predictions":bool((rounded>0).all()),
             "prediction_unique_values":int(check[TARGET].nunique()),"minimum":int(rounded.min()),
             "maximum":int(rounded.max()),"median":float(np.median(rounded)),
             "quantiles":{str(q):float(np.quantile(rounded,q)) for q in [.01,.1,.5,.9,.99]},
             "sha256":sha256(output_path),"test_source_sha256":sha256(input_path),
             "model_integrity_verified":True,"config_sha256":sha256(config_path),
             "rounding":"nearest integer yuan, minimum 1","components":components}
    (output_path.parent/(output_path.stem+"_checks.json")).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=True),flush=True)
    return submission

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=DATA_DIR/"test-reindex-test-v2.2.xlsx")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"submission.csv")
    parser.add_argument("--config",type=Path,default=ROOT/"models"/"final_config.json")
    args=parser.parse_args()
    predict_file(args.input,args.output,args.config)

if __name__ == "__main__":
    main()
