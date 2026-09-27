"""Build the same target-free feature schema for each supplied split."""
from __future__ import annotations
import json
import numpy as np
from prepare_data import ROOT, CACHE_DIR, REPORT_DIR, prepare_all, sha256
from house_features import build_features, get_categorical_columns

def main():
    frames = prepare_all()
    feature_source_hash = sha256(ROOT / "src" / "house_features.py")
    result = {"feature_source_sha256": feature_source_hash, "splits": {}}
    schema, categories = None, None
    for split, frame in frames.items():
        print(f"{split}: building features", flush=True)
        features = build_features(frame)
        cat = get_categorical_columns(features)
        if schema is None:
            schema, categories = list(features.columns), cat
        if list(features.columns) != schema or cat != categories:
            raise ValueError(f"Feature schema differs for {split}")
        if len(features) != len(frame) or not features.index.equals(frame.index):
            raise ValueError(f"Row alignment changed for {split}")
        if any(c in features for c in ("編號", "總價元")):
            raise ValueError("Identifier or target leaked into features")
        numeric = features.drop(columns=cat)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError(f"Infinite feature values in {split}")
        if any(str(t) != "float32" for t in numeric.dtypes):
            raise ValueError(f"Unexpected numeric feature types in {split}")
        if features[cat].isna().any().any():
            raise ValueError(f"Missing category strings in {split}")
        destination = CACHE_DIR / f"{split}_features.pkl"
        features.to_pickle(destination)
        summary = {
            "rows": len(features), "features": len(schema),
            "categorical_features": len(cat), "numeric_features": len(numeric.columns),
            "all_missing_numeric": [c for c in numeric if numeric[c].isna().all()],
            "constant_features_in_split": [c for c in features if features[c].nunique(dropna=False) <= 1],
            "invalid_transaction_dates": int(features["trade_date_invalid"].sum()),
            "out_of_range_transaction_dates": int(features["trade_date_out_of_range"].sum()),
            "input_source_sha256": json.loads((CACHE_DIR / f"{split}.json").read_text(encoding="utf-8"))["source_sha256"]}
        result["splits"][split] = summary
        print(f"{split}: checked {len(features):,} rows, {len(schema)} features", flush=True)
    result["columns"] = schema
    result["categorical_columns"] = categories
    (REPORT_DIR / "feature_schema.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All feature schemas, row alignments, types, and finite-value checks passed.", flush=True)

if __name__ == "__main__":
    main()
