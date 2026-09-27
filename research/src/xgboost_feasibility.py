"""One fixed, artificial two-tree XGBoost GPU capability probe.

--preflight-only runs the encoder's small checks once, plus inverse boundaries.
--run does not rerun those checks, read property data, evaluate MAE, or search.
The run directory must be new; place an external console log outside it.
Fitted training configuration and reloaded inference configuration are separate
evidence: UBJSON does not promise persistence of every training parameter.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
import xgboost_geo_features as encoder


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/xgboost_research/feasibility"
ENCODER_SHA = "17df97aed051d6c61532c4ee0bc07d53fac6e72eb9baf1b64bb070f5e3c51743"
SCHEMA_PATH = "runs/v1/models/final_config.json"
INSTALLATION_PATH = "outputs/xgboost_research/installation.json"
FIT_ROWS, QUERY_ROWS, DATA_SEED, ROUNDS = 512, 32, 20260926, 2
PARAMETERS = {
    "booster": "gbtree", "tree_method": "hist", "device": "cuda:0",
    "objective": "reg:absoluteerror", "eval_metric": "mae", "seed": 42,
    "nthread": 4, "eta": 0.065, "max_depth": 8,
    "max_cat_to_onehot": 4, "max_cat_threshold": 64,
    "subsample": 1.0, "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
}
INFERENCE_RUNTIME_PARAMETERS = {"device": "cuda:0", "nthread": 4}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def vector_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    with path.with_suffix(".sha256").open("x", encoding="ascii") as stream:
        stream.write(hashlib.sha256(data).hexdigest() + "\n")


def write_vector(path, values):
    with path.open("xb") as stream:
        np.save(stream, np.asarray(values, dtype="<f8"), allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())


def event(name, **fields):
    record = {"event": name, "at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


def source_hashes():
    result = encoder.source_hashes()
    result[relative(__file__)] = sha(__file__)
    return result


def inverse_transform(raw):
    values = np.asarray(raw, dtype="<f8")
    need(values.ndim == 1 and np.isfinite(values).all(), "Raw prediction is not finite 1D")
    with np.errstate(over="raise", invalid="raise"):
        result = np.maximum(values * 1e6, 0.0)
    need(np.isfinite(result).all(), "TWD inverse overflowed")
    return np.asarray(result, dtype="<f8")


def artificial_frames(schema):
    """Only names/types come from saved configuration; every value is artificial."""
    columns, cats = schema["feature_columns"], schema["categorical_columns"]
    need(len(columns) == len(set(columns)) == 87 and len(cats) == len(set(cats)) == 21
         and set(cats) <= set(columns), "Original schema must be 87/21")
    n = FIT_ROWS + QUERY_ROWS
    row = np.arange(n)
    rng = np.random.default_rng(DATA_SEED)
    values = {}
    for number, column in enumerate(columns):
        if column in cats:
            values[column] = ["__MISSING__" if i % 16 == 0 else f"c{number}_{i % 16}"
                              for i in row]
        else:
            values[column] = rng.normal(size=n).astype("float32")
    X = pd.DataFrame(values, columns=columns)
    geo = pd.DataFrame({c: np.zeros(n, dtype="float32") for c in encoder.base.NUMERIC})
    geo["geo_latitude"] = np.full(n, 25.0, dtype="float32")
    geo["geo_longitude"] = np.full(n, 121.0, dtype="float32")
    geo["geo_has_position"] = np.ones(n, dtype="float32")
    geo["geo_match_status"] = pd.Series(["road_level"] * n, dtype=object)
    geo["geo_grid_002"] = pd.Series(["1250:6050"] * n, dtype=object)
    geo = geo[encoder.base.FEATURES]
    fit, query = X.iloc[:FIT_ROWS].copy(), X.iloc[FIT_ROWS:].copy()
    fit_geo, query_geo = geo.iloc[:FIT_ROWS].copy(), geo.iloc[FIT_ROWS:].copy()
    query.loc[query.index[0], cats[0]] = "__QUERY_ONLY_UNSEEN__"
    query.loc[query.index[1], cats[0]] = "__MISSING__"
    query_geo.loc[query.index[0], "geo_grid_002"] = "__QUERY_ONLY_GRID__"
    # Query targets are never constructed. These are artificial positive TWD.
    fit_row = np.arange(FIT_ROWS, dtype="float64")
    raw_twd = (1.0 + (fit_row % 16) / 4.0 + 0.01 * (fit_row // 16)) * 1e6
    return fit, fit_geo, query, query_geo, np.asarray(raw_twd, dtype="<f8"), list(cats)


def check_model_schema(model, config, state):
    expected_types = ["c" if c in state["categorical_columns"] else "float"
                      for c in state["model_feature_columns"]]
    learner = config["learner"]
    need(model.num_boosted_rounds() == ROUNDS and model.num_features() == 95,
         "Native model rounds/features differ")
    need(int(learner["gradient_booster"]["gbtree_model_param"]["num_trees"]) == ROUNDS,
         "Native model must contain exactly two trees")
    need(model.feature_names == state["model_feature_columns"]
         and model.feature_types == expected_types and expected_types.count("c") == 23,
         "Native feature names/types differ from fitted categorical view")
    need(learner["objective"]["name"] == "reg:absoluteerror"
         and learner["learner_train_param"]["objective"] == "reg:absoluteerror"
         and learner["gradient_booster"]["name"] == "gbtree", "Native objective/booster differs")
    return {"rounds": model.num_boosted_rounds(), "trees": ROUNDS,
            "feature_names": list(model.feature_names), "feature_types": list(model.feature_types),
            "objective": learner["objective"]["name"]}


def check_fitted_configuration(config):
    learner = config["learner"]
    generic = learner["generic_param"]
    gradient = learner["gradient_booster"]
    training = gradient["tree_train_param"]
    need(generic["device"] == "cuda:0" and int(generic["nthread"]) == 4
         and int(generic["seed"]) == 42, "Fitted GPU/runtime/seed differs; no CPU fallback")
    need(gradient["gbtree_train_param"]["tree_method"] == "hist", "Fitted tree method is not hist")
    need([entry["name"] for entry in learner["metrics"]] == ["mae"], "Fitted metric differs")
    for key in ("max_depth", "max_cat_to_onehot", "max_cat_threshold"):
        need(int(training[key]) == PARAMETERS[key], "Fitted categorical/tree setting differs: " + key)
    need(np.isclose(float(training["eta"]), PARAMETERS["eta"], rtol=1e-7, atol=0),
         "Fitted learning rate differs")
    for key in ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"):
        need(float(training[key]) == 1.0, "Fitted sampling differs: " + key)


def preflight_only():
    need(sha(ROOT / "src/xgboost_geo_features.py") == ENCODER_SHA, "Encoder source changed")
    checks = encoder.self_check()  # Exactly once here; never called from run().
    need(checks["status"] == "passed", "Encoder preflight failed")
    need(np.array_equal(inverse_transform([-2.0, 0.0, 0.25, 3.0]), [0.0, 0.0, 250000.0, 3000000.0]),
         "Fixed TWD inverse differs")
    for raw in ([np.inf], [np.nan], [np.finfo("float64").max]):
        rejected = False
        try:
            inverse_transform(raw)
        except (ValueError, FloatingPointError):
            rejected = True
        need(rejected, "Nonfinite or overflow raw prediction was accepted")
    return {"status": "passed", "encoder": checks,
            "inverse_checks": ["fixed_nonnegative_million_scale", "nonfinite_and_overflow_rejected"],
            "source_sha256": source_hashes(), "xgboost_imported_here": False,
            "model_training_performed": False, "real_data_read": False, "file_outputs_written": False}


def run():
    # mkdir is itself exclusive. A stopped or failed attempt is never deleted/retried.
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(OUT / "run_claim.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "planned_fits": 1,
        "rounds": ROUNDS, "source_sha256": sha(__file__), "retry_allowed": False,
        "passed_parameters": PARAMETERS,
    })
    with (OUT / "events.jsonl").open("x", encoding="utf-8"):
        pass
    record = {
        "status": "not_completed", "synthetic_only": True,
        "fits_started": 0, "fits_completed": 0, "models_saved": 0,
        "models_reloaded": 0, "fully_verified_models": 0, "planned_fits": 1,
        "fit_rows": FIT_ROWS, "query_rows": QUERY_ROWS, "feature_count": 95,
        "categorical_feature_count": 23, "data_seed": DATA_SEED, "rounds": ROUNDS,
        "passed_parameters": copy.deepcopy(PARAMETERS), "sample_weight": "uniform; no weight argument",
        "real_property_data_or_models_loaded": False, "query_targets_generated": False,
        "MAE_computed": False, "search_performed": False, "eval_set_used": False,
        "early_stopping_used": False, "preflight_repeated_in_run": False,
        "read_scope": "source bytes, saved feature-name configuration and installation evidence only; artificial values",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "reload_runtime_parameters": INFERENCE_RUNTIME_PARAMETERS,
        "all_training_parameters_expected_to_persist": False,
        "fitted_configuration_is_training_authority": True,
    }
    phase = "verify_sources_and_installation"
    try:
        record["source_sha256"] = source_hashes()
        need(record["source_sha256"]["src/xgboost_geo_features.py"] == ENCODER_SHA, "Encoder identity changed")
        install_path = ROOT / INSTALLATION_PATH
        install_receipt = install_path.with_suffix(".sha256")
        need(install_receipt.read_text(encoding="ascii").strip() == sha(install_path), "Installation receipt mismatch")
        installation = json.loads(install_path.read_text(encoding="utf-8"))
        need(installation["status"] == "installed_and_readonly_import_verified"
             and installation["installed_version"] == "3.4.1", "Unexpected XGBoost installation")
        record["input_sha256"] = {p: sha(ROOT / p) for p in
            (SCHEMA_PATH, INSTALLATION_PATH, relative(install_receipt))}
        import xgboost as xgb
        need(xgb.__version__ == "3.4.1", "Probe requires installed XGBoost 3.4.1")
        record["runtime"]["xgboost"] = xgb.__version__
        record["native_build_info"] = xgb.build_info()
        need(record["native_build_info"]["USE_CUDA"] is True, "Installed build has no CUDA capability")
        phase = "prepare_artificial_features"
        schema = json.loads((ROOT / SCHEMA_PATH).read_text(encoding="utf-8"))
        X, geo, Xq, geoq, raw_y, cats = artificial_frames(schema)
        state = encoder.fit_feature_state(X, geo, cats)
        fit_view = encoder.encode_query(state, X, geo, cats)
        query_view = encoder.encode_query(state, Xq, geoq, cats)
        state_before = encoder.content_sha(state)
        need(pd.isna(query_view[cats[0]].iloc[0]) and pd.isna(query_view["geo_grid_002"].iloc[0]),
             "Artificial unknown categories were not mapped to NaN")
        need(query_view[cats[0]].iloc[1] == "__MISSING__"
             and "__MISSING__" in query_view[cats[0]].cat.categories,
             "Fitted missing string must remain distinct from unseen")
        record["artificial_frames"] = {
            "fit_X_sha256": encoder.base.frame_sha(X), "fit_geo_sha256": encoder.base.frame_sha(geo),
            "query_X_sha256": encoder.base.frame_sha(Xq), "query_geo_sha256": encoder.base.frame_sha(geoq),
            "fit_encoded_sha256": encoder.base.frame_sha(fit_view),
            "query_encoded_sha256": encoder.base.frame_sha(query_view),
            "fit_original_target_sha256": vector_sha(raw_y),
            "fit_scaled_target_sha256": vector_sha(raw_y / 1e6),
            "target_formula": "(1 + (fit_row % 16)/4 + .01*(fit_row // 16))*1e6 synthetic TWD; fit target /1e6",
            "query_unknown_is_missing": True, "fit_missing_token_remains_category": True,
            "feature_state_content_sha256": state["state_content_sha256"],
            "vocabulary_content_sha256": state["vocabulary_content_sha256"],
        }
        write_json(OUT / "feature_state.json", state)
        # No query label and no manual categorical codes: native receives pandas categories.
        train = xgb.DMatrix(fit_view, label=raw_y / 1e6, enable_categorical=True, nthread=4)
        query = xgb.DMatrix(query_view, enable_categorical=True, nthread=4)
        need(train.num_row() == FIT_ROWS and query.num_row() == QUERY_ROWS
             and train.get_weight().size == 0 and query.get_label().size == 0,
             "Artificial DMatrix row/weight/label scope differs")
        phase = "fit"
        record["fits_started"] = 1
        event("fit_started", fit_number=1, rounds=ROUNDS, fit_rows=FIT_ROWS)
        fit_started = time.monotonic()
        model = xgb.train(copy.deepcopy(PARAMETERS), train, num_boost_round=ROUNDS,
                          evals=(), verbose_eval=False)
        record["fit_seconds"] = time.monotonic() - fit_started
        record["fits_completed"] = 1
        event("fit_completed", completed_count=1)
        phase = "capture_fitted_configuration"
        fitted = json.loads(model.save_config())
        write_json(OUT / "fitted_config.json", fitted)
        record["fitted_native_model"] = check_model_schema(model, fitted, state)
        check_fitted_configuration(fitted)
        raw = np.asarray(model.predict(query, validate_features=True), dtype="<f8")
        need(raw.shape == (QUERY_ROWS,), "Artificial raw query length differs")
        twd = inverse_transform(raw)
        write_vector(OUT / "raw_z.npy", raw)
        write_vector(OUT / "predictions.npy", twd)
        phase = "save_and_reload"
        model_path = OUT / "model.ubj"
        need(not model_path.exists(), "Refuse to overwrite UBJSON model")
        model.save_model(model_path)
        record["models_saved"] = 1
        event("model_saved", path=relative(model_path), sha256=sha(model_path))
        loaded = xgb.Booster()
        loaded.load_model(model_path)
        record["models_reloaded"] = 1
        loaded_before = json.loads(loaded.save_config())
        write_json(OUT / "loaded_config_before_runtime.json", loaded_before)
        check_model_schema(loaded, loaded_before, state)
        loaded.set_param(copy.deepcopy(INFERENCE_RUNTIME_PARAMETERS))
        loaded_after = json.loads(loaded.save_config())
        write_json(OUT / "loaded_config_after_runtime.json", loaded_after)
        record["reloaded_native_model"] = check_model_schema(loaded, loaded_after, state)
        generic = loaded_after["learner"]["generic_param"]
        need(generic["device"] == "cuda:0" and int(generic["nthread"]) == 4,
             "Reload inference runtime was not restored")
        phase = "verify_state_and_full_query_replay"
        restored = json.loads((OUT / "feature_state.json").read_text(encoding="utf-8"))
        replay_view = encoder.encode_query(restored, Xq, geoq, cats)
        pd.testing.assert_frame_equal(query_view, replay_view)
        replay_query = xgb.DMatrix(replay_view, enable_categorical=True, nthread=4)
        replay_raw = np.asarray(loaded.predict(replay_query, validate_features=True), dtype="<f8")
        replay_twd = inverse_transform(replay_raw)
        saved_raw = np.load(OUT / "raw_z.npy", allow_pickle=False)
        saved_twd = np.load(OUT / "predictions.npy", allow_pickle=False)
        need(replay_raw.shape == replay_twd.shape == raw.shape == (QUERY_ROWS,)
             and raw.tobytes() == saved_raw.tobytes() == replay_raw.tobytes()
             and twd.tobytes() == saved_twd.tobytes() == replay_twd.tobytes(),
             "Full 32-row raw/TWD model or array serialization replay differs")
        need(encoder.content_sha(state) == state_before
             and encoder.base.frame_sha(fit_view) == record["artificial_frames"]["fit_encoded_sha256"]
             and encoder.base.frame_sha(query_view) == record["artificial_frames"]["query_encoded_sha256"],
             "Artificial fit/query state or encoded frames changed")
        record["prediction_replay"] = {
            "rows": QUERY_ROWS, "raw_bitwise_equal": True, "TWD_bitwise_equal": True,
            "raw_values_sha256": vector_sha(raw), "TWD_values_sha256": vector_sha(twd),
            "saved_array_dtype": "float64; native prediction cast without rounding",
            "state_JSON_roundtrip_equal": True, "no_query_vocabulary_learning": True,
        }
        record["fully_verified_models"] = 1
        event("model_verified", rows=QUERY_ROWS, raw_bitwise_equal=True, TWD_bitwise_equal=True)
        phase = "verify_unchanged_sources"
        for path, digest in {**record["source_sha256"], **record["input_sha256"]}.items():
            need(sha(ROOT / path) == digest, "Input/source changed during probe: " + path)
        record.update(status="synthetic_GPU_configuration_and_replay_verified",
                      source_and_inputs_unchanged=True,
                      configuration_persistence_scope="objective, names/types, rounds/trees and predictions checked; complete training-config equality not required")
    except Exception as exc:
        record.update(status="synthetic_GPU_probe_failed", failure_phase=phase,
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        event("run_failed", phase=phase, error_type=type(exc).__name__, error=str(exc))
    record.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, no_retry_or_search_performed=True)
    record["artifact_sha256"] = {relative(p): sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json(OUT / "feasibility.json", record)
    print(json.dumps({"status": record["status"], "path": relative(OUT / "feasibility.json"),
                      "sha256": sha(OUT / "feasibility.json"), "fits_started": record["fits_started"],
                      "fits_completed": record["fits_completed"], "fully_verified_models": record["fully_verified_models"]}), flush=True)
    return 0 if record["status"] == "synthetic_GPU_configuration_and_replay_verified" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight_only(), ensure_ascii=False, allow_nan=False), flush=True)
    else:
        raise SystemExit(run())
