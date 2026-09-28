"""Fresh-cutoff v22 NN preparation and explicitly invoked serial training.

prepare_nn fits new R47/R54 encoders and R60 grouped OOF/full priors using only
the caller's fit rows. train_nn is the only GPU entry point; importing or
preparing never starts a trainer. This module does not load housing datasets,
query labels, test files, old encoders, old priors or previous trained models.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import round6_geo as geo
import round47_nn_encode as encoder
import round54_numeric_encode as ramps
import round60_recency_price_prior as prior
from round19_unit_geo import effective_area

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "R90_fresh_cutoff_v22_NN_preparation_v1"
SOURCE_INDEX_CONTRACT = {
    "axis": "positions in the original canonical train matrix, before this cutoff split",
    "caller_field": "round90_offset_data.OffsetData.fit_indices",
    "not_raw_cache_indices": True,
    "raw_cache_mapping_role": "fit_source_cache_indices is only for extracting aligned fit y",
    "not_reset_fit_row_numbers": True,
    "consumer": "R60 project/fallback group OOF; missing-row singleton keys use these canonical positions",
    "caller_must_bind_to_split_manifest": True,
}
LOCKED_SOURCES = {
    "src/round47_nn_encode.py": "6bee3297f761d2cda0a0b9224aebc61e6748c9e56d8738d1fe344ea0abda4c47",
    "src/round54_numeric_encode.py": "fe73f4154bcd0811b7cdad09c58e1dfd7cba5350446193e90f70bd8a7ef0206e",
    "src/round48_nn_train.py": "38fb91e01eabb2ed3163f1260e16c826ca55592951c22ce1107ad68c24b37082",
    "src/round63_recency_weighted_prior_train.py": "26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a",
    "src/round60_recency_price_prior.py": "4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f",
    "src/round52_price_prior.py": "3617c88689976aa27e0a6430c75bb005cbe0c283746d32eddbc1755417e0bb07",
    "src/round47_nn_train.py": "68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7",
    "src/round52_nn_train.py": "0d27f93d9c48c7dc13cf0252a16eadd1a2aa8a7f53b0740021fc3d8b37c0740f",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True)
                          .to_numpy(dtype="<u8").tobytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe_path(path):
    result = Path(path)
    if not result.is_absolute():
        result = ROOT / result
    result = result.resolve()
    need(result.is_relative_to(ROOT), "NN artifact path must stay inside workspace")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def write_npy(path, a):
    with Path(path).open("xb") as f:
        np.save(f, a, allow_pickle=False)


def source_hashes():
    paths = list(LOCKED_SOURCES) + ["src/round6_geo.py", "src/round19_unit_geo.py",
                                  "src/v3_cluster_stability.py", "src/round90_offset_nn.py"]
    hashes = {p: sha(ROOT / p) for p in paths}
    need(all(hashes[p] == digest for p, digest in LOCKED_SOURCES.items()),
         "An immutable v22 NN source changed")
    return hashes


def _array_record(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "values_sha256": values_sha(a)}


def _positive(a, rows, name):
    need(a.dtype == np.float32 and a.shape == (rows,) and np.isfinite(a).all()
         and (a > 0).all(), name + " must be an ordered positive finite float32 vector")


def _save_npz(path, arrays):
    record = {key: _array_record(value) for key, value in arrays.items()}
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        need(set(loaded.files) == set(arrays), "NPZ schema changed on save")
        for key in arrays:
            need(_array_record(loaded[key]) == record[key], "NPZ roundtrip differs: " + key)
    return record


def prepare_nn(Xfit, yfit, Gtreefit, Xq, Gtreeq, source_indices, out):
    """Save fresh fit-only state and two trainer NPZs; return JSON-safe paths/SHA.

    source_indices MUST be OffsetData.fit_indices: positions in the original
    canonical train matrix before the cutoff split, aligned to Xfit/yfit.
    It is NOT fit_source_cache_indices (raw-cache mapping for extracting y),
    and NOT reset 0..n_fit-1 row numbers. The original R60 convention is kept
    because missing-row singleton keys and their fold hash depend on this axis.
    Caller remains responsible for split/raw-source/geography provenance.
    input_sha256['source_indices'] binds the supplied canonical vector; the
    caller must compare it with its saved split manifest before train_nn.
    Both geography frames must be the eight original R6 tree features.
    """
    started = time.monotonic()
    sources = source_hashes()
    need(isinstance(Xfit, pd.DataFrame) and isinstance(Xq, pd.DataFrame), "DataFrames required")
    need(len(Xfit) > 0 and len(Xq) > 0 and Xfit.index.is_unique and Xq.index.is_unique,
         "Nonempty unique fit/query indices required")
    need(list(Xfit.columns) == list(Xq.columns) and Xfit.dtypes.equals(Xq.dtypes),
         "Fit/query feature order or dtypes differ")
    need(not {"總價元", "y", "target", "ID", "id", "編號"}.intersection(Xfit.columns),
         "Targets/identifiers must not enter NN feature inputs")
    if isinstance(yfit, pd.Series):
        need(yfit.index.equals(Xfit.index), "Fit label Series index differs")
    y = np.asarray(yfit, dtype=np.float64)
    need(y.shape == (len(Xfit),) and np.isfinite(y).all() and (y > 0).all(),
         "Aligned positive finite fit prices required")
    indices = np.asarray(source_indices)
    need(indices.dtype == np.int64 and indices.shape == y.shape and (indices >= 0).all()
         and len(np.unique(indices)) == len(indices), "Ordered unique int64 source indices required")
    cats21 = [c for c in Xfit if pd.api.types.is_object_dtype(Xfit[c].dtype)
              or pd.api.types.is_string_dtype(Xfit[c].dtype)]
    cats = geo.categories(Xfit, cats21)
    xf = geo.augment_features(Xfit, Gtreefit)
    xq = geo.augment_features(Xq, Gtreeq)
    input_frames = {"fit_features": Xfit, "query_features": Xq,
                    "fit_tree_geo": Gtreefit, "query_tree_geo": Gtreeq}
    input_sha = {key: frame_sha(frame) for key, frame in input_frames.items()}
    input_sha.update(fit_target=values_sha(y), source_indices=values_sha(indices))
    out = safe_path(out)
    out.mkdir(parents=True, exist_ok=True)
    need(not any(out.iterdir()), "NN preparation directory must be empty; preserve previous evidence")
    write_json(out / "preparation_started.json", {"format": FORMAT, "started_at_utc": now(),
               "input_sha256": input_sha, "source_sha256": sources,
               "source_indices_contract": SOURCE_INDEX_CONTRACT, "query_labels_available": False})

    state = encoder.fit_encoder(xf, cats)
    tn, tc = encoder.encode(state, xf)
    qn, qc = encoder.encode(state, xq)
    write_json(out / "encoder_state.json", state)
    reloaded = read_json(out / "encoder_state.json")
    for original, restored in zip((tn, tc, qn, qc),
                                 (*encoder.encode(reloaded, xf), *encoder.encode(reloaded, xq))):
        need(_array_record(original) == _array_record(restored), "Encoder reload values differ")
    ramp_state = ramps.fit_state(tn)
    write_json(out / "ramp_state.json", ramp_state)
    ramp_reloaded = read_json(out / "ramp_state.json")
    need(ramp_state == ramp_reloaded, "Ramp JSON reload differs")
    card = np.asarray(state["cardinalities"], dtype=np.int64)
    ym = np.asarray(y / 1e6, dtype=np.float32)
    area = effective_area(Xfit)
    query_area = effective_area(Xq)
    af = np.asarray(area / 100, dtype=np.float32)
    qaf = np.asarray(query_area / 100, dtype=np.float32)
    for a, rows, label in ((ym, len(y), "million target"), (af, len(y), "fit area factor"),
                           (qaf, len(Xq), "query area factor")):
        _positive(a, rows, label)
    ramp_tn = ramps.transform(ramp_reloaded, tn)
    ramp_qn = ramps.transform(ramp_reloaded, qn)
    need(ramp_tn[:, :144].tobytes() == tn.tobytes()
         and ramp_qn[:, :144].tobytes() == qn.tobytes(), "R54 changed original numeric prefix")
    common = {"train_cat": tc, "train_y_million": ym, "query_cat": qc, "cardinalities": card}
    area_arrays = {**common, "train_num": ramp_tn, "query_num": ramp_qn,
                   "train_area_factor": af, "query_area_factor": qaf}
    area_info = _save_npz(out / "area_prepared.npz", area_arrays)
    del area_arrays, ramp_tn, ramp_qn

    oof, prior_state, folds, prior_evidence = prior.fit_oof_prior(Xfit, yfit, indices)
    query_prior, levels = prior.predict_prior(prior_state, Xq, return_diagnostics=True)
    write_json(out / "prior_state.json", prior_state)
    write_json(out / "prior_evidence.json", prior_evidence)
    reload_prior = prior.predict_prior(read_json(out / "prior_state.json"), Xq)
    need(query_prior.tobytes() == reload_prior.tobytes(), "Prior full-query reload differs")
    pf = np.asarray(area * oof / 1e6, dtype=np.float32)
    qpf = np.asarray(query_area * query_prior / 1e6, dtype=np.float32)
    w64, valid_dates, date_evidence = prior._weights(prior._dates(Xfit))
    weights = w64.astype(np.float32)
    need((weights[~valid_dates] == np.float32(1)).all(), "Invalid date weights must stay one")
    for a, rows, label in ((pf, len(y), "fit price factor"), (qpf, len(Xq), "query price factor"),
                           (weights, len(y), "fit date weight")):
        _positive(a, rows, label)
    for name, value in (("train_oof_unit_prior.npy", oof), ("query_unit_prior.npy", query_prior),
                        ("fold_ids.npy", folds), ("source_indices.npy", indices),
                        ("date_weights_float32.npy", weights)):
        write_npy(out / name, value)
    date_evidence.update(saved_dtype="float32", saved_values_sha256=values_sha(weights),
                         renormalized_after_cast=False, invalid_rows_exact_one=True)
    write_json(out / "date_weights_metadata.json", date_evidence)
    prior_arrays = {**common, "train_num": tn, "query_num": qn,
                    "train_price_factor_million": pf, "query_price_factor_million": qpf,
                    "train_sample_weight": weights}
    prior_info = _save_npz(out / "prior_prepared.npz", prior_arrays)
    need(sources == source_hashes(), "NN source changed during preparation")
    need(all(input_sha[key] == frame_sha(frame) for key, frame in input_frames.items())
         and input_sha["fit_target"] == values_sha(np.asarray(yfit, dtype=np.float64))
         and input_sha["source_indices"] == values_sha(indices), "Preparation mutated caller inputs")
    paths = {"saved_date_weights": "date_weights_float32.npy", "area_npz": "area_prepared.npz",
             "prior_npz": "prior_prepared.npz", "encoder_state": "encoder_state.json",
             "ramp_state": "ramp_state.json", "prior_state": "prior_state.json",
             "prior_evidence": "prior_evidence.json", "train_oof_unit_prior": "train_oof_unit_prior.npy",
             "query_unit_prior": "query_unit_prior.npy", "fold_ids": "fold_ids.npy",
             "source_indices": "source_indices.npy", "date_weights_metadata": "date_weights_metadata.json"}
    result = {"format": FORMAT, "status": "prepared_no_NN_training", "fit_rows": len(y),
              "query_rows": len(Xq), "categorical_columns": cats, "cardinalities": card.tolist(),
              "area_numeric_width": ramp_state["output_width"], "prior_numeric_width": 144,
              "area_arrays": area_info, "prior_arrays": prior_info,
              "source_sha256": sources, "input_sha256": input_sha,
              "source_indices_contract": SOURCE_INDEX_CONTRACT,
              **{key: rel(out / name) for key, name in paths.items()},
              "prior_query_level_counts": levels, "chronological_oof": False,
              "prior_training_role": "five group-held-out folds; complement fit labels/dates only",
              "query_prior_role": "same-cutoff full-fit state; no query labels or query-date weights",
              "date_weights_role": "N_prior only; N_area retains original R54 uniform total-price loss",
              "encoder_full_fit_and_query_reload_exact": True, "prior_full_query_reload_exact": True,
              "NPZ_array_reload_exact": True, "query_labels_available": False,
              "new_NN_fits": 0, "elapsed_seconds": time.monotonic() - started,
              "completed_at_utc": now(), "prepared_metadata": rel(out / "prepared.json")}
    result["artifact_sha256"] = {rel(p): sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "prepared.json", result)
    return json.loads(json.dumps(result))


def _expected_config(kind):
    # Read the immutable literal config without importing Torch into this runtime.
    tree = ast.parse((ROOT / "src/round47_nn_train.py").read_text(encoding="utf-8"))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets)]
    need(len(assignments) == 1, "R47 literal configuration missing")
    config = ast.literal_eval(assignments[0].value)
    if kind == "N_area":
        config.update(prediction_head="Softplus unit price; unit * area_factor yields million TWD",
                      output_units="1e4 TWD per square metre of effective area A",
                      area_formula="finite positive building_area else finite positive parking_area else 1",
                      area_factor_formula="A / 100",
                      loss_formula="abs(unit_output * train_area_factor - train_y_million).mean()",
                      initial_bias="inverseSoftplus(median_fit(train_y_million / train_area_factor))",
                      extra_sample_weights=False, price_or_unit_clipping=False)
    elif kind == "N_prior":
        config.update(prediction_head="Softplus dimensionless ratio; ratio * price_factor_million yields million TWD",
                      output_units="dimensionless positive price ratio",
                      price_factor_formula="effective_area_A * group_prior_TWD_per_m2 / 1e6",
                      price_factor_uses_fit_labels=True,
                      train_price_factor_construction="group-fold OOF fit-label prior supplied by driver",
                      query_price_factor_construction="full-fit label prior supplied by driver; no query labels",
                      initial_bias="inverseSoftplus(median_fit(train_y_million / train_price_factor_million))",
                      price_or_ratio_clipping=False, loss="fit-recency weighted MAE in million TWD",
                      loss_formula="mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))",
                      extra_sample_weights=True,
                      sample_weight_normalization="driver-supplied fit-global normalization; no batch renormalization")
    else:
        raise ValueError("Unknown fixed NN component")
    return config


def _verify_training(kind, directory, prepared, npz_path, trainer):
    meta = read_json(directory / "training.json")
    need(meta["status"] == "completed" and meta["configuration"] == _expected_config(kind),
         "NN configuration/status differs")
    need(meta["source_sha256"] == sha(trainer)
         and meta["architecture_source_sha256"] == LOCKED_SOURCES["src/round47_nn_train.py"]
         and meta["input_sha256"] == sha(npz_path)
         and meta["input_sha256"] == prepared["artifact_sha256"][rel(npz_path)],
         "NN source or prepared input identity differs")
    width = prepared["area_numeric_width" if kind == "N_area" else "prior_numeric_width"]
    need(meta["fit_rows"] == prepared["fit_rows"] and meta["query_rows"] == prepared["query_rows"]
         and meta["numeric_count"] == width and meta["cardinalities"] == prepared["cardinalities"],
         "NN shape/cardinality mismatch")
    need(meta["query_labels_available"] is False and meta["query_labels_used"] is False
         and meta["saved_model_reload_exact"] is True and meta["reload_rows"] == prepared["query_rows"]
         and meta["prediction_units"] == "TWD", "Full-query reload/label boundary differs")
    need([e["epoch"] for e in meta["epochs"]] == list(range(1, 31)), "Fixed 30 epochs not completed")
    records = prepared["area_arrays" if kind == "N_area" else "prior_arrays"]
    factor_keys = ("train_area_factor", "query_area_factor") if kind == "N_area" else (
        "train_price_factor_million", "query_price_factor_million", "train_sample_weight")
    for key in factor_keys:
        need(meta[key + "_sha256"] == records[key]["values_sha256"], "NN factor/weight SHA differs")
    epoch_keys = ["training_mae_million_with_dropout"]
    if kind == "N_prior":
        epoch_keys.append("training_weighted_mae_million_with_dropout")
    need(all(np.isfinite(e[key]) for e in meta["epochs"] for key in epoch_keys),
         "NN completed epochs include nonfinite loss")
    need(meta["extra_sample_weights"] is (kind == "N_prior"), "Wrong sample weight objective")
    if kind == "N_prior":
        need(meta["parent_trainer_source_sha256"] == LOCKED_SOURCES["src/round52_nn_train.py"]
             and meta["sample_weights_batch_renormalized"] is False
             and meta["sample_weights_recomputed_in_trainer"] is False
             and meta["epoch_selection_uses_training_metrics"] is False, "Prior objective contract differs")
    predictions = np.load(directory / "predictions_twd.npy", allow_pickle=False)
    need(predictions.dtype == np.float64 and predictions.shape == (prepared["query_rows"],)
         and np.isfinite(predictions).all() and (predictions > 0).all(), "Invalid native TWD predictions")
    receipt = {"component": kind, "fit_seconds": meta["fit_seconds"],
               "trainer_elapsed_seconds": meta["elapsed_seconds"], "full_query_reload_exact": True,
               "reload_verified_by": "immutable trainer reload of saved model.pt over every supplied query row",
               "reload_rows": len(predictions), "prediction_values_sha256": values_sha(predictions),
               "artifact_sha256": {rel(directory / name): sha(directory / name)
                                   for name in ("run.json", "training.json", "model.pt", "predictions_twd.npy")}}
    return predictions, receipt


def train_nn(prepared, out):
    """Explicit GPU entry: serial seed-42 R54-area then R63-prior, no retries.

    Returns {N_area: float64 TWD array, N_prior: float64 TWD array, receipt: dict}.
    The caller owns the GPU scheduling and must authorize this function call.
    """
    started = time.monotonic()
    need(prepared.get("format") == FORMAT and prepared.get("status") == "prepared_no_NN_training",
         "Unexpected NN preparation contract")
    need(read_json(safe_path(prepared["prepared_metadata"])) == prepared,
         "Prepared metadata differs from its saved receipt")
    need(prepared["source_sha256"] == source_hashes(), "NN source changed since preparation")
    for name, digest in prepared["artifact_sha256"].items():
        need(sha(safe_path(name)) == digest, "Prepared artifact changed: " + name)
    interpreter = ROOT / ".venv_torch/Scripts/python.exe"
    need(interpreter.is_file(), "Isolated Torch runtime is unavailable")
    out = safe_path(out)
    out.mkdir(parents=True, exist_ok=True)
    need(not any(out.iterdir()), "NN training output must be empty; no retry/overwrite")
    write_json(out / "fit_started.json", {"started_at_utc": now(), "source_sha256": source_hashes(),
               "prepared_metadata": prepared["prepared_metadata"],
               "prepared_metadata_sha256": sha(safe_path(prepared["prepared_metadata"])),
               "planned_NN_fits": 2, "serial_execution": True, "automatic_retry": False})
    result = {}
    receipts = []
    for kind, trainer_name, npz_key in (
        ("N_area", "src/round48_nn_train.py", "area_npz"),
        ("N_prior", "src/round63_recency_weighted_prior_train.py", "prior_npz"),
    ):
        trainer = ROOT / trainer_name
        npz_path = safe_path(prepared[npz_key])
        directory = out / kind
        directory.mkdir()
        command = [str(interpreter), "-u", "-B", "-X", "utf8", str(trainer),
                   "--data", str(npz_path), "--out", str(directory)]
        step = time.monotonic()
        try:
            with (directory / "stdout.log").open("x", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            need(process.returncode == 0, kind + " trainer failed; inspect stdout.log; no automatic retry")
            need(prepared["source_sha256"] == source_hashes(), "Source changed during NN training")
            predictions, receipt = _verify_training(kind, directory, prepared, npz_path, trainer)
        except Exception as exc:
            write_json(directory / "execution.json", {"status": "failed", "failure": repr(exc),
                       "elapsed_seconds": time.monotonic() - step, "automatic_retry": False})
            raise
        execution = {"status": "completed", "exit_code": process.returncode,
                     "elapsed_seconds": time.monotonic() - step, "automatic_retry": False}
        write_json(directory / "execution.json", execution)
        receipt["outer_elapsed_seconds"] = execution["elapsed_seconds"]
        receipts.append(receipt)
        result[kind] = predictions
    for name, digest in prepared["artifact_sha256"].items():
        need(sha(safe_path(name)) == digest, "Prepared artifact changed during training: " + name)
    receipt = {"format": FORMAT, "status": "two_native_NN_fits_completed", "new_NN_fits": 2,
               "components": receipts, "serial_execution": True, "query_labels_available": False,
               "full_query_saved_model_reload_exact": True, "source_sha256": source_hashes(),
               "prepared_metadata_sha256": sha(safe_path(prepared["prepared_metadata"])),
               "elapsed_seconds": time.monotonic() - started, "completed_at_utc": now(),
               "bitwise_retraining_determinism_claimed": False}
    write_json(out / "receipt.json", receipt)
    result["receipt"] = receipt
    return result


def self_check():
    """Small synthetic preparation check; no dataset read, trainer or GPU call."""
    import tempfile

    started = time.monotonic()
    n = 360
    cats = ["project_name", "address_city", "address_district", "address_raw",
            "address_road", "address_section", "transaction_type", "building_type"]
    cats += ["cat" + str(i) for i in range(13)]
    nums = ["building_area", "parking_area", "trade_day_index",
            "trade_date_invalid", "trade_date_out_of_range"] + ["num" + str(i) for i in range(61)]
    x = pd.DataFrame({c: np.repeat("c", n).astype(object) for c in cats})
    x["project_name"] = ["project" + str(i // 6) for i in range(n)]
    for i, c in enumerate(nums):
        x[c] = np.arange(n, dtype=np.float32) % 19 + i
    x["building_area"] = np.full(n, 100, dtype=np.float32)
    x["parking_area"] = np.zeros(n, dtype=np.float32)
    x["trade_day_index"] = np.arange(n, dtype=np.float32) + 1000
    x["trade_date_invalid"] = np.zeros(n, dtype=np.float32)
    x["trade_date_out_of_range"] = np.zeros(n, dtype=np.float32)
    x.loc[0, "trade_day_index"] = np.nan
    x.loc[0, "trade_date_invalid"] = 1
    q = x.iloc[:7].copy().reset_index(drop=True)
    q["project_name"] = "unknown"
    q["cat0"] = "unseen"
    g = pd.DataFrame({c: np.full(n, np.nan, dtype=np.float32) for c in geo.NUMERIC})
    g["geo_has_position"] = np.zeros(n, dtype=np.float32)
    g["geo_match_status"] = "ineligible_address"
    g["geo_grid_002"] = "__MISSING__"
    g = g[geo.FEATURES]
    gq = g.iloc[:7].copy().reset_index(drop=True)
    y = pd.Series(np.arange(n, dtype=np.float64) * 1000 + 1e7, index=x.index)
    before = x.copy(deep=True)
    workspace_temp_root = ROOT / "outputs"
    workspace_temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="round90_nn_synthetic_", dir=workspace_temp_root) as name:
        temporary = Path(name).resolve()
        need(temporary.is_relative_to(workspace_temp_root.resolve()), "Unsafe temporary directory")
        indices = np.arange(1000, 1000 + 2 * n, 2, dtype=np.int64)
        original_indices = indices.copy()
        prepared = prepare_nn(x, y, g, q, gq, indices, temporary / "prep")
        checks = {"caller_feature_values_unchanged": x.equals(before),
                  "no_native_fit_or_model_artifact": prepared["new_NN_fits"] == 0
                  and not list(temporary.rglob("model.pt")),
                  "saved_state_and_NPZ_reloads_exact": prepared["encoder_full_fit_and_query_reload_exact"]
                  and prepared["prior_full_query_reload_exact"] and prepared["NPZ_array_reload_exact"]}
        with np.load(safe_path(prepared["area_npz"]), allow_pickle=False) as area_npz, \
                np.load(safe_path(prepared["prior_npz"]), allow_pickle=False) as prior_npz:
            checks["area_uniform_prior_dateweighted_roles"] = (
                "train_sample_weight" not in area_npz.files and "train_sample_weight" in prior_npz.files
                and _expected_config("N_area")["extra_sample_weights"] is False
                and _expected_config("N_prior")["extra_sample_weights"] is True)
            checks["ramp_retains_prior_numeric_prefix"] = (
                area_npz["train_num"][:, :144].tobytes() == prior_npz["train_num"].tobytes())
            checks["invalid_date_float32_weight_exact_one"] = (
                prior_npz["train_sample_weight"].dtype == np.float32
                and prior_npz["train_sample_weight"][0] == 1)
            checks["query_unknown_category_maps_zero"] = bool(
                (area_npz["query_cat"][:, cats.index("cat0")] == 0).all())
            oof = np.load(safe_path(prepared["train_oof_unit_prior"]), allow_pickle=False)
            checks["fit_OOF_factor_exact_float32_formula"] = (
                prior_npz["train_price_factor_million"].tobytes()
                == np.asarray(effective_area(x) * oof / 1e6, dtype=np.float32).tobytes())
        checks["source_indices_unchanged_and_saved_in_order"] = (
            indices.tobytes() == original_indices.tobytes()
            == np.load(safe_path(prepared["source_indices"]), allow_pickle=False).tobytes())
        q_changed = q.copy(deep=True)
        q_changed["cat0"] = "different_unknown"
        q_changed["trade_day_index"] += 5000
        q_changed["building_area"] *= 3
        changed = prepare_nn(x, y, g, q_changed, gq, indices, temporary / "changed_query")
        checks["query_perturbation_cannot_change_fit_states_or_weights"] = all(
            sha(safe_path(prepared[key])) == sha(safe_path(changed[key]))
            for key in ("encoder_state", "ramp_state", "prior_state", "prior_evidence",
                        "saved_date_weights", "train_oof_unit_prior", "fold_ids"))
        folds = np.load(safe_path(prepared["fold_ids"]), allow_pickle=False)
        changed_y = y.copy()
        held = folds == 0
        need(held.any(), "Synthetic OOF held fold must not be empty")
        changed_y.loc[held] *= 7
        changed_oof, _, changed_folds, _ = prior.fit_oof_prior(x, changed_y, indices)
        checks["held_fold_labels_do_not_enter_own_training_prior"] = (
            changed_folds.tobytes() == folds.tobytes()
            and changed_oof[held].tobytes() == oof[held].tobytes())
        checks = {key: bool(value) for key, value in checks.items()}
        need(all(checks.values()), "Synthetic NN preparation check failed")
        return {"status": "synthetic_preparation_passed", "checks": checks, "fit_rows": n,
                "query_rows": len(q), "area_numeric_width": prepared["area_numeric_width"],
                "prior_numeric_width": 144, "elapsed_seconds": time.monotonic() - started,
                "first_preparation_elapsed_seconds": prepared["elapsed_seconds"],
                "source_sha256": source_hashes(), "real_data_read": False,
                "GPU_calls": 0, "native_NN_fits": 0,
                "trainer_execution_tested": False}
