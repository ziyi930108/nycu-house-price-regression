"""R97: fresh-cutoff v22/v24 reconstruction; query labels are unavailable here.

Seven serial native fits (five trees, two NNs), new fit-only encoder/prior/
reference/spatial states, complete saved-model/query reload, then seal predictions.
Scoring is a separate program invoked only after this process succeeds.
No test loader, old fitted model reuse, automatic retry or overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round97"
DOCUMENT = ROOT / "reports/round97_plan.md"
REQUEST = ROOT / "outputs/round97_launcher_20260929/research_request.json"
PREPARATION = ROOT / "outputs/round90/preparation"
PLAN_SHA = "8372960e832bd66203374c942e0f7988e66522aedf358c894d892e82669a3a08"
PREPARATION_SHA = "400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7"
LABEL_MANIFEST = ROOT / "outputs/round97_inputs/manifest.json"
LOCKED_HELPERS = {
    "src/round90_offset_data.py": "4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1",
    "src/round90_offset_trees.py": "666e9fc61d22565aae2f738767bfef471e12679d4e1c0bbe86cf366afa540110",
    "src/round90_offset_nn.py": "b473b246925dffe3d084c909679892f45e33f19ed0a6c18c100cfc836bc35e06",
    "src/round83_seed_unit.py": "c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b",
}
NATIVE_KEYS = ("raw", "light", "log", "unit42", "unit137", "N_area", "N_prior")
BRANCH_KEYS = ("N", "H", "U22", "U24")
FORMULAS = {
    "N": ".75*(.5*N_area+.5*N_prior)+.25*light",
    "H": ".75*(.5*raw+.5*log)+.25*light",
    "U22": ".75*unit42+.25*light",
    "U24": ".5*unit42+.5*unit137",
    "v22": "median(S(R(N)),S(R(H)),S(R(U22)))",
    "v24": "median(S(R(N)),S(R(H)),S(R(U24)))",
}

def need(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def safe(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    need(p.is_relative_to(ROOT), "Path outside workspace")
    return p

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    with path.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(path) + "\n")

def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def frame_sha(x):
    return hashlib.sha256(pd.util.hash_pandas_object(x, index=True).to_numpy(dtype="<u8").tobytes()).hexdigest()

def exact(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label)

def prices(a, n, name):
    a = np.asarray(a)
    need(a.dtype == np.float64 and a.shape == (n,) and np.isfinite(a).all()
         and (a >= 0).all(), "Invalid nonnegative float64 TWD vector: " + name)
    return a

def array_record(path, a):
    return {"path": rel(path), "sha256": sha(path), "values_sha256": values_sha(a),
            "dtype": str(a.dtype), "shape": list(a.shape)}

def save_array(path, a):
    a = np.asarray(a)
    with Path(path).open("xb") as f:
        np.save(f, a, allow_pickle=False)
    b = np.load(path, allow_pickle=False)
    exact(a, b, "Saved array full roundtrip: " + str(path))
    return array_record(path, a)

def save_pickle(path, value):
    with Path(path).open("xb") as f:
        pickle.dump(value, f, protocol=5)
    with Path(path).open("rb") as f:
        return pickle.load(f)

def state_sha(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()

def compose_bases(native):
    need(set(native) == set(NATIVE_KEYS), "Expected exactly seven fixed native vectors")
    n = len(native["raw"])
    for k in NATIVE_KEYS:
        prices(native[k], n, k)
    b = {
        "N": .75 * (.5 * native["N_area"] + .5 * native["N_prior"]) + .25 * native["light"],
        "H": .75 * (.5 * native["raw"] + .5 * native["log"]) + .25 * native["light"],
        "U22": .75 * native["unit42"] + .25 * native["light"],
        # R83 original U24 is the mean of native unit trees, with no LGB blend.
        "U24": .5 * native["unit42"] + .5 * native["unit137"],
    }
    for k in BRANCH_KEYS:
        prices(b[k], n, k)
    return b

def compose_final(spatial_vectors):
    need(set(spatial_vectors) == set(BRANCH_KEYS), "Exactly four spatial branches required")
    n = len(spatial_vectors["N"])
    for k, a in spatial_vectors.items():
        prices(a, n, k)
    return {
        "v22": np.median(np.column_stack([spatial_vectors[k] for k in ("N", "H", "U22")]), axis=1),
        "v24": np.median(np.column_stack([spatial_vectors[k] for k in ("N", "H", "U24")]), axis=1),
    }

def protected_hashes():
    paths = list((ROOT / "outputs").glob("submission_v*.csv"))
    paths += [ROOT / p for p in (
        "outputs/current_private_candidate.json", "outputs/v22_submission_focus_20260928/user_selection.json",
        "runs/versions.json", "deliverables/v22_github/REPORT.md")]
    for pattern in ("docs/*.pdf", "presentation/*.pptx"):
        paths += list((ROOT / "deliverables/v22_github").glob(pattern))
    need(all(p.is_file() for p in paths), "Missing protected evidence")
    return {rel(p): sha(p) for p in sorted(set(paths))}

def load_modules():
    # Verify immutable helpers before importing their fit APIs.
    for name, digest in LOCKED_HELPERS.items():
        need(sha(ROOT / name) == digest, "Changed immutable helper: " + name)
    import round90_offset_data as data
    import round90_offset_trees as trees
    import round90_offset_nn as neural
    import round83_seed_unit as seed
    import round2_neighborhood as reference
    import round8_spatial_reference as spatial
    return data, trees, neural, seed, reference, spatial

def source_hashes(modules, data_manifest):
    data, trees, neural, seed, reference, spatial = modules
    sources = {}
    for part in (data_manifest["source_sha256"], trees.source_hashes(),
                 neural.source_hashes(), seed.source_hashes(), LOCKED_HELPERS):
        for name, digest in part.items():
            need(name not in sources or sources[name] == digest, "Source authority conflict: " + name)
            need(sha(safe(name)) == digest, "Source authority SHA changed: " + name)
            sources[name] = digest
    for p in (__file__, reference.__file__, spatial.__file__, ROOT / "src/round97_prepare_fit_labels.py"):
        sources[rel(p)] = sha(p)
    return sources

def load_fit_labels(data, expected_sha):
    """Accept only a separately sealed fit-only export, never the raw pickle."""
    need(sha(LABEL_MANIFEST) == expected_sha, "Fit label manifest changed")
    m = read_json(LABEL_MANIFEST)
    need(m["status"] == "fit_labels_sealed_no_training" and m["format"] == "R97_fit_labels_v1", "Fit label export not complete")
    need(m["pure_X_manifest_sha256"] == PREPARATION_SHA, "Label export uses another preparation")
    need(m["fit_rows"] == len(data.fit_indices), "Fit label length mismatch")
    need(m["fit_canonical_values_sha256"] == values_sha(data.fit_indices), "Canonical fit label axis differs")
    need(m["fit_raw_cache_values_sha256"] == values_sha(data.fit_source_cache_indices),
         "Raw cache fit label axis differs")
    inherited = data.contract["inherited_provenance"]
    need(m["raw_train_path"] == inherited["raw_train_path"]
         and m["raw_train_sha256"] == inherited["raw_train_sha256"], "Fit label original source mismatch")
    need(m["source_sha256"] == sha(ROOT / "src/round97_prepare_fit_labels.py"), "Extractor source changed")
    need(m["dtype"] == "float64" and m["units"] == "TWD" and m["model_fits"] == 0
         and m["saved_labels_reload_exact"] is True and m["test_read"] is False
         and m["valid_read"] is False, "Fit-only export role differs")
    for name, digest in m["indices_sha256"].items():
        need(sha(safe(name)) == digest, "Fit-only export source index file changed")
    path = safe(m["fit_y_path"])
    need(path.parent == LABEL_MANIFEST.parent and path.name == "fit_y.npy", "Unexpected fit-only payload")
    need(sha(path) == m["fit_y_sha256"], "Fit-only prices file changed")
    y = np.load(path, allow_pickle=False)
    need(y.dtype == np.float64 and y.shape == (len(data.fit_indices),)
         and np.isfinite(y).all() and (y > 0).all(), "Invalid aligned fit-only prices")
    need(values_sha(y) == m["fit_y_values_sha256"], "Fit-only prices values changed")
    need(m["query_labels_exported"] is False and m["query_statistics_computed"] is False,
         "Extractor exported query information")
    return pd.Series(y, index=data.Xfit.index), m

def verify_nn_preparation(prepared, data, yfit, neural):
    need(prepared["status"] == "prepared_no_NN_training" and prepared["new_NN_fits"] == 0,
         "NN preparation unexpectedly trained models")
    need(prepared["source_indices_contract"] == neural.SOURCE_INDEX_CONTRACT
         and prepared["input_sha256"]["source_indices"] == values_sha(data.fit_indices),
         "NN OOF source axis must be canonical fit indices")
    need(prepared["input_sha256"]["fit_target"] == values_sha(yfit.to_numpy(dtype=np.float64)),
         "NN fit labels binding differs")
    for key, frame in (("fit_features", data.Xfit), ("query_features", data.Xquery),
                       ("fit_tree_geo", data.tree_fit), ("query_tree_geo", data.tree_query)):
        need(prepared["input_sha256"][key] == frame_sha(frame), "NN frame binding differs: " + key)
    for path, digest in prepared["artifact_sha256"].items():
        need(sha(safe(path)) == digest, "NN preparation artifact changed")
    w = np.load(safe(prepared["saved_date_weights"]), allow_pickle=False)
    need(w.dtype == np.float32 and w.shape == (len(yfit),) and np.isfinite(w).all() and (w > 0).all(),
         "Invalid saved fresh-cutoff date weights")
    with np.load(safe(prepared["prior_npz"]), allow_pickle=False) as z:
        exact(w, z["train_sample_weight"], "Tree/prior use identical saved float32 date weights")
        exact(np.asarray(yfit / 1e6, dtype=np.float32), z["train_y_million"], "Fit NN target binding")
    exact(np.load(safe(prepared["source_indices"]), allow_pickle=False), data.fit_indices,
          "Saved OOF indices differ from canonical fit axis")
    return w

def verify_unit_seed_pair(state137, state42, meta137, meta42):
    # R79 returns the original R19 state directly. Only R83 adds a wrapper.
    need(state137["legacy_feature_state"] == state42, "Unit feature state differs by seed")
    need(meta137["parameters"] == {**meta42["parameters"], "random_seed": 137}, "Seed-only difference violated")
    for k in ("fit_target_values_sha256", "fit_transformed_target_values_sha256", "fit_date_weights_sha256",
              "fit_sample_weights_sha256", "fit_original_feature_values_sha256",
              "fit_geo_feature_values_sha256", "fit_model_feature_values_sha256"):
        need(meta137[k] == meta42[k], "Unit fits have different data/weights: " + k)

def fit_seed137(folder, data, yfit, weights, seed, unit_metadata, seed42_state):
    from catboost import CatBoostRegressor
    folder.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    write_json(folder / "started.json", {"parameters": seed.parameters(), "fit_rows": len(yfit),
               "query_rows": len(data.Xquery), "query_labels_available": False, "automatic_retry": False})
    model, state, meta = seed.fit_model(data.Xfit, yfit, data.tree_fit, data.categorical_columns, weights)
    fit_seconds = time.monotonic() - tick
    model.save_model(str(folder / "model.cbm"), format="cbm")
    write_json(folder / "feature_state.json", state)
    write_json(folder / "fit_metadata.json", meta)
    # New-window seed42 and seed137 share identical feature state/target/weights.
    verify_unit_seed_pair(state, seed42_state, meta, unit_metadata)
    raw = seed.predict_raw(model, state, data.Xquery, data.tree_query, data.categorical_columns)
    pred = prices(seed.inverse_transform(raw, data.Xquery), len(data.Xquery), "unit137")
    saved_state = read_json(folder / "feature_state.json")
    restored = CatBoostRegressor()
    restored.load_model(str(folder / "model.cbm"), format="cbm")
    restored_raw = seed.predict_raw(restored, saved_state, data.Xquery, data.tree_query, data.categorical_columns)
    exact(raw, restored_raw, "Seed137 full query native reload")
    exact(pred, seed.inverse_transform(restored_raw, data.Xquery), "Seed137 full query TWD reload")
    save_array(folder / "raw_unit.npy", raw)
    rec = save_array(folder / "predictions_twd.npy", pred)
    summary = {"status": "completed", "new_native_fits": 1, "fit_seconds": fit_seconds,
               "elapsed_seconds": time.monotonic()-tick, "seed": 137, "parameters": seed.parameters(),
               "fit_rows": len(yfit), "query_rows": len(pred), "saved_model_reload_exact": True,
               "reload_rows": len(pred), "raw_reload_exact": True, "TWD_reload_exact": True,
               "seed42_same_target_weights_feature_state": True, "query_labels_available": False,
               "artifact_sha256": {rel(p): sha(p) for p in folder.iterdir() if p.is_file()}}
    write_json(folder / "metadata.json", summary)
    return pred, summary, rec

def build_and_apply_states(folder, data, yfit, bases, reference, spatial):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    ref = reference.fit_final_reference(data.Xfit, yfit)
    need(ref["fit_report"]["fitting_rows"] == len(yfit)
         and ref["fit_report"]["validation_targets_used_for_references"] is False,
         "Fresh reference fit contract differs")
    ref2 = save_pickle(folder / "reference.pkl", ref)
    write_json(folder / "reference_fit_report.json", ref["fit_report"])
    sp, summary = spatial.build(ref2, data.Xfit, data.spatial_fit)
    sp2 = save_pickle(folder / "spatial.pkl", sp)
    write_json(folder / "spatial_summary.json", summary)
    need(sp2["state_summary"] == summary and not summary["query_used_in_build"], "Spatial reload/fit role differs")
    before = {"ref": state_sha(ref), "ref2": state_sha(ref2), "sp": state_sha(sp), "sp2": state_sha(sp2)}
    vectors, outputs, reports = {}, {}, {}
    for key in BRANCH_KEYS:
        branch = folder / key
        branch.mkdir()
        B = prices(bases[key], len(data.Xquery), key + " base")
        P, D, rr = reference.predict_reference(ref, data.Xquery, B, variant="trend", return_diagnostics=True)
        P2, D2, _ = reference.predict_reference(ref2, data.Xquery, B, variant="trend", return_diagnostics=True)
        exact(P, P2, key + " reference full query state reload")
        pd.testing.assert_frame_equal(D, D2, check_exact=True)
        S, SD, sr = spatial.predict(sp, data.Xquery, data.spatial_query, B, P, D)
        S2, SD2, _ = spatial.predict(sp2, data.Xquery, data.spatial_query, B, P2, D2)
        exact(S, S2, key + " spatial full query state reload")
        pd.testing.assert_frame_equal(SD, SD2, check_exact=True)
        fallback = ~SD.applied.to_numpy(dtype=bool)
        exact(S[fallback], P[fallback], key + " spatial fallback")
        vectors[key] = {layer: save_array(branch / (layer + ".npy"), prices(a, len(B), key+layer))
                        for layer, a in (("base", B), ("reference", P), ("spatial", S))}
        for name, frame in (("reference_diagnostics", D), ("spatial_diagnostics", SD)):
            restored = save_pickle(branch / (name + ".pkl"), frame)
            pd.testing.assert_frame_equal(frame, restored, check_exact=True)
        outputs[key] = S
        reports[key] = {"reference": rr, "spatial": sr, "all_query_reload_exact": True,
                        "spatial_fallback_exact": True}
    after = {"ref": state_sha(ref), "ref2": state_sha(ref2), "sp": state_sha(sp), "sp2": state_sha(sp2)}
    need(before == after, "Reference/spatial predict mutated fit state")
    result = {"new_reference_fits": 1, "new_spatial_state_builds": 1, "query_labels_available": False,
              "reference_predict_calls": 8, "spatial_predict_calls": 8, "all_query_reload_exact": True,
              "fit_state_unchanged_during_all_predictions": True, "reports": reports,
              "elapsed_seconds": time.monotonic()-start}
    write_json(folder / "results.json", result)
    return vectors, outputs, result

def run(expected_plan_sha):
    need(sha(DOCUMENT) == expected_plan_sha == PLAN_SHA, "Fixed R97 plan SHA mismatch")
    request = read_json(REQUEST)
    need(request["approval_for_offset_rebuild"] is True and request["plan_sha256"] == expected_plan_sha,
         "Missing fixed offset rebuild authorization")
    need(request["prepared_manifest_sha256"] == PREPARATION_SHA, "Preparation identity differs")
    need(request["native_fits"] == {"trees": 5, "NN": 2} and request["formulas"] == FORMULAS,
         "Request changed fixed recipes/model count")
    need(sha(PREPARATION / "manifest.json") == PREPARATION_SHA, "Pure-X preparation changed")
    m = read_json(PREPARATION / "manifest.json")
    modules = load_modules()
    data_helper, trees, neural, seed, reference, spatial = modules
    sources = source_hashes(modules, m)
    frozen = request["frozen_source_and_plan"]
    for name, digest in {**sources, rel(DOCUMENT): expected_plan_sha}.items():
        need(frozen.get(name) == digest and sha(safe(name)) == digest, "Launch source not frozen: " + name)
    protected = protected_hashes()
    need(not OUT.exists(), "R97 output already exists; preserve failure, no retry/overwrite")
    OUT.mkdir()
    start = now();tick = time.monotonic();failure = None;exit_code = 1
    write_json(OUT / "started.json", {"started_at_utc": start, "plan_sha256": expected_plan_sha,
               "request_sha256": sha(REQUEST), "source_sha256": sources, "protected_before": protected,
               "new_native_tree_fits_planned": 5, "new_NN_fits_planned": 2, "automatic_retry": False})
    try:
        data = data_helper.load_prepared(PREPARATION, PREPARATION_SHA)
        need(len(data.fit_indices) == 272117 and len(data.query_indices) == 107169, "Fixed cutoff counts differ")
        for name, digest in m["source_sha256"].items():
            need(sources[name] == digest, "Preparation source differs from launch closure")
        yfit, labelmeta = load_fit_labels(data, request["fit_label_manifest_sha256"])
        inputs = {rel(PREPARATION / "manifest.json"): PREPARATION_SHA,
                  rel(LABEL_MANIFEST): sha(LABEL_MANIFEST),
                  labelmeta["fit_y_path"]: labelmeta["fit_y_sha256"]}
        # Original pure-X provenance files may be rehashed, never deserialize raw prices.
        for name, digest in m["input_sha256"].items():
            need("test" not in name.lower() and "valid.pkl" not in name.lower(), "Unexpected data dependency")
            need(sha(safe(name)) == digest, "Prepared source changed: " + name)
            inputs[name] = digest
        declared_raw = labelmeta["raw_train_path"]
        need(declared_raw not in inputs, "Raw price pickle must not enter driver read set")
        inputs[declared_raw] = labelmeta["raw_train_sha256"]  # inherited declaration, never opened here
        arrays = {}
        for name in data_helper.ARRAY_NAMES:
            arrays[name] = save_array(OUT / (name + ".npy"), getattr(data, name))
        frames = {}
        for name in data_helper.FRAME_NAMES:
            r = m["artifacts"][name]
            frames[name] = {**r, "frame_sha256": frame_sha(getattr(data, name))}
            inputs[r["path"]] = r["sha256"]
        identity = {k: frame_sha(getattr(data, k)) for k in data_helper.FRAME_NAMES}
        identity["fit_y"] = values_sha(yfit.to_numpy(dtype=np.float64))
        # Fail pure-X log-date contracts before spending on any native model.
        trees.log.fit_weights(data.Xfit)
        prepared = neural.prepare_nn(data.Xfit, yfit, data.tree_fit, data.Xquery,
                                     data.tree_query, data.fit_indices, OUT / "nn_prepared")
        weights = verify_nn_preparation(prepared, data, yfit, neural)
        rawtrees, tree_metadata = trees.fit_trees(data.Xfit, yfit, data.tree_fit, data.Xquery,
            data.tree_query, data.categorical_columns, weights, OUT / "trees")
        unitmeta = read_json(OUT / "trees/unit/fit_metadata.json")
        unitstate = read_json(OUT / "trees/unit/feature_state.json")
        u137, seedmeta, _ = fit_seed137(OUT / "trees/unit137", data, yfit, weights, seed, unitmeta, unitstate)
        nn = neural.train_nn(prepared, OUT / "nn_models")
        need(nn["receipt"]["new_NN_fits"] == 2 and nn["receipt"]["full_query_saved_model_reload_exact"],
             "Two NN completion/reload contract differs")
        native = {"raw": rawtrees["raw"], "light": rawtrees["light"], "log": rawtrees["log"],
                  "unit42": rawtrees["unit"], "unit137": u137,
                  "N_area": nn["N_area"], "N_prior": nn["N_prior"]}
        native_dir = OUT / "native_predictions";native_dir.mkdir()
        native_records = {k: save_array(native_dir / (k + ".npy"), prices(v, len(data.Xquery), k))
                          for k, v in native.items()}
        bases = compose_bases(native)
        branches, spatial_vectors, state_report = build_and_apply_states(
            OUT / "reference_spatial", data, yfit, bases, reference, spatial)
        final = compose_final(spatial_vectors)
        predictions = {k: save_array(OUT / (k + "_predictions.npy"), prices(v, len(data.Xquery), k))
                       for k, v in final.items()}
        after_identity = {k: frame_sha(getattr(data, k)) for k in data_helper.FRAME_NAMES}
        after_identity["fit_y"] = values_sha(yfit.to_numpy(dtype=np.float64))
        need(identity == after_identity, "Caller fit/query inputs were mutated")
        need(sources == source_hashes(modules, m), "Source changed during reconstruction")
        need(all(sha(safe(k)) == v for k, v in inputs.items() if k != declared_raw),
             "Driver-read inputs changed during reconstruction")
        after = protected_hashes()
        need(after == protected, "Protected submissions or formal artifacts changed")
        need(sha(DOCUMENT) == expected_plan_sha, "Plan changed")
        tree_fit_seconds = sum(float(v["fit_seconds"]) for v in tree_metadata.values()) + seedmeta["fit_seconds"]
        NN_fit_seconds = sum(float(v["fit_seconds"]) for v in nn["receipt"]["components"])
        seal = {
            "round": 97, "schema_version": 1, "status": "sealed_predictions_without_query_labels",
            "cutoff": "2023-02-11", "query_end": "2024-02-10",
            "all_fits_complete": True, "predictions_sealed": True, "sealed_at_utc": now(),
            "fit_before": "2023-02-11", "query_start": "2023-02-11", "query_end_inclusive": "2024-02-10",
            "fit_rows": len(data.Xfit), "query_rows": len(data.Xquery),
            "new_native_tree_fits": 5, "new_NN_fits": 2, "new_native_fits": 7,
            "new_reference_fits": 1, "new_spatial_state_builds": 1,
            "query_labels_available": False, "query_labels_used_for_training_or_prediction": False,
            "raw_price_table_deserialized_by_driver": False,
            "isolated_extractor_deserialized_full_raw_pickle": True,
            "query_price_values_exported_from_extractor": False, "test_read": False,
            "preparation_manifest": {"path": rel(PREPARATION / "manifest.json"), "sha256": PREPARATION_SHA},
            "fit_label_manifest": {"path": rel(LABEL_MANIFEST), "sha256": sha(LABEL_MANIFEST)},
            "raw_train_price_source_for_later_scorer": data.contract["inherited_provenance"],
            "formulas": FORMULAS, "arrays": arrays, "frames": frames, "branches": branches,
            "native_predictions": native_records, "predictions": predictions,
            "NN_preparation": prepared, "NN_receipt": nn["receipt"],
            "tree_receipts": tree_metadata, "seed137_receipt": seedmeta,
            "reference_spatial_receipt": state_report,
            "fit_seconds": {"trees": tree_fit_seconds, "NN": NN_fit_seconds,
                            "native_total": tree_fit_seconds + NN_fit_seconds},
            "new_query_metrics_computed": False, "scores": None,
            "selected_submission": "v22", "Private_score": None, "new_CSV": False,
            "source_sha256": sources, "input_sha256": inputs,
            "declared_only_input_paths_not_opened_by_driver": [declared_raw],
            "evaluation_source_frozen_by_separate_launcher": True,
            "plan_sha256": expected_plan_sha, "request_sha256": sha(REQUEST),
            "protected_before": protected, "protected_after": after,
            "input_frames_unchanged": True, "all_query_saved_model_and_state_reload_exact": True,
            "serial_native_training": True, "automatic_retry": False,
            "fresh_blind_validation_claimed": False,
            "limitations": ["New states remove cutoff leakage, but overlapping reused data make this no new blind test.",
                            "Saved-model reload is not deterministic retraining or clean-environment end-to-end proof."],
            "elapsed_seconds_before_seal": time.monotonic()-tick,
            "artifact_sha256": {rel(p): sha(p) for p in OUT.rglob("*") if p.is_file()},
        }
        write_json(OUT / "sealed_manifest.json", seal)
        need(read_json(OUT / "sealed_manifest.json") == seal, "Sealed metadata reload differs")
        exit_code = 0
        print(json.dumps({"status": seal["status"], "new_native_fits": 7,
              "sealed_manifest_sha256": sha(OUT / "sealed_manifest.json"), "query_metrics_computed": False}), flush=True)
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        write_json(OUT / "execution.json", {"exit_code": exit_code, "failure": failure,
            "started_at_utc": start, "completed_at_utc": now(),
            "elapsed_seconds": time.monotonic()-tick, "automatic_retry": False,
            "sealed_manifest_created": (OUT / "sealed_manifest.json").exists(),
            "query_labels_available": False, "test_read": False})

def self_check():
    """Synthetic only: recipes, type/shape refusal, serialization and write guards."""
    import tempfile
    checks = []
    def mark(name, ok):
        need(ok, name);checks.append(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, FileExistsError):
            checks.append(name);return
        raise AssertionError("Expected refusal: " + name)
    native = {k: np.array([10., 20., 30.]) + i * 10 for i, k in enumerate(NATIVE_KEYS)}
    before = {k: v.tobytes() for k, v in native.items()}
    b = compose_bases(native)
    exact(b["N"], .75*(.5*native["N_area"]+.5*native["N_prior"])+.25*native["light"], "N formula")
    mark("N_uses_both_fresh_NNs_with_original_LGB_share", True)
    exact(b["H"], .75*(.5*native["raw"]+.5*native["log"])+.25*native["light"], "H formula")
    mark("H_uses_fresh_raw_and_log_with_original_LGB_share", True)
    exact(b["U22"], .75*native["unit42"]+.25*native["light"], "U22 formula")
    mark("v22_unit_includes_original_LGB_share", True)
    exact(b["U24"], .5*native["unit42"]+.5*native["unit137"], "U24 formula")
    mark("v24_unit_mean_has_no_LGB_share", not np.array_equal(b["U24"], .75*b["U24"]+.25*native["light"]))
    f = compose_final(b)
    exact(f["v22"], np.median(np.column_stack([b["N"], b["H"], b["U22"]]), axis=1), "v22 median")
    exact(f["v24"], np.median(np.column_stack([b["N"], b["H"], b["U24"]]), axis=1), "v24 median")
    mark("two_complete_medians_keep_separate_unit_branches", True)
    mark("native_inputs_not_mutated", before == {k: v.tobytes() for k, v in native.items()})
    rejects("missing_native_component_rejected", lambda: compose_bases({k:v for k,v in native.items() if k!="raw"}))
    rejects("non_float64_native_rejected", lambda: compose_bases({**native, "raw": native["raw"].astype(np.float32)}))
    rejects("negative_native_rejected", lambda: compose_bases({**native, "raw": np.array([-1., 0., 1.])}))
    rejects("nonfinite_native_rejected", lambda: compose_bases({**native, "raw": np.array([1., np.nan, 2.])}))
    direct = {"state_content_sha256": "synthetic-direct-R19-state", "all_feature_names": ["a"]}
    wrapper = {"legacy_feature_state": direct}
    metadata42 = {k: "same" for k in ("fit_target_values_sha256", "fit_transformed_target_values_sha256",
        "fit_date_weights_sha256", "fit_sample_weights_sha256", "fit_original_feature_values_sha256",
        "fit_geo_feature_values_sha256", "fit_model_feature_values_sha256")}
    metadata42["parameters"] = {"random_seed": 42, "iterations": 2999}
    metadata137 = {**metadata42, "parameters": {"random_seed": 137, "iterations": 2999}}
    verify_unit_seed_pair(wrapper, direct, metadata137, metadata42)
    mark("actual_seed_pair_checker_accepts_direct_R19_seed42_state", True)
    rejects("seed42_wrapper_is_not_the_actual_R79_contract", lambda: verify_unit_seed_pair(
        wrapper, {"legacy_feature_state": direct}, metadata137, metadata42))
    rejects("seed_pair_different_weights_rejected", lambda: verify_unit_seed_pair(
        wrapper, direct, {**metadata137, "fit_sample_weights_sha256": "different"}, metadata42))
    # Temporary only, inside workspace; no datasets/models/imported fit APIs.
    with tempfile.TemporaryDirectory(prefix="r97_driver_synthetic_", dir=ROOT / "outputs") as name:
        d=Path(name);p=d/"vector.npy"
        record=save_array(p, native["raw"])
        mark("saved_vector_full_reload_identity", record["values_sha256"] == values_sha(native["raw"]))
        rejects("existing_vector_refuses_overwrite", lambda: save_array(p, native["raw"]))
        x={"v": np.array([np.nan, 1.]), "summary": {"query_used":False}}
        x2=save_pickle(d/"state.pkl",x)
        exact(x["v"],x2["v"],"Synthetic state nan bytes")
        mark("pickle_state_reload_retains_nan_values", x2["summary"] == x["summary"])
        write_json(d/"plain.json",{"a":1})
        mark("new_json_sidecar_binds_actual_bytes", (d/"plain.sha256").read_text().strip()==sha(d/"plain.json"))
        rejects("existing_json_refuses_overwrite", lambda: write_json(d/"plain.json",{"a":2}))
    return {"status":"passed","checks":checks,"real_data_read":False,"query_prices_read":False,
            "new_native_fits":0,"GPU_calls":0,"source_sha256":sha(__file__)}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--self-check",action="store_true")
    parser.add_argument("--plan-sha")
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(),ensure_ascii=False,indent=2))
    else:
        need(bool(args.plan_sha),"--plan-sha required for --run")
        run(args.plan_sha)
