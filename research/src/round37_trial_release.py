"""R37: one final R12 log fit plus unchanged saved R8 spatial state.

Separate trial output only. Freeze then run once; no score, model selection,
old authorization, reference refit, spatial rebuild, registry edit or upload.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import pickle
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import round12_log_geo as component
import round34_trial_release as prior

ROOT = prior.ROOT
OUT = ROOT / "outputs/round37"
DEST = OUT / "trial"
CSV = ROOT / "outputs/submission_v5_trial.csv"
DOCUMENT = ROOT / "reports/round37_trial_plan.md"
DOCUMENT_SHA = "eaf926b2112adf4ced1e7613b3f904f7179ddf5e4abc74b4e19e153c00a62759"
R12_PLAN = "outputs/round12/candidate_plan.json"
R12_SHA = "5b37d819e1fffb624b26740815fae10fb6597d5da84613026a6436132df3e25b"
R34_PLAN = "outputs/round34/trial_plan.json"
R34_PLAN_SHA = "aca3e5cb5f606a8cd2603efd4f56634cd8ef388fe760614f3b0b745f32b2add9"
R34_PROOF = "outputs/round34/trial/trial_release.json"
R34_PROOF_SHA = "67069dd953999d56c6e1e1fab70a5d7e0a0f98272a1bcde48a3bb538c5b64033"
R35_RESULT_SHA = "bf98f3ee111e7c89784d4338cc7545ecf8015728ecbc25f1bf085c4c6d695eb7"
R36_RESULT_SHA = "133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e"
SPATIAL_FOLDER = "outputs/round34/trial/"
V4_CSV = "outputs/submission_v4_trial.csv"
FORMULA = "B_log=.75*(.5*old_raw+.5*log_geo_TWD)+.25*old_unit; p_log,diag=reference_trend_once(B_log); R8.predict(saved_state,Xq,geoq,B_log,p_log,diag)"
need, sha, rel, read, locked = prior.need, prior.sha, prior.rel, prior.read, prior.locked
write_json, merge, verify, stamp = prior.write_json, prior.merge, prior.verify, prior.stamp
save_array, rounded, spatial = prior.save_array, prior.rounded, prior.spatial
PROTECTED = tuple(dict.fromkeys(prior.PROTECTED + (
    V4_CSV, R12_PLAN, "outputs/round12/frozen_candidate.json", "outputs/round12/accepted_candidate.json",
    R34_PLAN, R34_PROOF, "outputs/round35/reproduction_plan.json", "outputs/round35/results.json",
    "outputs/round36/combination_plan.json", "outputs/round36/results.json",
)))


def sources():
    pending, found = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                local = ROOT / "src" / (name.split(".")[0] + ".py")
                if local.is_file() and local not in found:
                    pending.append(local)
    return {rel(p): sha(p) for p in sorted(found)}


def runtime():
    return {**prior.runtime(), **{n: importlib.metadata.version(n) for n in ("catboost", "lightgbm")}}


def capture(mapping, name, expected=None):
    path = ROOT / name
    digest = sha(path)
    need(expected is None or digest == expected, "Authority differs: " + name)
    value = locked(path)
    mapping.update(merge(mapping, {name: digest, rel(path.with_suffix(".sha256")): sha(path.with_suffix(".sha256"))}))
    return value


def freeze():
    need(not (OUT / "trial_plan.json").exists() and not DEST.exists() and not CSV.exists(), "Trial already exists")
    need(sha(DOCUMENT) == DOCUMENT_SHA, "Fixed R37 document changed")
    evidence = {rel(DOCUMENT): DOCUMENT_SHA}
    snapshot = capture(evidence, "runs/v3/snapshot_manifest.json", prior.V3_SNAPSHOT)
    known = merge({n: v["original_sha256"] for n, v in snapshot["copied_files"].items()},
                  {v["original_path"]: v["sha256"] for v in snapshot["external_dependencies"]})
    original = capture(evidence, R12_PLAN, R12_SHA)
    r6 = capture(evidence, "outputs/round6/candidate_plan.json", original["old_plan_sha256"])
    need(known["outputs/round6/candidate_plan.json"] == original["old_plan_sha256"]
         and original["v3_snapshot_sha256"] == prior.V3_SNAPSHOT, "R12/R6 released lineage differs")
    need(original["parameters"] == component.parameters() and original["feature_rules"] == component.FEATURE_RULES
         and all(runtime()[n] == v for n, v in original["runtime"].items()), "Original log method/runtime differs")
    prior_plan = capture(evidence, R34_PLAN, R34_PLAN_SHA)
    proof = capture(evidence, R34_PROOF, R34_PROOF_SHA)
    execution = capture(evidence, "outputs/round34/execution.json")
    need(prior_plan["round"] == proof["round"] == 34 and proof["status"] == "experimental_trial_generated"
         and proof["plan_sha256"] == R34_PLAN_SHA and proof["path"] == V4_CSV
         and proof["source_sha256"] == prior_plan["source_sha256"]
         and proof["input_sha256"] == prior_plan["input_sha256"]
         and proof["saved_state_replay_bitwise_equal"] is True
         and proof["saved_state_reload_rows"] == 47073 and proof["v3_CSV_reproduced_exactly"] is True
         and execution["status"] == "completed" and execution["trial_release_sha256"] == R34_PROOF_SHA
         and execution["submission_sha256"] == proof["sha256"], "R34 final trial evidence incomplete")
    need(prior_plan["parameters"] == spatial.parameters() and prior_plan["runtime"] == prior.runtime(), "Saved spatial method/runtime differs")
    for number, result_sha, plan_name, expected_status, stages in (
        (35, R35_RESULT_SHA, "reproduction_plan.json", "both_periods_completed_no_adoption_decision", ["confirmation", "auxiliary"]),
        (36, R36_RESULT_SHA, "combination_plan.json", "three_periods_completed_no_adoption_decision", ["development", "confirmation", "auxiliary"]),
    ):
        folder = f"outputs/round{number}/"
        result = capture(evidence, folder + "results.json", result_sha)
        specification = capture(evidence, folder + plan_name, result["plan_sha256"])
        need(result["round"] == number and result["status"] == expected_status and result["checked_stages"] == stages
             and result["source_sha256"] == specification["source_sha256"]
             and result.get("formal_adoption_decision", result.get("new_formal_acceptance_decision")) is None,
             "Required reproduction/combination is incomplete")
        need(all(runtime()[n] == v for n, v in specification["runtime"].items()), "Research runtime differs")
        if number == 35:
            need(specification["original_R12_plan_sha256"] == R12_SHA
                 and specification["parameters"] == component.parameters(), "R35 log method differs")
        else:
            need(specification["formula"] == FORMULA and specification["R35_results_sha256"] == R35_RESULT_SHA
                 and specification["log_parameters"] == component.parameters()
                 and specification["spatial_parameters"] == spatial.parameters(), "R36 fixed combination differs")
        for stage in stages:
            item = capture(evidence, folder + stage + "/results.json", result["stage_result_sha256"][stage])
            need(item["stage"] == stage and item["status"] == "completed"
                 and item["plan_sha256"] == result["plan_sha256"], "Completed period authority differs")
    current_sources = sources()
    inherited_sources = merge(original["source_sha256"], prior_plan["source_sha256"])
    for name, digest in current_sources.items():
        if name != rel(Path(__file__)):
            need(inherited_sources.get(name) == digest, "Original dependency source differs: " + name)
    expected = merge(prior_plan["input_sha256"], evidence)
    for split in ("train", "valid"):
        for extension in ("pkl", "json"):
            name = f"data_cache/{split}.{extension}"
            digest = original["final_dependency_sha256" if extension == "pkl" else "dependency_sha256"][name]
            need(digest == r6["data_sha256"][name] == known[name], "Label cache lineage differs")
            expected = merge(expected, {name: digest})
    config = "runs/v1/models/final_config.json"
    expected = merge(expected, {config: original["dependency_sha256"][config], V4_CSV: proof["sha256"]})
    for name in ("spatial_state.pkl", "state_summary.json", "state_summary.sha256", "predictions.npy", "fit_indices.npy", "query_indices.npy"):
        path = SPATIAL_FOLDER + name
        expected = merge(expected, {path: proof["artifact_sha256"][path]})
    need(expected[rel(prior.SAMPLE)] == original["dependency_sha256"][rel(prior.SAMPLE)], "Sample identity differs")
    protected = {n: sha(ROOT / n) for n in PROTECTED}
    value = {"round": 37, "status": "trial_plan_frozen", "created_at_utc": stamp(),
             "trial_only": True, "formal_acceptance": False, "recommended_version": "v3", "registry_update_allowed": False,
             "submission_path": rel(CSV), "fit_rows": 467221, "query_rows": 47073, "geography_year": 2024,
             "parameters": component.parameters(), "feature_rules": component.FEATURE_RULES,
             "target_formula": component.TARGET_FORMULA, "inverse_formula": component.INVERSE_FORMULA,
             "spatial_parameters": spatial.parameters(), "formula": FORMULA, "runtime": runtime(),
             "source_sha256": current_sources, "input_sha256": expected, "protected_sha256": protected,
             "R12_plan_sha256": R12_SHA, "R34_release_sha256": R34_PROOF_SHA,
             "R35_results_sha256": R35_RESULT_SHA, "R36_results_sha256": R36_RESULT_SHA,
             "new_tree_models_planned": 1, "spatial_state_rebuilt": False, "reference_refitted": False,
             "labels_loaded": "local train.pkl and valid.pkl only, using saved canonical indices",
             "test_labels_loaded": False, "test_score_computed": False,
             "trial_public_mae": None, "trial_private_mae": None, "old_decisions_revised": False}
    need(current_sources == sources() and sha(DOCUMENT) == DOCUMENT_SHA, "Source changed during freeze")
    verify(evidence)
    write_json(OUT / "trial_plan.json", value)
    print(json.dumps({"status": value["status"], "plan_sha256": sha(OUT / "trial_plan.json")}), flush=True)


def raw_predict(model, state, X, geo, cats):
    component.check_state(state, X, cats)
    augmented = component.base.augment_features(X, geo)
    categories = component.base.categories(X, cats)
    need(int(model.tree_count_) == 2999 and list(model.feature_names_) == list(augmented.columns) == state["all_feature_names"], "Native model schema/trees differ")
    need(sorted(model.get_cat_feature_indices()) == sorted(augmented.columns.get_loc(c) for c in categories), "Native category positions differ")
    values = np.asarray(model.predict(Pool(augmented, cat_features=categories, thread_count=4), thread_count=4), dtype="float64")
    need(values.shape == (47073,) and np.isfinite(values).all(), "Invalid raw model output")
    return values


def adjustment(values, baseline, prices, baseline_prices):
    delta = values - baseline
    absolute = np.abs(delta)
    return {"changed_unrounded_rows": int(np.count_nonzero(delta)),
            "changed_rounded_rows": int(np.count_nonzero(prices != baseline_prices)),
            "minimum_TWD": float(delta.min()), "maximum_TWD": float(delta.max()),
            "mean_signed_TWD": float(delta.mean()), "median_signed_TWD": float(np.median(delta)),
            "mean_absolute_TWD": float(absolute.mean()), "p95_absolute_TWD": float(np.quantile(absolute, .95)),
            "is_prediction_error_metric": False}


def run():
    plan = locked(OUT / "trial_plan.json")
    need(plan["round"] == 37 and plan["trial_only"] is True and plan["formal_acceptance"] is False
         and plan["registry_update_allowed"] is False and plan["submission_path"] == rel(CSV), "Wrong trial identity")
    need(plan["source_sha256"] == sources() and plan["runtime"] == runtime()
         and plan["parameters"] == component.parameters() and plan["feature_rules"] == component.FEATURE_RULES
         and plan["spatial_parameters"] == spatial.parameters() and plan["formula"] == FORMULA
         and sha(DOCUMENT) == DOCUMENT_SHA, "Frozen source/runtime/method changed")
    need(not DEST.exists() and not CSV.exists(), "Refusing existing output")
    write_json(OUT / "run_claim.json", {"plan_sha256": sha(OUT / "trial_plan.json"), "started_at_utc": stamp(), "automatic_retry": False})
    started = time.monotonic()
    phase = "verify_inputs_before_deserialization"
    staging = OUT / (".trial-staging-" + uuid.uuid4().hex)
    progress = {"fits_started": 0, "fits_completed": 0, "models_saved": 0, "full_query_reload_verified": False}
    try:
        verify(merge(plan["input_sha256"], plan["source_sha256"], plan["protected_sha256"]))
        staging.mkdir(exist_ok=False)
        model_folder = staging / "model"
        model_folder.mkdir()
        phase = "final_original_X_and_label_alignment"
        schema = read(ROOT / "reports/feature_schema.json")
        frames = {s: pd.read_pickle(ROOT / f"data_cache/{s}_features.pkl") for s in ("train", "valid", "test")}
        for split, count in (("train", 455784), ("valid", 11994), ("test", 47073)):
            frame = frames[split]
            need(len(frame) == count and list(frame.columns) == schema["columns"] and len(frame.columns) == 87
                 and frame.index.equals(pd.RangeIndex(count)) and prior.TARGET not in frame, "Original feature schema/order differs")
        kt = prior.index(prior.GEO + "train_source_cache_indices.npy", 455784, 455255)
        kv = prior.index(prior.GEO + "valid_source_cache_indices.npy", 11994, 11994, True)
        kf = prior.index(prior.GEO + "full_source_cache_indices.npy", 467249, 467221)
        kq = prior.index(prior.GEO + "test_source_cache_indices.npy", 47073, 47073, True)
        X = pd.concat([frames["train"].iloc[kt], frames["valid"].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
        Xq = frames["test"].iloc[kq].reset_index(drop=True)
        need(len(X) == 467221 and X.dtypes.equals(Xq.dtypes), "Final feature alignment/dtypes differ")
        labels = {}
        for split, count in (("train", 455784), ("valid", 11994)):
            raw = pd.read_pickle(ROOT / f"data_cache/{split}.pkl")
            raw_meta = read(ROOT / f"data_cache/{split}.json")
            need(len(raw) == count and raw.index.equals(pd.RangeIndex(count))
                 and list(raw.columns) == raw_meta["columns"]
                 and raw_meta["source_sha256"] == schema["splits"][split]["input_source_sha256"], "Label/feature cache alignment differs")
            labels[split] = raw[prior.TARGET].to_numpy(dtype="float64", copy=True)
            need(np.isfinite(labels[split]).all() and (labels[split] > 0).all(), "Invalid local fit labels")
            del raw
        y = np.concatenate([labels["train"][kt], labels["valid"][kv]])[kf]
        del labels, frames
        old_meta = read(ROOT / (prior.MODEL + "metadata.json"))
        target_sha = component.values_sha256(y)
        need(y.shape == (467221,) and target_sha == old_meta["fit_target_values_sha256"], "Final train+valid target identity differs")
        fi = prior.index(prior.MODEL + "fit_indices.npy", 467221, 467221, True)
        qi = prior.index(prior.MODEL + "query_indices.npy", 47073, 47073, True)
        for name, values, upper in (("fit", fi, 467221), ("query", qi, 47073)):
            saved = prior.index(SPATIAL_FOLDER + name + "_indices.npy", upper, len(values), True)
            need(saved.tobytes() == values.tobytes(), "R34 final indices differ")
            save_array(model_folder / (name + "_indices.npy"), values)
        geo = pd.read_pickle(ROOT / (prior.GEO + "full.pkl"))
        geoq = pd.read_pickle(ROOT / (prior.GEO + "test.pkl"))
        need(geo.index.equals(X.index) and geoq.index.equals(Xq.index), "Final geo indices differ")
        before = {n: spatial.frame_sha(f) for n, f in (("X", X), ("Xq", Xq), ("geo", geo), ("geoq", geoq))}
        r34_proof = locked(ROOT / R34_PROOF)
        need(before == r34_proof["input_frame_sha256"]
             and before["geo"] == old_meta["fit_geo_feature_values_sha256"]
             and before["geoq"] == old_meta["query_geo_feature_values_sha256"], "Final frames differ from released trial")
        cats = schema["categorical_columns"]
        need(len(cats) == 21 and len(component.base.categories(X, cats)) == 23, "Original category schema differs")
        phase = "complete_v3_and_v4_control_replay"
        old_raw, unit, geo_raw, control = [prior.vector(n) for n in
            (prior.OLD_RAW, prior.OLD_UNIT, prior.MODEL + "predictions.npy", prior.V3_VECTOR)]
        old_base = .75 * (.5 * old_raw + .5 * geo_raw) + .25 * unit
        with (ROOT / prior.REFERENCE).open("rb") as stream:
            reference = pickle.load(stream)
        v3_replay, original_diagnostics, v3_reference_report = prior.predict_reference(reference, Xq, old_base, variant="trend", return_diagnostics=True)
        need(np.allclose(v3_replay, control, rtol=1e-12, atol=1e-5), "Unrounded v3 control differs")
        with (ROOT / (SPATIAL_FOLDER + "spatial_state.pkl")).open("rb") as stream:
            spatial_state = pickle.load(stream)
        spatial_summary = locked(ROOT / (SPATIAL_FOLDER + "state_summary.json"))
        need(spatial_state["parameters"] == spatial.parameters() and spatial_state["state_summary"] == spatial_summary
             and spatial_summary["fit_rows"] == len(X) and spatial_summary["query_used_in_build"] is False
             and spatial_summary["fit_X_sha256"] == before["X"] and spatial_summary["fit_geo_sha256"] == before["geo"]
             and spatial_summary["existing_reference_residual_sha256"] == component.values_sha256(reference["reference_residual"]), "Saved final spatial state/reference differs")
        v4 = prior.vector(SPATIAL_FOLDER + "predictions.npy")
        v4_replay, _, v4_report = spatial.predict(spatial_state, Xq, geoq, old_base, control, original_diagnostics)
        need(v4_replay.dtype == v4.dtype and v4_replay.tobytes() == v4.tobytes(), "Unrounded v4 trial differs")
        sample = pd.read_csv(prior.SAMPLE, encoding="utf-8-sig", usecols=[prior.ID], dtype={prior.ID: str}, keep_default_na=False)
        v2 = pd.read_csv(ROOT / "outputs/submission_v2.csv", encoding="utf-8-sig", usecols=[prior.ID], dtype={prior.ID: str})
        v3_csv = pd.read_csv(ROOT / "outputs/submission_v3.csv", encoding="utf-8-sig", dtype={prior.ID: str})
        v4_csv = pd.read_csv(ROOT / V4_CSV, encoding="utf-8-sig", dtype={prior.ID: str})
        ids = sample[prior.ID].tolist()
        need(len(ids) == 47073 and sample[prior.ID].is_unique and all(ids)
             and ids == v2[prior.ID].tolist() == v3_csv[prior.ID].tolist() == v4_csv[prior.ID].tolist(), "Sample/v2/v3/v4 IDs differ")
        for frame, values in ((v3_csv, control), (v4_csv, v4)):
            need(list(frame.columns) == [prior.ID, prior.TARGET] and frame[prior.TARGET].dtype == np.dtype("int64")
                 and (frame[prior.TARGET] > 0).all() and np.array_equal(rounded(values), frame[prior.TARGET].to_numpy()), "Existing CSV control differs")
        need(np.array_equal(rounded(v3_replay), v3_csv[prior.TARGET].to_numpy()), "v3 replay rounded prices differ")
        write_json(staging / "control_replay.json", {"rows": 47073, "v3_unrounded_rtol": 1e-12, "v3_unrounded_atol": 1e-5,
            "v3_max_abs_difference": float(np.max(np.abs(v3_replay - control))), "v3_CSV_exact": True,
            "v4_vector_bitwise_equal": True, "v4_CSV_exact": True, "v3_reference_report": v3_reference_report,
            "v4_prediction_report": v4_report, "old_tree_models_loaded": 0})
        phase = "fit_once"
        write_json(staging / "fit_started.json", {"event": "fit_started", "created_at_utc": stamp(),
            "plan_sha256": sha(OUT / "trial_plan.json"), "fit_rows": len(X), "query_rows": len(Xq),
            "fit_target_values_sha256": target_sha, "parameters": plan["parameters"], "automatic_retrain": False})
        progress["fits_started"] = 1
        print(json.dumps({"event": "fit_started", "fit_rows": len(X), "query_rows": len(Xq)}), flush=True)
        model, state, metadata = component.fit_model(X, y, geo, cats)
        progress["fits_completed"] = 1
        phase = "save_fitted_model_before_inference"
        model.save_model(str(model_folder / "model.cbm"))
        progress["models_saved"] = 1
        write_json(model_folder / "feature_state.json", state)
        metadata["component_source_sha256"] = metadata["source_sha256"]
        metadata.update({"round": 37, "stage": "final", "event": "fit_completed_model_saved",
            "created_at_utc": stamp(), "plan_sha256": sha(OUT / "trial_plan.json"), "source_sha256": plan["source_sha256"],
            "query_rows": len(Xq), "input_frame_sha256": before, "model_sha256": sha(model_folder / "model.cbm"),
            "feature_state_sha256": sha(model_folder / "feature_state.json"), "runtime": runtime(),
            "fit_indices_content_sha256": prior.hashlib.sha256(fi.astype("<i8").tobytes()).hexdigest(),
            "query_indices_content_sha256": prior.hashlib.sha256(qi.astype("<i8").tobytes()).hexdigest(),
            "test_labels_loaded": False, "reload_all_query_rows_verified": False})
        write_json(model_folder / "fit_metadata.json", metadata)
        print(json.dumps({"event": "fit_completed_model_saved"}), flush=True)
        need(metadata["parameters"] == plan["parameters"] and metadata["feature_count"] == 95
             and metadata["categorical_feature_count"] == 23 and metadata["iterations"] == 2999
             and metadata["sample_weight"] == "uniform" and metadata["feature_rules"] == component.FEATURE_RULES
             and metadata["fit_rows"] == 467221, "Fixed fitted component differs")
        need(metadata["fit_target_values_sha256"] == metadata["fit_original_target_values_sha256"] == target_sha
             and metadata["fit_transformed_target_values_sha256"] == component.values_sha256(component.transform_target(y)), "Fit target hashes differ")
        for key in ("weight_argument_passed_to_pool", "query_labels_used", "fit_uses_eval_set", "early_stopping_used",
                    "query_used_for_geographic_state", "target_used_for_geographic_state", "calibration_applied",
                    "extra_price_cap_applied", "reference_or_trend_applied_here"):
            need(metadata[key] is False, "Forbidden fit behavior: " + key)
        need(component.values_sha256(y) == target_sha, "Fit labels changed")
        del y
        phase = "save_and_reload_complete_query"
        z = raw_predict(model, state, Xq, geoq, cats)
        log_geo = component.inverse_transform(z)
        for name, values in (("raw_z", z), ("predictions", log_geo)):
            save_array(model_folder / (name + ".npy"), values)
        del model
        restored_model = CatBoostRegressor()
        restored_model.load_model(str(model_folder / "model.cbm"))
        restored_state = locked(model_folder / "feature_state.json")
        need(restored_state == state, "Saved feature state differs")
        restored_z = raw_predict(restored_model, restored_state, Xq, geoq, cats)
        restored_TWD = component.inverse_transform(restored_z)
        need(restored_z.dtype == z.dtype and restored_z.tobytes() == z.tobytes()
             and restored_TWD.dtype == log_geo.dtype and restored_TWD.tobytes() == log_geo.tobytes(), "Full raw/TWD model reload differs")
        for name, values in (("raw_z", z), ("predictions", log_geo)):
            saved = np.load(model_folder / (name + ".npy"), allow_pickle=False)
            need(saved.dtype == values.dtype and saved.shape == values.shape and saved.tobytes() == values.tobytes(), "Saved native vector differs")
        del restored_model
        progress["full_query_reload_verified"] = True
        metadata.update({"event": "complete_query_reload_verified", "reload_all_query_rows_verified": True,
            "reload_query_rows": 47073, "raw_and_TWD_reload_bitwise_equal": True,
            "raw_z_diagnostics": {"rows": len(z), "negative_count": int((z < 0).sum()), "zero_count": int((z == 0).sum()),
                "positive_count": int((z > 0).sum()), "minimum": float(z.min()), "maximum": float(z.max()),
                "zero_TWD_count": int((log_geo == 0).sum()), "used_for_tuning": False},
            "artifact_sha256": {rel(DEST / "model" / p.name): sha(p) for p in sorted(model_folder.iterdir()) if p.is_file()}})
        write_json(model_folder / "metadata.json", metadata)
        phase = "unchanged_R36_combination_with_saved_final_spatial_state"
        base_log = .75 * (.5 * old_raw + .5 * log_geo) + .25 * unit
        p_log, diagnostics, reference_report = prior.predict_reference(reference, Xq, base_log, variant="trend", return_diagnostics=True)
        prediction, detail, report = spatial.predict(spatial_state, Xq, geoq, base_log, p_log, diagnostics)
        need(prediction.dtype == np.dtype("float64") and prediction.shape == (47073,), "New combination vector schema differs")
        prices = rounded(prediction)
        fallback = ~detail["applied"].to_numpy(dtype=bool)
        need(prediction[fallback].tobytes() == p_log[fallback].tobytes(), "Fallback must preserve p_log")
        for name, values in (("base_log", base_log), ("p_log", p_log), ("predictions", prediction)):
            save_array(staging / (name + ".npy"), values)
            saved = np.load(staging / (name + ".npy"), allow_pickle=False)
            need(saved.dtype == values.dtype and saved.shape == values.shape and saved.tobytes() == values.tobytes(), "Saved combination vector differs")
        for name, value in (("reference_diagnostics", diagnostics), ("diagnostics", detail)):
            with (staging / (name + ".pkl")).open("xb") as stream:
                pickle.dump(value, stream, protocol=5)
        write_json(staging / "prediction_report.json", report)
        after = {n: spatial.frame_sha(f) for n, f in (("X", X), ("Xq", Xq), ("geo", geo), ("geoq", geoq))}
        need(after == before, "Original feature frames changed")
        phase = "new_CSV_roundtrip"
        staged_csv = staging / CSV.name
        pd.DataFrame({prior.ID: ids, prior.TARGET: prices}).to_csv(staged_csv, index=False, encoding="utf-8-sig", mode="x")
        back = pd.read_csv(staged_csv, encoding="utf-8-sig", dtype={prior.ID: str})
        need(list(back.columns) == [prior.ID, prior.TARGET] and back[prior.ID].tolist() == ids and back[prior.ID].is_unique
             and back[prior.TARGET].dtype == np.dtype("int64") and (back[prior.TARGET] > 0).all()
             and np.array_equal(back[prior.TARGET].to_numpy(), prices), "New trial CSV roundtrip differs")
        verify(plan["source_sha256"])
        verify(plan["protected_sha256"])
        proof = {"round": 37, "status": "experimental_trial_generated", "created_at_utc": stamp(),
            "trial_only": True, "formal_acceptance": False, "recommended_version": "v3", "registry_updated": False,
            "plan_sha256": sha(OUT / "trial_plan.json"), "source_sha256": plan["source_sha256"],
            "input_sha256": plan["input_sha256"], "runtime": runtime(), "parameters": plan["parameters"],
            "spatial_parameters": plan["spatial_parameters"], "formula": FORMULA, "fit_rows": 467221, "query_rows": 47073,
            "feature_count": 95, "categorical_feature_count": 23, "geography_year": 2024, "progress": progress,
            "R12_plan_sha256": R12_SHA, "R34_release_sha256": R34_PROOF_SHA,
            "R35_results_sha256": R35_RESULT_SHA, "R36_results_sha256": R36_RESULT_SHA,
            "input_frame_sha256": before, "fit_target_values_sha256": target_sha,
            "new_tree_models_trained": 1, "old_tree_models_loaded": 0, "spatial_state_rebuilt": False, "reference_refitted": False,
            "labels_loaded": "train and valid only", "test_labels_loaded": False,
            "full_query_raw_TWD_reload_bitwise_equal": True, "reload_rows": 47073,
            "v3_control_full_replay": True, "v3_CSV_exact": True, "v4_control_bitwise_equal": True, "v4_CSV_exact": True,
            "reference_trend_applications": 1, "spatial_future_base": "B_log", "fallback_baseline": "p_log",
            "fallback_rows": int(fallback.sum()), "fallback_bitwise_preserved": True,
            "prediction_changes_vs_v3": adjustment(prediction, control, prices, v3_csv[prior.TARGET].to_numpy()),
            "prediction_changes_vs_v4": adjustment(prediction, v4, prices, v4_csv[prior.TARGET].to_numpy()),
            "adjustments_are_not_prediction_error_metrics": True, "reference_report": reference_report, "prediction_report": report,
            "path": rel(CSV), "sha256": sha(staged_csv), "rounding": "max(rint(TWD),1)",
            "id_order_verified": True, "positive_int64_roundtrip": True, "UTF8_BOM": True,
            "test_mae": None, "public_mae": None, "private_mae": None, "uploaded_to_kaggle": False,
            "old_decisions_changed": False, "protected_before": plan["protected_sha256"],
            "protected_after": {n: sha(ROOT / n) for n in PROTECTED}, "elapsed_seconds": time.monotonic() - started,
            "artifact_sha256": {rel(DEST / p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file() and p != staged_csv}}
        need(proof["protected_before"] == proof["protected_after"], "Protected artifacts changed")
        write_json(staging / "trial_release.json", proof)
        phase = "publish_new_paths_only"
        need(not DEST.exists() and not CSV.exists() and staging.resolve().parent == OUT.resolve()
             and DEST.resolve().parent == OUT.resolve() and CSV.resolve().parent == (ROOT / "outputs").resolve(), "Unsafe/existing destination")
        staging.rename(DEST)
        (DEST / CSV.name).rename(CSV)
        need(sha(CSV) == proof["sha256"], "Published CSV differs")
        verify(plan["protected_sha256"])
        write_json(OUT / "execution.json", {"status": "completed", "finished_at_utc": stamp(), "progress": progress,
            "trial_release_sha256": sha(DEST / "trial_release.json"), "submission_sha256": sha(CSV),
            "protected_after_publication": {n: sha(ROOT / n) for n in PROTECTED}})
        print(json.dumps({"status": proof["status"], "path": rel(CSV), "sha256": proof["sha256"], "recommended_version": "v3"}), flush=True)
        return 0
    except Exception as exc:
        write_json(OUT / "failure.json", {"status": "technical_failure_preserved_no_automatic_retry", "phase": phase,
            "staging_path": rel(staging), "progress": progress, "created_at_utc": stamp(),
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "protected_after": {n: sha(ROOT / n) for n in PROTECTED}, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(run())
