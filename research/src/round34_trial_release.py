"""Separate R8 final-data trial CSV; no training, scoring, acceptance or registry edit.

Freeze then run once. All feature/array/pickle inputs are byte-verified first.
The original final reference already contains train+valid target-derived state.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import round8_spatial_reference as spatial
from round2_neighborhood import predict_reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round34"
DEST = OUT / "trial"
CSV = ROOT / "outputs/submission_v4_trial.csv"
DOCUMENT = ROOT / "reports/round34_trial_plan.md"
SAMPLE = ROOT / "台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/test-reindex-test-sample-submission-v2.csv"
ID, TARGET = "編號", "總價元"
V3_SNAPSHOT = "77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525"
R8_PLAN = "ea32d46733371ffcef27946be47e40dd1c32ac5c4ab1019eaa31a65773c4e08d"
HELPER = "de86397255d654f363f3b746d579923fe689aa41d9dc16335d0922fea05e1c32"
GEO = "outputs/round6/geography/final/"
MODEL = "outputs/round6/models/final/geography_catboost/"
REFERENCE = "outputs/round2/neighborhood/final_reference.pkl"
OLD_RAW = "outputs/round2/final_predictions/catboost_raw_test.npy"
OLD_UNIT = "outputs/round2/final_predictions/lightgbm_l1_unit63_test.npy"
V3_VECTOR = "outputs/round6/release/predictions.npy"
PROTECTED = (
    "outputs/submission_v2.csv", "outputs/submission_v3.csv", "runs/versions.json",
    "outputs/round6/candidate_plan.json", "outputs/round6/frozen_candidate.json", "outputs/round6/accepted_candidate.json",
    "outputs/round8/candidate_plan.json", "outputs/round8/frozen_candidate.json", "outputs/round8/accepted_candidate.json",
    "outputs/round32/candidate_plan.json", "outputs/round32/frozen_candidate.json", "outputs/round32/accepted_candidate.json",
    "outputs/round33/reproduction_plan.json", "outputs/round33/results.json",
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locked(path):
    path = Path(path)
    need(path.with_suffix(".sha256").read_text(encoding="ascii").strip() == sha(path), "Receipt differs: " + str(path))
    return read(path)


def write_json(path, value):
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    need(not path.exists() and not receipt.exists(), "Refusing overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with receipt.open("x", encoding="ascii") as stream:
        stream.write(sha(path) + "\n")


def merge(*maps):
    result = {}
    for mapping in maps:
        for name, digest in mapping.items():
            path = (ROOT / name).resolve()
            need(path.is_relative_to(ROOT.resolve()) and path.is_file(), "Missing/unsafe input: " + name)
            need(name not in result or result[name] == digest, "Conflicting identity: " + name)
            result[name] = digest
    return result


def verify(mapping):
    for name, digest in merge(mapping).items():
        need(sha(ROOT / name) == digest, "Changed input: " + name)


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
    return {"python": sys.version, **{n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "threadpoolctl")}}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    need(not (OUT / "trial_plan.json").exists() and not DEST.exists() and not CSV.exists(), "Trial already exists")
    snapshot_path = ROOT / "runs/v3/snapshot_manifest.json"
    need(sha(snapshot_path) == V3_SNAPSHOT, "Released v3 snapshot differs")
    snapshot = locked(snapshot_path)
    known = {n: v["original_sha256"] for n, v in snapshot["copied_files"].items()}
    known = merge(known, {v["original_path"]: v["sha256"] for v in snapshot["external_dependencies"]})
    v2_path = ROOT / "runs/v2/snapshot_manifest.json"
    need(sha(v2_path) == known[rel(v2_path)], "Released v2 snapshot differs")
    v2 = read(v2_path)["files_sha256"]
    r8_path = ROOT / "outputs/round8/candidate_plan.json"
    need(sha(r8_path) == R8_PLAN and sha(ROOT / "src/round8_spatial_reference.py") == HELPER, "Original R8 changed")
    r8 = locked(r8_path)
    need(r8["parameters"] == spatial.parameters(), "Spatial parameters changed")
    current_sources = sources()
    for name, digest in current_sources.items():
        if name != rel(Path(__file__)):
            need(r8["source_sha256"].get(name) == digest, "Original helper dependency changed: " + name)
    authority_names = ["outputs/round6/candidate_plan.json", "outputs/round6/release/release.json",
                       GEO + "manifest.json", MODEL + "metadata.json", "reports/feature_schema.json"]
    for name in authority_names:
        need(sha(ROOT / name) == known[name], "Released authority changed: " + name)
    release = locked(ROOT / authority_names[1])
    geography = locked(ROOT / (GEO + "manifest.json"))
    meta = read(ROOT / (MODEL + "metadata.json"))
    need(geography["stage"] == "final" and geography["snapshot_year"] == 2024
         and geography["geography_provenance_chain"]["verified"] is True, "Final geography differs")
    need(meta["outer_fit_rows"] == 467221 and meta["query_rows"] == 47073
         and meta["geography_manifest_sha256"] == sha(ROOT / (GEO + "manifest.json")), "Final partition differs")
    need(release["unrounded_prediction_sha256"] == known[V3_VECTOR]
         and release["reference_sha256"] == v2[REFERENCE] == known[REFERENCE], "Released prediction/reference identity differs")
    r33_path = ROOT / "outputs/round33/results.json"
    r33 = locked(r33_path)
    r33_plan_path = ROOT / "outputs/round33/reproduction_plan.json"
    r33_plan = locked(r33_plan_path)
    need(r33["status"] == "both_periods_completed_no_adoption_decision"
         and r33["checked_stages"] == ["confirmation", "auxiliary"]
         and r33["plan_sha256"] == sha(r33_plan_path)
         and r33["new_formal_acceptance_decision"] is None
         and r33_plan["parameters"] == spatial.parameters(), "R33 evidence incomplete/different")
    paths = [snapshot_path, snapshot_path.with_suffix(".sha256"), v2_path, r8_path, r8_path.with_suffix(".sha256"),
             DOCUMENT, SAMPLE, r33_path, r33_path.with_suffix(".sha256"), r33_plan_path, r33_plan_path.with_suffix(".sha256")]
    for stage in ("confirmation", "auxiliary"):
        p = ROOT / "outputs/round33" / stage / "results.json"
        need(sha(p) == r33["stage_result_sha256"][stage], "R33 period evidence differs")
        value = locked(p)
        need(value["stage"] == stage and value["status"] == "completed" and value["plan_sha256"] == sha(r33_plan_path), "R33 period contract differs")
        paths.extend([p, p.with_suffix(".sha256")])
    historical = ROOT / r33["original_R8_development"]["path"]
    need(sha(historical) == r33["original_R8_development"]["sha256"], "Original development evidence differs")
    locked(historical)
    paths.extend([historical, historical.with_suffix(".sha256")])
    for name in authority_names:
        paths.append(ROOT / name)
        if (ROOT / name).with_suffix(".sha256").is_file():
            paths.append((ROOT / name).with_suffix(".sha256"))
    ref_meta = "outputs/round2/neighborhood/final_reference.json"
    need(sha(ROOT / ref_meta) == v2[ref_meta], "Final reference metadata differs")
    expected = {n: known[n] for n in (V3_VECTOR, MODEL + "predictions.npy", MODEL + "fit_indices.npy",
                                    MODEL + "query_indices.npy", "outputs/submission_v3.csv")}
    expected.update({n: v2[n] for n in (OLD_RAW, OLD_UNIT, REFERENCE, ref_meta, "outputs/submission_v2.csv")})
    # Original large geography provenance remains inherited by this manifest.
    # Only the actual caches, index arrays and final geo frames are deserialized.
    for split in ("train", "valid", "test"):
        name = f"data_cache/{split}_features.pkl"
        need(geography["source_sha256"][name] == known[name], "Cache authority conflict")
        expected[name] = known[name]
    for item in geography["features"].values():
        if item["path"] in (GEO + "full.pkl", GEO + "test.pkl"):
            expected[item["path"]] = item["sha256"]
    expected = merge(expected, geography["index_files_sha256"], {rel(p): sha(p) for p in paths})
    protected = {n: sha(ROOT / n) for n in PROTECTED}
    value = {"round": 34, "status": "trial_plan_frozen", "created_at_utc": stamp(),
             "trial_only": True, "formal_acceptance": False, "recommended_version": "v3", "registry_update_allowed": False,
             "submission_path": rel(CSV), "fit_rows": 467221, "query_rows": 47073, "geography_year": 2024,
             "parameters": spatial.parameters(), "source_sha256": current_sources, "input_sha256": expected,
             "protected_sha256": protected, "runtime": runtime(), "R33_results_sha256": sha(r33_path),
             "raw_base_formula": ".75*(.5*old_raw+.5*geo_raw)+.25*old_unit", "reference_trend_applications": 1,
             "raw_labels_loaded": False, "reference_contains_historical_train_valid_labels": True,
             "new_tree_training": False, "test_labels_available": False, "test_score_computed": False,
             "trial_public_mae": None, "trial_private_mae": None, "old_decisions_revised": False,
             "expected_large_input_hashes": "inherited published contracts; verified before run deserialization"}
    need(current_sources == sources(), "Source changed during freeze")
    verify({rel(snapshot_path): V3_SNAPSHOT, rel(v2_path): known[rel(v2_path)], rel(r8_path): R8_PLAN})
    write_json(OUT / "trial_plan.json", value)
    print(json.dumps({"status": value["status"], "plan_sha256": sha(OUT / "trial_plan.json")}), flush=True)


def index(name, upper, count, identity=False):
    values = np.load(ROOT / name, allow_pickle=False)
    need(values.dtype == np.dtype("int64") and values.shape == (count,), "Index shape/dtype differs: " + name)
    need((values >= 0).all() and (values < upper).all() and (np.diff(values) > 0).all(), "Index range/order differs")
    if identity:
        need(np.array_equal(values, np.arange(count, dtype="int64")), "Expected complete index sequence")
    return values


def vector(name):
    values = np.load(ROOT / name, allow_pickle=False)
    need(values.dtype == np.dtype("float64") and values.shape == (47073,)
         and np.isfinite(values).all() and (values >= 0).all(), "Invalid TWD vector: " + name)
    return values


def rounded(values):
    need(np.isfinite(values).all() and (values >= 0).all() and (values < np.iinfo("int64").max).all(), "Invalid output prices")
    return np.maximum(np.rint(values), 1).astype("int64")


def save_array(path, values):
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def run():
    plan = locked(OUT / "trial_plan.json")
    need(plan["round"] == 34 and plan["trial_only"] is True and plan["formal_acceptance"] is False
         and plan["registry_update_allowed"] is False and plan["submission_path"] == rel(CSV), "Wrong trial contract")
    need(plan["source_sha256"] == sources() and plan["runtime"] == runtime()
         and plan["parameters"] == spatial.parameters(), "Frozen runtime/source/method changed")
    need(not DEST.exists() and not CSV.exists(), "Refusing existing trial output")
    write_json(OUT / "run_claim.json", {"plan_sha256": sha(OUT / "trial_plan.json"), "started_at_utc": stamp(), "automatic_retry": False})
    start = time.monotonic()
    phase = "input_hash_verification"
    staging = OUT / (".trial-staging-" + uuid.uuid4().hex)
    try:
        verify(merge(plan["input_sha256"], plan["source_sha256"], plan["protected_sha256"]))
        staging.mkdir(exist_ok=False)
        phase = "pure_X_saved_index_alignment"
        schema = read(ROOT / "reports/feature_schema.json")
        frames = {s: pd.read_pickle(ROOT / f"data_cache/{s}_features.pkl") for s in ("train", "valid", "test")}
        for split, count in (("train", 455784), ("valid", 11994), ("test", 47073)):
            frame = frames[split]
            need(len(frame) == count and list(frame.columns) == schema["columns"] and len(frame.columns) == 87
                 and frame.index.equals(pd.RangeIndex(count)) and TARGET not in frame, "Original X schema/order differs")
        kt = index(GEO + "train_source_cache_indices.npy", 455784, 455255)
        kv = index(GEO + "valid_source_cache_indices.npy", 11994, 11994, True)
        kf = index(GEO + "full_source_cache_indices.npy", 467249, 467221)
        kq = index(GEO + "test_source_cache_indices.npy", 47073, 47073, True)
        X = pd.concat([frames["train"].iloc[kt], frames["valid"].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
        Xq = frames["test"].iloc[kq].reset_index(drop=True)
        need(len(X) == 467221 and X.dtypes.equals(Xq.dtypes), "Full original X dtype/alignment differs")
        fi = index(MODEL + "fit_indices.npy", 467221, 467221, True)
        qi = index(MODEL + "query_indices.npy", 47073, 47073, True)
        geo, geoq = pd.read_pickle(ROOT / (GEO + "full.pkl")), pd.read_pickle(ROOT / (GEO + "test.pkl"))
        need(geo.index.equals(X.index) and geoq.index.equals(Xq.index), "Geo/X index differs")
        meta = read(ROOT / (MODEL + "metadata.json"))
        need(spatial.frame_sha(geo) == meta["fit_geo_feature_values_sha256"]
             and spatial.frame_sha(geoq) == meta["query_geo_feature_values_sha256"], "Saved geography values differ")
        before = {n: spatial.frame_sha(f) for n, f in (("X", X), ("Xq", Xq), ("geo", geo), ("geoq", geoq))}
        phase = "saved_component_and_reference_control_replay"
        old_raw, unit, geographic, control = [vector(n) for n in (OLD_RAW, OLD_UNIT, MODEL + "predictions.npy", V3_VECTOR)]
        base = .75 * (.5 * old_raw + .5 * geographic) + .25 * unit
        with (ROOT / REFERENCE).open("rb") as stream:
            reference = pickle.load(stream)
        replay, diagnostics, control_report = predict_reference(reference, Xq, base, variant="trend", return_diagnostics=True)
        need(np.allclose(replay, control, rtol=1e-12, atol=1e-5), "Full unrounded v3 control replay differs")
        sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", usecols=[ID], dtype={ID: str}, keep_default_na=False)
        old_csv = pd.read_csv(ROOT / "outputs/submission_v3.csv", encoding="utf-8-sig", dtype={ID: str})
        v2_ids = pd.read_csv(ROOT / "outputs/submission_v2.csv", encoding="utf-8-sig", usecols=[ID], dtype={ID: str})
        ids = sample[ID].tolist()
        need(len(ids) == 47073 and sample[ID].is_unique and all(ids)
             and ids == old_csv[ID].tolist() == v2_ids[ID].tolist(), "Sample/v2/v3 ID order differs")
        need(list(old_csv.columns) == [ID, TARGET] and old_csv[TARGET].dtype.kind in "iu"
             and np.array_equal(rounded(replay), old_csv[TARGET].to_numpy())
             and np.array_equal(rounded(control), old_csv[TARGET].to_numpy()), "Original v3 CSV is not reproduced exactly")
        phase = "unchanged_R8_final_state_and_prediction"
        model, summary = spatial.build(reference, X, geo)
        prediction, detail, report = spatial.predict(model, Xq, geoq, base, control, diagnostics)
        need(prediction.dtype == np.dtype("float64") and prediction.shape == (47073,), "Trial vector schema differs")
        prices = rounded(prediction)
        fallback = ~detail["applied"].to_numpy(dtype=bool)
        need(prediction[fallback].tobytes() == control[fallback].tobytes(), "Fallback rows changed")
        with (staging / "spatial_state.pkl").open("xb") as stream:
            pickle.dump(model, stream, protocol=5)
        with (staging / "spatial_state.pkl").open("rb") as stream:
            reloaded = pickle.load(stream)
        again, _, again_report = spatial.predict(reloaded, Xq, geoq, base, control, diagnostics)
        need(again.dtype == prediction.dtype and again.tobytes() == prediction.tobytes()
             and again_report == report, "Full saved-state replay differs")
        after = {n: spatial.frame_sha(f) for n, f in (("X", X), ("Xq", Xq), ("geo", geo), ("geoq", geoq))}
        need(after == before, "Feature frames changed")
        for name, values in (("predictions", prediction), ("fit_indices", fi), ("query_indices", qi)):
            save_array(staging / (name + ".npy"), values)
        with (staging / "diagnostics.pkl").open("xb") as stream:
            pickle.dump(detail, stream, protocol=5)
        write_json(staging / "state_summary.json", summary)
        write_json(staging / "prediction_report.json", report)
        staged_csv = staging / CSV.name
        pd.DataFrame({ID: ids, TARGET: prices}).to_csv(staged_csv, index=False, encoding="utf-8-sig", mode="x")
        back = pd.read_csv(staged_csv, encoding="utf-8-sig", dtype={ID: str})
        need(list(back.columns) == [ID, TARGET] and back[ID].tolist() == ids and back[ID].is_unique
             and back[TARGET].dtype == np.dtype("int64") and (back[TARGET] > 0).all()
             and np.array_equal(back[TARGET].to_numpy(), prices), "Trial CSV round trip differs")
        saved = np.load(staging / "predictions.npy", allow_pickle=False)
        need(saved.dtype == prediction.dtype and saved.tobytes() == prediction.tobytes(), "Saved vector differs")
        adjustment = prediction - control
        absolute = np.abs(adjustment)
        proof = {"round": 34, "status": "experimental_trial_generated", "trial_only": True,
                 "formal_acceptance": False, "recommended_version": "v3", "registry_updated": False,
                 "old_R8_decision_changed": False, "R33_promoted_to_formal_accepted": False,
                 "created_at_utc": stamp(), "plan_sha256": sha(OUT / "trial_plan.json"),
                 "source_sha256": plan["source_sha256"], "input_sha256": plan["input_sha256"],
                 "runtime": runtime(), "parameters": spatial.parameters(), "fit_rows": len(X), "query_rows": len(Xq),
                 "input_frame_sha256": before, "R33_results_sha256": plan["R33_results_sha256"],
                 "geography_year": 2024, "raw_labels_loaded": False, "saved_reference_uses_train_valid_labels": True,
                 "trees_trained": 0, "tree_models_loaded": 0, "v2_reference_refitted": False,
                 "control_full_rows_replayed": 47073, "control_unrounded_atol": 1e-5, "control_unrounded_rtol": 1e-12,
                 "control_max_abs_difference": float(np.max(np.abs(replay - control))), "v3_CSV_reproduced_exactly": True,
                 "saved_state_reload_rows": 47073, "saved_state_replay_bitwise_equal": True,
                 "fallback_rows": int(fallback.sum()), "fallback_bitwise_preserved": True,
                 "changed_unrounded_rows": int(np.count_nonzero(adjustment)),
                 "changed_rounded_rows": int(np.count_nonzero(prices != old_csv[TARGET].to_numpy())),
                 "adjustment_TWD": {"minimum": float(adjustment.min()), "maximum": float(adjustment.max()),
                                    "mean_signed": float(np.mean(adjustment)), "median_signed": float(np.median(adjustment)),
                                    "mean_absolute": float(np.mean(absolute)), "p95_absolute": float(np.quantile(absolute, .95))},
                 "adjustments_are_not_prediction_error_metrics": True, "prediction_report": report,
                 "control_replay_report": control_report, "path": rel(CSV), "sha256": sha(staged_csv),
                 "rounding": "max(rint(TWD),1)", "id_order_verified": True, "positive_int64_roundtrip": True,
                 "public_mae": None, "private_mae": None, "test_mae": None, "uploaded_to_kaggle": False,
                 "elapsed_seconds": time.monotonic() - start,
                 "protected_before": plan["protected_sha256"],
                 "artifact_sha256": {rel(DEST / p.name): sha(p) for p in sorted(staging.iterdir()) if p.is_file() and p != staged_csv}}
        verify(plan["protected_sha256"])
        verify(plan["source_sha256"])
        proof["protected_after"] = {n: sha(ROOT / n) for n in PROTECTED}
        need(proof["protected_before"] == proof["protected_after"], "Protected artifacts changed")
        write_json(staging / "trial_release.json", proof)
        phase = "publish_new_paths_only"
        need(not DEST.exists() and not CSV.exists() and staging.resolve().parent == OUT.resolve()
             and DEST.resolve().parent == OUT.resolve() and CSV.resolve().parent == (ROOT / "outputs").resolve(), "Unsafe/existing destination")
        staging.rename(DEST)
        (DEST / CSV.name).rename(CSV)
        need(sha(CSV) == proof["sha256"], "Published CSV changed")
        verify(plan["protected_sha256"])
        write_json(OUT / "execution.json", {"status": "completed", "finished_at_utc": stamp(),
                   "trial_release_sha256": sha(DEST / "trial_release.json"), "submission_sha256": sha(CSV),
                   "protected_after_publication": {n: sha(ROOT / n) for n in PROTECTED}})
        print(json.dumps({"status": proof["status"], "path": rel(CSV), "sha256": proof["sha256"],
                          "changed_rows": proof["changed_unrounded_rows"], "recommended_version": "v3"}), flush=True)
        return 0
    except Exception as exc:
        write_json(OUT / "failure.json", {"status": "incomplete_technical_failure", "phase": phase,
                   "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                   "created_at_utc": stamp(), "staging_path": rel(staging), "automatic_retry": False})
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
