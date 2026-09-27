"""One fixed mean of the completed v4/v5 unrounded vectors; trial CSV only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round38/trial_release"
DEST = OUT / "artifacts"
CSV = ROOT / "outputs/submission_v6_trial.csv"
DOCUMENT = ROOT / "reports/round38_trial_plan.md"
SAMPLE = "台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/test-reindex-test-sample-submission-v2.csv"
ID, TARGET = "編號", "總價元"
FORMULA = "0.5*R34_unrounded_TWD+0.5*R37_unrounded_TWD"
VECTORS = {"v4": "outputs/round34/trial/predictions.npy", "v5": "outputs/round37/trial/predictions.npy"}
CSVS = {"v3": "outputs/submission_v3.csv", "v4": "outputs/submission_v4_trial.csv", "v5": "outputs/submission_v5_trial.csv"}
AUTHORITIES = {
    "outputs/round34/trial/trial_release.json": "67069dd953999d56c6e1e1fab70a5d7e0a0f98272a1bcde48a3bb538c5b64033",
    "outputs/round37/trial_plan.json": "2679b8e87619dcfe7ec46bdb8141ce7d3cd4fc4b65aa07cf579a6c168619e6ef",
    "outputs/round37/trial/trial_release.json": "e7794550b976eca72fe3e6fb0bcdbdf02a24ce0b8e008139d328a2422bb827fe",
    "outputs/round37/execution.json": "e666419babbdae56ed14ca3df8bb0826a294915de0ec7ef5c547e7a69473b945",
    "outputs/round38/plan.json": "f885d8e2f37ee8655d1ca7e6af448572b847aaaee3b19cd96badaba297759f4b",
    "outputs/round38/results.json": "f14d84c7c9163cf894f5302fb6b6f8de0168ff56d2c80aa5bdd0c7f29ae5f947",
}
R37_SOURCE_SHA = "d77f03376e06b95644523b7e6d7a89da19bf467073ba58e8fd46f7b5695942ef"
PROTECTED = tuple(CSVS.values()) + ("outputs/submission_v2.csv", "runs/versions.json") + tuple(AUTHORITIES)


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


def now():
    return datetime.now(timezone.utc).isoformat()


def locked(path):
    path = Path(path)
    need(path.with_suffix(".sha256").read_text(encoding="ascii").strip() == sha(path), "Receipt differs: " + rel(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    need(not path.exists() and not receipt.exists(), "Refusing overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with receipt.open("x", encoding="ascii") as stream:
        stream.write(sha(path) + "\n")


def verify(mapping):
    for name, expected in mapping.items():
        path = (ROOT / name).resolve()
        need(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == expected, "Input differs/unsafe: " + name)


def bind(mapping, name, digest):
    need(name not in mapping or mapping[name] == digest, "Conflicting identity: " + name)
    mapping[name] = digest


def runtime():
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}


def freeze():
    need(not (OUT / "plan.json").exists() and not (OUT / "run_claim.json").exists()
         and not DEST.exists() and not CSV.exists(), "Existing trial attempt")
    verify(AUTHORITIES)
    values = {name: locked(ROOT / name) for name in AUTHORITIES}
    p37 = values["outputs/round37/trial_plan.json"]
    r34 = values["outputs/round34/trial/trial_release.json"]
    r37 = values["outputs/round37/trial/trial_release.json"]
    e37 = values["outputs/round37/execution.json"]
    p38, r38 = values["outputs/round38/plan.json"], values["outputs/round38/results.json"]
    need(r38["round"] == p38["round"] == 38 and r38["status"] == "three_periods_completed_no_adoption_decision"
         and r38["checked_stages"] == p38["stage_order"] == ["development", "confirmation", "auxiliary"]
         and r38["plan_sha256"] == AUTHORITIES["outputs/round38/plan.json"]
         and p38["formula"] == "0.5*spatial_v3+0.5*log_spatial_combination"
         and r38["formal_adoption_decision"] is None and r38["submission_generated"] is False, "R38 research contract differs")
    need(p37["round"] == r37["round"] == 37 and r37["plan_sha256"] == AUTHORITIES["outputs/round37/trial_plan.json"]
         and r37["source_sha256"] == p37["source_sha256"]
         and p37["source_sha256"]["src/round37_trial_release.py"] == R37_SOURCE_SHA
         and e37["status"] == "completed" and e37["trial_release_sha256"] == AUTHORITIES["outputs/round37/trial/trial_release.json"]
         and e37["submission_sha256"] == r37["sha256"] and e37["progress"]["full_query_reload_verified"] is True,
         "R37 completed release/source differs")
    inputs = dict(AUTHORITIES)
    for name in AUTHORITIES:
        receipt = (ROOT / name).with_suffix(".sha256")
        bind(inputs, rel(receipt), sha(receipt))
    bind(inputs, SAMPLE, p37["input_sha256"][SAMPLE])
    for version, proof in (("v4", r34), ("v5", r37)):
        need(proof["status"] == "experimental_trial_generated" and proof["query_rows"] == 47073
             and proof["path"] == CSVS[version] and proof["trial_only"] is True
             and proof["formal_acceptance"] is False, "Prior trial release contract differs")
        bind(inputs, VECTORS[version], proof["artifact_sha256"][VECTORS[version]])
        bind(inputs, CSVS[version], proof["sha256"])
    bind(inputs, CSVS["v3"], p37["input_sha256"][CSVS["v3"]])
    inherited_sources = {"src/round34_trial_release.py": r34["source_sha256"]["src/round34_trial_release.py"],
                         "src/round37_trial_release.py": R37_SOURCE_SHA,
                         "src/round38_fixed_average.py": p38["source_sha256"]["src/round38_fixed_average.py"]}
    verify(inherited_sources)
    source_map = {rel(Path(__file__)): sha(Path(__file__)), **inherited_sources}
    bind(inputs, rel(DOCUMENT), sha(DOCUMENT))
    need(all(runtime()[name] == p37["runtime"][name] for name in ("numpy", "pandas")), "Array/CSV runtime differs")
    protected = {name: sha(ROOT / name) for name in PROTECTED}
    for name in CSVS.values():
        need(protected[name] == inputs[name], "Protected CSV differs from release")
    plan = {"round": 38, "kind": "separate_fixed_average_final_trial", "created_at_utc": now(),
            "formula": FORMULA, "query_rows": 47073, "submission_path": rel(CSV),
            "runtime": runtime(), "source_sha256": source_map, "input_sha256": inputs, "protected_sha256": protected,
            "trial_only": True, "formal_acceptance": False, "recommended_version": "v3", "registry_update_allowed": False,
            "planned_after_R38_results": True, "model_training": False, "labels_loaded": False,
            "feature_or_model_objects_loaded": False, "price_scoring": False, "public_mae": None, "private_mae": None}
    verify(AUTHORITIES)
    verify(source_map)
    write(OUT / "plan.json", plan)
    print("trial_plan_frozen", sha(OUT / "plan.json"), flush=True)


def rounded(values):
    need(np.isfinite(values).all() and (values >= 0).all() and (values < np.iinfo("int64").max).all(), "Invalid output values")
    return np.maximum(np.rint(values), 1).astype("int64")


def changes(values, baseline):
    delta = values.astype("float64") - baseline.astype("float64")
    absolute = np.abs(delta)
    return {"changed_rows": int(np.count_nonzero(delta)), "minimum_TWD": float(delta.min()),
            "maximum_TWD": float(delta.max()), "mean_signed_TWD": float(delta.mean()),
            "mean_absolute_TWD": float(absolute.mean()), "p95_absolute_TWD": float(np.quantile(absolute, .95)),
            "is_error_or_score_improvement": False}


def run():
    plan = locked(OUT / "plan.json")
    need(plan["round"] == 38 and plan["kind"] == "separate_fixed_average_final_trial"
         and plan["formula"] == FORMULA and plan["submission_path"] == rel(CSV)
         and plan["trial_only"] is True and plan["formal_acceptance"] is False
         and plan["registry_update_allowed"] is False and plan["runtime"] == runtime(), "Frozen trial differs")
    need(not DEST.exists() and not CSV.exists(), "Refusing existing output")
    write(OUT / "run_claim.json", {"started_at_utc": now(), "plan_sha256": sha(OUT / "plan.json"), "automatic_retry": False})
    started, phase = time.monotonic(), "input_hash_verification"
    staging = OUT / (".release-staging-" + uuid.uuid4().hex)
    try:
        for mapping in (plan["input_sha256"], plan["source_sha256"], plan["protected_sha256"]):
            verify(mapping)
        staging.mkdir(exist_ok=False)
        phase = "two_saved_vectors_and_CSV_controls"
        vectors = {name: np.load(ROOT / path, allow_pickle=False) for name, path in VECTORS.items()}
        for values in vectors.values():
            need(values.dtype == np.dtype("float64") and values.shape == (47073,)
                 and np.isfinite(values).all() and (values >= 0).all(), "Invalid saved final vector")
        sample = pd.read_csv(ROOT / SAMPLE, encoding="utf-8-sig", usecols=[ID], dtype={ID: str}, keep_default_na=False)
        ids = sample[ID].tolist()
        need(len(ids) == 47073 and sample[ID].is_unique and all(ids), "Invalid sample IDs")
        csv_values = {}
        for name, path in CSVS.items():
            frame = pd.read_csv(ROOT / path, encoding="utf-8-sig", dtype={ID: str}, keep_default_na=False)
            need(list(frame.columns) == [ID, TARGET] and frame[ID].tolist() == ids and frame[ID].is_unique
                 and frame[TARGET].dtype == np.dtype("int64") and (frame[TARGET] > 0).all(), "Existing CSV schema/IDs differ")
            csv_values[name] = frame[TARGET].to_numpy()
            if name in vectors:
                need(np.array_equal(rounded(vectors[name]), csv_values[name]), "Original v4/v5 CSV does not replay")
        prediction = .5 * vectors["v4"] + .5 * vectors["v5"]
        prices = rounded(prediction)
        phase = "save_vector_and_CSV_roundtrip"
        with (staging / "predictions.npy").open("xb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        restored = np.load(staging / "predictions.npy", allow_pickle=False)
        need(restored.dtype == prediction.dtype and restored.shape == prediction.shape and restored.tobytes() == prediction.tobytes(), "Saved mean vector differs")
        staged_csv = staging / CSV.name
        pd.DataFrame({ID: ids, TARGET: prices}).to_csv(staged_csv, index=False, encoding="utf-8-sig", mode="x")
        with staged_csv.open("rb") as stream:
            need(stream.read(3) == b"\xef\xbb\xbf", "CSV BOM missing")
        back = pd.read_csv(staged_csv, encoding="utf-8-sig", dtype={ID: str}, keep_default_na=False)
        need(list(back.columns) == [ID, TARGET] and back[ID].tolist() == ids and back[ID].is_unique
             and back[TARGET].dtype == np.dtype("int64") and (back[TARGET] > 0).all()
             and np.array_equal(back[TARGET].to_numpy(), prices), "CSV roundtrip differs")
        for mapping in (plan["input_sha256"], plan["source_sha256"], plan["protected_sha256"]):
            verify(mapping)
        protected_after = {name: sha(ROOT / name) for name in PROTECTED}
        need(protected_after == plan["protected_sha256"], "Protected artifacts changed")
        proof = {"round": 38, "status": "experimental_trial_generated", "created_at_utc": now(),
                 "plan_sha256": sha(OUT / "plan.json"), "source_sha256": plan["source_sha256"],
                 "input_sha256": plan["input_sha256"], "runtime": runtime(), "formula": FORMULA, "query_rows": 47073,
                 "trial_only": True, "formal_acceptance": False, "recommended_version": "v3", "registry_updated": False,
                 "prior_research_decisions_changed": False, "planned_after_R38_results": True,
                 "two_unrounded_vectors_only": True, "v4_v5_CSV_reproduced_exactly": True,
                 "saved_prediction_bitwise_equal": True, "ID_order_verified": True, "positive_int64_roundtrip": True, "UTF8_BOM": True,
                 "rounded_prediction_changes_vs_CSV": {name: changes(prices, values) for name, values in csv_values.items()},
                 "unrounded_prediction_changes_vs_inputs": {name: changes(prediction, values) for name, values in vectors.items()},
                 "differences_are_not_errors": True, "models_trained_or_loaded": 0, "labels_or_features_loaded": False,
                 "test_mae": None, "public_mae": None, "private_mae": None, "uploaded_to_kaggle": False,
                 "path": rel(CSV), "sha256": sha(staged_csv), "rounding": "max(rint(TWD),1)",
                 "prediction_path": rel(DEST / "predictions.npy"), "prediction_sha256": sha(staging / "predictions.npy"),
                 "protected_before": plan["protected_sha256"], "protected_after": protected_after,
                 "elapsed_seconds": time.monotonic() - started}
        write(staging / "release.json", proof)
        phase = "publish_new_paths_only"
        need(not DEST.exists() and not CSV.exists() and staging.resolve().parent == OUT.resolve()
             and DEST.resolve().parent == OUT.resolve() and CSV.resolve().parent == (ROOT / "outputs").resolve(), "Unsafe/existing destination")
        staging.rename(DEST)
        (DEST / CSV.name).rename(CSV)
        need(sha(CSV) == proof["sha256"], "Published CSV differs")
        verify(plan["protected_sha256"])
        write(OUT / "execution.json", {"status": "completed", "exit_code": 0, "finished_at_utc": now(),
              "elapsed_seconds": time.monotonic() - started, "release_sha256": sha(DEST / "release.json"),
              "submission_sha256": sha(CSV), "protected_after_publication": {n: sha(ROOT / n) for n in PROTECTED}})
        print("experimental_trial_generated", rel(CSV), sha(CSV), flush=True)
        return 0
    except Exception as exc:
        write(OUT / "execution.json", {"status": "technical_failure_preserved", "exit_code": 1, "phase": phase,
              "finished_at_utc": now(), "elapsed_seconds": time.monotonic() - started, "staging_path": rel(staging),
              "error": str(exc), "traceback": traceback.format_exc(), "automatic_retry": False,
              "protected_after": {n: sha(ROOT / n) for n in PROTECTED}})
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
