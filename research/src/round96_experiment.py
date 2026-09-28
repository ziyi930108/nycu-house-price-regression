"""R96 fixed reference-confidence shrinkage, no native model load/fit/predict.

Only the parent-approved immutable request may launch the three saved-window
evaluations. Original reference/spatial states are reused, never refitted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round96_reference_shrink as helper

ROOT = old.ROOT
OUT = ROOT / "outputs/round96"
DOCUMENT = ROOT / "reports/round96_plan.md"
REQUEST = ROOT / "outputs/round96_launcher_20260929/research_request.json"
SELF_CHECK = ROOT / "reports/round96_helper_selfcheck.json"
STAGES = old.STAGES
sha, locked, write, rel, spatial = old.sha, old.locked, old.write, old.rel, old.spatial
FORMULA = "median(each original v22 branch spatial(base, Bfuture+.5*(reference-Bfuture), confidence*.5))"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def exact(a, b, label):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def vector(path, n):
    values = np.load(ROOT / path, allow_pickle=False)
    assert values.dtype == np.float64 and values.shape == (n,), path
    assert np.isfinite(values).all() and (values >= 0).all(), path
    return values


def save_array_checked(path, values):
    old.previous.save_array(path, values)
    exact(values, np.load(path, allow_pickle=False), "Saved array full roundtrip")


def state_digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def check_launch(plan_sha):
    assert sha(DOCUMENT) == plan_sha
    request = read_json(REQUEST)
    assert request["approval_for_three_evaluations"] is True
    assert request["plan_sha256"] == plan_sha
    frozen = request["frozen_source_and_plan"]
    required = (DOCUMENT, Path(__file__).resolve(), Path(helper.__file__).resolve(), SELF_CHECK)
    assert all(rel(p) in frozen for p in required), "Missing frozen source or self-check"
    for name, digest in frozen.items():
        assert sha(ROOT / name) == digest, "Frozen launch input changed: " + name
    parameters = helper.parameters()
    assert request["helper_parameters"] == parameters
    receipt = read_json(SELF_CHECK)
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0, "Helper CPU self-check did not pass"
    assert receipt["source_sha256"] == sha(Path(helper.__file__))
    assert receipt["plan_sha256"] == plan_sha and receipt["parameters"] == parameters
    assert receipt["source_hashes"] == helper.source_hashes()
    assert receipt["real_data_loaded"] is False and receipt["new_real_native_fits"] == 0
    # The root request binds the complete self-check bytes as well as final helper.
    # The helper's source dependencies must also belong to the closed launch map.
    for name, digest in helper.source_hashes().items():
        assert sha(ROOT / name) == digest == frozen[name], "Helper source not frozen"
    return request, frozen, parameters


def run(plan_sha):
    tick = time.monotonic()
    request, frozen, parameters = check_launch(plan_sha)
    inputs = {rel(REQUEST): sha(REQUEST)}

    def bind(name, digest):
        name = name.replace("\\", "/")
        assert name not in inputs or inputs[name] == digest, "Conflicting artifact SHA: " + name
        inputs[name] = digest

    def contract(name):
        path = ROOT / name
        sidecar = path.with_suffix(".sha256")
        assert path.is_file() and sidecar.is_file(), "Missing locked metadata: " + name
        value = locked(path)
        bind(name, sha(path))
        bind(rel(sidecar), sha(sidecar))
        return value

    records = {n: contract(f"outputs/round{n}/results.json") for n in (46, 48, 71, 79, 83)}
    plans = {n: contract(f"outputs/round{n}/plan.json") for n in (46, 48, 71, 79)}
    p79, r79, r83 = plans[79], records[79], records[83]
    assert r79["plan_sha256"] == sha(ROOT / "outputs/round79/plan.json")
    assert all(r["checked_stages"] == list(STAGES) for r in records.values())
    assert all(p["split_contract"] == p79["split_contract"] for p in plans.values())
    assert all(plans[n]["splits"] == p79["splits"] for n in (46, 48, 79))

    source = {}
    for previous in [*records.values(), *plans.values()]:
        for name, digest in previous["source_sha256"].items():
            assert name not in source or source[name] == digest, "Conflicting historical source: " + name
            assert sha(ROOT / name) == digest, "Historical source changed: " + name
            source[name] = digest
    for name, digest in helper.source_hashes().items():
        assert name not in source or source[name] == digest
        source[name] = digest
    source[rel(Path(__file__).resolve())] = sha(Path(__file__))
    for n in (46, 48, 79):
        for name, digest in plans[n]["input_sha256"].items():
            bind(name, digest)
    for name, digest in plans[71]["inherited_data_sha256"].items():
        bind(name, digest)

    branch_bindings = {}
    for stage in STAGES:
        spec = p79["split_contract"][stage]
        for n, record in records.items():
            item = record["stages"][stage]
            assert item["dates"] == p79["splits"][stage]
            assert item["fit_rows"] == spec["fit_rows"] and item["query_rows"] == spec["query_rows"]
            index_folder = f"outputs/round{n}/{stage}" + ("/model" if n == 46 else "")
            for kind in ("fit", "query"):
                name = f"{index_folder}/{kind}_indices.npy"
                bind(name, item["artifact_sha256"][name])
            if n in (79, 83):
                name = f"outputs/round{n}/{stage}/predictions.npy"
                assert item["prediction_sha256"] == item["artifact_sha256"][name]
                bind(name, item["prediction_sha256"])
        branch_bindings[stage] = {
            "nn": (71, f"outputs/round71/{stage}/seed42", "base.npy", "reference.npy", "nn_spatial.npy"),
            "log": (46, f"outputs/round46/{stage}", "base_recency_log.npy", "reference_recency_log.npy", "recency_log_spatial.npy"),
            "unit": (79, f"outputs/round79/{stage}", "base.npy", "reference.npy", "unit_spatial.npy"),
        }
        for n, prefix, basefile, reffile, spatialfile in branch_bindings[stage].values():
            for suffix in (basefile, reffile, spatialfile, "reference_diagnostics.pkl", "spatial_diagnostics.pkl"):
                name = prefix + "/" + suffix
                bind(name, records[n]["stages"][stage]["artifact_sha256"][name])
        name = f"outputs/round79/{stage}/model/metadata.json"
        bind(name, r79["stages"][stage]["artifact_sha256"][name])
        # These two metadata files have actual sidecars, checked before use.
        contract(name)
        contract(p79["bindings"][stage]["unit"]["state_summary"])
    for name, digest in {**inputs, **source, **frozen}.items():
        assert sha(ROOT / name) == digest, "Input/source changed before any arrays: " + name

    protected = {rel(p): sha(p) for p in sorted((ROOT / "outputs").glob("submission_v*.csv"))}
    for name in ("runs/versions.json", "outputs/current_private_candidate.json", "PRIVATE_SELECTION.md",
                 "CURRENT_STATUS.md", "outputs/round84/decision.json", "outputs/round79/selection/decision.json",
                 "outputs/v22_submission_focus_20260928/user_selection.json"):
        protected[name] = sha(ROOT / name)
    plan = dict(
        round=96, candidate="R96", formula=FORMULA, parameters=parameters, stage_order=list(STAGES),
        primary_control="v22", secondary_control="v24", planned_new_fits=0, new_NN_fits=0,
        new_native_tree_fits=0, text_plan_sha256=plan_sha, source_sha256=source, input_sha256=inputs,
        frozen_source_and_plan=frozen, launch_request_sha256=sha(REQUEST),
        branch_bindings=branch_bindings, split_contract=p79["split_contract"], splits=p79["splits"],
        target_group_weights=p79["target_group_weights"], protected_before=protected,
        query_labels_used_only_for_evaluation=True, helper_uses_query_labels=False,
        shared_loader_deserializes_canonical_y=True, adaptive_after_error_audit=True,
        fresh_blind_validation=False, Public_feedback_used_for_tuning=False, new_CSV=False,
        new_test_predictions=False, test_data_loaded=False, new_reference_or_spatial_states=0,
        automatic_adoption=False)
    write(OUT / "plan.json", plan)
    Xall, yall = old.previous.load_training_rows(p79)
    assert len(Xall) == 455255
    result = {k: v for k, v in plan.items() if k != "branch_bindings"}
    result.update(status="running", checked_stages=[], stages={}, new_fits=0, models_loaded=0,
                  native_models_loaded=0, model_predict_calls=0, reference_predict_calls=0,
                  spatial_predict_calls=0, reference_shrink_calls=0, saved_reference_state_loads=0,
                  saved_spatial_state_loads=0, plan_sha256=sha(OUT / "plan.json"),
                  runtime={"numpy": np.__version__, "pandas": pd.__version__})
    for stage in STAGES:
        stage_start = time.monotonic()
        folder = OUT / stage
        folder.mkdir()
        control_folder = folder / "control"
        control_folder.mkdir()
        binding = p79["bindings"][stage]
        unit_binding, spec = binding["unit"], p79["split_contract"][stage]
        indices = {}
        for kind in ("fit", "query"):
            values = np.load(ROOT / unit_binding[kind + "_indices"], allow_pickle=False)
            assert values.dtype == np.int64 and values.shape == (spec[kind + "_rows"],)
            assert (np.diff(values) > 0).all() and values.min() >= 0 and values.max() < len(Xall)
            assert old.original.value_sha(values, "<i8") == spec[kind + "_indices_content_sha256"]
            for n in records:
                name = f"outputs/round{n}/{stage}" + ("/model" if n == 46 else "") + f"/{kind}_indices.npy"
                exact(values, np.load(ROOT / name, allow_pickle=False), "All branch/control/R48 indices")
            indices[kind] = values
            save_array_checked(folder / (kind + "_indices.npy"), values)
        fi, qi = indices["fit"], indices["query"]
        assert not np.intersect1d(fi, qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        n = len(qi)
        cutoff = (pd.Timestamp(p79["splits"][stage][0]) - pd.Timestamp("2000-01-01")).days
        end = (pd.Timestamp(p79["splits"][stage][1]) - pd.Timestamp("2000-01-01")).days
        assert np.isfinite(X.trade_day_index).all() and (X.trade_day_index < cutoff).all()
        assert Xq.trade_day_index.between(cutoff, end).all()
        frames = {"fit_X": spatial.frame_sha(X), "query_X": spatial.frame_sha(Xq)}
        assert frames == r79["stages"][stage]["original_frame_sha256"]
        assert frames == records[71]["stages"][stage]["original_frame_sha256"]
        metadata = contract(f"outputs/round79/{stage}/model/metadata.json")
        assert frames == metadata["original_frame_sha256"]
        assert old.original.value_sha(yall[fi], "<f8") == metadata["fit_target_values_sha256"]
        geo = pd.read_pickle(ROOT / unit_binding["query_geo"])
        geo_before = spatial.frame_sha(geo)
        assert geo.index.equals(Xq.index)
        assert geo_before == unit_binding["manifest"]["features"]["query"]["frame_sha256"]
        with (ROOT / unit_binding["state"]).open("rb") as stream:
            state = pickle.load(stream)
        summary = contract(unit_binding["state_summary"])
        assert state["state_summary"] == summary and state["parameters"] == spatial.parameters()
        assert summary["fit_X_sha256"] == frames["fit_X"] and not summary["query_used_in_build"]
        assert summary["fit_geo_sha256"] == unit_binding["manifest"]["features"]["fit"]["frame_sha256"]
        with (ROOT / unit_binding["reference"]).open("rb") as stream:
            reference = pickle.load(stream)
        assert reference["fit_report"]["fitting_rows"] == len(fi)
        assert old.original.value_sha(reference["reference_residual"], "<f8") == summary["existing_reference_residual_sha256"]
        state_before = {"reference": state_digest(reference), "spatial": state_digest(state)}
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p79["splits"][stage][0], p79["target_group_weights"])

        originals, original_branch_metrics = {}, {}
        for branch, (_, prefix, basefile, reffile, spatialfile) in branch_bindings[stage].items():
            base, saved_ref, saved_spatial = (vector(prefix + "/" + name, n) for name in (basefile, reffile, spatialfile))
            saved_diag = pd.read_pickle(ROOT / prefix / "reference_diagnostics.pkl")
            saved_detail = pd.read_pickle(ROOT / prefix / "spatial_diagnostics.pkl")
            replay_ref, diag, rr = old.original.predict_reference(reference, Xq, base, variant="trend", return_diagnostics=True)
            replay_spatial, detail, sr = spatial.predict(state, Xq, geo, base, replay_ref, diag)
            exact(replay_ref, saved_ref, branch + " complete reference replay")
            exact(replay_spatial, saved_spatial, branch + " complete spatial replay")
            pd.testing.assert_frame_equal(diag, saved_diag, check_exact=True)
            pd.testing.assert_frame_equal(detail, saved_detail, check_exact=True)
            originals[branch] = dict(base=base, reference=replay_ref, spatial=replay_spatial,
                                     diagnostics=diag, spatial_diagnostics=detail, reference_report=rr, spatial_report=sr)
            original_branch_metrics[branch] = {name: score(values) for name, values in
                                               (("base", base), ("reference", replay_ref), ("spatial", replay_spatial))}
            for name in ("base", "reference", "spatial"):
                save_array_checked(control_folder / (branch + "_" + name + ".npy"), originals[branch][name])

        fixed = {key: vector(binding["paths"][key], n) for key in ("v11", "v12", "v5")}
        fixed["v22"] = vector(f"outputs/round79/{stage}/predictions.npy", n)
        fixed["v24"] = vector(f"outputs/round83/{stage}/predictions.npy", n)
        exact(originals["nn"]["spatial"], vector(binding["paths"]["nn"], n), "Original selected NN")
        exact(originals["log"]["spatial"], vector(binding["paths"]["log"], n), "Original selected log")
        exact(np.median(np.column_stack([originals[k]["spatial"] for k in ("nn", "log", "unit")]), axis=1),
              fixed["v22"], "Complete original v22 median")
        metrics = {key: score(values) for key, values in fixed.items()}
        for key in ("v11", "v12", "v5"):
            assert metrics[key] == r79["stages"][stage]["metrics"][key]
        assert metrics["v22"] == r79["stages"][stage]["metrics"]["R79"]
        assert metrics["v24"] == r83["stages"][stage]["metrics"]["R83"]
        for key, values in fixed.items():
            save_array_checked(control_folder / (key + ".npy"), values)
        write(folder / "control_verification.json", {
            "all_original_reference_spatial_vectors_and_diagnostics_exact": True,
            "original_v22_median_exact": True, "controls_full_metrics_exact": True,
            "old_model_load_or_predict_calls": 0, "original_control_vectors_saved": True})
        for name, digest in {**source, **frozen}.items():
            assert sha(ROOT / name) == digest

        candidates, branch_metrics, diagnostics, zero_confidence = {}, {}, {}, {}
        invariant_detail_columns = [c for c in originals["nn"]["spatial_diagnostics"].columns
                                    if c not in ("v2_confidence", "effective_remaining_weight", "price_adjustment")]
        for branch in ("nn", "log", "unit"):
            branch_folder = folder / branch
            branch_folder.mkdir()
            previous = originals[branch]
            base, original_ref, diag = previous["base"], previous["reference"], previous["diagnostics"]
            before_base, before_ref, before_diag = base.copy(), original_ref.copy(), diag.copy(deep=True)
            new_ref, new_diag, shrink_report = helper.shrink_reference(reference, Xq, base, original_ref, diag)
            exact(base, before_base, "Helper did not mutate base")
            exact(original_ref, before_ref, "Helper did not mutate original reference")
            pd.testing.assert_frame_equal(diag, before_diag, check_exact=True)
            expected_confidence = diag["confidence"].to_numpy(dtype=np.float64) * .5
            exact(new_diag["confidence"].to_numpy(), expected_confidence, "Reference confidence halved")
            pd.testing.assert_frame_equal(new_diag.drop(columns="confidence"), diag.drop(columns="confidence"), check_exact=True)
            no_conf = diag["confidence"].to_numpy(dtype=np.float64) == 0
            exact(new_ref[no_conf], original_ref[no_conf], "Zero confidence reference unchanged")
            branch_prediction, detail, spatial_report = spatial.predict(state, Xq, geo, base, new_ref, new_diag)
            exact(detail["v2_confidence"].to_numpy(), expected_confidence, "Spatial consumes new confidence")
            pd.testing.assert_frame_equal(detail[invariant_detail_columns],
                                          previous["spatial_diagnostics"][invariant_detail_columns], check_exact=True)
            fallback = ~detail["applied"].to_numpy(dtype=bool)
            exact(branch_prediction[fallback], new_ref[fallback], "New spatial fallback")
            exact(branch_prediction[no_conf], previous["spatial"][no_conf], "Zero confidence full branch unchanged")
            candidates[branch], zero_confidence[branch] = branch_prediction, no_conf
            branch_metrics[branch] = {"base": original_branch_metrics[branch]["base"],
                                      "reference": score(new_ref), "spatial": score(branch_prediction)}
            future = helper.future_base(reference, Xq, base)
            for name, values in (("base_future", future), ("reference", new_ref), ("spatial", branch_prediction)):
                save_array_checked(branch_folder / (name + ".npy"), values)
            for name, frame in (("reference_diagnostics", new_diag), ("spatial_diagnostics", detail)):
                with (branch_folder / (name + ".pkl")).open("xb") as stream:
                    pickle.dump(frame, stream, protocol=5)
                pd.testing.assert_frame_equal(frame, pd.read_pickle(branch_folder / (name + ".pkl")), check_exact=True)
            diagnostics[branch] = {
                "reference_shrink": shrink_report, "spatial_report": spatial_report,
                "zero_confidence_rows": int(no_conf.sum()), "zero_confidence_full_branch_bitwise_unchanged": True,
                "changed_reference_rows": int(np.count_nonzero(new_ref != original_ref)),
                "changed_spatial_rows": int(np.count_nonzero(branch_prediction != previous["spatial"])),
                "reference_confidence_exactly_halved": True, "spatial_used_new_confidence": True,
                "spatial_support_and_future_trend_diagnostics_unchanged": True,
                "mean_original_confidence": float(diag["confidence"].mean()),
                "mean_candidate_confidence": float(new_diag["confidence"].mean()),
                "mean_original_spatial_effective_weight": float(previous["spatial_diagnostics"]["effective_remaining_weight"].mean()),
                "mean_candidate_spatial_effective_weight": float(detail["effective_remaining_weight"].mean())}
        predictions = np.median(np.column_stack([candidates[k] for k in ("nn", "log", "unit")]), axis=1)
        assert predictions.dtype == np.float64 and predictions.shape == (n,)
        assert np.isfinite(predictions).all() and (predictions >= 0).all()
        all_zero = zero_confidence["nn"] & zero_confidence["log"] & zero_confidence["unit"]
        exact(predictions[all_zero], fixed["v22"][all_zero], "All-branch zero confidence final unchanged")
        metrics["R96"] = score(predictions)
        save_array_checked(folder / "predictions.npy", predictions)
        assert frames == {"fit_X": spatial.frame_sha(X), "query_X": spatial.frame_sha(Xq)}
        assert spatial.frame_sha(geo) == geo_before
        assert state_before == {"reference": state_digest(reference), "spatial": state_digest(state)}, "Saved fit states mutated"
        differences = old.previous.prior.log_run.differences
        item = dict(
            stage=stage, status="completed", fit_rows=len(fi), query_rows=n, dates=p79["splits"][stage],
            original_frame_sha256=frames, metrics=metrics,
            differences={key: differences(metrics[key], metrics["R96"]) for key in metrics if key != "R96"},
            branch_metrics=branch_metrics, original_branch_metrics=original_branch_metrics,
            branch_diagnostics=diagnostics, original_reference_spatial_replay_exact=True,
            original_diagnostics_replay_exact=True, original_v22_median_exact=True,
            controls_full_metrics_exact=True, original_control_vectors_saved=True,
            saved_states_and_query_frames_unchanged=True, saved_arrays_full_roundtrip_exact=True,
            saved_candidate_diagnostics_roundtrip_exact=True,
            all_branch_zero_confidence_rows=int(all_zero.sum()),
            all_branch_zero_confidence_final_bitwise_unchanged=True,
            changed_final_rows=int(np.count_nonzero(predictions != fixed["v22"])),
            query_labels_used_only_for_evaluation=True, new_fits=0, new_NN_fits=0, new_native_tree_fits=0,
            models_loaded=0, native_models_loaded=0, model_predict_calls=0, new_reference_or_spatial_states=0,
            saved_reference_state_loads=1, saved_spatial_state_loads=1,
            reference_predict_calls=3, spatial_predict_calls=6, reference_shrink_calls=3,
            elapsed_seconds=time.monotonic() - stage_start, prediction_sha256=sha(folder / "predictions.npy"),
            artifact_sha256={rel(p): sha(p) for p in folder.rglob("*") if p.is_file()})
        write(folder / "results.json", item)
        result["stages"][stage] = item
        result["checked_stages"].append(stage)
        for name in ("reference_predict_calls", "spatial_predict_calls", "reference_shrink_calls",
                     "saved_reference_state_loads", "saved_spatial_state_loads"):
            result[name] += item[name]
        print(json.dumps({"stage": stage, "R96": metrics["R96"]["overall_mae"],
                          "v22": metrics["v22"]["overall_mae"], "v24": metrics["v24"]["overall_mae"],
                          "weighted": metrics["R96"]["weighted_mae"]}), flush=True)
    assert result["checked_stages"] == list(STAGES)
    for name, digest in {**inputs, **source, **frozen, **protected}.items():
        assert sha(ROOT / name) == digest, "Dependency changed after evaluation: " + name
    result.update(status="completed", elapsed_seconds=time.monotonic() - tick,
                  protected_after={name: sha(ROOT / name) for name in protected}, automatic_adoption=False)
    write(OUT / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("--run required; no implicit evaluation")
    assert sha(DOCUMENT) == args.plan_sha and not OUT.exists(), "Plan changed or output exists"
    OUT.mkdir()
    started, tick, failure = old.now(), time.monotonic(), None
    write(OUT / "started.json", dict(started_at_utc=started, source_sha256=sha(Path(__file__)),
                                   command=sys.argv, automatic_retry=False))
    try:
        run(args.plan_sha)
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
    write(OUT / "execution.json", dict(started_at_utc=started, completed_at_utc=old.now(),
                                      elapsed_seconds=time.monotonic() - tick, exit_code=1 if failure else 0,
                                      failure=failure, source_sha256=sha(Path(__file__)), automatic_retry=False))
    raise SystemExit(1 if failure else 0)

