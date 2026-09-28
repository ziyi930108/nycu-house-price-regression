"""Coordinate three separate R98 CPU meta-fit processes, then seal; no scoring."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import round98_contract as c

HERE = Path(__file__).resolve().parent
REQUEST = HERE / "research_request.json"
OUT = ROOT / "outputs/round98"


def run(expected_request_sha):
    c.need(not OUT.exists(), "R98 run already exists; preserve all artifacts")
    c.need(c.sha(REQUEST) == expected_request_sha, "Request identity")
    request = c.locked(REQUEST)
    c.need(request["status"] == "research_frozen_before_meta_fit", "Request not frozen")
    c.verify_hashes(request["source_sha256"])
    c.need(c.protected_hashes() == request["protected_before"], "Protected outputs changed")
    OUT.mkdir()
    start, tick, code, failure = c.now(), time.monotonic(), 1, None
    completed = []
    c.write(OUT / "started.json", {"started_at_utc": start, "request_sha256": expected_request_sha,
            "coordinator_pid": os.getpid(), "stages": list(c.STAGES), "GPU_used": False})
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        stages = {}
        for s in c.STAGES:
            st = time.monotonic()
            cmd = [sys.executable, "-B", "-X", "utf8", "src/round98_temporal_stack.py", "--stage", s,
                   "--request-sha", expected_request_sha]
            with (HERE / (s + ".log")).open("x", encoding="utf8") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                c.write(HERE / (s + "_started.json"), {"started_at_utc": c.now(), "pid": proc.pid, "stage": s,
                        "command": cmd, "separate_stage_process": True})
                exit_code = proc.wait()
            c.write(HERE / (s + "_execution.json"), {"stage": s, "exit_code": exit_code,
                    "elapsed_seconds": time.monotonic() - st, "completed_at_utc": c.now(), "automatic_retry": False})
            c.need(exit_code == 0, "Stage process failed: " + s)
            sp, ep = OUT / s / "stage_seal.json", OUT / s / "execution.json"
            sm, execution = c.locked(sp), c.locked(ep)
            c.need(sm["status"] == "temporal_stack_stage_predictions_sealed" and execution["exit_code"] == 0,
                   "Stage not complete: " + s)
            c.need(sm["stage"] == s and sm["plan_sha256"] == c.PLAN_SHA and sm["query_labels_available"] is False
                   and sm["all_query_reload_exact"] is True and sm["new_meta_fits"] == 1 and sm["new_native_fits"] == 0,
                   "Stage contract differs")
            c.need(sm["parameters"] == request["helper_parameters"] and sm["protected_before"] == sm["protected_after"] == request["protected_before"], "Stage method/protection differs")
            stages[s] = {"stage_manifest": {"path": c.rel(sp), "sha256": c.sha(sp)},
                         "execution": {"path": c.rel(ep), "sha256": c.sha(ep)},
                         "predictions": sm["predictions"], "model_state": sm["model_state"]}
            completed.append(s)
            print(json.dumps({"completed_stage": s, "outer_seconds": time.monotonic() - st}), flush=True)
        c.verify_hashes(request["source_sha256"])
        c.need(c.protected_hashes() == request["protected_before"], "Protected outputs changed at seal")
        c.write(OUT / "sealed_manifest.json", {"schema_version": 1, "status": "all_temporal_stack_predictions_sealed",
                "stage_order": list(c.STAGES), "stages": stages, "new_meta_fits": 3, "new_native_fits": 0,
                "query_scores_computed": False, "all_query_reload_exact": True, "plan_sha256": c.PLAN_SHA,
                "preparation": request["preparation"], "source_sha256": request["source_sha256"],
                "launch_request": {"path": c.rel(REQUEST), "sha256": expected_request_sha},
                "protected_before": request["protected_before"], "protected_after": c.protected_hashes(),
                "separate_stage_processes": True, "cross_stage_target_role_overlap_documented": True,
                "selected_submission": "v22", "new_CSV": False, "Private_score": None,
                "started_at_utc": start, "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick})
        code = 0
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(OUT / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": start,
                "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick,
                "completed_stages": completed, "scoring_started": False, "new_native_fits": 0,
                "all_stage_predictions_sealed": code == 0, "automatic_retry": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-sha", required=True)
    run(parser.parse_args().request_sha)
