"""Run one sealed R98 evaluation process after all stage exits succeed."""
from pathlib import Path
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
REQUEST = HERE / "evaluation_request.json"


def run(expected):
    c.need(c.sha(REQUEST) == expected, "Frozen evaluation request differs")
    request = c.locked(REQUEST)
    c.need(request["parameters"] == c.evaluation_parameters() and request["plan_sha256"] == c.PLAN_SHA == c.sha(c.PLAN), "Scoring plan differs")
    c.verify_hashes(request["source_sha256"])
    execution = c.locked("outputs/round98/execution.json")
    c.need(execution["exit_code"] == 0 and execution["all_stage_predictions_sealed"] is True, "Main must finish successfully")
    c.need(not (HERE / "evaluation_execution.json").exists(), "Prior evaluation exists; no retry")
    start, tick = c.now(), time.monotonic()
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-X", "utf8", "src/round98_temporal_evaluate.py",
               "--sealed-manifest", "outputs/round98/sealed_manifest.json", "--evaluation-request", c.rel(REQUEST),
               "--out", "outputs/round98/evaluation"]
    code, failure = 1, None
    try:
        with (HERE / "evaluation.log").open("x", encoding="utf8") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            c.write(HERE / "evaluation_started.json", {"started_at_utc": start, "pid": proc.pid,
                    "request_sha256": expected, "command": command})
            code = proc.wait()
        c.need(code == 0, "Evaluation failed; read evaluation.log")
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(HERE / "evaluation_execution.json", {"exit_code": code, "failure": failure,
                "started_at_utc": start, "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick,
                "automatic_retry": False})
    print(json.dumps({"exit_code": code, "elapsed_seconds": time.monotonic() - tick}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request-sha", required=True)
    run(p.parse_args().request_sha)
