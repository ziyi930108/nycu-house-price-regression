"""Freeze implementation and evaluation contracts before any real R98 meta fit."""
from pathlib import Path
import ast
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import round98_contract as c
import round98_l1_stacker as helper
import round98_temporal_evaluate as evaluator
HERE = Path(__file__).resolve().parent


def merge(*mappings):
    result = {}
    for m in mappings:
        for k, v in m.items():
            c.need(k not in result or result[k] == v, "Conflicting frozen source")
            result[k] = v
    return result


def run():
    c.need(not (ROOT / "outputs/round98").exists(), "Actual fits already started")
    c.need(not (HERE / "research_request.json").exists() and not (HERE / "evaluation_request.json").exists(), "Freeze is immutable")
    c.need(c.sha(c.PLAN) == c.PLAN_SHA, "Plan changed")
    helper_check = c.read("reports/round98_helper_selfcheck.json")
    scorer_check = c.read("reports/round98_evaluator_selfcheck.json")
    c.need(helper_check["status"] == "passed" and helper_check["check_count"] == 42
           and helper_check["source_sha256"] == c.sha("src/round98_l1_stacker.py"), "Helper checks/source differ")
    c.need(scorer_check["status"] == "passed" and scorer_check["source_sha256"] == c.sha("src/round98_temporal_evaluate.py"), "Scorer checks/source differ")
    c.need(helper_check["parameters"] == helper.parameters(), "Helper method differs from checks")
    solver_sources = helper.source_hashes()
    c.verify_hashes(solver_sources)
    driver_sources = c.source_hashes("src/round98_temporal_stack.py")
    eval_sources = evaluator.source_hashes()
    sources = merge(driver_sources, eval_sources, c.source_hashes("src/round98_prepare_inputs.py"), solver_sources,
                    c.source_hashes(__file__, HERE / "run_pipeline.py", HERE / "run_evaluation.py"))
    for name in (c.PLAN, "outputs/round98_artifact_inventory/audit.py", c.INVENTORY,
                 "reports/round98_helper_selfcheck.json", "reports/round98_evaluator_selfcheck.json",
                 "reports/round98_helper_design.md", "reports/round98_validation_contract_review.md",
                 "reports/round98_artifact_inventory.md"):
        sources = merge(sources, {name: c.sha(name)})
    for name in sources:
        if name.endswith(".py") and not name.startswith(".venv"):
            ast.parse(c.safe(name).read_text(encoding="utf-8-sig"), filename=name)
    prep = c.locked("outputs/round98_inputs/manifest.json")
    c.need(prep["status"] == "prepared_temporal_meta_inputs" and prep["plan_sha256"] == c.PLAN_SHA, "Preparation incomplete")
    c.need(c.locked("outputs/round98_inputs/execution.json")["exit_code"] == 0, "Preparation did not exit 0")
    preparation = {"path": "outputs/round98_inputs/manifest.json", "sha256": c.sha("outputs/round98_inputs/manifest.json")}
    protected = c.protected_hashes()
    c.need(protected == prep["protected_before"] == prep["protected_after"], "Protected outputs differ")
    request = {"schema_version": 1, "status": "research_frozen_before_meta_fit", "created_at_utc": c.now(),
               "plan_sha256": c.PLAN_SHA, "stage_order": list(c.STAGES), "helper_parameters": helper.parameters(),
               "driver_source_sha256": driver_sources, "source_sha256": sources, "preparation": preparation,
               "protected_before": protected, "new_native_fits_planned": 0, "new_meta_fits_planned": 3,
               "independent_stage_processes_required": True, "query_scoring_allowed_before_all_sealed": False,
               "selected_submission": "v22", "new_CSV": False, "Private_score": None}
    c.write(HERE / "research_request.json", request)
    evaluation = {"schema_version": 1, "status": "evaluation_frozen_before_query_score", "created_at_utc": c.now(),
                  "parameters": evaluator.parameters(), "model_parameters": helper.parameters(),
                  "source_sha256": eval_sources, "plan_sha256": c.PLAN_SHA,
                  "research_request": {"path": c.rel(HERE / "research_request.json"), "sha256": c.sha(HERE / "research_request.json")},
                  "frozen_before_any_real_meta_fit": True}
    c.write(HERE / "evaluation_request.json", evaluation)
    print(json.dumps({"research_request_sha256": c.sha(HERE / "research_request.json"),
                      "evaluation_request_sha256": c.sha(HERE / "evaluation_request.json"),
                      "frozen_files": len(sources), "driver_import_closure": len(driver_sources),
                      "scorer_import_closure": len(eval_sources), "new_fits": 0}))


if __name__ == "__main__":
    run()
