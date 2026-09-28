"""R97 sealed-prediction evaluator for the fixed 2023-02-11 forecast origin.

No fitting, model loading, inference, tuning, test file or GPU operation.
Real query prices are deserialized only after the complete seal is verified.
The --self-check path uses only artificial arrays and temporary small files.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import round90_offset_data as data_helper
import v3_cluster_stability as methods

ROOT = Path(__file__).resolve().parents[1]
PLAN = "reports/round97_plan.md"
PREPARATION_SHA = "400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7"
RAW_PATH = "data_cache/train.pkl"
WEIGHTS = {"seen": 0.36617593949822613, "new_nonmissing": 0.6338240605017739, "missing": 0.0}
BANDS = (("<10m", 0.0, 1e7), ("10-20m", 1e7, 2e7), ("20-50m", 2e7, 5e7),
         ("50-100m", 5e7, 1e8), (">=100m", 1e8, np.inf))
BRANCHES = ("N", "H", "U22", "U24")
NATIVES = ("raw", "light", "log", "unit42", "unit137", "N_area", "N_prior")
LAYERS = ("base", "reference", "spatial")
CAUTION = ("Fixed saved predictions on a new forecast origin using historically reused, overlapping data. "
           "Not a blind holdout, a Private-score probability or a selection-adjusted test. "
           "Bootstrap excludes retraining, accumulated model selection and future distribution shift. "
           "True-price bands and error tails are descriptive only and cannot route predictions.")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT), "Path outside workspace: " + str(path))
    return p


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    p = Path(path)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    with p.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(p) + "\n")


def parameters():
    return {"round": 97, "cutoff": "2023-02-11", "query_end": "2024-02-10",
            "fit_rows": 272117, "query_rows": 107169, "raw_rows": 455784,
            "control": "v22", "candidate": "v24", "fixed_group_weights": WEIGHTS.copy(),
            "seed": 20260925, "replicates": 2000, "stage_index": 0,
            "family_order": ["project", "month"], "month_block_length": 3,
            "month_blocks": 4, "rng": "default_rng(SeedSequence([20260925,0,family_index]))",
            "quarters": {"Q1": [0, 91], "Q2": [92, 183], "Q3": [184, 275], "Q4": [276, None]},
            "price_bands": [b[0] for b in BANDS], "tail_quantiles": [.5, .9, .95, .99],
            "error_tail_top_rows": 10, "error_tail_top_fraction": .01,
            "selection_changed": False, "test_read": False, "new_CSV": False}


def source_hashes():
    """Bind the evaluator's actual local import closure without running loaders."""
    pending = [Path(__file__).resolve()]
    found = set()
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                q = ROOT / "src" / (name.split(".")[0] + ".py")
                if q.is_file() and q not in found:
                    pending.append(q)
    return {rel(p): sha(p) for p in sorted(found)}


def check_header(m):
    """First gate, called before any artifact or label loader."""
    need(m.get("schema_version") == 1, "Unknown sealed manifest schema")
    need(m.get("status") == "sealed_predictions_without_query_labels", "Predictions not sealed")
    need(m.get("all_fits_complete") is True and m.get("predictions_sealed") is True,
         "Seven fits and all predictions must be complete")
    need(m.get("new_native_tree_fits") == 5 and m.get("new_NN_fits") == 2, "Wrong native fit count")
    need(m.get("query_labels_available") is False, "Driver must not have query labels")
    need(m.get("all_query_saved_model_and_state_reload_exact") is True
         and m.get("input_frames_unchanged") is True, "Model/state full-query reload and input preservation required")
    need(m.get("new_query_metrics_computed") is False and m.get("scores") is None,
         "Seal must precede query scoring")
    need(m.get("selected_submission") == "v22" and m.get("new_CSV") is False,
         "Evaluation cannot change selected submission or export CSV")
    need(m.get("protected_before") == m.get("protected_after") and isinstance(m.get("protected_before"), dict),
         "Protected artifacts changed during model reconstruction")
    need(m.get("cutoff") == "2023-02-11" and m.get("query_end") == "2024-02-10", "Wrong fixed origin")
    need(set(m.get("predictions", {})) == {"v22", "v24"}, "Exactly two full fixed versions required")
    need(set(m.get("branches", {})) == set(BRANCHES), "Four fixed branches required")
    need(set(m.get("native_predictions", {})) == set(NATIVES), "Seven native vectors required")
    need(all(set(m["branches"][b]) == set(LAYERS) for b in BRANCHES), "Three layers per branch required")


def check_evaluation_request(request, sources, plan_sha):
    need(request.get("status") == "evaluation_frozen_before_query_score", "Evaluator not separately frozen")
    need(request.get("parameters") == parameters(), "Fixed scoring parameters differ")
    need(request.get("source_sha256") == sources, "Frozen evaluator source closure differs")
    need(request.get("plan_sha256") == plan_sha, "Evaluation plan changed")


def merge_hashes(*maps):
    result = {}
    for mapping in maps:
        for name, digest in mapping.items():
            key = rel(safe(name))
            need(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 "Invalid SHA: " + key)
            need(key not in result or result[key] == digest, "Conflicting SHA: " + key)
            result[key] = digest
    return result


def verify_hashes(mapping):
    for name, digest in mapping.items():
        need(sha(safe(name)) == digest, "Immutable input changed: " + name)


def load_array(record, rows, dtype):
    p = safe(record["path"])
    need(sha(p) == record["sha256"], "Array file SHA differs: " + str(p))
    a = np.load(p, allow_pickle=False)
    need(a.dtype == np.dtype(dtype) and a.shape == (rows,), "Array dtype/shape differs: " + str(p))
    need(record["dtype"] == str(a.dtype) and record["shape"] == list(a.shape)
         and record["values_sha256"] == values_sha(a), "Array semantic identity differs")
    if dtype == "float64":
        need(np.isfinite(a).all() and (a >= 0).all(), "Invalid TWD predictions")
    else:
        need((a >= 0).all() and (np.diff(a) > 0).all(), "Indices must be sorted unique nonnegative")
    return a


def exact(a, b, message):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), message)


def check_formulas(native, branch, full):
    expected = {"N": .75 * (.5 * native["N_area"] + .5 * native["N_prior"]) + .25 * native["light"],
                "H": .75 * (.5 * native["raw"] + .5 * native["log"]) + .25 * native["light"],
                "U22": .75 * native["unit42"] + .25 * native["light"],
                "U24": .5 * native["unit42"] + .5 * native["unit137"]}
    for name in BRANCHES:
        exact(branch[name]["base"], expected[name], "Fixed base formula differs: " + name)
    for name, unit in (("v22", "U22"), ("v24", "U24")):
        p = np.median(np.column_stack([branch[b]["spatial"] for b in ("N", "H", unit)]), axis=1)
        exact(p, full[name], "Fixed three-branch median differs: " + name)


def preflight(sealed_manifest, evaluation_request):
    """Verify every bound input/prediction and pure-X alignment; no prices deserialize."""
    path = safe(sealed_manifest)
    digest = sha(path)
    need(path.with_suffix(".sha256").read_text(encoding="ascii").strip() == digest, "Seal receipt differs")
    m = read(path)
    check_header(m)
    own = source_hashes()
    request_path = safe(evaluation_request)
    request_digest = sha(request_path)
    need(request_path.with_suffix(".sha256").read_text(encoding="ascii").strip() == request_digest, "Evaluation request receipt differs")
    request = read(request_path)
    check_evaluation_request(request, own, sha(ROOT / PLAN))
    need(m["plan_sha256"] == request["plan_sha256"], "Driver/evaluator fixed plan differs")
    execution_path = path.parent / "execution.json"
    execution = read(execution_path)
    need(execution["exit_code"] == 0 and execution["sealed_manifest_created"] is True
         and execution["query_labels_available"] is False, "Driver must exit successfully before price access")
    prep = m["preparation_manifest"]
    need(prep["sha256"] == PREPARATION_SHA, "Wrong target-free preparation")
    records = list(m["arrays"].values()) + list(m["frames"].values()) + list(m["predictions"].values())
    records += list(m["native_predictions"].values())
    records += [m["branches"][b][layer] for b in BRANCHES for layer in LAYERS]
    hashes = merge_hashes(m["input_sha256"], m["source_sha256"], m["artifact_sha256"],
                         {r["path"]: r["sha256"] for r in records}, {prep["path"]: prep["sha256"]},
                         {rel(path): digest, rel(path.with_suffix(".sha256")): sha(path.with_suffix(".sha256")),
                          rel(request_path): request_digest, rel(request_path.with_suffix(".sha256")): sha(request_path.with_suffix(".sha256")),
                          rel(execution_path): sha(execution_path), PLAN: request["plan_sha256"]}, own)
    need(RAW_PATH in hashes, "Original raw train SHA required")
    verify_hashes(hashes)
    d = data_helper.load_prepared(safe(prep["path"]).parent, PREPARATION_SHA)
    need(len(d.fit_indices) == 272117 and len(d.query_indices) == 107169, "Fixed split rows differ")
    need(hashes[RAW_PATH] == d.contract["inherited_provenance"]["raw_train_sha256"], "Raw price source identity differs")
    need(set(m["arrays"]) == set(data_helper.ARRAY_NAMES), "Index manifest keys differ")
    need(set(m["frames"]) == set(data_helper.FRAME_NAMES), "Frame manifest keys differ")
    for name in data_helper.ARRAY_NAMES:
        expected = getattr(d, name)
        a = load_array(m["arrays"][name], len(expected), "int64")
        exact(a, expected, "Driver/preparation index differs: " + name)
    for name in data_helper.FRAME_NAMES:
        rec = m["frames"][name]
        need(rec["sha256"] == d.contract["artifacts"][name]["sha256"]
             and rec["frame_sha256"] == d.contract["frames"][name]["frame_sha256"], "Driver/preparation frame differs: " + name)
    mapping_path = data_helper.CANONICAL
    need(hashes.get(mapping_path) == d.contract["input_sha256"][mapping_path], "Canonical mapping not bound")
    mapping = np.load(safe(mapping_path), allow_pickle=False)
    for role in ("fit", "query"):
        canonical, raw = getattr(d, role + "_indices"), getattr(d, role + "_source_cache_indices")
        exact(mapping[canonical], raw, "Canonical/raw source mapping differs: " + role)
    n = len(d.query_indices)
    native = {k: load_array(m["native_predictions"][k], n, "float64") for k in NATIVES}
    branch = {b: {layer: load_array(m["branches"][b][layer], n, "float64") for layer in LAYERS} for b in BRANCHES}
    full = {k: load_array(m["predictions"][k], n, "float64") for k in ("v22", "v24")}
    check_formulas(native, branch, full)
    return {"manifest": m, "manifest_path": rel(path), "manifest_sha256": digest, "data": d,
            "native": native, "branches": branch, "predictions": full, "input_sha256": hashes,
            "source_sha256": own, "seal_verified_without_price_deserialization": True}


def select_query_y(raw, indices, expected_rows):
    """Select raw-cache positions, never canonical positions or reset query ordinals."""
    a = np.asarray(indices)
    need(isinstance(raw, pd.DataFrame) and len(raw) == expected_rows and "總價元" in raw, "Wrong raw train table")
    need(a.dtype == np.int64 and a.ndim == 1 and len(a) > 0 and (a >= 0).all()
         and (a < len(raw)).all() and (np.diff(a) > 0).all(), "Invalid raw-cache query mapping")
    y = raw.iloc[a]["總價元"].to_numpy(dtype=np.float64, copy=True)
    need(np.isfinite(y).all() and (y > 0).all(), "Invalid query prices")
    return y


def load_query_y(checked):
    need(checked.get("seal_verified_without_price_deserialization") is True, "Price access requires verified seal")
    p = ROOT / RAW_PATH
    need(sha(p) == checked["input_sha256"][RAW_PATH], "Price source changed after seal verification")
    raw = pd.read_pickle(p)
    checked["raw_price_pickle_deserialized"] = True
    y = select_query_y(raw, checked["data"].query_source_cache_indices, 455784)
    del raw
    return y


def descriptive_metrics(y, prediction, fit, query):
    need(y.shape == prediction.shape == (len(query),), "Scoring alignment differs")
    need(np.isfinite(y).all() and (y > 0).all() and np.isfinite(prediction).all()
         and (prediction >= 0).all(), "Invalid scoring values")
    result = methods.original_score(y, prediction, fit, query, "2023-02-11", WEIGHTS)
    residual = prediction - y
    error = np.abs(residual)
    total = float(error.sum())
    bands = []
    for name, lo, hi in BANDS:
        mask = (y >= lo) & (y < hi)
        count = int(mask.sum())
        bands.append({"band": name, "rows": count,
                      "mae": float(error[mask].mean()) if count else None,
                      "mean_signed_error_prediction_minus_truth": float(residual[mask].mean()) if count else None,
                      "absolute_error_sum": float(error[mask].sum()),
                      "absolute_error_share": float(error[mask].sum() / total) if total else None,
                      "underprediction_fraction": float((residual[mask] < 0).mean()) if count else None})
    need(sum(b["rows"] for b in bands) == len(y), "Price bands are not a partition")
    descending = np.sort(error)[::-1]
    top_rows = min(10, len(error))
    top_fraction_rows = max(1, int(np.ceil(.01 * len(error))))
    result["price_bands"] = bands
    result["tails"] = {"mean_signed_error_prediction_minus_truth": float(residual.mean()),
                       "underprediction_fraction": float((residual < 0).mean()),
                       "absolute_error_sum": total, "maximum_absolute_error": float(error.max()),
                       "absolute_error_quantiles": {str(q): float(np.quantile(error, q, method="linear")) for q in (.5, .9, .95, .99)},
                       "top10_error_rows": top_rows, "top10_absolute_error_sum": float(descending[:top_rows].sum()),
                       "top10_error_share": float(descending[:top_rows].sum() / total) if total else None,
                       "top1pct_error_rows": top_fraction_rows,
                       "top1pct_absolute_error_sum": float(descending[:top_fraction_rows].sum()),
                       "top1pct_error_share": float(descending[:top_fraction_rows].sum() / total) if total else None,
                       "error_tail_membership": "Per-model largest absolute errors; not shared row IDs or prediction rules"}
    months = methods.relative_months(query.trade_day_index, "2023-02-11")
    result["relative_forecast_months"] = {str(i): {"rows": int((months == i).sum()),
        "mae": float(error[months == i].mean()) if (months == i).any() else None} for i in range(12)}
    return result


def paired(y, predictions, fit, query, canonical_query_indices):
    labels = methods.group_labels(fit, query)
    errors = np.abs(y[:, None] - np.column_stack([predictions[k] for k in ("v22", "v24")]))
    point = methods.point_values(errors, labels, WEIGHTS)
    keys, fallback = methods.cluster_keys(query, canonical_query_indices)
    months = methods.relative_months(query.trade_day_index, "2023-02-11")
    result = {"control": "v22", "candidate": "v24", "difference": "v24 minus v22", "seed": 20260925,
              "replicates": 2000, "stage_index": 0, "fallback": fallback, "methods": {}, "caution": CAUTION}
    arrays = {}
    for index, (name, function, groups) in enumerate((("project", methods.bootstrap_projects, keys),
                                                    ("month", methods.bootstrap_months, months))):
        samples, support = function(errors, labels, groups, WEIGHTS,
            np.random.default_rng(np.random.SeedSequence([20260925, 0, index])), 2000)
        summary = {}
        for metric, item in methods.summarize(samples, point).items():
            translated = {k: v for k, v in item.items() if k not in ("v2_mae", "v3_mae", "point_delta_v3_minus_v2")}
            translated.update(control_mae=item["v2_mae"], candidate_mae=item["v3_mae"], point_delta=item["point_delta_v3_minus_v2"])
            summary[metric] = translated
        result["methods"][name] = {"family_index": index, "summary": summary, "support": support}
        arrays[name] = samples
    return result, arrays


def comparisons(metrics):
    control, candidate = metrics["v22"], metrics["v24"]
    def difference(a, b):
        return {"control_mae": a, "candidate_mae": b, "delta": b - a,
                "relative_change_pct": 100 * (b - a) / a if a else None}
    result = {k: difference(control[k], candidate[k]) for k in ("overall_mae", "weighted_mae")}
    adverse = []
    for section in ("strata", "forecast_quarters", "months", "cities", "relative_forecast_months"):
        result[section] = {}
        for key, a in control[section].items():
            b = candidate[section][key]
            need(a["rows"] == b["rows"], "Group support differs")
            if a["rows"]:
                item = {"rows": a["rows"], **difference(a["mae"], b["mae"])}
                result[section][key] = item
                if (section == "forecast_quarters" or a["rows"] >= 500) and b["mae"] > 1.05 * a["mae"]:
                    adverse.append({"section": section, "group": key, **item})
    result["price_bands"] = []
    for a, b in zip(control["price_bands"], candidate["price_bands"]):
        need(a["band"] == b["band"] and a["rows"] == b["rows"], "Price support differs")
        item = {"band": a["band"], "rows": a["rows"], **(difference(a["mae"], b["mae"]) if a["rows"] else {})}
        result["price_bands"].append(item)
        if a["rows"] >= 500 and b["mae"] > 1.05 * a["mae"]:
            adverse.append({"section": "price_bands", **item})
    result["both_point_metrics_improve"] = all(candidate[k] < control[k] for k in ("overall_mae", "weighted_mae"))
    result["quarter_or_large_group_deterioration_above_5pct"] = adverse
    result["automatic_adoption"] = False
    return result


def render(result):
    lines = ["# R97：固定新切點的完整預測評估", "", "fit < 2023-02-11；query 2023-02-11～2024-02-10（含）。",
             "", "全部七支新模型與兩個完整預測封存、驗證後才載入 query 價格。原始 pickle 反序列化整張 train 表，僅按已封存 raw cache query 索引擷取價格作評分；未載入 test 或模型，未重新推論。",
             "", "這是已用過且重疊的歷史資料上的新起點檢查，不是新盲測。抽樣區間未校正多輪選型，抽樣改善比例不是 Private 勝率或 p 值。本輪不自動採用、不產 CSV；Private 未知。",
             "", "| 向量 | 整體 MAE（元） | 固定加權 MAE（元） |", "|---|---:|---:|"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines += ["", "| 分群／季度 | 列數 | v22 MAE | v24 MAE | 差值 v24−v22 |", "|---|---:|---:|---:|---:|"]
    for section in ("strata", "forecast_quarters"):
        for name, m in result["comparison"][section].items():
            lines.append(f"| {name} | {m['rows']:,} | {m['control_mae']:,.2f} | {m['candidate_mae']:,.2f} | {m['delta']:+,.2f} |")
    lines += ["", "| 真價帶 | 列數 | v22 MAE | v24 MAE | 差值 |", "|---|---:|---:|---:|---:|"]
    for m in result["comparison"]["price_bands"]:
        if m["rows"]:
            lines.append(f"| {m['band']} | {m['rows']:,} | {m['control_mae']:,.2f} | {m['candidate_mae']:,.2f} | {m['delta']:+,.2f} |")
        else:
            lines.append(f"| {m['band']} | 0 | — | — | — |")
    lines += ["", "| 配對抽樣 | 指標 | 點估計差值 | 2.5% | 97.5% | 有效／無效次數 |", "|---|---|---:|---:|---:|---:|"]
    number = lambda value: "—" if value is None else f"{value:+,.2f}"
    for family, value in result["paired"]["methods"].items():
        for metric, m in value["summary"].items():
            lines.append(f"| {family} | {metric} | {m['point_delta']:+,.2f} | {number(m['percentile_2_5'])} | {number(m['percentile_97_5'])} | {m['valid_replicates']}／{m['invalid_replicates']} |")
    lines += ["", "兩方法各 2,000 次，seed 20260925、單一窗口 index 0。project 為分層完整建案群抽樣；month 為四段連續三月的循環區塊。正權重層缺席時 weighted 留為無效，不重抽。", "",
              "所有月份、城市、相對月份、新舊建案、季度、五價帶、尾部偏差／誤差集中度與超過 5% 的季度或至少 500 列分群退步均保存在 JSON。單一誤差尾部按各模型自身排序，非同一組交易，未輸出逐筆 ID。",
              "", "[完整評估 JSON](results.json) · [project replicates](project_replicates.npy) · [month replicates](month_replicates.npy)", ""]
    return "\n".join(lines)


def evaluate(sealed_manifest, out, evaluation_request):
    out = safe(out)
    need(not out.exists(), "Evaluation output exists; no overwrite or automatic retry")
    out.mkdir(parents=True, exist_ok=False)
    started, tick, code, failure = now(), time.monotonic(), 1, None
    checked = {}
    try:
        checked = preflight(sealed_manifest, evaluation_request)
        write(out / "prelabel_gate.json", {"status": "verified_without_price_deserialization", "sealed_manifest": checked["manifest_path"],
              "sealed_manifest_sha256": checked["manifest_sha256"], "verified_input_count": len(checked["input_sha256"]),
              "parameters": parameters(), "source_sha256": checked["source_sha256"], "completed_at_utc": now()})
        y = load_query_y(checked)
        d = checked["data"]
        vectors = dict(checked["predictions"])
        vectors.update({"native/" + k: v for k, v in checked["native"].items()})
        vectors.update({b + "/" + layer: checked["branches"][b][layer] for b in BRANCHES for layer in LAYERS})
        metrics = {name: descriptive_metrics(y, p, d.Xfit, d.Xquery) for name, p in vectors.items()}
        boot, arrays = paired(y, checked["predictions"], d.Xfit, d.Xquery, d.query_indices)
        for family in boot["methods"].values():
            for metric, key in (("overall", "overall_mae"), ("weighted", "weighted_mae")):
                item = family["summary"][metric]
                need(abs(item["control_mae"] - metrics["v22"][key]) <= 1e-6
                     and abs(item["candidate_mae"] - metrics["v24"][key]) <= 1e-6, "Paired point metrics differ")
        artifacts = {rel(out / "prelabel_gate.json"): sha(out / "prelabel_gate.json")}
        for name, a in arrays.items():
            path = out / (name + "_replicates.npy")
            with path.open("xb") as f:
                np.save(f, a, allow_pickle=False)
            exact(np.load(path, allow_pickle=False), a, "Bootstrap save/reload differs")
            artifacts[rel(path)] = sha(path)
        result = {"round": 97, "status": "completed_sealed_prediction_evaluation", "created_at_utc": now(),
                  "parameters": parameters(), "sealed_manifest": checked["manifest_path"], "sealed_manifest_sha256": checked["manifest_sha256"],
                  "source_sha256": checked["source_sha256"], "input_sha256": checked["input_sha256"], "metrics": metrics,
                  "comparison": comparisons(metrics), "paired": boot, "artifact_sha256": artifacts,
                  "query_target_values_sha256": values_sha(y), "query_target_values_persisted": False,
                  "query_y_mapping": "raw_train.iloc[query_source_cache_indices]['總價元']; not canonical query_indices",
                  "raw_pickle_whole_table_deserialized_after_seal": True, "query_labels_used_in_training_or_predictions": False,
                  "model_fit_load_predict_calls": 0, "GPU_calls": 0, "test_files_read": 0, "fresh_blind_validation": False,
                  "selection_changed": False, "new_CSV": False, "caution": CAUTION,
                  "runtime": {"numpy": np.__version__, "pandas": pd.__version__}}
        verify_hashes(checked["input_sha256"])
        need(source_hashes() == checked["source_sha256"], "Evaluator changed during scoring")
        result["elapsed_seconds"] = time.monotonic() - tick
        with (out / "report.md").open("x", encoding="utf-8", newline="\n") as f:
            f.write(render(result))
        result["artifact_sha256"][rel(out / "report.md")] = sha(out / "report.md")
        write(out / "results.json", result)
        code = 0
        return result
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        write(out / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": started,
              "completed_at_utc": now(), "elapsed_seconds": time.monotonic() - tick,
              "query_labels_deserialized": checked.get("raw_price_pickle_deserialized", False), "source_sha256": sha(__file__), "automatic_retry": False})


def self_check():
    """Artificial-only tests. Does not load a dataset, model, seal or real prediction."""
    tick = time.monotonic()
    checks = []
    def mark(name, ok):
        need(bool(ok), name)
        checks.append(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, AssertionError):
            checks.append(name)
            return
        raise AssertionError("Expected rejection: " + name)
    header = {"schema_version": 1, "status": "sealed_predictions_without_query_labels", "all_fits_complete": True,
              "predictions_sealed": True, "new_native_tree_fits": 5, "new_NN_fits": 2,
              "query_labels_available": False, "cutoff": "2023-02-11", "query_end": "2024-02-10",
              "all_query_saved_model_and_state_reload_exact": True, "input_frames_unchanged": True,
              "new_query_metrics_computed": False, "scores": None, "selected_submission": "v22", "new_CSV": False,
              "protected_before": {}, "protected_after": {},
              "predictions": {k: {} for k in ("v22", "v24")}, "native_predictions": {k: {} for k in NATIVES},
              "branches": {b: {layer: {} for layer in LAYERS} for b in BRANCHES}}
    check_header(header)
    mark("complete_seal_header_accepted", True)
    for key, value in (("status", "running"), ("all_fits_complete", False), ("predictions_sealed", False),
                       ("new_native_tree_fits", 4), ("new_NN_fits", 3), ("query_labels_available", True),
                       ("cutoff", "2023-08-11"), ("query_end", "2024-02-11"),
                       ("all_query_saved_model_and_state_reload_exact", False), ("new_query_metrics_computed", True),
                       ("protected_after", {"changed": "artifact"})):
        rejects("reject_header_" + key, lambda key=key, value=value: check_header({**header, key: value}))
    rejects("label_access_requires_gate", lambda: load_query_y({}))
    request = {"status": "evaluation_frozen_before_query_score", "parameters": parameters(),
               "source_sha256": {"synthetic.py": "a" * 64}, "plan_sha256": "b" * 64}
    check_evaluation_request(request, request["source_sha256"], "b" * 64)
    mark("separate_evaluation_freeze_accepted", True)
    rejects("evaluation_changed_parameters_rejected", lambda: check_evaluation_request(
        {**request, "parameters": {**parameters(), "stage_index": 3}}, request["source_sha256"], "b" * 64))
    rejects("evaluation_changed_source_rejected", lambda: check_evaluation_request(request, {}, "b" * 64))
    raw = pd.DataFrame({"總價元": [900., 20., 700., 60., 500., 80.]}, index=[10, 11, 12, 13, 14, 15])
    y = select_query_y(raw, np.array([1, 3, 5], dtype=np.int64), 6)
    mark("raw_cache_positional_mapping_not_dataframe_label_or_canonical", np.array_equal(y, [20., 60., 80.]))
    rejects("duplicate_raw_indices_rejected", lambda: select_query_y(raw, np.array([1, 1], dtype=np.int64), 6))
    rejects("out_of_range_raw_indices_rejected", lambda: select_query_y(raw, np.array([6], dtype=np.int64), 6))
    rejects("wrong_raw_table_rows_rejected", lambda: select_query_y(raw, np.array([1], dtype=np.int64), 5))
    rejects("invalid_selected_query_price_rejected", lambda: select_query_y(raw.assign(總價元=np.nan), np.array([1], dtype=np.int64), 6))
    dates = pd.to_datetime(["2023-02-11", "2023-03-10", "2023-03-11", "2024-02-10"])
    mark("relative_month_inclusive_boundaries", np.array_equal(methods.relative_months((dates - pd.Timestamp("2000-01-01")).days, "2023-02-11"), [0, 0, 1, 11]))
    rejects("day_after_query_end_rejected_by_relative_month", lambda: methods.relative_months([(pd.Timestamp("2024-02-11") - pd.Timestamp("2000-01-01")).days], "2023-02-11"))
    rows = 8
    q = pd.DataFrame({"address_city": ["c"] * rows, "address_district": ["d"] * rows,
                      "project_name": ["seen", "seen", "new", "new", "new", "__MISSING__", "other", "other"],
                      "address_raw": ["__MISSING__"] * rows, "address_road": ["__MISSING__"] * rows,
                      "address_section": ["__MISSING__"] * rows,
                      "trade_day_index": [(pd.Timestamp("2023-02-11") - pd.Timestamp("2000-01-01")).days + i * 45 for i in range(rows)]})
    fit = q.iloc[:1].copy()
    target = np.array([1., 1e7, 2e7, 5e7, 1e8, 2e8, 4e7, 9e7])
    pred = target + np.arange(1, 9, dtype=np.float64)
    m = descriptive_metrics(target, pred, fit, q)
    mark("overall_MAE_handcomputed", m["overall_mae"] == 4.5)
    expected = WEIGHTS["seen"] * 1.5 + WEIGHTS["new_nonmissing"] * np.mean([3., 4., 5., 7., 8.])
    mark("fixed_group_weights_missing_zero_not_overall", m["weighted_mae"] == expected and m["strata"]["missing"]["rows"] == 1)
    mark("price_band_boundaries_partition", [v["rows"] for v in m["price_bands"]] == [1, 1, 2, 2, 2])
    mark("tail_bias_and_small_sample_top1pct", m["tails"]["mean_signed_error_prediction_minus_truth"] == 4.5 and m["tails"]["top1pct_error_rows"] == 1 and m["tails"]["top1pct_absolute_error_sum"] == 8.)
    zero = descriptive_metrics(target, target.copy(), fit, q)
    mark("zero_error_share_null", zero["tails"]["top10_error_share"] is None and zero["tails"]["absolute_error_sum"] == 0)
    miss = q.copy(); miss["project_name"] = "__MISSING__"
    rejects("missing_positive_weight_strata_rejected", lambda: descriptive_metrics(target, pred, fit, miss))
    labels = np.array(["seen", "seen", "new_nonmissing", "new_nonmissing"])
    errors = np.array([[10., 7.], [50., 47.], [100., 97.], [200., 197.]])
    point = methods.point_values(errors, labels, WEIGHTS)
    for family, fn, groups in (("project", methods.bootstrap_projects, np.array(["a", "a", "b", "c"])),
                                ("month", methods.bootstrap_months, np.array([0, 0, 6, 6]))):
        sample, support = fn(errors, labels, groups, WEIGHTS, np.random.default_rng(11), 100)
        summary = methods.summarize(sample, point)
        mark(family + "_constant_paired_difference", all(abs(v["point_delta_v3_minus_v2"] + 3.) < 1e-12 and abs(v["percentile_2_5"] + 3.) < 1e-12 and abs(v["percentile_97_5"] + 3.) < 1e-12 for v in summary.values()))
        same, _ = fn(errors[:, [0, 0]], labels, groups, WEIGHTS, np.random.default_rng(11), 100)
        mark(family + "_identical_models_exact", np.array_equal(same[:, 0], same[:, 1], equal_nan=True) and np.array_equal(same[:, 2], same[:, 3], equal_nan=True))
        if family == "month":
            mark("month_missing_weight_stratum_not_redrawn", summary["weighted"]["invalid_replicates"] > 0)
    native = {k: np.array([float(i + 1), float(10 + i)], dtype=np.float64) for i, k in enumerate(NATIVES)}
    base = {"N": .75 * (.5 * native["N_area"] + .5 * native["N_prior"]) + .25 * native["light"],
            "H": .75 * (.5 * native["raw"] + .5 * native["log"]) + .25 * native["light"],
            "U22": .75 * native["unit42"] + .25 * native["light"], "U24": .5 * native["unit42"] + .5 * native["unit137"]}
    branch = {k: {layer: a.copy() for layer in LAYERS} for k, a in base.items()}
    full = {k: np.median(np.column_stack([base[b] for b in ("N", "H", unit)]), axis=1) for k, unit in (("v22", "U22"), ("v24", "U24"))}
    check_formulas(native, branch, full)
    mark("four_base_and_two_median_exact", True)
    corrupted = {k: v.copy() for k, v in full.items()}; corrupted["v24"][0] += 1
    rejects("changed_median_rejected", lambda: check_formulas(native, branch, corrupted))
    branch["U24"]["base"] = .75 * base["U24"] + .25 * native["light"]
    rejects("U24_adding_LGB_rejected", lambda: check_formulas(native, branch, full))
    with tempfile.TemporaryDirectory(prefix="r97_eval_synthetic_", dir=ROOT / "outputs") as folder:
        fake_seal = Path(folder) / "unsealed.json"
        write(fake_seal, {**header, "all_fits_complete": False})
        rejects("unsealed_stops_before_request_artifacts_or_label_loading", lambda: preflight(fake_seal, Path(folder) / "nonexistent_request.json"))
        p = Path(folder) / "vector.npy"
        with p.open("xb") as f:
            np.save(f, pred, allow_pickle=False)
        rec = {"path": rel(p), "sha256": sha(p), "values_sha256": values_sha(pred), "dtype": "float64", "shape": [rows]}
        mark("small_saved_vector_identity", np.array_equal(load_array(rec, rows, "float64"), pred))
        rejects("changed_array_sha_rejected", lambda: load_array({**rec, "sha256": "0" * 64}, rows, "float64"))
        rejects("changed_values_sha_rejected", lambda: load_array({**rec, "values_sha256": "0" * 64}, rows, "float64"))
        rejects("wrong_record_dtype_rejected", lambda: load_array({**rec, "dtype": "float32"}, rows, "float64"))
        rejects("conflicting_record_hashes_rejected", lambda: merge_hashes({rel(p): "0" * 64}, {rel(p): "1" * 64}))
    return {"status": "passed", "checks": checks, "check_count": len(checks), "parameters": parameters(),
            "source_sha256": sha(__file__), "source_hashes": source_hashes(), "elapsed_seconds": time.monotonic() - tick,
            "synthetic_only": True, "real_price_or_prediction_files_loaded": 0, "model_fit_load_predict_calls": 0,
            "GPU_calls": 0, "completed_at_utc": now()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sealed-manifest", type=Path)
    parser.add_argument("--evaluation-request", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.self_check:
        need(args.sealed_manifest is None and args.evaluation_request is None, "Self-check cannot consume a real seal/request")
        receipt = safe(args.out)
        need(not receipt.exists(), "Self-check receipt exists; preserve attempts")
        try:
            result = self_check()
        except Exception:
            write(receipt, {"status": "failed", "failure": traceback.format_exc(), "source_sha256": sha(__file__), "synthetic_only": True})
            raise
        write(receipt, result)
        print(json.dumps({"status": result["status"], "checks": result["check_count"], "receipt": rel(receipt)}), flush=True)
    else:
        need(args.sealed_manifest is not None and args.evaluation_request is not None, "--sealed-manifest and --evaluation-request required")
        r = evaluate(args.sealed_manifest, args.out, args.evaluation_request)
        print(json.dumps({"status": r["status"], "comparison": r["comparison"]["overall_mae"], "elapsed_seconds": r["elapsed_seconds"]}), flush=True)
