"""Isolated R98 preparation: historical forward matrices and stage-specific meta labels.

This extractor necessarily deserializes the full raw price pickle. It exports
only each stage's pre-cutoff meta targets, with cross-stage label roles explicit.
No fitting, query scoring, test loading or model deserialization occurs here.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
import numpy as np
import pandas as pd
import round98_contract as c

OUT = c.ROOT / "outputs/round98_inputs"
ORIGIN = pd.Timestamp("2000-01-01")


def normalized_record(r):
    return {"path": r["path"], "sha256": r["file_sha256"], "dtype": r["dtype"],
            "shape": r["shape"], "values_sha256": r["values_sha256"]}


def day(date):
    return (pd.Timestamp(date) - ORIGIN).days


def assemble_meta(blocks, days, cutoff, source_order):
    indices, predictions, pieces = [], [], []
    for source in source_order:
        idx, matrix = blocks[source]["query_indices"], blocks[source]["predictions"]
        c.need(idx.dtype == np.int64 and matrix.dtype == np.float64 and matrix.shape == (len(idx), 3), "Forward block schema")
        positions = np.flatnonzero(np.isfinite(days[idx]) & (days[idx] < day(cutoff))).astype(np.int64)
        c.need(len(positions) > 0, "Empty source block")
        indices.append(idx[positions]); predictions.append(matrix[positions])
        pieces.append({"block": source, "selected_rows": len(positions),
                       "selected_source_positions_values_sha256": c.values_sha(positions)})
    ids = np.concatenate(indices).astype(np.int64, copy=False)
    matrix = np.concatenate(predictions).astype(np.float64, copy=False)
    c.need(len(np.unique(ids)) == len(ids), "Duplicate meta indices")
    c.need(np.isfinite(days[ids]).all() and (days[ids] < day(cutoff)).all(), "Future meta label")
    c.need(np.isfinite(matrix).all() and (matrix >= 0).all(), "Invalid forward total price")
    return ids, matrix, pieces


def select_meta_targets(raw, raw_ids):
    c.need(isinstance(raw, pd.DataFrame) and len(raw) == 455784 and raw.index.equals(pd.RangeIndex(455784)), "Raw table row axis")
    c.need(raw_ids.dtype == np.int64 and raw_ids.ndim == 1 and len(np.unique(raw_ids)) == len(raw_ids), "Raw meta mapping invalid")
    c.need((raw_ids >= 0).all() and (raw_ids < len(raw)).all(), "Raw meta index bounds")
    y = raw.iloc[raw_ids]["總價元"].to_numpy(dtype=np.float64, copy=True)
    c.need(y.shape == raw_ids.shape and np.isfinite(y).all() and (y > 0).all(), "Invalid meta targets")
    return y


def prepare(expected_inventory_sha):
    tick, start = time.monotonic(), c.now()
    c.need(not OUT.exists(), "Existing preparation preserved; no retry or overwrite")
    OUT.mkdir()
    code, failure, raw_opened = 1, None, False
    c.write(OUT / "started.json", {"started_at_utc": start, "scope": "pre-cutoff meta targets; no fit or query score"})
    try:
        c.need(c.sha(c.PLAN) == c.PLAN_SHA and c.sha(c.INVENTORY) == expected_inventory_sha, "Frozen plan/inventory changed")
        inv = c.read(c.INVENTORY)
        c.need(inv["status"] == "passed" and inv["branch_order"] == list(c.BRANCHES) and inv["query_or_fit_labels_loaded"] is False, "Pure-X inventory not passed")
        c.verify_hashes(inv["verified_input_sha256"])
        c.need(c.sha("outputs/round98_artifact_inventory/audit.py") == inv["script_sha256"], "Inventory source changed")
        sources = c.source_hashes(__file__)
        protected = c.protected_hashes()
        mapping = c.load_array(normalized_record(inv["canonical_mapping"]))
        c.need(mapping.shape == (455255,) and mapping.dtype == np.int64 and (np.diff(mapping) > 0).all(), "Canonical mapping")
        feature_sha = inv["verified_input_sha256"][c.FEATURES]
        c.need(c.sha(c.FEATURES) == feature_sha, "Pure-X source changed")
        Xraw = pd.read_pickle(c.safe(c.FEATURES))
        c.need(len(Xraw) == 455784 and "總價元" not in Xraw, "Feature target/axis")
        X = Xraw.iloc[mapping].reset_index(drop=True)
        del Xraw
        days = X.trade_day_index.to_numpy(dtype=np.float64)
        c.need(np.isnan(days).sum() == 8, "Date identity")
        loaded = {}
        for name, b in inv["blocks"].items():
            fit = c.load_array(normalized_record(b["fit_indices"]))
            query = c.load_array(normalized_record(b["query_indices"]))
            c.need((np.diff(fit) > 0).all() and (np.diff(query) > 0).all(), "Canonical sorted indices")
            c.exact(fit, np.flatnonzero(np.isfinite(days) & (days < day(b["cutoff"]))).astype(np.int64), "Fit date mask")
            c.exact(query, np.flatnonzero(np.isfinite(days) & (days >= day(b["cutoff"])) & (days <= day(b["query_end"]))).astype(np.int64), "Query date mask")
            matrix = np.column_stack([c.load_array(normalized_record(b["branches"][k])) for k in c.BRANCHES])
            control = c.load_array(normalized_record(b["control_v22"]))
            c.exact(np.median(matrix, axis=1), control, "Control v22 differs")
            loaded[name] = {"fit_indices": fit, "query_indices": query, "predictions": matrix, "control_v22": control}
        stage_values = {}
        for s in c.STAGES:
            spec = inv["meta_fit"][s]
            cutoff, end, n = c.SPLITS[s]
            c.need(inv["blocks"][s]["cutoff"] == cutoff and inv["blocks"][s]["query_end"] == end, "Fixed stage dates")
            expected_order = ["auxiliary"] if s == "development" else ["auxiliary", "development"]
            c.need(spec["source_order"] == expected_order, "Meta forward origin order")
            ids, P, pieces = assemble_meta(loaded, days, cutoff, expected_order)
            c.need(len(ids) == spec["rows"] and c.values_sha(ids) == spec["canonical_meta_indices_values_sha256"]
                   and c.values_sha(P) == spec["prediction_matrix_values_sha256"], "Meta selection identity")
            c.need(c.values_sha(mapping[ids]) == spec["raw_meta_source_indices_values_sha256"], "Meta raw mapping identity")
            for actual, expected in zip(pieces, spec["pieces"]):
                c.need(actual["selected_rows"] == expected["selected_rows"] and actual["selected_source_positions_values_sha256"] == expected["selected_source_positions_values_sha256"], "Prefix positions differ")
            q, f = loaded[s]["query_indices"], loaded[s]["fit_indices"]
            c.need(len(q) == n and not np.intersect1d(ids, q).size and np.isin(ids, f).all(), "Meta/query temporal separation")
            stage_values[s] = {"meta_predictions": P, "meta_indices": ids, "meta_source_cache_indices": mapping[ids],
                               "meta_trade_days": days[ids].copy(), "query_predictions": loaded[s]["predictions"],
                               "query_indices": q, "query_source_cache_indices": mapping[q], "query_trade_days": days[q].copy(),
                               "fit_indices": f, "control_v22": loaded[s]["control_v22"]}
        # Full price source is opened only inside this isolated preparer.
        raw_rec = inv["raw_train_source"]
        c.need(raw_rec["path"] == c.RAW and c.sha(c.RAW) == raw_rec["file_sha256"], "Course price source identity")
        raw = pd.read_pickle(c.safe(c.RAW)); raw_opened = True
        stage_info = {}
        for s in c.STAGES:
            values = stage_values[s]
            values["meta_y"] = select_meta_targets(raw, values["meta_source_cache_indices"])
            folder = OUT / s; folder.mkdir()
            records = {key: c.save_array(folder / (key + ".npy"), a) for key, a in values.items()}
            dates = inv["meta_fit"][s]["dates"]
            stage_info[s] = {"cutoff": c.SPLITS[s][0], "query_end": c.SPLITS[s][1], "query_rows": c.SPLITS[s][2],
                             "meta_rows": len(values["meta_indices"]), "meta_min_date": dates["first_date"],
                             "meta_max_date": dates["last_date"], "source_blocks": inv["meta_fit"][s]["pieces"],
                             "arrays": records, "meta_target_units": "TWD", "query_y_exported_to_this_stage": False,
                             "cross_stage_role_overlap": "Development query rows are legal later-stage meta labels; independent stage processes required."}
        del raw
        c.verify_hashes(inv["verified_input_sha256"])
        c.need(c.sha(c.RAW) == raw_rec["file_sha256"] and sources == c.source_hashes(__file__), "Sources changed during preparation")
        c.need(protected == c.protected_hashes(), "Protected selected outputs changed")
        manifest = {"schema_version": 1, "status": "prepared_temporal_meta_inputs", "plan_sha256": c.PLAN_SHA,
                    "inventory": {"path": c.INVENTORY, "sha256": expected_inventory_sha},
                    "raw_train": {"path": c.RAW, "sha256": raw_rec["file_sha256"], "rows": 455784},
                    "features": {"path": c.FEATURES, "sha256": feature_sha},
                    "canonical_mapping": {"path": c.CANONICAL, "sha256": inv["canonical_mapping"]["file_sha256"]},
                    "source_sha256": sources, "input_sha256": inv["verified_input_sha256"],
                    "stages": stage_info, "branch_order": list(c.BRANCHES), "stage_order": list(c.STAGES),
                    "raw_price_table_deserialized_in_extractor": True, "query_statistics_computed": False,
                    "new_native_fits": 0, "new_meta_fits": 0, "test_read": False, "valid_read": False,
                    "all_arrays_saved_reload_exact": True, "protected_before": protected, "protected_after": c.protected_hashes(),
                    "started_at_utc": start, "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick}
        c.write(OUT / "manifest.json", manifest)
        code = 0
        print(json.dumps({"status": manifest["status"], "meta_rows": {s: stage_info[s]["meta_rows"] for s in c.STAGES},
                          "elapsed_seconds": manifest["elapsed_seconds"], "model_fits": 0}))
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        c.write(OUT / "execution.json", {"exit_code": code, "failure": failure, "started_at_utc": start,
                "completed_at_utc": c.now(), "elapsed_seconds": time.monotonic() - tick,
                "raw_pickle_deserialized": raw_opened, "query_scores_computed": False, "new_fits": 0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory-sha", required=True)
    args = ap.parse_args()
    prepare(args.inventory_sha)
