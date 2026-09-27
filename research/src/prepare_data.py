"""Prepare trusted local caches; the supplied Excel and CSV files are never edited."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "台灣房屋銷售價格預測挑戰_回歸" / "ml-2026-iaii-regression"
CACHE_DIR = ROOT / "data_cache"
REPORT_DIR = ROOT / "reports"
TARGET = "總價元"
ID = "編號"
CACHE_VERSION = 1
FILENAMES = {"train": "train-v2.xlsx", "valid": "valid-v2.xlsx", "test": "test-reindex-test-v2.2.xlsx"}
SAMPLE_FILENAME = "test-reindex-test-sample-submission-v2.csv"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def column_summary(frame):
    result = {}
    for name in frame:
        series = frame[name]
        missing = series.isna() | series.astype("string").str.strip().eq("").fillna(False)
        result[name] = {"missing": int(missing.sum()), "missing_fraction": float(missing.mean()),
                        "distinct_nonmissing": int(series[~missing].nunique())}
    return result

def load_one(split):
    source = DATA_DIR / FILENAMES[split]
    digest = sha256(source)
    cache = CACHE_DIR / f"{split}.pkl"
    metadata = CACHE_DIR / f"{split}.json"
    if cache.exists() and metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if previous.get("source_sha256") == digest and previous.get("cache_version") == CACHE_VERSION:
            print(f"{split}: using verified source cache", flush=True)
            return pd.read_pickle(cache), previous
    print(f"{split}: reading {source.name}", flush=True)
    with pd.ExcelFile(source, engine="openpyxl") as workbook:
        if workbook.sheet_names != ["Sheet1"]:
            raise ValueError(f"Unexpected worksheets: {workbook.sheet_names}")
        frame = pd.read_excel(workbook, sheet_name="Sheet1", dtype=object)
    raw_rows = len(frame)
    rejected = []
    if TARGET in frame:
        prices = pd.to_numeric(frame[TARGET], errors="coerce")
        bad = prices.isna() | ~np.isfinite(prices)
        for index in frame.index[bad]:
            rejected.append({"excel_row": int(index) + 2, "id": str(frame.at[index, ID]),
                             "target_value": str(frame.at[index, TARGET]), "reason": "non_numeric_target"})
        # Never silently discard unknown label problems or price outliers.
        if any(item["id"] != "serial number" or item["target_value"] != "total price NTD" for item in rejected):
            raise ValueError(f"Unexpected malformed labels in {split}: {rejected}")
        frame = frame.loc[~bad].copy()
        frame[TARGET] = prices.loc[~bad].astype("float64")
        if (frame[TARGET] <= 0).any():
            raise ValueError(f"Nonpositive labeled prices in {split}")
    frame = frame.reset_index(drop=True)
    summary = {
        "cache_version": CACHE_VERSION, "source": str(source.relative_to(ROOT)), "source_sha256": digest,
        "raw_rows": raw_rows, "clean_rows": len(frame), "columns": list(frame.columns),
        "rejected_rows": rejected, "missing_id": int(frame[ID].isna().sum()),
        "duplicate_id_excess": int(frame[ID].duplicated().sum()),
        "exact_duplicate_row_excess": int(frame.duplicated().sum()), "column_summary": column_summary(frame)}
    if TARGET in frame:
        summary["target_quantiles"] = {str(q): float(value) for q, value in frame[TARGET].quantile([0,.25,.5,.75,.95,.99,1]).items()}
    temporary_cache = cache.with_suffix(".pkl.tmp")
    frame.to_pickle(temporary_cache)
    temporary_cache.replace(cache)
    metadata.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{split}: {len(frame):,} usable rows; {len(frame.columns)} columns; removed {len(rejected)} embedded headers", flush=True)
    return frame, summary

def prepare_all():
    CACHE_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    frames, summaries = {}, {}
    for split in FILENAMES:
        frames[split], summaries[split] = load_one(split)
    sample_path = DATA_DIR / SAMPLE_FILENAME
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError("Unexpected submission template columns")
    if list(frames["train"].columns) != list(frames["valid"].columns):
        raise ValueError("Train and validation columns differ")
    if [c for c in frames["train"] if c != TARGET] != list(frames["test"].columns):
        raise ValueError("Test features do not match training schema")
    if (frames["test"][ID].astype("string").tolist() != sample[ID].astype("string").tolist()
            or frames["test"][ID].isna().any() or not frames["test"][ID].is_unique):
        raise ValueError("Test IDs do not match the submission template exactly")
    summaries["submission_template"] = {
        "source": str(sample_path.relative_to(ROOT)), "source_sha256": sha256(sample_path),
        "rows": len(sample), "columns": list(sample.columns), "test_ids_match_in_order": True,
        "placeholder_distinct_values": int(sample[TARGET].nunique())}
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Input schemas and submission IDs checked.", flush=True)
    return frames

if __name__ == "__main__":
    prepare_all()
