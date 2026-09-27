"""Strict competition output validation, without test targets."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

COLUMNS = ["編號", "總價元"]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def ids(frame):
    if "編號" not in frame:
        raise ValueError("Missing 編號")
    result = frame["編號"].astype(str).str.strip()
    if result.isna().any() or result.eq("").any() or result.isin(["nan", "None", "<NA>"]).any() or result.duplicated().any():
        raise ValueError("Missing or duplicate ID")
    return result.tolist()

def rounded_prices(prediction):
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Predictions must be a finite vector")
    r = np.maximum(np.rint(p), 1)
    if np.any(r >= np.iinfo(np.int64).max):
        raise ValueError("Price exceeds int64")
    return r.astype(np.int64)

def write_submission(path, query_ids, sample_ids, prediction):
    if query_ids != sample_ids or len(prediction) != len(sample_ids):
        raise ValueError("Test/sample IDs and predictions must have identical order and count")
    result = pd.DataFrame({"編號":sample_ids, "總價元":rounded_prices(prediction)})
    path = Path(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        result.to_csv(stream, index=False, lineterminator="\r\n")
    actual = pd.read_csv(path, dtype={"編號":str})
    if list(actual) != COLUMNS or ids(actual) != sample_ids or not np.array_equal(actual["總價元"],result["總價元"]):
        raise RuntimeError("Submission roundtrip differs")
    return digest(path)
