"""Fixed R99 constants and file contracts; no data loading or training on import."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = "reports/round99_plan.md"
PLAN_SHA = "8918ba8a4a0b50378aca56649131678ce069350902c044a6678384b4e95d75c9"
STAGES = ("development", "offset", "confirmation")
BRANCHES = ("N", "H", "U")
STACKER_BRANCHES = ("N", "H", "U", "M")
SPLITS = {"development": ("2022-08-11", "2023-08-10", 81234),
          "offset": ("2023-02-11", "2024-02-10", 107169),
          "confirmation": ("2023-08-11", "2024-08-10", 133697)}
WEIGHTS = {"seen": 0.36617593949822613, "new_nonmissing": 0.6338240605017739, "missing": 0.0}
RAW = "data_cache/train.pkl"
FEATURES = "data_cache/train_features.pkl"
CANONICAL = "outputs/round6/geography/development/train_source_cache_indices.npy"
INVENTORY = "outputs/round98_artifact_inventory/results.json"
PREPARATION = "outputs/round98_inputs/manifest.json"
PREPARATION_SHA = "e496b2afcf7ebc69a4e248d13b691e3b2e6e991bfab4e510386ff77cccc97fc7"
PREPARATION_PLAN = "reports/round98_plan.md"
PREPARATION_PLAN_SHA = "394a65459349e479bb3e8a487b60516156705469fa51bfbe75fd8697bae3642b"
R98_SEALED_MANIFEST = "outputs/round98/sealed_manifest.json"
R98_SEALED_MANIFEST_SHA = "d0ff684e01bf18d590f9d8cdd402a6067aa7d85df1ea223fc6cf03df097bcbcc"
OLD_META_OBJECTIVE_TOLERANCE_TWD = 0.10
CAUTION = ("Earlier forward blocks fit a four-column N/H/U/median stacker after observing R98; historically reused data, "
           "not a blind holdout. Offset overlaps other windows. Missing pre-auxiliary block prevents "
           "the original three-August-window adoption gate. Bootstrap is conditional, not Private win probability.")


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(name):
    p = Path(name)
    p = ((ROOT / p) if not p.is_absolute() else p).resolve()
    need(p.is_relative_to(ROOT), "Path outside workspace: " + str(name))
    return p


def rel(path):
    return safe(path).relative_to(ROOT).as_posix()


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    p = safe(path)
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    with p.with_suffix(".sha256").open("x", encoding="ascii") as f:
        f.write(sha(p) + "\n")


def locked(path):
    p = safe(path)
    need(p.with_suffix(".sha256").read_text(encoding="ascii").strip() == sha(p), "Receipt differs: " + rel(p))
    return read(p)


def exact(a, b, name):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name)


def array_record(path, a):
    return {"path": rel(path), "sha256": sha(path), "dtype": str(a.dtype),
            "shape": list(a.shape), "values_sha256": values_sha(a)}


def save_array(path, a):
    p = safe(path)
    need(isinstance(a, np.ndarray) and not a.dtype.hasobject, "Only numeric arrays")
    with p.open("xb") as f:
        np.save(f, a, allow_pickle=False)
    exact(a, np.load(p, allow_pickle=False), "Saved array differs: " + rel(p))
    return array_record(p, a)


def load_array(record):
    p = safe(record["path"])
    need(sha(p) == record["sha256"], "Array file identity differs: " + rel(p))
    a = np.load(p, allow_pickle=False)
    need(not a.dtype.hasobject and str(a.dtype) == record["dtype"] and list(a.shape) == record["shape"], "Array schema differs")
    need(values_sha(a) == record["values_sha256"], "Array values differ")
    return a


def source_hashes(*entries):
    pending = [safe(p) for p in entries]
    found = set()
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                q = ROOT / "src" / (name.split(".")[0] + ".py")
                if q.is_file() and q not in found:
                    pending.append(q)
    return {rel(p): sha(p) for p in sorted(found)}


def verify_hashes(mapping, exclude=()):
    for name, expected in mapping.items():
        if name not in exclude:
            need(sha(name) == expected, "Immutable source changed: " + name)


def protected_hashes():
    names = read("outputs/github_publish_r97_20260929/protected_before.json")
    return {name: sha(name) for name in names}


def evaluation_parameters():
    return {"round": 99, "stage_order": list(STAGES), "branch_order": list(STACKER_BRANCHES),
            "source_branch_order": list(BRANCHES), "additional_control": "R98",
            "control": "v22", "candidate": "R99", "seed": 20260925, "replicates": 2000,
            "stage_indices": {s: i for i, s in enumerate(STAGES)}, "family_order": ["project", "month"],
            "family_indices": {"project": 0, "month": 1}, "month_block_length": 3, "month_blocks": 4,
            "fixed_group_weights": WEIGHTS, "large_group_minimum_rows": 500,
            "max_adverse_fraction": .05, "automatic_adoption": False, "new_CSV": False,
            "original_three_period_adoption_complete": False}


def matrix_identity(a):
    return {"dtype": str(a.dtype), "shape": list(a.shape), "values_sha256": values_sha(a)}


def stacker_matrix(base):
    need(isinstance(base, np.ndarray) and base.dtype == np.float64
         and base.ndim == 2 and base.shape[1] == len(BRANCHES), "Original matrix must be float64 n x 3")
    need(np.isfinite(base).all() and (base >= 0).all(), "Original branch predictions must be finite and nonnegative")
    median = np.median(base, axis=1)
    expanded = np.column_stack((base, median))
    exact(expanded[:, :3], base, "Expansion changed original branches")
    exact(expanded[:, 3], median, "Fourth column differs from row median")
    return expanded
