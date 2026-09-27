"""Select a conservative ensemble using local chronological validation.

No models are trained here. Run after every named validation and temporal
candidate has finished. The optional --names argument extends the default four
candidates without silently skipping missing or inconsistent artifacts.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from model_common import (
    ROOT, CACHE_DIR, REPORT_DIR, MODEL_DIR, EXPERIMENT_DIR,
    load_data, temporal_indices, mae,
)
from prepare_data import sha256

DEFAULT_NAMES = [
    "catboost_raw", "catboost_unit",
    "lightgbm_l1_raw63", "lightgbm_l1_unit63",
]
CUTOFF = "2024-04-01"
END = "2024-08-10"
MISSING = "__MISSING__"
PAIR_WEIGHTS = (0.25, 0.5, 0.75)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value



def _model_unique_mask(data: dict, path: Path) -> np.ndarray:
    """Derive the mask from current model inputs; verify any existing cache."""
    train, valid = data["X_train"], data["X_valid"]
    if list(train.columns) != list(valid.columns) or not train.dtypes.equals(valid.dtypes):
        raise ValueError("Train/validation schemas differ while checking model-feature overlap")
    train_hash = pd.util.hash_pandas_object(train, index=False).to_numpy()
    valid_hash = pd.util.hash_pandas_object(valid, index=False).to_numpy()
    overlap = pd.Series(valid_hash).isin(train_hash).to_numpy()
    # Hash matching narrows the search; exact row equality excludes accidental
    # hash collisions and uses pandas' consistent handling of missing values.
    for valid_index in np.flatnonzero(overlap):
        matches = np.flatnonzero(train_hash == valid_hash[valid_index])
        if not any(train.iloc[i].equals(valid.iloc[valid_index]) for i in matches):
            overlap[valid_index] = False
    mask = ~overlap
    if not mask.any():
        raise ValueError("No model-feature-unique validation rows remain")
    if path.exists():
        previous = np.load(path, allow_pickle=False)
        if (previous.dtype != np.dtype("bool") or previous.shape != mask.shape
                or not np.array_equal(previous, mask)):
            raise ValueError("Existing model-feature-unique mask does not match current features")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.npy")
        np.save(temporary, mask)
        temporary.replace(path)
    print(json.dumps({
        "model_feature_overlap_rows": int(overlap.sum()),
        "model_feature_unique_rows": int(mask.sum()),
    }), flush=True)
    return mask


def _artifact_paths(name: str) -> dict[str, Path]:
    return {
        "validation_metadata": EXPERIMENT_DIR / f"{name}.json",
        "validation_predictions": EXPERIMENT_DIR / f"{name}_valid.npy",
        "temporal_metadata": EXPERIMENT_DIR / f"{name}_temporal.json",
        "temporal_predictions": EXPERIMENT_DIR / f"{name}_temporal_valid.npy",
        "temporal_indices": EXPERIMENT_DIR / f"{name}_temporal_indices.npy",
    }


def _prediction(path: Path, expected_rows: int) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype="float64")
    if value.shape != (expected_rows,) or not np.isfinite(value).all():
        raise ValueError(f"Nonfinite or incorrectly shaped predictions: {path}")
    return value


def _family(metadata: dict) -> str:
    value = metadata.get("family", metadata.get("model_family", ""))
    family = str(value).strip().lower()
    if family not in {"catboost", "lightgbm"}:
        raise ValueError(f"Unsupported or missing model family: {value!r}")
    return family


def _check_recorded_mae(metadata: dict, actual: float, description: str) -> None:
    if "mae" in metadata:
        recorded = float(metadata["mae"])
        if not np.isfinite(recorded) or not np.isclose(recorded, actual, rtol=1e-9, atol=1e-5):
            raise ValueError(f"Saved MAE does not match aligned predictions: {description}")


def _load_candidate(name: str, paths: dict[str, Path], data: dict,
                    expected_time_indices: np.ndarray) -> dict:
    metadata = _read_json(paths["validation_metadata"])
    temporal_metadata = _read_json(paths["temporal_metadata"])
    if metadata.get("name") != name:
        raise ValueError(f"Validation metadata name mismatch: {name}")
    # CatBoost records the base candidate name, while the LightGBM temporal
    # runner records the base name plus the explicit temporal suffix.
    if temporal_metadata.get("name") not in {name, f"{name}_temporal"}:
        raise ValueError(f"Temporal metadata name mismatch: {name}")
    family = _family(metadata)
    mode = metadata.get("target_mode")
    if mode not in {"raw", "unit"}:
        raise ValueError(f"Invalid target mode for {name}: {mode!r}")
    if "target_mode" in temporal_metadata and temporal_metadata["target_mode"] != mode:
        raise ValueError(f"Temporal target mode differs for {name}")
    if "family" in temporal_metadata or "model_family" in temporal_metadata:
        if _family(temporal_metadata) != family:
            raise ValueError(f"Temporal model family differs for {name}")
    for field, expected in (("cutoff", CUTOFF), ("end", END)):
        if field in temporal_metadata and temporal_metadata[field] != expected:
            raise ValueError(f"Unexpected temporal {field} for {name}")
    saved_sources = metadata.get("data", {}).get("source_sha256")
    if saved_sources is not None and saved_sources != data["metadata"]["source_sha256"]:
        raise ValueError(f"Source data signatures differ for {name}")
    saved_features = metadata.get("features")
    if isinstance(saved_features, list) and saved_features != list(data["X_train"].columns):
        raise ValueError(f"Feature columns differ for {name}")

    parameter_key = "parameters" if family == "catboost" else "params"
    parameters = metadata.get(parameter_key)
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing {parameter_key} for {name}")
    iterations = metadata.get("best_iteration")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"Invalid validation best_iteration for {name}")

    indices = np.load(paths["temporal_indices"], allow_pickle=False)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, expected_time_indices)):
        raise ValueError(f"Temporal indices differ from the required chronological split: {name}")
    valid_prediction = _prediction(paths["validation_predictions"], len(data["y_valid"]))
    time_prediction = _prediction(paths["temporal_predictions"], len(expected_time_indices))
    _check_recorded_mae(metadata, mae(data["y_valid"], valid_prediction), name)
    _check_recorded_mae(
        temporal_metadata, mae(data["y_train"][expected_time_indices], time_prediction),
        f"{name} temporal",
    )
    return {
        "name": name, "family": family, "target_mode": mode,
        "iterations": iterations, "parameters": parameters,
        "metadata": metadata, "temporal_metadata": temporal_metadata,
        "valid_prediction": valid_prediction, "time_prediction": time_prediction,
        "artifact_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in paths.values()
        },
    }


def _model_component(candidate: dict, weight: float) -> dict:
    name, family = candidate["name"], candidate["family"]
    component = {
        "name": name,
        "family": family,
        "target_mode": candidate["target_mode"],
        "weight": float(weight),
        # This is the fitted tree count from the original validation candidate,
        # not the temporal stopping iteration or its configured iteration cap.
        "iterations": candidate["iterations"],
        "parameters": candidate["parameters"],
        "model_path": (f"models/final/{name}.cbm" if family == "catboost"
                       else f"models/final/{name}/model.txt"),
    }
    if family == "lightgbm":
        component["category_maps_path"] = f"models/final/{name}/category_maps.json"
    return component


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(names: list[str]) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError("Candidate names must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names):
        raise ValueError("Candidate names must be simple artifact basenames")
    paths = {name: _artifact_paths(name) for name in names}
    model_unique_mask_path = CACHE_DIR / "valid_model_unique_mask.npy"
    required = [path for candidate_paths in paths.values() for path in candidate_paths.values()]
    missing_files = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("All requested candidates must finish first. Missing files:\n" +
                                "\n".join(missing_files))

    data = load_data()
    fit_indices, time_indices = temporal_indices(data, CUTOFF, end=END)
    if not len(fit_indices) or not len(time_indices):
        raise ValueError("The required chronological fit and validation splits must both be nonempty")
    model_unique_mask = _model_unique_mask(data, model_unique_mask_path)
    clean_mask = data["valid_clean_mask"]
    y_valid = data["y_valid"]
    y_time = data["y_train"][time_indices]

    known_test_projects = (
        set(data["X_train"]["project_name"]) |
        set(data["X_valid"]["project_name"])
    ) - {MISSING}
    test_projects = data["X_test"]["project_name"]
    if not len(test_projects):
        raise ValueError("The test split must be nonempty")
    test_unseen_mask = ~test_projects.isin(known_test_projects).to_numpy()
    test_unseen_rate = float(test_unseen_mask.mean())

    known_time_projects = set(data["X_train"].iloc[fit_indices]["project_name"]) - {MISSING}
    seen_time_mask = data["X_train"].iloc[time_indices]["project_name"].isin(known_time_projects).to_numpy()
    if not seen_time_mask.any() or seen_time_mask.all():
        raise ValueError("Both seen and unseen projects are required for temporal reweighting")

    candidates = {
        name: _load_candidate(name, paths[name], data, time_indices) for name in names
    }

    def blend(parts: list[tuple[str, float]], key: str) -> np.ndarray:
        length = len(y_valid) if key == "valid_prediction" else len(y_time)
        result = np.zeros(length, dtype="float64")
        for name, weight in parts:
            result += weight * candidates[name][key]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite blended predictions")
        return result

    def score(parts: list[tuple[str, float]]) -> dict:
        pv = blend(parts, "valid_prediction")
        pt = blend(parts, "time_prediction")
        seen_mae = mae(y_time[seen_time_mask], pt[seen_time_mask])
        unseen_mae = mae(y_time[~seen_time_mask], pt[~seen_time_mask])
        return {
            "name": " + ".join(f"{weight:g}*{name}" for name, weight in parts),
            "components": json.dumps(
                [{"name": name, "weight": weight} for name, weight in parts],
                ensure_ascii=False, separators=(",", ":"),
            ),
            "component_count": len(parts),
            "reweighted_temporal_mae": (
                (1.0 - test_unseen_rate) * seen_mae + test_unseen_rate * unseen_mae
            ),
            "temporal_mae": mae(y_time, pt),
            "seen_project_mae": seen_mae,
            "unseen_project_mae": unseen_mae,
            "validation_mae": mae(y_valid, pv),
            "clean_validation_mae": mae(y_valid[clean_mask], pv[clean_mask]),
            "model_unique_validation_mae": mae(y_valid[model_unique_mask], pv[model_unique_mask]),
            "test_unseen_project_rate": test_unseen_rate,
        }

    schemes = [[(name, 1.0)] for name in names]
    schemes.extend(
        [(left, weight), (right, 1.0 - weight)]
        for left, right in combinations(names, 2)
        for weight in PAIR_WEIGHTS
    )
    expected_scheme_count = len(names) + len(PAIR_WEIGHTS) * len(names) * (len(names) - 1) // 2
    if len(schemes) != expected_scheme_count:
        raise AssertionError("Incomplete ensemble search grid")
    evaluated = [{"parts": parts, "scores": score(parts)} for parts in schemes]
    # Only chronological reweighting selects a model. Original validation MAE
    # breaks exact ties; clean/model-unique metrics are reported diagnostics.
    evaluated.sort(key=lambda item: (
        item["scores"]["reweighted_temporal_mae"],
        item["scores"]["validation_mae"],
    ))
    winner = evaluated[0]
    winning_parts = winner["parts"]
    winning_scores = winner["scores"]
    components = [_model_component(candidates[name], weight) for name, weight in winning_parts]
    selected_valid = blend(winning_parts, "valid_prediction")
    selected_time = blend(winning_parts, "time_prediction")

    validation = []
    temporal_validation = []
    temporal_baseline_mae = mae(
        y_time, np.full(len(y_time), np.median(data["y_train"][fit_indices]), dtype="float64")
    )
    for name in names:
        candidate = candidates[name]
        single = score([(name, 1.0)])
        validation.append({
            "name": name, "family": candidate["family"], "target_mode": candidate["target_mode"],
            "mae": single["validation_mae"], "clean_mae": single["clean_validation_mae"],
            "model_unique_mae": single["model_unique_validation_mae"],
            "best_iteration": candidate["iterations"],
        })
        temporal_validation.append({
            "name": name, "family": candidate["family"], "target_mode": candidate["target_mode"],
            "cutoff": CUTOFF, "end": END,
            "mae": single["temporal_mae"],
            "seen_project_mae": single["seen_project_mae"],
            "unseen_project_mae": single["unseen_project_mae"],
            "reweighted_temporal_mae": single["reweighted_temporal_mae"],
            "training_rows": int(len(fit_indices)), "validation_rows": int(len(time_indices)),
            "seen_project_rows": int(seen_time_mask.sum()),
            "unseen_project_rows": int((~seen_time_mask).sum()),
            "best_iteration": candidate["temporal_metadata"].get("best_iteration"),
            "median_baseline_mae": temporal_baseline_mae,
        })

    valid_prediction_path = EXPERIMENT_DIR / "selected_valid.npy"
    time_prediction_path = EXPERIMENT_DIR / "selected_temporal_valid.npy"
    time_indices_path = EXPERIMENT_DIR / "selected_temporal_indices.npy"
    selection_basis = (
        "只使用提供的本地無標籤測試資料之未見建案比例，將時間驗證集的已見／未見建案 MAE 重加權；"
        "比較全部單模型與兩兩模型的 0.25／0.50／0.75 保守粗網格，權重和為 1。"
        "以重加權時間 MAE 最小者選型，完全平手時才使用原驗證集 MAE。"
        "未查詢外部房價、內政部答案或測試價格標籤；未見建案重加權是分布近似，並非 Kaggle 成績保證。"
    )
    selection = {
        "name": winning_scores["name"],
        "components": components,
        **{key: winning_scores[key] for key in (
            "validation_mae", "clean_validation_mae", "model_unique_validation_mae",
            "temporal_mae", "reweighted_temporal_mae", "seen_project_mae",
            "unseen_project_mae", "test_unseen_project_rate",
        )},
        "test_unseen_project_rows": int(test_unseen_mask.sum()),
        "test_rows": int(len(test_projects)),
        "candidate_names": names,
        "schemes_evaluated": expected_scheme_count,
        "selection_basis": selection_basis,
        "validation_predictions_path": valid_prediction_path.relative_to(ROOT).as_posix(),
        "temporal_predictions_path": time_prediction_path.relative_to(ROOT).as_posix(),
        "temporal_indices_path": time_indices_path.relative_to(ROOT).as_posix(),
    }
    feature_hash = sha256(ROOT / "src" / "house_features.py")
    config = {
        "feature_source_sha256": feature_hash,
        "feature_columns": list(data["X_train"].columns),
        "categorical_columns": list(data["categorical_columns"]),
        "components": components,
        "selection_basis": selection_basis,
        "candidate_names": names,
        "source_sha256": data["metadata"]["source_sha256"],
        "selection_artifact_sha256": {
            name: candidate["artifact_sha256"] for name, candidate in candidates.items()
        },
        "model_unique_validation_mask_sha256": sha256(model_unique_mask_path),
    }
    baseline_prediction = np.full(len(y_valid), np.median(data["y_train"]), dtype="float64")
    results = {
        "metric": "MAE",
        "unit": "TWD",
        "baseline": {
            "name": "training_median",
            "mae": mae(y_valid, baseline_prediction),
            "clean_mae": mae(y_valid[clean_mask], baseline_prediction[clean_mask]),
        },
        "validation": validation,
        "temporal_validation": temporal_validation,
        "selection": selection,
        "final_submission": None,
        "constraints": {
            "no_external_property_data": True,
            "no_government_answer_lookup": True,
        },
        "validation_masks": {
            "raw_field_unique_rows": int(clean_mask.sum()),
            "model_feature_unique_rows": int(model_unique_mask.sum()),
            "total_rows": int(len(y_valid)),
        },
        "feature_source_sha256": feature_hash,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame([item["scores"] for item in evaluated])
    grid.insert(0, "rank", range(1, len(grid) + 1))
    grid.to_csv(REPORT_DIR / "ensemble_search.csv", index=False, encoding="utf-8-sig")
    _write_json(MODEL_DIR / "final_config.json", config)
    _write_json(REPORT_DIR / "results.json", results)
    np.save(valid_prediction_path, selected_valid)
    np.save(time_prediction_path, selected_time)
    np.save(time_indices_path, time_indices)
    print(json.dumps({
        "selected": winning_scores["name"],
        "schemes_evaluated": expected_scheme_count,
        "reweighted_temporal_mae": winning_scores["reweighted_temporal_mae"],
        "validation_mae": winning_scores["validation_mae"],
        "test_unseen_project_rate": test_unseen_rate,
        "config": "models/final_config.json",
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    args = parser.parse_args()
    run(args.names)


if __name__ == "__main__":
    main()
