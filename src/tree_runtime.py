"""Portable native-tree and reference/spatial inference for the fixed v22 recipe.

Only artifact files below the caller-supplied artifacts_dir are read. Historical
pure inference helpers are shipped byte-for-byte in the adjacent closure_modules.
No training, predictor .npy loading, selector state, or original workspace path.
The caller must verify the private artifact manifest before unpickling states.
"""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

_CLOSURE = Path(__file__).resolve().parent / 'closure_modules'
sys.path.insert(0, str(_CLOSURE))
from round6_geo import augment_features, content_sha
from round2_neighborhood import predict_reference
from round8_spatial_reference import predict as spatial_predict

UNIT_MODEL = 'outputs/round79/final_model/model/model.cbm'
UNIT_STATE = 'outputs/round79/final_model/model/feature_state.json'
LOG_MODEL = 'outputs/round46/trial/model/model.cbm'
LOG_STATE = 'outputs/round46/trial/model/feature_state.json'
RAW_MODEL = 'models/final/catboost_raw.cbm'
LGB_MODEL = 'models/final/lightgbm_l1_unit63/model.txt'
LGB_MAPS = 'models/final/lightgbm_l1_unit63/category_maps.json'
REFERENCE = 'outputs/round2/neighborhood/final_reference.pkl'
SPATIAL = 'outputs/round39/trial/artifacts/spatial_state.pkl'
SCHEMA = 'reports/feature_schema.json'


def _json(root, name):
    return json.loads((root/name).read_text(encoding='utf-8-sig'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def effective_area(X):
    building = X['building_area'].to_numpy(dtype='float64')
    parking = X['parking_area'].to_numpy(dtype='float64')
    return np.where(np.isfinite(building) & (building > 0), building,
                    np.where(np.isfinite(parking) & (parking > 0), parking, 1.0))


def _check_state(state, X, frame, cats):
    _require(state['state_content_sha256'] == content_sha({k:v for k,v in state.items()
             if k != 'state_content_sha256'}), 'Feature state content hash differs')
    _require(state['all_feature_names'] == list(frame), 'Saved state feature order differs')
    _require(state['categorical_columns'] == cats, 'Saved state category order differs')
    _require(state['geo_state']['original_columns'] == list(X), 'Original feature order differs')
    _require(state['geo_state']['original_dtypes'] == {c:str(X[c].dtype) for c in X},
             'Original feature dtypes differ from training')


def _catboost(root, model_path, frame, cats, inverse, threads, state=None, X=None):
    if state is not None:
        _check_state(state, X, frame, cats)
    model = CatBoostRegressor()
    model.load_model(str(root/model_path))
    _require(list(model.feature_names_) == list(frame), 'Native CatBoost feature order differs')
    _require(model.tree_count_ == 2999, 'Expected 2999 saved CatBoost trees')
    expected_cats = sorted(frame.columns.get_loc(c) for c in cats)
    _require(model.get_cat_feature_indices() == expected_cats, 'Native category indices differ')
    z = np.asarray(model.predict(Pool(frame, cat_features=cats, thread_count=threads),
                                 thread_count=threads), dtype='float64')
    _require(z.shape == (len(frame),) and np.isfinite(z).all(), 'Invalid native tree vector')
    with np.errstate(over='raise', invalid='raise'):
        values = np.maximum(inverse(z), 0.).astype('float64', copy=False)
    _require(np.isfinite(values).all(), 'Invalid TWD inverse')
    return values


def _encode_lightgbm(X, maps):
    """Exact train_lightgbm.encode_features inference arithmetic with frozen maps."""
    encoded = {}
    for i, column in enumerate(X):
        if column in maps:
            values = X[column].fillna('__MISSING__').astype(str)
            encoded[f'f{i:03d}'] = values.map(maps[column]).fillna(-1).astype('int32')
        else:
            encoded[f'f{i:03d}'] = pd.to_numeric(X[column], errors='coerce').replace(
                [np.inf, -np.inf], np.nan).astype('float32')
    return pd.DataFrame(encoded, index=X.index)


def predict_trees(X, G, artifacts_dir, threads=4):
    """Recompute four full TWD components from native models; never read predictions."""
    root = Path(artifacts_dir).resolve()
    _require(isinstance(threads, int) and threads >= 1, 'threads must be a positive integer')
    schema = _json(root, SCHEMA)
    _require(list(X) == schema['columns'] and len(X.columns) == 87, 'Original 87-feature schema differs')
    _require(not {'編號', '總價元', 'target', 'y'} & set(X.columns), 'IDs/labels cannot be model features')
    frame = augment_features(X, G)
    cats = list(schema['categorical_columns'])
    geo_cats = cats + ['geo_match_status', 'geo_grid_002']
    factor = effective_area(X) * 1e4
    components = {}
    components['C_unit'] = _catboost(root, UNIT_MODEL, frame, geo_cats,
        lambda z: z * factor, threads, _json(root, UNIT_STATE), X)
    components['C_log'] = _catboost(root, LOG_MODEL, frame, geo_cats,
        lambda z: np.expm1(z) * 1e6, threads, _json(root, LOG_STATE), X)
    components['C_raw'] = _catboost(root, RAW_MODEL, X, cats, lambda z: z * 1e6, threads)
    encoded = _encode_lightgbm(X, _json(root, LGB_MAPS))
    model = lgb.Booster(model_file=str(root/LGB_MODEL))
    _require(model.feature_name() == list(encoded), 'Native LightGBM feature order differs')
    _require(model.num_trees() == 3999, 'Expected 3999 saved LightGBM trees')
    z = np.asarray(model.predict(encoded, num_iteration=3999, num_threads=threads), dtype='float64')
    with np.errstate(over='raise', invalid='raise'):
        components['L_unit'] = np.maximum(z * factor, 0.)
    _require(all(p.shape == (len(X),) and np.isfinite(p).all() and p.dtype == np.float64
                 for p in components.values()), 'Invalid tree component vectors')
    return components


def load_states(artifacts_dir):
    """Load trusted, manifest-verified fitted reference and spatial states."""
    root = Path(artifacts_dir).resolve()
    with (root/REFERENCE).open('rb') as stream:
        reference = pickle.load(stream)
    with (root/SPATIAL).open('rb') as stream:
        spatial = pickle.load(stream)
    return reference, spatial


def apply_branch(X, GS, base, reference, state):
    """Apply fit-only comparable/trend and frozen spatial correction to one base."""
    prediction, diagnostic, _ = predict_reference(reference, X, base,
        variant='trend', return_diagnostics=True)
    final, _, _ = spatial_predict(state, X, GS, base, prediction, diagnostic)
    return final
