"""Original 87/21 L1 unit LightGBM with saved fit-date weights; no data loader."""
from pathlib import Path
import hashlib
import json
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import train_lightgbm as original

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'runs/v1/models/final_config.json'
COMPONENT = 'recency_lightgbm_l1_unit63'
TARGET_FORMULA = 'y_TWD / (effective_area * 1e4)'
INVERSE_FORMULA = 'maximum(raw_unit * (effective_area * 1e4), 0)'
WEIGHT_FORMULA = 'original_stage_area_weight * float64(saved_R63_train_sample_weight_float32)'
EXPECTED_PARAMETERS = {
    'objective': 'regression_l1', 'metric': 'None', 'n_estimators': 3999,
    'learning_rate': .04, 'num_leaves': 63, 'max_depth': -1, 'min_child_samples': 40,
    'colsample_bytree': .9, 'subsample': .9, 'subsample_freq': 1, 'reg_lambda': 2.,
    'cat_smooth': 20., 'cat_l2': 10., 'max_cat_to_onehot': 4, 'max_cat_threshold': 64,
    'random_state': 42, 'n_jobs': 4, 'verbosity': -1, 'deterministic': True, 'force_col_wise': True,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values_sha256(values, dtype='<f8'):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def _content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def _frame_sha(frame):
    return values_sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy(), '<u8')


def source_hashes():
    return {name: _sha(ROOT / name) for name in ('src/round81_recency_lgb.py',
        'src/train_lightgbm.py', 'src/model_common.py', 'src/prepare_data.py')}


def parameters():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    chosen = next(c for c in config['components'] if c['name'] == 'lightgbm_l1_unit63')
    assert chosen['family'] == 'lightgbm' and chosen['target_mode'] == 'unit'
    assert chosen['iterations'] == 3999
    result = {**chosen['parameters'], 'n_estimators': chosen['iterations'], 'n_jobs': 4}
    assert result == EXPECTED_PARAMETERS, 'Original fixed LightGBM parameters differ'
    return result


def _check_X(X):
    assert isinstance(X, pd.DataFrame) and X.shape[1] == 87 and X.columns.is_unique
    assert 'building_area' in X and 'parking_area' in X


def area_weights(X, stage):
    _check_X(X)
    assert stage in ('development', 'confirmation', 'auxiliary'), stage
    # Preserve the original R3 versus R6 floating-point operation order.
    weights = (original.price_factor(X, 'unit') / 1e6 if stage != 'auxiliary'
               else original.effective_area(X) / 100)
    assert weights.dtype == np.float64 and np.isfinite(weights).all() and (weights > 0).all()
    return weights


def _summary(values):
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and len(v) and np.isfinite(v).all() and (v > 0).all()
    total, squares = float(v.sum(dtype=np.float64)), float(np.dot(v, v))
    assert np.isfinite(total) and np.isfinite(squares) and squares > 0
    return {'rows': len(v), 'min': float(v.min()), 'max': float(v.max()), 'mean': float(v.mean()),
        'sum': total, 'sum_squares': squares, 'effective_sample_size': total * total / squares}


def check_state(state, X):
    _check_X(X)
    assert list(X.columns) == state['feature_columns'] and state['feature_count'] == 87
    assert state['categorical_feature_count'] == len(state['categorical_columns']) == 21
    assert list(state['safe_feature_names'].values()) == [f'f{i:03d}' for i in range(87)] or (
        [state['safe_feature_names'][c] for c in state['feature_columns']] == [f'f{i:03d}' for i in range(87)])
    assert state['parameters'] == parameters() and state['base_configuration_sha256'] == _sha(CONFIG_PATH)
    assert state['source_sha256'] == source_hashes()
    assert state['state_content_sha256'] == _content_sha({k: v for k, v in state.items() if k != 'state_content_sha256'})


def check_model(model, state):
    booster = model.booster_ if isinstance(model, lgb.LGBMRegressor) else model
    assert isinstance(booster, lgb.Booster)
    names = [state['safe_feature_names'][c] for c in state['feature_columns']]
    assert booster.feature_name() == names and booster.num_feature() == 87
    assert booster.num_trees() == booster.current_iteration() == 3999
    p = booster.params
    def param(*aliases):
        present = [p[k] for k in aliases if k in p]
        assert present, ('Missing native parameter', aliases)
        assert all(v == present[0] for v in present), ('Conflicting native aliases', aliases)
        return present[0]
    assert param('objective') == 'regression_l1'
    checks = {
        ('num_leaves',): 63, ('max_depth',): -1, ('min_data_in_leaf', 'min_child_samples'): 40,
        ('feature_fraction', 'colsample_bytree'): .9, ('bagging_fraction', 'subsample'): .9,
        ('bagging_freq', 'subsample_freq'): 1, ('lambda_l2', 'reg_lambda'): 2.,
        ('cat_smooth',): 20., ('cat_l2',): 10., ('max_cat_to_onehot',): 4,
        ('max_cat_threshold',): 64, ('seed', 'random_state'): 42, ('num_threads', 'n_jobs'): 4,
        ('deterministic',): True, ('force_col_wise',): True, ('learning_rate',): .04,
    }
    for aliases, wanted in checks.items():
        actual = param(*aliases)
        assert actual == wanted, (aliases, actual, wanted)
    return {'actual_trees': int(booster.num_trees()), 'actual_iterations': int(booster.current_iteration()),
        'actual_feature_count': int(booster.num_feature()), 'actual_objective': str(param('objective')),
        'actual_learning_rate': float(param('learning_rate')), 'actual_num_leaves': int(param('num_leaves')),
        'actual_seed': int(param('seed', 'random_state'))}


def predict_raw(model_or_booster, state, X):
    check_state(state, X)
    check_model(model_or_booster, state)
    # Supplying saved maps prevents learning anything from this query frame.
    encoded, _, safe, cats = original.encode_features({'train': X}, state['categorical_columns'],
                                                     category_maps=state['category_maps'])
    assert safe == state['safe_feature_names'] and cats == state['categorical_safe_names']
    booster = model_or_booster.booster_ if isinstance(model_or_booster, lgb.LGBMRegressor) else model_or_booster
    raw = np.asarray(booster.predict(encoded['train'], num_iteration=3999, num_threads=4), dtype=np.float64)
    assert raw.shape == (len(X),) and np.isfinite(raw).all()
    return raw


def inverse_transform(raw, X):
    _check_X(X)
    if isinstance(raw, pd.Series):
        assert raw.index.equals(X.index)
    z = np.asarray(raw, dtype=np.float64)
    assert z.shape == (len(X),) and np.isfinite(z).all()
    with np.errstate(over='raise', invalid='raise'):
        result = np.maximum(z * original.price_factor(X, 'unit'), 0.)
    assert result.dtype == np.float64 and np.isfinite(result).all() and (result >= 0).all()
    return result


def fit_model(X, y, cats, date_weights32, stage):
    """One fixed fit; caller verifies canonical indices and saved weight provenance."""
    started = time.monotonic()
    _check_X(X)
    assert len(X) and len(cats) == len(set(cats)) == 21 and set(cats).issubset(X.columns)
    for values in (y, date_weights32):
        if isinstance(values, pd.Series):
            assert values.index.equals(X.index)
    target = np.asarray(y, dtype=np.float64)
    dates = np.asarray(date_weights32)
    assert target.shape == (len(X),) and np.isfinite(target).all() and (target >= 0).all()
    assert dates.dtype == np.float32 and dates.shape == target.shape and np.isfinite(dates).all() and (dates > 0).all()
    dates = dates.copy()
    area = area_weights(X, stage)
    factor = original.price_factor(X, 'unit')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        transformed = target / factor
        weights = area * dates.astype(np.float64)
    assert np.isfinite(transformed).all() and np.isfinite(weights).all() and (weights > 0).all()
    encoded, maps, safe, categorical = original.encode_features({'train': X}, cats)
    config = parameters()
    state = {'component': COMPONENT, 'feature_columns': list(X.columns), 'feature_count': 87,
        'categorical_columns': list(cats), 'categorical_feature_count': 21,
        'safe_feature_names': safe, 'categorical_safe_names': categorical, 'category_maps': maps,
        'fit_rows': len(X), 'fit_original_feature_values_sha256': _frame_sha(X),
        'fit_encoded_feature_values_sha256': _frame_sha(encoded['train']),
        'category_mapping_fit_split': 'fit_only', 'unseen_category_code': -1,
        'category_order': 'pd.unique first occurrence in fit rows', 'parameters': config,
        'base_configuration_sha256': _sha(CONFIG_PATH), 'source_sha256': source_hashes(),
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'fit_target_used_in_feature_state': False, 'query_used_in_feature_state': False}
    state['state_content_sha256'] = _content_sha(state)
    fit_hashes = {'fit_target_values_sha256': values_sha256(target),
        'fit_transformed_target_values_sha256': values_sha256(transformed),
        'fit_area_weights_sha256': values_sha256(area), 'fit_date_weights_sha256': values_sha256(dates, '<f4'),
        'fit_sample_weights_sha256': values_sha256(weights), 'fit_factor_sha256': values_sha256(factor)}
    model = lgb.LGBMRegressor(**config)
    tick = time.monotonic()
    model.fit(encoded['train'], transformed, sample_weight=weights, categorical_feature=categorical)
    fit_seconds = time.monotonic() - tick
    native = check_model(model, state)
    check_state(state, X)
    assert _frame_sha(X) == state['fit_original_feature_values_sha256']
    assert values_sha256(target) == fit_hashes['fit_target_values_sha256']
    assert values_sha256(weights) == fit_hashes['fit_sample_weights_sha256']
    assert values_sha256(date_weights32, '<f4') == fit_hashes['fit_date_weights_sha256']
    report = {'family': 'lightgbm', 'component': COMPONENT, 'stage': stage, 'parameters': config,
        'fit_rows': len(X), 'feature_count': 87, 'categorical_feature_count': 21,
        'target_mode': 'unit', 'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'prediction_units': 'TWD', 'sample_weight': WEIGHT_FORMULA,
        'original_area_weight_formula': 'factor / 1e6' if stage != 'auxiliary' else 'effective_area / 100',
        'fit_date_weights_hash_encoding': 'ordered little-endian float32',
        'fit_sample_weights_hash_encoding': 'ordered little-endian float64',
        'fit_date_weights_summary': _summary(dates), 'fit_sample_weights_summary': _summary(weights),
        'fit_area_weights_summary': _summary(area), 'date_weights_recomputed': False,
        'date_weights_renormalized': False, 'combined_weights_renormalized': False,
        'date_weight_provenance_verified_in_helper': False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks': True,
        'weights_use_query': False, 'weights_use_fit_target': False, 'query_labels_used': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False, 'new_fits': 1,
        'reference_or_spatial_applied_here': False, 'extra_price_cap_applied': False,
        'feature_state_content_sha256': state['state_content_sha256'],
        'fit_original_feature_values_sha256': state['fit_original_feature_values_sha256'],
        'fit_encoded_feature_values_sha256': state['fit_encoded_feature_values_sha256'],
        'base_configuration_sha256': state['base_configuration_sha256'], 'source_sha256': source_hashes(),
        'fitted_booster_parameters': dict(model.booster_.params),
        'runtime': {'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'fit_elapsed_seconds': fit_seconds, 'elapsed_seconds': time.monotonic() - started,
        **fit_hashes, **native}
    return model, state, report
