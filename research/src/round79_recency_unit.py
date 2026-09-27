"""R19 unit CatBoost with saved R63 fit-date weights multiplying its area weights.

Only this fit Pool changes. The unchanged R19 state describes feature construction
and historical area-only training provenance, not the actual weights of this fit.
No loader, date-weight calculation, query-label input, trend, or spatial operation.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round19_unit_geo as original

base = original.base
COMPONENT = 'geography_recency_unit_catboost'
TARGET_MODE = original.TARGET_MODE
TARGET_FORMULA = original.TARGET_FORMULA
INVERSE_FORMULA = original.INVERSE_FORMULA
WEIGHT_FORMULA = 'effective_area(X) / 100 * float64(saved_R63_train_sample_weight_float32)'
parameters = original.parameters
effective_area = original.effective_area
transform_target = original.transform_target
inverse_transform = original.inverse_transform
fit_feature_state = original.fit_feature_state
check_state = original.check_state
augment_features = original.augment_features
categories = original.categories
predict_model = original.predict_model
values_sha256 = original.values_sha256
FEATURE_RULES = original._copy_json(original.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, sample_weight=WEIGHT_FORMULA,
    weights_normalized_to_mean_one=False, saved_date_weights_dtype='float32',
    date_weights_recomputed=False, date_weights_renormalized=False,
    weights_use_fit_target=False, weights_use_query=False,
    legacy_state_training_fields_are_provenance_only=True)


def source_hashes():
    return {**original.source_hashes(),
        'src/round79_recency_unit.py': original.sha256(Path(__file__))}


def _date_values(saved_date_weights, X):
    if isinstance(saved_date_weights, pd.Series):
        base.need(saved_date_weights.index.equals(X.index), 'Date-weight Series index differs from fit X')
    raw = np.asarray(saved_date_weights)
    base.need(raw.dtype == np.dtype('float32') and raw.shape == (len(X),),
              'Expected saved R63 float32 weights in exact fit-row order')
    base.need(np.isfinite(raw).all() and (raw > 0).all(), 'Date weights must be finite and positive')
    return np.array(raw, dtype=np.float32, order='C', copy=True)


def _summary(values):
    v = np.asarray(values, dtype=np.float64)
    total = float(v.sum(dtype=np.float64))
    squares = float(np.dot(v, v))
    base.need(len(v) > 0 and np.isfinite(total) and np.isfinite(squares)
              and total > 0 and squares > 0, 'Weight summary overflow or empty weights')
    return {'rows': len(v), 'min': float(v.min()), 'max': float(v.max()),
            'mean': float(v.mean()), 'sum': total, 'sum_squares': squares,
            'effective_sample_size': total * total / squares}


def check_model(model, state):
    result = original.check_model(model, state)
    actual = model.get_all_params()
    base.need(actual['random_seed'] == 42 and actual['loss_function'] == 'MAE'
              and actual['eval_metric'] == 'MAE', 'Native seed or objective changed')
    lr = float(actual['learning_rate'])
    base.need(np.isfinite(lr) and np.isclose(lr, .065, rtol=1e-7, atol=0),
              'Native learning rate differs from fixed R19 recipe')
    # Fit and saved learning rates may differ within the declared serialization
    # tolerance; callers must not require exact equality for this one diagnostic.
    return {**result, 'actual_learning_rate': lr}


def fit_model(X, y, geo, cats, saved_date_weights):
    """Fit once; the driver verifies the saved R63 weights' file/value provenance."""
    started = time.monotonic()
    original._check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty frame')
    raw = original._aligned_vector(y, X, 'Fit target')
    transformed = transform_target(raw, X)
    dates32 = _date_values(saved_date_weights, X)
    date_sha = hashlib.sha256(dates32.astype('<f4', copy=False).tobytes(order='C')).hexdigest()
    area_weights = original.sample_weights(X)
    with np.errstate(over='raise', invalid='raise'):
        weights = np.ascontiguousarray(area_weights * dates32.astype(np.float64), dtype=np.float64)
    base.need(np.isfinite(weights).all() and (weights > 0).all(), 'Combined Pool weights must be finite and positive')
    weight_sha = values_sha256(weights)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    config = parameters()
    base.need(config['iterations'] == 2999 and config['random_seed'] == 42
              and config['depth'] == 8 and config['learning_rate'] == .065,
              'Expected unchanged fixed R19 parameters')
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'], 'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha, 'Fit labels changed')
    base.need(values_sha256(weights) == weight_sha
              and values_sha256(area_weights) == state['fit_sample_weights_sha256'], 'Fit weights changed')
    base.need(hashlib.sha256(_date_values(saved_date_weights, X).astype('<f4', copy=False).tobytes(order='C')).hexdigest()
              == date_sha, 'Saved date-weight input changed')
    report = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'prediction_units': 'TWD', 'parameters': original._copy_json(config),
        'baseline_parameters': original._copy_json(state['baseline_parameters']),
        'changed_parameter_names': [], 'added_feature_names': [], 'removed_feature_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'], 'seed': 42,
        'fit_rows': len(X), 'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns), 'categorical_columns': list(state['categorical_columns']),
        'sample_weight': WEIGHT_FORMULA, 'weight_argument_passed_to_pool': True,
        'weight_dtype': 'float64', 'weights_normalized_to_mean_one': False,
        'date_weights_recomputed': False, 'date_weights_renormalized': False,
        'date_weight_provenance_verified_in_helper': False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks': True,
        'fit_effective_area_sha256': state['fit_effective_area_sha256'],
        'fit_factor_sha256': state['fit_factor_sha256'],
        'fit_area_weights_sha256': values_sha256(area_weights),
        'fit_date_weights_sha256': date_sha, 'fit_date_weights_hash_encoding': 'ordered little-endian float32',
        'fit_date_weights_summary': _summary(dates32),
        'fit_sample_weights_sha256': weight_sha, 'fit_sample_weights_hash_encoding': 'ordered little-endian float64',
        'fit_sample_weights_summary': _summary(weights),
        'loss_equivalence_before_nonnegative_inverse_clipping':
            '(A/100)*date_weight*abs(y/(A*1e4)-z) = date_weight*abs(y-A*1e4*z)/1e6',
        'feature_rules': original._copy_json(FEATURE_RULES),
        'legacy_state_training_fields_are_provenance_only': True,
        'legacy_state_weight_policy': original.WEIGHT_FORMULA,
        'legacy_state_component': state['component'],
        'feature_state_content_sha256': state['state_content_sha256'],
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha, 'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered little-endian float64',
        'source_sha256': source_hashes(), 'base_configuration_sha256': state['base_configuration_sha256'],
        'query_labels_used': False, 'weights_use_fit_target': False, 'weights_use_query': False,
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'reference_or_trend_applied_here': False, 'spatial_applied_here': False,
        'extra_price_cap_applied': False, 'calibration_applied': False,
        'elapsed_seconds': time.monotonic() - started, **actual,
    }
    return model, state, report
