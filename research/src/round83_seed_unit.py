"""R83: R79 unit/date-weighted CatBoost, with random_seed=137 only.

The immutable R19 inner state records feature provenance and historical seed42 /
area-only training declarations. The wrapper and fit metadata describe this fit.
No legacy module globals are modified. No loader, date recomputation, reference,
spatial operation, or query-label input is provided here.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
from catboost import CatBoostRegressor, Pool
import round79_recency_unit as prior

original = prior.original
base = original.base
COMPONENT = 'geography_recency_unit_seed137_catboost'
TARGET_MODE = prior.TARGET_MODE
TARGET_FORMULA = prior.TARGET_FORMULA
INVERSE_FORMULA = prior.INVERSE_FORMULA
WEIGHT_FORMULA = prior.WEIGHT_FORMULA
SEED = 137
effective_area = original.effective_area
transform_target = original.transform_target
inverse_transform = original.inverse_transform
augment_features = original.augment_features
categories = original.categories
values_sha256 = original.values_sha256
FEATURE_RULES = original._copy_json(prior.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, seed=SEED,
    changed_parameter_names=['random_seed'], legacy_feature_state_unchanged=True)


def source_hashes():
    return {**prior.source_hashes(),
            'src/round83_seed_unit.py': original.sha256(Path(__file__))}


def parameters():
    baseline = prior.parameters()
    base.need(baseline['random_seed'] == 42, 'Expected original R79 seed42')
    result = original._copy_json(baseline)
    result['random_seed'] = SEED
    base.need([k for k in result if result[k] != baseline[k]] == ['random_seed'],
              'Only random_seed may differ from R79')
    return result


def fit_feature_state(X, geo, cats):
    legacy = original.fit_feature_state(X, geo, cats)
    state = {'format_version': 1, 'component': COMPONENT,
        'legacy_feature_state': legacy,
        'legacy_feature_state_content_sha256': legacy['state_content_sha256'],
        'legacy_state_training_fields_are_provenance_only': True,
        'feature_rules': original._copy_json(FEATURE_RULES),
        'parameters': parameters(), 'baseline_parameters': prior.parameters(),
        'changed_parameter_names': ['random_seed'],
        'feature_columns': list(legacy['all_feature_names']),
        'categorical_columns': list(legacy['categorical_columns']),
        'fit_rows': legacy['fit_rows'], 'fit_state_uses_y': False,
        'fit_state_uses_query': False, 'query_labels_used': False,
        'source_sha256': source_hashes()}
    state['state_content_sha256'] = base.content_sha(state)
    return state


def _check_wrapper(state):
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'Wrapper state changed')
    base.need(state['component'] == COMPONENT and state['feature_rules'] == FEATURE_RULES
              and state['source_sha256'] == source_hashes(), 'Wrapper source or rules changed')
    base.need(state['parameters'] == parameters() and state['baseline_parameters'] == prior.parameters()
              and state['changed_parameter_names'] == ['random_seed'], 'Actual seed137 configuration changed')
    legacy = state['legacy_feature_state']
    base.need(state['legacy_feature_state_content_sha256'] == legacy['state_content_sha256']
              and state['legacy_state_training_fields_are_provenance_only'] is True,
              'Legacy feature provenance changed')
    base.need(state['feature_columns'] == legacy['all_feature_names']
              and state['categorical_columns'] == legacy['categorical_columns']
              and state['fit_rows'] == legacy['fit_rows'], 'Wrapper schema differs from legacy features')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declarations changed')
    return legacy


def check_state(state, X, cats):
    # Schema-only query validation: never compare query values with fit-value hashes.
    original.check_state(_check_wrapper(state), X, cats)


def check_model(model, state):
    legacy = _check_wrapper(state)
    capacity = original.check_model(model, legacy)
    actual = model.get_all_params()
    base.need(actual['random_seed'] == SEED and actual['loss_function'] == 'MAE'
              and actual['eval_metric'] == 'MAE', 'Native seed137 or objective changed')
    lr = float(actual['learning_rate'])
    base.need(np.isfinite(lr) and np.isclose(lr, .065, rtol=1e-7, atol=0),
              'Native learning rate differs beyond serialization tolerance')
    return {**capacity, 'actual_random_seed': int(actual['random_seed']),
            'actual_learning_rate': lr, 'actual_loss_function': actual['loss_function'],
            'actual_eval_metric': actual['eval_metric']}


def predict_raw(model, state, X, geo, cats):
    check_state(state, X, cats)
    augmented = augment_features(X, geo)
    base.need(list(augmented.columns) == state['feature_columns'], 'Prediction feature order changed')
    check_model(model, state)
    pool = Pool(augmented, cat_features=state['categorical_columns'], thread_count=4)
    raw = np.asarray(model.predict(pool), dtype=np.float64)
    base.need(raw.shape == (len(X),) and np.isfinite(raw).all(), 'Invalid full-query native unit output')
    return raw


def fit_model(X, y, geo, cats, saved_date_weights):
    """Exactly one fit; the driver validates the saved R63 date-weight provenance."""
    started = time.monotonic()
    original._check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty frame')
    raw = original._aligned_vector(y, X, 'Fit target')
    transformed = transform_target(raw, X)
    dates32 = prior._date_values(saved_date_weights, X)
    date_sha = hashlib.sha256(dates32.astype('<f4', copy=False).tobytes(order='C')).hexdigest()
    area_weights = original.sample_weights(X)
    with np.errstate(over='raise', invalid='raise'):
        weights = np.ascontiguousarray(area_weights * dates32.astype(np.float64), dtype=np.float64)
    base.need(np.isfinite(weights).all() and (weights > 0).all(), 'Combined Pool weights must be positive finite')
    weight_sha = values_sha256(weights)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    legacy = state['legacy_feature_state']
    augmented = augment_features(X, geo)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == legacy['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == legacy['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == legacy['fit_model_feature_values_sha256'], 'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha, 'Fit labels changed')
    base.need(values_sha256(weights) == weight_sha
              and values_sha256(area_weights) == legacy['fit_sample_weights_sha256'], 'Fit weights changed')
    base.need(hashlib.sha256(prior._date_values(saved_date_weights, X).astype('<f4', copy=False).tobytes(order='C')).hexdigest()
              == date_sha, 'Saved date-weight input changed')
    report = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'prediction_units': 'TWD', 'parameters': original._copy_json(config),
        'baseline_parameters': original._copy_json(state['baseline_parameters']),
        'changed_parameter_names': ['random_seed'], 'added_feature_names': [], 'removed_feature_names': [],
        'iterations': 2999, 'actual_iterations': actual['actual_trees'], 'seed': SEED,
        'fit_rows': len(X), 'feature_count': 95, 'categorical_feature_count': 23,
        'feature_columns': list(augmented.columns), 'categorical_columns': list(state['categorical_columns']),
        'sample_weight': WEIGHT_FORMULA, 'weight_argument_passed_to_pool': True,
        'weight_dtype': 'float64', 'weights_normalized_to_mean_one': False,
        'date_weights_recomputed': False, 'date_weights_renormalized': False,
        'date_weight_provenance_verified_in_helper': False,
        'date_weight_provenance_requires_driver_R63_file_indices_and_value_hash_checks': True,
        'fit_effective_area_sha256': legacy['fit_effective_area_sha256'],
        'fit_factor_sha256': legacy['fit_factor_sha256'],
        'fit_area_weights_sha256': values_sha256(area_weights),
        'fit_date_weights_sha256': date_sha, 'fit_date_weights_hash_encoding': 'ordered little-endian float32',
        'fit_date_weights_summary': prior._summary(dates32),
        'fit_sample_weights_sha256': weight_sha, 'fit_sample_weights_hash_encoding': 'ordered little-endian float64',
        'fit_sample_weights_summary': prior._summary(weights),
        'loss_equivalence_before_nonnegative_inverse_clipping':
            '(A/100)*date_weight*abs(y/(A*1e4)-z) = date_weight*abs(y-A*1e4*z)/1e6',
        'feature_rules': original._copy_json(FEATURE_RULES),
        'legacy_state_training_fields_are_provenance_only': True,
        'legacy_state_weight_policy': original.WEIGHT_FORMULA,
        'legacy_state_component': legacy['component'],
        'legacy_state_parameters': original._copy_json(legacy['parameters']),
        'legacy_feature_state_content_sha256': legacy['state_content_sha256'],
        'feature_state_content_sha256': state['state_content_sha256'],
        'fit_original_feature_values_sha256': legacy['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': legacy['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': legacy['fit_model_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha, 'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered little-endian float64',
        'source_sha256': source_hashes(), 'base_configuration_sha256': legacy['base_configuration_sha256'],
        'query_labels_used': False, 'weights_use_fit_target': False, 'weights_use_query': False,
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'reference_or_trend_applied_here': False, 'spatial_applied_here': False,
        'extra_price_cap_applied': False, 'calibration_applied': False,
        'elapsed_seconds': time.monotonic() - started, **actual,
    }
    return model, state, report
