"""R85 fixed square-root-area CatBoost representation.

Reuse R79's 95-column view, native configuration and saved float32 date weights.
F = sqrt(A/100), target = (y/1e6)/F, weight = F*date_weight.
The R19 pure-X feature state is retained as historical provenance; every actual
prediction must use this module's inverse, never R19's unit inverse.
No loader, query labels, date recomputation, clipping cap, reference or spatial fit.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round79_recency_unit as baseline

original = baseline.original
base = baseline.base
COMPONENT = 'geography_recency_sqrt_area_catboost'
TARGET_MODE = 'sqrt_area'
SCALE_FORMULA = 'sqrt(effective_area(X) / 100)'
TARGET_FORMULA = '(y_TWD / 1e6) / sqrt(effective_area(X) / 100)'
INVERSE_FORMULA = 'max((z * sqrt(effective_area(X) / 100)) * 1e6, 0)'
WEIGHT_FORMULA = 'sqrt(effective_area(X) / 100) * float64(saved_R63_train_sample_weight_float32)'
LOSS_EQUIVALENCE = 'F*date_weight*abs((y/1e6)/F-z) = date_weight*abs(y-(z*F)*1e6)/1e6, F=sqrt(A/100), before nonnegative projection'
parameters = baseline.parameters
effective_area = baseline.effective_area
fit_feature_state = baseline.fit_feature_state
check_state = baseline.check_state
check_model = baseline.check_model
augment_features = baseline.augment_features
categories = baseline.categories
values_sha256 = baseline.values_sha256
FEATURE_RULES = original._copy_json(baseline.FEATURE_RULES)
FEATURE_RULES.update(component=COMPONENT, fit_target=TARGET_FORMULA,
    prediction=INVERSE_FORMULA, sample_weight=WEIGHT_FORMULA,
    factor=SCALE_FORMULA, loss_equivalence=LOSS_EQUIVALENCE,
    fixed_area_exponent=0.5, prediction_output_units='TWD',
    raw_output_units='million TWD divided by sqrt(A/100)',
    inherited_R19_state_is_feature_provenance_not_active_target=True)


def source_hashes():
    return {**baseline.source_hashes(),
        'src/round85_sqrt_area.py': original.sha256(Path(__file__))}


def scale_factor(X):
    """Positive dimensionless F; this is not the old unit-price TWD factor."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = np.sqrt(effective_area(X) / 100.0)
    except FloatingPointError as exc:
        raise ValueError('Square-root area factor failed') from exc
    base.need(result.shape == (len(X),) and np.isfinite(result).all()
              and (result > 0).all(), 'Square-root factor must be finite and positive')
    return np.ascontiguousarray(result, dtype=np.float64)


def transform_target(y, X):
    raw = original._aligned_vector(y, X, 'Fit target')
    base.need((raw >= 0).all(), 'Fit target must be nonnegative')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = (raw / 1e6) / scale_factor(X)
    except FloatingPointError as exc:
        raise ValueError('Square-root target overflowed') from exc
    base.need(np.isfinite(result).all(), 'Square-root target must be finite')
    return np.ascontiguousarray(result, dtype=np.float64)


def inverse_transform(z, X):
    raw = original._aligned_vector(z, X, 'Model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            price = (raw * scale_factor(X)) * 1e6
    except FloatingPointError as exc:
        raise ValueError('Square-root TWD inverse overflowed') from exc
    base.need(np.isfinite(price).all(), 'Inverse TWD values must be finite')
    return np.maximum(price, 0).astype(np.float64, copy=False)


def sample_weights(X, saved_date_weights):
    dates32 = baseline._date_values(saved_date_weights, X)
    try:
        with np.errstate(over='raise', invalid='raise'):
            values = scale_factor(X) * dates32.astype(np.float64)
    except FloatingPointError as exc:
        raise ValueError('Square-root/date weights overflowed') from exc
    base.need(np.isfinite(values).all() and (values > 0).all(),
              'Combined weights must be finite and positive')
    return np.ascontiguousarray(values, dtype=np.float64)


def fit_model(X, y, geo, cats, saved_date_weights):
    """Return model, unchanged R19 pure-X state, actual R85 metadata."""
    started = time.monotonic()
    original._check_X(X)
    base.need(len(X) > 0, 'Cannot fit empty X')
    raw = original._aligned_vector(y, X, 'Fit target')
    transformed = transform_target(raw, X)
    dates32 = baseline._date_values(saved_date_weights, X)
    scale = scale_factor(X)
    weights = sample_weights(X, dates32)
    dates_sha = hashlib.sha256(dates32.astype('<f4', copy=False).tobytes()).hexdigest()
    raw_sha, target_sha = values_sha256(raw), values_sha256(transformed)
    weight_sha, scale_sha = values_sha256(weights), values_sha256(scale)
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    config = parameters()
    base.need(config['iterations'] == 2999 and config['random_seed'] == 42
              and config['depth'] == 8 and config['learning_rate'] == .065,
              'Expected the fixed R79 2999/depth8/seed42 parameters')
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, weight=weights,
                   cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == target_sha,
              'Fit labels changed')
    base.need(values_sha256(scale_factor(X)) == scale_sha
              and values_sha256(sample_weights(X, saved_date_weights)) == weight_sha,
              'Fit scaling/weights changed')
    base.need(hashlib.sha256(baseline._date_values(saved_date_weights, X).astype('<f4', copy=False).tobytes()).hexdigest()
              == dates_sha, 'Saved date weights changed')
    report = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'scale_formula': SCALE_FORMULA, 'fixed_area_exponent': .5,
        'raw_output_units': 'million TWD divided by sqrt(A/100)', 'prediction_units': 'TWD',
        'parameters': original._copy_json(config),
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
        'fit_effective_area_sha256': values_sha256(effective_area(X)),
        'fit_scale_factor_sha256': scale_sha, 'fit_scale_factor_summary': baseline._summary(scale),
        'fit_date_weights_sha256': dates_sha, 'fit_date_weights_hash_encoding': 'ordered little-endian float32',
        'fit_date_weights_summary': baseline._summary(dates32),
        'fit_sample_weights_sha256': weight_sha, 'fit_sample_weights_hash_encoding': 'ordered little-endian float64',
        'fit_sample_weights_summary': baseline._summary(weights),
        'fit_target_values_sha256': raw_sha, 'fit_transformed_target_values_sha256': target_sha,
        'target_value_hash_encoding': 'ordered little-endian float64',
        'loss_equivalence_before_nonnegative_inverse_clipping': LOSS_EQUIVALENCE,
        'feature_rules': original._copy_json(FEATURE_RULES),
        'legacy_state_training_fields_are_provenance_only': True,
        'legacy_state_target_mode': state['target_mode'],
        'legacy_state_weight_policy': original.WEIGHT_FORMULA,
        'legacy_state_component': state['component'],
        'legacy_state_unit_factor_sha256': state['fit_factor_sha256'],
        'feature_state_content_sha256': state['state_content_sha256'],
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'source_sha256': source_hashes(), 'base_configuration_sha256': state['base_configuration_sha256'],
        'query_labels_used': False, 'weights_use_fit_target': False, 'weights_use_query': False,
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False,
        'reference_or_trend_applied_here': False, 'spatial_applied_here': False,
        'extra_price_cap_applied': False, 'calibration_applied': False,
        'elapsed_seconds': time.monotonic() - started, **actual,
    }
    return model, state, report


def predict_model(model, state, X, geo, cats):
    """Return float64 TWD using R85 inverse; no query labels, weights or trend."""
    check_state(state, X, cats)
    check_model(model, state)
    frame = augment_features(X, geo)
    base.need(list(frame.columns) == state['all_feature_names'], 'Query columns differ')
    z = model.predict(Pool(frame, cat_features=state['categorical_columns'], thread_count=4),
                      thread_count=4)
    return inverse_transform(z, X)


def self_check():
    """Bounded mathematical checks only; imports no data and performs no fit."""
    checks = []
    def ok(name, truth):
        if not truth:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, operation):
        try:
            operation()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name + ' did not reject')
    index = pd.Index([2, 9, 4, 8, 5, 11, 7, 13])
    X = pd.DataFrame({'building_area':[100.,400.,25.,0.,np.nan,np.inf,-1.,1e4],
                      'parking_area':[30.,1.,80.,25.,0.,4.,np.inf,0.]}, index=index)
    before = X.copy(deep=True)
    A = np.array([100.,400.,25.,25.,1.,4.,1.,1e4])
    F = np.sqrt(A/100)
    y = pd.Series([1e6,4e6,2e6,0.,5e6,6e6,7e6,8e6], index=index)
    dates = np.array([1.,.5,.25,2.,1.,.125,1.5,3.], dtype=np.float32)
    z = np.array([.5,3.,1.,-1.,4.,0.,-2.,8.])
    ok('same_effective_area_fallbacks', np.array_equal(effective_area(X), A))
    ok('fixed_sqrt_dimensionless_scale', np.array_equal(scale_factor(X), F))
    ok('transform_correct_million_scale', np.array_equal(transform_target(y, X), (y.to_numpy()/1e6)/F))
    ok('inverse_correct_one_million_and_nonnegative_projection',
       np.array_equal(inverse_transform(z, X), np.maximum((z*F)*1e6,0)))
    w = sample_weights(X, dates)
    ok('saved_float32_dates_promoted_without_renormalization',
       np.array_equal(w, F*dates.astype(np.float64)) and w.dtype == np.float64)
    lhs = w*np.abs(transform_target(y, X)-z)
    rhs = dates.astype(np.float64)*np.abs(y.to_numpy()-(z*F)*1e6)/1e6
    ok('weighted_L1_matches_recency_total_TWD_L1_before_projection',
       np.allclose(lhs,rhs,rtol=2e-15,atol=2e-14))
    ok('target_inverse_roundtrip', np.allclose(inverse_transform(transform_target(y,X),X),
       y.to_numpy(),rtol=2e-15,atol=1e-8))
    ok('hundred_m2_matches_raw_million_target', transform_target(y, X)[0] == 1.)
    ok('quarter_and_four_times_area_multiply_scale_by_half_and_two', F[2] == .5 and F[1] == 2.)
    ok('same_rows_and_no_feature_mutation', X.equals(before))
    ok('query_labels_absent_from_predict_API',
       list(inspect.signature(predict_model).parameters) == ['model','state','X','geo','cats'])
    ok('fit_state_API_no_y_or_query',
       list(inspect.signature(fit_feature_state).parameters) == ['X','geo','cats'])
    reject('misordered_target_series_rejected', lambda: transform_target(y.iloc[::-1], X))
    reject('misordered_dates_series_rejected', lambda: sample_weights(X,pd.Series(dates,index=index[::-1])))
    reject('float64_date_input_rejected', lambda: sample_weights(X, dates.astype(np.float64)))
    reject('zero_date_input_rejected', lambda: sample_weights(X,np.zeros(len(X),dtype=np.float32)))
    reject('NaN_date_input_rejected', lambda: sample_weights(X,np.full(len(X),np.nan,dtype=np.float32)))
    reject('negative_target_rejected', lambda: transform_target(-np.ones(len(X)), X))
    reject('NaN_target_rejected', lambda: transform_target(np.full(len(X),np.nan), X))
    reject('wrong_target_shape_rejected', lambda: transform_target(np.ones((len(X),1)), X))
    reject('nonfinite_prediction_rejected', lambda: inverse_transform(np.full(len(X),np.inf), X))
    reject('prediction_overflow_rejected', lambda: inverse_transform(np.full(len(X),np.finfo(float).max), X))
    tiny=X.copy();tiny['building_area']=np.nextafter(0.,1.)
    reject('underflow_zero_scale_rejected', lambda: scale_factor(tiny))
    return {'status':'passed','check_count':len(checks),'checks':checks,
            'synthetic_only':True,'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False), flush=True)
