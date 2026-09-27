"""Fixed R21 additive fine-grid geographic categories and raw CatBoost.

Original R6 95 values, dtypes, order and index are preserved. Two fixed
0.005-degree category grids use the existing float32 coordinates cast to
float64; the second adds 0.0025 degrees to both axes before flooring.
All R6 passed model parameters remain unchanged. No training/query vocabulary,
label statistics, validation set, early stopping, trend or calibration is added.
Only the synthetic FakeCatBoost/Pool CLI is available here.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round6_geo as base
from prepare_data import ROOT, sha256

COMPONENT = 'geography_fine_grid_catboost'
TARGET_MODE = 'raw'
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
ADDED_FEATURES = ['geo_grid_0p005', 'geo_grid_0p005_shift']
MISSING_GRID = '__MISSING__'


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


FEATURE_RULES = {
    'version': 1,
    'component': COMPONENT,
    'original_features_unchanged': 87,
    'existing_geo_features_unchanged': 8,
    'existing_95_values_dtypes_order_and_index_unchanged': True,
    'feature_count': 97,
    'categorical_feature_count': 25,
    'geo_encoder_rules': _copy_json(base.FEATURE_RULES),
    'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
    'fit_target': TARGET_FORMULA,
    'prediction': INVERSE_FORMULA,
    'grid_degrees': 0.005,
    'grid_shift_degrees': 0.0025,
    'coordinate_input': 'existing float32 geography coordinates cast to float64',
    'regular_grid_formula': 'floor(lat_float64 / 0.005):floor(lon_float64 / 0.005)',
    'shifted_grid_formula': 'floor((lat_float64 + 0.0025) / 0.005):floor((lon_float64 + 0.0025) / 0.005)',
    'grid_encoding': 'str(int(latitude_cell)) + colon + str(int(longitude_cell))',
    'missing_grid': MISSING_GRID,
    'unknown_query_grid_policy': 'keep computed string; no fit-vocabulary filtering',
    'original_coarse_grid_policy': 'preserve existing geo_grid_002; never rebuild from float32',
    'grid_rounding_epsilon_or_boundary_adjustment': False,
    'additional_categorical_dtype': 'object containing strings',
    'fit_grid_counts_or_vocabulary_filter': False,
    'target_dtype': 'float64',
    'prediction_dtype': 'float64',
    'target_domain': 'finite nonnegative TWD',
    'sample_weight': 'uniform',
    'weight_argument_passed_to_pool': False,
    'baseline_depth': 8,
    'depth': 8,
    'baseline_boosting_type': 'Plain',
    'boosting_type': 'Plain',
    'changed_parameter_names': [],
    'removed_feature_names': [],
    'added_feature_names': list(ADDED_FEATURES),
    'baseline_feature_count': 95,
    'original95_preserved': True,
    'required_actual_grow_policy': 'SymmetricTree',
    'maximum_allowed_tree_leaf_count': 256,
    'baseline_iterations': 2999,
    'baseline_learning_rate': 0.065,
    'iterations': 2999,
    'learning_rate': 0.065,
    'native_learning_rate_comparison_rtol': 1e-7,
    'native_learning_rate_comparison_atol': 0.0,
    'threads': 4,
    'seed': 42,
    'no_eval_set': True,
    'no_early_stopping': True,
    'query_labels_used': False,
    'query_used_for_geographic_state': False,
    'target_used_for_geographic_state': False,
    'reference_scoring_grouping_or_dedup_keys_changed': False,
    'reference_or_trend_applied_here': False,
    'calibration_applied': False,
    'extra_price_cap_applied': False,
    'nonfinite_or_inverse_overflow': 'raise ValueError',
}


def source_hashes():
    return {name: sha256(ROOT / name) for name in (
        'src/round21_fine_grid_geo.py', 'src/round6_geo.py', 'src/prepare_data.py')}


def _base_configuration_sha256():
    return sha256(ROOT / 'runs/v1/models/final_config.json')


def _baseline_parameters():
    result = _copy_json(base.parameters())
    base.need(result['depth'] == 8 and result['iterations'] == 2999
              and result['learning_rate'] == 0.065 and result['thread_count'] == 4 and result['random_seed'] == 42
              and result['loss_function'] == 'MAE' and result['boosting_type'] == 'Plain',
              'Expected the fixed R6 depth-8 Plain recipe')
    return result


def parameters():
    """Return an independent deep copy of the exact original R6 recipe."""
    baseline = _baseline_parameters()
    result = _copy_json(baseline)
    base.need(result == baseline, 'R21 must not change any passed model parameter')
    return result


def _check_X(X):
    base.need(isinstance(X, pd.DataFrame), 'X must be a DataFrame')
    base.need(X.columns.is_unique and X.index.is_unique, 'X columns and indices must be unique')
    base.need(not {'編號', '總價元', 'y', 'target', 'ID', 'id'} & set(X.columns),
              'Identifiers and targets cannot enter model features')


def _float_vector(values, name):
    raw = values.to_numpy(copy=True) if isinstance(values, pd.Series) else np.asarray(values)
    base.need(raw.ndim == 1 and raw.dtype.kind in 'iuf', name + ' must be real numeric and one dimensional')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    base.need(np.isfinite(result).all(), name + ' must be finite')
    return result


def values_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def transform_target(y):
    result = _float_vector(y, 'Fit target')
    base.need((result >= 0).all(), 'Fit prices must be nonnegative')
    return result / 1e6


def inverse_transform(z):
    result = _float_vector(z, 'Model output')
    try:
        with np.errstate(over='raise', invalid='raise'):
            prices = result * 1e6
    except FloatingPointError as exc:
        raise ValueError('Raw TWD scaling overflowed') from exc
    base.need(np.isfinite(prices).all(), 'Raw TWD predictions must be finite')
    return np.maximum(prices, 0).astype(np.float64, copy=False)


def augment_features(X, geo):
    """Append only two fixed categorical grids to the unchanged R6 model view."""
    _check_X(X)
    baseline = base.augment_features(X, geo)
    base.need(len(baseline.columns) == 95 and baseline.columns.is_unique
              and baseline.index.equals(X.index), 'Expected original R6 95-feature schema')
    base.need(not set(ADDED_FEATURES) & set(baseline.columns), 'Fine-grid feature name collision')
    added = pd.DataFrame(index=geo.index)
    usable = geo['geo_has_position'].eq(1)
    lat = geo.loc[usable, 'geo_latitude'].to_numpy(dtype=np.float64, copy=True)
    lon = geo.loc[usable, 'geo_longitude'].to_numpy(dtype=np.float64, copy=True)
    for name, shift in zip(ADDED_FEATURES, (0.0, 0.0025)):
        values = pd.Series(MISSING_GRID, index=geo.index, dtype=object)
        lat_cells = np.floor(lat / 0.005) if shift == 0.0 else np.floor((lat + shift) / 0.005)
        lon_cells = np.floor(lon / 0.005) if shift == 0.0 else np.floor((lon + shift) / 0.005)
        values.loc[usable] = [str(int(a)) + ':' + str(int(b)) for a,b in zip(lat_cells,lon_cells)]
        added[name] = values
    result = pd.concat([baseline, added], axis=1)
    preserved = result.iloc[:, :95]
    base.need(list(result.columns) == list(baseline.columns) + ADDED_FEATURES
              and len(result.columns) == 97 and result.columns.is_unique
              and preserved.equals(baseline) and preserved.dtypes.equals(baseline.dtypes)
              and preserved.index.equals(baseline.index), 'Original 95 features or added grid schema changed')
    return result


def categories(X, supplied):
    baseline = base.categories(X, supplied)
    result = baseline + list(ADDED_FEATURES)
    base.need(len(baseline) == 23 and len(result) == len(set(result)) == 25,
              'Expected 23 original and two additional unique categorical columns')
    return result


def fit_feature_state(X, geo, cats):
    """Accept fit X and geography only; neither target nor query is an input."""
    _check_X(X)
    base.need(len(X) > 0, 'Cannot fit an empty feature frame')
    geo_state = _copy_json(base.fit_feature_state(X, geo))
    augmented = augment_features(X, geo)
    state = {
        'format_version': 1,
        'component': COMPONENT,
        'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA,
        'inverse_formula': INVERSE_FORMULA,
        'feature_rules': _copy_json(FEATURE_RULES),
        'source_sha256': source_hashes(),
        'base_configuration_sha256': _base_configuration_sha256(),
        'baseline_parameters': _baseline_parameters(),
        'parameters': parameters(),
        'changed_parameter_names': [],
        'removed_feature_names': [],
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
        'geo_state': geo_state,
        'original_categorical_columns': list(cats),
        'categorical_columns': categories(X, cats),
        'all_feature_names': list(augmented.columns),
        'fit_rows': int(len(X)),
        'fit_model_feature_values_sha256': base.frame_sha(augmented),
        'fit_model_feature_dtypes': {c: str(augmented[c].dtype) for c in augmented},
        'fit_state_uses_y': False,
        'fit_state_uses_query': False,
        'query_labels_used': False,
    }
    state['state_content_sha256'] = base.content_sha(state)
    return state


def check_state(state, X, cats):
    _check_X(X)
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['target_mode'] == TARGET_MODE
              and state['feature_rules'] == FEATURE_RULES, 'Fine-grid raw model rules changed')
    base.need(state['target_formula'] == TARGET_FORMULA and state['inverse_formula'] == INVERSE_FORMULA,
              'Raw target or inverse changed')
    base.need(state['source_sha256'] == source_hashes(), 'Helper or dependency source changed')
    base.need(state['base_configuration_sha256'] == _base_configuration_sha256()
              and state['baseline_parameters'] == _baseline_parameters()
              and state['parameters'] == parameters()
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True, 'Original R6 configuration or added grid schema changed')
    base.check_state(state['geo_state'], X)
    base.need(list(cats) == state['original_categorical_columns']
              and categories(X, cats) == state['categorical_columns'], 'Category schema changed')
    base.need(state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['geo_state']['all_feature_names']) == 95
              and len(state['all_feature_names']) == 97, 'Original R6 or added model columns changed')
    base.need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
              and state['query_labels_used'] is False, 'State data-use declaration changed')


def check_model(model, state):
    """Check native trees, learning rate, policies and categorical positions."""
    base.need(state['state_content_sha256'] == base.content_sha({
        k: v for k, v in state.items() if k != 'state_content_sha256'}), 'State content changed')
    base.need(state['component'] == COMPONENT and state['parameters']['depth'] == 8
              and state['parameters']['boosting_type'] == 'Plain'
              and state['parameters']['iterations'] == 2999 and state['parameters']['learning_rate'] == 0.065
              and state['changed_parameter_names'] == []
              and state['removed_feature_names'] == []
              and state['added_feature_names'] == ADDED_FEATURES
              and state['baseline_feature_count'] == 95 and state['original95_preserved'] is True
              and state['parameters'] == state['baseline_parameters']
              and state['all_feature_names'] == state['geo_state']['all_feature_names'] + ADDED_FEATURES
              and len(state['all_feature_names']) == 97, 'Wrong fine-grid raw model state')
    base.need(int(model.tree_count_) == 2999, 'Expected exactly 2999 native trees')
    base.need(list(model.feature_names_) == state['all_feature_names'], 'Native feature order differs')
    category_indices = np.asarray(model.get_cat_feature_indices())
    expected_categories = sorted(state['all_feature_names'].index(c) for c in state['categorical_columns'])
    base.need(category_indices.shape == (25,) and category_indices.dtype.kind in 'iu'
              and len(expected_categories) == len(set(expected_categories)) == 25
              and np.array_equal(category_indices, expected_categories), 'Native categorical positions differ')
    actual = model.get_all_params()
    native_learning_rate = float(actual['learning_rate'])
    base.need(actual['depth'] == 8 and actual['grow_policy'] == 'SymmetricTree'
              and actual['boosting_type'] == 'Plain', 'Native depth/grow/boosting policy differs')
    base.need(np.isclose(native_learning_rate, 0.065, rtol=1e-7, atol=0),
              'Native learning rate differs from the fixed 0.065 within storage tolerance')
    leaves = np.asarray(model.get_tree_leaf_counts())
    base.need(leaves.shape == (2999,) and leaves.dtype.kind in 'iu'
              and (leaves > 0).all() and (leaves <= 256).all(),
              'Expected 2999 positive integer leaf counts, each at most 256')
    return {
        'actual_model_depth': int(actual['depth']),
        'actual_boosting_type': actual['boosting_type'],
        'actual_grow_policy': actual['grow_policy'],
        'actual_trees': int(model.tree_count_),
        'maximum_tree_leaf_count': int(leaves.max()),
        'tree_leaf_counts_sha256': hashlib.sha256(leaves.astype('<i8').tobytes(order='C')).hexdigest(),
        'actual_learning_rate': native_learning_rate,
    }


def fit_model(X, y, geo, cats):
    """Train once on fit labels only, with uniform raw-price MAE."""
    started = time.monotonic()
    _check_X(X)
    if isinstance(y, pd.Series):
        base.need(y.index.equals(X.index), 'Fit target Series index differs from X')
    raw = _float_vector(y, 'Fit target')
    base.need(raw.shape == (len(X),) and len(X) > 0, 'Fit target length differs from X')
    transformed = transform_target(raw)
    raw_sha, transformed_sha = values_sha256(raw), values_sha256(transformed)
    state = fit_feature_state(X, geo, cats)
    augmented = augment_features(X, geo)
    config = parameters()
    model = CatBoostRegressor(**config)
    model.fit(Pool(augmented, transformed, cat_features=state['categorical_columns'], thread_count=4))
    check_state(state, X, cats)
    actual = check_model(model, state)
    base.need(parameters() == config, 'Parameters changed during fit')
    base.need(base.frame_sha(X) == state['geo_state']['fit_original_feature_values_sha256']
              and base.frame_sha(geo) == state['geo_state']['fit_geo_feature_values_sha256']
              and base.frame_sha(augmented) == state['fit_model_feature_values_sha256'],
              'Fit features changed')
    base.need(values_sha256(raw) == raw_sha and values_sha256(transformed) == transformed_sha,
              'Fit target changed')
    metadata = {
        'family': 'catboost', 'component': COMPONENT, 'target_mode': TARGET_MODE,
        'target_formula': TARGET_FORMULA, 'inverse_formula': INVERSE_FORMULA,
        'parameters': _copy_json(config), 'parameters_sha256': base.content_sha(config),
        'baseline_parameters': _copy_json(state['baseline_parameters']),
        'changed_parameter_names': [],
        'removed_feature_names': [],
        'added_feature_names': list(ADDED_FEATURES),
        'baseline_feature_count': 95,
        'original95_preserved': True,
        'iterations': 2999, 'actual_iterations': actual['actual_trees'],
        'features': 97, 'feature_count': 97, 'categorical_feature_count': 25,
        'native_categorical_feature_count': 25,
        'native_learning_rate_comparison_rtol': 1e-7,
        'native_learning_rate_comparison_atol': 0.0,
        'feature_columns': list(augmented.columns),
        'categorical_features': list(state['categorical_columns']),
        'categorical_columns': list(state['categorical_columns']),
        'sample_weight': 'uniform', 'weight_argument_passed_to_pool': False,
        'feature_rules': _copy_json(FEATURE_RULES), 'geo_feature_rules': _copy_json(base.FEATURE_RULES),
        'geo_encoder_rule_training_fields_are_baseline_provenance_only': True,
        'feature_state_content_sha256': state['state_content_sha256'],
        'script_sha256': state['source_sha256']['src/round21_fine_grid_geo.py'],
        'source_sha256': _copy_json(state['source_sha256']),
        'base_configuration_sha256': state['base_configuration_sha256'],
        'fit_rows': int(len(X)),
        'fit_original_feature_values_sha256': state['geo_state']['fit_original_feature_values_sha256'],
        'fit_geo_feature_values_sha256': state['geo_state']['fit_geo_feature_values_sha256'],
        'fit_model_feature_values_sha256': state['fit_model_feature_values_sha256'],
        'fit_target_values_sha256': raw_sha, 'fit_original_target_values_sha256': raw_sha,
        'fit_transformed_target_values_sha256': transformed_sha,
        'target_value_hash_encoding': 'ordered C-contiguous little-endian float64',
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'query_labels_used': False, 'fit_uses_eval_set': False, 'early_stopping_used': False,
        'calibration_applied': False, 'extra_price_cap_applied': False,
        'reference_or_trend_applied_here': False, 'reference_features_changed': False,
        'external_geography_used': True, 'external_property_answers_used': False,
        'elapsed_seconds': time.monotonic() - started,
        **actual,
    }
    return model, state, metadata


def predict_model(model, state, X, geo, cats):
    """Return unrounded nonnegative float64 TWD; no query labels or trend."""
    check_state(state, X, cats)
    augmented = augment_features(X, geo)
    base.need(list(augmented.columns) == state['all_feature_names'], 'Query feature order differs')
    check_model(model, state)
    raw = model.predict(Pool(augmented, cat_features=state['categorical_columns'], thread_count=4),
                        thread_count=4)
    result = inverse_transform(raw)
    base.need(result.shape == (len(X),), 'Prediction rows differ')
    return result


def self_check():
    """Full bounded checks with synthetic frames/FakePool; no real config or fit."""
    checks=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    idx=pd.Index([9,41,7,23],name='synthetic_row')
    cats=['category_'+str(i) for i in range(21)]
    X=pd.DataFrame({c:pd.Series(['category']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    for i in range(66):X['numeric_'+str(i)]=np.arange(len(idx),dtype='float32')
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype='float32') for c in base.NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[base.FEATURES]
    X_before,geo_before=X.copy(deep=True),geo.copy(deep=True)
    aug=augment_features(X,geo);expected=base.augment_features(X,geo)
    ok('original_R6_95_values_dtypes_order_index_exact',aug.iloc[:,:95].equals(expected)
       and aug.iloc[:,:95].dtypes.equals(expected.dtypes) and aug.index.equals(idx))
    ok('97_features_25_categories_only_two_fixed_grids',len(aug.columns)==97
       and categories(X,cats)==base.categories(X,cats)+ADDED_FEATURES and len(categories(X,cats))==25
       and aug.columns[-2:].tolist()==ADDED_FEATURES and aug[ADDED_FEATURES].eq(MISSING_GRID).all().all())
    import math
    boundary_lat=np.asarray([np.nextafter(np.float32(25.),np.float32(-np.inf)), np.float32(25.),
        np.nextafter(np.float32(25.),np.float32(np.inf)),np.float32(25.0025),
        np.nextafter(np.float32(25.0025),np.float32(-np.inf)),np.float32(25.005),np.float32(25.003)],dtype='float32')
    boundary_lon=np.asarray([np.nextafter(np.float32(121.),np.float32(-np.inf)), np.float32(121.),
        np.nextafter(np.float32(121.),np.float32(np.inf)),np.float32(121.0025),
        np.nextafter(np.float32(121.0025),np.float32(np.inf)),np.float32(121.005),np.float32(121.003)],dtype='float32')
    bx=X.iloc[np.zeros(len(boundary_lat),dtype=int)].copy();bx.index=pd.Index(range(100,107))
    bg=geo.iloc[np.zeros(len(boundary_lat),dtype=int)].copy();bg.index=bx.index
    for col in base.NUMERIC:bg[col]=np.zeros(len(bg),dtype='float32')
    bg['geo_latitude']=boundary_lat;bg['geo_longitude']=boundary_lon
    bg['geo_has_position']=np.ones(len(bg),dtype='float32')
    bg['geo_match_status']=pd.Series(['road_level']*len(bg),index=bg.index,dtype=object)
    bg['geo_grid_002']=pd.Series(['raw64-coarse-preserved']*len(bg),index=bg.index,dtype=object)
    boundary=augment_features(bx,bg)
    expected_regular=[str(math.floor(float(a)/.005))+':'+str(math.floor(float(b)/.005)) for a,b in zip(boundary_lat,boundary_lon)]
    expected_shift=[str(math.floor((float(a)+.0025)/.005))+':'+str(math.floor((float(b)+.0025)/.005)) for a,b in zip(boundary_lat,boundary_lon)]
    ok('float32_cast64_exact_boundary_floor_without_epsilon',boundary[ADDED_FEATURES[0]].tolist()==expected_regular)
    ok('both_axes_positive_half_cell_shift_exact',boundary[ADDED_FEATURES[1]].tolist()==expected_shift
       and boundary.iloc[-1][ADDED_FEATURES[0]]=='5000:24200' and boundary.iloc[-1][ADDED_FEATURES[1]]=='5001:24201')
    ok('nextafter_boundaries_are_not_snapped_or_rounded',boundary.iloc[0][ADDED_FEATURES[0]]=='4999:24199'
       and boundary.iloc[1][ADDED_FEATURES[0]]=='5000:24200'
       and expected_regular[5] != str(math.floor(25.005/.005))+':'+str(math.floor(121.005/.005)))
    ok('coarse_raw64_grid_never_rebuilt_from_float32',boundary['geo_grid_002'].equals(bg['geo_grid_002']))
    ok('fine_grid_batch_and_single_row_values_dtypes_index_identical',boundary.equals(pd.concat([
        augment_features(bx.iloc[[i]],bg.iloc[[i]]) for i in range(len(bx))])))
    ok('fine_grid_output_categories_are_object_strings',all(boundary[c].dtype==object
       and boundary[c].map(lambda value:isinstance(value,str)).all() for c in ADDED_FEATURES))
    ok('original_numeric_geography_bytes_preserved',all(boundary[c].to_numpy().tobytes()==bg[c].to_numpy().tobytes() for c in base.NUMERIC))
    badgeo=bg.copy();badgeo['geo_latitude']=badgeo['geo_latitude'].astype('float64')
    rejects('geo_float64_instead_of_existing_float32_rejected',lambda:augment_features(bx,badgeo))
    rejects('geo_column_order_changed_rejected',lambda:augment_features(bx,bg[bg.columns[::-1]]))
    badgeo=bg.copy();badgeo.loc[badgeo.index[0],'geo_has_position']=np.float32(0)
    rejects('missing_position_cannot_retain_usable_geometry',lambda:augment_features(bx,badgeo))
    badgeo=bg.copy();badgeo.loc[badgeo.index[0],'geo_latitude']=np.float32(np.nan)
    rejects('usable_geometry_must_be_finite',lambda:augment_features(bx,badgeo))
    collide=X.rename(columns={'numeric_0':ADDED_FEATURES[0]})
    rejects('added_grid_name_collision_rejected',lambda:augment_features(collide,geo))
    duplicate=X.copy();duplicate.index=[1,1,2,3]
    rejects('duplicate_row_index_rejected',lambda:augment_features(duplicate,geo))
    ok('state_API_fit_X_geo_cats_only',list(inspect.signature(fit_feature_state).parameters)==['X','geo','cats'])
    ok('raw_transform_and_inverse_once',np.array_equal(transform_target([0,1e6,2.5e6]),[0,1,2.5])
       and np.array_equal(inverse_transform([-1,0,2.5]),[0,0,2.5e6]))
    for name,value in [('negative',[-1.]),('nan',[np.nan]),('inf',[np.inf]),('string',['1']),('two_dim',[[1.]])]:
        rejects('invalid_target_'+name,lambda value=value:transform_target(value))
    for name,value in [('nan',[np.nan]),('inf',[np.inf]),('overflow',[np.finfo('float64').max]),('two_dim',[[1.]])]:
        rejects('invalid_inverse_'+name,lambda value=value:inverse_transform(value))
    fake_base={'iterations':2999,'depth':8,'thread_count':4,'random_seed':42,'loss_function':'MAE',
               'boosting_type':'Plain','task_type':'GPU','devices':'0','learning_rate':0.065,
               'l2_leaf_reg':6,'bootstrap_type':'Bernoulli','subsample':.85,'random_strength':.5,
               'one_hot_max_size':8,'max_ctr_complexity':2,'border_count':128,'gpu_ram_part':.75,
               'allow_writing_files':False,'verbose':300}
    saved_base=_copy_json(fake_base)
    class FakePool:
        def __init__(self,data,label=None,**kwargs):self.data,self.label,self.kwargs=data,label,kwargs
    class FakeModel:
        def __init__(self,**kwargs):
            self.config=_copy_json(kwargs);self.actual={**kwargs,'grow_policy':'SymmetricTree'}
            self.tree_count_=2999;self.leaves=np.full(2999,256,dtype=np.uint32)
            self.cat_indices=[]
        def fit(self,pool,**kwargs):
            if kwargs:raise AssertionError('No eval_set/earlystop/weights')
            self.feature_names_=list(pool.data.columns);self.fit_pool=pool
            self.cat_indices=sorted(self.feature_names_.index(c) for c in pool.kwargs['cat_features'])
            return self
        def predict(self,pool,**kwargs):
            if pool.label is not None:raise AssertionError('Query labels forbidden')
            self.last_query_pool=pool
            return pool.data['numeric_0'].to_numpy(dtype='float64')-.5
        def get_all_params(self):return dict(self.actual)
        def get_tree_leaf_counts(self):return self.leaves.copy()
        def get_cat_feature_indices(self):return list(self.cat_indices)
    original_model,original_pool=globals()['CatBoostRegressor'],globals()['Pool']
    original_params=base.parameters;original_config_sha=globals()['_base_configuration_sha256']
    globals()['CatBoostRegressor'],globals()['Pool']=FakeModel,FakePool
    base.parameters=lambda:fake_base
    globals()['_base_configuration_sha256']=lambda:'a'*64
    try:
        params=parameters()
        ok('passed_parameters_exactly_equal_original_R6',params==fake_base and params['iterations']==2999
           and params['learning_rate']==.065 and params['boosting_type']=='Plain')
        params['depth']=99
        ok('baseline_parameters_are_not_mutated',fake_base==saved_base and parameters()['depth']==8)
        for key,value in [('boosting_type','Ordered'),('iterations',3000),('learning_rate',.05)]:
            original=fake_base[key];fake_base[key]=value
            rejects('wrong_baseline_'+key,parameters);fake_base[key]=original
        y=pd.Series([0.0,1e6,2e6,3e6],index=idx,dtype='float64')
        model,state,meta=fit_model(X,y,geo,cats)
        ok('constructor_exact_original_R6_recipe_and_uniform_raw_Pool',model.config==parameters()
           and np.array_equal(model.fit_pool.label,y.to_numpy()/1e6)
           and set(model.fit_pool.kwargs)=={'cat_features','thread_count'}
           and model.fit_pool.kwargs['cat_features']==categories(X,cats))
        ok('fit_only_state_no_target_or_query_and_no_parameter_delta',state['baseline_parameters']==saved_base
           and state['parameters']==parameters() and state['changed_parameter_names']==[]
           and state['removed_feature_names']==[] and state['added_feature_names']==ADDED_FEATURES
           and state['baseline_feature_count']==95 and state['original95_preserved'] is True
           and state['all_feature_names']==state['geo_state']['all_feature_names']+ADDED_FEATURES
           and not state['fit_state_uses_y']
           and not state['fit_state_uses_query'] and not state['query_labels_used']
           and fit_feature_state(X,geo,cats)==state)
        actual=check_model(model,state)
        leaf_sha=hashlib.sha256(np.full(2999,256,dtype='<i8').tobytes()).hexdigest()
        ok('native_seven_fields_Plain_depth8_tree2999_leaf256_lr065',actual=={
            'actual_model_depth':8,'actual_boosting_type':'Plain','actual_grow_policy':'SymmetricTree',
            'actual_trees':2999,'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':leaf_sha,
            'actual_learning_rate':.065})
        ok('metadata_raw97_25_nativeLR_and_comparison_tolerance',all(meta[k]==v for k,v in actual.items())
           and meta['baseline_parameters']==saved_base and meta['parameters']==parameters()
           and meta['changed_parameter_names']==[] and meta['removed_feature_names']==[]
           and meta['feature_count']==97 and meta['categorical_feature_count']==25
           and meta['baseline_feature_count']==95 and meta['original95_preserved'] is True
           and meta['added_feature_names']==ADDED_FEATURES
           and meta['native_categorical_feature_count']==25 and meta['target_mode']=='raw'
           and meta['sample_weight']=='uniform' and not meta['weight_argument_passed_to_pool']
           and meta['native_learning_rate_comparison_rtol']==1e-7
           and meta['native_learning_rate_comparison_atol']==0.0
           and not meta['calibration_applied'] and not meta['extra_price_cap_applied']
           and not meta['reference_or_trend_applied_here'] and not meta['fit_uses_eval_set']
           and not meta['early_stopping_used'])
        restored=_copy_json(state)
        loaded=FakeModel(**_copy_json(model.config));loaded.feature_names_=list(model.feature_names_)
        loaded.cat_indices=list(model.cat_indices)
        pred=predict_model(loaded,restored,X,geo,cats)
        ok('JSON_state_and_simulated_model_reload_TWD_exact',np.array_equal(pred,predict_model(model,state,X,geo,cats))
           and pred.dtype==np.dtype('float64') and np.array_equal(pred,np.maximum(np.arange(len(X))-.5,0)*1e6))
        ok('batch_vs_single_row_prediction_exact',np.array_equal(pred,np.concatenate([
            predict_model(loaded,restored,X.iloc[[i]],geo.iloc[[i]],cats) for i in range(len(X))])))
        state_before=json.dumps(restored,ensure_ascii=False,sort_keys=True)
        unknown_X=bx.iloc[:2].copy();unknown_geo=bg.iloc[:2].copy()
        unknown_geo['geo_latitude']=np.asarray([26.31,26.315],dtype='float32')
        unknown_geo['geo_longitude']=np.asarray([120.61,120.615],dtype='float32')
        unknown=augment_features(unknown_X,unknown_geo)
        predict_model(loaded,restored,unknown_X,unknown_geo,cats)
        ok('unseen_query_cells_remain_strings_not_fit_vocab_missing',not unknown[ADDED_FEATURES].eq(MISSING_GRID).any().any()
           and loaded.last_query_pool.data[ADDED_FEATURES].equals(unknown[ADDED_FEATURES]))
        ok('query_never_mutates_or_enlarges_feature_state',json.dumps(restored,ensure_ascii=False,sort_keys=True)==state_before
           and not any('vocabulary' in key or 'grid_counts' in key for key in state))
        rejects('query_original_column_reorder_rejected',lambda:predict_model(model,state,X[X.columns[::-1]],geo,cats))
        rejects('empty_fit_rejected',lambda:fit_feature_state(X.iloc[:0],geo.iloc[:0],cats))
        ok('raw_target_hashes_not_log_or_unit',meta['fit_original_target_values_sha256']==values_sha256(y)
           and meta['fit_transformed_target_values_sha256']==values_sha256(y.to_numpy()/1e6))
        loaded.actual['learning_rate']=float(np.float32(.065))
        native_float32=check_model(loaded,restored)['actual_learning_rate']
        ok('native_float32_learning_rate_storage_accepted_and_recorded',native_float32==float(np.float32(.065)))
        for name,value in [('wrong_lr',.0325),('nan_lr',float('nan')),('zero_lr',0.0)]:
            loaded.actual['learning_rate']=value
            rejects('native_'+name+'_rejected',lambda:check_model(loaded,restored))
        loaded.actual['learning_rate']=.0325
        rejects('predict_checks_native_learning_rate',lambda:predict_model(loaded,restored,X,geo,cats))
        loaded.actual['learning_rate']=.065;loaded.actual['boosting_type']='Ordered'
        rejects('native_Ordered_rejected',lambda:check_model(loaded,restored))
        loaded.actual['boosting_type']='Plain';loaded.actual['depth']=6
        rejects('native_depth6_rejected',lambda:check_model(loaded,restored))
        loaded.actual['depth']=8;loaded.actual['grow_policy']='Lossguide'
        rejects('native_nonSymmetricTree_rejected',lambda:check_model(loaded,restored))
        loaded.actual['grow_policy']='SymmetricTree';loaded.leaves[0]=257
        rejects('leaf_over256_rejected',lambda:check_model(loaded,restored))
        loaded.leaves[0]=0
        rejects('zero_leaf_rejected',lambda:check_model(loaded,restored))
        loaded.leaves=np.full(2999,256.,dtype='float64')
        rejects('noninteger_leaf_counts_rejected',lambda:check_model(loaded,restored))
        loaded.leaves=np.full(2999,256,dtype=np.uint32);loaded.tree_count_=5998
        rejects('wrong_tree_count_rejected',lambda:check_model(loaded,restored))
        loaded.tree_count_=2999;loaded.leaves=loaded.leaves[:-1]
        rejects('wrong_leaf_vector_length_rejected',lambda:check_model(loaded,restored))
        loaded.leaves=np.full(2999,256,dtype=np.uint32);loaded.cat_indices=loaded.cat_indices[:-1]
        rejects('24_native_categories_rejected',lambda:check_model(loaded,restored))
        loaded.cat_indices=list(model.cat_indices);loaded.cat_indices[0]=30
        rejects('wrong_native_category_position_rejected',lambda:check_model(loaded,restored))
        loaded.cat_indices=list(model.cat_indices);loaded.feature_names_=list(reversed(loaded.feature_names_))
        rejects('wrong_native_feature_order_rejected',lambda:check_model(loaded,restored))
        loaded.feature_names_=list(model.feature_names_)
        bad=_copy_json(state);bad['parameters']['learning_rate']=.0325
        rejects('tampered_state_hash_rejected',lambda:check_state(bad,X,cats))
        bad=_copy_json(state);bad['source_sha256']['src/round21_fine_grid_geo.py']='b'*64
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('changed_helper_source_rejected',lambda:check_state(bad,X,cats))
        bad=_copy_json(state);bad['removed_feature_names']=['project_name']
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('feature_removal_rejected',lambda:check_state(bad,X,cats))
        bad_X=X.copy();bad_X['numeric_0']=bad_X['numeric_0'].astype('float64')
        rejects('query_dtype_drift_rejected',lambda:predict_model(model,state,bad_X,geo,cats))
        rejects('query_geo_index_misalignment_rejected',lambda:predict_model(model,state,X,geo.iloc[::-1],cats))
        rejects('target_Series_index_misalignment_rejected',lambda:fit_model(X,y.iloc[::-1],geo,cats))
        rejects('fit_target_length_rejected',lambda:fit_model(X,[1.],geo,cats))
        rejects('fit_target_negative_rejected',lambda:fit_model(X,[-1.,1.,2.,3.],geo,cats))
        rejects('fit_target_nonfinite_rejected',lambda:fit_model(X,[0.,np.nan,2.,3.],geo,cats))
        bad_X=X.copy();bad_X['y']=0.
        rejects('target_column_not_feature_rejected',lambda:augment_features(bad_X,geo))
        bad=_copy_json(state);bad['added_feature_names']=list(reversed(ADDED_FEATURES))
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('added_grid_order_tampering_rejected',lambda:check_state(bad,X,cats))
        bad=_copy_json(state);bad['all_feature_names']=bad['all_feature_names'][:95]
        bad['state_content_sha256']=base.content_sha({k:v for k,v in bad.items() if k!='state_content_sha256'})
        rejects('missing_added_model_features_rejected',lambda:check_state(bad,X,cats))
        ok('original_X_geo_and_baseline_unchanged',X.equals(X_before) and X.dtypes.equals(X_before.dtypes)
           and geo.equals(geo_before) and geo.dtypes.equals(geo_before.dtypes) and fake_base==saved_base)
        ok('no_unit_area_weight_metadata',not any(k in meta for k in
           ('fit_effective_area_sha256','fit_sample_weights_sha256','fit_factor_sha256','fit_weight_diagnostics')))
    finally:
        globals()['CatBoostRegressor'],globals()['Pool']=original_model,original_pool
        base.parameters=original_params;globals()['_base_configuration_sha256']=original_config_sha
    return {'status':'passed','checks':checks,'check_count':len(checks),'synthetic_only':True,
            'real_data_loaded':False,'real_model_training_performed':False,
            'real_model_or_configuration_loaded':False,'files_written':False,'source_sha256':source_hashes()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_check(),ensure_ascii=False),flush=True)
