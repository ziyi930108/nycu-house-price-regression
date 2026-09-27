"""R13 independent price replay. No fit, tuning, or additional window scoring.

Independent errors, grouping and fixed gates; shared canonical loader and frozen
legacy reference are explicitly reused. Only saved stage results are audited.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round13'
CONTROL = 'v3_control'
CANDIDATE = 'city_alias_geography_blend'
ALIAS_COLUMN = 'r13_address_city_canonical'
EXPECTED_HELPER_SHA256 = 'aed4d22c3bb911d163d42a3bef4eb47038dfd7ac8f34aec5c460da7ac57f6c42'
GROUPS = ('seen', 'new_nonmissing', 'missing')
QUARTERS = {'Q1': [0,91], 'Q2': [92,183], 'Q3': [184,275], 'Q4': [276,None]}
THRESHOLDS = {
    'development_min_overall_improvement': .01, 'development_min_weighted_improvement': .01,
    'development_max_quarter_deterioration': .05, 'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': .05, 'confirmation_max_overall_deterioration': .0,
    'confirmation_max_weighted_deterioration': .0, 'confirmation_min_improved_quarters': 2,
    'auxiliary_max_overall_deterioration': .03, 'auxiliary_max_weighted_deterioration': .03,
    'auxiliary_quarters_diagnostic_only': True,
}
SPLITS = {'development': ('2022-08-11','2023-08-10'),
          'confirmation': ('2023-08-11','2024-08-10'),
          'auxiliary': ('2021-08-11','2022-08-10')}
COUNTS = {'development': (240316,81234), 'confirmation': (321550,133697), 'auxiliary': (134805,105511)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def norm(values):
    values = values.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return values.mask(values.isin(['', 'nan', 'None', '<NA>', 'NaT']), '__MISSING__')

def labels(fit, query):
    def keys(frame):
        project = norm(frame['project_name'])
        return norm(frame['address_city']) + '|' + norm(frame['address_district']) + '|' + project, project.eq('__MISSING__')
    fit_key, fit_missing = keys(fit)
    query_key, query_missing = keys(query)
    known = set(fit_key[~fit_missing])
    return np.where(query_missing, 'missing', np.where(query_key.isin(known), 'seen', 'new_nonmissing'))

def scoring_weights(data):
    columns = ['address_city', 'address_district', 'project_name']
    fit = pd.concat([data['X_train'][columns], data['X_valid'][columns]], ignore_index=True)
    group = labels(fit, data['X_test'])
    return {key: float(np.mean(group == key)) for key in GROUPS}

def independent_score(y, prediction, fit, query, start, weights):
    y, prediction = np.asarray(y, dtype='float64'), np.asarray(prediction, dtype='float64')
    require(y.shape == prediction.shape == (len(query),) and len(y) > 0, 'Unaligned score arrays')
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), 'Nonfinite score arrays')
    require(set(weights) == set(GROUPS) and all(np.isfinite(v) and v >= 0 for v in weights.values()), 'Invalid scoring weights')
    require(abs(sum(weights.values()) - 1.) < 1e-12, 'Scoring weights do not sum to one')
    errors = np.abs(y - prediction)
    group = labels(fit, query)
    horizon = query['trade_day_index'].to_numpy(dtype='float64') - (pd.Timestamp(start) - pd.Timestamp('2000-01-01')).days
    require(np.isfinite(horizon).all() and (horizon >= 0).all(), 'Invalid forecast horizon')
    month = pd.to_datetime(query['trade_day_index'], unit='D', origin='2000-01-01').dt.strftime('%Y-%m').to_numpy()
    city = norm(query['address_city']).to_numpy()
    def part(mask):
        n = int(np.sum(mask))
        return {'rows': n, 'mae': float(errors[mask].mean()) if n else None}
    strata = {key: part(group == key) for key in GROUPS}
    weighted = 0.0
    for key, weight in weights.items():
        if weight > 0:
            require(strata[key]['rows'] > 0, 'Positive scoring weight has no stratum rows')
            weighted += weight * strata[key]['mae']
    quarters = {}
    for name, (low, high) in QUARTERS.items():
        mask = horizon >= low
        if high is not None:
            mask &= horizon <= high
        quarters[name] = {**part(mask), 'horizon_days': [low, high]}
    require(sum(v['rows'] for v in quarters.values()) == len(query), 'Quarter masks do not partition rows')
    return {'rows': len(errors), 'overall_mae': float(errors.mean()), 'weighted_mae': float(weighted),
            'strata': strata, 'forecast_quarters': quarters,
            'months': {str(key): part(month == key) for key in sorted(set(month))},
            'cities': {str(key): part(city == key) for key in sorted(set(city))}}

def independent_blend(old, geographic, unit, reference, query, apply_reference):
    # Preserve the frozen arithmetic order as well as its 37.5/37.5/25 meaning.
    base = .75 * (.5 * np.asarray(old, dtype='float64') + .5 * np.asarray(geographic, dtype='float64')) + .25 * np.asarray(unit, dtype='float64')
    result = np.asarray(apply_reference(reference, query, base, variant='trend'), dtype='float64')
    require(result.shape == (len(query),) and np.isfinite(result).all() and (result >= 0).all(), 'Invalid blended prices')
    return result

def compare_vector(saved, replay, name):
    require(saved.shape == replay.shape and np.isfinite(replay).all(), name + ' shape/finite mismatch')
    require(np.allclose(saved, replay, rtol=1e-12, atol=1e-5), name + ' full-query replay mismatch')
    return {'rows': len(saved), 'bitwise_equal': bool(np.array_equal(saved.view('uint64'), replay.view('uint64'))),
            'maximum_absolute_difference_twd': float(np.max(np.abs(saved - replay))),
            'rtol': 1e-12, 'atol_twd': 1e-5}

def validate_category_counts(original_count, augmented_count):
    require(original_count == 21, 'Original 87 features require 21 categorical columns')
    require(augmented_count == 24, 'Augmented 96 features require 24 categorical columns')

def independent_model_view(X, geo, original_categories):
    """Independently append the sole exact alias to the frozen R6 95-column view."""
    from round6_geo import augment_features as r6_augment, categories as r6_categories
    require(isinstance(X,pd.DataFrame) and X.columns.is_unique and X.index.is_unique,
            'Independent model view needs unique X columns and indices')
    require(ALIAS_COLUMN not in X and 'address_city' in X, 'Unexpected city alias input schema')
    original = r6_augment(X,geo)
    original_cats = r6_categories(X,original_categories)
    require(len(original.columns)==95 and len(original_cats)==23, 'Original R6 schema differs')
    result = original.copy(deep=True)
    alias = X['address_city'].astype(object).copy(deep=True)
    alias.loc[alias.eq('桃園縣').fillna(False)] = '桃園市'
    result[ALIAS_COLUMN] = alias
    require(result.iloc[:,:95].equals(original)
            and result.iloc[:,:95].dtypes.equals(original.dtypes)
            and result.index.equals(original.index), 'Independent alias changed the original 95')
    require(list(result.columns)==list(original.columns)+[ALIAS_COLUMN]
            and result[ALIAS_COLUMN].dtype==object, 'Wrong appended alias type/order')
    result_cats = original_cats + [ALIAS_COLUMN]
    validate_category_counts(len(original_categories),len(result_cats))
    return result,result_cats


def alias_self_checks():
    """Small synthetic-only model-view, grouping and raw-scale checks."""
    from round6_geo import NUMERIC, CATEGORICAL, FEATURES
    checks=[]
    values=['桃園縣','桃園市','桃園縣桃園市','新竹縣','新竹市','嘉義縣','嘉義市',
            '__MISSING__',None,np.nan,pd.NA,' 桃園縣','桃園縣路']
    idx=pd.Index([101,4,89,2,70,9,60,3,98,8,41,13,7],name='original_row')
    cats=['address_city']+['category_'+str(i) for i in range(20)]
    X=pd.DataFrame({c:pd.Series(['original']*len(idx),index=idx,dtype=object) for c in cats},index=idx)
    X['address_city']=pd.Series(values,index=idx,dtype=object)
    for j in range(66): X['numeric_'+str(j)]=np.arange(len(idx),dtype='float32')
    geo=pd.DataFrame({c:np.full(len(idx),np.nan,dtype='float32') for c in NUMERIC},index=idx)
    geo['geo_has_position']=np.zeros(len(idx),dtype='float32')
    geo['geo_match_status']=pd.Series(['ineligible_address']*len(idx),index=idx,dtype=object)
    geo['geo_grid_002']=pd.Series(['__MISSING__']*len(idx),index=idx,dtype=object)
    geo=geo[FEATURES]
    before_X,before_geo=X.copy(deep=True),geo.copy(deep=True)
    view,view_cats=independent_model_view(X,geo,cats)
    expected=pd.Series(['桃園市']+values[1:],index=idx,dtype=object,name=ALIAS_COLUMN)
    pd.testing.assert_series_equal(view[ALIAS_COLUMN],expected)
    checks.append('exact_alias_modern_child_non_Taoyuan_and_missing_cases')
    require(view.shape==(len(idx),96) and len(view_cats)==24,'Alias view 96/24 schema')
    checks.append('independent_alias_view_96features_24categories')
    require(view.iloc[:,:87].equals(X) and view.iloc[:,87:95].equals(geo),'Original values differ')
    require(view.iloc[:,:87].dtypes.equals(X.dtypes) and view.iloc[:,87:95].dtypes.equals(geo.dtypes),'Original dtypes differ')
    checks.append('independent_alias_preserves_original95_values_dtypes_order_index')
    pieces=[independent_model_view(X.iloc[a:b],geo.iloc[a:b],cats)[0] for a,b in [(0,4),(4,len(X))]]
    pd.testing.assert_frame_equal(pd.concat(pieces),view)
    checks.append('independent_alias_batches_equal_full_view')
    view.loc[idx[0],'address_city']='output_only'
    view.loc[idx[0],'geo_match_status']='output_only'
    require(X.equals(before_X) and geo.equals(before_geo),'Input mutation')
    checks.append('independent_alias_output_does_not_mutate_inputs')
    try: independent_model_view(X,geo.iloc[::-1],cats)
    except (RuntimeError,ValueError): checks.append('misaligned_geo_rejected')
    else: raise AssertionError('Misaligned geography not rejected')
    try: independent_model_view(X.assign(**{ALIAS_COLUMN:'x'}),geo,cats)
    except (RuntimeError,ValueError): checks.append('already_augmented_input_rejected')
    else: raise AssertionError('Already augmented input not rejected')
    fit=pd.DataFrame({'address_city':['桃園縣'],'address_district':['桃園市'],'project_name':['same']})
    query=pd.DataFrame({'address_city':['桃園市'],'address_district':['桃園市'],'project_name':['same']})
    require(labels(fit,query).tolist()==['new_nonmissing'],'Scoring must retain raw city keys')
    checks.append('scoring_city_keys_are_not_canonicalized')
    require(np.array_equal(inverse_replay([-1.,0.,2.]),[0.,0.,2e6]),'Raw million-TWD inverse differs')
    checks.append('independent_raw_inverse_without_log_or_double_scale')
    for name,array in [('nan',[np.nan]),('infinite',[np.inf]),('wrong_dimensions',[[1.]])]:
        try: inverse_replay(array)
        except RuntimeError: checks.append('independent_raw_inverse_rejects_'+name)
        else: raise AssertionError('Invalid raw vector accepted: '+name)
    try: inverse_replay([np.finfo('float64').max])
    except (RuntimeError,FloatingPointError): checks.append('independent_raw_inverse_overflow_rejected')
    else: raise AssertionError('Inverse scaling overflow accepted')
    return checks


def self_check():
    import copy
    checks = []
    def check(name, truth):
        require(truth, name); checks.append(name)
    def rejected(name, operation):
        try:
            operation()
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError(name)
    validate_category_counts(21, 24)
    checks.append('original_87_21cats_and_augmented_96_24cats_are_distinct')
    rejected('original_23cats_schema_rejected', lambda: validate_category_counts(23, 24))
    rejected('augmented_21cats_schema_rejected', lambda: validate_category_counts(21, 21))
    start = pd.Timestamp('2022-08-11')
    day = (start - pd.Timestamp('2000-01-01')).days
    fit = pd.DataFrame({'address_city': ['A'], 'address_district': ['D'], 'project_name': ['known']})
    query = pd.DataFrame({'address_city': ['A'] * 4, 'address_district': ['D'] * 4,
                          'project_name': ['known', 'new', None, 'known'], 'trade_day_index': np.array([0, 92, 184, 276]) + day})
    weights = {'seen': .2, 'new_nonmissing': .3, 'missing': .5}
    result = independent_score([11, 12, 13, 14], np.repeat(10., 4), fit, query, str(start.date()), weights)
    check('overall_mae_in_twd', result['overall_mae'] == 2.5)
    check('fixed_group_weighted_mae', result['weighted_mae'] == .2 * 2.5 + .3 * 2 + .5 * 3)
    check('four_nonoverlapping_forecast_quarters', [v['mae'] for v in result['forecast_quarters'].values()] == [1., 2., 3., 4.])
    check('seen_new_missing_semantics', labels(fit, query).tolist() == ['seen', 'new_nonmissing', 'missing', 'seen'])
    subset = query.iloc[:1].copy()
    absent = independent_score([11], [10], fit, subset, str(start.date()), {'seen': 1., 'new_nonmissing': 0., 'missing': 0.})
    check('empty_strata_and_quarters_are_null', absent['strata']['missing']['mae'] is None and absent['forecast_quarters']['Q2']['mae'] is None)
    rejected('empty_positive_weight_stratum_rejected', lambda: independent_score([11], [10], fit, subset, str(start.date()), weights))
    rejected('unmatched_prediction_rows_rejected', lambda: independent_score([11], [10, 12], fit, subset, str(start.date()), weights))
    base = {'overall_mae': 100., 'weighted_mae': 100., 'strata': {'seen': {'rows': 500, 'mae': 100.}, 'missing': {'rows': 0, 'mae': None}},
            'forecast_quarters': {q: {'rows': 1, 'mae': 100.} for q in QUARTERS}}
    metrics = {CONTROL: copy.deepcopy(base), CANDIDATE: copy.deepcopy(base)}
    metrics[CANDIDATE]['overall_mae'] = metrics[CANDIDATE]['weighted_mae'] = 99.
    check('exact_one_percent_passes', independent_development_decision(metrics)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['weighted_mae'] = 99.00001
    check('both_mae_gates_required', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['strata']['seen']['mae'] = 105.00001
    check('large_group_deterioration_rejected', not independent_development_decision(bad)['passed'])
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 105.00001
    check('quarter_deterioration_rejected', not independent_development_decision(bad)['passed'])
    calls = []
    def reference_spy(ref, q, p, variant):
        require(ref == 'fit_reference' and q is query and variant == 'trend', 'Wrong reference identity')
        calls.append(p.copy()); return p * 1.1
    got = independent_blend([1] * 4, [3] * 4, [5] * 4, 'fit_reference', query, reference_spy)
    check('single_trend_fixed_slot_original_query', len(calls) == 1 and np.array_equal(got, np.repeat(1.1 * (.375 + 1.125 + 1.25), 4)))
    return checks

def independent_decision(stage, metrics):
    require(stage in SPLITS and set(metrics) == {CONTROL, CANDIDATE}, 'Unexpected stage/candidate')
    base, value = metrics[CONTROL], metrics[CANDIDATE]
    improved = [q for q, item in base['forecast_quarters'].items()
                if item['rows'] and value['forecast_quarters'][q]['mae'] < item['mae']]
    large = all(value['strata'][g]['mae'] <= 1.05 * item['mae']
                for g,item in base['strata'].items() if item['rows'] >= 500)
    if stage == 'development':
        checks = {
            'overall_improves_at_least_1pct': value['overall_mae'] <= .99 * base['overall_mae'],
            'weighted_improves_at_least_1pct': value['weighted_mae'] <= .99 * base['weighted_mae'],
            'quarters_deteriorate_at_most_5pct': all(value['forecast_quarters'][q]['mae'] <= 1.05 * item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),
            'large_strata_deteriorate_at_most_5pct': large,
        }
    else:
        bound = 1. if stage == 'confirmation' else 1.03
        checks = {'overall_within_limit': value['overall_mae'] <= bound * base['overall_mae'],
                  'weighted_within_limit': value['weighted_mae'] <= bound * base['weighted_mae'],
                  'large_strata_deteriorate_at_most_5pct': large}
        if stage == 'confirmation': checks['at_least_two_quarters_improve'] = len(improved) >= 2
    passed = all(checks.values())
    return {'passed': passed, 'selected_candidate': CANDIDATE if passed else CONTROL,
            'checks': checks, 'improved_quarters': improved, 'candidate_reselection_performed': False}


def independent_development_decision(metrics):
    return independent_decision('development', metrics)


def extra_self_checks():
    checks = self_check()
    base = {'overall_mae': 100., 'weighted_mae': 100.,
            'strata': {'seen': {'rows':500,'mae':100.}, 'missing': {'rows':0,'mae':None}},
            'forecast_quarters': {q: {'rows': 1,'mae':100.} for q in QUARTERS}}
    metrics = {CONTROL: copy.deepcopy(base), CANDIDATE: copy.deepcopy(base)}
    require(not independent_decision('confirmation',metrics)['passed'], 'Confirmation requires 2 strict improvements')
    checks.append('confirmation_two_quarters_required')
    for q in ('Q1','Q2'): metrics[CANDIDATE]['forecast_quarters'][q]['mae'] = 99.
    require(independent_decision('confirmation',metrics)['passed'], 'Confirmation equal overall plus 2 quarters allowed')
    checks.append('confirmation_equal_means_and_two_quarters_pass')
    metrics[CANDIDATE]['weighted_mae'] = 100.00001
    require(not independent_decision('confirmation',metrics)['passed'], 'Confirmation no tolerance beyond zero')
    checks.append('confirmation_tiny_weighted_deterioration_fails')
    metrics[CANDIDATE]['overall_mae'] = metrics[CANDIDATE]['weighted_mae'] = 103.
    metrics[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 9999.
    require(independent_decision('auxiliary',metrics)['passed'], 'Auxiliary exact 3 percent / diagnostic quarters')
    checks.append('auxiliary_exact_three_percent_and_diagnostic_quarters')
    metrics[CANDIDATE]['overall_mae'] = 103.00001
    require(not independent_decision('auxiliary',metrics)['passed'], 'Auxiliary >3 percent must fail')
    checks.append('auxiliary_over_three_percent_fails')
    checks.extend(alias_self_checks())
    return checks


def inverse_replay(raw):
    raw = np.asarray(raw, dtype='float64')
    require(raw.ndim == 1 and np.isfinite(raw).all(), 'Invalid raw million-TWD predictions')
    with np.errstate(over='raise', invalid='raise'):
        result = raw * 1e6
    require(np.isfinite(result).all(), 'Nonfinite inverse prices')
    return np.maximum(result, 0.)


def verify(stage):
    """Read only the explicitly named saved stage; never fit or choose candidates."""
    require(stage in SPLITS, 'Unexpected stage')
    import round13_experiment as driver
    from catboost import CatBoostRegressor, Pool
    proof_path = OUT / 'audit' / (stage + '_replay.json')
    for p in (proof_path, proof_path.with_suffix('.sha256')):
        require(not p.exists(), 'Refusing to replace replay evidence')
    checks = extra_self_checks()
    driver.require_no_staging()
    plan = driver.authorize(stage)
    result = driver.read_result(stage)
    require(plan['thresholds'] == THRESHOLDS and plan['splits'] == {k:list(v) for k,v in SPLITS.items()}, 'Different fixed protocol')
    require(driver.CONTROL == CONTROL and driver.CANDIDATE == CANDIDATE, 'Different identities')
    require(plan['source_sha256']['src/round13_city_alias_geo.py']==EXPECTED_HELPER_SHA256, 'Unexpected city-alias helper identity')
    require(result['evaluated_candidates'] == [CONTROL,CANDIDATE], 'Unexpected scored candidate')
    require(result['query_labels_used_for_training'] is False and result['public_score_used'] is False
            and result['historically_untouched'] is False and result['private_score_known'] is False, 'Incorrect exposure declaration')
    require(result['source_sha256'] == plan['source_sha256'], 'Result source differs')
    captured = {}
    def capture(p, expected=None):
        p=Path(p).resolve()
        require(p.is_relative_to(ROOT) and p.is_file(), 'Missing/unsafe audit input')
        digest=sha(p); name=rel(p)
        require(expected is None or digest==expected, 'Input identity changed: '+name)
        require(name not in captured or captured[name]==digest, 'Input changed during capture: '+name)
        captured[name]=digest
        return digest
    def nested(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(key,str) and '/' in key and isinstance(item,str) and len(item)==64:
                    capture(ROOT/key,item)
                else: nested(item)
        elif isinstance(value,list):
            for item in value: nested(item)
    nested(plan); nested(result)
    for path in [Path(__file__), OUT/'candidate_plan.json',OUT/'evaluation'/stage/'results.json']:
        capture(path)
        if path.suffix=='.json': capture(path.with_suffix('.sha256'))
    freeze=driver.locked(OUT/'frozen_candidate.json')
    nested(freeze)
    for path in [OUT/'frozen_candidate.json',OUT/'frozen_candidate.sha256']:
        capture(path)
    if stage=='development':
        require(freeze['result_sha256']==sha(OUT/'evaluation/development/results.json')
                and freeze['passed']==result['decision']['passed']
                and freeze['later_stages_allowed']==result['decision']['passed'], 'Development freeze does not match result')
    else:
        require(freeze['passed'] is True and freeze['later_stages_allowed'] is True, 'No own development authorization')
    folder=OUT/'models'/stage
    meta=driver.locked(folder/'metadata.json')
    nested(meta)
    for path in folder.iterdir():
        require(path.is_file(), 'Unexpected nested model artifact')
        capture(path, meta.get('artifact_sha256',{}).get(path.name))
    required={'model.cbm','feature_state.json','predictions.npy','raw_z.npy','fit_indices.npy','query_indices.npy'}
    require(set(meta['artifact_sha256'])==required, 'Unexpected model artifact inventory')
    for split in ('train','valid','test'):
        for suffix in ('.pkl','_features.pkl','.json'): capture(ROOT/'data_cache'/(split+suffix))
    for name in ('valid_clean_mask.npy','valid_clean_mask.json'): capture(ROOT/'data_cache'/name)
    schema=read(ROOT/'reports/feature_schema.json')
    require(read(ROOT/'data_cache/valid_clean_mask.json')=={s:schema['splits'][s]['input_source_sha256'] for s in ('train','valid')}, 'Canonical mask would be rewritten')
    protected={rel(path):capture(path) for path in [ROOT/'outputs/submission_v2.csv',ROOT/'outputs/submission_v3.csv',ROOT/'runs/versions.json']}
    print(json.dumps({'event':'independent_replay_started','stage':stage,'training':False,'additional_windows':False}),flush=True)
    # The shared loader deserializes canonical train/valid targets; only this stage's query errors are calculated.
    data,X,y,Xq,yq,fi,qi=driver.partition(stage)
    require((len(X),len(Xq))==COUNTS[stage] and len(X.columns)==87, 'Canonical shape differs')
    require((result['fit_rows'],result['query_rows'])==COUNTS[stage], 'Saved result counts differ')
    validate_category_counts(len(data['categorical_columns']),24)
    require(not np.intersect1d(fi,qi).size, 'Fit and query overlap')
    start,end=SPLITS[stage]
    start_day=(pd.Timestamp(start)-pd.Timestamp('2000-01-01')).days
    end_day=(pd.Timestamp(end)-pd.Timestamp('2000-01-01')).days
    require(np.isfinite(X['trade_day_index']).all() and (X['trade_day_index']<start_day).all(), 'Fit time leakage')
    require(np.isfinite(Xq['trade_day_index']).all() and Xq['trade_day_index'].between(start_day,end_day).all(), 'Wrong query dates')
    for name,values in [('fit',fi),('query',qi)]:
        a=np.load(folder/(name+'_indices.npy'),allow_pickle=False)
        require(np.array_equal(a,values),'Model row indices differ')
        require(plan['split_contract'][stage][name+'_indices_content_sha256']==hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest(),'Plan row indices differ')
    weights=scoring_weights(data)
    require(weights==plan['target_group_weights']==result['target_group_weights'], 'Independent test-X group weights differ')
    geo,geoq,_=driver.load_geographic_partition(stage,fi,qi)
    state=read(folder/'feature_state.json')
    require(driver.component.fit_feature_state(X,geo,data['categorical_columns'])==state, 'Saved fit-only feature state differs')
    fit_view,fit_cats=independent_model_view(X,geo,data['categorical_columns'])
    require(driver.frame_sha(fit_view)==state['fit_model_feature_values_sha256']
            and list(fit_view.columns)==state['all_feature_names']
            and fit_cats==state['categorical_columns'], 'Independent fit model view differs')
    require(state['feature_rules']['alias_mapping']=={'桃園縣':'桃園市'}
            and state['feature_rules']['new_categorical_column']==ALIAS_COLUMN, 'Wrong fixed city projection')
    del fit_view
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq)]:
        require(meta[key]==driver.frame_sha(frame),'Saved frame differs: '+key)
    y64=np.asarray(y,dtype='float64')
    require(y64.shape==(len(X),) and np.isfinite(y64).all() and (y64>=0).all(), 'Invalid raw fit labels')
    raw_y_sha=hashlib.sha256(y64.astype('<f8').tobytes()).hexdigest()
    transformed_sha=hashlib.sha256((y64/1e6).astype('<f8').tobytes()).hexdigest()
    require(meta['fit_target_sha256']==meta['fit_target_values_sha256']==meta['fit_original_target_values_sha256']==raw_y_sha,'Wrong target hash')
    require(meta['fit_transformed_target_values_sha256']==transformed_sha,'Wrong million-TWD target hash')
    require(meta['target_mode']=='raw' and meta['component']=='geography_city_alias_raw_catboost'
            and meta['feature_count']==96 and meta['categorical_feature_count']==24
            and meta['calibration_applied'] is False and meta['extra_price_cap_applied'] is False
            and meta['reference_features_changed'] is False and meta['reference_or_trend_applied_here'] is False,
            'Wrong raw city-alias component semantics')
    require(meta['parameters']==plan['parameters'] and meta['sample_weight']=='uniform'
            and meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False
            and meta['query_labels_used'] is False and meta['prediction_units']=='TWD','Wrong training contract')
    require(meta['target_formula']=='y_TWD / 1e6' and meta['inverse_formula']=='max(z * 1e6, 0)','Wrong target units/formula')
    require(meta['plan_sha256']==sha(OUT/'candidate_plan.json') and meta['stage']==stage and meta['round']==13,'Wrong model provenance')
    require(meta['geo_manifest_sha256']==sha(driver.c6.OUT/'geography'/stage/'manifest.json'),'Wrong map manifest')
    model=CatBoostRegressor(); model.load_model(str(folder/'model.cbm'))
    augmented,cats=independent_model_view(Xq,geoq,data['categorical_columns'])
    require(model.tree_count_==2999 and len(augmented.columns)==96 and len(cats)==24,'Wrong model size')
    require(model.feature_names_==list(augmented.columns)==state['all_feature_names'],'Wrong model features')
    require(model.get_cat_feature_indices()==sorted(list(augmented.columns).index(c) for c in cats),'Wrong categorical indices')
    raw=np.asarray(model.predict(Pool(augmented,cat_features=cats,thread_count=4),thread_count=4),dtype='float64')
    saved_raw=np.load(folder/'raw_z.npy',allow_pickle=False)
    require(saved_raw.dtype==np.dtype('float64') and saved_raw.shape==raw.shape==(len(Xq),) and np.isfinite(raw).all() and np.array_equal(saved_raw.view('uint64'),raw.view('uint64')),'All-row raw z replay differs')
    prices=inverse_replay(raw)
    saved_prices=np.load(folder/'predictions.npy',allow_pickle=False)
    require(saved_prices.dtype==np.dtype('float64'), 'Saved TWD predictions must be float64')
    replays={'component_TWD':compare_vector(saved_prices,prices,'component_TWD')}
    raw_diagnostic={'rows':len(raw),'negative_count':int(np.sum(raw<0)),'zero_count':int(np.sum(raw==0)), 'min':float(raw.min()),'max':float(raw.max()),'positive_count':int(np.sum(raw>0)),
                    'zero_price_count':int(np.sum(prices==0)),
                    'raw_values_sha256':hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
                    'TWD_values_sha256':hashlib.sha256(prices.astype('<f8').tobytes()).hexdigest(),
                    'inverse_matches_component_all_rows':True,'used_for_tuning':False}
    # Actual driver metadata's complete schema is checked by its model_check; independently compare the core diagnostics.
    require(all(meta['raw_z_diagnostics'][k]==v for k,v in raw_diagnostic.items()),'Raw z diagnostics differ')
    reference,_=driver.c6.load_reference(stage,X)
    base=driver.c6.baseline_folder(stage)
    old=np.load(base/'catboost_raw.npy',allow_pickle=False)
    unit=np.load(base/'lightgbm_l1_unit63.npy',allow_pickle=False)
    old_geo=np.load(driver.c6.OUT/'models'/stage/'geography_catboost/predictions.npy',allow_pickle=False)
    control=independent_blend(old,old_geo,unit,reference,Xq,driver.predict_reference)
    candidate=independent_blend(old,prices,unit,reference,Xq,driver.predict_reference)
    original=np.load(driver.c6.OUT/'evaluation'/stage/'geography_blend.npy',allow_pickle=False)
    replays['original_v3_TWD']=compare_vector(original,control,'original_v3_TWD')
    for name,values in [(CONTROL,control),(CANDIDATE,candidate)]:
        saved=np.load(OUT/'evaluation'/stage/(name+'.npy'),allow_pickle=False)
        replays[name]=compare_vector(saved,values,name)
    metrics={name:independent_score(yq,values,X,Xq,start,weights) for name,values in [(CONTROL,control),(CANDIDATE,candidate)]}
    require(metrics==result['metrics'],'Independent all metrics differ')
    decision=independent_decision(stage,metrics)
    require(decision==result['decision'],'Independent decision differs')
    driver.require_no_staging(); driver.authorize(stage); driver.read_result(stage)
    for name,digest in captured.items(): require(sha(ROOT/name)==digest,'Audited input changed: '+name)
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'round':13,'stage':stage,'status':'passed',
           'candidate_accepted_in_this_stage':decision['passed'],'fit_rows':len(X),'query_rows':len(Xq),
           'source_sha256':sha(Path(__file__)),'plan_sha256':sha(OUT/'candidate_plan.json'),
           'result_sha256':sha(OUT/'evaluation'/stage/'results.json'),
           'saved_raw_z_bitwise_equal':True,'raw_z_diagnostics':raw_diagnostic,'replays':replays,
           'independent_metrics':metrics,'independent_decision':decision,'synthetic_checks':checks,
           'source_input_sha256':captured,'protected_artifacts_unchanged':protected,
           'fit_performed':False,'new_window_scored':False,'existing_query_errors_recomputed':True,
           'shared_canonical_loader_deserializes_train_valid_targets':True,
           'model_prediction_reimplemented_directly':True,'inverse_reimplemented_independently':True,
           'model_view_reimplemented_independently':True,'raw_z_units':'million_TWD',
           'alias_column':ALIAS_COLUMN,'alias_mapping':{'桃園縣':'桃園市'},
           'original_95_values_dtypes_order_index_preserved':True,
           'original_87_reference_and_scoring_keys_unchanged':True,
           'legacy_reference_function_reused':True,'public_score_used':False,'historically_untouched':False}
    driver.locked_write(proof_path,proof)
    print(json.dumps({'event':'independent_replay_passed','stage':stage,'query_rows':len(Xq),'proof_sha256':sha(proof_path),'raw_z_diagnostics':raw_diagnostic,'replays':replays}),flush=True)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['self-check','verify'])
    parser.add_argument('--stage',choices=list(SPLITS))
    args=parser.parse_args()
    if args.action=='self-check':
        checks=extra_self_checks()
        z=np.array([-1.,0.,2.])
        require(np.allclose(inverse_replay(z),[0.,0.,2e6],rtol=1e-14),'Inverse unit check')
        print(json.dumps({'checks':checks+['independent_inverse_units_and_zero_projection'],'real_data_read':False,'model_fit':False}),flush=True)
    else:
        require(args.stage is not None,'Verify requires an explicit saved stage')
        verify(args.stage)


if __name__=='__main__': main()
