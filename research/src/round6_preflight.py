"""Round-six draft preflight: synthetic gates and target-free geography contracts.

No data, plan, model or evaluation artifacts are loaded at import. The default
CLI uses synthetic inputs only. Optional evaluator/helper modules must expose
pure functions; no fitting or scoring of real observations is performed here.
Alignment receives a caller-supplied canonical keep mask: the established
model_common target-inclusive duplicate policy is reproduced outside this module.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

NUM = ['geo_latitude','geo_longitude','geo_log_road_length','geo_log_extent',
       'geo_has_position','geo_nearby_fragments']
CAT = ['geo_match_status','geo_grid_002']
USABLE = {'road_level','planar_intersection'}
RAW_GEO = {'source_id':'編號','source_address':'土地位置建物門牌',
           'source_district':'鄉鎮市區','source_transaction_type':'交易標的'}
RAW_NUM = {'geo_latitude':'geo_latitude','geo_longitude':'geo_longitude',
           'geo_log_road_length':'geo_road_length_m','geo_log_extent':'geo_extent_diagonal_m',
           'geo_nearby_fragments':'geo_nearby_fragment_grouping'}
THRESHOLDS = {
 'development_min_overall_improvement':.01,'development_min_weighted_improvement':.01,
 'development_max_quarter_deterioration':.05,'large_stratum_min_rows':500,
 'max_large_stratum_deterioration':.05,
 'confirmation_max_overall_deterioration':0.,'confirmation_max_weighted_deterioration':0.,
 'confirmation_min_improved_quarters':2,'auxiliary_max_overall_deterioration':.03,
 'auxiliary_max_weighted_deterioration':.03,'auxiliary_quarters_diagnostic_only':True}
SNAPSHOTS = {'auxiliary':(2021,'2021-08-11T00:00:00+08:00'),
             'development':(2022,'2022-08-11T00:00:00+08:00'),
             'confirmation':(2023,'2023-08-11T00:00:00+08:00'),
             'final':(2024,'2024-08-11T00:00:00+08:00')}
R3_MANIFEST_SHA = '96cee2c7977d9317f70e5e64b41cbf057db8fad529019a2e76fb37d1606a202f'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(root, expected):
    root = Path(root).resolve()
    require(bool(expected),'Empty provenance hash map')
    for relative, value in expected.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root),'Provenance path escapes workspace')
        require(path.is_file() and sha256(path) == value,f'Provenance SHA differs: {relative}')
    return {'verified_files':len(expected)}


def validate_snapshot(stage, contract):
    """Check an adapter contract populated from extraction receipts, not filenames.

    Required fields: snapshot_year, replication_timestamp, timestamp_evidence,
    raw_snapshot_sha256, source_sha256, input_sha256. Caller verifies hash maps
    with verify_hashes, including the extraction audit containing the timestamp.
    A planar XY intersection is allowed without grade connectivity evidence.
    """
    require(stage in SNAPSHOTS,'Unknown forecast stage')
    year, cutoff = SNAPSHOTS[stage]
    require(contract['snapshot_year'] == year,'Snapshot year differs from predeclared stage')
    timestamp = datetime.fromisoformat(contract['replication_timestamp'].replace('Z','+00:00'))
    require(timestamp.tzinfo is not None,'Snapshot timestamp needs explicit timezone')
    require(timestamp.year == year,'Actual snapshot timestamp is outside assigned year')
    require(timestamp < datetime.fromisoformat(cutoff),'Snapshot was not available before forecast origin')
    require(contract['timestamp_evidence'] == 'pbf_header_osmosis_replication_timestamp','Timestamp must come from PBF header')
    require(len(contract['raw_snapshot_sha256']) == 64,'Missing raw snapshot SHA')
    require(bool(contract['source_sha256']) and bool(contract['input_sha256']),'Missing source/input provenance')
    require(contract.get('geographic_extraction_target_free') is True,'Geographic extraction must be target free')
    require(contract.get('matching_target_free') is True,'Road matching must be target free')
    require(contract.get('exact_city_district_road_keys') is True,'Geographic keys must preserve city/district scope')
    require(contract.get('nearest_point_guessing') is False,'Do not fabricate nearest intersection or city')
    require(contract.get('query_label_feature_selection') is False,'Query labels cannot select geography rules')
    require(contract.get('grade_connectivity_verified') is False,'This experiment only claims planar intersections')
    return {'stage':stage,'snapshot_year':year,'strictly_before_origin':True,
            'origin_as_of_not_individual_transaction_as_of':True,
            'planar_intersections_allowed':True,'grade_connectivity_verified':False}


def _equal_series(left, right, label):
    try:
        pd.testing.assert_series_equal(left.reset_index(drop=True),right.reset_index(drop=True),
                                       check_names=False,check_dtype=False,check_exact=True)
    except AssertionError as exc:
        raise ValueError(f'Row identity differs: {label}') from exc


def validate_alignment(parsed, matched, raw_projection, retained_positions, canonical_projection):
    """Verify row positions and four non-price fields; never join on nonunique IDs.

    raw_projection is the 455784 header-clean cache before old duplicate removal;
    canonical_projection is load_data raw_train projected to the same four fields.
    retained_positions comes from the EXISTING target-inclusive duplicated mask.
    Synthetic/valid/test sizes also work; production wrapper must assert its counts.
    This checker does not infer or replace the canonical duplicate policy.
    """
    fields = list(RAW_GEO.values())
    require(set(raw_projection.columns) == set(fields),'Provide only the four raw geography fields')
    require(set(canonical_projection.columns) == set(fields),'Provide only canonical geography fields')
    require(len(parsed) == len(matched),'Parsed/matched row count differs')
    for field in ['source_row_index','source_excel_row','source_id','existing_feature_cache_row_index']:
        require(field in parsed and field in matched,f'Missing row provenance: {field}')
        _equal_series(parsed[field],matched[field],field)
    require(np.array_equal(parsed['source_row_index'].to_numpy(),np.arange(len(parsed))),
            'Original source rows were reordered or dropped')
    require(np.array_equal(parsed['source_excel_row'].to_numpy(),np.arange(len(parsed))+2),
            'Excel source positions changed')
    header = parsed['is_embedded_header'].to_numpy(dtype=bool)
    positions = parsed['existing_feature_cache_row_index'].to_numpy()
    require((positions[header] == -1).all(),'Embedded header acquired a cache row')
    require(np.array_equal(positions[~header],np.arange(len(raw_projection))),
            'Non-header cache row mapping is not contiguous and exact')
    geo = parsed.loc[~header].reset_index(drop=True)
    for source, raw in RAW_GEO.items():
        _equal_series(geo[source],raw_projection[raw],f'parsed to raw cache: {raw}')
    keep = np.asarray(retained_positions)
    require(keep.ndim == 1 and np.issubdtype(keep.dtype,np.integer),'Canonical positions must be an integer vector')
    require(len(keep) == len(canonical_projection),'Canonical retained row count differs')
    require((keep >= 0).all() and (keep < len(raw_projection)).all(),'Canonical position out of bounds')
    require((np.diff(keep) > 0).all(),'Canonical positions duplicated or reordered')
    for source, raw in RAW_GEO.items():
        _equal_series(geo.iloc[keep][source],canonical_projection[raw],f'canonical retained: {raw}')
    return {'parsed_rows':len(parsed),'embedded_headers':int(header.sum()),
            'header_clean_rows':len(raw_projection),'canonical_rows':len(keep),
            'retained_positions_int64_sha256':hashlib.sha256(keep.astype('<i8').tobytes()).hexdigest(),
            'join_on_id_used':False,'canonical_keep_policy_verified_here':False,
            'canonical_keep_policy':'caller reproduces existing all-columns-except-ID including target',
            'geography_values_and_row_identity_checked':True}


def expected_geo8(matched):
    """Independent encoder oracle for checks, never used to construct model inputs."""
    require('geo_match_status' in matched,'Missing geographic status')
    status = matched['geo_match_status'].astype(str)
    usable = status.isin(USABLE).to_numpy()
    result = pd.DataFrame(index=matched.index)
    for output, raw in RAW_NUM.items():
        values = pd.to_numeric(matched[raw],errors='raise').to_numpy(dtype='float64')
        if output in ['geo_latitude','geo_longitude']:
            require(np.isnan(values[~usable]).all(),'Rejected coordinates must be NaN')
            require(np.isfinite(values[usable]).all(),'Usable coordinates must be finite')
        else:
            require(np.isfinite(values[usable]).all(),'Usable geometry needs finite uncertainty fields')
            require((np.isnan(values) | np.isfinite(values)).all(),'Infinite geographic uncertainty')
            if output == 'geo_nearby_fragments':
                require((np.isnan(values) | np.isin(values,[0.,1.])).all(),'Invalid fragment flag')
            else:
                require((np.isnan(values) | (values >= 0)).all(),'Negative geographic length/extent')
        values = values.copy(); values[~usable] = np.nan
        if output in ['geo_log_road_length','geo_log_extent']:
            values = np.log1p(values)
        result[output] = values.astype('float32')
    lat = matched['geo_latitude'].to_numpy(dtype='float64')
    lon = matched['geo_longitude'].to_numpy(dtype='float64')
    require(((lat[usable] >= 20)&(lat[usable] <= 28)&(lon[usable] >= 116)&(lon[usable] <= 125)).all(),
            'Usable coordinate outside fixed Taiwan bounds, or fabricated 0,0')
    grid = np.full(len(matched),'__MISSING__',dtype=object)
    grid[usable] = [f'{int(np.floor(a/.02))}:{int(np.floor(b/.02))}' for a,b in zip(lat[usable],lon[usable])]
    result['geo_has_position'] = usable.astype('float32')
    result['geo_match_status'] = status
    result['geo_grid_002'] = grid
    return result[NUM+CAT]


def validate_feature_append(original_X, matched, geo8, augmented_X, original_cats):
    require(len(original_X.columns) == 87 and len(original_cats) == 21,'Original feature schema differs')
    require(len(set(original_cats)) == 21 and set(original_cats).issubset(original_X.columns),'Original categorical schema differs')
    require(list(geo8.columns) == NUM+CAT,'Geographic feature schema/order differs')
    require(list(augmented_X.columns) == list(original_X.columns)+NUM+CAT,'Expected exactly 87+8 features')
    require(len(original_X) == len(matched) == len(geo8) == len(augmented_X),'Feature rows differ')
    require(original_X.index.equals(matched.index) and original_X.index.equals(geo8.index),
            'Geographic feature index differs')
    pd.testing.assert_frame_equal(augmented_X[list(original_X.columns)],original_X,check_exact=True)
    pd.testing.assert_frame_equal(augmented_X[NUM+CAT],geo8,check_exact=True)
    oracle = expected_geo8(matched)
    for column in NUM:
        require(geo8[column].dtype == np.dtype('float32'),f'Expected float32 feature: {column}')
        np.testing.assert_array_equal(geo8[column].to_numpy(),oracle[column].to_numpy())
    for column in CAT:
        require(not geo8[column].isna().any(),f'Categorical feature contains null: {column}')
        _equal_series(geo8[column],oracle[column],column)
    return {'original_features_unchanged':True,'features':95,'categorical_features':23,
            'float64_grid_before_float32_conversion_verified':True,
            'unpositioned_fields_nan_not_zero':True,'planar_intersections_allowed':True}


def verify_archived_baseline(root, stage, paths):
    require(stage in ('development','confirmation'),'Auxiliary has a separately trained control')
    root = Path(root).resolve()
    manifest_path = root/'runs/round3_experiments/snapshot_manifest.json'
    require(sha256(manifest_path) == R3_MANIFEST_SHA,'Frozen R3 snapshot manifest changed')
    receipt = manifest_path.with_suffix('.sha256')
    require(receipt.read_text(encoding='utf-8').strip().split()[0] == R3_MANIFEST_SHA,'R3 manifest receipt changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    prefix = f'outputs/round3/baseline/{stage}/'
    require(bool(paths),'No baseline paths supplied')
    for item in paths:
        relative = Path(item).as_posix()
        require(relative.startswith(prefix),'Wrong stage or baseline folder')
        record = manifest['copied_files'][relative]
        require(record['original_sha256'] == record['backup_sha256'],'Archived original and backup hashes differ')
        verify_hashes(root,{relative:record['original_sha256'],record['backup_path']:record['backup_sha256']})
    return {'verified_files':len(paths),'r3_manifest_sha256':R3_MANIFEST_SHA,
            'historically_untouched':False,'loaded_model_or_predictions':False}


def metric(value):
    return {'overall_mae':value,'weighted_mae':value,
            'forecast_quarters':{f'Q{i}':{'rows':300,'mae':value} for i in range(1,5)},
            'strata':{'seen':{'rows':600,'mae':value},'new_nonmissing':{'rows':600,'mae':value},
                      'missing':{'rows':0,'mae':None}}}


def _rejected(call):
    try:
        call()
    except (ValueError,RuntimeError,AssertionError):
        return True
    return False


def check_evaluator(e, candidate, component):
    """Probe a supplied evaluator with invented metrics and arrays only."""
    require(e.THRESHOLDS == THRESHOLDS,'Thresholds differ from R5')
    require(set(e.FORMULAS) == {e.CONTROL,candidate},'Exactly one new candidate required')
    require(e.REQUIRED_COMPONENTS == {e.CONTROL:[],candidate:[component]},'Unexpected model components')
    pair = lambda value,base=None:{e.CONTROL:metric(100.) if base is None else base,candidate:value}
    require(e.development_selection(pair(metric(99.)))['selected_candidate'] == candidate,'1% equality failed')
    require(e.development_selection(pair(metric(99.000001)))['selected_candidate'] == e.CONTROL,'Sub-1% improvement admitted')
    for field in ['overall_mae','weighted_mae']:
        value = metric(98.); value[field] = 100.
        require(e.development_selection(pair(value))['selected_candidate'] == e.CONTROL,'Both main gates required')
    for section,key in [('forecast_quarters','Q4'),('strata','new_nonmissing')]:
        value = metric(98.); value[section][key]['mae'] = 105.
        require(e.development_selection(pair(value))['selected_candidate'] == candidate,'5% equality failed')
        value[section][key]['mae'] = 105.000001
        require(e.development_selection(pair(value))['selected_candidate'] == e.CONTROL,'5% deterioration admitted')
    for count,selected in [(499,candidate),(500,e.CONTROL)]:
        value,base = metric(98.),metric(100.)
        value['strata']['new_nonmissing'].update(rows=count,mae=106.)
        base['strata']['new_nonmissing']['rows'] = count
        require(e.development_selection(pair(value,base))['selected_candidate'] == selected,'Large group boundary changed')
    one = metric(98.)
    for quarter in ['Q2','Q3','Q4']:
        one['forecast_quarters'][quarter]['mae'] = 100.
    require(not e.stage_decision('confirmation',pair(one),candidate)['passed'],'Confirmation requires two improved quarters')
    two = copy.deepcopy(one); two['forecast_quarters']['Q2']['mae'] = 99.
    two['overall_mae'] = two['weighted_mae'] = 100.
    require(e.stage_decision('confirmation',pair(two),candidate)['passed'],'Confirmation equality failed')
    for field in ['overall_mae','weighted_mae']:
        bad = copy.deepcopy(two); bad[field] = 100.000001
        require(not e.stage_decision('confirmation',pair(bad),candidate)['passed'],'Confirmation main deterioration admitted')
        bad = metric(102.); bad[field] = 103.000001
        require(not e.stage_decision('auxiliary',pair(bad),candidate)['passed'],'Auxiliary >3% admitted')
    require(e.stage_decision('auxiliary',pair(metric(103.)),candidate)['passed'],'Auxiliary 3% equality failed')
    diagnostic = metric(102.); diagnostic['forecast_quarters']['Q4']['mae'] = 9999.
    require(e.stage_decision('auxiliary',pair(diagnostic),candidate)['passed'],'Auxiliary quarter became a new gate')
    for stage in ['confirmation','auxiliary']:
        bad = metric(98.); bad['strata']['seen']['mae'] = 105.000001
        require(not e.stage_decision(stage,pair(bad),candidate)['passed'],'Later large-stratum veto failed')
        require(not e.stage_decision(stage,{e.CONTROL:metric(100.)},e.CONTROL)['passed'],'Control cannot be a new accepted candidate')
        require(_rejected(lambda:e.stage_decision(stage,{**pair(metric(98.)),'extra':metric(90.)},candidate)), 'Unplanned candidate admitted')
    require(_rejected(lambda:e.development_selection({**pair(metric(98.)),'extra':metric(90.)})),'Unplanned development candidate admitted')
    original, calls = e.predict_reference, []
    query = pd.DataFrame({'original_feature':[1,2]})
    before = query.copy(deep=True)
    def probe(reference, supplied, base, variant):
        require(supplied is query and supplied.equals(before),'Reference did not receive original X')
        calls.append(variant)
        return base + 7.125
    try:
        e.predict_reference = probe
        result = e.build_predictions(list(e.FORMULAS),np.array([10.25,20.5]),np.array([20.25,40.5]),
                                    {component:np.array([30.25,60.5])},{},query)
    finally:
        e.predict_reference = original
    np.testing.assert_array_equal(result[e.CONTROL],np.array([19.875,32.625]))
    np.testing.assert_array_equal(result[candidate],np.array([27.375,47.625]))
    require(calls == ['trend','trend'],'Exactly one trend call per candidate required')
    return {'synthetic_gates_passed':True,'single_candidate_and_fixed_blend_passed':True,
            'original_X_reference_and_once_only_trend_passed':True,'real_scores_observed':False}


def synthetic_frames():
    index = pd.Index([9,2,17,4],name='synthetic_row')
    matched = pd.DataFrame({
        'geo_match_status':['road_level','planar_intersection','disconnected','ineligible_address'],
        'geo_latitude':[25.03999999,25.08,np.nan,np.nan],
        'geo_longitude':[121.57999999,121.64,np.nan,np.nan],
        'geo_road_length_m':[100.,500.,900.,np.nan],
        'geo_extent_diagonal_m':[40.,70.,500.,np.nan],
        'geo_nearby_fragment_grouping':[0.,1.,1.,np.nan]},index=index)
    cats = [f'old_cat_{i}' for i in range(21)]
    X = pd.DataFrame({name:['a','b','c','d'] for name in cats},index=index)
    for i in range(66):
        X[f'old_numeric_{i}'] = np.arange(4,dtype='float32') + i
    return X,matched,cats


def check_helper(helper):
    """Exercise only the proposed helper's encoder and append functions, not fit."""
    X,matched,cats = synthetic_frames()
    original,source = X.copy(deep=True),matched.copy(deep=True)
    geo8 = helper.encode_geo_features(matched)
    augmented = helper.augment_features(X,geo8)
    result = validate_feature_append(X,matched,geo8,augmented,cats)
    pd.testing.assert_frame_equal(X,original,check_exact=True)
    pd.testing.assert_frame_equal(matched,source,check_exact=True)
    return {**result,'helper_inputs_unmodified':True,'helper_model_fit_called':False}


def run_checks():
    checks = []
    contract = {'snapshot_year':2022,'replication_timestamp':'2022-01-01T21:21:26Z',
                'timestamp_evidence':'pbf_header_osmosis_replication_timestamp',
                'raw_snapshot_sha256':'a'*64,'source_sha256':{'synthetic.py':'b'*64},
                'input_sha256':{'synthetic.pbf':'a'*64},'geographic_extraction_target_free':True,
                'matching_target_free':True,'exact_city_district_road_keys':True,
                'nearest_point_guessing':False,'query_label_feature_selection':False,
                'grade_connectivity_verified':False}
    validate_snapshot('development',contract); checks.append('correct_pbf_timestamp_contract')
    require(_rejected(lambda:validate_snapshot('auxiliary',contract)),'2022 snapshot admitted into 2021 auxiliary')
    checks.append('wrong_snapshot_year_rejected')
    for change in [{'replication_timestamp':'2022-08-10T16:00:00Z'},
                   {'replication_timestamp':'2022-01-01T00:00:00'},
                   {'timestamp_evidence':'filename'}, {'nearest_point_guessing':True},
                   {'matching_target_free':False}, {'query_label_feature_selection':True}]:
        require(_rejected(lambda:validate_snapshot('development',{**contract,**change})),f'Invalid snapshot contract accepted: {change}')
    checks.append('origin_boundary_timezone_evidence_and_geographic_leakage_guards')
    raw = pd.DataFrame({'編號':[7,7,8,9],'土地位置建物門牌':['甲路','乙路','丙路','丁路'],
                        '鄉鎮市區':['甲區']*4,'交易標的':['房地']*4})
    parsed = pd.DataFrame({'source_row_index':np.arange(5),'source_excel_row':np.arange(5)+2,
                           'source_id':[7,'serial number',7,8,9],
                           'existing_feature_cache_row_index':[0,-1,1,2,3],
                           'is_embedded_header':[False,True,False,False,False],
                           'source_address':['甲路','address','乙路','丙路','丁路'],
                           'source_district':['甲區','district','甲區','甲區','甲區'],
                           'source_transaction_type':['房地','type','房地','房地','房地']})
    matched = parsed.copy(deep=True); keep = np.array([0,2,3],dtype='int64')
    canonical = raw.iloc[keep].reset_index(drop=True)
    validate_alignment(parsed,matched,raw,keep,canonical)
    checks.append('embedded_header_mapping_and_nonunique_ID_alignment')
    bad = matched.copy(deep=True); bad.loc[0,'existing_feature_cache_row_index'] = 1
    require(_rejected(lambda:validate_alignment(parsed,bad,raw,keep,canonical)),'Wrong cache map admitted')
    bad_canonical = canonical.copy(deep=True); bad_canonical.loc[0,'土地位置建物門牌'] = '乙路'
    require(_rejected(lambda:validate_alignment(parsed,matched,raw,keep,bad_canonical)), 'Same ID with wrong address admitted')
    require(_rejected(lambda:validate_alignment(parsed,matched,raw,keep[::-1],canonical)),'Reordered positions admitted')
    checks.append('cache_row_address_and_order_corruption_rejected')
    X,matched,cats = synthetic_frames(); geo8 = expected_geo8(matched)
    augmented = pd.concat([X,geo8],axis=1)
    validate_feature_append(X,matched,geo8,augmented,cats)
    require(geo8.loc[9,'geo_grid_002'] == '1251:6078','Grid must use original float64 coordinate')
    require(np.isnan(geo8.loc[17,'geo_log_road_length']),'Rejected road leaked uncertainty values')
    checks.append('95_features_23_cats_float64_floor_grid_and_planar_intersection')
    bad = augmented.copy(deep=True); bad.iloc[0,22] = 12345.
    require(_rejected(lambda:validate_feature_append(X,matched,geo8,bad,cats)),'Original feature mutation admitted')
    bad_geo = geo8.copy(deep=True); bad_geo.loc[17,'geo_latitude'] = 0.
    require(_rejected(lambda:validate_feature_append(X,matched,bad_geo,pd.concat([X,bad_geo],axis=1),cats)),'Zero coordinate imputation admitted')
    bad_geo = geo8.copy(deep=True); bad_geo.loc[9,'geo_grid_002'] = '1252:6079'
    require(_rejected(lambda:validate_feature_append(X,matched,bad_geo,pd.concat([X,bad_geo],axis=1),cats)),'Grid after float32 rounding admitted')
    checks.append('original_feature_zero_imputation_and_wrong_grid_rejected')
    return {'status':'passed','draft':True,'checks':checks,'uses_synthetic_inputs_only':True,
            'real_data_or_labels_loaded':False,'real_models_trained':False,'real_scores_observed':False,
            'plan_or_freeze_created':False,'production_alignment_snapshot_and_R6_evaluator_not_yet_certified':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator',help='Optional module exposing pure gate/blend functions')
    parser.add_argument('--candidate',help='Exact single candidate key in optional evaluator')
    parser.add_argument('--component',help='Exact component key in optional evaluator')
    parser.add_argument('--helper',help='Optional module exposing encode_geo_features and augment_features')
    args = parser.parse_args()
    result = run_checks()
    if args.evaluator:
        require(bool(args.candidate) and bool(args.component),'Candidate/component names required for evaluator probe')
        module = importlib.import_module(args.evaluator)
        result['evaluator_module'] = args.evaluator
        result['evaluator_synthetic_checks'] = check_evaluator(module,args.candidate,args.component)
        result['evaluator_source_sha256'] = sha256(module.__file__)
    if args.helper:
        module = importlib.import_module(args.helper)
        result['helper_module'] = args.helper
        result['helper_synthetic_checks'] = check_helper(module)
        result['helper_source_sha256'] = sha256(module.__file__)
    result['preflight_source_sha256'] = sha256(__file__)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
