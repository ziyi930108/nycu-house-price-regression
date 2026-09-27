"""Synthetic round-five gate, blend and mask-state checks; never fit a model.

Import is side-effect free. main() prints a JSON report; --output optionally
saves that report without overwriting different content. No real data/labels,
model artifacts, evaluation results, or plan/freeze contracts are read.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import round5_evaluate as e
import round5_project_mask as mask_helper
from round3_evaluate import project_keys as scoring_project_keys
from house_features import build_features

CANDIDATE = 'project_mask_blend'


def metric(value):
    return {'overall_mae':value,'weighted_mae':value,
            'forecast_quarters':{f'Q{i}':{'rows':300,'mae':value} for i in range(1,5)},
            'strata':{'seen':{'rows':600,'mae':value},'new_nonmissing':{'rows':600,'mae':value},
                      'missing':{'rows':0,'mae':None}}}


def pair(candidate, base=None):
    return {e.CONTROL:metric(100.) if base is None else base,CANDIDATE:candidate}


def rejected(call):
    try:
        call()
    except (RuntimeError,ValueError):
        return True
    return False


def check_gates():
    good, base = metric(98.), metric(100.)
    assert set(e.FORMULAS) == {e.CONTROL,CANDIDATE}
    assert e.REQUIRED_COMPONENTS == {e.CONTROL:[],CANDIDATE:['project_mask_catboost']}
    assert e.development_selection(pair(metric(99.)))['selected_candidate'] == CANDIDATE
    assert e.development_selection(pair(metric(99.000001)))['selected_candidate'] == e.CONTROL
    assert e.development_selection(pair(metric(99.5)))['selected_candidate'] == e.CONTROL
    for key in ['overall_mae','weighted_mae']:
        bad = copy.deepcopy(good); bad[key] = 100.
        assert not e.development_selection(pair(bad))['qualified_candidates']
    for field,key in [('forecast_quarters','Q4'),('strata','new_nonmissing')]:
        boundary = copy.deepcopy(good); boundary[field][key]['mae'] = 105.
        assert e.development_selection(pair(boundary))['selected_candidate'] == CANDIDATE
        bad = copy.deepcopy(boundary); bad[field][key]['mae'] = 105.000001
        assert e.development_selection(pair(bad))['selected_candidate'] == e.CONTROL
    small_base, small_candidate = copy.deepcopy(base), copy.deepcopy(good)
    small_base['strata']['new_nonmissing']['rows'] = 499
    small_candidate['strata']['new_nonmissing'].update(rows=499,mae=106.)
    assert e.development_selection(pair(small_candidate,small_base))['selected_candidate'] == CANDIDATE
    small_base['strata']['new_nonmissing']['rows'] = 500
    small_candidate['strata']['new_nonmissing']['rows'] = 500
    assert e.development_selection(pair(small_candidate,small_base))['selected_candidate'] == e.CONTROL
    assert rejected(lambda:e.development_selection({**pair(good),'unplanned_candidate':good}))
    assert e.stage_decision('confirmation',pair(good),CANDIDATE)['passed']
    one = copy.deepcopy(good)
    for quarter in ['Q2','Q3','Q4']:
        one['forecast_quarters'][quarter]['mae'] = 100.
    assert not e.stage_decision('confirmation',pair(one),CANDIDATE)['passed']
    two = copy.deepcopy(one); two['forecast_quarters']['Q2']['mae'] = 99.
    assert e.stage_decision('confirmation',pair(two),CANDIDATE)['passed']
    equal = copy.deepcopy(two); equal['overall_mae'] = equal['weighted_mae'] = 100.
    assert e.stage_decision('confirmation',pair(equal),CANDIDATE)['passed']
    for key in ['overall_mae','weighted_mae']:
        bad = copy.deepcopy(two); bad[key] = 100.000001
        assert not e.stage_decision('confirmation',pair(bad),CANDIDATE)['passed']
    assert e.stage_decision('auxiliary',pair(metric(103.)),CANDIDATE)['passed']
    for key in ['overall_mae','weighted_mae']:
        bad = metric(102.); bad[key] = 103.000001
        assert not e.stage_decision('auxiliary',pair(bad),CANDIDATE)['passed']
    for stage in ['confirmation','auxiliary']:
        bad = copy.deepcopy(good); bad['strata']['seen']['mae'] = 105.000001
        assert not e.stage_decision(stage,pair(bad),CANDIDATE)['passed']
        assert not e.stage_decision(stage,{e.CONTROL:base},e.CONTROL)['passed']
        assert rejected(lambda:e.stage_decision(stage,{**pair(good),'unplanned_candidate':good},CANDIDATE))
    diagnostic_only = metric(102.); diagnostic_only['forecast_quarters']['Q4']['mae'] = 9999.
    assert e.stage_decision('auxiliary',pair(diagnostic_only),CANDIDATE)['passed']
    assert good['strata']['missing']['mae'] is None


def check_blend():
    original, calls = e.predict_reference, []
    query = pd.DataFrame({'unmodified_reference_feature':[1,2]})
    query_before = query.copy(deep=True)
    def probe(reference, supplied, base, variant):
        assert supplied is query and supplied.equals(query_before)
        calls.append(variant)
        return base + 7.125
    try:
        e.predict_reference = probe
        result = e.build_predictions(list(e.FORMULAS),np.array([10.25,20.5]),np.array([20.25,40.5]),
            {'project_mask_catboost':np.array([30.25,60.5])},{},query)
    finally:
        e.predict_reference = original
    np.testing.assert_array_equal(result[e.CONTROL],np.array([19.875,32.625]))
    np.testing.assert_array_equal(result[CANDIDATE],np.array([27.375,47.625]))
    assert calls == ['trend','trend'] and query.equals(query_before)
    assert rejected(lambda:e.build_predictions(['unplanned_candidate'],np.ones(2),np.ones(2),{},{},query))


def check_mask_state():
    rows = []
    for project in ['合成建案甲','合成建案乙',None]:
        for floor in [3,7]:
            rows.append({'鄉鎮市區':'新市區','土地位置建物門牌':'臺南市新市區中正路10號',
                         '建案名稱':project,'棟及號':f'A棟{floor}樓','建物型態':'住宅大樓',
                         '建物移轉總面積平方公尺':100.+floor,'移轉層次':str(floor),'總樓層數':15})
    fit = build_features(pd.DataFrame(rows))
    query = fit.iloc[[0,2,4]].copy().reset_index(drop=True)
    query.loc[0,'project_name'] = '合成未見新案'
    before = query.copy(deep=True)
    state = mask_helper.fit_feature_state(fit)
    flags_before = copy.deepcopy(state['project_mask_flags'])
    verified = e.verify_mask_state(fit,query,state)
    assert verified['fit_X_state_reconstruction_verified'] and verified['query_mask_state_unchanged']
    assert query.equals(before) and state['project_mask_flags'] == flags_before
    bad = copy.deepcopy(state)
    key = next(iter(bad['project_mask_flags'])); bad['project_mask_flags'][key] = not bad['project_mask_flags'][key]
    assert rejected(lambda:e.verify_mask_state(fit,query,bad))
    # Distinguish real missing values from literal names exactly as the unchanged scorer does.
    sentinels = [None,np.nan,'NaN','null','nan','None','<NA>','NaT','__MISSING__','  合成建案甲  ','']
    normalized_query = pd.concat([fit.iloc[[0]]] * len(sentinels),ignore_index=True)
    normalized_query['project_name'] = pd.Series(sentinels,dtype=object)
    mk, mm = mask_helper.project_keys(normalized_query)
    sk, sm = scoring_project_keys(normalized_query)
    assert mk.reset_index(drop=True).equals(sk.reset_index(drop=True)), 'Mask keys differ from unchanged scoring keys'
    assert np.array_equal(mm.to_numpy(),sm.to_numpy()), 'Mask and scorer disagree about missing projects'
    helper_result = mask_helper.self_check()
    assert helper_result['status'] == 'passed' and helper_result['model_training_performed'] is False
    return helper_result


def run_checks():
    check_gates(); check_blend(); helper = check_mask_state()
    return {'status':'passed','single_predeclared_candidate_enforced':True,
            'development_1pct_boundary_and_both_main_metrics_verified':True,
            'quarter_and_large_stratum_5pct_boundaries_verified':True,'large_stratum_500_rows_boundary_verified':True,
            'confirmation_not_worse_and_two_quarters_verified':True,'auxiliary_3pct_boundary_verified':True,
            'auxiliary_quarters_diagnostic_only_verified':True,'control_cannot_be_accepted':True,
            'empty_missing_group_retained_as_null':True,'unrounded_TWD_blend_verified':True,
            'one_trend_call_per_candidate_verified':True,'reference_received_original_query':True,
            'fit_X_state_reconstruction_verified':True,'query_did_not_mutate_state_or_features':True,
            'tampered_mask_state_rejected':True,'mask_and_score_geographic_key_semantics_match':True,
            'helper_synthetic_checks':helper,'uses_synthetic_inputs_only':True,
            'real_models_trained':False,'real_scores_observed':False,'plan_or_freeze_created':False,
            'source_sha256':{'src/round5_evaluate.py':hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest(),
                'src/round5_project_mask.py':hashlib.sha256(Path(mask_helper.__file__).read_bytes()).hexdigest(),
                'src/round5_preflight.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new JSON report path; never overwrite different content')
    args = parser.parse_args()
    report = run_checks()
    payload = json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output is not None:
        path = args.output.resolve()
        if not path.is_relative_to(e.ROOT.resolve()):
            raise RuntimeError('Preflight output must stay inside the workspace')
        if path.exists():
            if path.read_text(encoding='utf-8') != payload:
                raise RuntimeError('Existing preflight report differs; refusing overwrite')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as handle:
                handle.write(payload)
    print(payload,flush=True)


if __name__ == '__main__':
    main()
