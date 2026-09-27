"""Record R76 evaluation without changing a submission or the selector."""
from pathlib import Path
import json
import round48_nn_experiment as old

ROOT=old.ROOT

def run():
    paths={'results':'outputs/round76/results.json','execution':'outputs/round76/execution.json',
        'diagnostics':'reports/round76_diagnostics.json','diagnostics_execution':'outputs/round76/diagnostics/execution.json',
        'paired':'outputs/round76/paired_vs_v19/results.json','review':'outputs/round76/results_review.json',
        'first_failed_execution':'outputs/round76_failed_attempt1/execution.json',
        'first_failed_diagnostics':'outputs/round76/diagnostics_failed_attempt1/execution.json',
        'diagnostics_repair':'outputs/round76/diagnostics_failed_attempt1/repair_note.json'}
    data={k:old.locked(ROOT/p) for k,p in paths.items()}
    r=data['results'];d=data['diagnostics'];b=data['paired'];review=data['review']
    assert old.sha(ROOT/paths['results'])=='a7ea0219deec6aacaf43be9abd9b62e41b0cd720d09274d6fa81c5fa34d81631'
    assert r['status']=='three_periods_completed_no_adoption_decision' and r['checked_stages']==list(old.STAGES)
    assert data['execution']['exit_code']==data['diagnostics_execution']['exit_code']==0
    assert data['first_failed_execution']['exit_code']==data['first_failed_diagnostics']['exit_code']==1
    assert review['status']=='completed_JSON_only_review' and b['status']=='completed'
    assert d['status']=='completed_saved_predictions_only'
    assert r['new_fits']==r['tree_NN_model_loads']==r['model_predict_calls']==0
    assert r['reference_predict_calls']==r['spatial_predict_calls']==6
    assert r['protected_before']==r['protected_after'] and len(r['protected_before'])==21
    assert all(old.sha(ROOT/n)==h for n,h in r['protected_before'].items())
    assert all(old.sha(ROOT/n)==h for n,h in r['source_sha256'].items())
    selector_path=ROOT/'outputs/current_private_candidate.json';selector_sha=old.sha(selector_path)
    assert selector_sha==r['selector_sha256_before']==r['selector_sha256_after']
    selection=json.loads(selector_path.read_text(encoding='utf8'))
    assert selection['private_priority']=='outputs/submission_v19_trial.csv' and selection['private_mae'] is None
    latest=r['stages']['confirmation']['differences']['v19']
    assert all(latest[k]['delta_TWD']<0 for k in ('overall','weighted'))
    assert all(latest['strata'][k]['delta_TWD']<0 for k in ('new_nonmissing','seen'))
    assert all(v['delta_TWD']<0 for v in latest['forecast_quarters'].values())
    assert all(b['stages']['confirmation'][m]['summary'][k]['percentile_97_5']<0 for m in ('project','month') for k in ('overall','weighted'))
    next_plan='reports/round77_plan.md';report='reports/round76_report.md'
    assert (ROOT/next_plan).is_file() and (ROOT/report).is_file()
    summary={}
    for s in old.STAGES:
        a=r['stages'][s];dg=d['stages'][s]
        summary[s]={'metrics':a['metrics'],'differences':a['differences'],
            'unit_branch_differences':a['unit_branch_differences'],'test_composition':dg['test_composition'],
            'price_bands':dg['price_bands'],
            'price_band_contributions_vs_v19_TWD':{v['band']:(v['R76']-v['v19'])*v['rows']/a['query_rows'] for v in dg['price_bands']},
            'paired_intervals':{m:b['stages'][s][m]['summary'] for m in ('project','month')},
            'changed_unit_rows':a['changed_unit_rows'],'changed_final_rows':a['changed_final_rows']}
    record={'round':76,'decision':'advance_fixed_area_unit_interaction_comparison_keep_v19_pending_release',
        'reason':'Complete R76 improves all three overall and fixed-weighted MAEs. Newest-period new/seen and all quarters improve, and both paired interval methods are negative there. Middle-period test-composition MAE worsens, and earlier-period quarter/price-band tradeoffs and reused-window uncertainty remain. Test the predeclared fixed R74-area/R76-unit combination before a separate final release decision.',
        'candidate':'R76','primary_control':'v19','summary':summary,
        'new_fits':0,'new_final_fits':0,'new_CSV':False,'new_historical_candidate':True,
        'main_seconds':r['elapsed_seconds'],'execution_seconds':data['execution']['elapsed_seconds'],
        'private_priority':selection['private_priority'],'submission_sha256':selection['sha256'],
        'known_public_best_filename':selection['known_public_best_filename'],
        'known_public_best_mae':selection['known_public_best_mae'],'private_mae':None,
        'selector_updated':False,'selector_sha256':selector_sha,'kaggle_selection_changed':False,
        'final_version_chosen':False,'GitHub_publication_deferred':True,
        'next_formal_experiment':{'round':77,'plan':next_plan,'sha256':old.sha(ROOT/next_plan),
            'primary_comparison':'R77 fixed area/unit combination minus R76','started_by_this_recorder':False},
        'report':report,'report_sha256':old.sha(ROOT/report),
        'evidence':{k:{'path':p,'sha256':old.sha(ROOT/p)} for k,p in paths.items()},
        'source_sha256':old.sha(Path(__file__)),
        'limitations':['Single fixed seed; no new seed-stability evidence.',
            'Oldest and middle windows have mixed quarters and bootstrap intervals crossing zero.',
            'Middle-period test-composition MAE worsens and its >=100m band worsens; oldest-period middle three price bands worsen.',
            'All historical windows have been reused; bootstrap does not correct adaptive selection.',
            'Unit branch alone worsens in all periods; standalone quality does not determine median ensemble quality.',
            'About 78 percent of the newest-period overall gain comes from the two true-price bands at least 50m TWD (2044 rows); lower three bands also improve.',
            'True-price bands are diagnostic only; no truth-based or per-ID prediction gate.',
            'No R76 final test vector or CSV has been produced; Public and Private remain unknown.'],
        'completed_at_utc':old.now()}
    assert old.sha(selector_path)==selector_sha
    old.write(ROOT/'outputs/round76/decision.json',record)
    print(json.dumps({'decision':record['decision'],'decision_sha256':old.sha(ROOT/'outputs/round76/decision.json'),
        'new_CSV':False,'selector_updated':False,'next_round':77}))

if __name__=='__main__':run()
