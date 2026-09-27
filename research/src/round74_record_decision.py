"""Record R74 evidence without promoting a candidate or changing submissions."""
from pathlib import Path
from datetime import datetime, timezone
import json
import round48_nn_experiment as old

ROOT=old.ROOT
STAGES=('development','confirmation','auxiliary')


def run():
    paths={'results':'outputs/round74/results.json','execution':'outputs/round74/execution.json',
           'diagnostics':'reports/round74_diagnostics.json','paired':'outputs/round74/paired_vs_v19/results.json',
           'review':'outputs/round74/results_review.json'}
    data={k:old.locked(ROOT/p) for k,p in paths.items()}
    r,d,p=data['results'],data['diagnostics'],data['paired']
    assert old.sha(ROOT/paths['results'])=='428f497383872490dfcc60051f1d25538ce69f845ec79f47c5017fcd50ddb717'
    assert r['checked_stages']==list(STAGES) and r['new_NN_fits']==3 and not r['new_submission_generated']
    assert r['status']=='three_periods_completed_no_adoption_decision' and data['execution']['exit_code']==0
    assert r['protected_before']==r['protected_after'] and len(r['protected_before'])==21
    assert all(old.sha(ROOT/n)==h for n,h in r['protected_before'].items())
    assert all(old.sha(ROOT/n)==h for n,h in r['source_sha256'].items())
    assert d['candidate']=='R74' and d['checked_stages']==list(STAGES) and not d['new_CSV']
    assert p['candidate']=='R74' and p['control']=='v19' and p['replicates']==2000 and p['seed']==20260925
    assert p['status']=='completed' and p['fits']==0 and p['model_predict_calls']==0 and not p['new_CSV']
    math_path=ROOT/'reports/round74_trainer_math_check.json'
    assert old.sha(math_path)=='10387b12d3c796ee202d21b410b17db119ccaa9546f6018cee14fec8010e31dd'
    math=json.loads(math_path.read_text(encoding='utf8'))
    assert math['exit_code']==0 and math['sources_unchanged'] and math['check']['model_fits']==math['check']['GPU_calls']==0
    paths['math_check']=old.rel(math_path)
    summaries={}
    for s in STAGES:
        a=r['stages'][s];diag=d['stages'][s]
        assert a['controls_full_metrics_exact'] and a['full_query_reload_exact'] and diag['all_full_saved_metrics_exact']
        assert a['preparation']['original_eight_arrays_bitwise_unchanged'] and not a['preparation']['query_labels_used']
        assert not a['preparation']['sample_weights_recomputed'] and not a['preparation']['sample_weights_renormalized']
        assert a['native_area_MAE_change_TWD']<0
        summaries[s]={'complete_MAE_TWD':a['metrics']['R74']['overall_mae'],'vs_v19':a['differences']['v19'],
            'native_area_MAE':a['native_MAE_diagnostic'],'native_area_delta_TWD':a['native_area_MAE_change_TWD'],
            'test_mix_MAE_TWD':diag['test_composition']['models']['R74'],
            'test_mix_delta_TWD':diag['test_composition']['models']['R74']-diag['test_composition']['models']['v19'],
            'price_band_deltas':[{'band':b['band'],'rows':b['rows'],'MAE_delta_TWD':b['R74']-b['v19'],
                                'contribution_to_overall_delta_TWD':(b['R74']-b['v19'])*b['rows']/a['query_rows']}
                               for b in diag['price_bands']],
            'bootstrap':{method:p['stages'][s][method]['summary'] for method in ('project','month')},
            'fit_seconds':a['fit_seconds'],'saved_weight_summary':a['preparation']['train_sample_weight_summary']}
    assert summaries['confirmation']['vs_v19']['overall']['delta_TWD']>0
    assert summaries['confirmation']['vs_v19']['strata']['new_nonmissing']['delta_TWD']>0
    assert summaries['confirmation']['test_mix_delta_TWD']>0
    selection_path=ROOT/'outputs/current_private_candidate.json'
    selection=json.loads(selection_path.read_text(encoding='utf8'))
    assert selection['private_priority']=='outputs/submission_v19_trial.csv'
    assert old.sha(ROOT/selection['private_priority'])==selection['sha256']
    assert selection['known_public_best_filename']=='submission_v18_trial.csv'
    assert selection['private_mae'] is None and not selection['final_version_chosen'] and not selection['kaggle_selection_changed']
    assert (ROOT/'reports/round74_report.md').is_file()
    decision={'round':74,'candidate':'R74 fit-date-weighted PLR area NN in fixed v19 pipeline',
        'decision':'withhold_adoption_keep_v19_preserve_area_weighting_direction',
        'reason':'Native area MAE improves in all three periods, but the complete newest-period overall, fixed new/seen weighting, test-composition, new-project, Q1 and Q4 metrics worsen. Historical gains warrant retaining the direction, not replacing v19 on the present evidence.',
        'proceed_final':False,'new_CSV':False,'historical_fits':3,'new_final_fits':0,
        'fit_seconds':sum(r['stages'][s]['fit_seconds'] for s in STAGES),
        'historical_main_seconds':r['elapsed_seconds'],'historical_execution_seconds':data['execution']['elapsed_seconds'],
        'private_priority':selection['private_priority'],'submission_sha256':selection['sha256'],
        'backup':selection['backup'],'known_public_best_filename':selection['known_public_best_filename'],
        'known_public_best_mae':selection['known_public_best_mae'],'private_mae':None,'public_mae':None,
        'scope':'local_recommendation_only','kaggle_selection_changed':False,'final_version_chosen':False,
        'GitHub_publication_deferred':True,'seed_or_weight_search':False,'individual_ID_rules':False,
        'Public_feedback_used_for_tuning':False,'fresh_blind_validation':False,
        'primary_comparison':'R74 complete minus v19; negative improves MAE',
        'native_comparison':'R74 native area minus R54 native area; not interchangeable with complete-pipeline effect',
        'summary':summaries,'evidence':{k:{'path':path,'sha256':old.sha(ROOT/path)} for k,path in paths.items()},
        'report':'reports/round74_report.md','report_is_editable_explanation':True,'source_sha256':old.sha(Path(__file__)),
        'limitations':[
            'Fixed seed42 only; same settings do not prove deterministic CUDA retraining or stability across initializations.',
            'Newest-period project and month intervals cross zero: the small regression is not proof of systematic harm or a Private forecast.',
            'Auxiliary project and month intervals are negative; development project intervals cross zero while month intervals are negative.',
            'All historical windows have been reused extensively; bootstrap does not remove adaptive-selection or future-shift uncertainty.',
            'Weighted-sample ESS describes concentration, not a count of independent transactions or projects.',
            'The >=100m true-price band is an evaluation diagnostic only; it is not an available test feature or prediction rule.',
            'No final fit, test prediction, submission CSV or Private score was produced for R74; final version remains undecided.',
        ],'future_direction_retained':True,'next_formal_experiment_selected':False,
        'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    old.write(ROOT/'outputs/round74/decision.json',decision)
    keys=('round','candidate','decision','reason','report','report_is_editable_explanation','proceed_final','new_CSV',
          'historical_fits','new_final_fits','fit_seconds','historical_main_seconds','historical_execution_seconds',
          'private_mae','public_mae','kaggle_selection_changed','final_version_chosen','GitHub_publication_deferred',
          'limitations','future_direction_retained','next_formal_experiment_selected','completed_at_utc')
    selection['latest_completed_research']={k:decision[k] for k in keys}
    selection['latest_completed_research'].update(decision_record='outputs/round74/decision.json',
          decision_sha256=old.sha(ROOT/'outputs/round74/decision.json'),evidence=decision['evidence'])
    selection['pending_research']={'round':74,'status':'completed_not_adopted_no_export','kind':'fixed_fit_date_weighted_area_NN',
          'planned_new_NN_fits':3,'actual_new_NN_fits':3,'historical_evaluation_started':True,
          'source_review_passed':True,'source_review_scope':'Root trainer diff and driver review; independent source-only trainer and driver reviews; fixed provenance checks reviewed by root',
          'plan':'reports/round74_plan.md','plan_sha256':r['plan_sha256'],
          'source_sha256':r['source_sha256']['src/round74_nn_experiment.py'],'completed_at_utc':decision['completed_at_utc']}
    selection['next_research_preparation']={'round':None,'status':'not_selected_after_area_weighting_evaluation',
          'formal_experiment_selected':False,'implementation_started':False,'historical_evaluation_started':False,
          'seed_or_weight_search':False,'Public_feedback_changed_design':False,
          'retained_direction':'Area date weighting; stability or error-transfer analysis has not yet been selected'}
    selection['prior_research_batches']=selection.get('prior_research_batches',[])+[selection['latest_research_batch']]
    selection['latest_research_batch']={'rounds':[74],'historical_NN_fits':3,'final_NN_fits':0,'fit_seconds':decision['fit_seconds'],'new_CSV':None}
    selection['latest_unreleased_alternative']='R74 fixed area recency-weighted MAE: native gains in all periods, mixed complete-pipeline evidence; no final CSV'
    for name in [*paths.values(),'reports/round74_plan.md','reports/round74_report.md','outputs/round74/decision.json',
                 'src/round74_nn_experiment.py','src/round74_recency_weighted_area_train.py']:
        assert (ROOT/name).is_file(),name
        if name not in selection['evidence']:selection['evidence'].append(name)
    selection['updated_at_utc']=decision['completed_at_utc']
    selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'decision':decision['decision'],'decision_sha256':old.sha(ROOT/'outputs/round74/decision.json'),
                      'fit_seconds':decision['fit_seconds'],'recommendation':selection['private_priority']}))


if __name__=='__main__':run()
