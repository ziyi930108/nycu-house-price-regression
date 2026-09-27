"""Record R75 diagnostic interpretation without altering the recommendation file."""
from pathlib import Path
import json
import round48_nn_experiment as old

ROOT=old.ROOT
STAGES=('development','confirmation','auxiliary')


def run():
    paths={'results':'outputs/round75/results.json','execution':'outputs/round75/execution.json',
           'review':'outputs/round75/results_review.json','plan':'outputs/round75/plan.json',
           'math_check':'reports/round75_projection_selfcheck.json'}
    data={k:old.locked(ROOT/p) for k,p in paths.items()}
    r=data['results']
    assert old.sha(ROOT/paths['results'])=='fd9c7eb4127bbe5a0a6a5ed52f20f56262cdf5ed394f36cdfd9ec0899456726e'
    assert r['status']=='completed_saved_vector_error_transfer_diagnostic' and r['checked_stages']==list(STAGES)
    assert data['execution']['exit_code']==0 and data['execution']['failure'] is None
    assert all(r[k]==0 for k in ('new_fits','model_loads','model_predict_calls','reference_predict_calls','spatial_predict_calls'))
    assert not r['new_CSV'] and not r['new_candidate_created'] and r['diagnostic_only']
    assert r['protected_before']==r['protected_after'] and len(r['protected_before'])==21
    assert all(old.sha(ROOT/n)==h for n,h in r['protected_before'].items())
    assert all(old.sha(ROOT/n)==h for n,h in r['source_sha256'].items())
    selector=ROOT/'outputs/current_private_candidate.json'
    before=old.sha(selector)
    assert before==r['selector_sha256_before']==r['selector_sha256_after']
    selection=json.loads(selector.read_text(encoding='utf8'))
    assert selection['private_priority']=='outputs/submission_v19_trial.csv'
    assert old.sha(ROOT/selection['private_priority'])==selection['sha256']
    assert selection['known_public_best_filename']=='submission_v18_trial.csv'
    assert not selection['kaggle_selection_changed'] and not selection['final_version_chosen'] and selection['private_mae'] is None
    summary={}
    for s in STAGES:
        a=r['stages'][s];projected=a['full']['projection'];overall=projected['overall']
        assert a['native_overall_and_complete_full_saved_metrics_exact'] and a['all_cohort_rows_and_error_contributions_reconciled']
        assert a['same_fixed_bounds_median_clip_identity_exact'] and a['native_mean_and_base_identity_exact']
        assert len(projected['transition_cells'])==10 and len(projected['error_sign_groups'])==3
        assert sum(x['rows'] for x in projected['transition_cells'].values())==a['query_rows']
        summary[s]={'rows':a['query_rows'],
             'six_layer_paired_MAE_changes_TWD':{k:a['layer_metrics'][k]['differences']['overall']['delta_TWD'] for k in r['layers']},
             'spatial_to_final_paired_delta_change_TWD':overall['transfer_effect_TWD'],
             'spatial_changed_rows':overall['predshift']['spatial']['changed_rows'],
             'final_changed_rows':overall['predshift']['final']['changed_rows'],
             'final_unchanged_rows':overall['predshift']['final']['unchanged_rows'],
             'transition_cells':projected['transition_cells'],'spatial_error_sign_groups':projected['error_sign_groups']}
    c=summary['confirmation']
    assert c['six_layer_paired_MAE_changes_TWD']['nn_spatial']<0<c['six_layer_paired_MAE_changes_TWD']['predictions']
    assert summary['auxiliary']['spatial_to_final_paired_delta_change_TWD']<0
    report='reports/round75_report.md';assert (ROOT/report).is_file()
    decision={'round':75,'kind':'saved_vector_error_transfer_diagnostic','decision':'diagnostic_complete_keep_v19_no_new_candidate',
        'finding':'Latest-period R74 improvements survive through the corrected NN branch but reverse at the fixed outer median. This does not justify removing the median: its effect differs across periods and it suppresses both gains and harms.',
        'diagnostic_only':True,'new_fits':0,'new_model_predictions':0,'new_candidate_created':False,'new_CSV':False,
        'main_seconds':r['elapsed_seconds'],'execution_seconds':data['execution']['elapsed_seconds'],
        'summary':summary,'primary_comparison':'R74 minus R71 seed42 at each saved layer; native area uses R54 control',
        'private_priority':selection['private_priority'],'submission_sha256':selection['sha256'],
        'backup':selection['backup'],'known_public_best_filename':selection['known_public_best_filename'],
        'known_public_best_mae':selection['known_public_best_mae'],'private_mae':None,
        'selector_updated':False,'selector_sha256_before':before,'kaggle_selection_changed':False,'final_version_chosen':False,
        'GitHub_publication_deferred':True,'next_formal_experiment_selected':False,
        'scope':'local diagnostic; no acceptance rule or model selection change',
        'evidence':{k:{'path':p,'sha256':old.sha(ROOT/p)} for k,p in paths.items()},
        'report':report,'report_is_editable_explanation':True,'source_sha256':old.sha(Path(__file__)),
        'limitations':[
            'This is a descriptive analysis of a single seed on repeatedly reused historical windows, not a fresh holdout.',
            'The clamp formula is exactly the existing median; no new cap or truth-based inference rule was introduced.',
            'A contraction of prediction changes does not guarantee improvement or a contraction of paired MAE gains.',
            'Stage-to-stage MAE differences are not isolated causal component effects.',
            'Price and paired-error-sign groups use validation truth only for diagnosis and must not be used as test prediction gates.',
            'Intermediate-layer prediction quality alone does not justify promoting that layer as a new submission.',
            'Private remains unknown; candidate, weight, threshold and final-version searches were not performed in this diagnostic.',
        ],'completed_at_utc':old.now()}
    assert old.sha(selector)==before
    decision['selector_sha256_after']=old.sha(selector)
    old.write(ROOT/'outputs/round75/decision.json',decision)
    print(json.dumps({'decision':decision['decision'],'decision_sha256':old.sha(ROOT/'outputs/round75/decision.json'),
                      'new_fits':0,'new_CSV':False,'selector_updated':False,'recommendation':selection['private_priority']}))


if __name__=='__main__':run()
