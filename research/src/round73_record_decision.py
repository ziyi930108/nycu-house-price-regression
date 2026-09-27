"""Record the fixed diagnostic-only R73 outcome; preserve the v19 recommendation."""
from pathlib import Path
from datetime import datetime, timezone
import json
import round48_nn_experiment as old

ROOT = old.ROOT
STAGES = ('development', 'confirmation', 'auxiliary')


def run():
    paths = {
        'results': 'outputs/round73/results.json',
        'execution': 'outputs/round73/execution.json',
        'review': 'outputs/round73/results_review.json',
        'paired': 'outputs/round72/paired_vs_null145/results.json',
        'zero_value_mechanism': 'outputs/round72/mechanism_review.json',
    }
    data = {k: old.locked(ROOT/p) for k, p in paths.items()}
    r, pair = data['results'], data['paired']
    assert old.sha(ROOT/paths['results']) == '50d8ee7cdcbc3eca9f76818cb7462061445c9f57f1877a6fc408624b2fa95a67'
    assert r['checked_stages'] == list(STAGES) and r['new_NN_fits'] == 3
    assert r['status'] == 'three_periods_completed_diagnostic_only'
    assert r['diagnostic_only'] and not r['eligible_for_export'] and not r['new_submission_generated']
    assert data['execution']['exit_code'] == 0
    assert r['protected_before'] == r['protected_after']
    assert all(old.sha(ROOT/p) == h for p, h in r['protected_before'].items())
    assert all(old.sha(ROOT/p) == h for p, h in r['source_sha256'].items())
    assert pair['candidate'] == 'R72' and pair['control'] == 'null145' and pair['replicates'] == 2000
    assert pair['status'] == 'completed' and pair['fits'] == 0 and pair['model_predict_calls'] == 0
    assert data['zero_value_mechanism']['execution']['exit_code'] == 0
    summaries = {}
    for stage in STAGES:
        a = r['stages'][stage]
        assert a['controls_full_metrics_exact'] and a['full_query_reload_exact']
        assert a['runtime_Torch_numpy_CUDA_GPU_exact_to_R72'] and a['configuration_factor_weights_bias_exact_to_R72']
        assert a['preparation']['original_144_prefix_bitwise_same_as_R72_and_R63']
        assert a['preparation']['other_seven_arrays_bitwise_same_as_R72_and_R63']
        assert not a['preparation']['query_labels_used']
        summaries[stage] = {
            'R72_minus_null145': a['R72_minus_null145'],
            'R72_minus_null145_native_MAE_TWD': a['R72_minus_null145_native_MAE_TWD'],
            'null145_minus_v19_background_only': a['differences']['v19'],
            'native_MAE': a['native_MAE_diagnostic'],
            'bootstrap': {m: pair['stages'][stage][m]['summary'] for m in ('project', 'month')},
        }
    selection_path = ROOT/'outputs/current_private_candidate.json'
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    assert selection['private_priority'] == 'outputs/submission_v19_trial.csv'
    assert old.sha(ROOT/selection['private_priority']) == selection['sha256']
    assert selection['known_public_best_filename'] == 'submission_v18_trial.csv'
    decision = {
        'round': 73, 'candidate': 'R73 null145 diagnostic control',
        'decision': 'diagnostic_completed_no_export_keep_v19',
        'reason': 'The road delta has mixed same-width effects: native improves only in the newest period; complete MAE changes are small, all project-bootstrap intervals cross zero, and Q3 worsens in every period. Do not promote the null control or add a hit flag from subgroup differences.',
        'diagnostic_only': True, 'eligible_for_export': False, 'new_CSV': False, 'proceed_final': False,
        'historical_fits': 3, 'new_final_fits': 0, 'fit_seconds': sum(r['stages'][s]['fit_seconds'] for s in STAGES),
        'historical_main_seconds': r['elapsed_seconds'], 'historical_execution_seconds': data['execution']['elapsed_seconds'],
        'private_priority': selection['private_priority'], 'submission_sha256': selection['sha256'],
        'backup': selection['backup'], 'known_public_best_filename': selection['known_public_best_filename'],
        'known_public_best_mae': selection['known_public_best_mae'], 'private_mae': None, 'public_mae': None,
        'scope': 'local_recommendation_only', 'kaggle_selection_changed': False, 'final_version_chosen': False,
        'GitHub_publication_deferred': True, 'seed_or_weight_search': False, 'individual_ID_rules': False,
        'Public_feedback_used_for_tuning': False, 'fresh_blind_validation': False,
        'primary_comparison': 'R72 minus R73 null145; negative means road delta improves MAE',
        'summary': summaries,
        'evidence': {k: {'path': p, 'sha256': old.sha(ROOT/p)} for k, p in paths.items()},
        'report': 'reports/round73_report.md', 'report_is_editable_explanation': True,
        'source_sha256': old.sha(Path(__file__)),
        'limitations': [
            'Fixed seed42 only; source implies equal initialization/draw shapes but original R72 initialization and RNG traces were not saved.',
            'CUDA retraining is not claimed bitwise deterministic; zero-column 145-input model is not equivalent to the 144-input v19 branch.',
            'The arithmetic decomposition of MAE differences is not a causal decomposition.',
            'All project-bootstrap intervals cross zero; month intervals differ by period and metric, so not all intervals cross zero.',
            'Repeated historical windows and posthoc mechanism groups do not remove adaptive-selection or future-shift uncertainty.',
            'This experiment does not prove all road features useless, and no hit-indicator model was trained.',
        ],
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    old.write(ROOT/'outputs/round73/decision.json', decision)
    keys = ('round','candidate','decision','reason','report','report_is_editable_explanation','diagnostic_only',
            'eligible_for_export','new_CSV','proceed_final','historical_fits','new_final_fits','fit_seconds',
            'historical_main_seconds','historical_execution_seconds','private_mae','public_mae',
            'kaggle_selection_changed','final_version_chosen','GitHub_publication_deferred','limitations','completed_at_utc')
    selection['latest_completed_research'] = {k: decision[k] for k in keys}
    selection['latest_completed_research'].update(decision_record='outputs/round73/decision.json',
        decision_sha256=old.sha(ROOT/'outputs/round73/decision.json'), evidence=decision['evidence'])
    selection['pending_research'].update(status='completed_diagnostic_only_no_export', actual_new_NN_fits=3,
        source_review_passed=True, source_review_scope='root and independent source review passed',
        completed_at_utc=decision['completed_at_utc'])
    selection['pending_research'].pop('live_exec_session_id', None)
    selection['next_research_preparation'] = {'round': None, 'status': 'not_selected_after_same_width_diagnostic',
        'formal_experiment_selected': False, 'implementation_started': False, 'historical_evaluation_started': False,
        'seed_or_weight_search': False, 'Public_feedback_changed_design': False}
    selection['prior_research_batches'] = selection.get('prior_research_batches', []) + [selection['latest_research_batch']]
    selection['latest_research_batch'] = {'rounds': [73], 'historical_NN_fits': 3, 'final_NN_fits': 0,
        'fit_seconds': decision['fit_seconds'], 'new_CSV': None, 'diagnostic_only': True}
    selection['latest_diagnostic_control'] = {'round': 73, 'description': 'same-width null145, excluded from export by fixed plan',
        'report': decision['report'], 'eligible_for_export': False}
    for name in [*paths.values(), 'reports/round73_report.md', 'reports/round73_plan.md', 'outputs/round73/decision.json']:
        assert (ROOT/name).is_file(), name
        if name not in selection['evidence']: selection['evidence'].append(name)
    selection['updated_at_utc'] = decision['completed_at_utc']
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'decision': decision['decision'], 'decision_sha256': old.sha(ROOT/'outputs/round73/decision.json'),
                      'fit_seconds': decision['fit_seconds'], 'recommendation': selection['private_priority']}))


if __name__ == '__main__':
    run()
