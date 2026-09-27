"""Record the bounded R76/R77 candidate-preparation decision from saved JSON only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('development', 'confirmation', 'auxiliary')
OUT = ROOT / 'outputs/round77/decision.json'
PINS = {
    'outputs/round77/results.json': '6e9b837a9bfe5918cb5200831ae31268d3ad204aec21430da1b0577de2be8337',
    'outputs/round77/results_review.json': '90aa860b08de99751dbbe0d00550ab0bbfcf3c40cba2ee21662bbaad914fdf68',
    'reports/round77_diagnostics.json': '23de0f9dc5eccd921ef5b776771db6317232d21031a5a0b78809b1fa8104fbd7',
    'outputs/round77/paired_vs_R76/results.json': '6621e83a91a10ff5cca1ff1a2bd5f4596102b510850b00707d833925e253b6ae',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    tick = time.monotonic()
    assert not OUT.exists() and not OUT.with_suffix('.sha256').exists(), 'No overwrite'
    inputs, sources = {}, {}

    def capture(name, expected=None):
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json', '.sha256', '.py', '.md')
        assert p.stat().st_size < 30_000_000, 'Small text/JSON only'
        h = sha(p)
        assert expected is None or h == expected, name
        assert name not in inputs or inputs[name] == h
        inputs[name] = h
        return p

    def locked(name):
        p = capture(name, PINS.get(name))
        receipt = p.with_suffix('.sha256').relative_to(ROOT).as_posix()
        assert capture(receipt).read_text(encoding='utf8').strip() == sha(p)
        return json.loads(p.read_text(encoding='utf8'))

    r, review, diag, paired = [locked(n) for n in PINS]
    execution = locked('outputs/round77/execution.json')
    diag_execution = locked('outputs/round77/diagnostics/execution.json')
    r76_decision = locked('outputs/round76/decision.json')
    assert execution['exit_code'] == diag_execution['exit_code'] == 0
    assert execution['failure'] is None and diag_execution['failure'] is None
    assert r['round'] == 77 and r['candidate'] == 'R77' and r['primary_control'] == 'R76'
    assert r['status'] == 'three_periods_completed_no_adoption_decision'
    assert r['checked_stages'] == review['checked_stages'] == diag['checked_stages'] == list(STAGES)
    assert review['status'] == 'completed_JSON_only_review' and not review['is_decision_authority']
    assert diag['candidate'] == 'R77' and diag['status'] == 'completed_saved_predictions_only'
    assert paired['candidate'] == 'R77' and paired['control'] == 'R76' and paired['status'] == 'completed'
    assert paired['seed'] == 20260925 and paired['replicates'] == 2000
    for key in ('new_fits', 'new_NN_fits', 'model_loads', 'model_predict_calls', 'reference_predict_calls', 'spatial_predict_calls'):
        assert r[key] == 0, key
    assert not r['new_CSV'] and not r['new_submission_generated']
    assert not r['Public_feedback_used_for_tuning'] and not r['seed_weight_threshold_search']
    assert diag['model_fit_load_predict_calls'] == paired['fits'] == paired['model_predict_calls'] == 0
    assert not diag['new_CSV'] and not paired['new_CSV']
    assert r['protected_before'] == r['protected_after'] and len(r['protected_before']) == 21
    for mapping in (r['source_sha256'], diag['source_sha256']):
        for name, expected in mapping.items():
            assert name.endswith('.py')
            capture(name, expected)
            sources[name] = expected
    for name, expected in (
        ('src/round77_results_review.py', review['source_sha256']),
        ('src/paired_candidate_bootstrap.py', paired['source_sha256']),
    ):
        capture(name, expected)
        sources[name] = expected
    assert execution['source_sha256'] == sources['src/round77_area_unit_combination.py']
    assert diag_execution['source_sha256'] == sources['src/candidate_diagnostics_v3.py']
    capture('reports/round77_plan.md', r['text_plan_sha256'])
    selector_name = 'outputs/current_private_candidate.json'
    selector_path = capture(selector_name, r['selector_sha256_before'])
    selector_sha = inputs[selector_name]
    assert selector_sha == r['selector_sha256_after']
    selection = json.loads(selector_path.read_text(encoding='utf8'))
    assert selection['private_priority'] == 'outputs/submission_v19_trial.csv'
    assert selection['private_mae'] is None and not selection['final_version_chosen']
    summaries = {}
    for s in STAGES:
        a, rv, dg, b = r['stages'][s], review['stages'][s], diag['stages'][s], paired['stages'][s]
        assert a['controls_full_metrics_exact'] and a['three_saved_cells_bitwise_exact']
        assert dg['all_full_saved_metrics_exact'] and b['point_metrics_reproduced']
        assert a['query_rows'] == rv['query_rows'] == dg['query_rows'] == b['rows']
        assert a['metrics'] == rv['metrics'] == dg['metrics']
        for metric in ('overall', 'weighted'):
            assert math.isclose(a['differences']['R76'][metric]['delta_TWD'],
                                rv['differences']['R76'][metric]['delta_TWD'], rel_tol=1e-11, abs_tol=1e-5)
        summaries[s] = {
            'four_cell_metrics': {k:a['metrics'][k] for k in ('v19','R74','R76','R77')},
            'R77_minus_R76': rv['differences']['R76'], 'R77_minus_v19': rv['differences']['v19'],
            'test_composition': rv['test_composition'],
            'test_composition_differences': rv['test_composition_differences'],
            'price_bands': rv['price_bands'], 'paired_R77_minus_R76': rv['paired'],
            'interaction_TWD': rv['interaction_TWD'], 'interaction_is_causal': False,
        }
    cf = summaries['confirmation']
    assert cf['R77_minus_R76']['overall']['delta_TWD'] > 0
    assert cf['test_composition_differences']['R76']['delta_TWD'] > 0
    assert cf['R77_minus_R76']['strata']['new_nonmissing']['delta_TWD'] > 0
    assert all(cf['paired_R77_minus_R76'][m][k]['percentile_2_5'] < 0 < cf['paired_R77_minus_R76'][m][k]['percentile_97_5']
               for m in ('project','month') for k in ('overall','weighted'))
    assert all(summaries[s]['R77_minus_R76']['overall']['delta_TWD'] < 0 for s in ('development','auxiliary'))
    tail = next(v for v in cf['price_bands'] if v['band'] == '>=100m')
    assert tail['differences']['R76']['delta_TWD'] > 0
    record = {
        'round':77, 'decision':'advance_two_separately_identified_final_candidates_R76_priority_R77_alternative',
        'scope':'Prepare separately identified submittable candidates after a new final plan; not a final-version choice or release.',
        'candidate_preparation_order':['R76','R77'], 'candidate_preparation_priority':'R76', 'candidate_preparation_alternative':'R77',
        'reason':'R76 has the lower newest confirmation overall and test-composition MAE. R77 improves development and auxiliary MAE, but worsens confirmation new-project and high-price metrics; the four confirmation paired intervals cross zero. Preserve both fixed methods as a documented tradeoff, with R76 preparation priority.',
        'summary':summaries,
        'key_differences_TWD':{
            'R77_minus_R76_confirmation_overall':cf['R77_minus_R76']['overall']['delta_TWD'],
            'R77_minus_R76_confirmation_test_mix':cf['test_composition_differences']['R76']['delta_TWD'],
            'R77_minus_R76_development_overall':summaries['development']['R77_minus_R76']['overall']['delta_TWD'],
            'R77_minus_R76_auxiliary_overall':summaries['auxiliary']['R77_minus_R76']['overall']['delta_TWD'],
            'R77_minus_R76_confirmation_new':cf['R77_minus_R76']['strata']['new_nonmissing']['delta_TWD'],
            'R77_minus_R76_confirmation_at_least_100m':tail['differences']['R76']['delta_TWD'],
        },
        'limitations':[
            'R76 newest-period gains versus v19 are concentrated in high prices: about 78% of the overall gain comes from >=50m bands (2044 rows); lower three bands also improve.',
            'R77 newest-period >=100m MAE is higher than R76 by about 733056 TWD, and new-project MAE by about 5202 TWD; Q1/Q4 and test-composition metrics also worsen.',
            'R77 improves the two earlier periods versus R76; this benefit is retained rather than dismissed.',
            'All four newest-period paired intervals cross zero; this does not establish equivalence or a Private win probability.',
            'Single fixed seed and extensively reused windows; conditional bootstrap does not remove adaptive-selection or temporal-shift risk.',
            'True-price bands are diagnostic only, not test-time routing or per-ID correction rules.',
            'Unit-branch standalone quality does not determine the quality of the final median; four-cell interaction is descriptive, not causal.',
            'Public/Private for these unreleased candidates remain unknown; no Public-based tuning or guarantee.',
        ],
        'R76_high_price_concentration_evidence':r76_decision['summary']['confirmation']['price_band_contributions_vs_v19_TWD'],
        'required_separate_final_plan':'reports/round76_77_final_plan.md',
        'separate_final_plan_required_before_training':True, 'proceed_final':False,
        'proceed_final_semantics':'This recorder does not launch or authorize a fit; final execution requires the separate final plan.',
        'new_fits':0,'new_final_fits':0,'model_or_array_loads':0,'new_scores':0,'new_bootstrap':False,'new_CSV':False,
        'final_version_chosen':False,'private_mae':None,'public_mae':None,
        'private_priority':selection['private_priority'],'submission_sha256':selection['sha256'],
        'known_public_best_filename':selection['known_public_best_filename'],'known_public_best_mae':selection['known_public_best_mae'],
        'selector_updated':False,'selector_sha256_before':selector_sha,'selector_sha256_after':selector_sha,
        'kaggle_selection_changed':False,'GitHub_publication_deferred':True,'human_documents_updated':False,
        'Public_feedback_used_for_tuning':False,'fresh_blind_validation':False,
        'evaluation_protected_files_declared_equal':21,'binary_files_independently_rehashed_by_recorder':False,
        'main_seconds':r['elapsed_seconds'],'execution_seconds':execution['elapsed_seconds'],
        'evidence_sha256':inputs,'verified_source_sha256':sources,'recorder_source_sha256':sha(Path(__file__)),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick,
    }
    assert all(sha(ROOT / name) == h for name,h in inputs.items()), 'Evidence changed'
    raw = (json.dumps(record,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    with OUT.open('xb') as f:f.write(raw)
    with OUT.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(hashlib.sha256(raw).hexdigest()+'\n')
    assert sha(selector_path) == selector_sha
    print(json.dumps({'decision':record['decision'],'decision_sha256':sha(OUT),'elapsed_seconds':record['elapsed_seconds'],
                      'selector_updated':False,'new_fits':0,'new_CSV':False},ensure_ascii=False))


if __name__ == '__main__':
    run()
