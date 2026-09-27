"""Record the R83 fixed complementary candidate decision; no training or release."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round83/decision.json'
SELECTOR_SHA = 'c993a861fa6766fb5c9da8c156521bb28886a459587653ee8734b2927b9e87a8'
PINS = {
    'outputs/round83/plan.json': '961792a507127c7e3a35c4637c386cbc353f36d8170c140db09518b49c4a5989',
    'outputs/round83/results.json': '8db07138c15ef2725d848c8b0ca65a6095bcc183b8550611377b3e165c59583b',
    'outputs/round83/execution.json': 'f840db3a37bf4c50b29366887cb7184e415b11cf3b5338a7b81f9358240db7ca',
    'reports/round83_diagnostics.json': '1001cb40a1d2e72a0d2268a6237f801e4255affe4e7e9241e86902eae2689b10',
    'outputs/round83/paired_vs_v23/results.json': '9a400bbcb8ed3eb1e4a0ac01ec8404b160abbc5ff48b466c365b3886b69d8655',
    'outputs/round83/results_review.json': '651ab12658e46cbb483a0725e6dceed28ce9dc2b33dc8d131f8a89da02d5ce2a',
    'reports/round83_report.md': '9afb18d38665c3ad3f041bab7b002beac1e52ea287413554c42184cba19f8316',
    'reports/round83_plan.md': 'c0cf070337ca86ab742479b664628413155f1e69745c7a80ae1f15e152c747c0',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run():
    assert not OUT.exists() and not OUT.with_suffix('.sha256').exists()
    observed = {}
    def read(name, expected=None):
        p = ROOT / name
        value = sha(p)
        assert value == (expected or PINS[name]), name
        assert p.with_suffix('.sha256').read_text(encoding='ascii').strip() == value
        observed[name] = value
        return json.loads(p.read_text(encoding='utf8'))
    for name, digest in PINS.items():
        assert sha(ROOT / name) == digest, name
        observed[name] = digest
    main = read('outputs/round83/results.json')
    plan = read('outputs/round83/plan.json')
    review = read('outputs/round83/results_review.json')
    diag = read('reports/round83_diagnostics.json')
    paired = read('outputs/round83/paired_vs_v23/results.json')
    for key, name in (('main', 'outputs/round83/execution.json'),
                      ('diagnostics', 'outputs/round83/diagnostics/execution.json'),
                      ('paired', 'outputs/round83/paired_vs_v23_execution.json')):
        e = read(name, review['input_sha256'][name])
        assert e == review['executions'][key] and e['exit_code'] == 0 and e.get('failure') is None
    stages = ['development', 'confirmation', 'auxiliary']
    assert main['status'] == 'three_periods_completed_no_adoption_decision'
    assert main['checked_stages'] == diag['checked_stages'] == review['checked_stages'] == stages
    assert main['new_fits'] == 3 and not main['new_CSV'] and not main['new_submission_generated']
    assert review['report_sha256'] == PINS['reports/round83_report.md']
    assert paired['candidate'] == 'R83' and paired['control'] == 'v23' and paired['replicates'] == 2000
    assert main['source_sha256'] == plan['source_sha256'] and main['plan_sha256'] == PINS['outputs/round83/plan.json']
    protected = plan['protected_before']
    assert main['protected_before'] == main['protected_after'] == protected and len(protected) == 25
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs').glob('submission_v*.csv')} == set(protected) - {'runs/versions.json'}
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    assert sha(ROOT / 'outputs/current_private_candidate.json') == SELECTOR_SHA
    selector = json.loads((ROOT / 'outputs/current_private_candidate.json').read_text(encoding='utf8'))
    assert selector['private_priority'] == 'outputs/submission_v22_trial.csv'
    assert selector['backup'] == 'outputs/submission_v20_trial.csv'
    assert selector['complementary_candidate'] == 'outputs/submission_v23_trial.csv'
    assert not (ROOT / 'outputs/submission_v24_trial.csv').exists()
    decision = dict(round=83, candidate='R83',
        status='advance_one_fixed_complementary_candidate_not_yet_released',
        proposed_filename='outputs/submission_v24_trial.csv',
        priority_retained=selector['private_priority'], backup_retained=selector['backup'],
        intended_complementary_replacement_of=selector['complementary_candidate'],
        complementary_role_changes_only_after_successful_release=True,
        final_plan_path='reports/round83_final_plan.md',
        reason='Development and oldest-period overall/weighted improvements versus v23 have eight negative paired intervals; newest overall is near-flat with mixed groups. Fixed seed averaging has useful complementary trial value without choosing a winning seed.',
        priority_retention_reason='Versus v22, development test-composition estimate worsens by 9367 TWD and newest >=100m per-row MAE worsens by about 909728 TWD; there is no paired v22 comparison in this fixed protocol. Retain v22 as a limited local priority, not a proved Private winner.',
        strongest_counterargument='R83 overall/weighted and new-project point metrics improve versus v22 in all three periods, including newest overall -10032 and new-project -16909 TWD. It may ultimately be a better priority; retaining v22 does not dismiss these gains.',
        latest_period_caution='Versus v23, newest overall -101, weighted +133, testmix +212 TWD and all four paired intervals cross zero. >=50m contributes -1189.61 TWD while other bands contribute +1088.22 TWD. This is near-flat, not broad newest-period improvement.',
        stage_differences={s:{k:main['stages'][s]['differences'][k] for k in ('v23','v22')} for s in stages},
        historical_new_fits=3, historical_fit_seconds=review['total_fit_seconds'],
        input_sha256=observed, source_sha256={Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))},
        selector_sha256_before=SELECTOR_SHA, protected_before=protected,
        new_fits_allowed=1, final_fit_seed=137, final_fit_helper='src/round83_seed_unit.py',
        reuse_seed42_NN_log=True, formula=main['formula'], new_hyperparameter_search=False,
        limitations=['Repeated historical windows are not fresh blind validation.',
            'Paired intervals against v23 cannot be applied to v22 or read as Private probabilities.',
            'Two seeds do not establish the distribution of training randomness.',
            'No all-period-wins gate, group routing, clipping change, Public tuning or transaction-answer lookup.',
            'The report has not tested changing the common scale of training weights; no conclusion on that effect is required for this decision.'],
        new_CSV=False, selector_updated=False, uploaded=False, GitHub_published=False,
        final_version_chosen=False, public_mae=None, private_mae=None,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    raw = (json.dumps(decision,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    with OUT.open('xb') as f: f.write(raw)
    with OUT.with_suffix('.sha256').open('x',encoding='ascii') as f: f.write(hashlib.sha256(raw).hexdigest()+'\n')
    print(json.dumps(dict(status=decision['status'],decision_sha256=sha(OUT),new_CSV=False,selector_updated=False)))


if __name__ == '__main__':
    run()
