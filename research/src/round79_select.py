"""Update only the local recommendation after verified R79/v22 CSV release; no model or CSV writes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, os, time, traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round79/selection'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
OLD_SHA = 'a488fa83d3a3581bc939d414897314d7caa443a803ca12096644b78d5245af56'
PLAN = 'reports/round79_final_plan.md'
PLAN_SHA = 'cd3dd3b71fc24e405c8ade92ee89bc928e8ad76a6feca258c79e66e65455ae2d'
RESEARCH = 'outputs/round79/decision.json'
RESEARCH_SHA = '12c5c188057ef36704f6e0553d1c9cfbd7d3d345dbbcac083275aa8fe9405ef5'
RELEASE = 'outputs/round79/final_export/release.json'
RELEASE_SOURCE = 'src/round79_release.py'
RELEASE_SOURCE_SHA = 'cd4b975430520c8297138353239d904b729cee5ec006fb0f1ba6d2a1d658b521'
PRIORITY = 'outputs/submission_v22_trial.csv'
BACKUP = 'outputs/submission_v20_trial.csv'
EARLIER = 'outputs/submission_v21_trial.csv'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def locked(name):
    path = ROOT / name
    assert sha(path) == path.with_suffix('.sha256').read_text(encoding='ascii').strip(), name
    return json.loads(path.read_text(encoding='utf8'))

def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(path) + '\n')

def protected_hashes(expected):
    actual = {name: sha(ROOT / name) for name in expected}
    assert actual == expected, 'A previous CSV or registry changed'
    return actual

def run():
    assert sha(SELECTOR) == OLD_SHA
    assert not SELECTOR.with_suffix('.sha256').exists(), 'Unexpected receipt for mutable selector'
    assert sha(ROOT / PLAN) == PLAN_SHA and sha(ROOT / RESEARCH) == RESEARCH_SHA
    research = locked(RESEARCH); release = locked(RELEASE)
    assert research['round'] == 79 and research['status'] == 'advance_to_one_fixed_final_candidate_not_yet_released'
    assert research['proposed_filename'] == PRIORITY and research['selector_sha256_before'] == OLD_SHA
    assert release['round'] == 79 and release['version'] == 22 and release['rows'] == 47073
    assert release['status'] == 'experimental_v22_trial_generated' and release['path'] == PRIORITY
    assert release['plan_path'] == PLAN and release['plan_sha256'] == PLAN_SHA
    assert release['decision_path'] == RESEARCH and release['decision_sha256'] == RESEARCH_SHA
    assert sha(ROOT / RELEASE_SOURCE) == RELEASE_SOURCE_SHA == release['source_sha256'][RELEASE_SOURCE]
    execution = locked('outputs/round79/final_export/execution.json')
    assert execution['exit_code'] == 0 and execution['source_sha256'] == RELEASE_SOURCE_SHA
    for key in ('positive_int64_full_roundtrip', 'original_unrounded_prediction_comparison_exact',
                'UTF8_BOM', 'all_sample_IDs_and_order_exact', 'exact_headers', 'old_CSV_unchanged',
                'composer_and_model_declared_full_query_raw_TWD_reload_exact'):
        assert release[key] is True, key
    for key in ('selector_updated', 'registry_updated', 'uploaded', 'kaggle_selection_changed', 'final_version_chosen'):
        assert release[key] is False, key
    assert release['public_mae'] is None and release['private_mae'] is None
    assert release['selector_sha256_before'] == release['selector_sha256_after'] == OLD_SHA
    assert release['model_fit_load_predict_calls_in_release'] == 0
    proof_name = release['candidate_proof_path']; proof = locked(proof_name)
    assert sha(ROOT / proof_name) == release['candidate_proof_sha256']
    assert proof['status'] == 'final_vector_and_export_payload_ready_no_CSV' and proof['round'] == 79
    assert proof['fit_rows'] == 467221 and proof['query_rows'] == 47073
    assert proof['plan_sha256'] == PLAN_SHA and proof['decision_sha256'] == RESEARCH_SHA
    assert proof['native_model_full_query_raw_TWD_reload_exact'] is True
    author_name = release['authoring_validation_path']; author = locked(author_name)
    assert sha(ROOT / author_name) == release['authoring_validation_sha256']
    assert author['all_saved_CSV_cells_exact'] is True and author['all_typed_cells_exact'] is True
    assert sha(ROOT / PRIORITY) == release['sha256'] == author['csv_sha256']
    protected = research['protected_before']
    assert len(protected) == 23 and 'runs/versions.json' in protected and PRIORITY not in protected
    assert release['protected_before'] == release['protected_after'] == proof['protected_before'] == proof['protected_after'] == protected
    before_protected = protected_hashes(protected)
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / 'outputs').glob('submission_v*.csv')} == {n for n in protected if n.endswith('.csv')} | {PRIORITY}
    for name, digest in research['input_sha256'].items():
        assert sha(ROOT / name) == digest, name
    before = SELECTOR.read_bytes(); previous = json.loads(before.decode('utf8'))
    assert hashlib.sha256(before).hexdigest() == OLD_SHA
    assert previous['private_priority'] == BACKUP and previous['sha256'] == protected[BACKUP]
    assert previous['backup'] == EARLIER and previous['backup_sha256'] == protected[EARLIER]
    assert previous['known_public_best_filename'] == 'outputs/submission_v18_trial.csv'
    assert previous['known_public_best_mae'] == 2388386.4613
    backup_name = 'outputs/round79/selection/previous_selector.json'
    with (ROOT / backup_name).open('xb') as handle: handle.write(before)
    with (ROOT / backup_name).with_suffix('.sha256').open('x', encoding='ascii') as handle: handle.write(OLD_SHA + '\n')
    completed = datetime.now(timezone.utc).isoformat()
    candidate = {'round': 79, 'path': PRIORITY, 'sha256': release['sha256'], 'rows': 47073,
                 'final_proof': proof_name, 'final_proof_sha256': release['candidate_proof_sha256'],
                 'public_mae': None, 'private_mae': None}
    updates = dict(updated_at_utc=completed, scope='local_recommendation_only',
        private_priority=PRIORITY, sha256=release['sha256'], backup=BACKUP, backup_sha256=protected[BACKUP],
        earlier_alternative=EARLIER, earlier_alternative_sha256=protected[EARLIER], previous_candidate=BACKUP,
        public_mae=None, private_mae=None, uploaded=False, kaggle_selection_changed=False,
        final_version_chosen=False, GitHub_publication_deferred=True,
        previous_selection_metadata={key: previous.get(key) for key in ('latest_completed_research', 'current_priority_release', 'backup_role')},
        latest_completed_research={'round': 79, 'status': 'verified_local_trial_released', 'candidate': 'R79',
            'decision': 'limited_local_v22_priority_with_v20_backup', 'decision_record': RESEARCH,
            'decision_sha256': RESEARCH_SHA, 'report': 'reports/round79_report.md',
            'new_final_tree_fits': 1, 'new_final_NN_fits': 0, 'completed_at_utc': completed, 'final_version_chosen': False},
        current_priority_release={'round': 79, 'candidate': 'R79',
            'decision': 'limited_local_private_preference_after_verified_CSV_release',
            'submission': PRIORITY, 'submission_sha256': release['sha256'], 'final_proof': proof_name,
            'final_proof_sha256': release['candidate_proof_sha256'], 'CSV_release': RELEASE, 'CSV_release_sha256': sha(ROOT / RELEASE),
            'backup': BACKUP, 'backup_sha256': protected[BACKUP], 'earlier_alternative': EARLIER,
            'public_mae': None, 'private_mae': None, 'uploaded': False, 'kaggle_selection_changed': False,
            'final_version_chosen': False, 'completed_at_utc': completed},
        backup_role='R76/v20 is the previous local priority and matched R79 control; its Public and Private MAEs remain unknown.',
        earlier_alternative_role='R77/v21 remains the earlier cross-period complementary alternative; its Public and Private MAEs remain unknown.',
        pending_research={'round': 79, 'status': 'completed_verified_local_trial_release', 'new_final_tree_fits': 1, 'new_final_NN_fits': 0},
        reason='Limited local preference for the fixed R79 recipe after verified v22 release: three reused periods improve overall, fixed-weight and test-composition MAE versus v20; newest and oldest paired intervals favor R79. Development and price-group regressions, seed42-only evidence and unknown Private truth remain. This is not final selection or Public-driven tuning.',
        limitations=list(research['limitations']) + ['The final-fit and CSV prerequisites in the original research decision are now satisfied; Private remains unknown.'],
        candidates=[candidate] + copy.deepcopy(previous.get('candidates', [])),
        final_release=RELEASE, final_release_sha256=sha(ROOT / RELEASE), previous_selector_backup=backup_name,
        evidence=list(dict.fromkeys(previous.get('evidence', []) + [PLAN, RESEARCH, proof_name, RELEASE] + list(research['input_sha256']))))
    updated = copy.deepcopy(previous)
    history = copy.deepcopy(previous.get('selection_history', []))
    history.append({'selector_backup': backup_name, 'selector_sha256': OLD_SHA,
                    'replaced_fields': {key: copy.deepcopy(previous[key]) for key in updates if key in previous}})
    updated.update(updates); updated['selection_history'] = history
    staged = OUT / 'new_selector.json'; write(staged, updated)
    after_protected = protected_hashes(protected)
    assert sha(ROOT / PRIORITY) == release['sha256'] and sha(SELECTOR) == OLD_SHA
    replacement = ROOT / 'outputs/current_private_candidate.next.json'
    with replacement.open('xb') as handle: handle.write(staged.read_bytes())
    os.replace(replacement, SELECTOR)
    assert sha(SELECTOR) == sha(staged)
    write(OUT / 'decision.json', dict(status='local_recommendation_updated', round=79,
        priority=PRIORITY, priority_sha256=release['sha256'], backup=BACKUP, backup_sha256=protected[BACKUP],
        earlier_alternative=EARLIER, earlier_alternative_sha256=protected[EARLIER], release_path=RELEASE,
        release_sha256=sha(ROOT / RELEASE), research_decision_path=RESEARCH, research_decision_sha256=RESEARCH_SHA,
        plan_path=PLAN, plan_sha256=PLAN_SHA, previous_selector_path=backup_name, previous_selector_sha256=OLD_SHA,
        new_selector_sha256=sha(SELECTOR), source_sha256=sha(Path(__file__)),
        protected_before=before_protected, protected_after=after_protected,
        known_public_best_filename=previous['known_public_best_filename'], known_public_best_mae=previous['known_public_best_mae'],
        new_CSV=False, new_fits=0, model_load_predict_calls=0, registry_updated=False,
        public_mae=None, private_mae=None, uploaded=False, kaggle_selection_changed=False,
        final_version_chosen=False, GitHub_published=False, completed_at_utc=completed))
    print(json.dumps({'status': 'local_recommendation_updated', 'priority': PRIORITY, 'backup': BACKUP}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True); parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); started = datetime.now(timezone.utc).isoformat(); code = 1; failure = None
    try:
        run(); code = 0
    except BaseException:
        failure = traceback.format_exc(); raise
    finally:
        write(OUT / 'execution.json', {'command': 'round79_select.py --run', 'exit_code': code,
            'failure': failure, 'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.perf_counter() - start, 'source_sha256': sha(Path(__file__)),
            'selector_sha256_after': sha(SELECTOR), 'automatic_retry': False})
