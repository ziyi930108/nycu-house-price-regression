"""Validate both Artifact-authored trial CSVs before exposing v20 and v21 locally."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round76_77/final_export'
AUTHOR = ROOT / 'outputs/round76_77/csv_authoring'
JOINT = ROOT / 'outputs/round76_77/final'
PLAN = ROOT / 'reports/round76_77_final_plan.md'
PLAN_SHA = '04612c7c862e634c3554331fffe31e64e63400a1c6fd68242b5ecb4aaed194a4'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
SPECS = ((76, 20), (77, 21))
PUBLISHED = []


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def at(name):
    p = (ROOT / name.replace('\\', '/')).resolve()
    assert p.is_relative_to(ROOT) and p != ROOT, name
    return p


def locked(p):
    assert sha(p) == p.with_suffix('.sha256').read_text(encoding='ascii').strip(), str(p)
    return json.loads(p.read_text(encoding='utf8'))


def write(p, obj):
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(p) + '\n')


def protected_now(expected):
    names = {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
    assert names == set(expected) - {'runs/versions.json'}, 'Existing CSV inventory changed'
    current = {n: sha(at(n)) for n in expected}
    assert current == expected
    return current


def run():
    assert sha(PLAN) == PLAN_SHA
    proof = locked(JOINT / 'trial_release.json')
    execution = locked(JOINT / 'execution.json')
    author_execution = locked(AUTHOR / 'execution.json')
    assert proof['status'] == 'final_vectors_and_export_payloads_ready_no_CSV'
    assert proof['rounds'] == [76, 77] and proof['query_rows'] == 47073 and proof['fit_rows'] == 467221
    assert execution['exit_code'] == author_execution['exit_code'] == 0
    assert proof['plan_sha256'] == execution['plan_sha256'] == PLAN_SHA
    assert proof['new_CSV'] is False and proof['selector_updated'] is False and proof['registry_updated'] is False
    assert proof['test_labels_loaded'] is False and proof['uploaded'] is False
    assert proof['new_fits_in_this_composer'] == proof['models_loaded_in_this_composer'] == proof['tree_NN_model_predict_calls_in_this_composer'] == 0
    assert proof['separately_completed_new_NN_fits'] == 1 and proof['old_v19_CSV_exact']
    assert execution['source_sha256'] == proof['source_sha256']['src/round76_77_final.py']
    builder = AUTHOR / 'export.mjs'
    builder_sha = sha(builder)
    assert author_execution['source_sha256'] == builder_sha
    # Joint composer already verified the prepared training archives. Keep their
    # declared identity, but do not repeat multi-GB prepared.npz byte reads here.
    verified, inherited_prepared = {}, {}
    for section in ('source_sha256', 'input_sha256', 'artifact_sha256'):
        for name, expected in proof[section].items():
            p = at(name)
            n = rel(p)
            assert isinstance(expected, str) and len(expected) == 64
            if p.name == 'prepared.npz' and section == 'input_sha256':
                assert p.is_file()
                inherited_prepared[n] = expected
                continue
            if n in verified:
                assert verified[n] == expected, n
            else:
                assert sha(p) == expected, n
                verified[n] = expected
    protected = proof['protected_before']
    assert len(protected) == 21 and 'runs/versions.json' in protected
    assert protected == proof['protected_after']
    protected_now(protected)
    selector_sha = sha(SELECTOR)
    assert selector_sha == proof['selector_sha256_before'] == proof['selector_sha256_after']
    ready = []
    for round_number, version in SPECS:
        folder = ROOT / ('outputs/round' + str(round_number) + '/final')
        author = AUTHOR / ('round' + str(round_number))
        child_path = folder / 'trial_release.json'
        child = locked(child_path)
        assert child == proof['candidates'][str(round_number)]
        assert proof['artifact_sha256'][rel(child_path)] == sha(child_path)
        assert child['status'] == 'final_vector_and_export_payload_ready_no_CSV'
        assert child['round'] == round_number and child['query_rows'] == 47073 and child['plan_sha256'] == PLAN_SHA
        payload_path = folder / 'export_payload.json'
        payload = locked(payload_path)
        authored_path = author / 'validation.json'
        authored = locked(authored_path)
        assert authored['status'] == 'authored_not_published' and authored['round'] == round_number
        assert authored['rows'] == payload['rows'] == 47073 and payload['round'] == round_number
        assert authored['source_sha256'] == builder_sha and authored['plan_sha256'] == PLAN_SHA
        assert authored['joint_proof_sha256'] == sha(JOINT / 'trial_release.json')
        assert authored['candidate_proof_sha256'] == sha(child_path)
        assert authored['payload_path'] == child['export_payload_path'] == rel(payload_path)
        assert sha(payload_path) == child['export_payload_sha256'] == authored['payload_sha256'] == proof['artifact_sha256'][rel(payload_path)]
        assert authored['all_typed_cells_exact'] and authored['all_saved_CSV_cells_exact']
        assert authored['rounding_performed_by_this_writer'] is False and authored['published'] is False
        assert authored['protected_before'] == authored['protected_after'] == protected
        assert authored['selector_sha256_before'] == authored['selector_sha256_after'] == selector_sha
        for key in ('prediction_path', 'prediction_sha256', 'sample_ID_path', 'sample_ID_sha256', 'rounding'):
            assert payload[key] == child[key], key
        prediction_path = at(child['prediction_path'])
        assert sha(prediction_path) == child['prediction_sha256'] == proof['artifact_sha256'][rel(prediction_path)]
        pred = np.load(prediction_path, allow_pickle=False)
        assert pred.dtype == np.float64 and pred.shape == (47073,) and np.isfinite(pred).all() and (pred > 0).all()
        sample_path = at(child['sample_ID_path'])
        assert sha(sample_path) == child['sample_ID_sha256']
        assert verified[rel(sample_path)] == child['sample_ID_sha256']
        with sample_path.open(encoding='utf-8-sig', newline='') as f:
            ids = [r['編號'] for r in csv.DictReader(f)]
        assert ids == payload['ids'] and len(ids) == len(set(ids)) == 47073
        assert all(isinstance(x, str) and x.isascii() and x.isdigit() for x in ids)
        staged = author / 'staged_submission.csv'
        assert sha(staged) == authored['csv_sha256'] and authored['staged_path'] == rel(staged)
        with staged.open('rb') as f:
            assert f.read(3) == b'\xef\xbb\xbf'
        with staged.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))
        assert rows[0] == payload['headers'] == ['編號', '總價元'] and len(rows) == 47074
        assert all(len(row) == 2 for row in rows) and [row[0] for row in rows[1:]] == ids
        assert all(row[1].isascii() and row[1].isdigit() and str(int(row[1])) == row[1] for row in rows[1:])
        values = [int(row[1]) for row in rows[1:]]
        assert all(0 < x <= 2**53 - 1 for x in values)
        assert all(type(x) is int for x in payload['prices']) and values == payload['prices']
        prices = np.asarray(values, dtype=np.int64)
        # Comparison only: never alter/re-round the authored CSV. Rounding was
        # performed once by the composer from each unrounded float64 candidate.
        rounded = np.maximum(np.rint(pred), 1)
        assert (rounded <= 2**53 - 1).all()
        assert np.array_equal(prices, rounded.astype(np.int64))
        target = ROOT / ('outputs/submission_v' + str(version) + '_trial.csv')
        assert not target.exists()
        ready.append((staged, target, {'round': round_number, 'version': version, 'path': rel(target), 'sha256': authored['csv_sha256'],
            'rows': 47073, 'prediction_path': rel(prediction_path), 'prediction_sha256': child['prediction_sha256'],
            'payload_path': rel(payload_path), 'payload_sha256': sha(payload_path), 'candidate_proof_path': rel(child_path), 'candidate_proof_sha256': sha(child_path),
            'authoring_validation_path': rel(authored_path), 'authoring_validation_sha256': sha(authored_path),
            'sample_ID_path': rel(sample_path), 'sample_ID_sha256': child['sample_ID_sha256'],
            'positive_int64_full_roundtrip': True, 'original_unrounded_prediction_comparison_exact': True,
            'UTF8_BOM': True, 'all_sample_IDs_and_order_exact': True, 'exact_headers': True,
            'prediction_min_TWD': float(pred.min()), 'prediction_max_TWD': float(pred.max()),
            'prediction_changes_vs_v19': {k: v for k, v in child.items() if k.endswith('_vs_v19') or '_vs_v19_' in k}}))
    # Both candidates passed every check. A pair of hard links is not one atomic
    # filesystem operation: retain any successful first publication on OS failure
    # and report it in execution.json; never silently roll back or retry.
    protected_now(protected)
    assert sha(SELECTOR) == selector_sha
    for staged, target, evidence in ready:
        assert not target.exists() and sha(staged) == evidence['sha256']
    for staged, target, evidence in ready:
        os.link(staged, target)
        PUBLISHED.append(rel(target))
        assert sha(target) == evidence['sha256']
    protected_after = {n: sha(at(n)) for n in protected}
    assert protected_after == protected and sha(SELECTOR) == selector_sha
    assert {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')} == (set(protected) - {'runs/versions.json'}) | set(PUBLISHED)
    result = {'rounds': [76, 77], 'status': 'experimental_v20_v21_trials_generated', 'candidates': {str(e['round']): e for _, _, e in ready},
        'joint_vector_evidence_path': rel(JOINT / 'trial_release.json'), 'joint_vector_evidence_sha256': sha(JOINT / 'trial_release.json'),
        'plan_path': rel(PLAN), 'plan_sha256': PLAN_SHA,
        'source_sha256': {rel(Path(__file__)): sha(Path(__file__)), rel(builder): builder_sha},
        'verified_dependency_sha256': verified, 'inherited_prepared_input_sha256': inherited_prepared,
        'prepared_archives_rehashed_in_this_release': False, 'both_candidates_verified_before_publication': True,
        'new_model_fits': 0, 'model_loads': 0, 'model_prediction_calls': 0, 'CSV_prices_modified': False,
        'protected_before': protected, 'protected_after': protected_after, 'old_CSV_unchanged': True,
        'selector_sha256_before': selector_sha, 'selector_sha256_after': sha(SELECTOR), 'selector_updated': False,
        'registry_updated': False, 'uploaded': False, 'kaggle_selection_changed': False, 'final_version_chosen': False,
        'public_mae': None, 'private_mae': None, 'prediction_changes_are_not_errors': True}
    write(OUT / 'release.json', result)
    print(json.dumps({'status': result['status'], 'candidates': {k: {'path': v['path'], 'sha256': v['sha256']} for k, v in result['candidates'].items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    assert not OUT.exists() and all(not (ROOT / ('outputs/submission_v' + str(v) + '_trial.csv')).exists() for _, v in SPECS), 'No overwrite or automatic retry'
    OUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    code, failure = 0, None
    try:
        run()
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure, 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'published_paths': PUBLISHED, 'automatic_retry': False})
    raise SystemExit(code)
