"""Verify one Artifact-authored v22 CSV against its unrounded vector, then publish locally."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round79/final_export'
AUTHOR = ROOT / 'outputs/round79/csv_authoring'
FINAL = ROOT / 'outputs/round79/final'
PLAN = ROOT / 'reports/round79_final_plan.md'
PLAN_SHA = 'cd3dd3b71fc24e405c8ade92ee89bc928e8ad76a6feca258c79e66e65455ae2d'
DECISION = ROOT / 'outputs/round79/decision.json'
DECISION_SHA = '12c5c188057ef36704f6e0553d1c9cfbd7d3d345dbbcac083275aa8fe9405ef5'
BUILDER_SHA = '1d14668ea53f52408c2e06f4a6d432ae99365d69079c634e06de31cfe272dc37'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
TARGET = ROOT / 'outputs/submission_v22_trial.csv'
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
    value = {name: sha(at(name)) for name in expected}
    assert value == expected
    return value


def run():
    assert sha(PLAN) == PLAN_SHA and sha(DECISION) == DECISION_SHA
    proof_path, payload_path = FINAL / 'trial_release.json', FINAL / 'export_payload.json'
    proof, payload = locked(proof_path), locked(payload_path)
    execution = locked(FINAL / 'execution.json')
    author_execution = locked(AUTHOR / 'execution.json')
    author = AUTHOR / 'round79'
    validation_path = author / 'validation.json'
    validation = locked(validation_path)
    builder = AUTHOR / 'export.mjs'
    assert sha(builder) == BUILDER_SHA == author_execution['source_sha256'] == validation['source_sha256']
    assert execution['exit_code'] == author_execution['exit_code'] == 0
    assert proof['status'] == 'final_vector_and_export_payload_ready_no_CSV' and proof['round'] == 79
    assert proof['fit_rows'] == 467221 and proof['query_rows'] == 47073
    assert proof['plan_sha256'] == PLAN_SHA and proof['decision_sha256'] == DECISION_SHA
    assert execution['source_sha256'] == proof['source_sha256']['src/round79_final.py'] == 'c6ad4103c7e75ff1954954b234a671e842a4918618d5ceaadc6ecb4881b97684'
    for key in ('new_fits_in_this_composer', 'new_NN_fits', 'models_loaded_in_this_composer',
                'tree_NN_model_predict_calls_in_this_composer', 'reference_or_spatial_refits'):
        assert proof[key] == 0
    assert proof['separately_completed_new_tree_fits'] == 1
    assert proof['reference_predict_calls'] == proof['spatial_predict_calls'] == 2
    for key in ('native_model_full_query_raw_TWD_reload_exact', 'saved_prediction_vector_reload_exact', 'old_v20_CSV_exact'):
        assert proof[key] is True
    for key in ('raw_labels_loaded', 'test_labels_loaded', 'new_CSV', 'selector_updated', 'registry_updated', 'uploaded', 'final_version_chosen'):
        assert proof[key] is False
    verified = {rel(PLAN): PLAN_SHA, rel(DECISION): DECISION_SHA,
                rel(proof_path): sha(proof_path), rel(FINAL / 'execution.json'): sha(FINAL / 'execution.json')}
    # Recheck actual source bytes and release-relevant small evidence. Original
    # data, prepared archives and model/raw-unit files remain composer authority;
    # do not deserialize or rehash those large artifacts in a CSV release.
    for name, digest in proof['source_sha256'].items():
        p = at(name)
        assert p.suffix in ('.py', '.md', '.json') and p.stat().st_size < 30_000_000
        assert sha(p) == digest, name
        verified[rel(p)] = digest
    model_result_path = at(proof['model_result_path'])
    assert sha(model_result_path) == proof['model_result_sha256']
    model_result = locked(model_result_path)
    assert model_result['status'] == 'final_unit_model_ready_no_CSV'
    assert model_result['text_plan_sha256'] == PLAN_SHA and model_result['decision_sha256'] == DECISION_SHA
    assert model_result['fit_rows'] == 467221 and model_result['query_rows'] == 47073
    assert model_result['new_fits'] == 1 and model_result['new_NN_fits'] == 0
    assert model_result['full_query_raw_TWD_reload_exact'] and model_result['new_CSV'] is False
    verified[rel(model_result_path)] = proof['model_result_sha256']
    assert validation['status'] == 'authored_not_published' and validation['round'] == payload['round'] == 79
    assert validation['version'] == 22 and validation['rows'] == payload['rows'] == 47073
    assert validation['intended_filename'] == payload['intended_filename'] == rel(TARGET)
    assert validation['plan_sha256'] == PLAN_SHA and validation['decision_sha256'] == DECISION_SHA
    assert validation['candidate_proof_path'] == rel(proof_path) and validation['candidate_proof_sha256'] == sha(proof_path)
    assert validation['payload_path'] == proof['export_payload_path'] == rel(payload_path)
    assert sha(payload_path) == proof['export_payload_sha256'] == validation['payload_sha256'] == proof['artifact_sha256'][rel(payload_path)]
    for key in ('utf8_BOM', 'ID_order_preserved', 'all_prices_safe_positive_integers', 'all_typed_cells_exact', 'all_saved_CSV_cells_exact'):
        assert validation[key] is True
    assert validation['rounding_performed_by_this_writer'] is False and validation['published'] is False
    assert validation['new_model_fits'] == 0
    protected = proof['protected_before']
    assert len(protected) == 23 and 'runs/versions.json' in protected and protected == proof['protected_after']
    assert validation['protected_before'] == validation['protected_after'] == protected
    protected_now(protected)
    selector_sha = sha(SELECTOR)
    assert selector_sha == proof['selector_sha256_before'] == proof['selector_sha256_after']
    assert selector_sha == validation['selector_sha256_before'] == validation['selector_sha256_after']
    for key in ('prediction_path', 'prediction_sha256', 'sample_ID_path', 'sample_ID_sha256', 'rounding'):
        assert payload[key] == proof[key], key
    assert payload['rounding'] == 'maximum(rint(unrounded candidate TWD),1).astype(int64), once'
    prediction_path = at(proof['prediction_path'])
    assert prediction_path == FINAL / 'predictions.npy'
    assert sha(prediction_path) == proof['prediction_sha256'] == proof['artifact_sha256'][rel(prediction_path)]
    pred = np.load(prediction_path, allow_pickle=False)
    assert pred.dtype == np.float64 and pred.shape == (47073,) and np.isfinite(pred).all() and (pred > 0).all()
    sample_path = at(proof['sample_ID_path'])
    assert sha(sample_path) == proof['sample_ID_sha256'] == proof['input_sha256'][rel(sample_path)]
    with sample_path.open(encoding='utf-8-sig', newline='') as f:
        ids = [r['編號'] for r in csv.DictReader(f)]
    assert ids == payload['ids'] and len(ids) == len(set(ids)) == 47073
    assert all(isinstance(x, str) and x.isascii() and x.isdigit() for x in ids)
    staged = author / 'staged_submission.csv'
    assert validation['staged_path'] == rel(staged) and sha(staged) == validation['csv_sha256']
    assert sha(author / 'preview.png') == validation['preview_sha256']
    with staged.open('rb') as f:
        assert f.read(3) == b'\xef\xbb\xbf'
    with staged.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == payload['headers'] == validation['columns'] == ['編號', '總價元'] and len(rows) == 47074
    assert all(len(row) == 2 for row in rows) and [row[0] for row in rows[1:]] == ids
    assert all(row[1].isascii() and row[1].isdigit() and str(int(row[1])) == row[1] for row in rows[1:])
    values = [int(row[1]) for row in rows[1:]]
    assert all(0 < x <= 2**53 - 1 for x in values)
    assert all(type(x) is int for x in payload['prices']) and values == payload['prices']
    prices = np.asarray(values, dtype=np.int64)
    # Comparison only; preserve the CSV integers rounded once by the composer.
    rounded = np.maximum(np.rint(pred), 1)
    assert (rounded <= 2**53 - 1).all() and np.array_equal(prices, rounded.astype(np.int64))
    verified.update({rel(payload_path): sha(payload_path), rel(prediction_path): sha(prediction_path),
                     rel(sample_path): sha(sample_path), rel(validation_path): sha(validation_path), rel(builder): BUILDER_SHA})
    protected_now(protected)
    assert sha(SELECTOR) == selector_sha and not TARGET.exists()
    assert all(sha(at(name)) == digest for name, digest in verified.items())
    assert sha(staged) == validation['csv_sha256']
    # A single hard link exposes the already-verified bytes and refuses overwrite.
    os.link(staged, TARGET)
    PUBLISHED.append(rel(TARGET))
    assert sha(TARGET) == validation['csv_sha256']
    after = {name: sha(at(name)) for name in protected}
    assert after == protected and sha(SELECTOR) == selector_sha
    assert {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')} == (set(protected) - {'runs/versions.json'}) | {rel(TARGET)}
    result = {'round': 79, 'version': 22, 'status': 'experimental_v22_trial_generated',
        'path': rel(TARGET), 'sha256': sha(TARGET), 'rows': 47073,
        'prediction_path': rel(prediction_path), 'prediction_sha256': proof['prediction_sha256'],
        'payload_path': rel(payload_path), 'payload_sha256': sha(payload_path),
        'candidate_proof_path': rel(proof_path), 'candidate_proof_sha256': sha(proof_path),
        'authoring_validation_path': rel(validation_path), 'authoring_validation_sha256': sha(validation_path),
        'sample_ID_path': rel(sample_path), 'sample_ID_sha256': proof['sample_ID_sha256'],
        'plan_path': rel(PLAN), 'plan_sha256': PLAN_SHA, 'decision_path': rel(DECISION), 'decision_sha256': DECISION_SHA,
        'positive_int64_full_roundtrip': True, 'original_unrounded_prediction_comparison_exact': True,
        'UTF8_BOM': True, 'all_sample_IDs_and_order_exact': True, 'exact_headers': True,
        'prediction_min_TWD': float(pred.min()), 'prediction_max_TWD': float(pred.max()),
        'prediction_changes_vs_v20': {k: v for k, v in proof.items() if k.endswith('_vs_v20') or '_vs_v20_' in k},
        'source_sha256': {rel(Path(__file__)): sha(Path(__file__)), rel(builder): BUILDER_SHA},
        'verified_release_dependency_sha256': verified, 'composer_input_sha256_inherited': proof['input_sha256'],
        'large_model_raw_or_prepared_artifacts_rehashed': False, 'model_fit_load_predict_calls_in_release': 0,
        'composer_declared_separately_completed_tree_fits': 1, 'composer_and_model_declared_full_query_raw_TWD_reload_exact': True,
        'independent_model_reload_performed_in_release': False,
        'unrounded_prediction_array_read_for_comparison': True, 'CSV_prices_modified': False,
        'protected_before': protected, 'protected_after': after, 'old_CSV_unchanged': True,
        'selector_sha256_before': selector_sha, 'selector_sha256_after': sha(SELECTOR), 'selector_updated': False,
        'registry_updated': False, 'uploaded': False, 'kaggle_selection_changed': False,
        'final_version_chosen': False, 'public_mae': None, 'private_mae': None, 'prediction_changes_are_not_errors': True}
    write(OUT / 'release.json', result)
    print(json.dumps({'status': result['status'], 'path': result['path'], 'sha256': result['sha256'], 'rows': 47073}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    assert not OUT.exists() and not TARGET.exists(), 'No overwrite or automatic retry'
    OUT.mkdir(parents=True, exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
    code, failure = 0, None
    try:
        run()
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure,
        'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - tick, 'source_sha256': sha(Path(__file__)),
        'published_paths': PUBLISHED, 'automatic_retry': False})
    raise SystemExit(code)
