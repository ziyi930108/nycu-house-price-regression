"""Final file/JSON/CSV QA for the fixed R6 release; no training or scoring.

Prediction/model/data blobs are only hashed. Only the two full-fit arange index
files and the released/sample CSV are decoded. Missing required evidence fails.
The output starts false, so an interrupted/failed check cannot retain old success.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/round6'
OUTPUT = ROOT/'reports/round6_final_checks.json'
PLAN_SHA = '606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
CANDIDATE, COMPONENT = 'geography_blend', 'geography_catboost'
STAGES = ('development', 'confirmation', 'auxiliary')
ID, TARGET = '編號', '總價元'
SAMPLE = ROOT/'台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/test-reindex-test-sample-submission-v2.csv'
DEV_GATES = {'overall_improves_at_least_1pct', 'weighted_improves_at_least_1pct', 'quarters_deteriorate_at_most_5pct', 'large_strata_deteriorate_at_most_5pct'}
LATER_GATES = {'overall_within_limit', 'weighted_within_limit', 'large_strata_deteriorate_at_most_5pct'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class Verifier:
    def __init__(self):
        self.files = {}

    def path(self, value):
        path = (ROOT/Path(value)).resolve()
        need(path.is_relative_to(ROOT), 'Path escapes workspace: ' + str(value))
        return path

    def digest(self, value):
        path = self.path(value)
        need(path.is_file(), 'Missing artifact: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        before = stamp(path)
        if key in self.files:
            need(self.files[key]['stat'] == before, 'Artifact changed during QA: ' + key)
            return self.files[key]['sha256']
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                h.update(chunk)
        need(stamp(path) == before, 'Artifact changed while hashing: ' + key)
        self.files[key] = {'sha256': h.hexdigest(), 'stat': before}
        return h.hexdigest()

    def expect(self, value, expected):
        need(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid SHA256 record')
        need(self.digest(value) == expected, 'SHA256 differs: ' + str(value))

    def mapping(self, records):
        need(isinstance(records, dict) and records, 'Missing artifact hash mapping')
        for path, digest in records.items():
            self.expect(path, digest)

    def json(self, value, locked=False):
        path = self.path(value)
        digest = self.digest(path)
        if locked:
            receipt = path.with_suffix('.sha256')
            self.digest(receipt)
            need(receipt.read_text(encoding='ascii').strip() == digest, 'Receipt differs: ' + str(path))
        def invalid(value):
            raise ValueError('Nonfinite JSON constant: ' + value)
        result = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid)
        need(isinstance(result, dict), 'Expected JSON object: ' + str(path))
        return result

    def unchanged(self):
        for name, record in self.files.items():
            need(self.path(name).is_file() and stamp(self.path(name)) == record['stat'], 'Artifact changed during QA: ' + name)


def check_arange(path, count):
    """Decode only int64 index NPYs, never predictions or training values."""
    with path.open('rb') as handle:
        need(handle.read(6) == b'\x93NUMPY', 'Invalid index NPY magic')
        version = tuple(handle.read(2))
        need(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported index NPY version')
        size = struct.unpack('<H' if version == (1, 0) else '<I', handle.read(2 if version == (1, 0) else 4))[0]
        header = ast.literal_eval(handle.read(size).decode('utf-8' if version == (3, 0) else 'latin1'))
        need(header['descr'] == '<i8' and header['shape'] == (count,) and header['fortran_order'] is False, 'Full index schema differs')
        raw = handle.read()
    need(len(raw) == count*8, 'Index length differs')
    need(all(value[0] == index for index, value in enumerate(struct.iter_unpack('<q', raw))), 'Final indices are not canonical arange')


def csv_ids(path, check_prices=False):
    ids = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        need(reader.fieldnames == [ID, TARGET], 'CSV columns/order differ: ' + str(path))
        for row in reader:
            need(set(row) == {ID, TARGET} and isinstance(row[ID], str) and row[ID] != '', 'Malformed CSV row')
            ids.append(row[ID])
            if check_prices:
                text = row[TARGET]
                need(isinstance(text, str) and re.fullmatch(r'[1-9][0-9]*', text), 'CSV price is not a positive integer')
                need(int(text) <= 9223372036854775807, 'CSV price exceeds int64')
    need(len(ids) == 47073 and len(set(ids)) == 47073, 'CSV row count or ID uniqueness differs')
    return ids


def markdown_links(path):
    """Inspect ordinary local links; skip fenced/inline code, URLs and anchors."""
    text = path.read_text(encoding='utf-8')
    need('\ufffd' not in text, 'Replacement character in report: ' + str(path))
    visible, fence = [], None
    for line in text.splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            token = match.group(1)
            if fence is None:
                fence = token[0]
            elif fence == token[0]:
                fence = None
            continue
        if fence is None:
            visible.append(line)
    clean = re.sub(r'(`+).*?\1', '', '\n'.join(visible))
    links = []
    for value in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', clean):
        value = value.strip()
        if value.startswith('<') and '>' in value:
            value = value[1:value.index('>')]
        else:
            value = value.split(' ', 1)[0]
        if not value or value.startswith('#') or value.startswith('//'):
            continue
        if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', value) and not re.match(r'^[A-Za-z]:[\\/]', value):
            continue
        local = unquote(value.split('#', 1)[0].split('?', 1)[0])
        target = (path.parent/local).resolve()
        need(target.is_relative_to(ROOT) and target.exists(), 'Broken/unsafe local report link: ' + value)
        links.append(target.relative_to(ROOT).as_posix())
    return links


def check_all():
    v = Verifier()
    plan = v.json(OUT/'candidate_plan.json', locked=True)
    v.expect(OUT/'candidate_plan.json', PLAN_SHA)
    frozen = v.json(OUT/'frozen_candidate.json', locked=True)
    accepted = v.json(OUT/'accepted_candidate.json', locked=True)
    release = v.json(OUT/'release/release.json', locked=True)
    identities = {'plan_sha256': PLAN_SHA, 'frozen_candidate_sha256': v.digest(OUT/'frozen_candidate.json'), 'accepted_candidate_sha256': v.digest(OUT/'accepted_candidate.json'), 'release_sha256': v.digest(OUT/'release/release.json')}
    need(plan['round'] == 6 and set(plan['candidates']) == {CANDIDATE}, 'Wrong candidate plan')
    need(frozen['selected_candidate'] == accepted['selected_candidate'] == CANDIDATE and frozen['required_components'] == [COMPONENT], 'Candidate selection differs')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == PLAN_SHA and accepted['frozen_candidate_sha256'] == identities['frozen_candidate_sha256'], 'Selection/acceptance identities differ')
    need(accepted['accepted'] is True and accepted['deployment_decision'] == CANDIDATE and accepted['candidate_reselection_performed'] is False, 'Fixed backtests were not accepted')
    for name in ('fit_source_sha256', 'evaluation_source_sha256', 'geography_transform_source_sha256', 'research_evidence_sha256', 'data_sha256'):
        v.mapping(plan[name])
    need(frozen['source_sha256'] == plan['evaluation_source_sha256'], 'Frozen source closure differs')
    v.mapping(frozen['development_artifact_sha256']);v.mapping(accepted['evidence_sha256'])
    need(set(accepted['stage_decisions']) == {'confirmation', 'auxiliary'}, 'Missing finalized stage decisions')
    stage_records = {}
    for stage in STAGES:
        path = OUT/'evaluation'/stage/'results.json';result = v.json(path, locked=True)
        need(result['stage'] == stage and result['plan_sha256'] == PLAN_SHA, 'Stage/plan differs')
        need(result['candidate_formulas'] == plan['candidate_formulas'] and result['thresholds'] == plan['thresholds'], 'Saved stage rules differ')
        need(result['source_sha256'] == plan['evaluation_source_sha256'], 'Stage source differs')
        decision = result['decision'];need(decision['selected_candidate'] == CANDIDATE, 'Stage selected another candidate')
        if stage == 'development':
            item = decision['candidate_checks'][CANDIDATE];flags = item['checks'];expected = DEV_GATES
            need(item['eligible'] is True and decision['qualified_candidates'] == [CANDIDATE], 'Development did not qualify')
            need(result['frozen_candidate_sha256'] is None, 'Development has later freeze')
        else:
            flags = decision['checks'];expected = LATER_GATES | ({'at_least_two_quarters_improve'} if stage == 'confirmation' else set())
            need(decision['passed'] is True and decision['candidate_reselection_performed'] is False, 'Later stage did not pass')
            need(accepted['stage_decisions'][stage] == decision and result['frozen_candidate_sha256'] == identities['frozen_candidate_sha256'], 'Final/stage frozen decision differs')
        need(set(flags) == expected and all(value is True for value in flags.values()), 'A saved fixed gate did not pass')
        for name in ('input_artifact_sha256', 'output_artifact_sha256'):
            v.mapping(result[name])
        for suffix in ('.json', '.sha256'):
            key = (OUT/'evaluation'/stage/('results'+suffix)).relative_to(ROOT).as_posix()
            need(accepted['evidence_sha256'].get(key) == v.digest(key), 'Accepted stage evidence differs')
        stage_records[stage] = {'result_sha256': v.digest(path), 'all_saved_gates_passed': True, 'checks': flags}
    for name in ('plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256'):
        need(release[name] == identities[name], 'Release identity differs: ' + name)
    need(release['candidate'] == CANDIDATE and release['release_version'] == 'v3' and release['experiment_round'] == 6, 'Wrong release')
    need(release['rows'] == 47073 and release['training_rows'] == 467221 and release['columns'] == [ID, TARGET], 'Release row/schema metadata differs')
    need(release['formula'] == plan['candidate_formulas'][CANDIDATE] and release['path'] == 'outputs/submission_v3.csv', 'Release formula/path differs')
    for flag in ('full_component_replayed_all_rows', 'v2_csv_reproduced_exactly', 'id_order_verified', 'positive_finite_integer_prices'):
        need(release[flag] is True, 'Missing saved release verification: ' + flag)
    v.expect('src/round6_release.py', release['release_source_sha256'])
    v.mapping(release['full_component_sha256']);v.mapping(release['baseline_component_sha256'])
    v.expect(OUT/'release/predictions.npy', release['unrounded_prediction_sha256'])
    v.expect('outputs/round2/neighborhood/final_reference.pkl', release['reference_sha256'])
    folder = OUT/'models/final'/COMPONENT
    meta = v.json(folder/'metadata.json');state = v.json(folder/'feature_state.json')
    need(meta['stage'] == 'final' and meta['component'] == COMPONENT and meta['plan_sha256'] == PLAN_SHA and meta['frozen_candidate_sha256'] == identities['frozen_candidate_sha256'], 'Full model identity differs')
    need(meta['parameters'] == plan['components'][COMPONENT]['parameters'] and meta['iterations'] == 2999 and meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23, 'Full model parameters/schema differ')
    need(meta['outer_fit_rows'] == 467221 and meta['query_rows'] == 47073 and meta['reload_64_rows_verified'] is True, 'Full model rows/reload evidence differ')
    need(meta['fit_source_sha256'] == plan['fit_source_sha256'], 'Full fitting source closure differs')
    for flag in ('fit_uses_eval_set', 'early_stopping_used', 'query_labels_used'):
        need(meta[flag] is False, 'Unexpected full fitting label/early-stop policy')
    for filename, key in [('model.cbm', 'model_sha256'), ('feature_state.json', 'feature_state_sha256'), ('predictions.npy', 'prediction_sha256'), ('fit_indices.npy', 'fit_indices_sha256'), ('query_indices.npy', 'query_indices_sha256')]:
        v.expect(folder/filename, meta[key])
        relative = (folder/filename).relative_to(ROOT).as_posix()
        need(release['full_component_sha256'].get(relative) == meta[key], 'Release/full metadata file hash differs')
    check_arange(folder/'fit_indices.npy', 467221);check_arange(folder/'query_indices.npy', 47073)
    state_hash = hashlib.sha256(json.dumps({k:value for k,value in state.items() if k != 'state_content_sha256'}, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    need(state_hash == state['state_content_sha256'] == meta['feature_state_content_sha256'], 'Feature state content identity differs')
    need(state['fit_rows'] == 467221 and len(state['original_columns']) == 87 and len(state['all_feature_names']) == 95, 'Full feature state schema differs')
    geo_path = OUT/'geography/final/manifest.json';geo = v.json(geo_path, locked=True)
    need(geo['stage'] == 'final' and geo['snapshot_year'] == meta['geography_snapshot_year'] == 2024, 'Final geography year differs')
    need(meta['geography_manifest_sha256'] == release['geography_manifest_sha256'] == v.digest(geo_path), 'Final geography identity differs')
    need(meta['geography_manifest_path'] == release['geography_manifest_path'] == geo_path.relative_to(ROOT).as_posix(), 'Final geography path differs')
    need(geo['feature_rules'] == state['feature_rules'] == meta['geo_feature_rules'] == plan['components'][COMPONENT]['feature_rules'], 'Final geographic rules differ')
    need(geo['features']['full']['rows'] == 467221 and geo['features']['test']['rows'] == 47073, 'Final geographic row counts differ')
    need(geo['geographic_values_learned_from_targets'] is False and geo['existing_target_containing_dedup_rule_reproduced_for_alignment'] is True, 'Geography target/dedup policy differs')
    v.mapping(geo['source_sha256']);v.mapping(geo['index_files_sha256'])
    for artifact in geo['features'].values():
        v.expect(artifact['path'], artifact['sha256'])
    submission = ROOT/release['path'];v.expect(submission, release['sha256']);v.expect('outputs/submission_v2.csv', V2_SHA)
    need(release['v2_unchanged_sha256'] == V2_SHA, 'Preserved v2 identity differs')
    v.digest(SAMPLE)
    need(csv_ids(submission, check_prices=True) == csv_ids(SAMPLE), 'Sample/submission ID order differs')
    report = v.json('reports/round6_results.json')
    need(report['status'] == 'released_v3' and report['recommendation'] == 'v3' and report['backtest_accepted'] is True and report['acceptance_finalized'] is True, 'Final result report is stale/unaccepted')
    need(report['recommended_submission'] == release['path'] and report['new_submission_verified_by_this_report'] is True and report['new_submission']['sha256'] == release['sha256'] and report['new_submission']['rows'] == 47073, 'Report submission differs')
    need(report['plan_sha256'] == PLAN_SHA and report['frozen_candidate_sha256'] == identities['frozen_candidate_sha256'] and report['accepted_candidate_sha256'] == identities['accepted_candidate_sha256'] and report['release'] == release, 'Report provenance differs')
    for stage in STAGES:
        need(report['stages'][stage]['status'] == 'evaluated' and report['stages'][stage]['result_sha256'] == stage_records[stage]['result_sha256'], 'Report stage is stale')
    v.expect('src/round6_report.py', report['report_script_sha256']);v.mapping(report['receipt_verified_inputs'])
    for relative in report['missing_optional_inputs']:
        need(not v.path(relative).exists(), 'Report omits an existing optional input')
    links = {}
    for relative in ('README_round6.md', 'reports/optimization_round6_report.md', 'reports/round6_preprocessing_record.md', 'reports/round6_work_log.md', 'reports/round6_validation_plan.md'):
        path = v.path(relative);v.digest(path);links[relative] = markdown_links(path)
    for relative in ('README_round6.md', 'reports/optimization_round6_report.md'):
        need('outputs/submission_v3.csv' in links[relative], 'Report does not link the new submission')
    replay_path = OUT/'release/replay_verification.json'
    replay = {'present': False, 'status': 'not_saved'}
    if replay_path.exists():
        has_receipt = replay_path.with_suffix('.sha256').exists()
        record = v.json(replay_path, locked=has_receipt)
        need(record['release_sha256'] == identities['release_sha256'] and record['submission_sha256'] == release['sha256'], 'Replay release/submission identity differs')
        source = record.get('release_source_sha256', record.get('source_sha256'))
        need(source == release['release_source_sha256'], 'Replay release source differs')
        if 'release_source_sha256' in record and 'source_sha256' in record:
            need(record['release_source_sha256'] == record['source_sha256'], 'Replay source aliases disagree')
        for flag in ('all_predictions_recomputed', 'v2_reproduced_exactly'):
            if flag in record:
                need(record[flag] is True, 'Replay did not verify: ' + flag)
        replay = {'present': True, 'status': 'sha_verified', 'receipt_verified': has_receipt, 'sha256': v.digest(replay_path)}
    else:
        need(not replay_path.with_suffix('.sha256').exists(), 'Orphan replay receipt')
    v.unchanged()
    return {'all_checks_passed': True, **identities, 'submission_sha256': release['sha256'], 'submission_path': release['path'], 'rows': 47073, 'full_fit_rows': 467221, 'csv_columns': [ID, TARGET], 'sample_id_order_verified': True, 'unique_ids_verified': True, 'positive_integer_prices_verified': True, 'full_fit_indices_verified': True, 'full_model_and_geo_hashes_verified': True, 'v2_sha256': V2_SHA, 'v2_unchanged': True, 'frozen_source_hashes_unchanged': True, 'stages': stage_records, 'report_status': 'released_v3', 'report_local_links_checked': {k:len(values) for k,values in links.items()}, 'optional_replay_verification': replay, 'targets_or_prediction_arrays_deserialized': False, 'models_loaded': False, 'training_or_scoring_performed': False, 'verified_file_sha256': {name:record['sha256'] for name,record in v.files.items()}}


def write_result(value):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base = {'all_checks_passed': False, 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'script_path': 'src/round6_final_checks.py', 'script_sha256': script_sha256, 'checker_source_sha256': script_sha256}
    write_result({**base, 'status': 'running'})
    try:
        result = check_all()
    except Exception as error:
        write_result({**base, 'status': 'failed', 'error': type(error).__name__ + ': ' + str(error), 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
        raise
    final = {**base, **result, 'status': 'passed', 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    write_result(final)
    print(json.dumps({k:final[k] for k in ('all_checks_passed', 'plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256', 'release_sha256', 'submission_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
