"""Audit and freeze round two without changing raw inputs or the v1 snapshot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import uuid
from pathlib import Path

from prepare_data import ROOT, DATA_DIR, FILENAMES, SAMPLE_FILENAME, ID, TARGET, sha256

EXPECTED_VARIANTS = {'v2', 'v2_anchor', 'v2_mix'}
EXPECTED_ROWS = 47073
CONFIG = 'models/round2/release_config.json'
RESULTS = 'reports/round2_results.json'
PROOF = 'reports/round2_notebook_checks.json'
RUN_MANIFEST = 'reports/round2_run_manifest.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    return json.loads(safe_path(ROOT, relative).read_text(encoding='utf-8-sig'))


def safe_path(base, relative):
    """Reject traversal, absolute artifact paths, and links outside the chosen root."""
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, f'Unsafe artifact path: {relative}')
    base = base.resolve()
    result = (base / value).resolve()
    require(result.is_relative_to(base) and result != base, f'Path leaves archive root: {relative}')
    return result


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), f'Invalid expected SHA: {path}')
    require(path.is_file(), f'Required file is missing: {path}')
    actual = sha256(path)
    require(actual == expected, f'SHA-256 mismatch: {path}')
    return actual


def verify_v1():
    base = ROOT / 'runs' / 'v1'
    manifest = read('runs/v1/snapshot_manifest.json')
    expected = manifest['files_sha256']
    require(manifest['version'] == 'v1' and expected, 'Invalid v1 snapshot manifest')
    for relative, digest in expected.items():
        check_hash(safe_path(base, relative), digest)
    return {
        'snapshot_manifest_sha256': sha256(base / 'snapshot_manifest.json'),
        'verified_files': len(expected),
    }


def audit_submission(name, expected_ids, config_sha, test_sha):
    relative = f'outputs/submission_{name}.csv'
    path = safe_path(ROOT, relative)
    prices = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        require(next(reader, None) == [ID, TARGET], f'Unexpected CSV header: {name}')
        for index, row in enumerate(reader):
            require(len(row) == 2, f'Extra or missing CSV column: {name}, row {index + 2}')
            require(index < len(expected_ids) and row[0] == expected_ids[index], f'CSV ID order differs: {name}, row {index + 2}')
            require(re.fullmatch(r'[0-9]+', row[1]) is not None, f'Noninteger/nonfinite price: {name}, row {index + 2}')
            value = int(row[1])
            require(value > 0, f'Nonpositive price: {name}, row {index + 2}')
            prices.append(value)
    require(len(prices) == EXPECTED_ROWS, f'Unexpected CSV row count: {name}')
    checks = read(f'outputs/submission_{name}_checks.json')
    expected = {
        'name': name, 'rows': len(prices), 'columns': [ID, TARGET],
        'sha256': sha256(path), 'config_sha256': config_sha,
        'test_source_sha256': test_sha, 'id_order_verified': True,
        'positive_finite_integer_prices': True, 'minimum': min(prices),
        'maximum': max(prices), 'median': statistics.median(prices),
    }
    for key, value in expected.items():
        require(checks.get(key) == value, f'Submission check differs: {name}/{key}')
    require(Path(checks['path']).resolve() == path, f'Submission check path differs: {name}')
    require(checks['seen_project_rows'] + checks['unseen_project_rows'] == EXPECTED_ROWS, f'Project counts differ: {name}')
    return checks


def audit():
    config = read(CONFIG)
    results = read(RESULTS)
    original = read('reports/run_manifest.json')['input_sha256']
    required_inputs = set(FILENAMES.values()) | {SAMPLE_FILENAME}
    require(set(original) == required_inputs, 'Original manifest must describe the four supplied inputs')
    for filename, digest in original.items():
        check_hash(safe_path(DATA_DIR, filename), digest)
    v1 = verify_v1()
    require(original == read('runs/v1/reports/run_manifest.json')['input_sha256'], 'Original input manifest differs from frozen v1')
    config_sha = sha256(ROOT / CONFIG)
    sources = {split: original[filename] for split, filename in FILENAMES.items()}
    require(config['source_sha256'] == sources, 'Release input provenance differs from original manifest')
    require(results['source_sha256'] == sources, 'Report input provenance differs')
    require(results['release_config_sha256'] == config_sha, 'Report was generated from another release configuration')
    require(results['metric'] == 'MAE' and results['unit'] == 'TWD', 'Unexpected report metric/unit')
    require(results['variants'] == config['variants'], 'Report variants or metrics differ from release configuration')
    require(results['final_training']['components'] == config['components'], 'Report model records differ from release configuration')
    require(results['final_training']['rows'] == config['training_rows'] and results['final_training']['sources'] == config['training_sources'], 'Report final-training rows or sources differ')
    require(results['baseline'] == read('runs/v1/reports/results.json')['selection'], 'Report baseline differs from v1')
    require(config['recommended_variant'] == 'v2', 'Unexpected recommended variant')
    names = [variant['name'] for variant in config['variants']]
    require(len(names) == 3 and set(names) == EXPECTED_VARIANTS, 'Release must contain v2, v2_anchor, and v2_mix exactly once')
    components = config['components']
    expected_artifacts = {CONFIG: config_sha}
    expected_artifacts.update(config['inference_source_sha256'])
    expected_artifacts[config['known_projects_path']] = config['known_projects_sha256']
    require(config['inference_source_sha256'], 'Inference source hashes are missing')
    for name, component in components.items():
        for path_key, hash_key in [('model_path', 'model_sha256'), ('category_maps_path', 'category_maps_sha256')]:
            if path_key in component:
                relative, digest = component[path_key], component[hash_key]
                require(relative not in expected_artifacts or expected_artifacts[relative] == digest, f'Conflicting model SHA: {relative}')
                expected_artifacts[relative] = digest
        require('model_path' in component, f'Model path is missing: {name}')
        require(all(dep in components for dep in component.get('dependencies', [])), f'Unknown component dependency: {name}')
        if 'metadata_path' in component:
            metadata = read(component['metadata_path'])
            for key in ['model_sha256', 'category_maps_sha256', 'training_rows']:
                if key in component:
                    require(metadata[key] == component[key], f'Final component metadata differs: {name}/{key}')
    for relative, digest in expected_artifacts.items():
        check_hash(safe_path(ROOT, relative), digest)
    for variant in config['variants']:
        require(set(variant['gates']) == {'seen', 'unseen'}, f'Invalid gates: {variant["name"]}')
        for parts in variant['gates'].values():
            require(all(name in components and math.isfinite(weight) and weight >= 0 for name, weight in parts), 'Invalid release component or weight')
            require(math.isclose(sum(weight for _, weight in parts), 1.0), 'Release weights do not sum to one')
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in variant['metrics'].values()), 'Invalid release metric')
    with (DATA_DIR / SAMPLE_FILENAME).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        require(next(reader, None) == [ID, TARGET], 'Unexpected sample submission header')
        sample_rows = list(reader)
    require(all(len(row) == 2 for row in sample_rows), 'Unexpected sample submission columns')
    expected_ids = [row[0] for row in sample_rows]
    require(len(expected_ids) == EXPECTED_ROWS and len(set(expected_ids)) == EXPECTED_ROWS, 'Invalid sample ID count/uniqueness')
    submissions = {name: audit_submission(name, expected_ids, config_sha, sources['test']) for name in names}
    report_submissions = results['submissions']
    require(len(report_submissions) == 3 and {s['name'] for s in report_submissions} == EXPECTED_VARIANTS, 'Report submission variants differ')
    for item in report_submissions:
        require(item == submissions[item['name']], f'Report submission checks are stale: {item["name"]}')

    proof = read(PROOF)
    for key in ['all_cells_passed', 'all_submissions_reproduced_exactly', 'release_config_unchanged', 'verification_passed']:
        require(proof.get(key) is True, f'Notebook verification did not pass: {key}')
    require(proof['release_config_before_sha256'] == proof['release_config_after_sha256'] == config_sha, 'Notebook used another configuration')
    require(results['notebook_verification'] == proof, 'Regenerate report_round2.py after Notebook verification')
    require(set(proof['csv_sha256']) == set(proof['submission_sha256']) == EXPECTED_VARIANTS, 'Notebook proof omits a submission')
    for name, checks in submissions.items():
        item = proof['csv_sha256'][name]
        require(item['path'] == f'outputs/submission_{name}.csv', f'Notebook CSV path differs: {name}')
        require(item['unchanged'] is True and item['before_sha256'] == item['after_sha256'] == proof['submission_sha256'][name] == checks['sha256'], f'Notebook CSV proof differs: {name}')
    check_hash(ROOT / 'solution_round2.ipynb', proof['notebook_sha256'])
    notebook = read('solution_round2.ipynb')
    executable = [c for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    require(len(executable) == proof['executed_code_cells'] == proof['expected_executable_code_cells'] and executable, 'Notebook executed-cell count differs')
    require(all(c['execution_count'] is not None and not any(o['output_type'] == 'error' for o in c['outputs']) for c in executable), 'Notebook contains an unexecuted or failed cell')
    return config, original, v1, submissions, proof, expected_artifacts


def artifact_paths(config, expected_artifacts):
    required = {
        CONFIG, RESULTS, PROOF, 'requirements.txt', 'requirements-round2.txt',
        'README_round2.md', 'solution_round2.ipynb', 'reports/optimization_round2_report.md',
        'reports/round2_model_comparison.png', 'reports/run_manifest.json',
        'reports/data_quality.json', 'reports/feature_schema.json', 'reports/split_audit.json',
        'runs/v1/models/final_config.json', 'runs/v1/reports/results.json',
        'runs/v1/reports/final_training.json', 'runs/v1/outputs/submission.csv',
    } | set(expected_artifacts)
    for name in EXPECTED_VARIANTS:
        required.update({f'outputs/submission_{name}.csv', f'outputs/submission_{name}_checks.json'})
    for component in config['components'].values():
        if 'metadata_path' in component:
            required.add(component['metadata_path'])
    required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'src').glob('*.py'))
    required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'reports').glob('round2*') if path.is_file() and path.name != Path(RUN_MANIFEST).name)
    allowed = {'.json', '.csv', '.npy', '.md', '.png', '.py'}
    excluded_dirs = {'__pycache__', 'notebook_runtime', 'matplotlib_cache', 'catboost_info', 'logs', 'models'}
    # Scan only explicit output trees. Never recursively traverse runs or data/cache directories.
    for folder in [ROOT / 'outputs' / 'round2', ROOT / 'outputs' / 'experiments']:
        require(folder.is_dir(), f'Required experiment folder is missing: {folder}')
        for path in folder.rglob('*'):
            parts = path.relative_to(folder).parts
            if path.is_file() and path.suffix.lower() in allowed and not excluded_dirs.intersection(parts):
                required.add(path.relative_to(ROOT).as_posix())
    for branch in ['catboost', 'lightgbm', 'neighborhood']:
        manifest = f'outputs/round2/{branch}/candidates.json'
        required.add(manifest)
        for candidate in read(manifest):
            for key in ['valid_predictions', 'temporal_predictions', 'temporal_indices', 'metadata_path']:
                if candidate.get(key):
                    required.add(candidate[key])
    for relative in required:
        require(safe_path(ROOT, relative).is_file(), f'Artifact is missing: {relative}')
        require(not relative.startswith('runs/v2/'), 'Refusing recursive v2 archive inclusion')
    return sorted(required)


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_snapshot(base, expected):
    require(base.is_dir() and not base.is_symlink(), f'Invalid snapshot directory: {base}')
    actual_files = {path.relative_to(base).as_posix() for path in base.rglob('*') if path.is_file()}
    require(actual_files == set(expected), 'Existing v2 contains different, missing, or extra files; no overwrite is allowed')
    for relative, digest in expected.items():
        check_hash(safe_path(base, relative), digest)


def version_index(config, submissions, snapshot_sha):
    versions = read('runs/versions.json')
    entries = versions['versions']
    require(isinstance(entries, list) and sum(v['version'] == 'v1' for v in entries) == 1, 'Version index must preserve one v1 entry')
    require(len({v['version'] for v in entries}) == len(entries), 'Duplicate versions in existing index')
    additions = []
    for variant in config['variants']:
        name = variant['name']
        entry = {
            'version': name, 'label': variant['label'],
            'submission': f'runs/v2/outputs/submission_{name}.csv',
            'sha256': submissions[name]['sha256'], 'rows': EXPECTED_ROWS,
            'snapshot': 'runs/v2', 'snapshot_manifest_sha256': snapshot_sha,
            'release_config': 'runs/v2/' + CONFIG,
            'metrics': variant['metrics'], **variant['metrics'],
            'kaggle_score': None, 'status': 'completed',
        }
        previous = next((v for v in entries if v['version'] == name), None)
        require(previous is None or previous == entry, f'Existing version entry differs: {name}; no overwrite is allowed')
        if previous is None:
            additions.append(entry)
    return {**versions, 'versions': entries + additions, 'recommended_version': 'v2'}


def main():
    config, original, v1, submissions, proof, expected_artifacts = audit()
    paths = artifact_paths(config, expected_artifacts)
    artifacts = {relative: sha256(safe_path(ROOT, relative)) for relative in paths}
    require(all(artifacts[p] == h for p, h in expected_artifacts.items()), 'A configured artifact changed during the audit')
    run_manifest = {
        'version': 'v2', 'inputs_unchanged': True, 'input_sha256': original,
        'input_paths': {name: (DATA_DIR / name).relative_to(ROOT).as_posix() for name in original},
        'artifacts_sha256': artifacts, 'release_config_sha256': artifacts[CONFIG],
        'submissions': submissions, 'notebook_sha256': proof['notebook_sha256'],
        'notebook_proof_sha256': artifacts[PROOF], 'notebook_verified': True,
        'v1_snapshot_verified': v1,
        'original_data_not_duplicated': True, 'feature_caches_not_duplicated': True,
        'intermediate_training_models_and_logs_excluded': True,
        'source_policy': 'Only the four supplied local files; no external property prices or government answer lookup.',
    }
    generated = {RUN_MANIFEST: json_bytes(run_manifest)}
    files_sha = {**artifacts, RUN_MANIFEST: bytes_sha(generated[RUN_MANIFEST])}
    snapshot = {
        'version': 'v2', 'files_sha256': files_sha,
        'run_manifest_sha256': files_sha[RUN_MANIFEST],
        'manifest_self_excluded': True,
        'manifest_hash_recorded_in': 'runs/versions.json',
        'original_data_not_duplicated': True, 'feature_caches_not_duplicated': True,
    }
    generated['snapshot_manifest.json'] = json_bytes(snapshot)
    snapshot_sha = bytes_sha(generated['snapshot_manifest.json'])
    expected_snapshot = {**files_sha, 'snapshot_manifest.json': snapshot_sha}
    versions = version_index(config, submissions, snapshot_sha)
    runs = (ROOT / 'runs').resolve()
    destination = safe_path(ROOT, 'runs/v2')
    require(destination.parent == runs, 'Unexpected snapshot destination')
    if destination.exists():
        verify_snapshot(destination, expected_snapshot)
    else:
        runs.mkdir(parents=True, exist_ok=True)
        staging = safe_path(runs, f'.v2-staging-{uuid.uuid4().hex}')
        staging.mkdir()
        try:
            for relative in paths:
                target = safe_path(staging, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_path(ROOT, relative), target)
            for relative, content in generated.items():
                target = safe_path(staging, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            verify_snapshot(staging, expected_snapshot)
            for relative, digest in artifacts.items():
                check_hash(safe_path(ROOT, relative), digest)
            require(verify_v1() == v1, 'v1 changed while building v2')
            require(not destination.exists(), 'v2 appeared while building the snapshot; refusing overwrite')
            # Both fully resolved targets are direct children of the intended runs directory.
            require(staging.resolve().parent == runs and destination.parent == runs, 'Unsafe snapshot move')
            staging.rename(destination)
        finally:
            if staging.exists():
                resolved = staging.resolve()
                require(resolved.parent == runs and resolved.name.startswith('.v2-staging-') and resolved.is_relative_to(ROOT.resolve()), 'Unsafe staging cleanup')
                shutil.rmtree(resolved)
    verify_snapshot(destination, expected_snapshot)
    require(verify_v1() == v1, 'v1 verification changed before finalizing the index')
    atomic_write(ROOT / RUN_MANIFEST, generated[RUN_MANIFEST])
    atomic_write(ROOT / 'runs' / 'versions.json', json_bytes(versions))
    check_hash(ROOT / RUN_MANIFEST, files_sha[RUN_MANIFEST])
    require(read('runs/versions.json') == versions, 'Version index round-trip differs')
    print(json.dumps({
        'archive': str(destination), 'artifacts': len(files_sha),
        'snapshot_manifest_sha256': snapshot_sha, 'inputs_unchanged': True,
        'v1_unchanged': True, 'recommended_version': 'v2',
        'submission_sha256': {name: checks['sha256'] for name, checks in submissions.items()},
    }, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()