"""Small, import-safe guards for the frozen v25 inference draft."""
from pathlib import Path, PurePosixPath
import hashlib
import json

SELECTED_SHA = '36231cd17a6cbeafefeb62a7989d207fa20d6c5f7aaf99857e4d9addd25597c1'
NATIVE_KEYS = {'C_unit', 'C_log', 'C_raw', 'L_unit', 'N_area', 'N_prior'}

def need(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def safe_relative(root, rel):
    p = PurePosixPath(rel)
    need(isinstance(rel, str) and rel and '\\' not in rel and ':' not in rel
         and not p.is_absolute() and '..' not in p.parts and str(p) == rel,
         'Noncanonical relative artifact/source path')
    q = (Path(root) / rel).resolve()
    need(q.is_relative_to(Path(root).resolve()), 'Path escapes declared root')
    return q

def check_config(config, manifest):
    need(config['version'] == 'v25' and config['experiment'] == 'R107'
         and config['selected_submission'] == 'v25'
         and config['rows'] == 47073 and config['expected_csv_sha256'] == SELECTED_SHA
         and config['Private_score'] is None, 'Selected v25 identity differs')
    need(config['unit_weight'] == .75 and config['lgb_weight'] == .25
         and config['tree_threads'] == 4 and config['neural_batch_size'] == 4096,
         'Frozen inference arithmetic/settings differ')
    files = manifest['files']
    paths = [e['path'] for e in files]
    need(len(files) == 17 and len(set(paths)) == 17, 'Expected exactly 17 unique artifacts')
    need(sum(p.endswith(('.pt', '.cbm')) for p in paths) == 5
         and 'models/final/lightgbm_l1_unit63/model.txt' in paths,
         'Expected four trees and two neural checkpoints')
    for e in files:
        safe_relative(Path.cwd(), e['path'])
        need(isinstance(e['bytes'], int) and e['bytes'] > 0
             and len(e['sha256']) == 64
             and all(c in '0123456789abcdef' for c in e['sha256']), 'Invalid artifact identity')
    models = config['neural_models']
    need(set(models) == {'N_area', 'N_prior'}, 'Both neural roles required')
    for key, n in [('N_area',603), ('N_prior',144)]:
        s = models[key]
        need(s['numeric_count'] == n and len(s['cardinalities']) == 23
             and s['configuration']['hidden'] == [512,256,128]
             and s['configuration']['dropout'] == .1, 'Neural architecture differs')
        e = next((e for e in files if e['path'] == s['model_path']), None)
        need(e is not None and e['sha256'] == s['expected_model_sha256'], 'Neural manifest/spec differ')
    need(models['N_prior']['model_path'] == 'outputs/round107/final/model/model.pt'
         and models['N_prior']['configuration']['smooth_l1_beta_million'] == .1,
         'R107 prior checkpoint required')

def verify_closure(root, expected_sha=None):
    root = Path(root).resolve()
    path = root/'closure.json'
    actual = digest(path)
    if expected_sha is not None:
        need(actual == expected_sha, 'Reviewed source closure SHA differs')
    closure = read(path)
    paths = [e['path'] for e in closure['files']]
    need(len(paths) == len(set(paths)), 'Duplicate closure path')
    observed = {p.relative_to(root).as_posix() for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p != path}
    need(observed == set(paths), 'Source closure exact file set differs')
    for e in closure['files']:
        p = safe_relative(root, e['path'])
        need(p.is_file() and p.stat().st_size == e['bytes'] and digest(p) == e['sha256'],
             'Runtime source/config identity differs: '+e['path'])
    config = read(root/'config/v25.json')
    manifest_path = root/'artifacts/manifest.json'
    need(digest(manifest_path) == config['artifact_manifest_sha256'], 'Artifact manifest SHA differs')
    manifest = read(manifest_path)
    check_config(config, manifest)
    return config, manifest, actual

def verify_artifacts(root, manifest):
    root = Path(root).resolve()
    result = []
    for e in manifest['files']:
        p = safe_relative(root, e['path'])
        need(p.is_file() and p.stat().st_size == e['bytes'] and digest(p) == e['sha256'],
             'Private artifact identity differs: '+e['path'])
        result.append({'path':e['path'], 'sha256':e['sha256'], 'bytes':e['bytes']})
    return result

def require_new_output(path, source_root):
    path = Path(path).absolute()
    need(not path.exists(), 'Output directory already exists; use an exclusive new directory')
    need(not path.resolve().is_relative_to(Path(source_root).resolve()),
         'Output directory must be outside the frozen source closure')
    return path

def check_query_schema(columns, rows, query_ids, sample_columns, sample_ids):
    need('總價元' not in columns and not {'target','y'} & set(columns), 'Query labels forbidden')
    need(list(sample_columns) == ['編號','總價元'], 'Sample schema differs')
    need(rows == 47073 and len(query_ids) == rows and query_ids == sample_ids,
         'Test/sample row count or exact ID order differs')
    need(all(isinstance(v,str) and v.isascii() and v.isdigit() for v in query_ids)
         and len(set(query_ids)) == rows, 'Expected unique nonempty digit IDs')

def check_checkpoint_metadata(checkpoint, spec):
    need(set(checkpoint) == {'numeric_count','cardinalities','configuration','state_dict'},
         'Unexpected checkpoint keys')
    need(all(checkpoint[k] == spec[k] for k in ['numeric_count','cardinalities','configuration']),
         'Full frozen role-specific checkpoint configuration differs')
