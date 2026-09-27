"""R81 saved-baseline native parameter compatibility: one load, zero fit/predict."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round81_preflight.json'
HELPER_SHA = '8ce19dbc9800737d8de394a2bf266f6514cadd9d738a3dec1944434803b2ab3c'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run():
    assert not OUT.exists() and not OUT.with_suffix('.sha256').exists(), 'Never repeat or overwrite this preflight'
    started = time.monotonic()
    record = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'source_path': 'src/round81_preflight.py',
        'source_sha256': sha(__file__), 'helper_sha256': HELPER_SHA,
        'new_fits': 0, 'predict_calls': 0, 'models_loaded': 0,
        'real_features_labels_or_NPZ_read': False, 'source_modified': False, 'automatic_retry': False}
    code = 1
    try:
        hp = ROOT / 'src/round81_recency_lgb.py'
        assert sha(hp) == HELPER_SHA
        cp = ROOT / 'runs/v1/models/final_config.json'
        mp = ROOT / 'outputs/round3/baseline/development/lightgbm_l1_unit63.json'
        config = json.loads(cp.read_text(encoding='utf-8-sig'))
        meta = json.loads(mp.read_text(encoding='utf-8-sig'))
        model_path = ROOT / meta['model_path']
        assert model_path.resolve() == (ROOT / 'outputs/round3/baseline/development/lightgbm_unit63.txt').resolve()
        model_sha = sha(model_path)
        assert model_sha == meta['model_sha256'], 'Saved baseline model identity differs'
        record.update(model_path=model_path.relative_to(ROOT).as_posix(), model_sha256=model_sha,
            input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (cp, mp)})
        record['input_sha256'][record['model_path']] = model_sha
        columns = config['feature_columns']
        assert len(columns) == len(set(columns)) == 87 and len(config['categorical_columns']) == 21
        state = {'feature_columns': columns, 'safe_feature_names': {c: f'f{i:03d}' for i, c in enumerate(columns)}}
        import lightgbm as lgb
        import round81_recency_lgb as helper
        assert helper.parameters() == meta['params']
        record['method_source_sha256'] = helper.source_hashes()
        tick = time.monotonic()
        model = lgb.Booster(model_file=str(model_path))
        record['model_load_seconds'] = time.monotonic() - tick
        record['models_loaded'] = 1
        record['loaded_native_parameters'] = dict(model.params)
        record['check_model_result'] = helper.check_model(model, state)
        record['lightgbm_version'] = lgb.__version__
        record['status'] = 'passed'
        code = 0
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    record.update(exit_code=code, elapsed_seconds=time.monotonic() - started,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    with OUT.open('x', encoding='utf8', newline='\n') as f:
        json.dump(record, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    digest = sha(OUT)
    with OUT.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(digest + '\n')
    print(json.dumps({'status': record['status'], 'exit_code': code, 'elapsed_seconds': record['elapsed_seconds'],
        'path': OUT.relative_to(ROOT).as_posix(), 'sha256': digest, 'error': record.get('error')}, ensure_ascii=False), flush=True)
    return code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
