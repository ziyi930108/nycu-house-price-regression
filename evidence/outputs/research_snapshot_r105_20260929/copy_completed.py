"""Copy completed R105 source and aggregate evidence; never train or publish."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'deliverables/v22_github'
HERE = Path(__file__).resolve().parent
DEST = REPO / 'evidence/outputs/research_snapshot_r105_20260929'
NAV = ['README.md', 'HANDOFF.md', 'research/README.md', 'docs/RESEARCH_UPDATES_20260929_NATIVE_AVERAGE.md']
STATE_NAMES = {'encoder_state', 'ramp_state', 'lgb_category_state', 'category_state'}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def inspect(value):
    if isinstance(value, dict):
        assert not {'global_median', 'reference_global_median', 'full_fit_global_median', 'cells'}.intersection(value), 'Full prior state forbidden'
        for v in value.values(): inspect(v)
    elif isinstance(value, list):
        assert not (len(value) > 100 and all(isinstance(v, (int, float)) for v in value)), 'Row vector forbidden'
        for v in value: inspect(v)


def main():
    assert not DEST.exists(), 'Preserve snapshot receipt'
    for name in ['outputs/round105/execution.json', 'outputs/round105/evaluation/execution.json',
                 'outputs/round105_summary/execution.json', 'outputs/round105_launcher_20260929/execution.json']:
        assert read(ROOT/name)['exit_code'] == 0, name
    for name in ['reports/round105_report.md', 'outputs/round105_summary/receipt.json',
                 'reports/round105_next_direction_inventory.md']:
        assert (ROOT/name).is_file(), name
    baseline = read(REPO/'SOURCE_MANIFEST.json')
    for row in baseline['files']: assert sha(REPO/row['path']) == row['sha256'], row['path']
    paths = set(HERE.glob('*.py'))
    folders = ['round105', 'round105_summary', 'round105_evaluator_prefreeze',
               'round105_launcher_20260929', 'round105_results_review', 'round105_driver_prefreeze']
    for r in (105,):
        paths.update((ROOT/'src').glob(f'round{r}_*.py'))
        paths.update((ROOT/'src').glob(f'summarize_round{r}.py'))
        paths.update((ROOT/'reports').glob(f'round{r}_*'))
    for folder in folders: paths.update((ROOT/'outputs'/folder).rglob('*'))
    paths = sorted(p for p in paths if p.is_file() and p.suffix in {'.md', '.py', '.json', '.sha256', '.patch'}
                   and '__pycache__' not in p.parts and p.stem not in STATE_NAMES)
    for p in paths:
        assert not {'prefix_states', 'states', 'private', 'data'}.intersection(p.relative_to(ROOT).parts)
        if p.suffix == '.json': inspect(read(p))
        name = p.relative_to(ROOT).as_posix()
        assert not (REPO/(('research/' if name.startswith('src/') else 'evidence/')+name)).exists(), name
    copied = []; DEST.mkdir(parents=True)
    save(DEST/'baseline.json', {'files':baseline['files'], 'manifest_sha256':sha(REPO/'SOURCE_MANIFEST.json')})
    for name in NAV[:3]:
        q = DEST/'navigation_before'/name; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes((REPO/name).read_bytes())
    for p in paths:
        name = p.relative_to(ROOT).as_posix(); relative = ('research/' if name.startswith('src/') else 'evidence/')+name
        q = REPO/relative; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(p.read_bytes())
        assert sha(q) == sha(p)
        copied.append({'source':name, 'destination':relative, 'sha256':sha(p), 'bytes':p.stat().st_size})
    save(DEST/'copy_manifest.json', {'files':copied, 'copied_count':len(copied), 'copied_bytes':sum(v['bytes'] for v in copied),
         'excluded':['raw data', 'labels', 'arrays', 'models', 'full encoders/category/ramp/prior states'], 'new_model_fits':0})
    print(json.dumps({'copied_count':len(copied), 'copied_bytes':sum(v['bytes'] for v in copied)}))


if __name__ == '__main__': main()
