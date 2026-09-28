"""Isolated fit-label extraction for fixed R97; no training or query scoring.

The inherited pickle necessarily materializes a complete source table here.
Only the verified fit indices are selected/exported. The model driver consumes
the sealed fit-only NPY and never deserializes the whole price table.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import traceback
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round97_inputs'
MANIFEST = ROOT / 'outputs/round90/preparation/manifest.json'
EXPECTED = '400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):h.update(block)
    return h.hexdigest()

def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def run():
    tick=time.monotonic()
    assert not OUT.exists(), 'Preserve old extraction; never overwrite'
    OUT.mkdir()
    started=datetime.now(timezone.utc).isoformat()
    source_sha=sha(Path(__file__))
    write(OUT/'started.json',dict(started_at_utc=started,source_sha256=source_sha,
        scope='fit-only export; original full pickle materialized only inside this extractor'))
    try:
        assert sha(MANIFEST)==EXPECTED
        m=json.loads(MANIFEST.read_text(encoding='utf-8'))
        bindings={}
        arrays={}
        for key in ('fit_indices','fit_source_cache_indices'):
            item=m['artifacts'][key];p=ROOT/item['path']
            assert sha(p)==item['sha256']
            a=np.load(p,allow_pickle=False)
            assert a.dtype==np.int64 and a.shape==(272117,) and (np.diff(a)>0).all()
            assert hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()==m['arrays'][key]['content_sha256']
            bindings[item['path']]=item['sha256'];arrays[key]=a
        provenance=m['inherited_provenance']
        raw=ROOT/provenance['raw_train_path']
        assert sha(raw)==provenance['raw_train_sha256']
        frame=pd.read_pickle(raw)
        assert len(frame)==455784 and frame.index.equals(pd.RangeIndex(len(frame)))
        assert arrays['fit_source_cache_indices'].max()<len(frame)
        y=frame.iloc[arrays['fit_source_cache_indices']]['總價元'].to_numpy(dtype=np.float64,copy=True)
        del frame
        assert y.shape==(272117,) and np.isfinite(y).all() and (y>0).all()
        with (OUT/'fit_y.npy').open('xb') as f:np.save(f,y,allow_pickle=False)
        restored=np.load(OUT/'fit_y.npy',allow_pickle=False)
        assert restored.dtype==y.dtype and restored.tobytes()==y.tobytes()
        assert sha(raw)==provenance['raw_train_sha256'] and sha(Path(__file__))==source_sha
        receipt=dict(status='fit_labels_sealed_no_training',format='R97_fit_labels_v1',
            fit_rows=len(y),dtype='float64',units='TWD',fit_y_path='outputs/round97_inputs/fit_y.npy',
            fit_y_sha256=sha(OUT/'fit_y.npy'),fit_y_values_sha256=hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),
            pure_X_manifest_path='outputs/round90/preparation/manifest.json',pure_X_manifest_sha256=EXPECTED,
            raw_train_path=provenance['raw_train_path'],raw_train_sha256=provenance['raw_train_sha256'],
            indices_sha256=bindings,fit_canonical_values_sha256=m['arrays']['fit_indices']['content_sha256'],
            fit_raw_cache_values_sha256=m['arrays']['fit_source_cache_indices']['content_sha256'],
            source_sha256=source_sha,extractor_materialized_full_source_pickle=True,
            query_labels_exported=False,query_statistics_computed=False,model_fits=0,
            test_read=False,valid_read=False,saved_labels_reload_exact=True,
            started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
        write(OUT/'manifest.json',receipt)
        write(OUT/'execution.json',dict(exit_code=0,elapsed_seconds=time.monotonic()-tick))
        print(json.dumps(receipt,ensure_ascii=False))
    except Exception:
        write(OUT/'execution.json',dict(exit_code=1,elapsed_seconds=time.monotonic()-tick,failure=traceback.format_exc()))
        raise

if __name__=='__main__':run()
