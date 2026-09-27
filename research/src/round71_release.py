"""Verify the R71 saved-vector payload and authored CSV, then expose v19 locally."""
from pathlib import Path
import argparse, csv, hashlib, json, os, time, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round71/final_export'
FINAL=ROOT/'outputs/round71/final'
AUTHOR=ROOT/'outputs/round71/csv_authoring'
CSV=ROOT/'outputs/submission_v19_trial.csv'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def locked(p):
    assert sha(p)==p.with_suffix('.sha256').read_text(encoding='ascii').strip(),str(p)
    return json.loads(p.read_text(encoding='utf8'))
def write(p,obj):
    with p.open('x',encoding='utf8',newline='\n') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
def run():
    plan=locked(ROOT/'outputs/round71/final_plan.json')
    proof=locked(FINAL/'trial_release.json');execution=locked(FINAL/'execution.json')
    authored=locked(AUTHOR/'validation.json')
    assert execution['exit_code']==0 and proof['status']=='final_vectors_and_export_payload_ready_no_CSV'
    assert plan['round']==proof['round']==authored['round']==71 and plan['proceed_final']
    assert proof['new_NN_fits']==proof['model_predict_calls']==proof['models_loaded']==0
    assert not proof['new_CSV'] and proof['query_rows']==47073 and proof['old_v17_v18_CSV_exact']
    assert proof['final_plan_sha256']==sha(ROOT/'outputs/round71/final_plan.json')
    for section in ('source_sha256','input_sha256','artifact_sha256'):
        for n,h in proof[section].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/plan['csv_writer'])==plan['csv_writer_sha256']==authored['source_sha256']
    assert authored['all_saved_CSV_cells_exact'] and authored['all_typed_cells_exact']
    assert not authored['rounding_performed_by_this_writer']
    payload_path=ROOT/proof['export_payload_path'];payload=locked(payload_path)
    assert sha(payload_path)==proof['export_payload_sha256']==authored['payload_sha256']
    prediction_path=ROOT/proof['prediction_path']
    assert sha(prediction_path)==proof['prediction_sha256']==payload['prediction_sha256']
    pred=np.load(prediction_path,allow_pickle=False)
    assert pred.dtype==np.float64 and pred.shape==(47073,) and np.isfinite(pred).all() and (pred>0).all()
    sample=ROOT/proof['sample_ID_path'];assert sha(sample)==proof['sample_ID_sha256']==payload['sample_ID_sha256']
    with sample.open(encoding='utf-8-sig',newline='') as f:ids=[r['編號'] for r in csv.DictReader(f)]
    assert ids==payload['ids'] and len(ids)==len(set(ids))==47073
    staged=AUTHOR/'staged_submission.csv'
    assert sha(staged)==authored['csv_sha256'] and staged.read_bytes()[:3]==b'\xef\xbb\xbf'
    with staged.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
    assert rows[0]==payload['headers']==['編號','總價元'] and len(rows)==47074
    assert all(len(r)==2 for r in rows) and [r[0] for r in rows[1:]]==ids
    assert all(r[1].isascii() and r[1].isdigit() and str(int(r[1]))==r[1] for r in rows[1:])
    prices=np.asarray([int(r[1]) for r in rows[1:]],dtype=np.int64)
    assert (prices>0).all() and prices.tolist()==payload['prices']
    # Independent comparison with original float64 prediction; CSV prices are never re-rounded.
    expected=np.maximum(np.rint(pred),1).astype(np.int64)
    assert np.array_equal(prices,expected)
    protected=proof['protected_before']
    assert protected==proof['protected_after']==authored['protected_before']==authored['protected_after']
    assert len(protected)==20 and all(sha(ROOT/n)==h for n,h in protected.items())
    assert {rel(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}==set(protected)-{'runs/versions.json'}
    assert not CSV.exists();os.link(staged,CSV)
    assert sha(CSV)==sha(staged)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    release={'round':71,'candidate':'R71','status':'experimental_v19_trial_generated','path':rel(CSV),'sha256':sha(CSV),
      'query_rows':47073,'fit_rows_in_reused_models':467221,'seed':42,'prior_numeric_count':144,
      'formula':proof['formula'],'new_NN_fits':0,'model_predict_calls':0,'reference_predict_calls':1,'spatial_predict_calls':1,
      'prediction_path':proof['prediction_path'],'prediction_sha256':proof['prediction_sha256'],
      'vector_release_path':rel(FINAL/'trial_release.json'),'vector_release_sha256':sha(FINAL/'trial_release.json'),
      'authoring_validation_path':rel(AUTHOR/'validation.json'),'authoring_validation_sha256':sha(AUTHOR/'validation.json'),
      'final_plan_sha256':sha(ROOT/'outputs/round71/final_plan.json'),'source_sha256':{rel(Path(__file__)):sha(Path(__file__)),plan['csv_writer']:plan['csv_writer_sha256']},
      'UTF8_BOM':True,'exact_headers':True,'all_sample_IDs_and_order_exact':True,'positive_int64_full_roundtrip':True,
      'original_unrounded_prediction_comparison_exact':True,'no_rounding_in_export_or_release':True,
      'protected_before':protected,'protected_after':{n:sha(ROOT/n) for n in protected},'old_CSV_unchanged':True,
      'prediction_min_TWD':float(pred.min()),'prediction_max_TWD':float(pred.max()),
      'prediction_changes_vs_controls':proof['prediction_changes_vs_controls'],'prediction_changes_are_not_errors':True,
      'private_mae':None,'public_mae':None,'uploaded':False,'kaggle_selection_changed':False,'registry_updated':False,'final_version_chosen':False}
    write(OUT/'release.json',release)
    print(json.dumps({'status':release['status'],'rows':47073,'sha256':release['sha256']},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
    assert not OUT.exists() and not CSV.exists(),'No overwrite or automatic retry'
    OUT.mkdir();tick=time.monotonic();code=0;failure=None
    try:run()
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
