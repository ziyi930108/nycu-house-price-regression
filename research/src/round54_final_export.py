"""Export the verified R54 full branch from the shared R55 final fit; no new fit."""
from pathlib import Path
import json,os,time
import numpy as np
import pandas as pd
import round48_final_branch as common
ROOT=common.ROOT;OUT=ROOT/'outputs/round54/final_export';CSV=ROOT/'outputs/submission_v15_trial.csv'
PARENT=ROOT/'outputs/round55/final/trial_release.json'
sha,write=common.sha,common.write
rel=lambda p:Path(p).relative_to(ROOT).as_posix()

def run():
 tick=time.monotonic()
 assert PARENT.is_file() and not OUT.exists() and not CSV.exists(),'Parent not ready or destination exists'
 p=json.loads(PARENT.read_text(encoding='utf8'))
 r=json.loads((ROOT/'outputs/round54/results.json').read_text(encoding='utf8'))
 assert p['status']=='experimental_v14_trial_generated' and p['fit_rows']==467221 and p['query_rows']==47073
 assert p['saved_model_reload_exact'] and p['reload_rows']==47073 and p['new_NN_fits']==1
 assert r['checked_stages']==['development','confirmation','auxiliary'] and r['status']=='three_periods_completed_no_adoption_decision'
 for name,h in r['source_sha256'].items():assert p['source_sha256'][name]==h==sha(ROOT/name),name
 protected={rel(f):sha(f) for f in (ROOT/'outputs').glob('submission_v*.csv')}
 def vector(name):
  path=ROOT/name
  assert sha(path)==p['artifact_sha256'][name],name
  a=np.load(path,allow_pickle=False)
  assert a.dtype==np.float64 and a.shape==(47073,) and np.isfinite(a).all() and (a>0).all()
  return a
 parent_dir='outputs/round55/final/'
 candidate=vector(parent_dir+'r54_predictions.npy')
 nn=vector(parent_dir+'nn_mean_spatial.npy')
 refs=['outputs/round46/trial/recency_log_spatial.npy','outputs/round40/trial/unit_spatial.npy']
 for name in refs:assert sha(ROOT/name)==p['input_sha256'][name]
 log,unit=[np.load(ROOT/name,allow_pickle=False) for name in refs]
 assert np.array_equal(candidate,np.median(np.column_stack([nn,log,unit]),axis=1))
 v13name='outputs/round53/final/predictions.npy'
 assert sha(ROOT/v13name)==p['input_sha256'][v13name]
 v13=np.load(ROOT/v13name,allow_pickle=False)
 v14=vector(parent_dir+'predictions.npy')
 assert np.array_equal(v14,.5*v13+.5*candidate)
 qpath=ROOT/(parent_dir+'query_indices.npy')
 assert sha(qpath)==p['artifact_sha256'][rel(qpath)] and np.array_equal(np.load(qpath,allow_pickle=False),np.arange(47073,dtype=np.int64))
 sample=common.old.SAMPLE
 assert sha(sample)==p['input_sha256'][rel(sample)]
 ids=pd.read_csv(sample,encoding='utf-8-sig',usecols=[common.old.ID],dtype={common.old.ID:str})[common.old.ID]
 assert len(ids)==47073 and ids.is_unique and ids.notna().all()
 rounded=common.old.rounded(candidate)
 OUT.mkdir(parents=True);staged=OUT/CSV.name
 frame=pd.DataFrame({common.old.ID:ids,common.old.TARGET:rounded})
 frame.to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
 common.csv_check(staged,ids.tolist(),rounded)
 assert staged.read_bytes()[:3]==b'\xef\xbb\xbf' and not CSV.exists()
 os.link(staged,CSV)
 with (OUT/'predictions.npy').open('xb') as f:np.save(f,candidate,allow_pickle=False)
 changes={}
 for key,a in [('v13',v13),('v14',v14)]:
  diff=candidate-a
  changes[key]={'changed_rounded_rows':int(np.count_nonzero(rounded!=common.old.rounded(a))),
   'mean_absolute_prediction_change_TWD':float(np.abs(diff).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(diff).max())}
 detail=json.loads((ROOT/(parent_dir+'prediction_changes.json')).read_text(encoding='utf8'))
 assert sha(ROOT/(parent_dir+'prediction_changes.json'))==p['artifact_sha256'][parent_dir+'prediction_changes.json']
 large=[{'ID':a['ID'],'pure_X':a['pure_X'],'v15_TWD':a['R54_final_TWD'],'v14_TWD':a['candidate_TWD'],
    'v13_TWD':a['v13_TWD'],'true_price_unknown':True} for a in detail['requested_94345_m2_rows']]
 proof={'round':54,'status':'experimental_v15_full_R54_generated','path':rel(CSV),'sha256':sha(CSV),
   'parent_final_release':rel(PARENT),'parent_final_release_sha256':sha(PARENT),
   'historical_results':'outputs/round54/results.json','historical_results_sha256':sha(ROOT/'outputs/round54/results.json'),
   'source_sha256':{rel(Path(__file__)):sha(Path(__file__))},'formula':r['formula'],'source_recipe_exact':True,
   'fit_rows':467221,'query_rows':47073,'new_fits_for_export':0,'shared_final_fit_seconds':p['fit_seconds'],
   'shared_final_model':'outputs/round55/final/model/model.pt','shared_final_numeric_count':p['numeric_count'],
   'saved_model_reload_exact':True,'sample_ID_order_verified':True,'positive_int64_roundtrip':True,'UTF8_BOM':True,
   'prediction_path':rel(OUT/'predictions.npy'),'prediction_sha256':sha(OUT/'predictions.npy'),
   'prediction_changes_vs_controls':changes,'large_area_case':large,'prediction_changes_are_not_errors':True,
   'test_labels_loaded':False,'public_mae':None,'private_mae':None,'uploaded':False,'kaggle_selection_changed':False,
   'protected_before':protected,'protected_after':{name:sha(ROOT/name) for name in protected},
   'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':common.old.stamp()}
 assert proof['protected_before']==proof['protected_after']
 write(OUT/'release.json',proof)
 print(json.dumps({k:proof[k] for k in ['status','sha256','new_fits_for_export','elapsed_seconds','prediction_changes_vs_controls','large_area_case']},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
