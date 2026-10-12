"""Conditional single final R63 weighted-MAE prior NN; unchanged R60 priors, v17."""
from pathlib import Path
import argparse,contextlib,json,os,pickle,sys,time,traceback
import numpy as np
import pandas as pd
import round63_nn_experiment as experiment
import round48_final_branch as final48

ROOT=experiment.ROOT;OUT=ROOT/'outputs/round63/final';CSV=ROOT/'outputs/submission_v17_trial.csv'
old=final48.old;spatial=experiment.spatial
sha,read,write=experiment.sha,experiment.read,experiment.write
rel=experiment.rel
AUTHORITIES={
 'outputs/round53/final/trial_release.json':'89bf546437d5e11eead362241677c2c84eff3e210a353c177cf1fd6d42701092',
 'outputs/round55/final/trial_release.json':'e3b3f4333a4d8aa056728819cb2853e8c296f5d2dd5f03e1a2f067e612badb0d',
 'outputs/round60/final/trial_release.json':'aff1246b3ce746acced6f81044e166433bbf1d0ac989cc8a2e62dc32dbd0b591'}

def prepare_weights(prior_data,X,y,baseline,prior_evidence):
 expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million'}
 with np.load(prior_data,allow_pickle=False) as saved:
  assert set(saved.files)==expected
  arrays={k:saved[k].copy() for k in saved.files}
 assert arrays['train_num'].shape==(467221,144) and arrays['query_num'].shape==(47073,144)
 assert arrays['train_y_million'].dtype==np.float32 and arrays['train_y_million'].tobytes()==np.asarray(y/1e6,dtype=np.float32).tobytes()
 for key in ('train_price_factor_million','query_price_factor_million'):
  assert experiment.old.original.value_sha(arrays[key],'<f4')==baseline[key+'_sha256']
 weights64,valid,date_summary=experiment.date_weights._weights(experiment.date_weights._dates(X))
 assert weights64.shape==valid.shape==(467221,) and int(valid.sum())==467213 and int((~valid).sum())==8
 previous_dates=prior_evidence['prior_evidence']['full_fit_date_support']
 for key in ('reference_rows','valid_date_rows','neutral_date_rows','date_values_sha256','invalid_flags_sha256','out_of_range_flags_sha256','valid_mask_sha256','weights_sha256','max_valid_reference_trade_day_index'):
  assert date_summary[key]==previous_dates[key],key
 weights32=weights64.astype(np.float32)
 assert np.isfinite(weights32).all() and (weights32>0).all() and np.array_equal(weights32[~valid],np.ones(8,dtype=np.float32))
 v=weights32.astype(np.float64);total=float(v.sum(dtype=np.float64));squares=float(np.dot(v,v))
 summary={'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),'sum':total,'sum_squares':squares,'effective_sample_size':total*total/squares}
 assert np.isfinite(list(summary.values())).all() and abs(summary['mean']-1)<1e-7
 assert 0<summary['effective_sample_size']<=len(X)*(1+1e-12)
 path=OUT/'prepared.npz'
 with path.open('xb') as f:np.savez(f,**arrays,train_sample_weight=weights32)
 with np.load(path,allow_pickle=False) as saved:
  assert set(saved.files)==expected|{'train_sample_weight'}
  for key,value in arrays.items():assert saved[key].dtype==value.dtype and saved[key].shape==value.shape and saved[key].tobytes()==value.tobytes(),key
  assert saved['train_sample_weight'].dtype==np.float32 and saved['train_sample_weight'].tobytes()==weights32.tobytes()
 prep={'fit_rows':467221,'query_rows':47073,'numeric_count':144,'prepared_path':rel(path),'prepared_sha256':sha(path),
  'source_prepared_path':rel(prior_data),'source_prepared_sha256':sha(prior_data),'all_eight_R60_arrays_reused_without_changes':True,
  'exact_NPZ_keys':sorted(expected|{'train_sample_weight'}),'date_weight_summary_float64':date_summary,
  'train_sample_weight_sha256':experiment.old.original.value_sha(weights32,'<f4'),'train_sample_weight_summary':summary,
  'neutral_date_positions':np.flatnonzero(~valid).tolist(),'neutral_weights_exactly_one':True,'weights_use_fit_X_only':True,
  'weights_use_targets':False,'weights_use_query_dates':False,'float32_weights_renormalized':False,'batch_weights_renormalized':False,
  'encoder_refit':False,'prior_refit':False,'fit_labels_used_for_prior':True,'query_labels_used':False,
  'prior_preparation_path':'outputs/round60/final/prior_preparation.json','prior_preparation_sha256':sha(ROOT/'outputs/round60/final/prior_preparation.json')}
 write(OUT/'preparation.json',prep)
 return path,prep


def run():
 tick=time.monotonic();inputs={};docs=[]
 def bind(name,digest):
  path=ROOT/name;assert sha(path)==digest,name
  assert name not in inputs or inputs[name]==digest
  inputs[name]=digest;return path
 for name,h in AUTHORITIES.items():docs.append(read(bind(name,h)))
 p53,p55,p60=docs
 assert p60['round']==60 and p60['status']=='experimental_v16_trial_generated'
 assert p60['fit_rows']==467221 and p60['query_rows']==47073 and p60['saved_model_reload_exact']
 history=read(ROOT/'outputs/round63/results.json')
 final_plan=read(ROOT/'outputs/round63/final_plan.json')
 assert final_plan['proceed_to_final'] and final_plan['round']==63
 assert final_plan['results_sha256']==sha(ROOT/'outputs/round63/results.json')
 assert history['round']==63 and history['candidate']=='R63' and history['checked_stages']==list(experiment.STAGES)
 assert history['new_NN_fits']==3 and history['formula']==experiment.FORMULA
 assert history['status']=='three_periods_completed_no_adoption_decision'
 for name,h in history['source_sha256'].items():assert sha(ROOT/name)==h,name
 assert sha(experiment.TRAINER)==experiment.TRAINER_SHA
 source={**history['source_sha256'],rel(Path(__file__)):sha(Path(__file__)),
  rel(Path(final48.__file__)):sha(Path(final48.__file__))}
 known={}
 for doc in (p53,p55,p60):
  for section in ('input_sha256','artifact_sha256'):
   for name,h in doc[section].items():
    assert name not in known or known[name]==h,name
    known[name]=h
 def checked(name):return bind(name,known[name])
 def vector(name):return experiment.old.vector(checked(name),47073)
 protected={rel(p):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
 protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
 for name in ('outputs/round63/results.json','outputs/round63/final_plan.json'):
  bind(name,sha(ROOT/name))
 frames={s:pd.read_pickle(checked(f'data_cache/{s}_features.pkl')) for s in ('train','valid','test')}
 for split,n in (('train',455784),('valid',11994),('test',47073)):
  assert len(frames[split])==n and frames[split].index.equals(pd.RangeIndex(n))
  assert len(frames[split].columns)==87 and old.TARGET not in frames[split]
 indexes={}
 for name,n,size,identity in [('train',455784,455255,False),('valid',11994,11994,True),
   ('full',467249,467221,False),('test',47073,47073,True)]:
  a=np.load(checked(old.GEO+name+'_source_cache_indices.npy'),allow_pickle=False)
  assert a.dtype==np.int64 and a.shape==(size,) and (np.diff(a)>0).all() and a.min()>=0 and a.max()<n
  if identity:assert np.array_equal(a,np.arange(size,dtype=np.int64))
  indexes[name]=a
 kt,kv,kf,kq=[indexes[name] for name in ('train','valid','full','test')]
 X=pd.concat([frames['train'].iloc[kt],frames['valid'].iloc[kv]],ignore_index=True).iloc[kf].reset_index(drop=True)
 Xq=frames['test'].iloc[kq].reset_index(drop=True)
 labels={s:pd.read_pickle(checked(f'data_cache/{s}.pkl'))[old.TARGET].to_numpy(dtype=np.float64,copy=True) for s in ('train','valid')}
 assert len(labels['train'])==455784 and len(labels['valid'])==11994
 y=np.concatenate([labels['train'][kt],labels['valid'][kv]])[kf]
 assert y.shape==(467221,) and np.isfinite(y).all() and (y>0).all()
 target_sha=experiment.old.original.value_sha(y,'<f8')
 assert target_sha==p53['fit_target_values_sha256']==p55['fit_target_values_sha256']==p60['fit_target_values_sha256']
 del labels,frames
 geo,geoq=[pd.read_pickle(checked(old.GEO+n)) for n in ('full.pkl','test.pkl')]
 before={name:spatial.frame_sha(f) for name,f in [('X',X),('Xq',Xq),('geo',geo),('geoq',geoq)]}
 assert before==p53['input_frame_sha256']==p55['input_frame_sha256']==p60['input_frame_sha256']
 ind={}
 for name,n in [('fit',467221),('query',47073)]:
  a=np.load(checked(f'outputs/round53/final/{name}_indices.npy'),allow_pickle=False)
  other=np.load(checked(f'outputs/round55/final/{name}_indices.npy'),allow_pickle=False)
  recent=np.load(checked(f'outputs/round60/final/{name}_indices.npy'),allow_pickle=False)
  assert np.array_equal(a,recent)
  assert a.dtype==np.int64 and np.array_equal(a,np.arange(n,dtype=np.int64)) and np.array_equal(a,other)
  ind[name]=a;experiment.old.previous.save_array(OUT/f'{name}_indices.npy',a)
 prefix='outputs/round60/final/'
 baseline=json.loads(checked(prefix+'model/training.json').read_text(encoding='utf8'))
 prior_data=checked(prefix+'prepared.npz')
 prior_evidence=read(checked(prefix+'prior_preparation.json'))
 assert baseline['input_sha256']==sha(prior_data) and baseline['numeric_count']==144
 assert baseline['source_sha256']==experiment.PARENT_SHA and baseline['saved_model_reload_exact']
 assert baseline['fit_rows']==467221 and baseline['query_rows']==baseline['reload_rows']==47073
 assert baseline['configuration']['seed']==42 and baseline['configuration']['epochs']==30
 assert baseline['prediction_units']=='TWD' and prior_evidence==p60['prior_preparation']
 assert not prior_evidence['query_labels_used'] and not prior_evidence['encoder_refit']
 for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256','price_factor_million_summary'):
  assert baseline[key]==prior_evidence[key]
 area_native=vector('outputs/round55/final/model/predictions_twd.npy')
 area_meta=json.loads(checked('outputs/round55/final/model/training.json').read_text(encoding='utf8'))
 assert area_meta['fit_rows']==467221 and area_meta['query_rows']==47073 and area_meta['saved_model_reload_exact']
 assert area_meta['source_sha256']==experiment.old.TRAINER_SHA and area_meta['prediction_units']=='TWD'
 recency=vector(final48.RECENCY);unit=vector(final48.inherited.UNIT_BRANCH)
 v16=vector(prefix+'predictions.npy')
 v15=vector('outputs/round55/final/r54_predictions.npy');v13=vector('outputs/round53/final/predictions.npy')
 v5=vector(final48.inherited.V5)
 experiment.old.exact(np.median(np.column_stack([vector('outputs/round55/final/nn_mean_spatial.npy'),recency,unit]),axis=1),v15,'v15 complete median')
 experiment.old.exact(np.median(np.column_stack([vector(prefix+'nn_spatial.npy'),recency,unit]),axis=1),v16,'v16 complete median')
 sample=pd.read_csv(checked(rel(old.SAMPLE)),encoding='utf-8-sig',usecols=[old.ID],dtype={old.ID:str},keep_default_na=False)
 ids=sample[old.ID].tolist();assert len(ids)==47073 and sample[old.ID].is_unique and all(ids)
 for label,value in [('v16',v16),('v15',v15),('v13',v13),('v5',v5)]:
  final48.csv_check(ROOT/f'outputs/submission_{label}_trial.csv',ids,old.rounded(value))
 manifest=json.loads(checked(final48.inherited.GEO_MANIFEST).read_text(encoding='utf8'))
 newquery=pd.read_pickle(checked(manifest['features']['query']['path']))
 assert spatial.frame_sha(newquery)==manifest['features']['query']['frame_sha256'] and newquery.index.equals(Xq.index)
 with checked(old.REFERENCE).open('rb') as f:reference=pickle.load(f)
 with checked(final48.FINAL+'spatial_state.pkl').open('rb') as f:state=pickle.load(f)
 summary=json.loads(checked(final48.FINAL+'state_summary.json').read_text(encoding='utf8'))
 assert state['state_summary']==summary and summary['fit_rows']==467221 and not summary['query_used_in_build']
 assert summary['fit_X_sha256']==before['X'] and summary['fit_geo_sha256']==manifest['features']['fit']['frame_sha256']
 assert summary['parameters']==spatial.parameters() and reference['fit_report']['fitting_rows']==467221
 assert summary['existing_reference_residual_sha256']==experiment.old.original.value_sha(reference['reference_residual'],'<f8')
 raw,old_unit=vector(old.OLD_RAW),vector(old.OLD_UNIT)
 write(OUT/'input_contract.json',{'round':63,'formula':experiment.FORMULA,'new_NN_fits':1,
  'source_sha256':source,'input_sha256':inputs,'protected_sha256':protected,'query_labels_used':False,
  'reused_R54_area_native':True,'original_144_encoding_reused':True,'prior_refit':False,
  'original_eight_R60_arrays_reused':True,'weight_scope':'full final fit only; invalid dates neutral 1'})
 data,prep=prepare_weights(prior_data,X,y,baseline,prior_evidence)
 new_native,training=experiment.train(OUT,data,baseline,prep)
 mean=.5*area_native+.5*new_native;B=.75*(.5*raw+.5*mean)+.25*old_unit
 p,diag,rr=experiment.old.original.predict_reference(reference,Xq,B,variant='trend',return_diagnostics=True)
 branch,detail,sr=spatial.predict(state,Xq,newquery,B,p,diag)
 fallback=~detail['applied'].to_numpy(dtype=bool);experiment.old.exact(branch[fallback],p[fallback],'spatial fallback')
 candidate=np.median(np.column_stack([branch,recency,unit]),axis=1)
 for name,value in [('native_mean',mean),('base',B),('reference',p),('nn_spatial',branch),('predictions',candidate)]:
  assert value.dtype==np.float64 and value.shape==(47073,) and np.isfinite(value).all() and (value>0).all()
  experiment.old.previous.save_array(OUT/(name+'.npy'),value)
 for name,value in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
  with (OUT/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
 changes={}
 for label,value in [('v16',v16),('v15',v15),('v13',v13),('v5',v5)]:
  d=candidate-value;ix=np.argsort(-np.abs(d),kind='stable')[:12]
  changes[label]={'changed_unrounded_rows':int(np.count_nonzero(d)),
   'changed_rounded_rows':int(np.count_nonzero(old.rounded(candidate)!=old.rounded(value))),
   'mean_absolute_prediction_change_TWD':float(np.abs(d).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(d).max()),
   'top12':[{'position':int(i),'ID':ids[i],'candidate_TWD':float(candidate[i]),'control_TWD':float(value[i]),
    'signed_prediction_change_TWD':float(d[i]),'building_area':float(Xq.iloc[i]['building_area']) if pd.notna(Xq.iloc[i]['building_area']) else None} for i in ix]}
 write(OUT/'prediction_changes.json',{'changes':changes,'prediction_changes_are_not_errors':True,'test_labels_loaded':False,'no_row_adjustment_or_price_cap':True})
 assert before=={name:spatial.frame_sha(f) for name,f in [('X',X),('Xq',Xq),('geo',geo),('geoq',geoq)]}
 assert all(sha(ROOT/name)==h for name,h in source.items())
 assert all(sha(ROOT/name)==h for name,h in protected.items())
 prices=old.rounded(candidate);assert prices.dtype==np.int64 and (prices>0).all()
 staged=OUT/CSV.name
 pd.DataFrame({old.ID:ids,old.TARGET:prices}).to_csv(staged,index=False,encoding='utf-8-sig',mode='x')
 assert staged.read_bytes()[:3]==b'\xef\xbb\xbf'
 final48.csv_check(staged,ids,prices)
 assert not CSV.exists();os.link(staged,CSV)
 proof={'round':63,'status':'experimental_v17_trial_generated','formula':experiment.FORMULA,
  'fit_rows':467221,'query_rows':47073,'fit_target_values_sha256':target_sha,'input_frame_sha256':before,
  'source_sha256':source,'input_sha256':inputs,'new_NN_fits':1,'numeric_count':144,'seed':42,'epochs':30,
  'reused_R54_area_native':True,'base_encoder_refit':False,'prior_refit':False,
  'prior_preparation':prior_evidence,'sample_weight_preparation':prep,
  'training_configuration':training['configuration'],'loss_configuration_overrides':training['loss_configuration_overrides'],
  'train_sample_weight_sha256':training['train_sample_weight_sha256'],'train_sample_weight_summary':training['train_sample_weight_summary'],
  'fit_seconds':training['fit_seconds'],'saved_model_reload_exact':training['saved_model_reload_exact'],'reload_rows':training['reload_rows'],
  'reference_predict_calls':1,'spatial_predict_calls':1,'reference_report':rr,'spatial_report':sr,'new_spatial_state_builds':0,
  'v16_v15_v13_v5_CSV_exact':True,'old_CSV_unchanged':True,'test_labels_loaded':False,'sample_ID_order_verified':True,
  'positive_int64_roundtrip':True,'UTF8_BOM':True,'rounding':'max(rint(candidate),1), once',
  'new_CSV':True,'path':rel(CSV),'sha256':sha(CSV),'prediction_path':rel(OUT/'predictions.npy'),'prediction_sha256':sha(OUT/'predictions.npy'),
  'prediction_changes_are_not_errors':True,'prediction_changes_vs_controls':changes,
  'public_mae':None,'private_mae':None,'uploaded':False,'kaggle_selection_changed':False,'registry_updated':False,
  'artifact_sha256':{rel(f):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='execution.log'},
  'protected_before':protected,'protected_after':{name:sha(ROOT/name) for name in protected},
  'elapsed_seconds':time.monotonic()-tick,'completed_at_utc':experiment.old.now()}
 assert proof['protected_before']==proof['protected_after']
 write(OUT/'trial_release.json',proof)
 print(json.dumps({'status':proof['status'],'fit_seconds':proof['fit_seconds'],'sha256':proof['sha256']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:parser.error('--run is required; no implicit final execution')
 assert not OUT.exists() and not CSV.exists(),'Existing final output or CSV; do not overwrite'
 OUT.mkdir();tick=time.monotonic();code=0;failure=None
 write(OUT/'started.json',{'source_sha256':sha(Path(__file__)),'started_at_utc':experiment.old.now(),'automatic_retry':False})
 with (OUT/'execution.log').open('x',encoding='utf8') as f:
  with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
   try:run()
   except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
 event={'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'source_sha256':sha(Path(__file__))}
 write(OUT/'execution.json',event);print(json.dumps(event),flush=True);raise SystemExit(code)
