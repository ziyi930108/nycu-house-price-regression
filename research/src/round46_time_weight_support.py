from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
x=pd.read_pickle(root/'data_cache/train_features.pkl')
k=np.load(root/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
x=x.iloc[k].reset_index(drop=True)
r={'method':'fit-only time weights exp2(-(max_fit_trade_day-fit_trade_day)/365), normalized by fit mean; no prices used','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stages':{}}
for stage in ['development','confirmation','auxiliary']:
 p=root/'outputs/round39/evaluation'/stage/'fit_indices.npy'
 fi=np.load(p,allow_pickle=False)
 f=x.iloc[fi]
 d=f.trade_day_index.to_numpy(dtype=np.float64)
 assert np.isfinite(d).all()
 age=d.max()-d
 assert (age>=0).all()
 u=np.exp2(-age/365)
 w=u/u.mean()
 assert np.isfinite(w).all() and (w>0).all()
 shares={}
 for lo,hi,label in [(0,365,'latest_1year'),(365,730,'year_1_to_2'),(730,np.inf,'older_2years')]:
  mask=(age>=lo)&(age<hi)
  shares[label]={'rows':int(mask.sum()),'row_share':float(mask.mean()),'weight_share':float(w[mask].sum()/w.sum())}
 r['stages'][stage]={'rows':len(f),'finite_dates':bool(np.isfinite(d).all()),'invalid_date_flag_rows':int((f.trade_date_invalid>0).sum()),'out_of_range_flag_rows':int((f.trade_date_out_of_range>0).sum()),'min_day':float(d.min()),'max_day':float(d.max()),'min_weight':float(w.min()),'max_weight':float(w.max()),'mean_weight':float(w.mean()),'kish_weight_concentration_ratio':float(w.sum()**2/(len(w)*np.square(w).sum())),'age_bands':shares,'fit_indices_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
r['caution']='Kish ratio summarizes weight concentration only; not a measure of model effective sample size or future accuracy.'
out=root/'outputs/error_budget_20260926/round46_time_weight_support.json'
assert not out.exists()
out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r['stages'],ensure_ascii=False))
