from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
raw=pd.read_pickle(root/'data_cache/train.pkl')
k=np.load(root/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
y=raw['總價元'].to_numpy(dtype=np.float64)[k]
result={'method':'fit-only w=(1+y/1e6)/mean_fit(1+y/1e6); diagnostic only; no model call or coefficient search','stages':{}}
for stage in ['development','confirmation','auxiliary']:
 fi=np.load(root/'outputs/round39/evaluation'/stage/'fit_indices.npy',allow_pickle=False)
 u=1+y[fi]/1e6
 w=u/u.mean()
 ordered=np.sort(w)
 top=max(1,int(np.ceil(.01*len(w))))
 result['stages'][stage]={'rows':int(len(w)),'min_weight':float(w.min()),'max_weight':float(w.max()),'mean_weight':float(w.mean()),'effective_n_from_weight_concentration':float(w.sum()**2/np.square(w).sum()),'effective_n_fraction':float(w.sum()**2/np.square(w).sum()/len(w)),'top_1pct_rows':top,'top_1pct_share_of_weight':float(ordered[-top:].sum()/w.sum()),'fit_indices_sha256':hashlib.sha256((root/'outputs/round39/evaluation'/stage/'fit_indices.npy').read_bytes()).hexdigest()}
result['effective_n_caution']='Kish weight-concentration diagnostic only; correlated transactions and model complexity not reflected; not validation sample size.'
result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=root/'outputs/error_budget_20260926/weighted_log_concentration.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result['stages'],ensure_ascii=False))
