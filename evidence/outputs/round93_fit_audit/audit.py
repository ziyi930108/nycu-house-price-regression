"""Read-only, fit-X counts and existing date/area weights; no label arrays or models."""
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np,pandas as pd
import round5_project_mask as identity
import round79_recency_unit as old
tick=time.monotonic()
sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
plan=json.loads((ROOT/"outputs/round79/plan.json").read_text(encoding="utf8"))
plan93="reports/round93_plan.md"
assert sha(plan93)=="ac3ced3aa70ea3bb2548e4f4d4476da4c5b2e9ba2bd4795a5a673ee97f884054"
paths=["data_cache/train_features.pkl","outputs/round6/geography/development/train_source_cache_indices.npy"]
for p in paths:assert sha(p)==plan["input_sha256"][p]
Xraw=pd.read_pickle(ROOT/paths[0]);ids=np.load(ROOT/paths[1],allow_pickle=False)
assert len(Xraw)==455784 and ids.shape==(455255,) and ids.dtype==np.int64
Xall=Xraw.iloc[ids].reset_index(drop=True)
records={}
for stage in ("development","confirmation","auxiliary"):
 b=plan["bindings"][stage];fitpath=b["unit"]["fit_indices"]
 assert sha(fitpath)==plan["input_sha256"][fitpath]
 fi=np.load(ROOT/fitpath,allow_pickle=False);X=Xall.iloc[fi].reset_index(drop=True)
 for col in ("address_city","address_district","project_name"):
  assert not identity._normal(X[col]).str.contains("|",regex=False).any(),"Unsafe project-key separator"
 key,missing=identity.project_keys(X)
 freq=key[~missing].value_counts()
 counts=key.map(freq).where(~missing,1).to_numpy(dtype=np.int64)
 q=1/np.sqrt(counts.astype(np.float64))
 path=b["prepared"];assert sha(path)==plan["input_sha256"][path]
 with np.load(ROOT/path,allow_pickle=False) as z:dates=z["train_sample_weight"].copy()
 assert dates.dtype==np.float32 and dates.shape==(len(fi),)
 w0=old.original.sample_weights(X)*dates.astype(np.float64)
 c=float(w0.sum(dtype=np.float64)/(w0*q).sum(dtype=np.float64))
 w1=w0*q*c
 assert np.isfinite(w1).all() and (w1>0).all()
 assert np.isclose(w0.sum(),w1.sum(),rtol=2e-15,atol=0)
 bygroup=pd.DataFrame({"key":key,"old":w0,"new":w1}).groupby("key",sort=False)[["old","new"]].sum()
 records[stage]={"fit_rows":len(fi),"named_projects":len(freq),"missing_name_rows":int(missing.sum()),
 "singleton_project_count":int(freq.eq(1).sum()),"largest_project_rows":int(freq.max()),
 "project_count_quantiles":{str(v):float(freq.quantile(v)) for v in (0,.25,.5,.75,.9,.99,1)},
 "row_group_count_quantiles":{str(v):float(np.quantile(counts,v)) for v in (0,.25,.5,.75,.9,.99,1)},
 "normalization_c":c,"old_total_weight":float(w0.sum()),"new_total_weight":float(w1.sum()),
 "row_multiplier_min":float((q*c).min()),"row_multiplier_max":float((q*c).max()),
 "maximum_group_pool_weight_share_before":float(bygroup.old.max()/w0.sum()),
 "maximum_group_pool_weight_share_after":float(bygroup.new.max()/w1.sum()),
 "counts_values_sha256":hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
 "q_values_sha256":old.values_sha256(q),"weights_values_sha256":old.values_sha256(w1),
 "fit_indices_file_sha256":sha(fitpath),"prepared_file_sha256":sha(path),
 "query_or_price_arrays_read":False}
result={"status":"passed","exit_code":0,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
 "elapsed_seconds":time.monotonic()-tick,"plan_sha256":sha(plan93),
 "source_sha256":{str(Path(__file__).relative_to(ROOT)).replace("\\","/"):sha(Path(__file__).relative_to(ROOT)),**old.source_hashes(),"src/round5_project_mask.py":sha("src/round5_project_mask.py")},
 "input_sha256":{p:sha(p) for p in paths},"stages":records,
 "new_model_fits":0,"query_scores":False,"hypothesis_changed_after_audit":False}
out=Path(__file__).with_name("results.json")
with out.open("x",encoding="utf8") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
out.with_suffix(".sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest()+"\n")
print(json.dumps({"status":result["status"],"seconds":result["elapsed_seconds"],"stages":records}))
