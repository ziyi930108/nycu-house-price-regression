"""R98 read-only artifact/date audit; no labels, models, fit, predict or scores."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/round98_artifact_inventory"
AUTHORITIES = {
 "outputs/round79/plan.json": "7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f",
 "outputs/round79/results.json": "7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6",
 "outputs/round71/results.json": "8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833",
 "outputs/round97/sealed_manifest.json": "429e5405652b4e9d54308d6f09d6722ab540694b13f92e2ad8577277b85893cb",
 "outputs/round90/preparation/manifest.json": "400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7",
}
FEATURES = "data_cache/train_features.pkl"
MAPPING = "outputs/round6/geography/development/train_source_cache_indices.npy"
STAGES = ("auxiliary", "development", "confirmation")
ORIGIN = pd.Timestamp("2000-01-01")
BRANCHES = ("N", "H", "U")
read_hashes = {}
authority_hashes = {}
def need(ok, message):
 if not ok: raise AssertionError(message)
def digest(path):
 h = hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def file(name, expected=None):
 need(name != "data_cache/train.pkl" and not (name.startswith("data_cache/") and any(x in Path(name).name.lower() for x in ("test", "valid"))) and "fit_y" not in name, "Forbidden label/test path")
 p=ROOT/name
 need(p.is_file(), "Missing "+name)
 h=digest(p)
 if expected is not None:need(h==expected,"File SHA differs: "+name)
 if name in read_hashes:need(read_hashes[name]==h,"Input mutated: "+name)
 read_hashes[name]=h
 return p
def read(name, expected=None):
 p=file(name,expected)
 d=json.loads(p.read_text(encoding="utf-8-sig"))
 authority_hashes[name]=read_hashes[name]
 return d
def values(a,dtype):
 return hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()
def same(a,b,label):
 need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label)
def array(name,expected,kind,authority):
 p=file(name,expected);a=np.load(p,allow_pickle=False)
 need(a.ndim==1,"Not vector "+name)
 dtype=np.dtype("int64" if kind=="index" else "float64")
 need(a.dtype==dtype and np.isfinite(a).all(),"Type/nonfinite "+name)
 if kind=="index":need(np.unique(a).size==len(a) and (a>=0).all(),"Invalid index "+name)
 else:need((a>=0).all(),"Negative prediction "+name)
 return a,{"path":name,"file_sha256":read_hashes[name],"dtype":str(a.dtype),"shape":list(a.shape),"values_sha256":values(a,"<i8" if kind=="index" else "<f8"),"authority":authority}
def boundary(s):return (pd.Timestamp(s)-ORIGIN).days
def date_summary(days):
 need(np.isfinite(days).all() and len(days)>0,"Empty/invalid dates")
 return {"rows":len(days),"first_date":str((ORIGIN+pd.Timedelta(days=int(days.min()))).date()),"last_date":str((ORIGIN+pd.Timedelta(days=int(days.max()))).date()),"trade_day_index_values_sha256":values(days,"<f8")}
def run(out):
 started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
 need(not (out/"results.json").exists(),"Refuse overwrite")
 docs={n:read(n,h) for n,h in AUTHORITIES.items()}
 p79=docs["outputs/round79/plan.json"];r79=docs["outputs/round79/results.json"];r71=docs["outputs/round71/results.json"]
 seal=docs["outputs/round97/sealed_manifest.json"];p90=docs["outputs/round90/preparation/manifest.json"]
 need(r79["plan_sha256"]==AUTHORITIES["outputs/round79/plan.json"],"R79 plan binding")
 # R46 branch identity is independently pinned in R79 inputs; its results/plan
 # additionally prove this authority's native origin. No MAE fields are read.
 r46=read("outputs/round46/results.json")
 p46=read("outputs/round46/plan.json",r46["plan_sha256"])
 p71=read("outputs/round71/plan.json",r71["machine_plan_sha256"])
 sources={}
 for d in [p79,r79,p46,r46,p71,r71,seal,p90]:
  for n,h in d["source_sha256"].items():
   need(n.startswith("src/") and n.endswith(".py"),"Unexpected source key "+n)
   if n in sources:need(sources[n]==h,"Conflicting historic source "+n)
   sources[n]=h
 for n,h in sources.items():file(n,h)
 schema=read("reports/feature_schema.json",p79["input_sha256"]["reports/feature_schema.json"])
 need(p90["input_sha256"][FEATURES]==p79["input_sha256"][FEATURES],"Feature cache binding")
 k,kr=array(MAPPING,p79["input_sha256"][MAPPING],"index",{"path":"outputs/round79/plan.json","field":"input_sha256"})
 need(len(k)==455255 and values(k,"<i8")==p90["canonical_source_indices_content_sha256"],"Canonical K identity")
 raw=pd.read_pickle(file(FEATURES,p79["input_sha256"][FEATURES]))
 need(len(raw)==455784 and list(raw)==schema["columns"],"Pure X source schema")
 need(not any(str(c).lower() in {"總價元","y","target","price","total_price"} for c in raw),"Target column in X")
 X=raw.iloc[k].reset_index(drop=True);del raw
 days=X.trade_day_index.to_numpy(dtype=np.float64)
 need(np.isnan(days).sum()==8,"Canonical invalid-date identity")
 blocks={};loaded={}
 for s in STAGES:
  start,end=p79["splits"][s];spec=p79["split_contract"][s]
  need(p71["split_contract"][s]==spec and p46["split_contract"][s]==spec,"Split contract differs")
  rr={46:r46["stages"][s],71:r71["stages"][s],79:r79["stages"][s]}
  for n,d in rr.items():
   need(d["dates"]==[start,end] and d["fit_rows"]==spec["fit_rows"] and d["query_rows"]==spec["query_rows"],"Stage origin/rows differ")
  need(rr[46]["query_labels_used_for_fit_or_predict"] is False and rr[71]["query_labels_used_only_by_scorer"] is True and rr[79]["query_labels_used_only_for_evaluation"] is True,"Origin query-label contract")
  axes={};idx={}
  for role in ("fit","query"):
   name=f"outputs/round79/{s}/{role}_indices.npy"
   a,rec=array(name,rr[79]["artifact_sha256"][name],"index",{"path":"outputs/round79/results.json","field":f"stages.{s}.artifact_sha256"})
   need(rec["values_sha256"]==spec[role+"_indices_content_sha256"],"Canonical index semantic hash")
   mask=np.isfinite(days)&((days<boundary(start)) if role=="fit" else ((days>=boundary(start))&(days<=boundary(end))))
   same(a,np.flatnonzero(mask).astype(np.int64),"Date-mask/order differs")
   for rn,sub in [(71,""),(46,"model/")]:
    other=f"outputs/round{rn}/{s}/{sub}{role}_indices.npy"
    aa,ar=array(other,rr[rn]["artifact_sha256"][other],"index",{"path":f"outputs/round{rn}/results.json","field":f"stages.{s}.artifact_sha256"})
    same(a,aa,"Cross-branch canonical alignment")
   idx[role]=a;axes[role+"_indices"]=rec
  need(np.intersect1d(idx["fit"],idx["query"]).size==0,"Fit/query overlap")
  paths={"N":p79["bindings"][s]["paths"]["nn"],"H":p79["bindings"][s]["paths"]["log"],"U":f"outputs/round79/{s}/unit_spatial.npy"}
  need(paths["N"]==f"outputs/round71/{s}/seed42/nn_spatial.npy" and paths["H"]==f"outputs/round46/{s}/recency_log_spatial.npy","Branch identity")
  preds={};records={}
  for b,rn in [("N",71),("H",46),("U",79)]:
   name=paths[b];expected=rr[rn]["artifact_sha256"][name]
   if b in ("N","H"):need(p79["input_sha256"][name]==expected,"R79 input branch binding")
   preds[b],records[b]=array(name,expected,"prediction",{"path":f"outputs/round{rn}/results.json","field":f"stages.{s}.artifact_sha256","R79_input_binding":b in ("N","H")})
   need(len(preds[b])==len(idx["query"]),"Prediction row count")
  name=f"outputs/round79/{s}/predictions.npy";h=rr[79]["artifact_sha256"][name]
  need(h==rr[79]["prediction_sha256"],"R79 control double binding")
  control,cr=array(name,h,"prediction",{"path":"outputs/round79/results.json","field":f"stages.{s}.artifact_sha256/prediction_sha256"})
  same(np.median(np.column_stack([preds[b] for b in BRANCHES]),axis=1),control,"Original control median differs")
  blocks[s]={"cutoff":start,"query_end":end,**axes,"fit_rows":len(idx["fit"]),"query_rows":len(idx["query"]),"fit_dates":date_summary(days[idx["fit"]]),"query_dates":date_summary(days[idx["query"]]),"branches":records,"control_v22":cr,"feature_path":FEATURES,"feature_row_axis":"raw_cache_then_canonical_mapping_then_query_indices","feature_file_sha256":read_hashes[FEATURES],"canonical_mapping":kr,"control_median_bitwise_exact":True,"model_origin":{"native_fit_before":start,"branch_query_index_exact_across_R46_R71_R79":True,"query_labels_fit_or_prediction":False,"fit_state_reloaded_or_refit_in_this_audit":False}}
  loaded[s]={"fit":idx["fit"],"query":idx["query"],"matrix":np.column_stack([preds[b] for b in BRANCHES])}
 need(seal["predictions_sealed"] and seal["all_fits_complete"] and seal["all_query_saved_model_and_state_reload_exact"],"R97 not sealed")
 need(not seal["query_labels_used_for_training_or_prediction"] and not seal["test_read"],"R97 scope")
 start,end=seal["cutoff"],seal["query_end"];idx={};axes={}
 for role in ("fit","query"):
  rec=seal["arrays"][role+"_indices"];a,ar=array(rec["path"],rec["sha256"],"index",{"path":"outputs/round97/sealed_manifest.json","field":"arrays."+role+"_indices"})
  need(ar["values_sha256"]==rec["values_sha256"],"R97 semantic axis")
  mask=np.isfinite(days)&((days<boundary(start)) if role=="fit" else ((days>=boundary(start))&(days<=boundary(end))))
  same(a,np.flatnonzero(mask).astype(np.int64),"R97 canonical temporal order")
  src=seal["arrays"][role+"_source_cache_indices"];rawidx,_=array(src["path"],src["sha256"],"index",{"path":"outputs/round97/sealed_manifest.json","field":"arrays."+role+"_source_cache_indices"})
  same(k[a],rawidx,"R97 raw/canonical mapping")
  need(values(a,"<i8")==p90["arrays"][role+"_indices"]["content_sha256"],"R90/R97 index identity")
  idx[role]=a;axes[role+"_indices"]=ar
 xrec=seal["frames"]["Xquery"];qX=pd.read_pickle(file(xrec["path"],xrec["sha256"]))
 need(xrec["sha256"]==p90["artifacts"]["Xquery"]["sha256"],"R90 feature identity")
 pd.testing.assert_frame_equal(qX,X.iloc[idx["query"]].reset_index(drop=True),check_exact=True)
 preds={};records={}
 for b,key in [("N","N"),("H","H"),("U","U22")]:
  r=seal["branches"][key]["spatial"];preds[b],records[b]=array(r["path"],r["sha256"],"prediction",{"path":"outputs/round97/sealed_manifest.json","field":f"branches.{key}.spatial"})
  need(records[b]["values_sha256"]==r["values_sha256"] and records[b]["shape"]==r["shape"],"R97 branch semantic identity")
 r=seal["predictions"]["v22"];control,cr=array(r["path"],r["sha256"],"prediction",{"path":"outputs/round97/sealed_manifest.json","field":"predictions.v22"})
 need(cr["values_sha256"]==r["values_sha256"],"R97 control semantic identity")
 same(np.median(np.column_stack([preds[b] for b in BRANCHES]),axis=1),control,"R97 control median")
 blocks["offset"]={"cutoff":start,"query_end":end,**axes,"fit_rows":len(idx["fit"]),"query_rows":len(idx["query"]),"fit_dates":date_summary(days[idx["fit"]]),"query_dates":date_summary(days[idx["query"]]),"branches":records,"control_v22":cr,"feature_path":xrec["path"],"feature_row_axis":"query_local_RangeIndex_exact_canonical_query_order","feature_file_sha256":xrec["sha256"],"canonical_feature_path":FEATURES,"canonical_mapping":kr,"control_median_bitwise_exact":True,"model_origin":{"native_fit_before":start,"seven_new_native_fits":seal["new_native_fits"],"query_labels_fit_or_prediction":False,"fit_state_reloaded_or_refit_in_this_audit":False}}
 loaded["offset"]={"fit":idx["fit"],"query":idx["query"],"matrix":np.column_stack([preds[b] for b in BRANCHES])}
 # Annual forward blocks must not overlap each other. The target offset overlaps
 # other query windows, but its meta-fit is cut strictly before its own origin.
 pair_overlaps={}
 for i,a in enumerate(STAGES):
  for b in STAGES[i+1:]:
   n=int(np.intersect1d(loaded[a]["query"],loaded[b]["query"]).size)
   need(n==0,"Annual forward block overlap");pair_overlaps[a+"__"+b]=n
 meta={}
 for target,sources_used in [("development",["auxiliary"]),("offset",["auxiliary","development"]),("confirmation",["auxiliary","development"])]:
  cutoff=blocks[target]["cutoff"];pieces=[];ii=[];pp=[]
  for s in sources_used:
   q=loaded[s]["query"];mask=days[q]<boundary(cutoff);pos=np.flatnonzero(mask).astype(np.int64);chosen=q[pos]
   need(len(chosen)>0 and boundary(blocks[s]["cutoff"])<boundary(cutoff),"Source origin not earlier")
   pieces.append({"block":s,"date_mask":{"kind":"trade_day_index_strictly_before_target_cutoff","cutoff":cutoff},"source_rows":len(q),"selected_rows":len(chosen),"all_source_rows_selected":bool(mask.all()),"selected_source_positions_values_sha256":values(pos,"<i8"),"canonical_indices_values_sha256":values(chosen,"<i8"),"raw_source_indices_values_sha256":values(k[chosen],"<i8"),"dates":date_summary(days[chosen]),"source_order_preserved":True})
   ii.append(chosen);pp.append(loaded[s]["matrix"][pos])
  mi=np.concatenate(ii);mp=np.concatenate(pp,axis=0)
  need(np.unique(mi).size==len(mi) and not np.intersect1d(mi,loaded[target]["query"]).size,"Meta overlaps target or duplicates")
  need(np.isin(mi,loaded[target]["fit"]).all(),"Meta rows outside target native fit")
  need((days[mi]<boundary(cutoff)).all(),"Future meta row")
  meta[target]={"cutoff":cutoff,"source_order":sources_used,"pieces":pieces,"rows":len(mi),"canonical_meta_indices_values_sha256":values(mi,"<i8"),"raw_meta_source_indices_values_sha256":values(k[mi],"<i8"),"prediction_matrix_values_sha256":values(mp,"<f8"),"prediction_matrix_shape":list(mp.shape),"branch_order":list(BRANCHES),"dates":date_summary(days[mi]),"target_query_overlap_rows":0,"all_meta_rows_in_target_native_fit":True,"duplicates":0,"row_sorting_applied":False,"labels_read":False}
 raw_source={"path":"data_cache/train.pkl","file_sha256":p79["input_sha256"]["data_cache/train.pkl"],"authority":"outputs/round79/plan.json.input_sha256","opened_or_rehashed":False}
 need(seal["raw_train_price_source_for_later_scorer"]["raw_train_sha256"]==raw_source["file_sha256"],"Raw identity declaration differs")
 for b in blocks.values():b["raw_train_source"]=raw_source;b["authority_sha256"]=dict(authority_hashes)
 result={"schema_version":1,"status":"passed","started_at_utc":started,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-tick,"script_sha256":digest(Path(__file__)),"blocks":blocks,"meta_fit":meta,"branch_order":list(BRANCHES),"canonical_rows":len(X),"invalid_date_rows":8,"canonical_mapping":kr,"raw_train_source":raw_source,"authority_sha256":authority_hashes,"verified_source_sha256":sources,"verified_input_sha256":read_hashes,"annual_query_pair_overlaps":pair_overlaps,"new_fits":0,"models_loaded":0,"model_predictions_computed":0,"scores_computed":0,"query_or_fit_labels_loaded":False,"metrics_fields_used":False,"test_read":False,"row_price_or_prediction_values_written":False,"limitations":["Not a blind holdout: model formulas were selected using repeatedly inspected historical periods.","Only existing saved predictions, pure X and index/date identities audited; no model/state deserialization.","All metadata files retain historical aggregate score fields, but this audit never accesses those fields.","Prediction values may be read for identity and control median checks; true price tables and targets are never opened."]}
 out.mkdir(parents=True,exist_ok=True)
 with (out/"results.json").open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({"status":"passed","elapsed_seconds":result["elapsed_seconds"],"source_files":len(sources),"verified_files":len(read_hashes),"meta_rows":{k:v["rows"] for k,v in meta.items()},"offset_development_prefix":meta["offset"]["pieces"][1]["selected_rows"]},ensure_ascii=False))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=DEFAULT_OUT);args=p.parse_args();run(args.out.resolve())
