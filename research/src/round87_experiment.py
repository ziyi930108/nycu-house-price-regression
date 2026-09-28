"""R87 fixed equal native mixture, reused chronological states; CPU only, zero fits."""
from pathlib import Path
import argparse, contextlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round79_recency_unit as previous_model

ROOT=old.ROOT
OUT=ROOT/"outputs/round87"
DOCUMENT=ROOT/"reports/round87_plan.md"
STAGES=old.STAGES
sha,read,write,rel,spatial=old.sha,old.locked,old.write,old.rel,old.spatial
FORMULA="median(saved_NN,saved_log,spatial(reference(.375*R79_native_TWD+.375*R85_native_TWD+.25*saved_LGB_unit_TWD)))"

def exact(a,b,label):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label

def vector(path,n):
    a=np.load(ROOT/path,allow_pickle=False)
    assert a.dtype==np.float64 and a.shape==(n,) and np.isfinite(a).all() and (a>=0).all(),path
    return a

def run(plan_sha):
    tick=time.monotonic()
    assert sha(DOCUMENT)==plan_sha
    p79=read(ROOT/"outputs/round79/plan.json")
    r79=read(ROOT/"outputs/round79/results.json")
    r83=read(ROOT/"outputs/round83/results.json")
    r85=read(ROOT/"outputs/round85/results.json")
    assert r85["status"]=="three_periods_completed_no_adoption_decision" and r85["new_fits"]==3
    assert r85["checked_stages"]==list(STAGES)
    assert r79["plan_sha256"]==sha(ROOT/"outputs/round79/plan.json")
    assert r79["checked_stages"]==r83["checked_stages"]==list(STAGES)
    inputs=dict(p79["input_sha256"])
    source={**p79["source_sha256"],**r85["source_sha256"],rel(Path(__file__)):sha(Path(__file__))}
    for name in ("outputs/round79/plan.json","outputs/round79/results.json","outputs/round83/results.json","outputs/round85/results.json"):
        inputs[name]=sha(ROOT/name)
    for s in STAGES:
        for r,record in ((79,r79),(83,r83),(85,r85)):
            st=record["stages"][s]
            for name,digest in st["artifact_sha256"].items():
                if name.endswith(("predictions.npy","fit_indices.npy","query_indices.npy","base.npy","reference.npy","unit_spatial.npy","metadata.json","native_predictions.npy","date_weights.npy","pool_weights.npy")):
                    inputs[name.replace("\\","/")]=digest
        for suffix in ("model/metadata.json","model/feature_state.json"):
            path=f"outputs/round79/{s}/{suffix}"
            inputs[path]=r79["stages"][s]["artifact_sha256"][path]
    for name,digest in {**inputs,**source}.items():
        assert sha(ROOT/name)==digest,"Dependency changed: "+name
    protected={rel(p):sha(p) for p in sorted((ROOT/"outputs").glob("submission_v*.csv"))}
    for name in ("runs/versions.json","outputs/current_private_candidate.json","PRIVATE_SELECTION.md","CURRENT_STATUS.md","outputs/round84/decision.json","outputs/round79/selection/decision.json","outputs/v22_submission_focus_20260928/user_selection.json"):
        protected[name]=sha(ROOT/name)
    plan={"round":87,"candidate":"R87","stage_order":list(STAGES),"planned_new_fits":0,
          "primary_control":"v22","secondary_control":"v24","formula":FORMULA,
          "parameters":"no new model; unchanged saved R79 and R85","text_plan_sha256":plan_sha,
          "source_sha256":source,"input_sha256":inputs,"bindings":p79["bindings"],
          "splits":p79["splits"],"split_contract":p79["split_contract"],
          "target_group_weights":p79["target_group_weights"],"protected_before":protected,
          "fresh_blind_validation":False,"adaptive_after_R85_results":True,"Public_feedback_used_for_tuning":False,
          "query_labels_used_only_for_evaluation":True,"new_CSV":False,"new_test_predictions":False,
          "new_NN_fits":0,"new_reference_or_spatial_states":0,"automatic_adoption":False}
    write(OUT/"plan.json",plan)
    Xall,yall=old.previous.load_training_rows(p79)
    assert len(Xall)==455255
    result={k:v for k,v in plan.items() if k not in ("bindings","parameters")}
    result.update(status="running",checked_stages=[],stages={},new_fits=0,plan_sha256=sha(OUT/"plan.json"),
                  runtime={"numpy":np.__version__,"pandas":pd.__version__},models_loaded=0,model_predict_calls=0,
                  reference_predict_calls=0,spatial_predict_calls=0)
    for s in STAGES:
        start=time.monotonic();folder=OUT/s;folder.mkdir()
        bnd=p79["bindings"][s];b=bnd["unit"];spec=p79["split_contract"][s]
        indices={}
        for k in ("fit","query"):
            v=np.load(ROOT/b[k+"_indices"],allow_pickle=False)
            assert v.dtype==np.int64 and v.shape==(spec[k+"_rows"],) and (np.diff(v)>0).all()
            assert old.original.value_sha(v,"<i8")==spec[k+"_indices_content_sha256"]
            assert v.min()>=0 and v.max()<len(Xall)
            indices[k]=v;old.previous.save_array(folder/(k+"_indices.npy"),v)
            for r in (79,83,85):
                exact(v,np.load(ROOT/f"outputs/round{r}/{s}/{k}_indices.npy",allow_pickle=False),"Control indices")
        fi,qi=indices["fit"],indices["query"]
        assert not np.intersect1d(fi,qi).size
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
        yfit=yall[fi].copy();n=len(qi)
        cutoff=(pd.Timestamp(p79["splits"][s][0])-pd.Timestamp("2000-01-01")).days
        end=(pd.Timestamp(p79["splits"][s][1])-pd.Timestamp("2000-01-01")).days
        assert np.isfinite(X.trade_day_index).all() and (X.trade_day_index<cutoff).all()
        assert Xq.trade_day_index.between(cutoff,end).all()
        frames={"fit_X":spatial.frame_sha(X),"query_X":spatial.frame_sha(Xq)}
        assert frames==r79["stages"][s]["original_frame_sha256"]
        metadata79=read(ROOT/f"outputs/round79/{s}/model/metadata.json")
        assert frames==metadata79["original_frame_sha256"]
        assert previous_model.values_sha256(yfit)==metadata79["fit_target_values_sha256"]
        with np.load(ROOT/bnd["prepared"],allow_pickle=False) as z:
            w=z["train_sample_weight"].copy()
            exact(z["train_y_million"],np.asarray(yfit/1e6,dtype=np.float32),"Fit labels match prepared")
        prep=bnd["preparation"]
        assert w.dtype==np.float32 and w.shape==(len(fi),) and np.isfinite(w).all() and (w>0).all()
        assert old.original.value_sha(w,"<f4")==prep["train_sample_weight_sha256"]==metadata79["fit_date_weights_sha256"]
        ds=prep["date_weight_summary_float64"]
        assert ds["weight_rules"]["half_life_days"]==365 and not ds["weight_rules"]["weights_use_query"] and not ds["weight_rules"]["weights_use_targets"]
        for col,key in (("trade_day_index","date_values_sha256"),("trade_date_invalid","invalid_flags_sha256"),("trade_date_out_of_range","out_of_range_flags_sha256")):
            assert previous_model.values_sha256(X[col].to_numpy(dtype=np.float64))==ds[key]
        geo=pd.read_pickle(ROOT/b["query_geo"])
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo)==b["manifest"]["features"]["query"]["frame_sha256"]
        with (ROOT/b["state"]).open("rb") as f: state_spatial=pickle.load(f)
        ss=read(ROOT/b["state_summary"])
        assert state_spatial["state_summary"]==ss and state_spatial["parameters"]==spatial.parameters()
        assert ss["fit_X_sha256"]==frames["fit_X"] and not ss["query_used_in_build"]
        assert ss["fit_geo_sha256"]==b["manifest"]["features"]["fit"]["frame_sha256"]
        with (ROOT/b["reference"]).open("rb") as f: reference=pickle.load(f)
        assert reference["fit_report"]["fitting_rows"]==len(fi)
        assert previous_model.values_sha256(reference["reference_residual"])==ss["existing_reference_residual_sha256"]
        fixed={k:vector(bnd["paths"][k],n) for k in ("nn","log","v11","v12","v5")}
        fixed["v22"]=vector(f"outputs/round79/{s}/predictions.npy",n)
        fixed["v24"]=vector(f"outputs/round83/{s}/predictions.npy",n)
        lgb=vector(b["old_unit"],n)
        native79=vector(f"outputs/round79/{s}/model/native_predictions.npy",n)
        base79=.75*native79+.25*lgb
        exact(base79,vector(f"outputs/round79/{s}/base.npy",n),"v22 base")
        pr,diag,_=old.original.predict_reference(reference,Xq,base79,variant="trend",return_diagnostics=True)
        ub,detail,_=spatial.predict(state_spatial,Xq,geo,base79,pr,diag)
        exact(pr,vector(f"outputs/round79/{s}/reference.npy",n),"v22 reference")
        exact(ub,vector(f"outputs/round79/{s}/unit_spatial.npy",n),"v22 spatial")
        exact(np.median(np.column_stack([fixed["nn"],fixed["log"],ub]),axis=1),fixed["v22"],"v22 median")
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p79["splits"][s][0],p79["target_group_weights"])
        metrics={k:score(fixed[k]) for k in ("v11","v12","v5","v22","v24")}
        for k in ("v11","v12","v5"):assert metrics[k]==r79["stages"][s]["metrics"][k]
        assert metrics["v22"]==r79["stages"][s]["metrics"]["R79"]
        assert metrics["v24"]==r83["stages"][s]["metrics"]["R83"]
        native85=vector(f"outputs/round85/{s}/model/native_predictions.npy",n)
        meta85=read(ROOT/f"outputs/round85/{s}/model/metadata.json")
        assert meta85["original_frame_sha256"]==frames
        assert meta85["fit_target_values_sha256"]==metadata79["fit_target_values_sha256"]
        assert meta85["fit_date_weights_sha256"]==metadata79["fit_date_weights_sha256"]
        assert meta85["reload_all_query_rows_verified"] and meta85["reload_raw_exact"] and meta85["reload_TWD_exact"]
        assert meta85["target_mode"]=="sqrt_area" and meta85["prediction_units"]=="TWD"
        native=.5*native79+.5*native85
        base=.375*native79+.375*native85+.25*lgb
        old.previous.save_array(folder/"mean_native.npy",native)
        pr,diag,rr=old.original.predict_reference(reference,Xq,base,variant="trend",return_diagnostics=True)
        branch,detail,sr=spatial.predict(state_spatial,Xq,geo,base,pr,diag)
        fallback=~detail["applied"].to_numpy(dtype=bool)
        exact(branch[fallback],pr[fallback],"Spatial fallback")
        pred=np.median(np.column_stack([fixed["nn"],fixed["log"],branch]),axis=1)
        assert pred.dtype==np.float64 and pred.shape==(n,) and np.isfinite(pred).all() and (pred>0).all()
        metrics["R87"]=score(pred)
        for k,v in (("base",base),("reference",pr),("unit_spatial",branch),("predictions",pred)):
            old.previous.save_array(folder/(k+".npy"),v)
        for k,v in (("reference_diagnostics",diag),("spatial_diagnostics",detail)):
            with (folder/(k+".pkl")).open("xb") as f:pickle.dump(v,f,protocol=5)
        assert frames=={"fit_X":spatial.frame_sha(X),"query_X":spatial.frame_sha(Xq)}
        diff=old.previous.prior.log_run.differences
        item={"stage":s,"status":"completed","fit_rows":len(fi),"query_rows":n,"dates":p79["splits"][s],
              "original_frame_sha256":frames,"metrics":metrics,"differences":{k:diff(metrics[k],metrics["R87"]) for k in metrics if k!="R87"},
              "unit_branch_metrics":{k:score(v) for k,v in (("native",native),("base",base),("reference",pr),("spatial",branch))},
              "reference_report":rr,"spatial_report":sr,"controls_full_metrics_exact":True,
              "v22_base_reference_spatial_median_full_vectors_exact":True,"query_labels_used_only_for_evaluation":True,
              "parent_native_reload_evidence_verified":True,"independent_model_reload_performed":False,
              "date_weight_provenance_verified":True,"adaptive_after_R85_results":True,
              "new_fits":0,"new_NN_fits":0,"models_loaded":0,"model_predict_calls":0,
              "reference_predict_calls":2,"spatial_predict_calls":2,"elapsed_seconds":time.monotonic()-start,
              "prediction_sha256":sha(folder/"predictions.npy"),"artifact_sha256":{rel(p):sha(p) for p in folder.rglob("*") if p.is_file()}}
        write(folder/"results.json",item);result["stages"][s]=item;result["checked_stages"].append(s);result["reference_predict_calls"]+=2;result["spatial_predict_calls"]+=2
        print(json.dumps({"stage":s,"R87":metrics["R87"]["overall_mae"],"v22":metrics["v22"]["overall_mae"],"v24":metrics["v24"]["overall_mae"]}),flush=True)
    assert result["new_fits"]==0 and result["checked_stages"]==list(STAGES)
    assert all(sha(ROOT/n)==h for n,h in {**inputs,**source,**protected}.items())
    assert sha(DOCUMENT)==plan_sha
    result.update(status="three_periods_completed_no_adoption_decision",elapsed_seconds=time.monotonic()-tick,protected_after=protected)
    write(OUT/"results.json",result)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--plan-sha",required=True);args=ap.parse_args()
    assert sha(DOCUMENT)==args.plan_sha and not OUT.exists(),"Plan changed or output exists"
    OUT.mkdir();tick=time.monotonic();started=old.now();error=None
    write(OUT/"started.json",{"started_at_utc":started,"source_sha256":sha(Path(__file__)),"command":sys.argv})
    try:run(args.plan_sha)
    except Exception:error=traceback.format_exc();print(error,flush=True)
    write(OUT/"execution.json",{"started_at_utc":started,"completed_at_utc":old.now(),"elapsed_seconds":time.monotonic()-tick,
                              "exit_code":1 if error else 0,"failure":error,"source_sha256":sha(Path(__file__))})
    raise SystemExit(1 if error else 0)
