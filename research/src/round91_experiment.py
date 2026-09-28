"""R91 fixed signed total-price residual on group-OOF prior, same v22 ensemble; three chronological fits."""
from pathlib import Path
import argparse, contextlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor, Pool
import round48_nn_experiment as old
import round79_recency_unit as previous_model
import round91_prior_residual as helper
import round60_recency_price_prior as price_prior

ROOT=old.ROOT
OUT=ROOT/"outputs/round91"
DOCUMENT=ROOT/"reports/round91_plan.md"
STAGES=old.STAGES
sha,read,write,rel,spatial=old.sha,old.locked,old.write,old.rel,old.spatial
FORMULA="median(saved_NN,saved_log,spatial(reference(.75*prior_residual_TWD+.25*saved_LGB_unit_TWD)))"

def exact(a,b,label):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label

def vector(path,n):
    a=np.load(ROOT/path,allow_pickle=False)
    assert a.dtype==np.float64 and a.shape==(n,) and np.isfinite(a).all() and (a>=0).all(),path
    return a

def raw_predict(model,state,X,G,cats,prior):
    helper.check_state(state,X,cats)
    helper.check_model(model,state)
    z=np.asarray(model.predict(Pool(helper.augment_features(X,G),cat_features=state["categorical_columns"],thread_count=4),thread_count=4),dtype=np.float64)
    assert z.shape==(len(X),) and np.isfinite(z).all()
    return z

def run(plan_sha):
    tick=time.monotonic()
    assert sha(DOCUMENT)==plan_sha
    p79=read(ROOT/"outputs/round79/plan.json")
    r79=read(ROOT/"outputs/round79/results.json")
    r83=read(ROOT/"outputs/round83/results.json")
    assert r79["plan_sha256"]==sha(ROOT/"outputs/round79/plan.json")
    assert r79["checked_stages"]==r83["checked_stages"]==list(STAGES)
    r60=read(ROOT/"outputs/round60/results.json")
    p60=read(ROOT/"outputs/round60/plan.json")
    assert p60["split_contract"]==p79["split_contract"]
    assert all(r60["stages"][s]["dates"]==p79["splits"][s] for s in STAGES)
    inputs=dict(p79["input_sha256"])
    source={**p79["source_sha256"],**helper.source_hashes(),rel(Path(__file__)):sha(Path(__file__))}
    for name in ("outputs/round79/plan.json","outputs/round79/results.json","outputs/round83/results.json","outputs/round60/results.json","outputs/round60/plan.json"):
        inputs[name]=sha(ROOT/name)
    for s in STAGES:
        for suffix in ("train_oof_unit_prior.npy","query_unit_prior.npy","prior_fold_ids.npy","price_prior_state.json","prior_preparation.json"):
            name=f"outputs/round60/{s}/{suffix}"
            inputs[name]=r60["stages"][s]["artifact_sha256"][name]
        for r,record in ((79,r79),(83,r83)):
            st=record["stages"][s]
            for name,digest in st["artifact_sha256"].items():
                if name.endswith(("predictions.npy","fit_indices.npy","query_indices.npy","base.npy","reference.npy","unit_spatial.npy","metadata.json")):
                    inputs[name.replace("\\","/")]=digest
        for suffix in ("model/metadata.json","model/feature_state.json"):
            path=f"outputs/round79/{s}/{suffix}"
            inputs[path]=r79["stages"][s]["artifact_sha256"][path]
    for name,digest in {**inputs,**source}.items():
        assert sha(ROOT/name)==digest,"Dependency changed: "+name
    check_receipt=ROOT/"reports/round91_helper_selfcheck.json"
    receipt=json.loads(check_receipt.read_text(encoding="utf-8-sig"))
    assert receipt["status"]=="passed"
    assert all(sha(ROOT/n)==h for n,h in receipt["source_sha256"].items())
    inputs[rel(check_receipt)]=sha(check_receipt)
    protected={rel(p):sha(p) for p in sorted((ROOT/"outputs").glob("submission_v*.csv"))}
    for name in ("runs/versions.json","outputs/current_private_candidate.json","PRIVATE_SELECTION.md","CURRENT_STATUS.md","outputs/v22_submission_focus_20260928/user_selection.json","outputs/round84/decision.json","outputs/round79/selection/decision.json"):
        protected[name]=sha(ROOT/name)
    plan={"round":91,"candidate":"R91","stage_order":list(STAGES),"planned_new_fits":3,
          "primary_control":"v22","secondary_control":"v24","formula":FORMULA,
          "parameters":helper.parameters(),"text_plan_sha256":plan_sha,
          "source_sha256":source,"input_sha256":inputs,"bindings":p79["bindings"],
          "splits":p79["splits"],"split_contract":p79["split_contract"],
          "target_group_weights":p79["target_group_weights"],"protected_before":protected,
          "fresh_blind_validation":False,"Public_feedback_used_for_tuning":False,
          "query_labels_used_only_for_evaluation":True,"new_CSV":False,"new_test_predictions":False,
          "new_NN_fits":0,"new_reference_or_spatial_states":0,"OOF_prior_replay_checks":3,"automatic_adoption":False,"prior_as_offset_not_feature":True,"fit_target_is_signed_residual":True}
    write(OUT/"plan.json",plan)
    cats=json.loads((ROOT/"reports/feature_schema.json").read_text(encoding="utf-8"))["categorical_columns"]
    Xall,yall=old.previous.load_training_rows(p79)
    assert len(Xall)==455255
    result={k:v for k,v in plan.items() if k not in ("bindings","parameters")}
    result.update(status="running",checked_stages=[],stages={},new_fits=0,plan_sha256=sha(OUT/"plan.json"),
                  runtime={"catboost":catboost.__version__,"numpy":np.__version__,"pandas":pd.__version__})
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
            for r in (79,83):
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
        Gfull=pd.read_pickle(ROOT/bnd["tree_geo"])
        assert len(Gfull)==len(Xall) and Gfull.index.equals(Xall.index)
        G,Gq=Gfull.iloc[fi].reset_index(drop=True),Gfull.iloc[qi].reset_index(drop=True)
        del Gfull
        assert spatial.frame_sha(G)==bnd["R19_metadata"]["fit_geo_sha256"]
        assert spatial.frame_sha(Gq)==bnd["R19_metadata"]["query_geo_sha256"]
        assert helper.values_sha256(yfit)==metadata79["fit_target_values_sha256"]
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
        prior_dir=ROOT/f"outputs/round60/{s}"
        train_prior=vector(f"outputs/round60/{s}/train_oof_unit_prior.npy",len(fi))
        query_prior=vector(f"outputs/round60/{s}/query_unit_prior.npy",n)
        prior_prep=read(prior_dir/"prior_preparation.json")
        evidence=prior_prep["prior_evidence"]
        assert evidence["fit_target_sha256"]==helper.values_sha256(yfit)
        assert evidence["source_indices_sha256"]==old.original.value_sha(fi,"<i8")
        assert evidence["oof_prior_sha256"]==helper.values_sha256(train_prior)
        assert evidence["uses_fit_labels"] and not evidence["uses_query_labels"] and not evidence["chronological_oof"]
        assert prior_prep["fit_rows"]==len(fi) and prior_prep["query_rows"]==n and not prior_prep["query_labels_used"]
        prior_begin=time.monotonic()
        replay_oof,replay_state,replay_folds,replay_evidence=price_prior.fit_oof_prior(X,yfit,fi)
        exact(replay_oof,train_prior,"Full OOF prior replay from fit only")
        exact(replay_folds,np.load(prior_dir/"prior_fold_ids.npy",allow_pickle=False),"Prior folds")
        saved_prior_state=read(prior_dir/"price_prior_state.json")
        assert replay_state==saved_prior_state
        assert replay_evidence==evidence
        exact(price_prior.predict_prior(saved_prior_state,Xq),query_prior,"Query prior from fit state")
        with np.load(ROOT/bnd["prepared"],allow_pickle=False) as z:
            for k,frame,prior in (("train",X,train_prior),("query",Xq,query_prior)):
                factor=np.asarray(helper.effective_area(frame)*prior/1e6,dtype=np.float32)
                exact(factor,z[k+"_price_factor_million"],"Prior factor matches R63 prepared")
        write(folder/"prior_verification.json",{"fit_rows":len(fi),"query_rows":n,
            "fit_only_OOF_rebuild_exact":True,"full_fit_state_exact":True,"query_prediction_exact":True,
            "R63_factor_values_exact":True,"chronological_oof":False,"query_labels_used":False,
            "elapsed_seconds":time.monotonic()-prior_begin,"prior_source_sha256":evidence["source_sha256"]})
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
        model_dir=folder/"model";model_dir.mkdir()
        write(model_dir/"fit_started.json",{"started_at_utc":old.now(),"parameters":helper.parameters(),"source_sha256":source,"query_labels_used":False})
        begin=time.monotonic();failure=None
        try:
            with (model_dir/"fit.log").open("x",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                model,state,metadata=helper.fit_model(X,yfit,G,cats,w,train_prior)
        except Exception:
            failure=traceback.format_exc();raise
        finally:
            fit_seconds=time.monotonic()-begin
            write(model_dir/"execution.json",{"exit_code":1 if failure else 0,"failure":failure,"elapsed_seconds":fit_seconds})
        assert state["legacy_feature_state"]==bnd["R19_state"]
        assert metadata["feature_count"]==95 and metadata["categorical_feature_count"]==23
        exact(helper.sample_weights(X,w),w.astype(np.float64),"R91 date-only Pool weights")
        raw=raw_predict(model,state,Xq,Gq,cats,query_prior)
        native=helper.inverse_transform(raw,Xq,query_prior)
        exact(native,helper.predict_model(model,state,Xq,Gq,cats,query_prior),"Public helper prediction API matches explicit inverse")
        model.save_model(str(model_dir/"model.cbm"),format="cbm")
        write(model_dir/"feature_state.json",state)
        loaded=CatBoostRegressor();loaded.load_model(str(model_dir/"model.cbm"),format="cbm")
        raw2=raw_predict(loaded,read(model_dir/"feature_state.json"),Xq,Gq,cats,query_prior)
        exact(raw,raw2,"All query raw reload")
        exact(native,helper.inverse_transform(raw2,Xq,query_prior),"All query TWD reload")
        for k,v in (("raw_z",raw),("native_predictions",native),("date_weights",w),("pool_weights",helper.sample_weights(X,w))):
            old.previous.save_array(model_dir/(k+".npy"),v)
        metadata.update(round=91,stage=s,fit_seconds=fit_seconds,query_rows=n,original_frame_sha256=frames,
                        fit_indices_content_sha256=spec["fit_indices_content_sha256"],query_indices_content_sha256=spec["query_indices_content_sha256"],
                        reload_all_query_rows_verified=True,reload_raw_exact=True,reload_TWD_exact=True,
                        date_weight_provenance_verified=True,plan_sha256=result["plan_sha256"],runtime=result["runtime"])
        metadata["artifact_sha256"]={rel(p):sha(p) for p in model_dir.iterdir() if p.is_file()}
        write(model_dir/"metadata.json",metadata)
        del model,loaded
        base=.75*native+.25*lgb
        pr,diag,rr=old.original.predict_reference(reference,Xq,base,variant="trend",return_diagnostics=True)
        branch,detail,sr=spatial.predict(state_spatial,Xq,geo,base,pr,diag)
        fallback=~detail["applied"].to_numpy(dtype=bool)
        exact(branch[fallback],pr[fallback],"Spatial fallback")
        pred=np.median(np.column_stack([fixed["nn"],fixed["log"],branch]),axis=1)
        assert pred.dtype==np.float64 and pred.shape==(n,) and np.isfinite(pred).all() and (pred>0).all()
        metrics["R91"]=score(pred)
        for k,v in (("base",base),("reference",pr),("unit_spatial",branch),("predictions",pred)):
            old.previous.save_array(folder/(k+".npy"),v)
        for k,v in (("reference_diagnostics",diag),("spatial_diagnostics",detail)):
            with (folder/(k+".pkl")).open("xb") as f:pickle.dump(v,f,protocol=5)
        assert frames=={"fit_X":spatial.frame_sha(X),"query_X":spatial.frame_sha(Xq)}
        diff=old.previous.prior.log_run.differences
        item={"stage":s,"status":"completed","fit_rows":len(fi),"query_rows":n,"dates":p79["splits"][s],
              "original_frame_sha256":frames,"metrics":metrics,"differences":{k:diff(metrics[k],metrics["R91"]) for k in metrics if k!="R91"},
              "unit_branch_metrics":{k:score(v) for k,v in (("native",native),("base",base),("reference",pr),("spatial",branch))},
              "reference_report":rr,"spatial_report":sr,"controls_full_metrics_exact":True,
              "v22_base_reference_spatial_median_full_vectors_exact":True,"query_labels_used_only_for_evaluation":True,
              "full_query_raw_TWD_reload_exact":True,"date_weight_provenance_verified":True,
              "new_fits":1,"new_NN_fits":0,"fit_seconds":fit_seconds,"elapsed_seconds":time.monotonic()-start,
              "prediction_sha256":sha(folder/"predictions.npy"),"artifact_sha256":{rel(p):sha(p) for p in folder.rglob("*") if p.is_file()}}
        write(folder/"results.json",item);result["stages"][s]=item;result["checked_stages"].append(s);result["new_fits"]+=1
        print(json.dumps({"stage":s,"fit_seconds":fit_seconds,"R91":metrics["R91"]["overall_mae"],"v22":metrics["v22"]["overall_mae"],"v24":metrics["v24"]["overall_mae"]}),flush=True)
    assert result["new_fits"]==3 and result["checked_stages"]==list(STAGES)
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
