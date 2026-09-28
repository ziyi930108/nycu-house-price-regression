"""R89 fixed type/use time slopes with coherent reference/spatial; three CPU fits."""
from pathlib import Path
import argparse,json,pickle,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old
import round89_group_trend as helper

ROOT=old.ROOT
OUT=ROOT/"outputs/round89"
DOCUMENT=ROOT/"reports/round89_plan.md"
STAGES=old.STAGES
sha,read,write,rel,spatial=old.sha,old.locked,old.write,old.rel,old.spatial
FORMULA="median(R89_group_normalized_reference_and_spatial_on_fixed_v22_NN_log_unit_bases)"
def exact(a,b,label):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label
def vector(path,n):
    v=np.load(ROOT/path,allow_pickle=False)
    assert v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>=0).all(),path
    return v
def run(plan_sha):
    tick=time.monotonic();assert sha(DOCUMENT)==plan_sha
    records={n:read(ROOT/f"outputs/round{n}/results.json") for n in (46,71,79,83)}
    plans={n:read(ROOT/f"outputs/round{n}/plan.json") for n in (46,71,79)}
    p79=plans[79];r79=records[79];r83=records[83]
    support_audit=read(ROOT/"outputs/round89_fit_audit/results.json")
    assert support_audit["new_candidate_fits"]==0 and not support_audit["query_MAE_computed"]
    assert r79["plan_sha256"]==sha(ROOT/"outputs/round79/plan.json")
    assert all(r["checked_stages"]==list(STAGES) for r in records.values())
    assert all(p["split_contract"]==p79["split_contract"] for p in plans.values())
    inputs=dict(p79["input_sha256"]);source=dict(p79["source_sha256"])
    def bind(name,digest):
        assert name not in inputs or inputs[name]==digest,name
        inputs[name]=digest
    bind("outputs/round89_fit_audit/results.json",sha(ROOT/"outputs/round89_fit_audit/results.json"))
    for n,r in records.items():
        name=f"outputs/round{n}/results.json";bind(name,sha(ROOT/name))
        for name,h in r["source_sha256"].items():
            assert name not in source or source[name]==h,name
            source[name]=h
    for n in plans:
        name=f"outputs/round{n}/plan.json";bind(name,sha(ROOT/name))
    for name,h in helper.source_hashes().items():
        assert name not in source or source[name]==h,name
        source[name]=h
    source[rel(Path(__file__))]=sha(Path(__file__))
    branch_bindings={}
    for s in STAGES:
        gm=p79["bindings"][s]["unit"]["manifest"]["features"]["fit"]
        bind(gm["path"],gm["sha256"])
        for n,r in records.items():
            a=r["stages"][s]
            assert a["dates"]==p79["splits"][s] and a["fit_rows"]==p79["split_contract"][s]["fit_rows"] and a["query_rows"]==p79["split_contract"][s]["query_rows"]
            idxfolder=f"outputs/round{n}/{s}"+("/model" if n==46 else "")
            for k in ("fit","query"):
                path=f"{idxfolder}/{k}_indices.npy";bind(path,a["artifact_sha256"][path])
            if n in (79,83):
                path=f"outputs/round{n}/{s}/predictions.npy";bind(path,a["prediction_sha256"])
        branch_bindings[s]={
            "unit":(79,f"outputs/round79/{s}","base.npy","reference.npy","unit_spatial.npy"),
            "nn":(71,f"outputs/round71/{s}/seed42","base.npy","reference.npy","nn_spatial.npy"),
            "log":(46,f"outputs/round46/{s}","base_recency_log.npy","reference_recency_log.npy","recency_log_spatial.npy")}
        for _,(n,folder,b,r,u) in branch_bindings[s].items():
            for suffix in (b,r,u,"reference_diagnostics.pkl","spatial_diagnostics.pkl"):
                name=folder+"/"+suffix;bind(name,records[n]["stages"][s]["artifact_sha256"][name])
        name=f"outputs/round79/{s}/model/metadata.json";bind(name,r79["stages"][s]["artifact_sha256"][name])
    for name,h in {**inputs,**source}.items():assert sha(ROOT/name)==h,"Input/source changed: "+name
    protected={rel(p):sha(p) for p in sorted((ROOT/"outputs").glob("submission_v*.csv"))}
    for name in ("runs/versions.json","outputs/current_private_candidate.json","PRIVATE_SELECTION.md","CURRENT_STATUS.md","outputs/round84/decision.json","outputs/round79/selection/decision.json","outputs/v22_submission_focus_20260928/user_selection.json"):
        protected[name]=sha(ROOT/name)
    plan=dict(round=89,candidate="R89",formula=FORMULA,stage_order=list(STAGES),
       primary_control="v22",secondary_control="v24",planned_new_fits=3,planned_new_native_fits=0,new_NN_fits=0,
       parameters=helper.parameters(),fit_family="CPU_group_trend_wrapper_with_rebuilt_spatial",
       text_plan_sha256=plan_sha,source_sha256=source,input_sha256=inputs,
       branch_bindings=branch_bindings,split_contract=p79["split_contract"],splits=p79["splits"],
       target_group_weights=p79["target_group_weights"],protected_before=protected,
       query_labels_used_only_for_evaluation=True,policy_uses_query_labels=False,
       adaptive_after_error_audit=True,fresh_blind_validation=False,Public_feedback_used_for_tuning=False,
       new_CSV=False,new_test_predictions=False,test_data_loaded=False,planned_new_reference_normalizations=3,planned_new_spatial_builds=3,
       automatic_adoption=False)
    write(OUT/"plan.json",plan)
    Xall,yall=old.previous.load_training_rows(p79);assert len(Xall)==455255
    result={k:v for k,v in plan.items() if k!="branch_bindings"}
    result.update(status="running",checked_stages=[],stages={},new_fits=0,new_native_fits=0,native_models_loaded=0,native_model_predict_calls=0,wrapper_loads=0,wrapper_predict_calls=0,
       reference_control_replay_calls=0,spatial_control_replay_calls=0,plan_sha256=sha(OUT/"plan.json"),
       runtime=dict(numpy=np.__version__,pandas=pd.__version__))
    for s in STAGES:
        start=time.monotonic();folder=OUT/s;folder.mkdir()
        bnd=p79["bindings"][s];b=bnd["unit"];spec=p79["split_contract"][s]
        indices={}
        for k in ("fit","query"):
            v=np.load(ROOT/b[k+"_indices"],allow_pickle=False)
            assert v.dtype==np.int64 and v.shape==(spec[k+"_rows"],) and (np.diff(v)>0).all()
            assert old.original.value_sha(v,"<i8")==spec[k+"_indices_content_sha256"]
            assert v.min()>=0 and v.max()<len(Xall)
            for n in (46,71,79,83):
                p=f"outputs/round{n}/{s}"+("/model" if n==46 else "")+f"/{k}_indices.npy"
                exact(v,np.load(ROOT/p,allow_pickle=False),"All branch/control indices")
            indices[k]=v;old.previous.save_array(folder/(k+"_indices.npy"),v)
        fi,qi=indices["fit"],indices["query"];assert not np.intersect1d(fi,qi).size
        X=Xall.iloc[fi].reset_index(drop=True);Xq=Xall.iloc[qi].reset_index(drop=True);n=len(qi)
        cutoff=(pd.Timestamp(p79["splits"][s][0])-pd.Timestamp("2000-01-01")).days
        end=(pd.Timestamp(p79["splits"][s][1])-pd.Timestamp("2000-01-01")).days
        assert np.isfinite(X.trade_day_index).all() and (X.trade_day_index<cutoff).all() and Xq.trade_day_index.between(cutoff,end).all()
        frames=dict(fit_X=spatial.frame_sha(X),query_X=spatial.frame_sha(Xq))
        assert frames==r79["stages"][s]["original_frame_sha256"]==records[71]["stages"][s]["original_frame_sha256"]
        m=read(ROOT/f"outputs/round79/{s}/model/metadata.json")
        assert frames==m["original_frame_sha256"] and old.original.value_sha(yall[fi],"<f8")==m["fit_target_values_sha256"]
        geo=pd.read_pickle(ROOT/b["query_geo"])
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo)==b["manifest"]["features"]["query"]["frame_sha256"]
        gm=b["manifest"]["features"]["fit"];fitgeo=pd.read_pickle(ROOT/gm["path"])
        assert fitgeo.index.equals(X.index) and spatial.frame_sha(fitgeo)==gm["frame_sha256"]
        with (ROOT/b["state"]).open("rb") as f:state=pickle.load(f)
        ss=read(ROOT/b["state_summary"])
        assert state["state_summary"]==ss and state["parameters"]==spatial.parameters()
        assert ss["fit_X_sha256"]==frames["fit_X"] and not ss["query_used_in_build"]
        assert ss["fit_geo_sha256"]==b["manifest"]["features"]["fit"]["frame_sha256"]
        with (ROOT/b["reference"]).open("rb") as f:reference=pickle.load(f)
        assert reference["fit_report"]["fitting_rows"]==len(fi)
        assert old.original.value_sha(reference["reference_residual"],"<f8")==ss["existing_reference_residual_sha256"]
        modeldir=folder/"model";modeldir.mkdir()
        write(modeldir/"fit_started.json",dict(started_at_utc=old.now(),parameters=helper.parameters(),query_labels_used=False))
        fit_tick=time.monotonic();failure=None
        try:
            wrapper,meta=helper.fit_model(X,yall[fi].copy(),fitgeo,reference,state)
            helper.validate_model(wrapper)
            expected_support={(r["building_type"],r["main_use"]):
                (r["support"]["rows"],r["support"]["independent_project_keys"],r["support"]["r2_project_groups"],r["support"]["combined_date_span_days"])
                for r in support_audit["stages"][s]["fixed_R89_supported_groups"]}
            observed_support={k:(r["rows"],r["independent_projects"],r["reference_projects"],r["span_days"])
                for k,r in wrapper["group_trends"].items() if r["supported"]}
            assert observed_support==expected_support,"Independent fit-only support audit differs"
            meta["independent_fit_support_audit_exact"]=True
        except Exception:
            failure=traceback.format_exc();raise
        finally:
            fit_seconds=time.monotonic()-fit_tick
            write(modeldir/"execution.json",dict(elapsed_seconds=fit_seconds,exit_code=1 if failure else 0,failure=failure))
        wrapper_identity=helper.state_identity(wrapper)
        blob=pickle.dumps(wrapper,protocol=5)
        with (modeldir/"model.pkl").open("xb") as f:f.write(blob)
        with (modeldir/"model.pkl").open("rb") as f:loaded=pickle.load(f)
        helper.validate_model(loaded)
        assert helper.state_identity(loaded)==wrapper_identity,"Wrapper value identity roundtrip exact"
        meta.update(stage=s,fit_seconds=fit_seconds,original_frame_sha256=frames,
           fit_indices_content_sha256=spec["fit_indices_content_sha256"],
           query_indices_content_sha256=spec["query_indices_content_sha256"],
           model_file_sha256=sha(modeldir/"model.pkl"),plan_sha256=sha(OUT/"plan.json"),
           native_models_fitted=0,query_labels_used_in_fit=False)
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p79["splits"][s][0],p79["target_group_weights"])
        originals={};candidates={};branch_stats={};masks={}
        call_counts=dict(global_reference=0,global_spatial=0,group_reference=0,group_spatial=0)
        for branch,(_,prefix,basefile,reffile,spatialfile) in branch_bindings[s].items():
            base=vector(prefix+"/"+basefile,n);saved_ref=vector(prefix+"/"+reffile,n);original=vector(prefix+"/"+spatialfile,n)
            saved_diag=pd.read_pickle(ROOT/prefix/"reference_diagnostics.pkl")
            saved_detail=pd.read_pickle(ROOT/prefix/"spatial_diagnostics.pkl")
            pr,diag,rr=old.original.predict_reference(reference,Xq,base,variant="trend",return_diagnostics=True)
            output,detail,sr=spatial.predict(state,Xq,geo,base,pr,diag)
            exact(pr,saved_ref,branch+" reference replay");exact(output,original,branch+" spatial replay")
            pd.testing.assert_frame_equal(diag,saved_diag,check_exact=True)
            pd.testing.assert_frame_equal(detail,saved_detail,check_exact=True)
            out=helper.predict_model(wrapper,Xq,geo,base)
            replay=helper.predict_model(loaded,Xq,geo,base)
            for evaluated in (out,replay):
                for key in call_counts:call_counts[key]+=evaluated["report"][key+"_calls"]
            for k in ("reference","spatial","global_reference_prediction","global_spatial_prediction","affected_mask"):
                exact(np.asarray(out[k]),np.asarray(replay[k]),branch+" wrapper reload "+k)
            for k in ("reference_diagnostics","spatial_diagnostics","global_reference_diagnostics","global_spatial_diagnostics"):
                pd.testing.assert_frame_equal(out[k],replay[k],check_exact=True)
            exact(out["global_reference_prediction"],saved_ref,branch+" helper global reference")
            exact(out["global_spatial_prediction"],original,branch+" helper global spatial")
            pd.testing.assert_frame_equal(out["global_reference_diagnostics"],saved_diag,check_exact=True)
            pd.testing.assert_frame_equal(out["global_spatial_diagnostics"],saved_detail,check_exact=True)
            candidate=np.asarray(out["spatial"]);mask=np.asarray(out["affected_mask"])
            assert candidate.dtype==np.float64 and candidate.shape==(n,) and np.isfinite(candidate).all() and (candidate>=0).all()
            assert mask.dtype==np.bool_ and mask.shape==(n,)
            exact(candidate[~mask],original[~mask],branch+" fallback unchanged")
            exact(np.asarray(out["reference"])[~mask],saved_ref[~mask],branch+" reference fallback unchanged")
            originals[branch]=original;candidates[branch]=candidate;masks[branch]=mask
            branch_stats[branch]=dict(affected_query_rows=int(mask.sum()),affected_fraction=float(mask.mean()),
                changed_rows=int(np.count_nonzero(candidate!=original)),original_metrics=score(original),
                candidate_reference_metrics=score(out["reference"]),candidate_metrics=score(candidate),
                reference_replayed_exact=True,spatial_replayed_exact=True,diagnostics_replayed_exact=True,
                fallback_unchanged=True,wrapper_reload_all_rows_and_diagnostics_exact=True,
                candidate_report=out["report"])
            for key,value in (("reference",out["reference"]),("spatial",candidate),("affected_mask",mask)):
                old.previous.save_array(folder/(branch+"_"+key+".npy"),value)
            for key in ("reference_diagnostics","spatial_diagnostics"):
                with (folder/(branch+"_"+key+".pkl")).open("xb") as f:pickle.dump(out[key],f,protocol=5)
        assert all(np.array_equal(masks["unit"],masks[k]) for k in ("nn","log")),"Same model/X must produce same affected group mask"
        assert helper.state_identity(wrapper)==wrapper_identity and helper.state_identity(loaded)==wrapper_identity,"Predict mutated fitted wrapper"
        meta.update(wrapper_reload_all_query_three_branches_exact=True,model_unchanged_by_predict=True)
        write(modeldir/"metadata.json",meta)
        fixed={k:vector(bnd["paths"][k],n) for k in ("v11","v12","v5")}
        fixed["v22"]=vector(f"outputs/round79/{s}/predictions.npy",n);fixed["v24"]=vector(f"outputs/round83/{s}/predictions.npy",n)
        exact(originals["nn"],vector(bnd["paths"]["nn"],n),"Correct fixed NN branch")
        exact(originals["log"],vector(bnd["paths"]["log"],n),"Correct fixed log branch")
        exact(np.median(np.column_stack([originals[k] for k in ("nn","log","unit")]),axis=1),fixed["v22"],"Original v22 median")
        metrics={k:score(v) for k,v in fixed.items()}
        for k in ("v11","v12","v5"):assert metrics[k]==r79["stages"][s]["metrics"][k]
        assert metrics["v22"]==r79["stages"][s]["metrics"]["R79"] and metrics["v24"]==r83["stages"][s]["metrics"]["R83"]
        pred=np.median(np.column_stack([candidates[k] for k in ("nn","log","unit")]),axis=1)
        assert np.isfinite(pred).all() and (pred>=0).all()
        union=masks["unit"]|masks["nn"]|masks["log"]
        exact(pred[~union],fixed["v22"][~union],"All unmasked final rows unchanged")
        metrics["R89"]=score(pred);old.previous.save_array(folder/"predictions.npy",pred)
        diff=old.previous.prior.log_run.differences
        item=dict(stage=s,status="completed",fit_rows=len(fi),query_rows=n,dates=p79["splits"][s],
            original_frame_sha256=frames,metrics=metrics,differences={k:diff(metrics[k],metrics["R89"]) for k in metrics if k!="R89"},
            branch_policy_diagnostics=branch_stats,affected_query_rows=int(union.sum()),fit_seconds=fit_seconds,
            changed_final_rows=int(np.count_nonzero(pred!=fixed["v22"])),branch_masks_identical=all(np.array_equal(masks["unit"],masks[k]) for k in ("nn","log")),
            controls_full_metrics_exact=True,original_v22_median_exact=True,all_original_reference_spatial_and_diagnostics_exact=True,
            query_labels_used_only_for_evaluation=True,adaptive_after_error_audit=True,new_fits=1,new_native_fits=0,new_NN_fits=0,
            native_models_loaded=0,native_model_predict_calls=0,new_reference_normalizations=1,new_spatial_builds=1,
            wrapper_loads=1,wrapper_predict_calls=6,reference_control_replay_calls=3,spatial_control_replay_calls=3,
            wrapper_internal_call_counts=call_counts,
            elapsed_seconds=time.monotonic()-start,
            prediction_sha256=sha(folder/"predictions.npy"),artifact_sha256={rel(p):sha(p) for p in folder.rglob("*") if p.is_file()})
        write(folder/"results.json",item);result["stages"][s]=item;result["checked_stages"].append(s)
        result["reference_control_replay_calls"]+=3;result["spatial_control_replay_calls"]+=3
        result["new_fits"]+=1;result["wrapper_loads"]+=1;result["wrapper_predict_calls"]+=6
        print(json.dumps(dict(stage=s,R89=metrics["R89"]["overall_mae"],v22=metrics["v22"]["overall_mae"],
           weighted=metrics["R89"]["weighted_mae"],mask_rows=int(union.sum()),changed_rows=item["changed_final_rows"])),flush=True)
    assert result["checked_stages"]==list(STAGES)
    assert all(sha(ROOT/n)==h for n,h in {**inputs,**source,**protected}.items())
    assert sha(DOCUMENT)==plan_sha
    result.update(status="three_periods_completed_no_adoption_decision",elapsed_seconds=time.monotonic()-tick,protected_after=protected)
    write(OUT/"results.json",result)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--plan-sha",required=True);a=ap.parse_args()
    assert a.plan_sha and sha(DOCUMENT)==a.plan_sha and not OUT.exists(),"Plan changed or output exists"
    OUT.mkdir();tick=time.monotonic();started=old.now();error=None
    write(OUT/"started.json",dict(started_at_utc=started,source_sha256=sha(Path(__file__)),command=sys.argv,automatic_retry=False))
    try:run(a.plan_sha)
    except Exception:error=traceback.format_exc();print(error,flush=True)
    write(OUT/"execution.json",dict(started_at_utc=started,completed_at_utc=old.now(),elapsed_seconds=time.monotonic()-tick,
          exit_code=1 if error else 0,failure=error,source_sha256=sha(Path(__file__)),automatic_retry=False))
    raise SystemExit(1 if error else 0)
