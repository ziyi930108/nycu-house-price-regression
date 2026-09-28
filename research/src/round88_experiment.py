"""R88 fixed unsupported-reference fallback ablation; CPU, no model load or fit."""
from pathlib import Path
import argparse,json,pickle,sys,time,traceback
import numpy as np
import pandas as pd
import round48_nn_experiment as old

ROOT=old.ROOT
OUT=ROOT/"outputs/round88"
DOCUMENT=ROOT/"reports/round88_plan.md"
STAGES=old.STAGES
sha,read,write,rel,spatial=old.sha,old.locked,old.write,old.rel,old.spatial
FORMULA="median(each original v22 branch replaced by its base iff reference source=base_fallback and spatial.applied=False)"
def exact(a,b,label):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label
def vector(path,n):
    v=np.load(ROOT/path,allow_pickle=False)
    assert v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>=0).all(),path
    return v
def fallback_policy(base,original,source,applied):
    """Pure X/state-derived policy; no targets, indices, model, or mutation."""
    base=np.asarray(base);original=np.asarray(original);source=np.asarray(source);applied=np.asarray(applied)
    n=len(base)
    assert base.dtype==original.dtype==np.float64 and base.shape==original.shape==source.shape==applied.shape==(n,)
    assert np.isfinite(base).all() and np.isfinite(original).all() and (base>=0).all() and (original>=0).all()
    assert applied.dtype==np.bool_ and not pd.isna(source).any()
    mask=(source=="base_fallback") & ~applied
    values=np.where(mask,base,original)
    return values,mask
def self_check():
    b=np.array([10.,20.,30.,40.]);p=np.array([11.,22.,33.,44.])
    cases=[
       ("all_mask",np.array(["base_fallback"]*4),np.zeros(4,dtype=bool),b),
       ("no_mask_reference",np.array(["project"]*4),np.zeros(4,dtype=bool),p),
       ("no_mask_spatial",np.array(["base_fallback"]*4),np.ones(4,dtype=bool),p),
       ("mixed",np.array(["base_fallback","project","base_fallback","base_fallback"]),np.array([False,False,True,False]),np.array([10.,22.,33.,40.]))]
    checks=[]
    for name,src,a,expected in cases:
        before=[x.copy() for x in (b,p,src,a)]
        v,m=fallback_policy(b,p,src,a);exact(v,expected,name)
        assert np.array_equal(m,(src=="base_fallback")&~a)
        for x,y in zip((b,p,src,a),before):exact(x,y,"Input unchanged")
        assert not np.shares_memory(v,b) and not np.shares_memory(v,p)
        checks.append(name)
    return dict(status="passed",checks=checks,inputs_unchanged=True,zero_model_fit_load_predict=True,
                real_data_loaded=False,query_labels_in_policy_API=False,source_sha256=sha(Path(__file__)))

def run(plan_sha):
    tick=time.monotonic();assert sha(DOCUMENT)==plan_sha
    records={n:read(ROOT/f"outputs/round{n}/results.json") for n in (46,71,79,83)}
    plans={n:read(ROOT/f"outputs/round{n}/plan.json") for n in (46,71,79)}
    p79=plans[79];r79=records[79];r83=records[83]
    assert r79["plan_sha256"]==sha(ROOT/"outputs/round79/plan.json")
    assert all(r["checked_stages"]==list(STAGES) for r in records.values())
    assert all(p["split_contract"]==p79["split_contract"] for p in plans.values())
    inputs=dict(p79["input_sha256"]);source=dict(p79["source_sha256"])
    def bind(name,digest):
        assert name not in inputs or inputs[name]==digest,name
        inputs[name]=digest
    for n,r in records.items():
        name=f"outputs/round{n}/results.json";bind(name,sha(ROOT/name))
        for name,h in r["source_sha256"].items():
            assert name not in source or source[name]==h,name
            source[name]=h
    for n in plans:
        name=f"outputs/round{n}/plan.json";bind(name,sha(ROOT/name))
    source[rel(Path(__file__))]=sha(Path(__file__))
    branch_bindings={}
    for s in STAGES:
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
    plan=dict(round=88,candidate="R88",formula=FORMULA,stage_order=list(STAGES),
       primary_control="v22",secondary_control="v24",planned_new_fits=0,new_NN_fits=0,
       text_plan_sha256=plan_sha,source_sha256=source,input_sha256=inputs,
       branch_bindings=branch_bindings,split_contract=p79["split_contract"],splits=p79["splits"],
       target_group_weights=p79["target_group_weights"],protected_before=protected,
       query_labels_used_only_for_evaluation=True,policy_uses_query_labels=False,
       adaptive_after_error_audit=True,fresh_blind_validation=False,Public_feedback_used_for_tuning=False,
       new_CSV=False,new_test_predictions=False,test_data_loaded=False,new_reference_or_spatial_states=0,
       automatic_adoption=False)
    write(OUT/"plan.json",plan)
    Xall,yall=old.previous.load_training_rows(p79);assert len(Xall)==455255
    result={k:v for k,v in plan.items() if k!="branch_bindings"}
    result.update(status="running",checked_stages=[],stages={},new_fits=0,models_loaded=0,model_predict_calls=0,
       reference_predict_calls=0,spatial_predict_calls=0,plan_sha256=sha(OUT/"plan.json"),
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
        with (ROOT/b["state"]).open("rb") as f:state=pickle.load(f)
        ss=read(ROOT/b["state_summary"])
        assert state["state_summary"]==ss and state["parameters"]==spatial.parameters()
        assert ss["fit_X_sha256"]==frames["fit_X"] and not ss["query_used_in_build"]
        assert ss["fit_geo_sha256"]==b["manifest"]["features"]["fit"]["frame_sha256"]
        with (ROOT/b["reference"]).open("rb") as f:reference=pickle.load(f)
        assert reference["fit_report"]["fitting_rows"]==len(fi)
        assert old.original.value_sha(reference["reference_residual"],"<f8")==ss["existing_reference_residual_sha256"]
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p79["splits"][s][0],p79["target_group_weights"])
        originals={};candidates={};branch_stats={};masks={}
        for branch,(_,prefix,basefile,reffile,spatialfile) in branch_bindings[s].items():
            base=vector(prefix+"/"+basefile,n);saved_ref=vector(prefix+"/"+reffile,n);original=vector(prefix+"/"+spatialfile,n)
            saved_diag=pd.read_pickle(ROOT/prefix/"reference_diagnostics.pkl")
            saved_detail=pd.read_pickle(ROOT/prefix/"spatial_diagnostics.pkl")
            pr,diag,rr=old.original.predict_reference(reference,Xq,base,variant="trend",return_diagnostics=True)
            output,detail,sr=spatial.predict(state,Xq,geo,base,pr,diag)
            exact(pr,saved_ref,branch+" reference replay");exact(output,original,branch+" spatial replay")
            pd.testing.assert_frame_equal(diag,saved_diag,check_exact=True)
            pd.testing.assert_frame_equal(detail,saved_detail,check_exact=True)
            candidate,mask=fallback_policy(base,original,diag["source"].to_numpy(),detail["applied"].to_numpy())
            _,saved_mask=fallback_policy(base,original,saved_diag["source"].to_numpy(),saved_detail["applied"].to_numpy())
            exact(mask,saved_mask,branch+" mask replay")
            exact(candidate[~mask],original[~mask],branch+" unmasked unchanged")
            exact(candidate[mask],base[mask],branch+" masked raw base")
            originals[branch]=original;candidates[branch]=candidate;masks[branch]=mask
            before=score(original);after=score(candidate)
            errors0=np.abs(yall[qi]-original);errors1=np.abs(yall[qi]-candidate)
            branch_stats[branch]=dict(mask_rows=int(mask.sum()),mask_fraction=float(mask.mean()),
                changed_rows=int(np.count_nonzero(candidate!=original)),original_metrics=before,candidate_metrics=after,
                masked_original_mae=float(errors0[mask].mean()) if mask.any() else None,
                masked_candidate_mae=float(errors1[mask].mean()) if mask.any() else None,
                delta_total_absolute_error_TWD=float((errors1-errors0).sum()),
                reference_replayed_exact=True,spatial_replayed_exact=True,diagnostics_replayed_exact=True,
                unmasked_unchanged=True,masked_equals_base=True,reference_report=rr,spatial_report=sr)
            old.previous.save_array(folder/(branch+"_candidate.npy"),candidate)
            old.previous.save_array(folder/(branch+"_fallback_mask.npy"),mask)
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
        metrics["R88"]=score(pred);old.previous.save_array(folder/"predictions.npy",pred)
        diff=old.previous.prior.log_run.differences
        item=dict(stage=s,status="completed",fit_rows=len(fi),query_rows=n,dates=p79["splits"][s],
            original_frame_sha256=frames,metrics=metrics,differences={k:diff(metrics[k],metrics["R88"]) for k in metrics if k!="R88"},
            branch_policy_diagnostics=branch_stats,union_mask_rows=int(union.sum()),
            changed_final_rows=int(np.count_nonzero(pred!=fixed["v22"])),branch_masks_identical=all(np.array_equal(masks["unit"],masks[k]) for k in ("nn","log")),
            controls_full_metrics_exact=True,original_v22_median_exact=True,all_original_reference_spatial_and_diagnostics_exact=True,
            query_labels_used_only_for_evaluation=True,adaptive_after_error_audit=True,new_fits=0,new_NN_fits=0,
            models_loaded=0,model_predict_calls=0,new_reference_or_spatial_states=0,
            reference_predict_calls=3,spatial_predict_calls=3,elapsed_seconds=time.monotonic()-start,
            prediction_sha256=sha(folder/"predictions.npy"),artifact_sha256={rel(p):sha(p) for p in folder.iterdir() if p.is_file()})
        write(folder/"results.json",item);result["stages"][s]=item;result["checked_stages"].append(s)
        result["reference_predict_calls"]+=3;result["spatial_predict_calls"]+=3
        print(json.dumps(dict(stage=s,R88=metrics["R88"]["overall_mae"],v22=metrics["v22"]["overall_mae"],
           weighted=metrics["R88"]["weighted_mae"],mask_rows=int(union.sum()),changed_rows=item["changed_final_rows"])),flush=True)
    assert result["checked_stages"]==list(STAGES)
    assert all(sha(ROOT/n)==h for n,h in {**inputs,**source,**protected}.items())
    assert sha(DOCUMENT)==plan_sha
    result.update(status="three_periods_completed_no_adoption_decision",elapsed_seconds=time.monotonic()-tick,protected_after=protected)
    write(OUT/"results.json",result)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--plan-sha");ap.add_argument("--self-check",action="store_true");a=ap.parse_args()
    if a.self_check:
        print(json.dumps(self_check()),flush=True);raise SystemExit(0)
    assert a.plan_sha and sha(DOCUMENT)==a.plan_sha and not OUT.exists(),"Plan changed or output exists"
    OUT.mkdir();tick=time.monotonic();started=old.now();error=None
    write(OUT/"started.json",dict(started_at_utc=started,source_sha256=sha(Path(__file__)),command=sys.argv,automatic_retry=False))
    try:run(a.plan_sha)
    except Exception:error=traceback.format_exc();print(error,flush=True)
    write(OUT/"execution.json",dict(started_at_utc=started,completed_at_utc=old.now(),elapsed_seconds=time.monotonic()-tick,
          exit_code=1 if error else 0,failure=error,source_sha256=sha(Path(__file__)),automatic_retry=False))
    raise SystemExit(1 if error else 0)
