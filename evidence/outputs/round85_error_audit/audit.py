"""Read-only R85 audit of preserved v22 temporal errors; never trains or alters predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    start=time.perf_counter()
    manifest={}
    def path(name):
        p=ROOT/name; manifest[name]=sha(p); return p
    def read(name): return json.loads(path(name).read_text("utf-8"))
    plan=read("outputs/round79/plan.json")
    old=read("outputs/round79/results.json")
    kt=np.load(path("outputs/round6/geography/development/train_source_cache_indices.npy"),allow_pickle=False)
    raw=pd.read_pickle(path("data_cache/train.pkl"))
    xraw=pd.read_pickle(path("data_cache/train_features.pkl"))
    assert len(raw)==len(xraw)==455784 and kt.dtype==np.int64 and len(kt)==455255
    assert (np.diff(kt)>0).all()
    X=xraw.iloc[kt].reset_index(drop=True)
    y=raw["總價元"].to_numpy(float)[kt]
    assert np.isfinite(y).all() and (y>0).all()
    for name in manifest:
        if name in plan["input_sha256"]: assert manifest[name]==plan["input_sha256"][name],name
    def metric(p,t,mask=None,overall_n=None,overall_error=None):
        if mask is not None: p,t=p[mask],t[mask]
        d=p-t; a=np.abs(d)
        if len(a)==0: return {"rows":0}
        return {"rows":len(a),"mae":float(a.mean()),"mean_pred_minus_true":float(d.mean()),
                "median_pred_minus_true":float(np.median(d)),"overprediction_fraction":float((d>0).mean()),
                "error_share":None if overall_error is None else float(a.sum()/overall_error),
                "whole_period_mae_contribution":None if overall_n is None else float(a.sum()/overall_n)}
    def categorical(p,t,g):
        g=np.asarray(g,dtype=str)
        return {v:metric(p,t,g==v,len(t),np.abs(p-t).sum()) for v in sorted(set(g))}
    result={"purpose":"descriptive_error_audit_not_model_selection","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "new_fits":0,"new_predictions":0,"test_labels_accessed":False,"test_features_accessed":False,
        "inputs_unchanged":False,"bias_sign":"prediction minus observed price; positive means overprediction",
        "true_price_bins":"posthoc diagnostics only; unavailable at deployment; never used for routing",
        "window_limitation":"Repeatedly inspected historical windows, not fresh blind evidence",
        "stages":{}}
    for s in ("auxiliary","development","confirmation"):
        def vec(n):
            z=np.load(path(n),allow_pickle=False); assert z.shape==(len(qi),) and np.isfinite(z).all(); return z
        qi=np.load(path(f"outputs/round79/{s}/query_indices.npy"),allow_pickle=False)
        fi=np.load(path(f"outputs/round79/{s}/fit_indices.npy"),allow_pickle=False)
        assert not np.intersect1d(fi,qi).size and (np.diff(qi)>0).all()
        q=X.iloc[qi].reset_index(drop=True);t=y[qi]
        branches={k:vec(n) for k,n in {
            "v22":f"outputs/round79/{s}/predictions.npy",
            "unit_base":f"outputs/round79/{s}/base.npy",
            "unit_reference":f"outputs/round79/{s}/reference.npy",
            "unit_spatial":f"outputs/round79/{s}/unit_spatial.npy",
            "nn_spatial":f"outputs/round71/{s}/seed42/nn_spatial.npy",
            "log_spatial":f"outputs/round46/{s}/recency_log_spatial.npy"}.items()}
        p=branches["v22"]
        assert np.array_equal(p,np.median(np.column_stack([branches[k] for k in ["unit_spatial","nn_spatial","log_spatial"]]),axis=1))
        assert sha(ROOT/f"outputs/round79/{s}/predictions.npy")==old["stages"][s]["prediction_sha256"]
        assert float(np.abs(p-t).mean())==old["stages"][s]["metrics"]["R79"]["overall_mae"]
        for name,digest in old["stages"][s]["artifact_sha256"].items():
            name=name.replace("\\","/")
            if name in manifest: assert manifest[name]==digest,name
        geo=pd.read_pickle(path(plan["bindings"][s]["unit"]["query_geo"]))
        ref=pd.read_pickle(path(f"outputs/round79/{s}/reference_diagnostics.pkl"))
        spatial=pd.read_pickle(path(f"outputs/round79/{s}/spatial_diagnostics.pkl"))
        assert len(q)==len(geo)==len(ref)==len(spatial)
        area_bins=pd.cut(q.building_area,[-np.inf,0,50,100,200,500,1000,5000,np.inf]).astype(str).to_numpy()
        true_bins=pd.cut(t,[-np.inf,1e7,2e7,5e7,1e8,np.inf],right=False).astype(str)
        known=set(X.iloc[fi].project_name)-{"__MISSING__"}
        groups={"building_area_m2":area_bins,"true_price_TWD":true_bins,
            "transaction_type":q.transaction_type,"building_type":q.building_type,
            "address_missing":q.address_raw.eq("__MISSING__").astype(str),
            "address_road_missing":q.address_road.eq("__MISSING__").astype(str),
            "geo_has_position":geo.geo_has_position.astype(str),"geo_match_status":geo.geo_match_status,
            "partial_share_flag":q.remark_partial_share.astype(str),
            "project_seen_in_fit":q.project_name.isin(known).astype(str),
            "unit_reference_source":ref.source,"unit_spatial_reason":spatial.reason}
        selected={"industrial_office":q.building_type.eq("廠辦").to_numpy(),
            "area_above_500m2":q.building_area.gt(500).to_numpy(),
            "parking_only":q.transaction_type.eq("車位").to_numpy(),
            "partial_share":q.remark_partial_share.eq(1).to_numpy(),
            "common_area_100_to_200m2":q.building_area.between(100,200,inclusive="right").to_numpy()}
        st={"rows":len(qi),"fit_rows":len(fi),"overall":metric(p,t),
            "groups":{k:categorical(p,t,g) for k,g in groups.items()},
            "selected_group_branch_metrics":{}}
        for k,m in selected.items():
            st["selected_group_branch_metrics"][k]={
                "rows":int(m.sum()),"projects":int(q.loc[m,"project_name"].nunique()),
                "largest_project_rows":int(q.loc[m,"project_name"].value_counts().max()) if m.any() else 0,
                "reference_source_counts":{str(k):int(v) for k,v in ref.loc[m,"source"].value_counts().items()},
                "reference_confidence_mean":float(ref.loc[m,"confidence"].mean()) if m.any() else None,
                "spatial_applied_rows":int(spatial.loc[m,"applied"].sum()),
                "branches":{b:metric(v,t,m,len(t),np.abs(p-t).sum()) for b,v in branches.items()}}
        d=np.abs(p-t); ix=np.argsort(d)[::-1]
        st["worst_error_concentration"]={str(k):{"rows":k,"error_share":float(d[ix[:k]].sum()/d.sum())} for k in [1,10,100,1000]}
        result["stages"][s]=st
    manifest["outputs/round85_error_audit/audit.py"]=sha(Path(__file__))
    result["input_sha256"]=manifest
    assert all(sha(ROOT/n)==h for n,h in manifest.items())
    result["inputs_unchanged"]=True
    result["elapsed_seconds"]=time.perf_counter()-start
    target=OUT/"results.json"
    with target.open("x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"result":str(target),"sha256":sha(target),"input_files":len(manifest),"elapsed_seconds":result["elapsed_seconds"]}))
if __name__=="__main__": main()
