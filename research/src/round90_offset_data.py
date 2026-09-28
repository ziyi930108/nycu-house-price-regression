"""Fixed R90 offset-window, target-free data preparation.

Public API:
  load_inputs() -> OffsetData (read-only original sources, no prices)
  prepare(out=DEFAULT_OUT) -> OffsetData (new immutable pure-X artifacts)
  load_prepared(out, expected_manifest_sha) -> OffsetData
  self_check() -> synthetic-only checks, no local dataset reads

No train/valid/test price table, model, encoder, prior, prediction or scorer is
loaded. Fit labels must be separately extracted by the caller with the returned
fit_source_cache_indices after verifying the inherited raw-train SHA.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/"outputs/round90/preparation"
CUTOFF="2023-02-11"
QUERY_END="2024-02-10"
ORIGIN_UTC="2023-02-10T16:00:00+00:00"
PLAN="outputs/round79/plan.json"
PLAN_SHA="7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f"
R6_MANIFEST="outputs/round6/geography/confirmation/manifest.json"
R39_MANIFEST="outputs/round39/geography/confirmation/manifest.json"
R39_SHA="556ccfa3ad60c2360df3e5b220f737c6cd1d9a157dca5f02a4238f2eba00c6be"
FEATURES="data_cache/train_features.pkl"
SCHEMA="reports/feature_schema.json"
CANONICAL="outputs/round6/geography/development/train_source_cache_indices.npy"
EXTRACTION="outputs/geography/osm_2023/audit.json"
FORBIDDEN_COLUMNS={"總價元","編號","target","y","ID","id"}
GEO_NUMERIC=["geo_latitude","geo_longitude","geo_log_road_length","geo_log_extent","geo_has_position","geo_nearby_fragments"]
GEO_CATEGORICAL=["geo_match_status","geo_grid_002"]
GEO_COLUMNS=GEO_NUMERIC+GEO_CATEGORICAL
FRAME_NAMES=("Xfit","Xquery","tree_fit","tree_query","spatial_fit","spatial_query")
ARRAY_NAMES=("fit_indices","query_indices","fit_source_cache_indices","query_source_cache_indices")
PARAMETERS={"round":90,"role":"offset_time_stability_check_not_new_model_recipe",
    "fit_before":CUTOFF,"query_start":CUTOFF,"query_end_inclusive":QUERY_END,
    "origin_utc":ORIGIN_UTC,"snapshot_year":2023,"valid_included":False,
    "test_read":False,"price_tables_read":False,"query_prices_read":False,
    "canonical_rows":455255,"raw_feature_cache_rows":455784,
    "tree_geo":"original_R6_2023","spatial_geo":"R39_2023_pure_X_city_road_fallback",
    "new_models_encoders_priors_reference_spatial_or_scores":False}

@dataclass
class OffsetData:
    fit_indices: np.ndarray
    query_indices: np.ndarray
    fit_source_cache_indices: np.ndarray
    query_source_cache_indices: np.ndarray
    Xfit: pd.DataFrame
    Xquery: pd.DataFrame
    tree_fit: pd.DataFrame
    tree_query: pd.DataFrame
    spatial_fit: pd.DataFrame
    spatial_query: pd.DataFrame
    categorical_columns: list[str]
    contract: dict

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def values_sha(a,dtype="<i8"):
    return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype="<u8").tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def relative(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def safe_path(name):
    p=(ROOT/name).resolve()
    need(p.is_relative_to(ROOT.resolve()),"Path outside workspace: "+str(name))
    return p
def json_read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def json_write(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with path.with_suffix(".sha256").open("x",encoding="ascii") as f:f.write(sha(path)+"\n")
def check_index(index,rows,label):
    a=np.asarray(index)
    need(a.dtype==np.dtype("int64") and a.ndim==1 and len(a)>0,label+" must be nonempty int64")
    need((a>=0).all() and (a<rows).all() and (np.diff(a)>0).all(),label+" must be sorted unique in bounds")
def check_target_free(frame,label):
    need(isinstance(frame,pd.DataFrame) and frame.columns.is_unique,label+" invalid DataFrame")
    need(not FORBIDDEN_COLUMNS.intersection(frame.columns),label+" contains target or ID")
def canonical_features(raw,indices,columns=None):
    check_target_free(raw,"raw features")
    need(raw.index.equals(pd.RangeIndex(len(raw))),"Raw feature cache index differs")
    check_index(indices,len(raw),"canonical source indices")
    if columns is not None:need(list(raw.columns)==list(columns),"Original feature schema differs")
    return raw.iloc[indices].reset_index(drop=True).copy(deep=True)
def temporal_partition(features):
    check_target_free(features,"canonical features")
    days=features["trade_day_index"].to_numpy(dtype="float64")
    first=(pd.Timestamp(CUTOFF)-pd.Timestamp("2000-01-01")).days
    last=(pd.Timestamp(QUERY_END)-pd.Timestamp("2000-01-01")).days
    fi=np.flatnonzero(np.isfinite(days)&(days<first)).astype("int64")
    qi=np.flatnonzero(np.isfinite(days)&(days>=first)&(days<=last)).astype("int64")
    need(not np.intersect1d(fi,qi).size,"New fit/query overlap")
    return fi,qi
def remap_geography(parts,requested,canonical_rows):
    """Map role-independent geo frames by canonical row ID; never by old role."""
    check_index(requested,canonical_rows,"requested canonical indices")
    frames=[];seen=[];columns=None;dtypes=None
    for indices,frame in parts:
        check_index(indices,canonical_rows,"source canonical indices")
        check_target_free(frame,"geographic source")
        need(len(indices)==len(frame) and frame.index.equals(pd.RangeIndex(len(frame))),"Geo/index length or local index mismatch")
        if columns is None:columns=list(frame.columns);dtypes=frame.dtypes
        need(list(frame.columns)==columns and frame.dtypes.equals(dtypes),"Mixed geographic schemas")
        f=frame.copy(deep=True);f.index=pd.Index(indices)
        frames.append(f);seen.append(indices)
    need(bool(frames),"No geography partitions")
    all_indices=np.concatenate(seen)
    need(len(np.unique(all_indices))==len(all_indices),"Overlapping geography source partitions")
    joined=pd.concat(frames).sort_index()
    need(np.isin(requested,all_indices).all(),"Requested canonical geography row missing")
    out=joined.loc[requested].reset_index(drop=True).copy(deep=True)
    need(out.dtypes.equals(dtypes),"Remapping changed geographic dtypes")
    return out
def check_geo(frame):
    check_target_free(frame,"geo")
    need(list(frame.columns)==GEO_COLUMNS,"Geographic schema differs")
    need(all(str(frame[c].dtype)=="float32" for c in GEO_NUMERIC),"Geographic numeric dtype differs")
    need(all(frame[c].map(lambda x:isinstance(x,str)).all() for c in GEO_CATEGORICAL),"Geographic strings differ")
    usable=frame.geo_has_position.eq(1)
    need(frame.geo_has_position.isin([0,1]).all(),"Invalid geo availability")
    need(usable.equals(frame.geo_match_status.isin(["road_level","planar_intersection"])),"Geo status differs")
    need(frame.loc[~usable,[c for c in GEO_NUMERIC if c!="geo_has_position"]].isna().all().all(),"Missing geo has values")
    need(frame.loc[~usable,"geo_grid_002"].eq("__MISSING__").all(),"Missing geo has grid")
    need(np.isfinite(frame.loc[usable,GEO_NUMERIC].to_numpy()).all(),"Usable geo nonfinite")
    need(frame.loc[usable,"geo_latitude"].between(20,28).all() and frame.loc[usable,"geo_longitude"].between(116,125).all(),"Geo outside study bounds")
def spatial_preserves_tree(tree,spatial):
    check_geo(tree);check_geo(spatial)
    need(tree.index.equals(spatial.index) and tree.dtypes.equals(spatial.dtypes),"R6/R39 alignment differs")
    preserved=tree.geo_has_position.eq(1)|spatial.geo_has_position.eq(0)
    pd.testing.assert_frame_equal(tree.loc[preserved],spatial.loc[preserved],check_exact=True)
    added=~preserved
    need(tree.loc[added,"geo_has_position"].eq(0).all() and
         tree.loc[added,["geo_latitude","geo_longitude"]].isna().all().all() and
         spatial.loc[added,"geo_match_status"].eq("road_level").all(),"R39 changed existing geography")
    return int(added.sum())

def _validate_bundle(data):
    c=data.contract
    need(c["parameters"]==PARAMETERS,"New origin contract changed")
    n=PARAMETERS["canonical_rows"]
    for name in ("fit_indices","query_indices"):check_index(getattr(data,name),n,name)
    need(not np.intersect1d(data.fit_indices,data.query_indices).size,"Stored fit/query overlap")
    for name in ARRAY_NAMES:
        a=getattr(data,name)
        need(values_sha(a)==c["arrays"][name]["content_sha256"] and len(a)==c["arrays"][name]["rows"],"Stored array identity differs: "+name)
    for name in FRAME_NAMES:
        f=getattr(data,name);check_target_free(f,name)
        record=c["frames"][name]
        need(frame_sha(f)==record["frame_sha256"] and {k:str(f[k].dtype) for k in f}==record["dtypes"] and list(f)==record["columns"],"Stored frame identity differs: "+name)
        expected=len(data.fit_indices) if name.endswith("fit") else len(data.query_indices)
        need(len(f)==expected and f.index.equals(pd.RangeIndex(expected)),"Stored local frame alignment differs: "+name)
    first=(pd.Timestamp(CUTOFF)-pd.Timestamp("2000-01-01")).days
    last=(pd.Timestamp(QUERY_END)-pd.Timestamp("2000-01-01")).days
    d=data.Xfit.trade_day_index.to_numpy(dtype=float);q=data.Xquery.trade_day_index.to_numpy(dtype=float)
    need(np.isfinite(d).all() and (d<first).all(),"Stored fit dates invalid")
    need(np.isfinite(q).all() and (q>=first).all() and (q<=last).all(),"Stored query dates invalid")
    need(data.categorical_columns==c["categorical_columns"] and len(data.categorical_columns)==21,"Categories differ")
    for role in ("fit","query"):spatial_preserves_tree(getattr(data,"tree_"+role),getattr(data,"spatial_"+role))
    return data

def load_inputs():
    """Read only verified pure-X tables/indices and small provenance documents."""
    start=time.monotonic();inputs={};sources={};inherited={}
    def bind(name,digest):
        p=safe_path(name);actual=sha(p)
        need(actual==digest,"Changed source: "+name)
        need(name not in inputs or inputs[name]==digest,"Conflicting source SHA")
        inputs[name]=actual
        return p
    def small(name,digest,receipt=False):
        p=bind(name,digest)
        if receipt:
            rp=p.with_suffix(".sha256")
            need(rp.read_text(encoding="ascii").strip()==digest,"Receipt differs: "+name)
            inputs[relative(rp)]=sha(rp)
        return json_read(p)
    p79=small(PLAN,PLAN_SHA,True)
    m6=small(R6_MANIFEST,p79["input_sha256"][R6_MANIFEST],True)
    m39=small(R39_MANIFEST,R39_SHA,True)
    need(m39==p79["bindings"]["confirmation"]["unit"]["manifest"],"R39 manifest differs from locked R79 binding")
    need(m39["original_manifest_sha256"]==inputs[R6_MANIFEST],"R39/R6 identity differs")
    for name in ("src/house_features.py","src/model_common.py","src/prepare_data.py","src/round6_geo.py","src/round6_prepare_geo.py"):
        bind(name,p79["source_sha256"][name]);sources[name]=inputs[name]
    for name,digest in m39["source_sha256"].items():
        if name.startswith("src/"):bind(name,digest);sources[name]=digest
    sources[relative(__file__)]=sha(__file__)
    ext=small(EXTRACTION,m6["source_sha256"][EXTRACTION],True)
    timestamp=ext["source"]["header"]["replication_timestamp"]
    need(ext["source"]["header"]["replication_timestamp_header_key"]=="osmosis_replication_timestamp","Not actual replication header")
    need(timestamp==m6["replication_timestamp"]==m39["replication_timestamp"],"Mixed geography snapshots")
    need(m6["snapshot_year"]==m39["snapshot_year"]==2023,"Wrong geography year")
    stamp=pd.Timestamp(timestamp)
    need(stamp.tzinfo is not None and stamp<pd.Timestamp(ORIGIN_UTC) and stamp.year==2023,"Geography snapshot after new origin")
    need(ext["source"]["sha256"]==m6["source_pbf_sha256"],"PBF source identity differs")
    # Preserve the already verified chain without opening test/valid or re-reading the PBF.
    chain=m6["geography_provenance_chain"]
    need(chain["verified"] and chain["matches_to_index_to_extraction_PBF_verified"] and chain["recorded_source_input_output_hashes_verified"],"Historical geographic provenance absent")
    need(m6["geographic_values_learned_from_targets"] is False and m6["price_model_trained"] is False,"R6 geography was not pure X")
    need(m39["spatial_only"] is True and m39["rules"]["new_frames_use"]=="R8 spatial only; never new tree-model inputs","R39 scope differs")
    for name,key in ((m6["geo_index_audit_path"],"geo_index_audit_sha256"),(m6["matching_audit_path"],"matching_audit_sha256")):
        a=small(name,m6[key])
        if name==m6["matching_audit_path"]:
            need(a["target_values_read"] is False and a["model_fitting_or_scoring_performed"] is False,"Matcher used targets")
    inherited.update(raw_train_path="data_cache/train.pkl",raw_train_sha256=p79["input_sha256"]["data_cache/train.pkl"],
        raw_train_opened_or_rehashed=False,original_pbf_sha256=m6["source_pbf_sha256"],
        PBF_opened_or_rehashed=False,full_geographic_chain_rehashed=False,
        historical_chain=m6["geography_provenance_chain"],new_origin_timestamp_checked=True)
    schema=json_read(bind(SCHEMA,p79["input_sha256"][SCHEMA]))
    Xraw=pd.read_pickle(bind(FEATURES,p79["input_sha256"][FEATURES]))
    kt=np.load(bind(CANONICAL,p79["input_sha256"][CANONICAL]),allow_pickle=False)
    need(len(Xraw)==455784 and len(kt)==455255,"Canonical source dimensions differ")
    Xall=canonical_features(Xraw,kt,schema["columns"]);del Xraw
    need(len(Xall.columns)==87 and schema["feature_source_sha256"]==sources["src/house_features.py"],"Frozen features changed")
    cats=[c for c in Xall if not pd.api.types.is_numeric_dtype(Xall[c])]
    need(len(cats)==21,"Expected 21 original categorical columns")
    fi,qi=temporal_partition(Xall)
    check_index(fi,len(Xall),"new fit");check_index(qi,len(Xall),"new query")
    rec=m6["features"]["train"]
    tree_all=pd.read_pickle(bind(rec["path"],rec["sha256"]))
    need(len(tree_all)==len(Xall)==rec["rows"] and tree_all.index.equals(Xall.index),"Full canonical tree geo alignment differs")
    check_geo(tree_all)
    parts=[];old_cut=(pd.Timestamp("2023-08-11")-pd.Timestamp("2000-01-01")).days
    old_end=(pd.Timestamp("2024-08-10")-pd.Timestamp("2000-01-01")).days
    dates=Xall.trade_day_index.to_numpy(dtype=float)
    for role in ("fit","query"):
        idxrec=m39["indices"][role]
        ix=np.load(bind(idxrec["path"],idxrec["sha256"]),allow_pickle=False)
        check_index(ix,len(Xall),"old "+role)
        need(len(ix)==idxrec["rows"] and values_sha(ix)==idxrec["content_sha256"],"Old role index identity differs")
        oldmask=np.isfinite(dates)&((dates<old_cut) if role=="fit" else ((dates>=old_cut)&(dates<=old_end)))
        need(np.array_equal(ix,np.flatnonzero(oldmask)),"Old geo role dates differ")
        gr=m39["features"][role];g=pd.read_pickle(bind(gr["path"],gr["sha256"]))
        need(len(g)==gr["rows"]==len(ix) and frame_sha(g)==gr["frame_sha256"],"Old R39 geo values differ")
        need({k:str(g[k].dtype) for k in g}==gr["dtypes"],"Old R39 geo dtype differs")
        original=tree_all.iloc[ix].reset_index(drop=True)
        need(gr["original_geo_path"]==rec["path"] and gr["original_geo_sha256"]==rec["sha256"] and frame_sha(original)==gr["original_partition_frame_sha256"],"R39 original frame binding differs")
        spatial_preserves_tree(original,g);parts.append((ix,g))
    arrays={"fit_indices":fi,"query_indices":qi,"fit_source_cache_indices":kt[fi].copy(),"query_source_cache_indices":kt[qi].copy()}
    frames={"Xfit":Xall.iloc[fi].reset_index(drop=True).copy(deep=True),"Xquery":Xall.iloc[qi].reset_index(drop=True).copy(deep=True),
        "tree_fit":tree_all.iloc[fi].reset_index(drop=True).copy(deep=True),"tree_query":tree_all.iloc[qi].reset_index(drop=True).copy(deep=True),
        "spatial_fit":remap_geography(parts,fi,len(Xall)),"spatial_query":remap_geography(parts,qi,len(Xall))}
    additions={role:spatial_preserves_tree(frames["tree_"+role],frames["spatial_"+role]) for role in ("fit","query")}
    contract={"round":90,"parameters":PARAMETERS.copy(),"created_at_utc":now(),"fit_rows":len(fi),"query_rows":len(qi),
        "categorical_columns":cats,"input_sha256":inputs,"source_sha256":sources,"inherited_provenance":inherited,
        "canonical_source_indices_content_sha256":values_sha(kt),"geography_replication_timestamp":timestamp,
        "new_origin_utc":ORIGIN_UTC,"spatial_added_rows":additions,
        "arrays":{k:{"rows":len(a),"dtype":str(a.dtype),"content_sha256":values_sha(a)} for k,a in arrays.items()},
        "frames":{k:{"rows":len(f),"columns":list(f),"dtypes":{c:str(f[c].dtype) for c in f},"frame_sha256":frame_sha(f)} for k,f in frames.items()},
        "verification":{"new_date_masks_complete":True,"fit_query_disjoint":True,"canonical_mapping_preserved":True,
            "R39_both_old_partitions_remapped_by_canonical_index":True,"new_geography_rows_all_covered":True,
            "R6_existing_positions_bitwise_preserved":True,"prices_deserialized":False,"test_files_opened":False,
            "new_models_or_fitted_states":False,"fresh_blind_validation":False},
        "elapsed_seconds":time.monotonic()-start}
    for name,digest in inputs.items():need(sha(safe_path(name))==digest,"Source changed during preparation: "+name)
    return _validate_bundle(OffsetData(**arrays,**frames,categorical_columns=cats,contract=contract))

def prepare(out=DEFAULT_OUT):
    """Make a new immutable pure-X preparation; never overwrite an existing dir."""
    out=Path(out).resolve()
    need(out.is_relative_to(ROOT.resolve()) and not out.exists(),"Unsafe or existing preparation directory")
    out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();start=now();failure=None;code=1
    try:
        data=load_inputs();artifacts={}
        for name in ARRAY_NAMES:
            p=out/(name+".npy")
            with p.open("xb") as f:np.save(f,getattr(data,name),allow_pickle=False)
            artifacts[name]={"path":relative(p),"sha256":sha(p),"kind":"int64_array"}
        for name in FRAME_NAMES:
            p=out/(name+".pkl")
            with p.open("xb") as f:getattr(data,name).to_pickle(f)
            artifacts[name]={"path":relative(p),"sha256":sha(p),"kind":"target_free_dataframe"}
        manifest={**data.contract,"artifacts":artifacts,"status":"prepared_target_free"}
        json_write(out/"manifest.json",manifest)
        restored=load_prepared(out,sha(out/"manifest.json"))
        need(restored.fit_indices.tobytes()==data.fit_indices.tobytes(),"Reloaded fit indices differ")
        code=0
        return restored
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        json_write(out/"execution.json",{"exit_code":code,"failure":failure,"started_at_utc":start,
            "completed_at_utc":now(),"elapsed_seconds":time.monotonic()-tick,
            "source_sha256":sha(__file__),"automatic_retry":False,"models_fitted":0,"price_tables_read":0,"test_files_read":0})

def load_prepared(out,expected_manifest_sha):
    """Load only newly prepared X/index artifacts under a caller-bound SHA."""
    out=Path(out).resolve();need(out.is_relative_to(ROOT.resolve()),"Preparation outside workspace")
    path=out/"manifest.json"
    need(sha(path)==expected_manifest_sha and path.with_suffix(".sha256").read_text().strip()==expected_manifest_sha,"Preparation manifest differs")
    m=json_read(path);need(m["status"]=="prepared_target_free" and m["parameters"]==PARAMETERS,"Wrong preparation")
    need(m["source_sha256"][relative(__file__)]==sha(__file__),"Data helper source changed")
    values={}
    for name in (*ARRAY_NAMES,*FRAME_NAMES):
        r=m["artifacts"][name];p=safe_path(r["path"])
        need(p.parent==out and sha(p)==r["sha256"],"Prepared artifact changed: "+name)
        values[name]=np.load(p,allow_pickle=False) if name in ARRAY_NAMES else pd.read_pickle(p)
    return _validate_bundle(OffsetData(**values,categorical_columns=m["categorical_columns"],contract=m))

def self_check():
    """Synthetic date/row-map/role-isolation checks; no real paths are opened."""
    passed=[]
    def mark(name,ok):need(ok,name);passed.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,AssertionError):passed.append(name);return
        raise AssertionError("Expected rejection: "+name)
    first=(pd.Timestamp(CUTOFF)-pd.Timestamp("2000-01-01")).days
    last=(pd.Timestamp(QUERY_END)-pd.Timestamp("2000-01-01")).days
    raw=pd.DataFrame({"trade_day_index":[first-1,first,last,last+1,np.nan,first-2],"n":np.arange(6,dtype=np.float32)})
    keep=np.array([0,1,2,4,5],dtype=np.int64);before=raw.copy(deep=True)
    x=canonical_features(raw,keep);fi,qi=temporal_partition(x)
    mark("fixed_date_start_inclusive_end_inclusive_invalid_excluded",np.array_equal(fi,[0,4]) and np.array_equal(qi,[1,2]))
    mark("canonical_not_raw_position_indices",np.array_equal(keep[fi],[0,5]) and np.array_equal(keep[qi],[1,2]))
    mark("canonical_input_unchanged",raw.equals(before))
    rejects("target_column_rejected",lambda:canonical_features(raw.assign(總價元=1),keep))
    rejects("duplicate_canonical_indices_rejected",lambda:canonical_features(raw,np.array([0,0],dtype=np.int64)))
    rejects("out_of_range_canonical_rejected",lambda:canonical_features(raw,np.array([6],dtype=np.int64)))
    a=pd.DataFrame({"lat":np.array([10,30],dtype=np.float32),"status":["a","c"]})
    b=pd.DataFrame({"lat":np.array([20,40],dtype=np.float32),"status":["b","d"]})
    abefore=a.copy(deep=True);bbefore=b.copy(deep=True)
    parts=[(np.array([0,2],dtype=np.int64),a),(np.array([1,3],dtype=np.int64),b)]
    remapped=remap_geography(parts,np.array([0,1,3],dtype=np.int64),5)
    mark("cross_old_role_mapping_preserves_values_order_dtypes",np.array_equal(remapped.lat,[10,20,40]) and remapped.status.tolist()==["a","b","d"] and remapped.dtypes.equals(a.dtypes))
    remapped.iloc[0,0]=99
    mark("geography_inputs_unmutated_and_copied",a.equals(abefore) and b.equals(bbefore))
    rejects("missing_canonical_geo_row_rejected",lambda:remap_geography(parts,np.array([4],dtype=np.int64),5))
    rejects("overlapping_old_geo_partitions_rejected",lambda:remap_geography([parts[0],parts[0]],np.array([0],dtype=np.int64),5))
    rejects("wrong_old_partition_length_rejected",lambda:remap_geography([(np.array([0],dtype=np.int64),a)],np.array([0],dtype=np.int64),5))
    rejects("mixed_geography_dtypes_rejected",lambda:remap_geography([parts[0],(parts[1][0],b.astype({"lat":"float64"}))],np.array([0],dtype=np.int64),5))
    return {"status":"passed","synthetic_only":True,"real_dataset_files_opened":0,"checks":passed}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--self-check",action="store_true")
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False,indent=2))
    else:
        d=prepare(args.out)
        print(json.dumps({"status":"prepared_target_free","fit_rows":len(d.fit_indices),
            "query_rows":len(d.query_indices),"manifest":relative(args.out/"manifest.json"),
            "manifest_sha256":sha(args.out/"manifest.json")},ensure_ascii=False))
