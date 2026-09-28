"""R89 fit-only support audit; never extracts query targets or predicts new prices.

Run from any directory:
  .venv/Scripts/python.exe -B -X utf8 outputs/round89_fit_audit/audit.py
Use --output-dir for an independent rerun; existing output evidence is never replaced.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="3"
import argparse, hashlib, json, pickle, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import round2_neighborhood as r2

STAGES=("development","confirmation","auxiliary")
KT="outputs/round6/geography/development/train_source_cache_indices.npy"
MISSING="__MISSING__"
SPAN_BINS=(0,30,90,180,365,730)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def value_sha(a,dtype="<f8"):
    return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def frame_sha(a):
    return hashlib.sha256(pd.util.hash_pandas_object(a,index=True).to_numpy(dtype="<u8").tobytes()).hexdigest()
def read_locked(path):
    path=Path(path)
    assert path.with_suffix(".sha256").read_text().strip()==sha(path),str(path)
    return json.loads(path.read_text(encoding="utf-8"))
def write(path,data):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False,sort_keys=True)
        f.write("\n")
    with path.with_suffix(".sha256").open("x",encoding="ascii") as f:f.write(sha(path)+"\n")
def now():return datetime.now(timezone.utc).isoformat()
def normalized(series):
    v=series.fillna(MISSING).astype(str).str.strip()
    return v.mask(v.isin(["","nan","None","<NA>","NaT"]),MISSING)
def finite_summary(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {"count":0,"min":None,"p25":None,"median":None,"p75":None,"p90":None,"max":None}
    q=np.quantile(a,[0,.25,.5,.75,.9,1])
    return dict(count=len(a),**{k:float(v) for k,v in zip(("min","p25","median","p75","p90","max"),q)})
def year_coverage(w):
    return {str(int(year)):{"rows":len(z),"r2_project_groups":int(z["project"].nunique()),
        "independent_project_keys":int(z["independent"].nunique())}
        for year,z in w.groupby("calendar_year",sort=True)}
def support(w):
    """Receives feature-only rows already satisfying the original >=8 rule."""
    if w.empty:
        return {"rows":0,"r2_project_groups":0,"independent_project_keys":0,"cities":0,
            "date_span_days":None,"year_coverage":{},"project_date_span_days":finite_summary([]),
            "project_time_variance_year2":finite_summary([]),"time_design_sum_squares_year2":0.,
            "equal_project_weighted_time_sum_squares_year2":0.,"maximum_single_project_rows":0,
            "maximum_single_project_row_share":None,"span_bins":{}}
    grouped=w.groupby("project",sort=True)
    project=grouped.agg(rows=("day","size"),min_day=("day","min"),max_day=("day","max"))
    project["span"]=project.max_day-project.min_day
    t=w.day.to_numpy(dtype=float)/365.25
    means=grouped.day.transform("mean").to_numpy(dtype=float)/365.25
    centered=t-means
    count=grouped.day.transform("size").to_numpy(dtype=float)
    work=w.assign(squared=centered**2)
    project["time_variance"]=work.groupby("project").squared.mean()
    bins={}
    for days in SPAN_BINS:
        # 0 describes strictly positive spans; the other boundaries are inclusive.
        keep=project["span"].gt(0) if days==0 else project["span"].ge(days)
        z=w.loc[w.project.isin(project.index[keep])]
        bins[str(days)]={"rule":"span>0" if days==0 else f"span>={days}",
            "rows":len(z),"r2_project_groups":int(keep.sum()),
            "independent_project_keys":int(z.independent.nunique()),
            "combined_date_span_days":float(z.day.max()-z.day.min()) if len(z) else None,
            "combined_date_span_at_least_730_days":bool(len(z) and z.day.max()-z.day.min()>=730),
            "year_coverage":year_coverage(z)}
    return {"rows":len(w),"r2_project_groups":len(project),
        "independent_project_keys":int(w.independent.nunique()),"cities":int(w.city.nunique()),
        "date_span_days":float(w.day.max()-w.day.min()),"year_coverage":year_coverage(w),
        "project_date_span_days":finite_summary(project["span"]),
        "project_time_variance_year2":finite_summary(project.time_variance),
        "time_design_sum_squares_year2":float(np.sum(centered**2)),
        "equal_project_weighted_time_sum_squares_year2":float(np.sum(centered**2/count)),
        "maximum_single_project_rows":int(project.rows.max()),
        "maximum_single_project_row_share":float(project.rows.max()/len(w)),
        "span_bins":bins}
def audit_stage(stage,plan,Xraw,raw,kt,provenance):
    start=time.monotonic()
    b=plan["bindings"][stage]["unit"]
    paths=[b["fit_indices"],b["reference"]]
    for name in paths:
        actual=sha(ROOT/name);assert actual==plan["input_sha256"][name],name
        provenance[name]=actual
    fi=np.load(ROOT/b["fit_indices"],allow_pickle=False)
    contract=plan["split_contract"][stage]
    assert fi.dtype==np.dtype("int64") and fi.shape==(contract["fit_rows"],) and (np.diff(fi)>0).all()
    assert value_sha(fi,"<i8")==contract["fit_indices_content_sha256"]
    # Select fit rows BEFORE accessing the target column; never create yall/yquery.
    X=Xraw.iloc[kt[fi]].reset_index(drop=True)
    y=raw.iloc[kt[fi],raw.columns.get_loc("總價元")].to_numpy(dtype="float64")
    m=plan["bindings"][stage]["R19_metadata"]
    assert frame_sha(X)==m["fit_original_feature_values_sha256"]
    assert value_sha(y)==m["fit_target_values_sha256"]
    assert np.isfinite(y).all() and (y>0).all()
    cutoff=(pd.Timestamp(plan["splits"][stage][0])-pd.Timestamp("2000-01-01")).days
    all_feature_dates=Xraw.iloc[kt]["trade_day_index"].to_numpy(dtype="float64")
    assert np.array_equal(fi,np.flatnonzero(np.isfinite(all_feature_dates)&(all_feature_dates<cutoff)))
    del all_feature_dates
    days=r2.numeric(X,"trade_day_index")
    assert np.isfinite(days).all() and days.max()<cutoff
    with (ROOT/b["reference"]).open("rb") as f:reference=pickle.load(f)
    assert reference["parameters"]==r2.PARAMETERS
    assert reference["fit_report"]["fitting_rows"]==len(X)
    area=r2.numeric(X,"building_area")
    area_mask=np.isfinite(area)&(area>=15)&(area<=1500)
    unit=np.divide(y,area,out=np.full(len(area),np.nan),where=area>0)
    bounds=np.quantile(unit[area_mask],[.001,.999])
    assert np.array_equal(bounds,reference["fit_report"]["reference_unit_price_bounds"])
    finite_mask=np.isfinite(reference["reference_residual"])
    reconstructed=area_mask&(unit>=bounds[0])&(unit<=bounds[1])
    assert np.array_equal(reconstructed,finite_mask)
    assert int(finite_mask.sum())==reference["fit_report"]["reference_rows"]
    distance,h,fill=r2.matrices(X)
    assert np.array_equal(fill,reference["fill_values"])
    assert np.array_equal(distance,reference["reference_distance"])
    projects=r2.group_keys(X)["project"]
    with threadpool_limits(limits=3):
        replay_beta,replay_report=r2.fit_hedonic(X,y,area,h,finite_mask,projects)
    beta=np.asarray(reference["beta"],dtype=float)
    assert np.allclose(replay_beta,beta,rtol=0,atol=1e-12)
    assert replay_report["fit_rows"]==reference["fit_report"]["hedonic"]["fit_rows"]
    assert replay_report["project_groups"]==reference["fit_report"]["hedonic"]["project_groups"]
    assert replay_report["maximum_fit_day_index"]==reference["fit_report"]["hedonic"]["maximum_fit_day_index"]
    residual=np.log(unit[finite_mask])-h[finite_mask]@beta
    assert np.allclose(residual,reference["reference_residual"][finite_mask],rtol=0,atol=1e-12)
    recent=finite_mask&np.isfinite(days)&(days>=days.max()-4*365.25)&X.project_name.ne(MISSING).to_numpy()
    recent_keys=projects.loc[recent]
    counts=recent_keys.map(recent_keys.value_counts())
    selected=np.zeros(len(X),dtype=bool)
    selected[np.flatnonzero(recent)]=counts.to_numpy()>=8
    assert selected.sum()==replay_report["fit_rows"]
    types=normalized(X.building_type);uses=normalized(X.main_use)
    cities=normalized(X.address_city);districts=normalized(X.address_district);names=normalized(X.project_name)
    independent=cities+"|"+districts+"|"+names
    w=pd.DataFrame({"type":types,"use":uses,"city":cities,"project":projects,
        "independent":independent,"day":days,
        "calendar_year":(pd.Timestamp("2000-01-01")+pd.to_timedelta(days,unit="D")).year,
        "reference":finite_mask,"recent":recent,"selected":selected,
        "named":names.ne(MISSING)})
    all_selected=w.loc[selected].copy()
    totals=support(all_selected)
    groups=[]
    for (style,use),z in w.groupby(["type","use"],sort=True,dropna=False):
        zselected=z.loc[z.selected]
        entry={"building_type":str(style),"main_use":str(use),
            "group_labels_present":bool(style!=MISSING and use!=MISSING),
            "all_fit_rows":len(z),"named_fit_rows":int(z.named.sum()),
            "independent_named_fit_project_keys":int(z.loc[z.named,"independent"].nunique()),
            "reference_finite_residual_rows":int(z.reference.sum()),
            "recent_reference_named_rows_before_project_minimum":int(z.recent.sum()),
            "recent_r2_projects_before_minimum":int(z.loc[z.recent,"project"].nunique()),
            "selected_support":support(zselected)}
        long=entry["selected_support"]["span_bins"].get("180",{})
        entry["fixed_R89_support_eligible"]=bool(entry["group_labels_present"] and
            long.get("rows",0)>=1000 and long.get("independent_project_keys",0)>=30 and
            long.get("combined_date_span_at_least_730_days",False))
        groups.append(entry)
    assert sum(g["all_fit_rows"] for g in groups)==len(X)
    assert sum(g["selected_support"]["rows"] for g in groups)==totals["rows"]
    return {"stage":stage,"cutoff_exclusive":plan["splits"][stage][0],
        "fit_rows":len(X),"fit_indices_content_sha256":value_sha(fi,"<i8"),
        "fit_X_sha256":frame_sha(X),"fit_target_values_sha256":value_sha(y),
        "fit_indices_path":b["fit_indices"],"saved_reference_path":b["reference"],
        "fit_minimum_date":str(pd.Timestamp("2000-01-01")+pd.Timedelta(days=float(days.min())))[:10],
        "fit_maximum_date":str(pd.Timestamp("2000-01-01")+pd.Timedelta(days=float(days.max())))[:10],
        "reference_unit_price_quantile_bounds":bounds.tolist(),
        "reference_finite_rows":int(finite_mask.sum()),"reference_mask_content_sha256":value_sha(finite_mask,"|b1"),
        "hedonic_selected_mask_content_sha256":value_sha(selected,"|b1"),
        "global_beta_saved":beta.tolist(),"global_beta_replayed":replay_beta.tolist(),
        "global_beta_replay_exact":bool(np.array_equal(beta,replay_beta)),
        "global_beta_replay_max_abs_difference":float(np.max(np.abs(beta-replay_beta))),
        "global_beta_features":r2.PARAMETERS["hedonic_features"],
        "global_hedonic_report_replayed":replay_report,
        "reference_residual_replay_max_abs_difference":float(np.max(np.abs(residual-reference["reference_residual"][finite_mask]))),
        "selected_totals":totals,"type_use_groups":groups,"group_count":len(groups),
        "fixed_R89_supported_groups":[{"building_type":g["building_type"],"main_use":g["main_use"],
            "support":g["selected_support"]["span_bins"]["180"]} for g in groups if g["fixed_R89_support_eligible"]],
        "verification":{"canonical_fit_indices_exact":True,"fit_features_exact":True,"fit_target_hash_exact":True,
            "all_fit_dates_before_cutoff":True,"reference_parameters_exact":True,"reference_mask_exact":True,
            "reference_fill_and_distance_exact":True,"original_global_hedonic_replayed":True,
            "group_slope_estimated":False,"query_target_vector_extracted":False,"candidate_predictions_made":False},
        "elapsed_seconds":time.monotonic()-start}
def report(result,path):
    lines=["# R89：僅 fit 資料的分群時間支持稽核","",
        "三個原年度窗口均完成；本次只盤點建物型態 × 主要用途的案內時間支持，並重播原全域 hedonic 係數。沒有估計新群斜率、沒有產生候選預測或 MAE，未讀取 test。","",
        "## 邊界与來源","",
        "原 train.pkl 是整表 pickle，反序列化包含全表；程式先以 canonical fit 索引選列，再抽取該期總價欄，從未建立 query 價格向量。train_features 僅含無目標特徵，所有日期／支持統計只使用 fit 列。R79 plan 含歷史附帶評分，本程式不存取其 metrics 分支。","",
        "沿原 R2：面積 15–1500 平方公尺，單價界限取 fit 面積合格列的 0.001／0.999 分位；核對保存 reference finite residual mask。全域 hedonic 選擇原 fit 最晚日往前 4 × 365.25 日，案名非缺值、每個 R2 案至少 8 列，原 6 次 Huber IRLS。只重播舊係數作來源核驗，不拟合任何新候選。","",
        "R2 案鍵為 city|district|type|use|project；獨立案計數另用 R8 的正規化 city|district|project。同一 type×use 中兩者通常一致，但不能把字串鍵當成經濟上真正獨立建案，沒有做建案別名合併。年度案數跨年可重複。","",
        "日期跨度在原 hedonic 選定列內計算。固定描述 bins 為 >0、≥30／90／180／365／730 日，並非根據 query 成績設門檻。time design 平方和以案內中心化的交易年份計；案內變異數用母體分母 n。per-project equal 版本每列乘 1/n，仅为設計支持統計。","",
        "## 三期核驗","",
        "| 期別 | fit 列 | reference 列 | hedonic 列 | R2 案／獨立案 | 原時間係數 | beta 重播最大差 |","|---|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        s=result["stages"][stage];t=s["selected_totals"]
        lines.append(f"| {stage} | {s['fit_rows']:,} | {s['reference_finite_rows']:,} | {t['rows']:,} | {t['r2_project_groups']:,}／{t['independent_project_keys']:,} | {s['global_beta_saved'][3]:.10f} | {s['global_beta_replay_max_abs_difference']:.3g} |")
    lines+=["","## 全部群組支持","",
        "以下列出所有 fit 中出現的群，不按 query 或未知 Private 選群。180 日案數／列數只包括案自身日期跨度≥180日的原 hedonic 支持；該子集總跨度與單案集中度詳見 JSON。"]
    for stage in STAGES:
        s=result["stages"][stage]
        lines+=["",f"### {stage}（fit 截止 {s['cutoff_exclusive']} 之前）","",
            "| 建物型態 | 用途 | fit 列 | 選定列 | 獨立案 | ≥90日案 | ≥180日案／列 | ≥365日案 | 案跨度中位(日) | time SS(年²) | 最大單案占列 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g in s["type_use_groups"]:
            t=g["selected_support"];bins=t["span_bins"]
            b90=bins.get("90",{});b180=bins.get("180",{});b365=bins.get("365",{})
            med=t["project_date_span_days"]["median"]
            share=t["maximum_single_project_row_share"]
            lines.append(f"| {g['building_type']} | {g['main_use']} | {g['all_fit_rows']:,} | {t['rows']:,} | {t['independent_project_keys']:,} | {b90.get('independent_project_keys',0):,} | {b180.get('independent_project_keys',0):,}／{b180.get('rows',0):,} | {b365.get('independent_project_keys',0):,} | {med if med is not None else '—'} | {t['time_design_sum_squares_year2']:.3f} | {format(share,'.2%') if share is not None else '—'} |")
        lines+=["","原選定支持的年度分布："+json.dumps(s["selected_totals"]["year_coverage"],ensure_ascii=False)+"。"]
    lines+=["","## 預先固定 R89 規格的 fit 支持","",
        "主執行緒在看到本次實際支持輸出之前已固定：先沿原 hedonic eligibility，再僅用案內跨度≥180日之案；群需至少1000列、30個 city|district|project 鍵、支持列總跨度≥730日且type/use非缺值。以下只核對此固定支持規格，不估新斜率，不依query修改門檻。收縮 prior200／cap0.5及以每案1/n作初始權重是後續候選規則，本稽核未執行新候選IRLS。",""]
    for stage in STAGES:
        items=result["stages"][stage]["fixed_R89_supported_groups"]
        lines.append(f"- {stage}："+("；".join(f"{x['building_type']} × {x['main_use']}（{x['support']['rows']:,}列／{x['support']['independent_project_keys']:,}案／{x['support']['combined_date_span_days']:.0f}日）" for x in items) or "沒有群滿足固定規格")+"。")
    lines+=["","## 解讀與限制","",
        "多列數不必然代表時間斜率可辨識：同案在極短時間售出的千筆交易，仍可能只有極少案內日期變化。群的整體日曆跨度也不能取代案內跨度；本稽核同時保留這兩種量、獨立案鍵数及單案列占比。支持充足僅表示可估計，不能證明斜率穩定或 Private 改善。","",
        "原全域 beta 的前三個非時間係數若固定，群內時間斜率仍可能受到隨時間改變的未觀察戶型、區域組成／建案銷售階段影響；equal-project weighting 可降低大案列數主導，但不消除混雜。三期重複窗口已被長期使用，後續任何候選仍須明記 adaptive research。","",
        "本次不更動原 reference／spatial state、模型、CSV、選版或報告簡報。群別新 slope 及完整一致的 reference／spatial 轉換由另一個固定候選實驗負責，本報告不代表候選已評估。","",
        "## 重跑與證據","",
        "- 程式：[audit.py](../outputs/round89_fit_audit/audit.py)",
        "- 聚合結果：[results.json](../outputs/round89_fit_audit/results.json)",
        "- R79 來源契約：[plan.json](../outputs/round79/plan.json)",
        "- 原公式：[round2_neighborhood.py](../src/round2_neighborhood.py)",
        "- 記錄 audit 程式 SHA、原 source SHA、全部實際讀取 input SHA、逐期 fit 索引／X／y content SHA 與原 beta。JSON 不包含逐筆價格、地址、案名或新價格向量。","",
        "重跑：.venv/Scripts/python.exe -B -X utf8 outputs/round89_fit_audit/audit.py --output-dir outputs/round89_fit_audit_rerun",
        "既有輸出拒絕覆寫。另輸出目錄會產生自身 report.md，預設本輪才写 reports/r89_fit_support_audit.md。","",
        f"實際稽核耗時 {result['elapsed_seconds']:.3f} 秒；不含後续討論。"]
    with path.open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+"\n")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="outputs/round89_fit_audit")
    args=parser.parse_args()
    out=(ROOT/args.output_dir).resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/"results.json").exists() and not (out/"execution.json").exists()
    started=now();tick=time.monotonic();provenance={};code=1;failure=None
    result={"round":89,"task":"fit_only_support_audit","started_at_utc":started,
        "new_candidate_fits":0,"global_hedonic_verification_replays":0,"new_group_slopes":0,
        "candidate_prediction_calls":0,"query_price_vectors_extracted":0,"query_MAE_computed":0,
        "test_files_read":0,"new_CSV":0,"full_train_pickle_deserialized":True,
        "fixed_R89_support_contract":{"project_span_days_min":180,"group_rows_min":1000,
            "independent_projects_min":30,"group_support_date_span_days_min":730,
            "type_use_nonmissing":True,"fixed_before_support_results":True,
            "group_key":"type|use","independent_project_key":"normalized city|district|project",
            "slope_or_shrinkage_not_fitted_in_this_audit":True},
        "stage_order":list(STAGES),"stages":{},"input_sha256":provenance,
        "audit_source_sha256":sha(__file__)}
    try:
        planpath=ROOT/"outputs/round79/plan.json"
        plan=read_locked(planpath);provenance["outputs/round79/plan.json"]=sha(planpath)
        source_names=("src/round2_neighborhood.py","src/model_common.py","src/prepare_data.py",
            "src/round6_geo.py","src/round8_spatial_reference.py","src/round3_reference.py")
        result["source_sha256"]={}
        for name in source_names:
            actual=sha(ROOT/name);assert actual==plan["source_sha256"][name]
            result["source_sha256"][name]=actual
        for name in ("data_cache/train.pkl","data_cache/train_features.pkl",KT,"reports/feature_schema.json"):
            actual=sha(ROOT/name);assert actual==plan["input_sha256"][name];provenance[name]=actual
        Xraw=pd.read_pickle(ROOT/"data_cache/train_features.pkl")
        raw=pd.read_pickle(ROOT/"data_cache/train.pkl")
        kt=np.load(ROOT/KT,allow_pickle=False)
        schema=json.loads((ROOT/"reports/feature_schema.json").read_text(encoding="utf-8"))
        assert len(raw)==len(Xraw)==455784 and raw.index.equals(pd.RangeIndex(455784)) and raw.index.equals(Xraw.index)
        assert list(Xraw.columns)==schema["columns"] and len(Xraw.columns)==87
        assert "總價元" not in Xraw and "編號" not in Xraw
        assert kt.dtype==np.dtype("int64") and kt.shape==(455255,) and (np.diff(kt)>0).all()
        assert kt.min()>=0 and kt.max()<len(raw)
        result["canonical_source_index_content_sha256"]=value_sha(kt,"<i8")
        for stage in STAGES:
            result["stages"][stage]=audit_stage(stage,plan,Xraw,raw,kt,provenance)
            result["global_hedonic_verification_replays"]+=1
            print(stage,"fit-only audit completed",flush=True)
        for name,digest in provenance.items():assert sha(ROOT/name)==digest
        result["status"]="three_fit_windows_audited_no_candidate_or_query_score"
        code=0
    except Exception:
        failure=traceback.format_exc();result["status"]="incomplete_technical_failure";result["failure"]=failure
        print(failure,flush=True)
    result.update(completed_at_utc=now(),elapsed_seconds=time.monotonic()-tick)
    write(out/"results.json",result)
    if code==0:
        rp=ROOT/"reports/r89_fit_support_audit.md" if args.output_dir=="outputs/round89_fit_audit" else out/"report.md"
        report(result,rp)
    write(out/"execution.json",{"exit_code":code,"failure":failure,"started_at_utc":started,
        "completed_at_utc":now(),"elapsed_seconds":time.monotonic()-tick,
        "source_sha256":sha(__file__),"results_sha256":sha(out/"results.json"),"automatic_retry":False})
    print(result["status"],"results_sha256",sha(out/"results.json"),flush=True)
    return code
if __name__=="__main__":raise SystemExit(main())
