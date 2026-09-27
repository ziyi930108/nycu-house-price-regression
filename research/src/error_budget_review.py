"""Post-hoc error contribution review of saved v5/v7; no fitting or submissions."""
from pathlib import Path
import hashlib, json, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from round3_evaluate import group_labels

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/error_budget_20260926'
REPORT=ROOT/'reports/error_budget_20260926.md'
STAGES=('development','confirmation','auxiliary')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(frame, group):
    all_error=frame.a5.sum()
    out=frame.groupby(group,dropna=False,observed=True).agg(rows=('a5','size'),
        v5_mae=('a5','mean'),v7_mae=('a7','mean'),v5_signed_error=('e5','mean'),
        v7_signed_error=('e7','mean'),v5_abs_error_sum=('a5','sum'),
        v7_abs_error_sum=('a7','sum')).reset_index()
    out['row_percent']=out.rows/len(frame)*100
    out['v5_error_percent']=out.v5_abs_error_sum/all_error*100
    out['v5_contribution_to_overall_mae']=out.v5_abs_error_sum/len(frame)
    out['v7_minus_v5_mae']=out.v7_mae-out.v5_mae
    return out.sort_values('v5_abs_error_sum',ascending=False).to_dict('records')

def main():
    started=time.monotonic()
    OUT.mkdir(exist_ok=False)
    keep=np.load(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
    xraw=pd.read_pickle(ROOT/'data_cache/train_features.pkl')
    raw=pd.read_pickle(ROOT/'data_cache/train.pkl')
    assert len(xraw)==len(raw)==455784 and len(keep)==455255
    X=xraw.iloc[keep].reset_index(drop=True)
    y=raw['總價元'].to_numpy(dtype='float64')[keep]
    del raw,xraw
    test=pd.read_pickle(ROOT/'data_cache/test_features.pkl')
    p39=read(ROOT/'outputs/round39/evaluation/plan.json')
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Post-hoc error contribution, not a new model or untouched validation',
        'private_labels_available':False,'fit_count':0,'prediction_calls':0,'source_sha256':sha(Path(__file__)),
        'data_identities_from_existing_plan':{k:v for k,v in p39['input_sha256'].items() if k in ['data_cache/train.pkl','data_cache/train_features.pkl','outputs/round6/geography/development/train_source_cache_indices.npy']},
        'actual_price_groups_are_diagnostic_only_not_deployable_rules':True,'stages':{}}
    lines=['# 主要誤差來源：保存的 v5／v7 預測','',
        '本報告在既有三個年度上做事後診斷，沒有重新訓練、預測或修改提交。所有價格標籤皆來自原訓練資料的歷史驗證區段，沒有 test／Private 標籤。這些窗口已反覆使用，不能當作新的盲測。',
        '', '正的平均有號誤差代表高估，負值代表低估。「誤差占比」是該群絕對誤差總和占全部誤差的比例，不是只看小群的高 MAE。真實價格分組只供診斷，部署時無法按未知真實價格分流。','']
    rows=[]
    for stage in STAGES:
        d36=ROOT/'outputs/round36'/stage
        d39=ROOT/'outputs/round39/evaluation'/stage
        r36,r39=read(d36/'results.json'),read(d39/'results.json')
        fi,qi=[np.load(d39/(key+'_indices.npy'),allow_pickle=False) for key in ['fit','query']]
        for key,arr in [('fit',fi),('query',qi)]: assert np.array_equal(arr,np.load(d36/(key+'_indices.npy'),allow_pickle=False))
        assert not np.intersect1d(fi,qi).size
        q=X.iloc[qi].reset_index(drop=True)
        yy=y[qi]
        paths={'v5':d36/'log_spatial_combination.npy','v7':d39/'city_road_spatial_half_raw_half_log.npy'}
        preds={}
        for name,path in paths.items():
            source=r36 if name=='v5' else r39
            assert sha(path)==source['artifact_sha256'][path.relative_to(ROOT).as_posix()]
            preds[name]=np.load(path,allow_pickle=False)
            assert preds[name].shape==(len(q),) and np.isfinite(preds[name]).all()
        e5,e7=preds['v5']-yy,preds['v7']-yy
        a5,a7=np.abs(e5),np.abs(e7)
        assert abs(a5.mean()-r36['metrics']['log_spatial_combination']['overall_mae'])<1e-6
        assert abs(a7.mean()-r39['metrics']['city_road_spatial_half_raw_half_log']['overall_mae'])<1e-6
        geo=pd.read_pickle(ROOT/p39['prepared_geography'][stage]['features']['query']['path'])
        bins=[-np.inf,1e7,2e7,5e7,1e8,np.inf]
        labels=['<1000萬','1000–2000萬','2000–5000萬','5000萬–1億','>=1億']
        f=pd.DataFrame({'a5':a5,'a7':a7,'e5':e5,'e7':e7,
            'city':q.address_city.fillna('__MISSING__').astype(str),
            'building_type':q.building_type.fillna('__MISSING__').astype(str),
            'main_use':q.main_use.fillna('__MISSING__').astype(str),
            'project_group':group_labels(X.iloc[fi].reset_index(drop=True),q),
            'true_price_band':pd.cut(yy,bins,right=False,labels=labels).astype(str),
            'v5_predicted_price_band':pd.cut(preds['v5'],bins,right=False,labels=labels).astype(str),
            'has_geography':geo.geo_has_position.map({0.:'missing',1.:'located'}),
            'calendar_month':q.trade_year.astype(int).astype(str)+'-'+q.trade_month.astype(int).astype(str).str.zfill(2)})
        groups={key:summarize(f,key) for key in ['city','building_type','main_use','project_group','true_price_band','v5_predicted_price_band','has_geography','calendar_month']}
        concentration=[]
        for fraction in [.001,.01,.05,.1]:
            n=max(1,int(np.ceil(len(q)*fraction)))
            contribution=float(np.sort(a5)[-n:].sum()/a5.sum()*100)
            concentration.append({'top_error_fraction':fraction,'rows':n,'v5_abs_error_percent':contribution})
        record={'query_rows':len(q),'v5_mae':float(a5.mean()),'v7_mae':float(a7.mean()),
                'saved_mae_reproduced':True,'v5_signed_error':float(e5.mean()),'v7_signed_error':float(e7.mean()),
                'error_concentration':concentration,'groups':groups,'prediction_sha256':{name:sha(path) for name,path in paths.items()}}
        result['stages'][stage]=record
        lines += [f'## {stage}：{len(q):,} 筆','',f'v5 MAE {a5.mean():,.0f} 元；v7 MAE {a7.mean():,.0f} 元。v5 誤差最大 1% 的資料占全部絕對誤差 {concentration[1]["v5_abs_error_percent"]:.2f}%。','']
        for dimension,title in [('city','城市'),('project_group','建案是否見過'),('true_price_band','真實價格'),('building_type','建物型態')]:
            lines += [f'{title}（依誤差貢獻排序）：','', '| 分組 | 筆數 | 資料占比 | 誤差占比 | v5 MAE | v7 MAE | v5 平均高估／低估 |','|---|---:|---:|---:|---:|---:|---:|']
            for row in groups[dimension][:6]:
                lines.append(f'| {row[dimension]} | {row["rows"]:,} | {row["row_percent"]:.2f}% | {row["v5_error_percent"]:.2f}% | {row["v5_mae"]:,.0f} | {row["v7_mae"]:,.0f} | {row["v5_signed_error"]:+,.0f} |')
            lines += ['']
        for dimension,records in groups.items():
            for row in records:
                row=dict(row); group=row.pop(dimension)
                rows.append({'stage':stage,'dimension':dimension,'group':group,**row})
    test_city=test.address_city.fillna('__MISSING__').astype(str).value_counts(dropna=False)
    result['test_X_only_city_composition']=[{'city':str(k),'rows':int(v),'percent':float(v/len(test)*100)} for k,v in test_city.items()]
    result['elapsed_seconds']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT/'group_contributions.csv',index=False,encoding='utf-8-sig')
    lines += ['完整城市、用途、地理覆蓋、月份與價格分組數字見 [JSON](../outputs/error_budget_20260926/results.json)／[分組表](../outputs/error_budget_20260926/group_contributions.csv)。', '', f'本次耗時 {result["elapsed_seconds"]:.2f} 秒，訓練 0 次、推論 0 次。']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(REPORT),'elapsed_seconds':result['elapsed_seconds'],'stages':{k:{'mae':v['v5_mae'],'concentration':v['error_concentration'],'top_city':v['groups']['city'][:3],'project_groups':v['groups']['project_group']} for k,v in result['stages'].items()}},ensure_ascii=False))

if __name__=='__main__': main()
