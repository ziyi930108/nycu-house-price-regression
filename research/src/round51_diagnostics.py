"""One CPU-only saved-vector diagnostic for the fixed three-seed R51 method."""
from __future__ import annotations
import argparse, contextlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round49_combination as common

old=common.old; ROOT=common.ROOT; STAGES=common.STAGES; SCORER=common.SCORER
OUT=ROOT/'outputs/round51/diagnostics'
REPORT=ROOT/'reports/round51_diagnostics.md'; JSON=ROOT/'reports/round51_diagnostics.json'
MODELS=('v10','v11','v12','R51'); SEEDS=('42','137','2026')
KEYS={'v10':'v10_control','v11':'v11_control','v12':'v12_control','R51':'nn_seed_mean_half_v10'}
AREA_QUANTILE=.99
FINAL_AREA_METADATA='outputs/round48/final/model/training.json'
FINAL_AREA_SHA='9d85822d0db3a59ae6ba297ffd8f873baa640609a2cdd8cab24456753805cae7'
need,sha,rel,write,locked=common.need,common.sha,common.rel,common.write,common.locked


def effective_area(frame):
    b=frame['building_area'].to_numpy(dtype=np.float64);p=frame['parking_area'].to_numpy(dtype=np.float64)
    return np.where(np.isfinite(b)&(b>0),b,np.where(np.isfinite(p)&(p>0),p,1.0))


def scalar_stats(y,p):
    return {'rows':len(y),'mae':float(np.abs(p-y).mean()),'mean_signed_error':float((p-y).mean()),
            'median_prediction':float(np.median(p)),'min_prediction':float(p.min()),'max_prediction':float(p.max())} if len(y) else None


def native_group(y,seed,mean,mask):
    n=int(mask.sum())
    if not n:return {'rows':0,'models':{},'dispersion':None,'pairwise':{},'all_seeds_underpredict_fraction':None,'all_seeds_overpredict_fraction':None}
    yy=y[mask];stack=np.column_stack([seed[k][mask] for k in SEEDS]);spread=stack.std(axis=1,ddof=0)
    pair={}
    for i,j in ((0,1),(0,2),(1,2)):
        d=np.abs(stack[:,i]-stack[:,j]);pair[SEEDS[i]+'_'+SEEDS[j]]={'mean_absolute_prediction_difference':float(d.mean()),'p95_absolute_prediction_difference':float(np.quantile(d,.95)),'max_absolute_prediction_difference':float(d.max())}
    return {'rows':n,'median_truth':float(np.median(yy)),
        'models':{**{k:scalar_stats(yy,seed[k][mask]) for k in SEEDS},'mean':scalar_stats(yy,mean[mask])},
        'dispersion':{'definition':'population std across the three fixed native predictions per row; not a confidence interval','mean_row_std':float(spread.mean()),'median_row_std':float(np.median(spread)),'p95_row_std':float(np.quantile(spread,.95)),'max_row_std':float(spread.max())},
        'pairwise':pair,'all_seeds_underpredict_fraction':float((stack<yy[:,None]).all(axis=1).mean()),'all_seeds_overpredict_fraction':float((stack>yy[:,None]).all(axis=1).mean())}


def render(result):
    labels={'development':'開發','confirmation':'確認','auxiliary':'輔助'}
    lines=['# R51：固定 NN seeds 平均的保存預測診斷','',
      '只讀三個固定 seed（42／137／2026）原生 TWD 與已完成組合向量，沒有重訓、模型載入／推論、挑 seed 或改權重。v11=R49，v12=R50；R51先平均 native TWD，reference／spatial各一次，再依固定原式組合。本程式不重跑這些修正。','',
      '| 期別 | 方法 | 整體MAE | 固定加權MAE |','|---|---|---:|---:|']
    for s in STAGES:
        for n in MODELS:
            m=result['stages'][s]['metrics'][n];lines.append(f"| {labels[s]} | {n} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','## 固定 test 城市×新舊建案組成','',
      '沿用R49固定test-X權重、歷史cell至少200筆；僅對有支持的test質量正規化，不是完整test／Private預期MAE。','',
      '| 期別 | 涵蓋test質量 | 未支持筆數 | v10 | v11 | v12 | R51 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        d=result['stages'][s]['test_composition'];lines.append('| '+' | '.join([labels[s],f"{d['test_mass_covered']:.4%}",str(d['unsupported_test_rows']),*[f"{d['models'][n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 五個固定真價bands','', '| 期別 | 真價 | 筆數 | v10 | v11 | v12 | R51 |','|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        for b in result['stages'][s]['price_bands']:
            lines.append('| '+' | '.join([labels[s],b['band'],str(b['rows']),*['—' if b[n] is None else f"{b[n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 純X面積範圍','', '| 期別 | fit A99(m²) | fit最大A | query最大A | query≥fit A99 | query>fit最大A |','|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        a=result['stages'][s]['area_group']
        lines.append(f"| {labels[s]} | {a['fit_A_p99']:,.4f} | {a['fit_A_max']:,.2f} | {a['query_A_max']:,.2f} | {a['query_rows_at_or_above']} | {a['query_rows_above_fit_max']} |")
    lines+=['','## 原生NN：共同偏差與seed差異','',
      '下表是NN原生總價輸出，尚未融合/reference/spatial。signed error=預測−真價；負值代表平均低估。大面積群固定取query有效面積A≥該期fit A第99百分位，A沿原R48有限正建物面積→車位面積→1規則；不按query真價或誤差挑面積門檻。',
      '', '| 期別 | 群組 | 筆數 | seed | MAE | 平均signed error | 預測中位數 |','|---|---|---:|---|---:|---:|---:|']
    for s in STAGES:
        for group,g in result['stages'][s]['native_groups'].items():
            for n,m in g['models'].items():
                lines.append(f"| {labels[s]} | {group} | {g['rows']} | {n} | {m['mae']:,.2f} | {m['mean_signed_error']:+,.2f} | {m['median_prediction']:,.2f} |")
    lines+=['','| 期別 | fit A99門檻(m²) | 群組 | 平均逐列seed標準差 | 三支同時低估 | 三支同時高估 |','|---|---:|---|---:|---:|---:|']
    for s in STAGES:
        a=result['stages'][s]
        for group,g in a['native_groups'].items():
            if g['rows']:lines.append(f"| {labels[s]} | {a['area_group']['fit_A_p99']:.4f} | {group} | {g['dispersion']['mean_row_std']:,.2f} | {g['all_seeds_underpredict_fraction']:.2%} | {g['all_seeds_overpredict_fraction']:.2%} |")
    lines+=['','## 新舊建案與季度','', '| 期別 | 群組 | 筆數 | v10 | v11 | v12 | R51 |','|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        mm=result['stages'][s]['metrics']
        for f,keys in (('strata',('seen','new_nonmissing','missing')),('forecast_quarters',('Q1','Q2','Q3','Q4'))):
            for k in keys:
                vals=['—' if mm[n][f][k]['mae'] is None else f"{mm[n][f][k]['mae']:,.2f}" for n in MODELS]
                lines.append('| '+' | '.join([labels[s],k,str(mm['R51'][f][k]['rows']),*vals])+' |')
    improvements=[result['stages'][s]['differences']['v12'][k]['relative_change_pct'] for s in STAGES for k in ('overall','weighted')]
    recommendation=('三期整體與加權均優於v12，可列為下一步候選；仍須分開看大面積／高價共同偏差，不能由平均seed保證外推安全。' if all(v<0 for v in improvements) else
      '相對v12未同時改善三期整體與加權，不能僅因seed平均就宣稱更穩健或直接優先採用；應保留各期得失。')
    lines+=['','## 建議與界線','',recommendation,
      '三支平均可平滑固定seed分歧，但若三支共享方向性誤差，平均不會自動消除它。表中seed標準差是三個固定模型的描述性分散，不是可泛化的不確定性區間。完整兩兩差異、極值與群組counts保存在JSON。',
      f"既有R48 final純X紀錄：fit最大面積約 {result['final_area_extrapolation']['fit_max_A_approx']:,.2f} m²，test最大約 {result['final_area_extrapolation']['test_max_A_approx']:,.2f} m²（{result['final_area_extrapolation']['test_max_to_fit_max_ratio']:.4f}倍；從float32 A/100還原的近似值）。這不是新的test誤差，也不能由seed平均證明共同外推風險已消失。歷史各期另列 `beyond_fit_max_area` 群，空群為0、不猜偏差。",
      '窗口已多輪重用、真價bands是事後誤差描述；沒有新bootstrap或選型後顯著性結論，Private未知。沒有更動CSV／registry或提出額外權重。',
      '',f"唯一診斷耗時 {result['elapsed_seconds']:.3f}秒；fit/model-load/predict均為0。原scorer所有control與R51保存指標精確重現。",
      '', '[機器診斷](round51_diagnostics.json) · [R51結果](../outputs/round51/results.json) · [診斷來源](../src/round51_diagnostics.py)','']
    with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def run():
    tick=time.monotonic();inputs={}
    def small(path):
        p=ROOT/path;v=locked(p);inputs[path]=sha(p);inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'));return v
    p48=small('outputs/round48/plan.json');r51=small('outputs/round51/results.json');r50=small('outputs/round50/results.json');r49=small('outputs/round49/results.json')
    need(r51['checked_stages']==r50['checked_stages']==r49['checked_stages']==list(STAGES),'Expected all three completed periods')
    mixpath='outputs/error_budget_20260926/round46_diagnostics.json';mix=json.loads((ROOT/mixpath).read_text(encoding='utf8'));inputs[mixpath]=sha(ROOT/mixpath)
    need(mix['minimum_historical_cell_rows']==200,'Original mix changed')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[name]=p48['input_sha256'][name]
    specs={}
    for s in STAGES:
        a=r51['stages'][s];spec={}
        for role,suffix in (('fit','fit_indices.npy'),('query','query_indices.npy'),('R51','predictions.npy'),('mean','nn_mean_twd.npy')):
            name=f'outputs/round51/{s}/{suffix}';inputs[name]=a['artifact_sha256'][name];spec[role]=name
        for seed in SEEDS:
            item=a['seed_predictions'][seed];spec[seed]=item['path'];inputs[item['path']]=item['sha256']
        for n,path,known in [('v12',f'outputs/round50/{s}/predictions.npy',r50['stages'][s]['prediction_sha256']),('v11',f'outputs/round49/{s}/predictions.npy',r49['stages'][s]['prediction_sha256'])]:
            spec[n]=path;inputs[path]=known
        path=r50['bindings'][s]['v10'];spec['v10']=path;inputs[path]=r50['input_sha256'][path];specs[s]=spec
    source=common.merge(old.sources(),{'src/round49_combination.py':sha(ROOT/'src/round49_combination.py'),'src/round51_diagnostics.py':sha(Path(__file__)),'src/round3_evaluate.py':sha(ROOT/'src/round3_evaluate.py')})
    inputs[FINAL_AREA_METADATA]=FINAL_AREA_SHA
    old.prior.c6.verify_hashes(common.merge(inputs,source));Xall,yall=old.load_training_rows(p48)
    final_meta=json.loads((ROOT/FINAL_AREA_METADATA).read_text(encoding='utf8'));farea=final_meta['area_factor_summary']
    need(farea['train']['rows']==467221 and farea['query']['rows']==47073,'Final X-only area metadata differs')
    final_area={'metadata_path':FINAL_AREA_METADATA,'metadata_sha256':FINAL_AREA_SHA,'fit_max_area_factor':farea['train']['max'],'test_max_area_factor':farea['query']['max'],'fit_max_A_approx':farea['train']['max']*100,'test_max_A_approx':farea['query']['max']*100,'test_max_to_fit_max_ratio':farea['query']['max']/farea['train']['max'],'test_labels_used':False,'new_final_predictions_loaded':False}
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float);counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(int(counts.sum())==47073 and np.isclose(weights.sum(),1),'Test composition differs')
    result={'round':51,'status':'incomplete','checked_stages':[],'stages':{},'source_sha256':source,'input_sha256':inputs,'fit_calls':0,'model_load_calls':0,'model_predict_calls':0,'new_bootstrap':False,'new_seed_or_weight_search':False,'final_area_extrapolation':final_area,'fixed_area_group':{'fit_quantile':AREA_QUANTILE,'quantile_method':'linear','rule':'query_A >= fit_A_p99','effective_area':'finite positive building_area else finite positive parking_area else 1'}}
    for s in STAGES:
        a=r51['stages'][s];b=specs[s];spec=p48['split_contract'][s];fi,qi=[np.load(ROOT/b[k],allow_pickle=False) for k in ('fit','query')]
        for k,ind in (('fit',fi),('query',qi)):
            need(ind.dtype==np.int64 and ind.shape==(spec[k+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==spec[k+'_indices_content_sha256'],'Canonical indices differ')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);y=yall[qi]
        v={n:old.vector(ROOT/b[n],len(qi)) for n in MODELS};seed={n:old.vector(ROOT/b[n],len(qi)) for n in SEEDS};mean=old.vector(ROOT/b['mean'],len(qi))
        old.exact((seed['42']+seed['137']+seed['2026'])/3.0,mean,'Fixed seed mean')
        score=lambda z:SCORER.score(y,z,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={n:score(v[n]) for n in MODELS}
        need(all(metrics[n]==a['metrics'][KEYS[n]] for n in MODELS),'R51 full scorer replay differs')
        need(metrics['v12']==r50['stages'][s]['metrics']['R50'] and metrics['v11']==r49['stages'][s]['metrics']['R49'],'Saved comparator metrics differ')
        groups=common.norm(Xq.address_city)+'|'+pd.Series(common.group_labels(X,Xq));hist=groups.value_counts();support=hist.reindex(weights.index,fill_value=0)>=200;coverage=float(weights[support].sum());sw=weights[support]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in MODELS}
        need(coverage==r50['stages'][s]['test_composition']['test_mass_covered'] and all(condition[n]==r50['stages'][s]['test_composition']['models'][k] for n,k in (('v10','v10'),('v11','R49'),('v12','R50'))),'Mix control differs')
        bands=[]
        for label,lo,hi in [('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf)]:
            m=(y>=lo)&(y<hi);bands.append({'band':label,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-y[m]).mean()) if m.any() else None for n in MODELS}})
        af,aq=effective_area(X),effective_area(Xq);threshold=float(np.quantile(af,AREA_QUANTILE,method='linear'));large=aq>=threshold
        masks={'all':np.ones(len(qi),dtype=bool),'>=100m':y>=1e8,'large_area_fit_p99':large,'beyond_fit_max_area':aq>af.max()}
        native={name:native_group(y,seed,mean,mask) for name,mask in masks.items()}
        item={'stage':s,'fit_rows':len(fi),'query_rows':len(qi),'metrics':metrics,'all_full_saved_metrics_exact':True,'native_mean_bitwise_exact':True,
          'differences':{n:old.prior.log_run.differences(metrics[n],metrics['R51']) for n in ('v10','v11','v12')},
          'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~support].sum()),'models':condition},
          'price_bands':bands,'area_group':{'fit_A_p99':threshold,'fit_rows_at_or_above':int((af>=threshold).sum()),'query_rows_at_or_above':int(large.sum()),'fit_A_max':float(af.max()),'query_A_max':float(aq.max()),'query_rows_above_fit_max':int(masks['beyond_fit_max_area'].sum())},'native_groups':native}
        result['stages'][s]=item;result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'overall':{n:m['overall_mae'] for n,m in metrics.items()},'weighted':{n:m['weighted_mae'] for n,m in metrics.items()},'test_mix':condition,'fit_A99':threshold}),flush=True)
    result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick);write(JSON,result);render(result)


def execute():
    need(not OUT.exists() and not REPORT.exists() and not JSON.exists(),'Existing diagnostic; preserve it')
    OUT.mkdir(parents=True);start=old.now();tick=time.monotonic();code=1;failure=None
    with (OUT/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:run();code=0
        except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':51,'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('--run required')
    raise SystemExit(execute())
