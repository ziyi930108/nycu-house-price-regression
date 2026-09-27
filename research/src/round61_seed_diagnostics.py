"""Fixed saved-output R61 seed diagnostics; no model operations or bootstrap."""
from __future__ import annotations
import contextlib, json, re, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import candidate_diagnostics_v3 as base

common=base.common; old=base.old; ROOT=base.ROOT; STAGES=base.STAGES; SCORER=base.SCORER
need,sha,rel,write,locked=base.need,base.sha,base.rel,base.write,base.locked
SEEDS=('42','137','2026')
MODELS=('R61','R61_ordinary','v16','v15','v13','v11','v12','v5',
        *('ordinary_seed'+s for s in SEEDS),*('recency_seed'+s for s in SEEDS))
MAIN=('R61','R61_ordinary','v16','v15','v5')
OUT=ROOT/'outputs/round61/diagnostics'


def relative(value):
    p=Path(value.replace('\\','/'))
    need(not p.is_absolute() and '..' not in p.parts,'Expected a root-relative artifact path')
    return p.as_posix()


def hash_map(mapping):
    result={}
    for name,digest in mapping.items():
        name=relative(name)
        need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Invalid saved artifact SHA')
        need(name not in result or result[name]==digest,'Conflicting normalized paths')
        result[name]=digest
    return result


def range_of(values):
    return {'by_seed':values,'minimum':min(values.values()),'maximum':max(values.values()),
            'span':max(values.values())-min(values.values()),'is_confidence_interval':False}


def render(result,path):
    names={'development':'開發','confirmation':'確認','auxiliary':'輔助'}
    table=lambda cells:'| '+' | '.join(cells)+' |'
    fmt=lambda x:'—' if x is None else f'{x:,.2f}'
    lines=['# R61：固定三種子保存預測診斷','',
        '僅使用保存的完整TWD向量與原canonical標籤／scorer；沒有模型載入、訓練、predict、reference或spatial重播。各seed的native MAE直接引用本輪保存結果，未重新載入native向量。',
        'R61為recency三種子native平均後的完整候選；R61_ordinary為ordinary三種子版本。單seed比較也都是完整pipeline結果，不是在這裡把最終向量重新平均。',
        '',table(['期別','方法','整體MAE','固定加權MAE','固定test組成MAE']),table(['---','---','---:','---:','---:'])]
    for s in STAGES:
        a=result['stages'][s]
        for n in MAIN:
            m=a['metrics'][n]
            lines.append(table([names[s],n,fmt(m['overall_mae']),fmt(m['weighted_mae']),fmt(a['test_composition']['models'][n])]))
    lines+=['','test組成沿既定city×seen/new、歷史格至少200筆，只對有支持的test質量重正規化；不是完整Private預期MAE。']
    for s in STAGES:
        t=result['stages'][s]['test_composition']
        lines.append(f"{names[s]}覆蓋 {t['test_mass_covered']:.4%}，未支持 {t['unsupported_test_rows']:,} 筆。")
    lines+=['','## 同seed：ordinary → recency','',
        '每格依序為ordinary／recency／差額（recency−ordinary），負值為較低MAE；native欄是保存的原始NN總價MAE。',
        '',table(['期別','seed','完整整體MAE','完整加權MAE','保存native MAE']),table(['---','---','---:','---:','---:'])]
    for s in STAGES:
        for seed in SEEDS:
            p=result['stages'][s]['paired_seeds'][seed]
            cells=[]
            for key in ('overall_mae','weighted_mae','native_mae'):
                cells.append(f"{fmt(p['ordinary'][key])} / {fmt(p['recency'][key])} / {p['delta_recency_minus_ordinary'][key]:+,.2f}")
            lines.append(table([names[s],seed,*cells]))
    lines+=['','## 各方法的三seed範圍','',
        '以下只是三個固定seed的最小值至最大值，不是Private信賴區間，也不是獨立資料重抽樣。',
        '',table(['期別','方法','完整整體MAE範圍','完整加權MAE範圍','保存native MAE範圍']),table(['---','---','---:','---:','---:'])]
    for s in STAGES:
        for method in ('ordinary','recency'):
            r=result['stages'][s]['seed_ranges'][method]
            lines.append(table([names[s],method,*[f"{fmt(r[k]['minimum'])}–{fmt(r[k]['maximum'])}" for k in ('overall_mae','weighted_mae','native_mae')]]))
    lines+=['','## 兩個mean的分組差異','',
        '列相對各自seed42對照的MAE變化：recency mean對v16、ordinary mean對v15。正值是退步，缺樣本列不解讀。',
        '',table(['期別','群組','筆數','R61對v16','R61_ordinary對v15']),table(['---','---','---:','---:','---:'])]
    for s in STAGES:
        a=result['stages'][s]
        for field in ('strata','forecast_quarters'):
            for name,v in a['metrics']['R61'][field].items():
                changes=[]
                for model,control in (('R61','v16'),('R61_ordinary','v15')):
                    delta=a['mean_differences'][model][control][field][name]['relative_change_pct']
                    changes.append('—' if delta is None else f'{delta:+.4f}%')
                lines.append(table([names[s],name,str(v['rows']),*changes]))
    lines+=['','## 五個固定真價bands','',table(['期別','真價','筆數',*MAIN]),table(['---','---','---:',*['---:']*len(MAIN)])]
    for s in STAGES:
        for b in result['stages'][s]['price_bands']:
            lines.append(table([names[s],b['band'],str(b['rows']),*[fmt(b[n]) for n in MAIN]]))
    lines+=['','全14個方法的完整metrics、test組成與價位群保留於JSON。沒有搜尋seed、權重、門檻或另做bootstrap；窗口已多輪用於選型，Private未知。',
        '',f"本次CPU診斷 {result['elapsed_seconds']:.3f}秒；所有完整保存metrics精確重現，v15/v16及原三個test組成對照精確吻合。",
        '', '[完整JSON](round61_diagnostics.json) · [R61原結果](../outputs/round61/results.json) · [診斷來源](../src/round61_seed_diagnostics.py)','']
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))


def diagnose():
    tick=time.monotonic(); inputs={}
    def small(name):
        p=ROOT/name; value=locked(p); inputs[name]=sha(p)
        inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256')); return value
    r=small('outputs/round61/results.json'); p48=small('outputs/round48/plan.json')
    r50=small('outputs/round50/results.json')
    history={'v16':(small('reports/round60_diagnostics.json'),'R60'),
             'v15':(small('reports/round54_diagnostics.json'),'R54')}
    need(r['round']==61 and r['checked_stages']==list(STAGES),'R61 must have all three completed stages')
    inherited=hash_map(r['input_sha256']); specs={}
    for s in STAGES:
        a=r['stages'][s]; artifact=hash_map(a['artifact_sha256'])
        declared=common.merge(inherited,artifact)
        need(set(MODELS)<=set(a['metrics']) and set(MODELS)<=set(a['prediction_paths']),'Missing fixed model labels')
        b={n:relative(a['prediction_paths'][n]) for n in MODELS}
        b.update({k:f'outputs/round61/{s}/{k}_indices.npy' for k in ('fit','query')})
        need(b['ordinary_seed42']==b['v15'] and b['recency_seed42']==b['v16'],'Seed42 must directly reuse v15/v16 paths')
        for path in b.values():
            need(path in declared,'Artifact lacks a saved SHA: '+path); inputs[path]=declared[path]
        specs[s]=b
    mixpath='outputs/error_budget_20260926/round46_diagnostics.json'
    mix=json.loads((ROOT/mixpath).read_text(encoding='utf-8')); inputs[mixpath]=sha(ROOT/mixpath)
    need(mix['minimum_historical_cell_rows']==200,'Fixed test composition rule changed')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[name]=p48['input_sha256'][name]
    source=common.merge(old.sources(),{rel(p):sha(p) for p in (Path(__file__),Path(base.__file__),Path(common.__file__),ROOT/'src/round3_evaluate.py')})
    old.prior.c6.verify_hashes(common.merge(inputs,source)); Xall,yall=old.load_training_rows(p48)
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float)
    counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(counts.sum()==47073 and np.isclose(weights.sum(),1),'Fixed test composition differs')
    result={'round':61,'status':'incomplete','candidate':'R61','model_order':list(MODELS),'seeds':list(SEEDS),
        'source_sha256':source,'input_sha256':inputs,'bindings':specs,'checked_stages':[],'stages':{},
        'model_fit_load_predict_calls':0,'native_MAE_recomputed':False,'new_bootstrap':False,'new_CSV':False,
        'seed_ranges_are_confidence_intervals':False}
    for s in STAGES:
        a=r['stages'][s]; b=specs[s]; c=p48['split_contract'][s]
        fi,qi=[np.load(ROOT/b[k],allow_pickle=False) for k in ('fit','query')]
        for k,ind in (('fit',fi),('query',qi)):
            need(ind.dtype==np.int64 and ind.shape==(c[k+'_rows'],) and len(ind)==a[k+'_rows'] and
                 (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and
                 SCORER.value_sha(ind,'<i8')==c[k+'_indices_content_sha256'],'Canonical indices differ')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True); y=yall[qi]
        vectors={path:old.vector(ROOT/path,len(qi)) for path in set(b[n] for n in MODELS)}
        v={n:vectors[b[n]] for n in MODELS}
        metrics_by_path={path:SCORER.score(y,p,X,Xq,p48['splits'][s][0],p48['target_group_weights']) for path,p in vectors.items()}
        metrics={n:metrics_by_path[b[n]] for n in MODELS}
        need(all(metrics[n]==a['metrics'][n] for n in MODELS),'Full saved metrics differ')
        for n,(h,key) in history.items():need(metrics[n]==h['stages'][s]['metrics'][key],'Historical v15/v16 metrics differ')
        groups=common.norm(Xq.address_city)+'|'+pd.Series(common.group_labels(X,Xq)); hist=groups.value_counts()
        supported=hist.reindex(weights.index,fill_value=0)>=200; coverage=float(weights[supported].sum()); sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in MODELS}
        t50=r50['stages'][s]['test_composition']
        need(coverage==t50['test_mass_covered'] and all(condition[n]==t50['models'][k] for n,k in [('v11','R49'),('v12','R50'),('v5','v5')]),'Original test-mix controls differ')
        for n,(h,key) in history.items():
            saved=h['stages'][s]['test_composition']
            need(saved['minimum_historical_cell_rows']==200 and saved['test_mass_covered']==coverage and
                 saved['unsupported_test_rows']==int(counts[~supported].sum()) and saved['models'][key]==condition[n],'v15/v16 test-mix differs')
        bands=[]
        for name,lo,hi in base.BANDS:
            mask=(y>=lo)&(y<hi)
            bands.append({'band':name,'rows':int(mask.sum()),**{n:float(np.abs(v[n][mask]-y[mask]).mean()) if mask.any() else None for n in MODELS}})
        native=a['native_MAE']; paired={}; ranges={}
        for method in ('ordinary','recency'):
            need(set(native[method])==set(SEEDS) and all(np.isfinite(native[method][seed]) and native[method][seed]>=0 for seed in SEEDS),'Invalid saved native MAE')
            ranges[method]={key:range_of({seed:float(native[method][seed] if key=='native_mae' else metrics[method+'_seed'+seed][key]) for seed in SEEDS}) for key in ('overall_mae','weighted_mae','native_mae')}
        for seed in SEEDS:
            pair={method:{key:float(native[method][seed] if key=='native_mae' else metrics[method+'_seed'+seed][key]) for key in ('overall_mae','weighted_mae','native_mae')} for method in ('ordinary','recency')}
            pair['delta_recency_minus_ordinary']={key:pair['recency'][key]-pair['ordinary'][key] for key in pair['ordinary']}
            paired[seed]=pair
        item={'stage':s,'fit_rows':len(fi),'query_rows':len(qi),'metrics':metrics,'all_full_saved_metrics_exact':True,
            'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition},
            'historical_v15_v16_test_mix_exact':True,'price_bands':bands,'saved_native_MAE':native,
            'paired_seeds':paired,'seed_ranges':ranges,
            'mean_differences':{n:{control:old.prior.log_run.differences(metrics[control],metrics[n]) for control in ('v16','v15','v13','v11','v12','v5')} for n in ('R61','R61_ordinary')}}
        result['stages'][s]=item; result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'R61':metrics['R61']['overall_mae'],'R61_ordinary':metrics['R61_ordinary']['overall_mae'],'test_mix':{n:condition[n] for n in MAIN}}),flush=True)
    result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick)
    write(ROOT/'reports/round61_diagnostics.json',result); render(result,ROOT/'reports/round61_diagnostics.md')


def execute():
    need(not OUT.exists() and not (ROOT/'reports/round61_diagnostics.json').exists() and not (ROOT/'reports/round61_diagnostics.md').exists(),'No overwrite/retry')
    OUT.mkdir(parents=True); tick=time.monotonic(); start=old.now(); code=1; failure=None
    with (OUT/'run.log').open('x',encoding='utf-8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:diagnose(); code=0
        except Exception:failure=traceback.format_exc(); print(failure,flush=True)
    event={'round':61,'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),
        'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),
        'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',event); print(json.dumps(event),flush=True); return code


if __name__=='__main__':
    raise SystemExit(execute())
