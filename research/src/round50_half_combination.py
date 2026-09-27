"""One fixed post-test-X robustness comparison: half saved R49 plus half v10."""
from __future__ import annotations
import argparse, contextlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round49_combination as prior

old=prior.old; ROOT=prior.ROOT; OUT=ROOT/'outputs/round50'; REPORT=ROOT/'reports/round50_report.md'
STAGES=prior.STAGES; SCORER=prior.SCORER; MODELS=('v10','R49','v5','R50')
FORMULA='0.5 * saved_R49_complete_unrounded_TWD + 0.5 * saved_v10_complete_unrounded_TWD'
RATIONALE='Proposed after observing final test-X and model prediction disagreement; no test labels, not a pre-test blind proposal.'
need,sha,rel,write,locked=prior.need,prior.sha,prior.rel,prior.write,prior.locked


def render(result):
    names={'development':'開發','confirmation':'確認','auxiliary':'輔助'}
    lines=['# R50：全體固定 50% R49＋50% v10','',
      '在看過 final test 特徵及模型預測分歧後，針對兩個面積分支可能相關外推的風險，提出這唯一一個固定比例。全部列一律以未取整 TWD 做 0.5×R49＋0.5×v10；沒有逐ID、面積門檻、cap、比例搜尋或新模型。R49 是 v11 的歷史同法向量。',
      '', '這是事後提出的歷史比較，沒有 test y，不是事前盲測或 Private 分數。負百分比表示 MAE 降低。','',
      '| 期別 | 方法 | 整體MAE | 加權MAE |','|---|---|---:|---:|']
    for s in STAGES:
        for n in MODELS:
            m=result['stages'][s]['metrics'][n];lines.append(f"| {names[s]} | {n} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','| 期別 | 對v10 整體/加權 | 對R49 整體/加權 | 對v5 整體/加權 |','|---|---:|---:|---:|']
    for s in STAGES:
        d=result['stages'][s]['differences'];parts=[f"{d[n]['overall']['relative_change_pct']:+.4f}% / {d[n]['weighted']['relative_change_pct']:+.4f}%" for n in ('v10','R49','v5')]
        lines.append('| '+' | '.join([names[s],*parts])+' |')
    lines+=['','## 固定 test 城市×新舊建案組成','',
      '沿原固定 test-X 權重及歷史 cell 至少200筆規則；只在有支持的 test 質量內重新正規化。不是完整 test 或 Private 預期 MAE。','',
      '| 期別 | test質量涵蓋 | 未支持筆數 | v10 | R49 | v5 | R50 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        d=result['stages'][s]['test_composition'];lines.append('| '+' | '.join([names[s],f"{d['test_mass_covered']:.4%}",str(d['unsupported_test_rows']),*[f"{d['models'][n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 全部固定真價 bands','', '| 期別 | 真價 | 筆數 | v10 | R49 | v5 | R50 |','|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        for b in result['stages'][s]['price_bands']:
            lines.append('| '+' | '.join([names[s],b['band'],str(b['rows']),*['—' if b[n] is None else f"{b[n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 新舊建案與季度','', '| 期別 | 分組 | 筆數 | v10 | R49 | v5 | R50 |','|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        mm=result['stages'][s]['metrics']
        for family,keys in (('strata',('seen','new_nonmissing','missing')),('forecast_quarters',('Q1','Q2','Q3','Q4'))):
            for k in keys:
                values=['—' if mm[n][family][k]['mae'] is None else f"{mm[n][family][k]['mae']:,.2f}" for n in MODELS]
                lines.append('| '+' | '.join([names[s],k,str(mm['R50'][family][k]['rows']),*values])+' |')
    lines+=['',f"三期完整完成，fit/predict/reference/spatial 呼叫均為0，運算耗時 {result['elapsed_seconds']:.3f}秒。所有對照的完整原scorer metrics及既有R49/v10 test-mix精確重現。保留舊CSV與registry，沒有產新CSV。",
      '', '重用年度窗口且未校正多輪選型；沒有新增 bootstrap，不把 R48 或其他向量區間移用到 R50。各子群得失全列，未設新採用門檻。',
      '', '[完整結果](../outputs/round50/results.json) · [執行紀錄](../outputs/round50/execution.json) · [來源](../src/round50_half_combination.py)','']
    with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def run():
    tick=time.monotonic(); inputs={}
    def small(path):
        p=ROOT/path;v=locked(p);inputs[path]=sha(p);inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'));return v
    p48=small('outputs/round48/plan.json');p49=small('outputs/round49/input_contract.json');r49=small('outputs/round49/results.json')
    mixpath=ROOT/p49['target_composition_reused_from'];mix=json.loads(mixpath.read_text(encoding='utf8'));inputs[rel(mixpath)]=sha(mixpath)
    need(r49['checked_stages']==list(STAGES) and mix['minimum_historical_cell_rows']==200,'Prior windows incomplete')
    for p in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[p]=p48['input_sha256'][p]
    specs={}
    for s in STAGES:
        b=p49['bindings'][s]; specs[s]={k:b[k] for k in ('fit_indices','query_indices','v10','v5')}
        for p in specs[s].values():inputs[p]=p49['input_sha256'][p]
        p=f'outputs/round49/{s}/predictions.npy';specs[s]['R49']=p;inputs[p]=r49['stages'][s]['prediction_sha256']
    source=prior.merge(old.sources(),{'src/round49_combination.py':sha(ROOT/'src/round49_combination.py'),'src/round50_half_combination.py':sha(Path(__file__)),'src/round3_evaluate.py':sha(ROOT/'src/round3_evaluate.py')})
    old.prior.c6.verify_hashes(prior.merge(inputs,source))
    protected={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    Xall,yall=old.load_training_rows(p48);weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float);counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(int(counts.sum())==47073 and np.isclose(weights.sum(),1),'Test composition changed')
    result={'round':50,'status':'incomplete','formula':FORMULA,'rationale':RATIONALE,'source_sha256':source,'input_sha256':inputs,'bindings':specs,'checked_stages':[],'stages':{},'fit_calls':0,'model_predict_calls':0,'reference_or_spatial_calls':0,'new_submission_generated':False,'formal_adoption_decision':None,'protected_before':protected}
    for s in STAGES:
        b=specs[s];a=r49['stages'][s];folder=OUT/s;folder.mkdir()
        fi,qi=[np.load(ROOT/b[k+'_indices'],allow_pickle=False) for k in ('fit','query')]
        c=p48['split_contract'][s]
        for k,ind in (('fit',fi),('query',qi)):
            need(ind.dtype==np.int64 and ind.shape==(c[k+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==c[k+'_indices_content_sha256'],'Canonical indices changed')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);y=yall[qi]
        v={n:old.vector(ROOT/b[n],len(qi)) for n in ('v10','R49','v5')}
        score=lambda z:SCORER.score(y,z,X,Xq,p48['splits'][s][0],p48['target_group_weights'])
        metrics={n:score(v[n]) for n in ('v10','R49','v5')};need(all(metrics[n]==a['metrics'][n] for n in metrics),'Full control metrics differ')
        v['R50']=.5*v['R49']+.5*v['v10'];need(np.isfinite(v['R50']).all() and (v['R50']>=0).all(),'Invalid candidate TWD');metrics['R50']=score(v['R50']);old.save_array(folder/'predictions.npy',v['R50'])
        groups=prior.norm(Xq.address_city)+'|'+pd.Series(prior.group_labels(X,Xq));hist=groups.value_counts();supported=hist.reindex(weights.index,fill_value=0)>=200;coverage=float(weights[supported].sum());sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in MODELS}
        need(coverage==a['test_composition']['test_mass_covered'] and all(condition[n]==a['test_composition']['models'][n] for n in ('v10','R49')),'Existing composition replay differs')
        bands=[]
        for label,lo,hi in [('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf)]:
            m=(y>=lo)&(y<hi);bands.append({'band':label,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-y[m]).mean()) if m.any() else None for n in MODELS}})
        item={'round':50,'stage':s,'fit_rows':len(fi),'query_rows':len(qi),'dates':p48['splits'][s],'metrics':metrics,'differences':{n:old.prior.log_run.differences(metrics[n],metrics['R50']) for n in ('v10','R49','v5')},'controls_full_metrics_exact':True,'prediction_sha256':sha(folder/'predictions.npy'),'price_bands':bands,'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition,'cells':a['test_composition']['cells']}}
        write(folder/'results.json',item);result['stages'][s]=item;result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'overall':{n:m['overall_mae'] for n,m in metrics.items()},'weighted':{n:m['weighted_mae'] for n,m in metrics.items()},'differences':{n:(d['overall']['relative_change_pct'],d['weighted']['relative_change_pct']) for n,d in item['differences'].items()},'test_mix':condition}),flush=True)
    result.update(status='three_periods_completed_no_adoption_decision',elapsed_seconds=time.monotonic()-tick,protected_after={p:sha(ROOT/p) for p in protected})
    need(result['protected_after']==protected,'Protected files changed');write(OUT/'results.json',result);render(result)


def execute():
    need(not OUT.exists() and not REPORT.exists(),'R50 already exists; no overwrite');OUT.mkdir(parents=True);tick=time.monotonic();start=old.now();code=1;failure=None
    with (OUT/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:run();code=0
        except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':50,'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:p.error('--run required')
    raise SystemExit(execute())
