"""Reusable CPU diagnostics for a saved candidate against fixed v11/v12/true-v5 controls."""
from __future__ import annotations
import argparse,contextlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round49_combination as common
old=common.old;ROOT=common.ROOT;STAGES=common.STAGES;SCORER=common.SCORER
need,sha,rel,write,locked=common.need,common.sha,common.rel,common.write,common.locked
BANDS=(('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf))


def render(result,path):
    names={'development':'開發','confirmation':'確認','auxiliary':'輔助'};candidate=result['candidate'];models=('v11','v12','v5',candidate)
    lines=[f"# R{result['round']}：固定保存預測診斷",'',
      '僅重算保存候選及v11（R49）、v12（R50）、真正v5（R36）的同窗指標；未載入模型、fit、predict、reference、spatial，未選比例、改規則或新增bootstrap。',
      '', '| 期別 | 方法 | 整體MAE | 固定加權MAE |','|---|---|---:|---:|']
    for s in STAGES:
        for n in models:
            m=result['stages'][s]['metrics'][n];lines.append(f"| {names[s]} | {n} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','| 期別 | 對v11 整體/加權 | 對v12 整體/加權 | 對v5 整體/加權 |','|---|---:|---:|---:|']
    for s in STAGES:
        ds=result['stages'][s]['differences'];parts=[f"{ds[n]['overall']['relative_change_pct']:+.4f}% / {ds[n]['weighted']['relative_change_pct']:+.4f}%" for n in models[:-1]]
        lines.append('| '+' | '.join([names[s],*parts])+' |')
    lines+=['','## 固定test組成','',
      '原R49沿用的city×seen/new test-X權重；歷史query cell至少200筆，僅對有支持的test質量重正規化。不是完整test／Private預期MAE。',
      '',f'| 期別 | test涵蓋 | 未支持筆數 | v11 | v12 | v5 | {candidate} |','|---|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        d=result['stages'][s]['test_composition'];lines.append('| '+' | '.join([names[s],f"{d['test_mass_covered']:.4%}",str(d['unsupported_test_rows']),*[f"{d['models'][n]:,.2f}" for n in models]])+' |')
    lines+=['','## 五個固定真價bands','',f'| 期別 | 真價 | 筆數 | v11 | v12 | v5 | {candidate} |','|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        for b in result['stages'][s]['price_bands']:lines.append('| '+' | '.join([names[s],b['band'],str(b['rows']),*['—' if b[n] is None else f"{b[n]:,.2f}" for n in models]])+' |')
    lines+=['','窗口已多輪使用，真價bands是事後誤差描述，不能用於推論時逐筆規則。完整原scorer含新舊建案、所有季度與城市指標保留於JSON。未設新採用門檻、不產CSV；Private未知。',
      '',f"唯一CPU診斷 {result['elapsed_seconds']:.3f}秒；全部保存指標及既有test-mix對照精確重現。",
      '',f"[完整JSON](round{result['round']}_diagnostics.json) · [原結果](../outputs/round{result['round']}/results.json) · [通用診斷程式](../src/candidate_diagnostics.py)",'']
    with path.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def diagnose(number,candidate):
    tick=time.monotonic();inputs={};models=('v11','v12','v5',candidate)
    need(number>0 and candidate not in models[:-1],'Round/candidate collides with fixed controls')
    def small(name):
        p=ROOT/name;v=locked(p);inputs[name]=sha(p);inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'));return v
    p48=small('outputs/round48/plan.json');r49=small('outputs/round49/results.json');r50=small('outputs/round50/results.json');r=small(f'outputs/round{number}/results.json')
    need(r['checked_stages']==r49['checked_stages']==r50['checked_stages']==list(STAGES),'All three periods must be complete')
    mixpath='outputs/error_budget_20260926/round46_diagnostics.json';mix=json.loads((ROOT/mixpath).read_text(encoding='utf8'));inputs[mixpath]=sha(ROOT/mixpath);need(mix['minimum_historical_cell_rows']==200,'Fixed mix rule changed')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[name]=p48['input_sha256'][name]
    specs={}
    for s in STAGES:
        a=r['stages'][s];b={k:f'outputs/round{number}/{s}/{k}_indices.npy' for k in ('fit','query')}
        for name in b.values():inputs[name]=a.get('artifact_sha256',{}).get(name,sha(ROOT/name))
        name=f'outputs/round{number}/{s}/predictions.npy';digest=a.get('artifact_sha256',{}).get(name,a.get('prediction_sha256'))
        need(isinstance(digest,str) and len(digest)==64,'Candidate prediction SHA required in saved stage result');inputs[name]=digest;b[candidate]=name
        for n,previous,round_no in [('v11',r49,49),('v12',r50,50)]:
            name=f'outputs/round{round_no}/{s}/predictions.npy';b[n]=name;inputs[name]=previous['stages'][s]['prediction_sha256']
        name=r50['bindings'][s]['v5'];b['v5']=name;inputs[name]=r50['input_sha256'][name];specs[s]=b
    source=common.merge(old.sources(),{n:sha(ROOT/n) for n in ('src/candidate_diagnostics.py','src/round49_combination.py','src/round3_evaluate.py')})
    old.prior.c6.verify_hashes(common.merge(inputs,source));Xall,yall=old.load_training_rows(p48)
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float);counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(counts.sum()==47073 and np.isclose(weights.sum(),1),'Fixed test composition differs')
    result={'round':number,'candidate':candidate,'status':'incomplete','source_sha256':source,'input_sha256':inputs,'bindings':specs,'checked_stages':[],'stages':{},'model_fit_load_predict_calls':0,'new_bootstrap':False,'new_CSV':False}
    for s in STAGES:
        b=specs[s];a=r['stages'][s];c=p48['split_contract'][s];fi,qi=[np.load(ROOT/b[k],allow_pickle=False) for k in ('fit','query')]
        for k,ind in (('fit',fi),('query',qi)):
            need(ind.dtype==np.int64 and ind.shape==(c[k+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==c[k+'_indices_content_sha256'],'Canonical indices differ')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);y=yall[qi];v={n:old.vector(ROOT/b[n],len(qi)) for n in models}
        metrics={n:SCORER.score(y,v[n],X,Xq,p48['splits'][s][0],p48['target_group_weights']) for n in models};need(all(metrics[n]==a['metrics'][n] for n in models),'Full saved metrics differ')
        need(metrics['v11']==r49['stages'][s]['metrics']['R49'] and metrics['v12']==r50['stages'][s]['metrics']['R50'] and metrics['v5']==r50['stages'][s]['metrics']['v5'],'Control authority differs')
        groups=common.norm(Xq.address_city)+'|'+pd.Series(common.group_labels(X,Xq));hist=groups.value_counts();supported=hist.reindex(weights.index,fill_value=0)>=200;coverage=float(weights[supported].sum());sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in models}
        need(coverage==r50['stages'][s]['test_composition']['test_mass_covered'] and all(condition[n]==r50['stages'][s]['test_composition']['models'][key] for n,key in [('v11','R49'),('v12','R50'),('v5','v5')]),'Fixed test-mix control differs')
        bands=[]
        for name,lo,hi in BANDS:
            m=(y>=lo)&(y<hi);bands.append({'band':name,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-y[m]).mean()) if m.any() else None for n in models}})
        item={'stage':s,'fit_rows':len(fi),'query_rows':len(qi),'metrics':metrics,'all_full_saved_metrics_exact':True,'differences':{n:old.prior.log_run.differences(metrics[n],metrics[candidate]) for n in models[:-1]},'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition},'price_bands':bands}
        result['stages'][s]=item;result['checked_stages'].append(s);print(json.dumps({'stage':s,'test_mix':condition,'candidate_overall':metrics[candidate]['overall_mae'],'candidate_weighted':metrics[candidate]['weighted_mae']}),flush=True)
    result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick)
    write(ROOT/f'reports/round{number}_diagnostics.json',result);render(result,ROOT/f'reports/round{number}_diagnostics.md')


def execute(number,candidate):
    out=ROOT/f'outputs/round{number}/diagnostics';need(not out.exists() and not (ROOT/f'reports/round{number}_diagnostics.md').exists() and not (ROOT/f'reports/round{number}_diagnostics.json').exists(),'No overwrite/retry')
    out.mkdir(parents=True);tick=time.monotonic();start=old.now();code=1;failure=None
    with (out/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:diagnose(number,candidate);code=0
        except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':number,'candidate':candidate,'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(out/'run.log'),'automatic_retry':False}
    write(out/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',type=int,required=True);p.add_argument('--candidate',required=True);args=p.parse_args()
    raise SystemExit(execute(args.round,args.candidate))
