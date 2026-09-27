"""R76 compatibility adapter: unchanged diagnostics, original control authorities.

Derived from candidate_diagnostics_v3.py SHA256
49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128.
R76 main results omit v11/v12/v5; their full metric checks use R49/R50/R36.
No scoring, test-mix, price-band or model-selection formula is changed.
"""
from __future__ import annotations
import argparse, contextlib, json, re, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round49_combination as common

old=common.old; ROOT=common.ROOT; STAGES=common.STAGES; SCORER=common.SCORER
need,sha,rel,write,locked=common.need,common.sha,common.rel,common.write,common.locked
BANDS=(('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf))
DEFAULT_CONTROLS=(('v11',49,'R49'),('v12',50,'R50'))


def extra_control(value):
    parts=value.split(':')
    if len(parts)!=3 or not all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',x) for x in (parts[0],parts[-1])) or not parts[1].isdigit() or int(parts[1])<=0:
        raise argparse.ArgumentTypeError('Expected label:positive_round:metric_key, e.g. v15:54:R54')
    return parts[0],int(parts[1]),parts[2]


def validate_controls(number,candidate,extras):
    need(number==76 and candidate=='R76','This compatibility adapter is restricted to R76')
    labels=['v11','v12','v5',*[x[0] for x in extras],candidate]
    need(number>0 and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',candidate) is not None,'Invalid round/candidate')
    need(len(labels)==len(set(labels)),'Duplicate or colliding candidate/control label')
    need(all(n!=number for _,n,_ in extras),'Extra control cannot be the current round')


def artifact_hash(stage,path,prediction=False):
    mapping={}
    for key,value in stage.get('artifact_sha256',{}).items():
        key=key.replace('\\','/')
        need(key not in mapping or mapping[key]==value,'Conflicting normalized artifact paths')
        mapping[key]=value
    digest=mapping.get(path)
    if prediction:
        direct=stage.get('prediction_sha256')
        need(digest is None or direct is None or digest==direct,'Prediction SHA declarations differ')
        digest=digest or direct
        need(isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest) is not None,'Saved prediction SHA required')
    return digest if digest is not None else sha(ROOT/path)


def render(result,path):
    names={'development':'開發','confirmation':'確認','auxiliary':'輔助'}
    models=result['model_order']; candidate=result['candidate']; controls=models[:-1]
    table=lambda cells:'| '+' | '.join(cells)+' |'
    lines=[f"# R{result['round']}：固定保存預測診斷",'',
        '僅比較保存候選與指定對照的同窗指標；未載入模型、fit、predict、reference、spatial，未選比例、改規則或新增bootstrap。',
        '', '對照：'+ '；'.join(f"{x['label']}＝R{x['round']}／{x['metric_key']}" for x in result['control_specs'])+'。',
        '',table(['期別','方法','整體MAE','固定加權MAE']),table(['---','---','---:','---:'])]
    for s in STAGES:
        for n in models:
            m=result['stages'][s]['metrics'][n]
            lines.append(table([names[s],n,f"{m['overall_mae']:,.2f}",f"{m['weighted_mae']:,.2f}"]))
    lines+=['',table(['期別',*[f'對{n} 整體/加權' for n in controls]]),table(['---',*['---:']*len(controls)])]
    for s in STAGES:
        ds=result['stages'][s]['differences']
        lines.append(table([names[s],*[f"{ds[n]['overall']['relative_change_pct']:+.4f}% / {ds[n]['weighted']['relative_change_pct']:+.4f}%" for n in controls]]))
    lines+=['','## 固定test組成','',
        '原R49沿用的city×seen/new test-X權重；歷史query cell至少200筆，僅對有支持的test質量重正規化。不是完整test／Private預期MAE。',
        '',table(['期別','test涵蓋','未支持筆數',*models]),table(['---','---:','---:',*['---:']*len(models)])]
    for s in STAGES:
        d=result['stages'][s]['test_composition']
        lines.append(table([names[s],f"{d['test_mass_covered']:.4%}",str(d['unsupported_test_rows']),*[f"{d['models'][n]:,.2f}" for n in models]]))
    lines+=['','## 五個固定真價bands','',table(['期別','真價','筆數',*models]),table(['---','---','---:',*['---:']*len(models)])]
    for s in STAGES:
        for b in result['stages'][s]['price_bands']:
            lines.append(table([names[s],b['band'],str(b['rows']),*['—' if b[n] is None else f"{b[n]:,.2f}" for n in models]]))
    lines+=['','窗口已多輪使用，真價bands是事後誤差描述，不能用於推論時逐筆規則。完整原scorer含新舊建案、所有季度與城市指標保留於JSON。未設新採用門檻、不產CSV；Private未知。',
        '',f"唯一CPU診斷 {result['elapsed_seconds']:.3f}秒；全部候選及對照的保存完整指標精確重現，三個固定test-mix對照不變。額外既有diagnostics只在存在時核對，核對狀態見JSON。",
        '',f"[完整JSON](round{result['round']}_diagnostics.json) · [原結果](../outputs/round{result['round']}/results.json) · [R76 相容診斷程式](../src/round76_diagnostics.py)",'']
    with path.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def diagnose(number,candidate,extras=()):
    validate_controls(number,candidate,extras)
    tick=time.monotonic(); inputs={}; cache={}; controls=(*DEFAULT_CONTROLS,*extras)
    models=('v11','v12','v5',*[x[0] for x in extras],candidate)
    def small(name):
        if name not in cache:
            p=ROOT/name; cache[name]=locked(p); inputs[name]=sha(p)
            inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'))
        return cache[name]
    p48=small('outputs/round48/plan.json'); r=small(f'outputs/round{number}/results.json')
    authority={label:small(f'outputs/round{n}/results.json') for label,n,_ in controls}; r50=authority['v12']
    r36=small('outputs/round36/results.json')
    need(all(v['checked_stages']==list(STAGES) for v in [r,r36,*authority.values()]),'All three periods must be complete')
    prior_diagnostics={}
    for label,n,key in extras:
        path=f'reports/round{n}_diagnostics.json'
        if (ROOT/path).exists():
            prior_diagnostics[label]=(small(path),key)
    mixpath='outputs/error_budget_20260926/round46_diagnostics.json'
    mix=json.loads((ROOT/mixpath).read_text(encoding='utf8')); inputs[mixpath]=sha(ROOT/mixpath)
    need(mix['minimum_historical_cell_rows']==200,'Fixed mix rule changed')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[name]=p48['input_sha256'][name]
    specs={}; extra_indices={}
    for s in STAGES:
        a=r['stages'][s]; b={k:f'outputs/round{number}/{s}/{k}_indices.npy' for k in ('fit','query')}
        for name in b.values():inputs[name]=artifact_hash(a,name)
        name=f'outputs/round{number}/{s}/predictions.npy'; inputs[name]=artifact_hash(a,name,True); b[candidate]=name
        for label,n,key in controls:
            previous=authority[label]['stages'][s]; need(key in previous['metrics'],'Control metric key absent')
            name=f'outputs/round{n}/{s}/predictions.npy'; b[label]=name; inputs[name]=artifact_hash(previous,name,True)
            if label not in ('v11','v12'):
                paths={k:f'outputs/round{n}/{s}/{k}_indices.npy' for k in ('fit','query')}
                extra_indices[s,label]=paths
                for path in paths.values():inputs[path]=artifact_hash(previous,path)
        name=r50['bindings'][s]['v5']
        need(name==f'outputs/round36/{s}/log_spatial_combination.npy','True v5 source path differs')
        b['v5']=name; inputs[name]=r50['input_sha256'][name]; specs[s]=b
    source=common.merge(old.sources(),{n:sha(ROOT/n) for n in ('src/round76_diagnostics.py','src/candidate_diagnostics_v3.py','src/round49_combination.py','src/round3_evaluate.py')})
    need(source['src/candidate_diagnostics_v3.py']=='49cf3a89e8a1d60defb8f49bc63ef0f8f88ded528af1e3687684d70c8a4ae128','Original generic source identity differs')
    old.prior.c6.verify_hashes(common.merge(inputs,source)); Xall,yall=old.load_training_rows(p48)
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float)
    counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(counts.sum()==47073 and np.isclose(weights.sum(),1),'Fixed test composition differs')
    result={'round':number,'candidate':candidate,'status':'incomplete','model_order':list(models),
        'control_specs':[{'label':n,'round':rn,'metric_key':key} for n,rn,key in (*DEFAULT_CONTROLS,('v5',36,'log_spatial_combination'),*extras)],
        'source_sha256':source,'input_sha256':inputs,'bindings':specs,'checked_stages':[],'stages':{},'model_fit_load_predict_calls':0,'new_bootstrap':False,'new_CSV':False}
    for s in STAGES:
        b=specs[s]; a=r['stages'][s]; c=p48['split_contract'][s]
        fi,qi=[np.load(ROOT/b[k],allow_pickle=False) for k in ('fit','query')]
        for k,ind in (('fit',fi),('query',qi)):
            need(ind.dtype==np.int64 and ind.shape==(c[k+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==c[k+'_indices_content_sha256'],'Canonical indices differ')
            for label,_,_ in extras:
                other=np.load(ROOT/extra_indices[s,label][k],allow_pickle=False)
                need(other.dtype==np.int64 and np.array_equal(other,ind),'Extra control canonical indices differ')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True); y=yall[qi]
        v={n:old.vector(ROOT/b[n],len(qi)) for n in models}
        metrics={n:SCORER.score(y,v[n],X,Xq,p48['splits'][s][0],p48['target_group_weights']) for n in models}
        need(metrics[candidate]==a['metrics'][candidate],'Full candidate saved metrics differ')
        default_authority={'v11':authority['v11']['stages'][s]['metrics']['R49'],
            'v12':authority['v12']['stages'][s]['metrics']['R50'],
            'v5':r36['stages'][s]['metrics']['log_spatial_combination']}
        need(all(metrics[n]==default_authority[n] for n in default_authority),'Original default-control full metrics differ')
        need(all(metrics[n]==a['metrics'][n] for n in default_authority if n in a['metrics']),'Candidate-recorded default full metrics differ')
        for label,_,key in controls:
            need(metrics[label]==authority[label]['stages'][s]['metrics'][key],'Control authority full metrics differ')
            if label in a['metrics']:need(metrics[label]==a['metrics'][label],'Candidate-recorded control metrics differ')
        need(metrics['v5']==r50['stages'][s]['metrics']['v5'],'True v5 authority differs')
        groups=common.norm(Xq.address_city)+'|'+pd.Series(common.group_labels(X,Xq)); hist=groups.value_counts()
        supported=hist.reindex(weights.index,fill_value=0)>=200; coverage=float(weights[supported].sum()); sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in models}
        need(coverage==r50['stages'][s]['test_composition']['test_mass_covered'] and all(condition[n]==r50['stages'][s]['test_composition']['models'][key] for n,key in [('v11','R49'),('v12','R50'),('v5','v5')]),'Fixed test-mix control differs')
        extra_mix_checks={label:'not_available' for label,_,_ in extras}
        for label,(diag,key) in prior_diagnostics.items():
            saved=diag['stages'][s]['test_composition']
            need(saved['minimum_historical_cell_rows']==200 and saved['test_mass_covered']==coverage and saved['unsupported_test_rows']==int(counts[~supported].sum()) and saved['models'][key]==condition[label],'Existing extra-control test-mix differs')
            extra_mix_checks[label]='exact'
        bands=[]
        for name,lo,hi in BANDS:
            m=(y>=lo)&(y<hi); bands.append({'band':name,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-y[m]).mean()) if m.any() else None for n in models}})
        result['stages'][s]={'stage':s,'fit_rows':len(fi),'query_rows':len(qi),'metrics':metrics,'all_full_saved_metrics_exact':True,
            'differences':{n:old.prior.log_run.differences(metrics[n],metrics[candidate]) for n in models[:-1]},
            'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition},
            'extra_control_existing_test_mix_checks':extra_mix_checks,'price_bands':bands}
        result['checked_stages'].append(s)
        print(json.dumps({'stage':s,'test_mix':condition,'candidate_overall':metrics[candidate]['overall_mae'],'candidate_weighted':metrics[candidate]['weighted_mae']}),flush=True)
    result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick)
    write(ROOT/f'reports/round{number}_diagnostics.json',result); render(result,ROOT/f'reports/round{number}_diagnostics.md')


def execute(number,candidate,extras=()):
    validate_controls(number,candidate,extras)
    out=ROOT/f'outputs/round{number}/diagnostics'
    need(not out.exists() and not (ROOT/f'reports/round{number}_diagnostics.md').exists() and not (ROOT/f'reports/round{number}_diagnostics.json').exists(),'No overwrite/retry')
    out.mkdir(parents=True); tick=time.monotonic(); start=old.now(); code=1; failure=None
    with (out/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:diagnose(number,candidate,extras); code=0
        except Exception:failure=traceback.format_exc(); print(failure,flush=True)
    event={'round':number,'candidate':candidate,'extra_controls':list(extras),'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(out/'run.log'),'automatic_retry':False}
    write(out/'execution.json',event); print(json.dumps(event),flush=True); return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round',type=int,required=True); p.add_argument('--candidate',required=True)
    p.add_argument('--extra-control',type=extra_control,action='append',default=[],metavar='LABEL:ROUND:KEY')
    args=p.parse_args(); raise SystemExit(execute(args.round,args.candidate,args.extra_control))
