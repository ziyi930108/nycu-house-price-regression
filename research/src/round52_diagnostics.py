"""Short R52 saved-output diagnostic: fixed bands/mix, native bias and prior counts."""
from __future__ import annotations
import argparse,contextlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round49_combination as common
from round51_diagnostics import effective_area,scalar_stats
old=common.old;ROOT=common.ROOT;STAGES=common.STAGES;SCORER=common.SCORER
OUT=ROOT/'outputs/round52/diagnostics';REPORT=ROOT/'reports/round52_diagnostics.md';JSON=ROOT/'reports/round52_diagnostics.json'
MODELS=('v12','v5','R52');KEYS={'v12':'v12_control','v5':'v5_log_spatial','R52':'nn_price_prior_half_v10'}
need,sha,rel,write,locked=common.need,common.sha,common.rel,common.write,common.locked


def report(r):
    label={'development':'開發','confirmation':'確認','auxiliary':'輔助'}
    lines=['# R52：保存預測的固定診斷','',
      '只讀完整三期保存預測；沒有模型載入、fit/predict、新bootstrap、調prior規則或權重。v12沿R50、v5取真正R36。原生NN比較R52 ratio×價格因子與R48 seed42 area-linked模型，並非比較不同seed。',
      '', '| 期別 | 方法 | 整體MAE | 加權MAE |','|---|---|---:|---:|']
    for s in STAGES:
        for n,m in r['stages'][s]['metrics'].items():lines.append(f"| {label[s]} | {n} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','## 固定test組成','', '沿R49 city×seen/new權重及mincell200，僅对有歷史支持的test質量重正規化；不是完整test／Private預期誤差。','',
      '| 期別 | 涵蓋test質量 | 未支持筆數 | v12 | v5 | R52 |','|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        d=r['stages'][s]['test_composition'];lines.append('| '+' | '.join([label[s],f"{d['test_mass_covered']:.4%}",str(d['unsupported_test_rows']),*[f"{d['models'][n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 五個固定真價bands','', '| 期別 | 真價 | 筆數 | v12 | v5 | R52 |','|---|---|---:|---:|---:|---:|']
    for s in STAGES:
        for b in r['stages'][s]['price_bands']:lines.append('| '+' | '.join([label[s],b['band'],str(b['rows']),*['—' if b[n] is None else f"{b[n]:,.2f}" for n in MODELS]])+' |')
    lines+=['','## 原生NN共同偏差','', 'signed error=預測−真價；負值是平均低估。面積群固定query有效面積A≥該期fit A99（linear），不依query價或誤差選門檻。數字在融合/reference/spatial之前。','',
      '| 期別 | 群組 | 筆數 | 原生模型 | MAE | 平均signed error | 預測中位數 |','|---|---|---:|---|---:|---:|---:|']
    for s in STAGES:
        for g,d in r['stages'][s]['native_groups'].items():
            for n,m in d['models'].items():
                if m:lines.append(f"| {label[s]} | {g} | {d['rows']} | {n} | {m['mae']:,.2f} | {m['mean_signed_error']:+,.2f} | {m['median_prediction']:,.2f} |")
    lines+=['','## prior支援（引用driver證據）','', '不重建prior、不重新選層級；OOF是按建案分fold，不是時間OOF。價格因子使用fit標籤，並非純X。','',
      '| 期別 | prior層級 | full-fit有效cells | 最小cell支持 | OOF訓練被使用筆數 | query被使用筆數 |','|---|---|---:|---:|---:|']
    for s in STAGES:
        p=r['stages'][s]['prior_support'];levels={ '|'.join(x['columns']):x for x in p['full_state_level_support']}
        for k,q in p['query_prior_level_counts'].items():
            z=levels.get(k,{});lines.append('| '+' | '.join([label[s],k.replace('|','＋'),str(z.get('cell_count','—')),str(z.get('minimum_observed_support','—')),str(p['oof_training_level_counts'].get(k,0)),str(q)])+' |')
    signs=[r['stages'][s]['differences_vs_v12'][k]['relative_change_pct'] for s in STAGES for k in ('overall','weighted')]
    note='三期兩個整體指標均改善，可列為候選但Private未知。' if all(x<0 for x in signs) else '未同時改善三期兩個整體指標，不應僅凭prior表示或原生改善便直接替換v12。'
    lines+=['','## 簡短判讀','',note,'',
      '高價和面積群是描述性切片，不增加任何逐ID或價格規則。窗口已多輪使用，沒有選型後顯著性或Private保證；原生偏差與完整pipeline效果應分開看。完整新舊建案、季度、prior來源及counts均留JSON。',
      '',f"唯一CPU診斷 {r['elapsed_seconds']:.3f}秒；所有完整保存metrics及v12/v5固定test-mix精確重現。沒有新CSV。",
      '', '[完整JSON](round52_diagnostics.json) · [R52結果](../outputs/round52/results.json) · [診斷source](../src/round52_diagnostics.py)','']
    with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def run():
    tick=time.monotonic();inputs={}
    def small(n):
        p=ROOT/n;v=locked(p);inputs[n]=sha(p);inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'));return v
    p48=small('outputs/round48/plan.json');r48=small('outputs/round48/results.json');r50=small('outputs/round50/results.json');r52=small('outputs/round52/results.json')
    need(all(r['checked_stages']==list(STAGES) for r in (r48,r50,r52)),'Three completed windows required')
    mixpath='outputs/error_budget_20260926/round46_diagnostics.json';mix=json.loads((ROOT/mixpath).read_text(encoding='utf8'));inputs[mixpath]=sha(ROOT/mixpath);need(mix['minimum_historical_cell_rows']==200,'Mix rule changed')
    for n in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[n]=p48['input_sha256'][n]
    specs={}
    for s in STAGES:
        a=r52['stages'][s];b={}
        for role,name in [('fit',f'outputs/round52/{s}/fit_indices.npy'),('query',f'outputs/round52/{s}/query_indices.npy'),('R52',f'outputs/round52/{s}/predictions.npy'),('native_R52',a['native_predictions_path'])]:b[role]=name;inputs[name]=a['artifact_sha256'][name]
        name=f'outputs/round48/{s}/model/predictions_twd.npy';b['native_R48']=name;inputs[name]=r48['stages'][s]['artifact_sha256'][name]
        name=f'outputs/round50/{s}/predictions.npy';b['v12']=name;inputs[name]=r50['stages'][s]['prediction_sha256']
        name=r50['bindings'][s]['v5'];b['v5']=name;inputs[name]=r50['input_sha256'][name];specs[s]=b
    source=common.merge(old.sources(),{n:sha(ROOT/n) for n in ('src/round52_diagnostics.py','src/round51_diagnostics.py','src/round49_combination.py','src/round3_evaluate.py')})
    old.prior.c6.verify_hashes(common.merge(inputs,source));Xall,yall=old.load_training_rows(p48)
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float);counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64);need(counts.sum()==47073 and np.isclose(weights.sum(),1),'Invalid fixed mix')
    result={'round':52,'status':'incomplete','source_sha256':source,'input_sha256':inputs,'checked_stages':[],'stages':{},'model_fit_load_predict_calls':0,'new_bootstrap':False,'prior_rules_reselected':False,'new_CSV':False}
    for s in STAGES:
        a=r52['stages'][s];b=specs[s];fi,qi=[np.load(ROOT/b[k],allow_pickle=False) for k in ('fit','query')]
        for k,ind in (('fit',fi),('query',qi)):
            c=p48['split_contract'][s];need(ind.dtype==np.int64 and ind.shape==(c[k+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==c[k+'_indices_content_sha256'],'Canonical indices differ')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);y=yall[qi];v={n:old.vector(ROOT/b[n],len(qi)) for n in (*MODELS,'native_R48','native_R52')}
        metrics={n:SCORER.score(y,v[n],X,Xq,p48['splits'][s][0],p48['target_group_weights']) for n in MODELS};need(all(metrics[n]==a['metrics'][KEYS[n]] for n in MODELS),'Original full metrics differ')
        groups=common.norm(Xq.address_city)+'|'+pd.Series(common.group_labels(X,Xq));hist=groups.value_counts();supported=hist.reindex(weights.index,fill_value=0)>=200;coverage=float(weights[supported].sum());sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-y)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in MODELS}
        need(coverage==r50['stages'][s]['test_composition']['test_mass_covered'] and condition['v12']==r50['stages'][s]['test_composition']['models']['R50'] and condition['v5']==r50['stages'][s]['test_composition']['models']['v5'],'Mix controls differ')
        bands=[]
        for label,lo,hi in [('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf)]:
            m=(y>=lo)&(y<hi);bands.append({'band':label,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-y[m]).mean()) if m.any() else None for n in MODELS}})
        threshold=float(np.quantile(effective_area(X),.99,method='linear'));masks={'all':np.ones(len(y),dtype=bool),'>=100m':y>=1e8,'fit_A99':effective_area(Xq)>=threshold}
        native={k:{'rows':int(m.sum()),'models':{n:scalar_stats(y[m],v[n][m]) for n in ('native_R48','native_R52')}} for k,m in masks.items()}
        prep=a['prior_preparation'];ev=prep['prior_evidence'];oof={}
        for f in ev['folds']:
            for k,n in f['held_prior_level_counts'].items():oof[k]=oof.get(k,0)+n
        query=prep['query_prior_level_counts'];need(sum(oof.values())==len(fi) and sum(query.values())==len(qi),'Prior evidence support counts differ')
        support={'source':'saved stage.prior_preparation; no prior refit/predict','full_state_level_support':ev['full_state_level_support'],'oof_training_level_counts':oof,'query_prior_level_counts':query,'groups':ev['groups'],'rules':ev['rules'],'uses_fit_labels':ev['uses_fit_labels'],'uses_query_labels':ev['uses_query_labels'],'chronological_oof':ev['chronological_oof']}
        item={'stage':s,'metrics':metrics,'all_full_metrics_exact':True,'differences_vs_v12':old.prior.log_run.differences(metrics['v12'],metrics['R52']),'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition},'price_bands':bands,'fit_A99':threshold,'native_groups':native,'prior_support':support}
        result['stages'][s]=item;result['checked_stages'].append(s);print(json.dumps({'stage':s,'test_mix':condition,'native_all':native['all'],'native_high':native['>=100m']}),flush=True)
    result.update(status='completed_saved_predictions_only',elapsed_seconds=time.monotonic()-tick);write(JSON,result);report(result)


def execute():
    need(not OUT.exists() and not REPORT.exists() and not JSON.exists(),'No overwrite/retry');OUT.mkdir(parents=True);start=old.now();tick=time.monotonic();code=1;failure=None
    with (OUT/'run.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:run();code=0
        except Exception:failure=traceback.format_exc();print(failure,flush=True)
    event={'round':52,'command':sys.argv,'started_at_utc':start,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',event);print(json.dumps(event),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('--run required')
    raise SystemExit(execute())
