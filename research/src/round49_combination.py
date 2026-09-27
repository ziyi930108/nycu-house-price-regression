"""R49 one fixed saved-branch median comparison; no model or reference calls."""
from __future__ import annotations
import argparse,contextlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round39_city_road_experiment as old
from round3_evaluate import group_labels,norm
ROOT=old.ROOT;OUT=ROOT/'outputs/round49';REPORT=ROOT/'reports/round49_report.md'
STAGES=('development','confirmation','auxiliary')
FORMULA='median(saved_R48_nn_area_spatial_TWD,saved_R46_recency_log_spatial_TWD,saved_R40_unit_spatial_TWD)'
CANDIDATE='R49';need,sha,rel,write,locked,merge=old.need,old.sha,old.rel,old.write,old.locked,old.merge
SCORER=old.prior.log_run.original


def report(result):
    labels={'development':'開發期','confirmation':'確認期','auxiliary':'輔助期'}
    lines=['# R49：固定 NN area raw＋recency log＋unit 中位數','',
    '將已完成 R48 的完整 NN area 空間分支與 R46 recency-log 空間分支、R40 unit 空間分支逐列取 median。這是兩個既有改動的固定組合；無新fit、模型predict、reference或spatial呼叫，沒有比例搜尋、clip或逐ID規則，三期完整執行。',
    '', '金額為元，負百分比表示MAE降低。全部為同窗歷史評分，不是Public分數。','',
    '| 期別 | 方法 | 整體MAE | 固定加權MAE |','|---|---|---:|---:|']
    for stage in STAGES:
        for name,m in result['stages'][stage]['metrics'].items():
            lines.append(f"| {labels[stage]} | {name} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','| 期別 | 對v10整體/加權 | 對R48整體/加權 | 對v5整體/加權 |','|---|---:|---:|---:|']
    for stage in STAGES:
        ds=result['stages'][stage]['differences'];pairs=[f"{ds[n]['overall']['relative_change_pct']:+.4f}% / {ds[n]['weighted']['relative_change_pct']:+.4f}%" for n in ('v10','R48','v5')]
        lines.append('| '+' | '.join([labels[stage],*pairs])+' |')
    lines+=['','## 城市×新舊建案的固定test組成診斷','',
    '沿既有 R46 診斷的 test-X city×seen/new/missing 權重；每窗只納歷史query至少200筆的cell，按涵蓋test質量重新正規化。沒有支持的test質量明列，不能當成完整test／Private預期MAE。此診斷使用真價僅計各cell誤差，不調整組合。','',
    '| 期別 | 涵蓋test質量 | 不支持test筆數 | v10條件MAE | R48條件MAE | R49條件MAE |','|---|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        d=result['stages'][stage]['test_composition'];ms=d['models']
        lines.append(f"| {labels[stage]} | {d['test_mass_covered']:.4%} | {d['unsupported_test_rows']:,} | {ms['v10']:,.2f} | {ms['R48']:,.2f} | {ms['R49']:,.2f} |")
    lines+=['','## 固定真價bands','', '| 期別 | 真價區間 | 筆數 | v10 MAE | R48 MAE | R49 MAE |','|---|---|---:|---:|---:|---:|']
    for stage in STAGES:
        for d in result['stages'][stage]['price_bands']:
            vals=['—' if d[n] is None else f"{d[n]:,.2f}" for n in ('v10','R48','R49')]
            lines.append('| '+' | '.join([labels[stage],d['band'],f"{d['rows']:,}",*vals])+' |')
    lines+=['','## 新舊建案與季度','', '| 期別 | 群組/季 | 筆數 | v10 MAE | R48 MAE | R49 MAE |','|---|---|---:|---:|---:|---:|']
    for stage in STAGES:
        mm=result['stages'][stage]['metrics']
        for family,names in (('strata',('seen','new_nonmissing','missing')),('forecast_quarters',('Q1','Q2','Q3','Q4'))):
            for name in names:
                vals=['—' if mm[m][family][name]['mae'] is None else f"{mm[m][family][name]['mae']:,.2f}" for m in ('v10','R48','R49')]
                lines.append('| '+' | '.join([labels[stage],name,str(mm['R49'][family][name]['rows']),*vals])+' |')
    signs={s:result['stages'][s]['differences']['v10'] for s in STAGES}
    lines+=['','## 解讀界線','',
    '各期及各子群的得失如上，不將局部改善說成所有分組勝出，也未設新硬門檻或自動採用。原年度標籤已跨輪重用；按真價分组僅為事後描述，推論時不知道真價，不能作逐筆規則。沒有測試標籤、Private機率推定或新CSV。',
    '', f"唯一run全部三期完成；模型fit/predict為0，耗時 {result['elapsed_seconds']:.3f}秒。v10與R48分支組合逐位重現，v10/R48/真正R36 v5全部原scorer指標精確重現。原CSV及registry SHA不變。",
    '', '[完整結果](../outputs/round49/results.json) · [輸入身分與固定方法](../outputs/round49/input_contract.json) · [程式](../src/round49_combination.py)','']
    with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))


def run():
    tick=time.monotonic();inputs={}
    def small(name):
        p=ROOT/name;v=locked(p);inputs[name]=sha(p);inputs[rel(p.with_suffix('.sha256'))]=sha(p.with_suffix('.sha256'));return v
    p48=small('outputs/round48/plan.json');r48=small('outputs/round48/results.json');r46=small('outputs/round46/results.json')
    mixpath=ROOT/'outputs/error_budget_20260926/round46_diagnostics.json'
    mix=json.loads(mixpath.read_text(encoding='utf8'));inputs[rel(mixpath)]=sha(mixpath)
    need(r48['checked_stages']==r46['checked_stages']==list(STAGES) and mix['minimum_historical_cell_rows']==200,'Incomplete prior stages/mix contract')
    for n in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',old.KT):inputs[n]=p48['input_sha256'][n]
    specs={}
    for stage in STAGES:
        a=r48['stages'][stage];b=r46['stages'][stage];bind=p48['bindings'][stage]
        need(a['fit_rows']==b['fit_rows'] and a['query_rows']==b['query_rows'] and a['dates']==b['dates'],'Stage row/date mismatch')
        spec={}
        for role,name,known in [
            ('fit_indices',f'outputs/round48/{stage}/fit_indices.npy',a['artifact_sha256']),
            ('query_indices',f'outputs/round48/{stage}/query_indices.npy',a['artifact_sha256']),
            ('r46_qi',f'outputs/round46/{stage}/model/query_indices.npy',b['artifact_sha256']),
            ('newraw',f'outputs/round48/{stage}/nn_area_spatial.npy',a['artifact_sha256']),
            ('newlog',f'outputs/round46/{stage}/recency_log_spatial.npy',b['artifact_sha256']),
            ('R48',f'outputs/round48/{stage}/predictions.npy',a['artifact_sha256']),
            ('v10',f'outputs/round46/{stage}/recency_log_three_branch_median.npy',b['artifact_sha256']),
            *[(n,bind[n],p48['input_sha256']) for n in ('unit_spatial','raw_spatial','old_log_spatial','v5')]]:
            spec[role]=name;inputs.update(merge(inputs,{name:known[name]}))
        specs[stage]=spec
    src=merge(old.sources(),{'src/round49_combination.py':sha(Path(__file__)),'src/round3_evaluate.py':sha(ROOT/'src/round3_evaluate.py')})
    protected={rel(f):sha(f) for f in sorted((ROOT/'outputs').glob('submission_v*.csv'))};protected['runs/versions.json']=sha(ROOT/'runs/versions.json')
    contract={'round':49,'formula':FORMULA,'stage_order':list(STAGES),'source_sha256':src,'input_sha256':inputs,'bindings':specs,
              'minimum_historical_cell_rows':200,'fixed_true_price_bounds':[0,10000000,20000000,50000000,100000000,None],
              'target_composition_reused_from':rel(mixpath),'new_fit_or_model_predict_calls':0,'new_gate':None,'no_early_stop_by_scores':True}
    write(OUT/'input_contract.json',contract)
    old.prior.c6.verify_hashes(merge(inputs,src))
    Xall,yall=old.load_training_rows(p48)
    weights=pd.Series({k:v['weight'] for k,v in mix['target_cells'].items()},dtype=float)
    counts=pd.Series({k:v['rows'] for k,v in mix['target_cells'].items()},dtype=np.int64)
    need(int(counts.sum())==47073 and np.isclose(weights.sum(),1),'Target composition changed')
    result={'round':49,'status':'incomplete','formula':FORMULA,'checked_stages':[],'stages':{},'input_contract_sha256':sha(OUT/'input_contract.json'),
            'protected_before':protected,'source_sha256':src,'fit_calls':0,'model_predict_calls':0,'reference_or_spatial_calls':0,'formal_adoption_decision':None,'new_submission_generated':False}
    for stage in STAGES:
        b=specs[stage];a=r48['stages'][stage];r=r46['stages'][stage];folder=OUT/stage;folder.mkdir()
        fi,qi=[np.load(ROOT/b[k+'_indices'],allow_pickle=False) for k in ('fit','query')]
        need(np.array_equal(qi,np.load(ROOT/b['r46_qi'],allow_pickle=False)),'R46 query order differs')
        for key,ind in (('fit',fi),('query',qi)):
            c=p48['split_contract'][stage];need(ind.dtype==np.int64 and ind.shape==(c[key+'_rows'],) and (np.diff(ind)>0).all()
                and ind.min()>=0 and ind.max()<len(Xall) and SCORER.value_sha(ind,'<i8')==c[key+'_indices_content_sha256'],'Indices mismatch')
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True)
        need(old.spatial.frame_sha(X)==a['original_frame_sha256']['fit_X'] and old.spatial.frame_sha(Xq)==a['original_frame_sha256']['query_X'],'X order mismatch')
        v={k:old.vector(ROOT/b[k],len(qi)) for k in ('newraw','newlog','unit_spatial','raw_spatial','old_log_spatial','R48','v10','v5')}
        old.exact(np.median(np.column_stack([v['newraw'],v['old_log_spatial'],v['unit_spatial']]),axis=1),v['R48'],'R48 saved median')
        old.exact(np.median(np.column_stack([v['raw_spatial'],v['newlog'],v['unit_spatial']]),axis=1),v['v10'],'v10 saved median')
        yy=yall[qi];score=lambda z:SCORER.score(yy,z,X,Xq,p48['splits'][stage][0],p48['target_group_weights'])
        metrics={n:score(v[n]) for n in ('v10','R48','v5')}
        need(metrics['v10']==r['metrics']['recency_log_three_branch_median'] and metrics['R48']==a['metrics']['nn_area_three_branch_median']
             and metrics['v5']==a['metrics']['v5_log_spatial'],'Control full metrics differ')
        candidate=np.median(np.column_stack([v['newraw'],v['newlog'],v['unit_spatial']]),axis=1)
        need(candidate.shape==yy.shape and np.isfinite(candidate).all() and (candidate>=0).all(),'Candidate not TWD')
        v['R49']=candidate;metrics['R49']=score(candidate);old.save_array(folder/'predictions.npy',candidate)
        groups=norm(Xq.address_city)+'|'+pd.Series(group_labels(X,Xq));hist=groups.value_counts();supported=hist.reindex(weights.index,fill_value=0)>=200
        coverage=float(weights[supported].sum());sw=weights[supported]/coverage
        condition={n:float((pd.Series(np.abs(v[n]-yy)).groupby(groups).mean().reindex(sw.index)*sw).sum()) for n in ('v10','R48','R49')}
        need(coverage==mix['stages'][stage]['test_mass_covered'] and condition['v10']==mix['stages'][stage]['models']['R46']['target_mix_supported_mae'],'Fixed composition control differs')
        bands=[]
        for label,lo,hi in [('<10m',0,1e7),('10-20m',1e7,2e7),('20-50m',2e7,5e7),('50-100m',5e7,1e8),('>=100m',1e8,np.inf)]:
            m=(yy>=lo)&(yy<hi);bands.append({'band':label,'rows':int(m.sum()),**{n:float(np.abs(v[n][m]-yy[m]).mean()) if m.any() else None for n in ('v10','R48','R49')}})
        cells=[{'cell':key,'test_rows':int(counts[key]),'test_weight':float(weights[key]),'historical_rows':int(hist.get(key,0)),'supported':bool(supported[key])} for key in weights.index]
        item={'round':49,'stage':stage,'status':'completed','fit_rows':len(fi),'query_rows':len(qi),'dates':p48['splits'][stage],'metrics':metrics,
              'differences':{n:old.prior.log_run.differences(metrics[n],metrics['R49']) for n in ('v10','R48','v5')},
              'controls_vector_and_full_metrics_exact':True,'prediction_sha256':sha(folder/'predictions.npy'),
              'test_composition':{'minimum_historical_cell_rows':200,'test_mass_covered':coverage,'unsupported_test_rows':int(counts[~supported].sum()),'models':condition,'cells':cells},'price_bands':bands}
        write(folder/'results.json',item);result['stages'][stage]=item;result['checked_stages'].append(stage)
        print(json.dumps({'stage':stage,'overall':{n:m['overall_mae'] for n,m in metrics.items()},'weighted':{n:m['weighted_mae'] for n,m in metrics.items()},'composition':condition}),flush=True)
    result.update(status='three_periods_completed_no_adoption_decision',elapsed_seconds=time.monotonic()-tick,protected_after={n:sha(ROOT/n) for n in protected})
    need(result['protected_before']==result['protected_after'],'Protected files changed')
    write(OUT/'results.json',result);report(result)


def execute():
    need(not OUT.exists() and not REPORT.exists(),'R49 output exists; do not overwrite');OUT.mkdir(parents=True)
    started=old.now();tick=time.monotonic();code=1;failure=None
    with (OUT/'run.log').open('x',encoding='utf8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:run();code=0
            except Exception:failure=traceback.format_exc();print(failure,flush=True)
    receipt={'round':49,'command':sys.argv,'started_at_utc':started,'completed_at_utc':old.now(),'elapsed_seconds':time.monotonic()-tick,'exit_code':code,'failure':failure,'source_sha256':sha(Path(__file__)),'log_sha256':sha(OUT/'run.log'),'automatic_retry':False}
    write(OUT/'execution.json',receipt);print(json.dumps(receipt),flush=True);return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('--run is required')
    raise SystemExit(execute())
