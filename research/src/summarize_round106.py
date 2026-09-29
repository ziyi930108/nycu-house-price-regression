"""R106 aggregate-only report. No model, row array, price-table or bootstrap loading."""
from pathlib import Path
import argparse,json,time,traceback
import summarize_round100 as common
ROOT=common.ROOT;STAGES=common.STAGES;METRICS=common.MEASURES
OUT=ROOT/'outputs/round106_summary';REPORT=ROOT/'reports/round106_report.md'
need,sha,write=common.need,common.sha,common.write
LAYERS=('native','base','reference','spatial','median')

def collect():
    inputs={}
    def read(name):
        p=ROOT/name
        need(name.startswith('outputs/round106/') and p.suffix=='.json','Only R106 aggregate metadata allowed')
        inputs[name]=sha(p)
        side=p.with_suffix('.sha256')
        if side.exists():need(side.read_text(encoding='ascii').strip()==inputs[name],'SHA sidecar differs')
        return json.loads(p.read_text(encoding='utf8'))
    def bound(rec):
        v=read(rec['path']);need(inputs[rec['path']]==rec['sha256'],'Bound metadata changed');return v
    seal=read('outputs/round106/sealed_manifest.json');result=read('outputs/round106/evaluation/results.json')
    main=read('outputs/round106/execution.json');evaluation=read('outputs/round106/evaluation/execution.json')
    for v in (main,evaluation):need(v['exit_code']==0 and not v.get('failure'),'Unfinished execution')
    need(seal['round']==result['round']==106 and result['status']=='completed_three_window_evaluation','Wrong results')
    need(result['sealed_manifest']=={'path':'outputs/round106/sealed_manifest.json','sha256':inputs['outputs/round106/sealed_manifest.json']},'Result seal differs')
    need(seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0 and seal['all_query_reload_exact'],'Fit/reload contract differs')
    need(seal['protected_before']==seal['protected_after'],'Protected artifacts changed')
    need(result['conditional_intervals_reported']==24 and result['Private_score'] is None and not result['new_CSV'],'Scope differs')
    stages={};intervals=[];fit_total=0.
    for stage in STAGES:
        r=result['stages'][stage];s=bound(seal['stages'][stage]['stage_manifest']);ex=bound(seal['stages'][stage]['execution'])
        need(ex['exit_code']==0 and not ex.get('failure') and s['new_NN_fits']==2 and s['all_query_reload_exact'],'Incomplete stage')
        arms={}
        for arm in ('C','T'):
            a=s['arms'][arm];tr=bound(a['training']);te=bound(a['execution'])
            need(tr['status']=='completed' and tr['saved_model_reload_exact'] and a['saved_model_reload_exact'] and len(tr['epochs'])==30 and te['exit_code']==0,'Incomplete trainer')
            need(tr['fit_seconds']==a['fit_seconds'],'Fit time differs');fit_total+=a['fit_seconds']
            arms[arm]={'fit_seconds':a['fit_seconds'],'training':a['training'],'trainer_process_seconds':te['elapsed_seconds']}
        need(all(r['layer_metrics'][a]['median']==r['metrics'][a] for a in ('v22','C','T')),'Median identity differs')
        comparisons={k:r[k] for k in ('comparison','comparison_vs_C','C_vs_v22')}
        bands={};groups={}
        for label,key in [('T-v22','comparison'),('T-C','comparison_vs_C'),('C-v22','C_vs_v22')]:
            d=comparisons[key];b=[dict(v,overall_delta_contribution_TWD=v['rows']*v['delta']/r['query_rows'] if v['rows'] else 0.) for v in d['price_bands']]
            need(sum(v['rows'] for v in b)==r['query_rows'] and abs(sum(v['overall_delta_contribution_TWD'] for v in b)-d['overall_mae']['delta'])<1e-6,'Price-band identity differs')
            bands[label]=b;groups[label]={'strata':d['strata'],'adverse':d['quarter_or_large_group_deterioration_above_5pct']}
        for control in ('v22','C'):
            for family in ('project','month'):
                for measure in ('overall','weighted'):
                    v=r['paired'][control]['methods'][family]['summary'][measure];lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    need(v['valid_replicates']+v['invalid_replicates']==2000,'Replicate count differs')
                    direction='unavailable' if lo is None or hi is None else 'negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero'
                    intervals.append({'stage':stage,'control':control,'family':family,'metric':measure,'direction':direction,**v})
        stages[stage]={'metrics':r['metrics'],'layer_metrics':r['layer_metrics'],'groups':groups,'price_bands':bands,'fit':arms,'stage_execution_seconds':ex['elapsed_seconds']}
    need(len(intervals)==24 and abs(fit_total-seal['fit_seconds'])<1e-6,'Aggregate count/time differs')
    return {'round':106,'status':'aggregate_summary_completed','stages':stages,'research_gates':result['research_gates'],
        'intervals':intervals,'interval_counts':{k:sum(v['direction']==k for v in intervals) for k in ('negative','crosses_zero','positive','unavailable')},
        'timing':{'six_NN_fit_seconds':fit_total,'main_execution_seconds':main['elapsed_seconds'],'evaluation_execution_seconds':evaluation['elapsed_seconds']},
        'input_sha256':inputs,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0,
        'selected_submission':'v22','Private_score':None,'new_CSV':False}

def render(r):
    f=lambda v:f'{v:,.2f}'
    pair=lambda d:' / '.join(f(d[k]) for k in METRICS)
    delta=lambda a,b:' / '.join(f(a[k]-b[k]) for k in METRICS)
    lines=['# R106：四個身分類別的聯動 dropout','',
        'C為原逐欄獨立0.1遮罩的重訓；T只令案名、完整門牌、棟及戶、棟別四欄共享案名原mask。沿原R63 144數值／23類別、價格因子及日期weighted MAE，三窗共六NN fit。唯一提交仍v22，Private未知，無新CSV。','',
        '所有值為新臺幣元；每格為整體／固定加權MAE，差值負數較好。C−v22展示原設定重訓偏移，不能隱去。','',
        '|窗口|v22|C|T|T−v22|T−C|C−v22|','|---|---:|---:|---:|---:|---:|---:|']
    for stage,s in r['stages'].items():
        m=s['metrics'];lines.append('|'+stage+'|'+'|'.join(pair(m[a]) for a in ('v22','C','T'))+'|'+delta(m['T'],m['v22'])+'|'+delta(m['T'],m['C'])+'|'+delta(m['C'],m['v22'])+'|')
    lines+=['','## 固定門檻','','```json',json.dumps(r['research_gates'],ensure_ascii=False,indent=2),'```','',
        'Primary為相對v22的跨窗採用研究條件，mechanism為相對同設定重訓C的條件；單seed不能證明跨seed因果。native不是門檻，仍完整列出。兩項全過亦不自動取代使用者選定。','',
        '## 五層與分群','','|窗口|層|v22|C|T|T−v22|T−C|','|---|---|---:|---:|---:|---:|---:|']
    for stage,s in r['stages'].items():
        for layer in LAYERS:
            m={a:s['layer_metrics'][a][layer] for a in ('v22','C','T')};lines.append('|'+stage+'|'+layer+'|'+'|'.join(pair(m[a]) for a in ('v22','C','T'))+'|'+delta(m['T'],m['v22'])+'|'+delta(m['T'],m['C'])+'|')
    lines+=['','|窗口|比較|seen差|new差|>5%違例數|最大相對退步|','|---|---|---:|---:|---:|---|']
    for stage,s in r['stages'].items():
        for label,g in s['groups'].items():
            bad=g['adverse'];w=max(bad,key=lambda v:v['relative_change_pct']) if bad else None
            desc=f"{w['section']}/{w.get('group',w.get('band'))}，{w['rows']}列，{f(w['delta'])}元／{w['relative_change_pct']:.3f}%" if w else '無'
            lines.append('|'+stage+'|'+label+'|'+f(g['strata']['seen']['delta'])+'|'+f(g['strata']['new_nonmissing']['delta'])+'|'+str(len(bad))+'|'+desc+'|')
    lines+=['','所有城市、月份、季度、固定群與空群保留在原評分JSON；互相重疊的群不能加總。五價帶的全窗貢獻另存摘要JSON，核對加總至1e−6元，只是事後診斷，不按未知真價調價。','',
        '## 全部24個條件配對區間','','|窗口|比較|方法／指標|點差|2.5%～97.5%|有效／無效|','|---|---|---|---:|---|---:|']
    for v in r['intervals']:
        bounds='缺失' if v['percentile_2_5'] is None else f(v['percentile_2_5'])+'～'+f(v['percentile_97_5'])
        lines.append(f"|{v['stage']}|T−{v['control']}|{v['family']}/{v['metric']}|{f(v['point_delta'])}|{bounds}|{v['valid_replicates']}/{v['invalid_replicates']}|")
    lines+=['',json.dumps(r['interval_counts'],ensure_ascii=False)+'；每類原固定2000次，本摘要不重抽。CI不是Private勝率。','',
        '## 時間與限制','','|步驟|秒|','|---|---:|']
    for stage,s in r['stages'].items():
        for arm,a in s['fit'].items():lines.append(f"|{stage} {arm} fit|{a['fit_seconds']:.3f}|")
    for k,v in r['timing'].items():lines.append(f'|{k}|{v:.3f}|')
    lines+=['','fit已包含於main，不能重複相加，以上非聊天或開發工時。原unknown embedding已有訓練，聯動不遮numeric prior或其他地址／地理，並不等同真新案；各欄邊際人工遮罩率不變，也不保證CUDA重訓逐位確定。',
        '反覆年度選型、單seed、既有R42/R18遮罩反證與2023非價格地理的歷史可用性限制保留，不是新盲測或完整historical-as-of。無mask率／欄位／seed／alpha搜尋，未全量fit、未新CSV。保存模型重載不是乾淨環境從零重現。',
        '本摘要只讀JSON metadata與聚合結果，不載模型、逐筆array或真價、不重算MAE或bootstrap。',
        '使用者要求本輪完成後暫停；R106完成與文件交接後停止研究，不啟動R107或背景訓練。',
        '[計畫](round106_plan.md) · [完整評分](../outputs/round106/evaluation/results.json) · [摘要JSON](../outputs/round106_summary/summary.json) · [摘要程式](../src/summarize_round106.py)','']
    return '\n'.join(lines)

def run():
    need(not OUT.exists() and not REPORT.exists(),'Preserve existing summary');OUT.mkdir();t=time.monotonic();failure=None
    sources={str(Path(p).relative_to(ROOT)).replace('\\','/'):sha(p) for p in (__file__,common.__file__)}
    try:
        r=collect();r['source_sha256']=sources;write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(all(sha(ROOT/p)==h for p,h in {**sources,**r['input_sha256']}.items()),'Source/input changed')
        artifacts={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (REPORT,OUT/'summary.json')}
        write(OUT/'receipt.json',{'status':'completed','artifact_sha256':artifacts,'source_sha256':sources,'input_sha256':r['input_sha256'],'elapsed_seconds':time.monotonic()-t,'new_fit_predict_score_bootstrap_calls':0})
        print(json.dumps({'status':'completed','artifacts':artifacts,'elapsed_seconds':time.monotonic()-t}))
    except Exception:failure=traceback.format_exc();raise
    finally:write(OUT/'execution.json',{'exit_code':int(failure is not None),'failure':failure,'elapsed_seconds':time.monotonic()-t,'automatic_retry':False,'new_fit_predict_score_bootstrap_calls':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:ap.error('--run required after all training and evaluation exit0')
    run()
