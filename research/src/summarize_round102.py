"""R102 aggregate-only summary. Run only after all six fits and evaluation exit0."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import summarize_round100 as common

ROOT=common.ROOT; OUT=ROOT/'outputs/round102_summary'; REPORT=ROOT/'reports/round102_report.md'
STAGES=common.STAGES; PAIRS=common.PAIRS; KEYS=common.MEASURES
BASE=tuple('outputs/round102/'+p for p in ('evaluation/results.json','sealed_manifest.json','execution.json','evaluation/execution.json'))
ALLOWED=set(BASE)|{f'outputs/round102/{s}/{p}' for s in STAGES for p in ('stage_seal.json','execution.json','C/execution.json','T/execution.json')}
need,sha,write=common.need,common.sha,common.write


def collect(inputs):
    def read(name,expected=None):
        need(name in ALLOWED,'Only saved aggregate/timing JSON allowed'); path=ROOT/name; digest=sha(path)
        need(expected is None or digest==expected,'Bound JSON changed: '+name)
        side=path.with_suffix('.sha256')
        if side.exists(): need(side.read_text(encoding='ascii').strip()==digest,'Sidecar changed'); inputs[side.relative_to(ROOT).as_posix()]=sha(side)
        inputs[name]=digest; return json.loads(path.read_text(encoding='utf-8-sig'))
    result,seal,main,evaluation=(read(p) for p in BASE)
    need(main['exit_code']==evaluation['exit_code']==0 and main['failure'] is None and evaluation['failure'] is None,'Incomplete execution')
    need(result['round']==102 and result['status']=='completed_three_window_six_fit_evaluation' and result['checked_stages']==list(STAGES),'Incomplete results')
    need(result['sealed_manifest']=={'path':BASE[1],'sha256':inputs[BASE[1]]} and seal['status']=='all_prior_input_predictions_sealed','Seal differs')
    need(seal['new_NN_fits']==6 and seal['new_native_tree_fits']==seal['prior_build_calls']==0 and seal['all_query_reload_exact'],'Fit/reload scope differs')
    need(result['conditional_intervals_reported']==24 and not result['new_CSV'] and result['selected_submission']=='v22' and result['Private_score'] is None,'Selection/scope differs')
    stages={}; intervals=[]
    for s in STAGES:
        r=result['stages'][s]; binding=seal['stages'][s]
        ss=read(binding['stage_manifest']['path'],binding['stage_manifest']['sha256']); ex=read(binding['execution']['path'],binding['execution']['sha256'])
        need(ex['exit_code']==0 and ex['failure'] is None and ss['new_NN_fits']==2 and ss['numeric_count']==145,'Incomplete stage')
        need(ss['all_query_reload_exact'] and r['full_original_v22_and_v24_metrics_replay_exact'] and r['R102_C_is_not_R100_C'],'Control identity differs')
        comparisons={name:r[key] for name,key in PAIRS.items()}; decompositions={}
        for name,d in comparisons.items():
            bands=[{**b,'overall_delta_contribution_TWD':b['rows']*b['delta']/r['query_rows'] if b['rows'] else 0.} for b in d['price_bands']]
            need(sum(b['rows'] for b in bands)==r['query_rows'] and abs(sum(b['overall_delta_contribution_TWD'] for b in bands)-d['overall_mae']['delta'])<1e-6,'Aggregate band identity differs')
            decompositions[name]=bands
        for control in ('v22','C'):
            for family in ('project','month'):
                for measure in ('overall','weighted'):
                    v=r['paired'][control]['methods'][family]['summary'][measure]; lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    need(v['valid_replicates']+v['invalid_replicates']==2000,'Saved interval count differs')
                    direction='unavailable' if lo is None or hi is None else 'negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero'
                    intervals.append({'stage':s,'control':control,'family':family,'metric':measure,'direction':direction,**v})
        layers={arm:{**r['layer_metrics'][arm],'median':r['metrics'][arm]} for arm in ('v22','C','T')}
        deltas={layer:{pair:{k:layers[a][layer][k]-layers[b][layer][k] for k in KEYS} for pair,a,b in [('T-v22','T','v22'),('T-C','T','C'),('C-v22','C','v22')]} for layer in ('native','base','reference','spatial','median')}
        times={}
        for arm in ('C','T'):
            a=ss['arms'][arm]; ae=read(a['execution']['path'],a['execution']['sha256'])
            need(ae['exit_code']==0 and ae['failure'] is None and a['saved_model_reload_exact'],'Incomplete NN')
            times[arm]={'fit_seconds':a['fit_seconds'],'trainer_process_seconds':ae['elapsed_seconds']}
        need(abs(sum(v['fit_seconds'] for v in times.values())-ss['fit_seconds'])<1e-8,'Fit time differs')
        stages[s]={'saved_stage_results':r,'comparisons':comparisons,'layer_deltas':deltas,'price_band_decomposition':decompositions,'timing':{'arms':times,'stage_execution_seconds':ex['elapsed_seconds']}}
    need(len(intervals)==24,'All intervals required'); fit=sum(a['fit_seconds'] for s in stages.values() for a in s['timing']['arms'].values())
    need(abs(fit-seal['fit_seconds'])<1e-8 and fit<main['elapsed_seconds'],'Timing hierarchy differs')
    return {'round':102,'status':'aggregate_summary_completed','stages':stages,'saved_intervals':intervals,'research_gates':result['research_gates'],
        'interval_directions':{k:sum(v['direction']==k for v in intervals) for k in ('negative','crosses_zero','positive','unavailable')},
        'timing':{'NN_fit_total_seconds':fit,'main_execution_seconds':main['elapsed_seconds'],'evaluation_execution_seconds':evaluation['elapsed_seconds'],'fit_included_in_main':True,'new_NN_fits':6,'new_native_tree_fits':0,'prior_build_calls':0,'chat_time_measured':False},
        'original_results':{'path':BASE[0],'sha256':inputs[BASE[0]]},'input_sha256':inputs,'selected_submission':'v22','Private_score':None,'new_CSV':False,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0}


def render(r):
    f=lambda x:'—' if x is None else f'{x:,.2f}'
    pair=lambda x:' / '.join(f(x[k]) for k in KEYS)
    g=r['research_gates']; lines=['# R102：明示時間先驗與同寬零欄對照','',f"Primary門檻：**{g['primary']['adoption_research_gate_passed']}**；mechanism門檻：**{g['mechanism']['mechanism_gate_passed']}**。v22唯一提交不變，Private未知，無新CSV。",'',
        'T為145欄log-prior候選，C為145欄正零控制；兩者factor、loss列與另七陣列皆同R100T。**本輪C不是R100原OOF先驗C**，零欄不另選為候選。僅讀保存聚合JSON，沒有新評分。差值為前者減後者，負值較好；金額為元。','',
        '|窗口|v22整體／加權|C整體／加權|T整體／加權|','|---|---:|---:|---:|']
    for s,a in r['stages'].items():
        m=a['saved_stage_results']['metrics']; lines.append('|'+s+'|'+'|'.join(pair(m[k]) for k in ('v22','C','T'))+'|')
    lines+=['','|窗口|T−v22整體／加權|T−C|C−v22|T−v24|','|---|---:|---:|---:|---:|']
    for s,a in r['stages'].items(): lines.append('|'+s+'|'+'|'.join(pair({k:a['comparisons'][p][k]['delta'] for k in KEYS}) for p in PAIRS)+'|')
    lines+=['','## 兩個門檻分開判斷','','|門檻|條件|通過|','|---|---|---|']
    for name in ('primary','mechanism'):
        for key,value in g[name]['components'].items(): lines.append(f'|{name}|{key}|{value}|')
    lines+=['','若native未改善，不用後處理或median收益宣稱ratio表示機制成功。即使探索門檻通過也不是採用、Private保證或乾淨環境重現。','',
        '## 由原生到完整模型','','每格為整體／加權差；native是prior NN，spatial是N分支，median才是完整模型。','','|窗口|層級|T−v22原分量|T−C|C−v22原分量|','|---|---|---:|---:|---:|']
    for s,a in r['stages'].items():
        for layer,d in a['layer_deltas'].items(): lines.append('|'+s+'|'+layer+'|'+'|'.join(pair(d[k]) for k in ('T-v22','T-C','C-v22'))+'|')
    lines+=['','## 分群正反證','','|窗口|比較|seen差|new差|>5%群數|最大相對退步群|','|---|---|---:|---:|---:|---|']; notes=[]
    for s,a in r['stages'].items():
        for p,d in a['comparisons'].items():
            bad=d['quarter_or_large_group_deterioration_above_5pct']; w=max(bad,key=lambda b:b['relative_change_pct']) if bad else None
            label=f"{w['section']}/{w.get('group',w.get('band'))}，{w['rows']}列，{f(w['delta'])}元／{w['relative_change_pct']:.3f}%" if w else '無'
            lines.append(f"|{s}|{p}|{f(d['strata'].get('seen',{}).get('delta'))}|{f(d['strata'].get('new_nonmissing',{}).get('delta'))}|{len(bad)}|{label}|")
        cities=a['comparisons']['T-v22']['cities']; supported=[(k,v) for k,v in cities.items() if v['rows']>=500]
        if supported:
            low=min(supported,key=lambda kv:kv[1]['delta']); high=max(supported,key=lambda kv:kv[1]['delta'])
            notes.append(f"\n{s}城市正反端點（至少500列，按群內MAE差）：最低{low[0]} {f(low[1]['delta'])}元；最高{high[0]} {f(high[1]['delta'])}元。最低亦可能仍退步；非另設選型門檻。\n")
    lines+=notes
    lines+=['','## 五價帶及尾部貢獻','','|窗口|真價帶|列數|T−v22群MAE差|T−C群MAE差|全窗貢獻T−v22|全窗貢獻T−C|','|---|---|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        for u,v in zip(a['price_band_decomposition']['T-v22'],a['price_band_decomposition']['T-C']): lines.append(f"|{s}|{u['band']}|{u['rows']}|{f(u.get('delta'))}|{f(v.get('delta'))}|{f(u['overall_delta_contribution_TWD'])}|{f(v['overall_delta_contribution_TWD'])}|")
    lines+=['','全窗貢獻＝保存群列數×群內MAE差÷全窗列數；五带加總核至1e−6元。這是事後聚合診斷，不是因果證明或依未知真價推論分流。完整群及空群保存在原結果與摘要JSON。','',
        '## 全部24個保存配對區間','','|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|','|---|---|---|---:|---|---:|---|']
    for v in r['saved_intervals']: lines.append(f"|{v['stage']}|T−{v['control']}|{v['family']}/{v['metric']}|{f(v['point_delta'])}|{f(v['percentile_2_5'])}～{f(v['percentile_97_5'])}|{v['valid_replicates']}/{v['invalid_replicates']}|{v['direction']}|")
    lines+=['',f"區間方向計數：{json.dumps(r['interval_directions'],ensure_ascii=False)}。固定2000次，未重抽；不是Private勝率。",'', '## 實測時間與限制','','|窗口|C fit秒|T fit秒|stage秒|','|---|---:|---:|---:|']
    for s,a in r['stages'].items():
        t=a['timing']; lines.append(f"|{s}|{t['arms']['C']['fit_seconds']:.3f}|{t['arms']['T']['fit_seconds']:.3f}|{t['stage_execution_seconds']:.3f}|")
    t=r['timing']; lines+=['',f"六NN fit共{t['NN_fit_total_seconds']:.3f}秒，已含於main {t['main_execution_seconds']:.3f}秒，不重複相加；獨立評分另{t['evaluation_execution_seconds']:.3f}秒。0新樹、0prior重建，未量開發聊天工時。",'單seed、反覆年度窗口、沒有初始RNG trace；同寬不等於完全因果隔離。全eligible fit標準化、encoder/query prior/R/S的outer-fit範圍仍在，不能稱所有輸入逐列as-of。bootstrap不含重訓、累積選型或未來分布移轉。','', '[固定計畫](round102_plan.md) · [原完整結果](../outputs/round102/evaluation/results.json) · [摘要JSON](../outputs/round102_summary/summary.json) · [來源](../src/summarize_round102.py)','']
    return '\n'.join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(),'Existing summary preserved'); OUT.mkdir(); tick=time.monotonic(); inputs={}; code=1; failure=None
    sources={str(Path(p).relative_to(ROOT).as_posix()):sha(p) for p in (__file__,common.__file__)}
    try:
        r=collect(inputs); r.update(source_sha256=sources,created_at_utc=datetime.now(timezone.utc).isoformat()); write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(all(sha(ROOT/p)==h for p,h in {**inputs,**sources}.items()),'Source/input changed')
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/'summary.json',OUT/'summary.sha256')}
        write(OUT/'receipt.json',{'status':'completed','source_sha256':sources,'input_sha256':inputs,'artifact_sha256':artifacts,'saved_interval_count':24,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0,'elapsed_seconds':time.monotonic()-tick}); code=0
        print(json.dumps({'status':'completed','artifacts':artifacts,'elapsed_seconds':time.monotonic()-tick}))
    except Exception: failure=traceback.format_exc(); raise
    finally: write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':sources,'input_sha256':inputs,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'new_fit_predict_score_bootstrap_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); a=p.parse_args()
    if not a.run:p.error('--run required after main/evaluator exit0')
    run()
