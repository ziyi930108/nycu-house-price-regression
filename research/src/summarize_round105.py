"""Summarize completed R105 aggregate JSON only; never load arrays, labels or models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import summarize_round100 as common

ROOT=common.ROOT;OUT=ROOT/'outputs/round105_summary';REPORT=ROOT/'reports/round105_report.md'
STAGES=common.STAGES;KEYS=common.MEASURES;LAYERS=('native','base','reference','spatial','median')
PAIRS={'T-v22':'comparison','T-C':'comparison_vs_C','C-v22':'C_vs_v22'}
BASE='outputs/round105/'
ALLOWED={BASE+p for p in ('evaluation/results.json','sealed_manifest.json','execution.json','evaluation/execution.json')}
ALLOWED|={BASE+s+'/'+p for s in STAGES for p in ('stage_seal.json','execution.json')}
need,sha,write=common.need,common.sha,common.write
ZERO_KEYS=('new_fit_calls','new_native_fits','new_NN_fits','new_native_tree_fits','prior_or_R_S_build_calls','fit_seconds')
FLAGS=('all_query_reload_exact','all_saved_vectors_reload_exact','all_endpoints_replay_exact','original_v22_all_five_layers_replay_exact','original_R102T_all_five_layers_replay_exact','R_S_state_and_frames_unchanged')


def collect(inputs):
    def read(name,expected=None):
        need(name in ALLOWED,'Only saved aggregate/timing JSON allowed: '+name);path=ROOT/name;digest=sha(path)
        need(expected is None or expected==digest,'Bound metadata changed: '+name)
        side=path.with_suffix('.sha256')
        if side.exists():need(side.read_text(encoding='ascii').strip()==digest,'Metadata sidecar differs');inputs[side.relative_to(ROOT).as_posix()]=sha(side)
        inputs[name]=digest;return json.loads(path.read_text(encoding='utf-8-sig'))
    def bound(rec):return read(rec['path'],rec['sha256'])
    def complete(ex):need(ex['exit_code']==0 and ex.get('failure') is None,'Incomplete execution')
    main=read(BASE+'execution.json');evaluation=read(BASE+'evaluation/execution.json');complete(main);complete(evaluation)
    seal=read(BASE+'sealed_manifest.json');need(seal['round']==105 and seal['status']=='all_native_average_predictions_sealed' and seal['stage_order']==list(STAGES),'Wrong root seal')
    need(all(seal[k]==0 for k in ZERO_KEYS) and all(seal[k] is True for k in FLAGS),'Root zero-fit/replay scope differs')
    need(seal['protected_before']==seal['protected_after'],'Saved protected identities differ')
    stage_times={}
    for s in STAGES:
        ss=bound(seal['stages'][s]['stage_manifest']);ex=bound(seal['stages'][s]['execution']);complete(ex)
        need(ss['stage']==s and ss['status']=='native_average_stage_predictions_sealed' and all(ss[k] is True for k in FLAGS) and all(ss[k]==0 for k in ZERO_KEYS),'Incomplete stage reload/zero-fit scope')
        stage_times[s]=ex['elapsed_seconds']
    result=read(BASE+'evaluation/results.json')
    need(result['round']==105 and result['status']=='completed_three_window_evaluation' and set(result['stages'])==set(STAGES),'Wrong evaluation stages')
    need(result['new_fits']==result['model_load_predict_calls']==0 and result['variance_reduction_measured'] is False,'Scorer scope differs')
    need(result['sealed_manifest']=={'path':BASE+'sealed_manifest.json','sha256':inputs[BASE+'sealed_manifest.json']},'Evaluation seal differs')
    need(result['conditional_intervals_reported']==24 and result['selected_submission']=='v22' and result['Private_score'] is None and not result['new_CSV'],'Scope changed')
    stages={};intervals=[]
    for s in STAGES:
        r=result['stages'][s];need(r['both_endpoint_five_layer_metrics_replay_exact'],'Endpoint metric replay differs')
        comparisons={p:r[k] for p,k in PAIRS.items()};bands={}
        for p,d in comparisons.items():
            b=[{**v,'overall_delta_contribution_TWD':v['rows']*v['delta']/r['query_rows'] if v['rows'] else 0.} for v in d['price_bands']]
            need(len(b)==5 and sum(v['rows'] for v in b)==r['query_rows'],'Five bands must partition query')
            need(abs(sum(v['overall_delta_contribution_TWD'] for v in b)-d['overall_mae']['delta'])<1e-6,'Aggregate band identity differs');bands[p]=b
        for arm in ('v22','C','T'):need(r['layer_metrics'][arm]['median']==r['metrics'][arm],'Median/final identity differs')
        deltas={layer:{p:{k:r['layer_metrics'][a][layer][k]-r['layer_metrics'][b][layer][k] for k in KEYS}
                 for p,a,b in [('T-v22','T','v22'),('T-C','T','C'),('C-v22','C','v22')]} for layer in LAYERS}
        for control in ('v22','C'):
            for family in ('project','month'):
                for metric in ('overall','weighted'):
                    v=r['paired'][control]['methods'][family]['summary'][metric];lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    need(v['valid_replicates']+v['invalid_replicates']==2000,'Replicate count differs')
                    need(lo is None or hi is None or lo<=hi,'Reversed interval')
                    direction='unavailable' if lo is None or hi is None else 'negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero'
                    intervals.append({'stage':s,'control':control,'family':family,'metric':metric,'direction':direction,**v})
        stages[s]={'saved_stage_results':r,'comparisons':comparisons,'layer_deltas':deltas,'price_band_decomposition':bands,'stage_execution_seconds':stage_times[s]}
    need(len(intervals)==24,'All intervals required')
    return {'round':105,'status':'aggregate_summary_completed','stages':stages,'research_gates':result['research_gates'],
        'saved_intervals':intervals,'interval_directions':{k:sum(v['direction']==k for v in intervals) for k in ('negative','crosses_zero','positive','unavailable')},
        'timing':{'main_execution_seconds':main['elapsed_seconds'],'evaluation_inner_seconds':result['elapsed_seconds'],
                  'evaluation_execution_seconds':evaluation['elapsed_seconds'],'stage_times_included_in_main':True,'new_fit_calls':0,'chat_time_measured':False},
        'original_results':{'path':BASE+'evaluation/results.json','sha256':inputs[BASE+'evaluation/results.json']},'input_sha256':inputs,
        'selected_submission':'v22','user_reported_Public_MAE_TWD':1998568.5706,'Private_score':None,'new_CSV':False,
        'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0,
        'author_role':'Source-direction reviewer and aggregate summarizer; not an independent retraining audit'}


def render(r):
    f=lambda x:'—' if x is None else f'{x:,.2f}'
    pair=lambda x:' / '.join(f(x[k]) for k in KEYS)
    lines=['# R105：兩個 prior NN 的固定原生等權平均','',
        f"Primary **{r['research_gates']['primary']['adoption_research_gate_passed']}**；相對 R102T 端點補充 **{r['research_gates']['supplementary_endpoint']['supplementary_endpoint_gate_passed']}**。唯一提交 v22 不變，Private 未知，無新 CSV。",'',
        '候選 T=.5P+.5Q；P 為 v22 原 prior NN，Q 為 R102T。**本輪 C 是完整 R102T 端點**，不是 R102zeroC 或 R104 loss control。平均只在原生 TWD 層，再沿原 base／reference／spatial／median。無新 fit，兩端五層均重播後才評分。',
        '只讀保存 aggregate JSON，未載逐筆陣列／真價／模型或重算 MAE／bootstrap。金額為元，每格為整體／固定加權；差值負數較好。','',
        '|窗口|v22 整體／加權|C=R102T|T|T−v22|T−C|','|---|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        m=a['saved_stage_results']['metrics'];d=a['layer_deltas']['median'];lines.append('|'+s+'|'+'|'.join(pair(m[k]) for k in ('v22','C','T'))+'|'+pair(d['T-v22'])+'|'+pair(d['T-C'])+'|')
    lines+=['','## 固定門檻','','Primary 與相對 C 的完整三窗補充分開呈現，不要求 native 勝兩端；補充不是因果機制證明。保存的完整 gate 結果如下：','','```json',json.dumps(r['research_gates'],ensure_ascii=False,indent=2),'```','',
        '兩項全過也不自動替換使用者選定的 v22；未通過則保留反證，不依結果追加 alpha 或子群候選。','',
        '## 五層結果','','native 為 prior NN；spatial 是 N 分支，median 才是完整模型。層間 MAE 差不能相加為因果貢獻。','','|窗口|層級|v22 整體／加權|C 整體／加權|T 整體／加權|T−v22|T−C|','|---|---|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        for layer,d in a['layer_deltas'].items():
            lm=a['saved_stage_results']['layer_metrics'];lines.append('|'+s+'|'+layer+'|'+'|'.join(pair(lm[k][layer]) for k in ('v22','C','T'))+'|'+pair(d['T-v22'])+'|'+pair(d['T-C'])+'|')
    lines+=['','## 固定分群的正反證','','|窗口|比較|seen 差|new 差|>5%違例數|最大相對退步|','|---|---|---:|---:|---:|---|']
    for s,a in r['stages'].items():
        for p in ('T-v22','T-C'):
            d=a['comparisons'][p];bad=d['quarter_or_large_group_deterioration_above_5pct'];w=max(bad,key=lambda b:b['relative_change_pct']) if bad else None
            label=f"{w['section']}/{w.get('group',w.get('band'))}，{w['rows']}列，{f(w['delta'])}元／{w['relative_change_pct']:.3f}%" if w else '無'
            lines.append(f"|{s}|{p}|{f(d['strata'].get('seen',{}).get('delta'))}|{f(d['strata'].get('new_nonmissing',{}).get('delta'))}|{len(bad)}|{label}|")
    lines+=['','|窗口|比較|至少500列城市群的最低／最高 MAE 差|','|---|---|---|']
    for s,a in r['stages'].items():
        for p in ('T-v22','T-C'):
            cities=[(k,v) for k,v in a['comparisons'][p]['cities'].items() if v['rows']>=500]
            if cities:
                lo=min(cities,key=lambda kv:kv[1]['delta']);hi=max(cities,key=lambda kv:kv[1]['delta'])
                lines.append(f"|{s}|{p}|{lo[0]} {f(lo[1]['delta'])}／{hi[0]} {f(hi[1]['delta'])}|")
    lines+=['','最低也可能退步，最高也可能改善。城市、新舊案、季度是重疊分群，不能相加；所有月份、固定群及空群保留於摘要 saved_stage_results。','',
        '|窗口|真價帶|列數|T−v22 群MAE差|T−C 群MAE差|全窗貢獻 T−v22|全窗貢獻 T−C|','|---|---|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        for u,v in zip(a['price_band_decomposition']['T-v22'],a['price_band_decomposition']['T-C']):
            need(u['band']==v['band'] and u['rows']==v['rows'],'Band labels differ')
            lines.append(f"|{s}|{u['band']}|{u['rows']}|{f(u['delta'])}|{f(v['delta'])}|{f(u['overall_delta_contribution_TWD'])}|{f(v['overall_delta_contribution_TWD'])}|")
    lines+=['','全窗貢獻＝保存群列數×群MAE差÷全窗列數，五帶加總核至1e−6元；這是事後聚合診斷，不是新增門檻或依未知真價分流。','',
        '## 全部24個保存配對區間','','|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|','|---|---|---|---:|---|---:|---|']
    for v in r['saved_intervals']:lines.append(f"|{v['stage']}|T−{v['control']}|{v['family']}/{v['metric']}|{f(v['point_delta'])}|{f(v['percentile_2_5'])}～{f(v['percentile_97_5'])}|{v['valid_replicates']}/{v['invalid_replicates']}|{v['direction']}|")
    lines+=['',f"方向計數：{json.dumps(r['interval_directions'],ensure_ascii=False)}。每組固定2000次、seed20260925；本摘要沒有重抽。CI不是Private勝率。",'',
        '## 實測時間與解讀界線','','|步驟|秒數|','|---|---:|']
    for s,a in r['stages'].items():lines.append(f"|{s} driver stage|{a['stage_execution_seconds']:.3f}|")
    t=r['timing'];lines += [f"|driver main|{t['main_execution_seconds']:.3f}|",f"|scorer 內層／execution|{t['evaluation_inner_seconds']:.3f}／{t['evaluation_execution_seconds']:.3f}|",'',
        '三期 stage 已含於 main，scorer 內層已含於其 execution，不能重複相加。0 新 fit；driver 會載同窗已學 R/S 狀態並套用預測，不是全程無模型狀態讀取。以上非聊天／開發工時。',
        'MAE 凸性只保證原生等權損失不高於兩端損失的平均，不保證勝過任一端，更不保證 R/S／median 後改善。沒有多seed重訓或方差估計，不能宣稱已降低統計方差。固定 R/S 未針對新平均重新最佳化，保留這個限制。',
        '本候選受 R102–R104 結果啟發，仍是重用窗口的適應性研究；2023 靜態非價格地理晚於早期 origin，不是完整 historical-as-of 或新盲測。CI不涵蓋累積選型、重訓或未來分布變化。',
        '使用者唯一選定仍為 v22，已回報 Public MAE 1,998,568.5706 元，Private 未知。無新 CSV、上傳或最終勾選；來源與保存核對不等於乾淨環境端到端重訓。','',
        '[固定計畫](round105_plan.md) · [全部聚合結果](../outputs/round105/evaluation/results.json) · [摘要JSON](../outputs/round105_summary/summary.json) · [來源](../src/summarize_round105.py)','']
    return '\n'.join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(),'Existing summary preserved');OUT.mkdir();tick=time.monotonic();inputs={};code=1;failure=None
    sources={Path(p).relative_to(ROOT).as_posix():sha(p) for p in (__file__,common.__file__)}
    try:
        r=collect(inputs);r.update(source_sha256=sources,created_at_utc=datetime.now(timezone.utc).isoformat());write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(all(sha(ROOT/p)==h for p,h in {**sources,**inputs}.items()),'Source/metadata changed')
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/'summary.json',OUT/'summary.sha256')}
        write(OUT/'receipt.json',{'status':'completed','source_sha256':sources,'input_sha256':inputs,'artifact_sha256':artifacts,'saved_interval_count':24,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0,'elapsed_seconds':time.monotonic()-tick});code=0
        print(json.dumps({'status':'completed','artifacts':artifacts,'elapsed_seconds':time.monotonic()-tick}))
    except Exception:failure=traceback.format_exc();raise
    finally:write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':sources,'input_sha256':inputs,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('--run required only after three stages and evaluation exit 0')
    run()
