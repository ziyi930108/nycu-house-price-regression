"""R100 summary using saved aggregate JSON only; no arrays, labels or rescoring."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round100_summary'
REPORT=ROOT/'reports/round100_summary.md'
STAGES=('development','confirmation','auxiliary')
BASE=('outputs/round100/evaluation/results.json','outputs/round100/sealed_manifest.json',
      'outputs/round100/execution.json','outputs/round100/evaluation/execution.json')
TIMELINE=('outputs/round100_launcher_20260929/research_request.json',
          'outputs/round100_launcher_20260929/evaluation_request.json',
          'outputs/round100_launcher_20260929/evaluation_request_v2.json',
          'outputs/round100_evaluator_v2_revision/revision.json',
          'outputs/round100_evaluator_v2_revision/actual_metadata_check.json')
ALLOWED=set(BASE+TIMELINE)|{f'outputs/round100/{s}/{p}' for s in STAGES for p in
    ('stage_seal.json','execution.json','temporal_prior_evidence.json','C/execution.json','T/execution.json')}
PAIRS={'T-v22':'comparison','T-C':'comparison_vs_C','C-v22':'C_vs_v22','T-v24':'comparison_vs_v24'}
MEASURES=('overall_mae','weighted_mae')


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


def collect(inputs):
    def read(name,expected=None):
        need(name in ALLOWED,'Only saved aggregate JSON may be opened')
        p=ROOT/name;digest=sha(p)
        if expected is not None:need(digest==expected,'Bound JSON changed: '+name)
        side=p.with_suffix('.sha256')
        if side.is_file():
            need(side.read_text(encoding='ascii').strip()==digest,'JSON sidecar differs')
            inputs[side.relative_to(ROOT).as_posix()]=sha(side)
        inputs[name]=digest
        return json.loads(p.read_text(encoding='utf-8-sig'))
    result,seal,main,evaluation=(read(p) for p in BASE)
    need(main['exit_code']==evaluation['exit_code']==0 and main['failure'] is None and evaluation['failure'] is None,'Incomplete run')
    need(result['status']=='completed_three_window_six_fit_evaluation' and result['checked_stages']==list(STAGES),'Incomplete scoring')
    need(result['sealed_manifest']['sha256']==inputs[BASE[1]] and seal['status']=='all_temporal_prior_predictions_sealed','Seal identity')
    need(seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0 and seal['all_query_reload_exact'],'Six fits/reload incomplete')
    need(result['conditional_intervals_reported']==24 and result['new_CSV'] is False and result['Private_score'] is None,'Scope differs')
    launch,old_eval,new_eval,revision,metadata_check=(read(p) for p in TIMELINE)
    need(seal['request']=={'path':TIMELINE[0],'sha256':inputs[TIMELINE[0]]},'Original research request identity')
    need(result['evaluation_request']=={'path':TIMELINE[2],'sha256':inputs[TIMELINE[2]]},'Revised evaluator identity')
    need(len(launch['frozen_source_and_plan'])==45 and len(launch['input_authority_sha256'])==11,'Original freeze inventory')
    need(old_eval['parameters']==new_eval['parameters'] and old_eval['plan_sha256']==new_eval['plan_sha256'], 'Scientific evaluation changed')
    need(revision['status']=='engineering_revision_before_real_evaluation'
         and revision['old_frozen_sources_modified'] is False and revision['true_evaluation_executed'] is False
         and revision['true_fit_failure'] is False and revision['model_training_already_running'] is True
         and revision['model_configuration_or_metrics_bootstrap_adoption_rules_changed'] is False,'Revision scope differs')
    need(result['source_sha256'][revision['new_source']['path']]==revision['new_source']['sha256']
         and metadata_check['status']=='passed_actual_metadata_contract_check','Final evaluator source differs')
    stages={};intervals=[]
    for s in STAGES:
        r=result['stages'][s];binding=seal['stages'][s]
        ss=read(binding['stage_manifest']['path'],binding['stage_manifest']['sha256'])
        execution=read(binding['execution']['path'],binding['execution']['sha256'])
        evidence=read(ss['temporal_prior_evidence']['path'],ss['temporal_prior_evidence']['sha256'])
        need(execution['exit_code']==0 and execution['failure'] is None and ss['new_NN_fits']==2,'Incomplete stage')
        need(ss['all_query_reload_exact'] and r['full_original_v22_and_v24_metrics_replay_exact'],'Original/reload control failed')
        need(evidence['saved_state_records_complete'] and evidence['prefix_states_saved'],'Missing prefix state receipt')
        comparisons={k:r[v] for k,v in PAIRS.items()}
        decompositions={}
        for name,d in comparisons.items():
            bands=[{**b,'overall_delta_contribution_TWD':b['rows']*b['delta']/r['query_rows'] if b['rows'] else 0.}
                   for b in d['price_bands']]
            total=sum(b['overall_delta_contribution_TWD'] for b in bands)
            need(sum(b['rows'] for b in bands)==r['query_rows'] and abs(total-d['overall_mae']['delta'])<=1e-6,'Price-band aggregate identity')
            need(bands[-1]['band']=='>=100m','Tail band identity')
            decompositions[name]={'bands':bands,'sum_TWD':total,
                'at_least_100m_contribution_TWD':bands[-1]['overall_delta_contribution_TWD'],
                'other_bands_combined_contribution_TWD':total-bands[-1]['overall_delta_contribution_TWD'],
                'post_hoc_aggregate_diagnostic':True,'not_an_inference_rule_or_new_gate':True}
        for control in ('v22','C'):
            for family in ('project','month'):
                for measure in ('overall','weighted'):
                    v=r['paired'][control]['methods'][family]['summary'][measure]
                    need(v['valid_replicates']+v['invalid_replicates']==2000,'Saved replicate count differs')
                    lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    direction='unavailable' if lo is None or hi is None else ('negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero')
                    intervals.append({'stage':s,'control':control,'family':family,'metric':measure,'direction':direction,**v})
        layers={**r['layer_metrics']}
        for arm in ('v22','C','T'):
            layers[arm]={**layers[arm],'median':r['metrics'][arm]}
        layer_deltas={}
        for layer in ('native','base','reference','spatial','median'):
            layer_deltas[layer]={pair:{key:layers[candidate][layer][key]-layers[control][layer][key] for key in MEASURES}
                for pair,candidate,control in [('T-v22','T','v22'),('T-C','T','C'),('C-v22','C','v22')]}
        arm_times={}
        for arm in ('C','T'):
            a=ss['arms'][arm];ex=read(a['execution']['path'],a['execution']['sha256'])
            need(ex['exit_code']==0 and ex['failure'] is None and a['saved_model_reload_exact'],'Incomplete arm')
            arm_times[arm]={'fit_seconds':a['fit_seconds'],'trainer_process_seconds':ex['elapsed_seconds']}
        need(abs(sum(a['fit_seconds'] for a in arm_times.values())-ss['fit_seconds'])<1e-8,'Fit timing differs')
        stages[s]={'cutoff':r['cutoff'],'query_end':r['query_end'],'fit_rows':r['fit_rows'],'eligible_rows':r['eligible_rows'],
            'query_rows':r['query_rows'],'warmup_rows':evidence['warmup_rows'],'invalid_date_rows':evidence['invalid_date_rows'],
            'prefix_state_count':evidence['prefix_state_count'],'metrics':r['metrics'],'comparisons':comparisons,
            'layer_metrics':layers,'layer_deltas':layer_deltas,'decompositions':decompositions,
            'timing':{'arms':arm_times,'prior_preparation_seconds':evidence['elapsed_seconds'],
                      'stage_execution_seconds':execution['elapsed_seconds']}}
    need(len(intervals)==24,'All twenty-four saved intervals required')
    fit=sum(x['timing']['arms'][a]['fit_seconds'] for x in stages.values() for a in ('C','T'))
    prior=sum(x['timing']['prior_preparation_seconds'] for x in stages.values())
    need(fit<main['elapsed_seconds'] and prior<main['elapsed_seconds'],'Timing hierarchy differs')
    directions={k:sum(v['direction']==k for v in intervals) for k in ('negative','crosses_zero','positive','unavailable')}
    return {'round':100,'status':'aggregate_summary_completed','stage_order':list(STAGES),'stages':stages,
        'saved_intervals':intervals,'interval_directions':directions,'research_gate':result['research_gate'],
        'timing':{'NN_fit_total_seconds':fit,'prior_preparation_total_seconds':prior,'main_execution_seconds':main['elapsed_seconds'],
                  'fit_and_prior_already_included_in_main':True,'evaluation_execution_seconds':evaluation['elapsed_seconds'],
                  'new_NN_fits':6,'new_native_tree_fits':0,'developer_chat_time_measured':False},
        'engineering_timeline':{'research_frozen_source_count':45,'input_authority_count':11,
            'revision':revision,'metadata_check':metadata_check,'original_and_v2_evaluation_parameters_equal':True,
            'original_research_request':seal['request'],'executed_evaluation_request':result['evaluation_request']},
        'selected_submission':'v22','Private_score':None,'new_CSV':False,'input_sha256':inputs,
        'reads_row_arrays_or_labels':False,'new_fit_predict_score_bootstrap_calls_by_summary':0,
        'limitations':['Adaptive experiment on reused three August windows; not blind validation.',
            'Single-seed matched-row control does not remove stochastic training variability or support-distribution differences.',
            'Only fit price prior is time-forward; encoder/query prior/R/S retain original full outer-fit scope.',
            'Saved bootstrap excludes refitting and cumulative selection, and is not Private probability.',
            'True-price-band decomposition is post-hoc aggregate diagnosis, not a causal proof or inference routing rule.']}


def render(r):
    f=lambda v:f'{v:,.2f}'
    pair=lambda d:f'{f(d["overall_mae"])} / {f(d["weighted_mae"])}'
    gate=r['research_gate']['adoption_research_gate_passed']
    statement='固定探索門檻通過，僅值得另立採用研究' if gate else '固定探索門檻未通過'
    lines=['# R100 時間前置先驗：完成摘要','',f'2026-09-29。**{statement}；v22 唯一提交不變，Private 未知，沒有新 CSV 或自動採用。**','',
        'T 為時間前置且排除 held fold 的 prior；C 只使用相同 eligible 訓練列，prior 仍為原 R60 分組 OOF。兩臂各三次 NN fit，其餘分支、query prior、encoder 與 R/S 保留原同窗完整 fit 狀態。此摘要只读已保存聚合 JSON，不載逐筆資料或重算分數。','',
        '所有差值是前者減後者，負數較好；表內「整體／加權」均為新臺幣元，後者是固定 seen/new 組成評分，不是日期訓練 loss。','',
        '|窗口|v22 整體／加權|C 整體／加權|T 整體／加權|eligible／原 fit|','|---|---:|---:|---:|---:|']
    for s in STAGES:
        a=r['stages'][s];m=a['metrics']
        lines.append(f'|{s}|{pair(m["v22"])}|{pair(m["C"])}|{pair(m["T"])}|{a["eligible_rows"]:,} / {a["fit_rows"]:,}|')
    lines+=['','|窗口|T−v22|T−C|C−v22|T−v24|','|---|---:|---:|---:|---:|']
    for s in STAGES:
        d=r['stages'][s]['comparisons']
        vals=[pair({k:d[p][k]['delta'] for k in MEASURES}) for p in PAIRS]
        lines.append('|'+s+'|'+'|'.join(vals)+'|')
    lines+=['','T−C 才是相同 loss rows 下的先驗對照；T−v22 同時含暖身資料移除。C 不是可依結果另挑的候選，較低訓練 loss 或單一窗口改善也不代表 Private 改善。','',
        '## 原生 NN 到完整模型','',
        '每格為整體／加權差。native 是 prior NN，spatial 是完整 N 分支，median 才是與固定 H/U 合併的最終總價；各層改善不可互相代稱。','',
        '|窗口|層級|T−原模型|T−C|C−原模型|','|---|---|---:|---:|---:|']
    for s in STAGES:
        for layer,d in r['stages'][s]['layer_deltas'].items():
            lines.append(f'|{s}|{layer}|{pair(d["T-v22"])}|{pair(d["T-C"])}|{pair(d["C-v22"])}|')
    lines+=['','## 固定門檻與分群反證','',
        '|門檻|結果|','|---|---|']
    for k,v in r['research_gate']['components'].items():lines.append(f'|{k}|{v}|')
    lines+=[f'','confirmation 整體改善比例：'+f'{r["research_gate"]["confirmation_fractional_improvement"]*100:.4f}%（固定需求至少 0.5%）。','',
        '|窗口|群|列數|T−v22|T−C|C−v22|','|---|---|---:|---:|---:|---:|']
    for s in STAGES:
        d=r['stages'][s]['comparisons']
        for name,a in d['T-v22']['strata'].items():
            lines.append(f'|{s}|{name}|{a["rows"]:,}|{f(a["delta"])}|{f(d["T-C"]["strata"][name]["delta"])}|{f(d["C-v22"]["strata"][name]["delta"])}|')
    lines+=['','以下列每窗、每個主要對照退步超過 5% 的固定季度或至少 500 列群數，並列最大相對退步群。完整清單不省略地保存在本摘要 JSON 的 comparisons，群可重疊，不能相加當獨立列數。','',
        '|窗口|比較|超限群數|最大相對退步群／列數|MAE 差／增幅|','|---|---|---:|---|---:|']
    for s in STAGES:
        for p in ('T-v22','T-C','C-v22','T-v24'):
            adverse=r['stages'][s]['comparisons'][p]['quarter_or_large_group_deterioration_above_5pct']
            if adverse:
                a=max(adverse,key=lambda v:v['relative_change_pct'])
                lines.append(f'|{s}|{p}|{len(adverse)}|{a["section"]}/{a.get("group",a.get("band"))} / {a["rows"]:,}|{f(a["delta"])} / {a["relative_change_pct"]:.3f}%|')
            else:lines.append(f'|{s}|{p}|0|無|—|')
    lines+=['','## 價格帶正反證與億元群貢獻','',
        '|窗口|真價帶|列數|T−v22 帶內 MAE|T−C 帶內 MAE|對全窗貢獻 T−v22|對全窗貢獻 T−C|',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        d=r['stages'][s]['decompositions']
        for a,b in zip(d['T-v22']['bands'],d['T-C']['bands']):
            if not a['rows']:continue
            lines.append(f'|{s}|{a["band"]}|{a["rows"]:,}|{f(a["delta"])}|{f(b["delta"])}|{f(a["overall_delta_contribution_TWD"])}|{f(b["overall_delta_contribution_TWD"])}|')
    lines+=['','各帶對全窗貢獻 = 保存群列數 × 帶內 MAE 差 ÷ 全窗列數，五帶加總核對 overall 差至 0.000001 元。這是事後聚合診斷，不是新門檻、因果證明或依未知真價推論分流。','']
    for s in STAGES:
        d=r['stages'][s]['decompositions'];a=d['T-v22'];b=d['T-C']
        lines.append(f'{s}：億元群 {a["bands"][-1]["rows"]:,} 筆，對全窗 T−v22 貢獻 {f(a["at_least_100m_contribution_TWD"])} 元，其餘價帶合計 {f(a["other_bands_combined_contribution_TWD"])} 元；T−C 的億元貢獻為 {f(b["at_least_100m_contribution_TWD"])} 元。')
    lines+=['','## 全部 24 個保存配對區間','',
        '每族固定 2,000 次；只轉錄保存端點與有效次數，沒有重新 bootstrap。負向表示整段小於零，正向表示整段大於零，跨零保留不確定性。','',
        '|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|','|---|---|---|---:|---|---:|---|']
    for a in r['saved_intervals']:
        lo,hi=a['percentile_2_5'],a['percentile_97_5'];bounds='無' if lo is None else f'{f(lo)} ～ {f(hi)}'
        lines.append(f'|{a["stage"]}|T−{a["control"]}|{a["family"]}/{a["metric"]}|{f(a["point_delta"])}|{bounds}|{a["valid_replicates"]} / {a["invalid_replicates"]}|{a["direction"]}|')
    lines+=['',f'全部方向計數：{json.dumps(r["interval_directions"],ensure_ascii=False)}。兩個對照的所有區間皆保留，不擇優呈現；它們不是 Private 勝率。','',
        '## 工程修正時間線','',
        '真訓練前已固定 45 份來源／計畫與 11 份歷史及前置 JSON。訓練進行中、真評分尚未開始時，來源審查發現 scorer 把 pure-X 專用計畫 d536… 的身分誤要求為科學計畫 bbb…；後者實際另綁於 pure-X request。保留原 a86… scorer、request 與來源副本，只新增 _v2 窄工程修正及 42 項人工檢查，兩版評估參數完全相同。原錯誤版未執行真評分，這不是模型 fit 失敗、模型改動或看到 query 分數後改設計。',
        '[修正收據](../outputs/round100_evaluator_v2_revision/revision.json)與[實際 metadata 核驗](../outputs/round100_evaluator_v2_revision/actual_metadata_check.json)保留完整細節；本次結果由另行固定的 v2 request 產生。','',
        '## 實測時間與限制','',
        '|窗口|C NN fit 秒|T NN fit 秒|prior 準備秒|stage 全流程秒|prefix states|','|---|---:|---:|---:|---:|---:|']
    for s in STAGES:
        a=r['stages'][s];t=a['timing']
        lines.append(f'|{s}|{t["arms"]["C"]["fit_seconds"]:.3f}|{t["arms"]["T"]["fit_seconds"]:.3f}|{t["prior_preparation_seconds"]:.3f}|{t["stage_execution_seconds"]:.3f}|{a["prefix_state_count"]}|')
    t=r['timing']
    lines+=['',f'六次 NN fit 共 **{t["NN_fit_total_seconds"]:.3f} 秒**；prior 準備共 **{t["prior_preparation_total_seconds"]:.3f} 秒**，兩者均已包含主流程 **{t["main_execution_seconds"]:.3f} 秒**，不可再相加。獨立評分與固定抽樣另 **{t["evaluation_execution_seconds"]:.3f} 秒**。沒有新樹模型 fit；開發、聊天與等待時間未量成訓練工時。',
        '三個年度窗口已多輪重用，單 seed 控制不能消除隨機訓練變異。只有 fit prior 逐列時間前置；encoder、query prior、R/S 仍取原完整 outer fit，不能宣稱全部輸入皆為逐列 as-of。bootstrap 不涵蓋重訓、累積選型或未來市場移轉。完整正反證據保留，未由此改 v22 唯一提交或產生新 CSV，Private 仍未知。','',
        '[固定計畫](round100_plan.md) · [原評估結果](../outputs/round100/evaluation/results.json) · [聚合摘要](../outputs/round100_summary/summary.json) · [摘要程式](../src/summarize_round100.py)','']
    return '\n'.join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(),'Existing summary preserved; no overwrite')
    OUT.mkdir();tick=time.monotonic();start=datetime.now(timezone.utc).isoformat();inputs={};code=1;failure=None
    try:
        r=collect(inputs);r['source_sha256']=sha(__file__);r['created_at_utc']=datetime.now(timezone.utc).isoformat()
        write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(all(sha(ROOT/n)==h for n,h in inputs.items()),'Input aggregate JSON changed')
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/'summary.json',OUT/'summary.sha256')}
        write(OUT/'receipt.json',{'status':'completed','source_sha256':sha(__file__),'input_sha256':inputs,'artifact_sha256':artifacts,
            'saved_interval_count':24,'aggregate_band_identities_verified':True,'reads_row_arrays_or_labels':False,
            'new_fit_predict_score_bootstrap_calls':0,'selected_submission':'v22','Private_score':None,'new_CSV':False,
            'started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick})
        code=0;print(json.dumps({'status':'completed','artifacts':artifacts,'timing':r['timing'],'elapsed_seconds':time.monotonic()-tick}))
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':sha(__file__),'input_sha256':inputs,
            'started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick,
            'automatic_retry':False,'new_fit_predict_score_bootstrap_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('--run required after main and evaluator completion')
    run()
