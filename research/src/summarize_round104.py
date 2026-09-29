"""R104 metadata-only report; execute only after teacher, students and scorer exit 0.

No NumPy, Torch, price table, prediction vector, model or bootstrap loader.
Arithmetic below only decomposes already-saved aggregate metrics.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import summarize_round100 as common

ROOT=common.ROOT; OUT=ROOT/'outputs/round104_summary'; REPORT=ROOT/'reports/round104_report.md'
STAGES=common.STAGES; KEYS=common.MEASURES
LAYERS=('native','base','reference','spatial','median')
PAIRS={'T-v22':'comparison','T-C':'comparison_vs_C','C-v22':'C_vs_v22'}
BASE=tuple('outputs/round104/'+s for s in ('evaluation/results.json','sealed_manifest.json','execution.json','evaluation/execution.json'))
TEACHER='outputs/round104_teacher/models/'
ALLOWED=set(BASE)|{TEACHER+p for p in ('sealed_manifest.json','execution.json','area/training.json','lgb/training.json')}
ALLOWED|={f'outputs/round104/{s}/{p}' for s in STAGES for p in
          ('stage_seal.json','execution.json','preparation/preparation.json','C/execution.json','T/execution.json','C/model/training.json','T/model/training.json')}
need,sha,write=common.need,common.sha,common.write


def interval_direction(low,high):
    if low is None or high is None:return 'unavailable'
    need(low<=high,'Interval endpoints reversed')
    return 'negative' if high<0 else 'positive' if low>0 else 'crosses_zero'


def collect(inputs):
    def read(name,expected=None):
        need(name in ALLOWED,'Only explicitly allowed aggregate/timing JSON may be opened: '+name)
        path=ROOT/name; digest=sha(path)
        need(expected is None or digest==expected,'Bound metadata changed: '+name)
        side=path.with_suffix('.sha256')
        if side.exists():
            need(side.read_text(encoding='ascii').strip()==digest,'Metadata sidecar differs')
            inputs[side.relative_to(ROOT).as_posix()]=sha(side)
        inputs[name]=digest
        return json.loads(path.read_text(encoding='utf-8-sig'))
    def bound(record):return read(record['path'],record['sha256'])
    def complete(ex):need(ex['exit_code']==0 and ex['failure'] is None,'Incomplete execution')
    result,seal,main,evaluation=(read(p) for p in BASE)
    complete(main);complete(evaluation)
    need(result['round']==104 and result['status']=='completed_three_window_evaluation','Incomplete R104 evaluation')
    need(result['sealed_manifest']=={'path':BASE[1],'sha256':inputs[BASE[1]]},'Evaluation seal identity')
    need(seal['status']=='all_base_aware_predictions_sealed' and seal['stage_order']==list(STAGES),'Incomplete root seal')
    need(seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0 and seal['new_teacher_native_fits']==2
         and seal['new_native_fits_total']==8 and seal['all_query_reload_exact'],'Eight native fit/reload scope differs')
    need(result['conditional_intervals_reported']==24 and result['selected_submission']=='v22'
         and not result['new_CSV'] and result['Private_score'] is None,'Evaluation scope differs')
    teacher=bound(seal['teacher_seal']); teacher_ex=bound(seal['teacher_execution']);complete(teacher_ex)
    need(result['teacher_seal']==seal['teacher_seal'] and teacher['status']=='pre_auxiliary_teachers_sealed'
         and teacher['new_NN_fits']==teacher['new_native_tree_fits']==1 and teacher['new_native_fits']==2
         and teacher['all_query_reload_exact'],'Teacher scope differs')
    teacher_times={}
    for name in ('area','lgb'):
        a=teacher['teachers'][name];tr=bound(a['training'])
        need(tr['status']=='completed' and tr['fit_seconds']==a['fit_seconds'] and a['saved_model_reload_exact'],'Teacher fit metadata differs')
        teacher_times[name]={'fit_seconds':a['fit_seconds'],'training':a['training']}
    need(abs(sum(a['fit_seconds'] for a in teacher_times.values())-teacher['fit_seconds'])<1e-8,'Teacher fit sum differs')
    for doc in (seal,teacher,result):need(doc['protected_before']==doc['protected_after'],'Saved protected identity changed')
    stages={};intervals=[]
    for s in STAGES:
        r=result['stages'][s]; binding=seal['stages'][s]
        ss=bound(binding['stage_manifest']);ex=bound(binding['execution']);complete(ex)
        prep=bound(ss['preparation'])
        need(ss['status']=='base_aware_stage_predictions_sealed' and ss['new_NN_fits']==2
             and ss['new_native_tree_fits']==0 and ss['all_query_reload_exact'],'Incomplete student stage')
        need(r['full_original_v22_metrics_replay_exact'] and ss['original_v22_all_five_layers_replay_exact'],'v22 replay differs')
        need(ss['same_prepared_for_C_and_T'] and ss['R_S_state_and_frames_unchanged'],'Matched control/state differs')
        need(r['student_fit_rows']==ss['student_fit_rows']==prep['fit_rows'] and r['query_rows']==prep['query_rows'],'Student row metadata differs')
        comparisons={p:r[k] for p,k in PAIRS.items()};decompositions={}
        for p,d in comparisons.items():
            bands=[{**b,'overall_delta_contribution_TWD':b['rows']*b['delta']/r['query_rows'] if b['rows'] else 0.} for b in d['price_bands']]
            need(len(bands)==5 and sum(b['rows'] for b in bands)==r['query_rows'],'Five price bands must partition query')
            need(abs(sum(b['overall_delta_contribution_TWD'] for b in bands)-d['overall_mae']['delta'])<1e-6,'Saved aggregate band identity differs')
            decompositions[p]=bands
        for arm in ('v22','C','T'):need(r['layer_metrics'][arm]['median']==r['metrics'][arm],'Final/median aggregate identity differs')
        deltas={layer:{p:{k:r['layer_metrics'][a][layer][k]-r['layer_metrics'][b][layer][k] for k in KEYS}
                       for p,a,b in (('T-v22','T','v22'),('T-C','T','C'),('C-v22','C','v22'))} for layer in LAYERS}
        old=r['historical_R102T_point_background']
        background={layer:{k:r['layer_metrics']['T'][layer][k]-old['layer_metrics'][layer][k] for k in KEYS} for layer in LAYERS}
        need(all(abs(background['median'][k]-old['T_minus_R102T'][k])<1e-6 for k in KEYS),'R102 background delta differs')
        for control in ('v22','C'):
            for family in ('project','month'):
                for measure in ('overall','weighted'):
                    v=r['paired'][control]['methods'][family]['summary'][measure]
                    need(v['valid_replicates']+v['invalid_replicates']==2000,'Saved replicate count differs')
                    intervals.append({'stage':s,'control':control,'family':family,'metric':measure,
                                      'direction':interval_direction(v['percentile_2_5'],v['percentile_97_5']),**v})
        times={}
        for arm in ('C','T'):
            a=ss['arms'][arm];ae=bound(a['execution']);complete(ae);tr=bound(a['training'])
            need(tr['status']=='completed' and tr['objective']==arm and a['saved_model_reload_exact']
                 and tr['fit_seconds']==a['fit_seconds'] and len(tr['epochs'])==30,'Student fit metadata differs')
            times[arm]={'fit_seconds':a['fit_seconds'],'trainer_process_seconds':ae['elapsed_seconds'],
                        'trainer_total_seconds':tr['elapsed_seconds'],
                        'negative_equivalent_residual_target_rows':tr['negative_equivalent_residual_target_rows']}
        need(abs(sum(a['fit_seconds'] for a in times.values())-ss['fit_seconds'])<1e-8,'Student fit sum differs')
        stages[s]={'saved_stage_results':r,'comparisons':comparisons,'layer_deltas':deltas,
                   'T_minus_R102T_layer_deltas':background,'price_band_decomposition':decompositions,
                   'sample_coverage':{k:prep[k] for k in ('source_outer_fit_rows','source_eligible_rows','source_ineligible_rows',
                       'eligible_without_forward_teacher_rows','teacher_counts','fit_rows','query_rows',
                       'source_sample_weight_sum_float64','selected_sample_weight_sum_float64')},
                   'timing':{'arms':times,'stage_execution_seconds':ex['elapsed_seconds'],'preparation_seconds':prep['elapsed_seconds']}}
    need(len(intervals)==24,'All 24 intervals required')
    parts={f'{layer}_all_three_{key}_improve':all(stages[s]['layer_deltas'][layer]['T-C'][key]<0 for s in STAGES)
           for layer in ('base','median') for key in KEYS}
    gates=result['research_gates']
    need(parts==gates['mechanism']['components'] and all(parts.values())==gates['mechanism']['mechanism_gate_passed'],'Saved mechanism gate differs')
    need(not gates['mechanism']['native_improvement_required'],'R104 does not require native improvement')
    student_fit=sum(a['fit_seconds'] for s in stages.values() for a in s['timing']['arms'].values())
    need(abs(student_fit-seal['student_fit_seconds'])<1e-8,'Total student fit time differs')
    return {'round':104,'status':'aggregate_summary_completed','stages':stages,'research_gates':gates,
        'saved_intervals':intervals,'interval_directions':{k:sum(v['direction']==k for v in intervals) for k in ('negative','crosses_zero','positive','unavailable')},
        'teacher':{k:teacher[k] for k in ('origin','query_end','fit_rows','query_rows','geography')},
        'timing':{'teacher_arms':teacher_times,'teacher_fit_seconds':teacher['fit_seconds'],
                  'teacher_execution_seconds':teacher_ex['elapsed_seconds'],'student_fit_seconds':student_fit,
                  'eight_native_fit_seconds':teacher['fit_seconds']+student_fit,'main_execution_seconds':main['elapsed_seconds'],
                  'evaluation_inner_seconds':result['elapsed_seconds'],'evaluation_execution_seconds':evaluation['elapsed_seconds'],
                  'new_native_fits':8,'new_NN_fits':7,'new_native_tree_fits':1,'prior_or_R_S_build_calls':0,
                  'fit_is_included_in_process_times':True,'chat_time_measured':False},
        'original_results':{'path':BASE[0],'sha256':inputs[BASE[0]]},'input_sha256':inputs,
        'selected_submission':'v22','Private_score':None,'new_CSV':False,'row_arrays_or_labels_read':False,
        'new_fit_predict_score_bootstrap_calls':0,'author_role':'R104 trainer and student main driver author; aggregate summary is not a fully independent audit'}


def render(r):
    f=lambda x:'—' if x is None else f'{x:,.2f}'
    pair=lambda x:' / '.join(f(x[k]) for k in KEYS)
    g=r['research_gates']; lines=['# R104：以固定 base 為目標的配對學生實驗','',
        f"Primary門檻 **{g['primary']['adoption_research_gate_passed']}**；mechanism門檻 **{g['mechanism']['mechanism_gate_passed']}**。v22唯一提交不變，Private未知，無新CSV。",'',
        'C、T使用同一教師覆蓋子集、145數值欄／23類別欄、factor、日期權重、初始化及30 epochs。C=.375×原生加權L1；T為加權|.375p+教師offset−y|。C是本輪重訓的梯度尺度對照，不能當成原R102T。教師offset不供query推論，也不裁切負的等價殘差目標。',
        '本程式只讀封存聚合及訓練metadata；評分由另一作者的scorer完成。摘要作者曾實作trainer及main driver，並非全程獨立審查。以下差值前者減後者，負值較好；單位TWD。','',
        '|窗口|v22整體／加權|新C整體／加權|T整體／加權|原R102T整體／加權|','|---|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        z=a['saved_stage_results'];m=z['metrics'];lines.append('|'+s+'|'+'|'.join(pair(m[k]) for k in ('v22','C','T'))+'|'+pair(z['historical_R102T_point_background']['metrics'])+'|')
    lines+=['','|窗口|T−v22整體／加權|T−C|C−v22|T−原R102T（點比較）|','|---|---:|---:|---:|---:|']
    for s,a in r['stages'].items():lines.append('|'+s+'|'+'|'.join(pair({k:a['comparisons'][p][k]['delta'] for k in KEYS}) for p in PAIRS)+'|'+pair(a['T_minus_R102T_layer_deltas']['median'])+'|')
    lines+=['','## 事前固定門檻','','|門檻|條件|通過|','|---|---|---|']
    for name in ('primary','mechanism'):
        for k,v in g[name]['components'].items():lines.append(f'|{name}|{k}|{v}|')
    lines+=['','Primary要求三期完整模型整體及固定加權均改善、confirmation至少0.5%，且固定季度或至少500列群沒有>5%退步。Mechanism要求base及median的整體／加權指標三期均勝新C；不要求native改善。兩門檻分開，通過也不等於自動採用或Private保證。','',
        '## 五層結果','','每格為整體／加權差。native是prior NN，spatial仍只是一支N，median才是完整模型。','','|窗口|層級|T−v22原分量|T−C|C−v22原分量|T−原R102T|','|---|---|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        for layer,d in a['layer_deltas'].items():lines.append('|'+s+'|'+layer+'|'+'|'.join(pair(d[p]) for p in PAIRS)+'|'+pair(a['T_minus_R102T_layer_deltas'][layer])+'|')
    lines+=['','## 固定分群與正反證','','|窗口|比較|seen差|new差|>5%群數|最大相對退步群|','|---|---|---:|---:|---:|---|']
    for s,a in r['stages'].items():
        for p,d in a['comparisons'].items():
            bad=d['quarter_or_large_group_deterioration_above_5pct'];worst=max(bad,key=lambda b:b['relative_change_pct']) if bad else None
            label=f"{worst['section']}/{worst.get('group',worst.get('band'))}，{worst['rows']}列，{f(worst['delta'])}元／{worst['relative_change_pct']:.3f}%" if worst else '無'
            lines.append(f"|{s}|{p}|{f(d['strata'].get('seen',{}).get('delta'))}|{f(d['strata'].get('new_nonmissing',{}).get('delta'))}|{len(bad)}|{label}|")
    lines+=['','|窗口|比較|固定群（季度或至少500列）|列數|MAE差|相對差%|','|---|---|---|---:|---:|---:|']
    for s,a in r['stages'].items():
        for p in ('T-v22','T-C'):
            d=a['comparisons'][p]
            for section in ('strata','forecast_quarters','cities'):
                for name,v in d[section].items():
                    if section=='forecast_quarters' or v['rows']>=500:lines.append(f"|{s}|{p}|{section}/{name}|{v['rows']}|{f(v['delta'])}|{f(v['relative_change_pct'])}|")
    lines+=['','季度為原評分的預測期季度，不改成曆年季度；群重疊，不能把城市、新舊案與季度差額相加。完整月份、全部退步門檻清單與空群保留於摘要saved_stage_results。','',
        '## 五價帶與尾部','','|窗口|真價帶|列數|T−v22群MAE差|T−C群MAE差|全窗貢獻T−v22|全窗貢獻T−C|','|---|---|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        for u,v in zip(a['price_band_decomposition']['T-v22'],a['price_band_decomposition']['T-C']):
            need(u['band']==v['band'] and u['rows']==v['rows'],'Band labels differ')
            lines.append(f"|{s}|{u['band']}|{u['rows']}|{f(u.get('delta'))}|{f(v.get('delta'))}|{f(u['overall_delta_contribution_TWD'])}|{f(v['overall_delta_contribution_TWD'])}|")
    lines+=['','全窗貢獻＝群列數×保存群MAE差÷全窗列數，五帶加總核至1e−6元。這是事後診斷，不是依未知真價分流；原始signed error、underprediction與各模型尾部摘要完整保留，模型各自最大誤差列不視為同一批樣本。','',
        '## 全部24個保存配對區間','','|窗口|比較|抽樣／指標|點差|2.5%～97.5%|有效／無效|方向|','|---|---|---|---:|---|---:|---|']
    for v in r['saved_intervals']:lines.append(f"|{v['stage']}|T−{v['control']}|{v['family']}/{v['metric']}|{f(v['point_delta'])}|{f(v['percentile_2_5'])}～{f(v['percentile_97_5'])}|{v['valid_replicates']}/{v['invalid_replicates']}|{v['direction']}|")
    lines+=['',f"方向計數：{json.dumps(r['interval_directions'],ensure_ascii=False)}。沿固定seed與每組2000次保存結果，摘要未重抽。R102T、C−v22仅點比較，沒有追加bootstrap；區間不是Private勝率。",'',
        '## 樣本覆蓋與實測成本','','|窗口|outer fit列|原eligible列|新學生列|eligible未有教師列|query列|保留日期權重質量比例|','|---|---:|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        v=a['sample_coverage'];mass=v['selected_sample_weight_sum_float64']/v['source_sample_weight_sum_float64']
        lines.append(f"|{s}|{v['source_outer_fit_rows']}|{v['source_eligible_rows']}|{v['fit_rows']}|{v['eligible_without_forward_teacher_rows']}|{v['query_rows']}|{mass:.6f}|")
    lines+=['','|窗口|C fit秒|T fit秒|C程序秒|T程序秒|準備秒|stage秒|','|---|---:|---:|---:|---:|---:|---:|']
    for s,a in r['stages'].items():
        t=a['timing'];x=t['arms'];lines.append(f"|{s}|{x['C']['fit_seconds']:.3f}|{x['T']['fit_seconds']:.3f}|{x['C']['trainer_process_seconds']:.3f}|{x['T']['trainer_process_seconds']:.3f}|{t['preparation_seconds']:.3f}|{t['stage_execution_seconds']:.3f}|")
    t=r['timing'];teacher=r['teacher']
    lines+=['',f"新增早期教師origin {teacher['origin']}、query截至{teacher['query_end']}：fit {teacher['fit_rows']}列／query {teacher['query_rows']}列。area NN fit {t['teacher_arms']['area']['fit_seconds']:.3f}秒，LGB fit {t['teacher_arms']['lgb']['fit_seconds']:.3f}秒；教師程序 {t['teacher_execution_seconds']:.3f}秒。較後aux/dev教師沿用原forward向量。",
        f"八次native fit＝7 NN＋1 LGB：教師fit {t['teacher_fit_seconds']:.3f}秒＋六學生fit {t['student_fit_seconds']:.3f}秒＝{t['eight_native_fit_seconds']:.3f}秒。學生main程序 {t['main_execution_seconds']:.3f}秒，獨立scorer內層 {t['evaluation_inner_seconds']:.3f}秒／execution {t['evaluation_execution_seconds']:.3f}秒。fit已含於各自程序耗時，不重複加總；0新prior/R/S建模，未量聊天開發工時。",'',
        '## 解讀界線','',
        '教師使用固定2023非價格地圖，晚於早期origin；只稱價格標籤依時間先後的forward預測，不能稱完整歷史as-of。原R90 guard未削弱；學生encoder/scaler/factor仍沿outer-fit範圍，也不是每個fit row逐列as-of。教師offset分布與實際query原area/LGB分布可能不同，教師覆蓋子集也改變訓練母體。',
        '單seed、30固定epochs，反覆利用既有年度窗口，並非盲測。matched-gradient C只控制部分因素，不證明因果；base loss也不保證reference、spatial或median受益。bootstrap僅條件於已保存預測，不含重新訓練、累積選型與未來分布變化。產物可追溯不等於乾淨環境端到端重現。',
        'v22仍是使用者唯一選定提交；Private未知，未查外部價格答案、未生成或上傳新CSV。','',
        '[固定計畫](round104_plan.md) · [完整聚合結果](../outputs/round104/evaluation/results.json) · [摘要JSON](../outputs/round104_summary/summary.json) · [摘要來源](../src/summarize_round104.py)','']
    return '\n'.join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(),'Existing summary preserved')
    OUT.mkdir();tick=time.monotonic();inputs={};code=1;failure=None
    sources={Path(p).relative_to(ROOT).as_posix():sha(p) for p in (__file__,common.__file__)}
    try:
        r=collect(inputs);r.update(source_sha256=sources,created_at_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(all(sha(ROOT/p)==h for p,h in {**sources,**inputs}.items()),'Source/metadata changed during summary')
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/'summary.json',OUT/'summary.sha256')}
        write(OUT/'receipt.json',{'status':'completed','source_sha256':sources,'input_sha256':inputs,'artifact_sha256':artifacts,
            'saved_interval_count':24,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0,
            'author_role':r['author_role'],'elapsed_seconds':time.monotonic()-tick})
        code=0;print(json.dumps({'status':'completed','artifacts':artifacts,'elapsed_seconds':time.monotonic()-tick}))
    except Exception:failure=traceback.format_exc();raise
    finally:write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':sources,'input_sha256':inputs,
        'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'row_arrays_or_labels_read':False,'new_fit_predict_score_bootstrap_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('--run required, only after all real teacher/student/scorer executions exit 0')
    run()
