"""Summarize sealed R101 aggregate JSON only; never arrays, labels, models or scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round101_summary'
REPORT=ROOT/'reports/round101_report.md'
STAGES=('development','confirmation','auxiliary')
VIEWS=('unweighted','original_date_weight','effective_ratio_loss_weight')
BANDS=('<0.9','[0.9,1.1]','>1.1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(ok,message):
    if not ok:raise ValueError(message)
def write(p,value):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
def locked(p):
    p=Path(p);need(p.with_suffix('.sha256').read_text().strip()==sha(p),'Sidecar mismatch: '+str(p));return read(p)
def p50(d):return d['quantiles']['0.5']
def fmt(v,digits=4):return '—' if v is None else f'{v:,.{digits}f}'
def pct(v):return '—' if v is None else f'{100*v:.2f}%'

def fit_selection(group,total_rows):
    ds=group['distributions']
    ratios={a:ds['target_ratio_'+a] for a in ('C','T')}
    return {'rows':group['rows'],'row_fraction':group['rows']/total_rows,
        'date_weight_mass_fraction':ds['prior_T_over_C']['original_date_weight']['weight_mass_fraction'],
        'prior_T_over_C':ds['prior_T_over_C'],'factor_T_over_C':ds['factor_T_over_C'],
        'log_prior_T_over_C':ds['log_prior_T_over_C'],'log_factor_T_over_C':ds['log_factor_T_over_C'],
        'prior_bands':group['T_over_C_prior_bands'],'target_ratios':ratios,
        'effective_ratio_weights':{a:ds['effective_ratio_weight_'+a] for a in ('C','T')},
        'selected_raw_support_rows':{a:ds['support_raw_rows_'+a] for a in ('C','T')}}

def verify_partition(groups,rows):
    need(sum(g['rows'] for g in groups.values())==rows,'Rows do not partition complete scope')
    need(math.isclose(sum(g['date_weight_mass_fraction'] for g in groups.values()),1.,abs_tol=1e-12),'Date mass partition')
    for arm in ('C','T'):
        need(math.isclose(sum(g['target_ratios'][arm]['effective_ratio_loss_weight']['weight_mass_fraction'] for g in groups.values()),1.,abs_tol=1e-12),'Effective loss mass partition')

def compact(stage):
    n=stage['eligible_rows'];f=stage['fit'];q=stage['query']
    fg={name:{key:fit_selection(group,n) for key,group in groups.items()} for name,groups in f['groups'].items()}
    need(len(fg['support_transition'])==25,'Missing fixed transitions')
    for groups in fg.values():verify_partition(groups,n)
    for g in [f['all'],*f['groups']['age_days'].values()]:
        need(sum(r['rows'] for r in g['T_over_C_prior_bands'].values())==g['rows'],'Prior band partition')
    querygroups=q['groups'];need(len(querygroups['relative_forecast_month'])==12 and len(querygroups['support_level'])==5,'Query fixed groups')
    for groups in querygroups.values():need(sum(g['rows'] for g in groups.values())==stage['query_rows'],'Query group partition')
    for name in ('query_labels_read','raw_price_pickle_opened','other_stage_arrays_opened','row_level_derived_arrays_written','new_CSV'):
        need(stage[name] is False,'Unexpected diagnostic data boundary')
    for name in ('model_loads','model_predict_calls','prior_builds','new_native_fits','MAE_calls','bootstrap_calls'):
        need(stage[name]==0,'Unexpected model/score operation')
    for arm in ('C','T'):
        need(p50(f['all']['distributions']['target_ratio_'+arm]['unweighted'])==stage['initialization_replay'][arm]['median'],'Initial median discrepancy')
    return {'cutoff':stage['cutoff'],'query_end':stage['query_end'],'fit_rows':stage['fit_rows'],
        'eligible_rows':n,'query_rows':stage['query_rows'],'elapsed_seconds':stage['elapsed_seconds'],
        'initialization_replay':stage['initialization_replay'],'fit_all':fit_selection(f['all'],n),
        'fit_groups':fg,'query_all':q['all'],'query_groups':querygroups,
        'reference_date_ESS_status':'Not in original immutable date summaries; intentionally null, not reconstructed.',
        'all_reference_date_ESS_null':all(r['saved_reference_date_support']['reference_date_weight_ESS'] is None
            for r in stage['reference_support']['C_fold_complements']+stage['reference_support']['T_quarter_prefixes'])
            and stage['reference_support']['query_full_fit']['reference_date_weight_ESS'] is None,
        'formula_checks':stage['formula_checks']}

def render(result):
    stages=result['stages'];lines=[
        '# R101：價格先驗表示、支持與有效損失權重診斷',
        '',
        '2026-09-29。三個獨立 stage 診斷均 exit 0；本摘要只讀已保存聚合 JSON，沒有載任何 y／NPZ／模型、重建 prior、推論、MAE、bootstrap 或新 CSV。v22 仍為唯一提交，Private 未知。',
        '',
        '[固定計畫](round101_plan.md)與[主收據](../outputs/round101/results.json)界定本輪；[完整聚合摘要](../outputs/round101_summary/summary.json)保留全部固定季度、年齡、25 種支持轉移及 query 分組，沒有依結果刪群。',
        '',
        '## 結果先看這裡',
        '',
        'T 的 fit prior 通常低於 C，target ratio 因而較高；這在最近 365 日仍可觀察。惟 query 的 T/C 輸出方向依年份不同，最近窗口中位數反而低於 1，不能把分布變化當成單一校正係數或 R100 退步的已證實原因。wF 的 ESS 在前兩期減少、auxiliary 增加，亦不是所有窗口一致變得更集中。',
        '',
        'C 與 T 取相同 eligible 列；C 是原 group-OOF prior，T 是日曆季前置且排除同 held fold 的 prior。以下普通分位數採 linear；明確加權分位數採 stable inverse-CDF，不插值。ESS 只描述權重集中度，不是獨立樣本数或支持可靠率。',
        '',
        '## fit 先驗與三個預定比例帶',
        '',
        '|窗口|eligible 列|T/C 普通中位|原 w 中位|<0.9 w 質量|0.9–1.1 w 質量|>1.1 w 質量|最小／最大|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name,s in stages.items():
        a=s['fit_all'];p=a['prior_T_over_C'];b=a['prior_bands'];u=p['unweighted']
        lines.append(f"|{name}|{s['eligible_rows']:,}|{fmt(p50(u))}|{fmt(p50(p['original_date_weight']))}|{pct(b[BANDS[0]]['fraction_full_eligible_w'])}|{pct(b[BANDS[1]]['fraction_full_eligible_w'])}|{pct(b[BANDS[2]]['fraction_full_eligible_w'])}|{fmt(u['min'])}／{fmt(u['max'])}|")
    lines+=['','極低與大於 1 的比例均保留，未 clip、未據此調價。原 w 質量以同 stage 全部 eligible 的 w 總和為分母；三帶包含端點 0.9、1.1。','','## target ratio 與實際有效權重','','固定代數：mean(w·|F·r−y|)=mean(wF·|r−y/F|)，F>0。這是 float64 診斷，不聲稱與 float32 trainer 逐位相同。每列 ratio 使用實際保存的 y32／F32 轉 float64；初始化中位數與 inverseSoftplus 已對保存訓練 JSON 精確核對。','','|窗口／組|普通 ratio 中位|w 中位|wF 中位|w ESS|wF ESS|initial bias|','|---|---:|---:|---:|---:|---:|---:|']
    for name,s in stages.items():
        for arm in ('C','T'):
            r=s['fit_all']['target_ratios'][arm]
            lines.append(f"|{name}／{arm}|{fmt(p50(r[VIEWS[0]]))}|{fmt(p50(r[VIEWS[1]]))}|{fmt(p50(r[VIEWS[2]]))}|{fmt(r[VIEWS[1]]['ESS'],1)}|{fmt(r[VIEWS[2]]['ESS'],1)}|{fmt(s['initialization_replay'][arm]['inverseSoftplus'])}|")
    lines+=['','## 距 cutoff 不足 365 日的固定子群','','下表質量以全 eligible 為分母；近期子群沒有重新正規化。其他兩個年齡帶與所有已出現季度亦完整保留在 JSON。','','|窗口|近期列數／列占比|近期原 w 質量|prior T/C 普通／w 中位|ratio C 普通／w／wF 中位|ratio T 普通／w／wF 中位|C／T wF ESS|','|---|---|---:|---|---|---|---|']
    for name,s in stages.items():
        a=s['fit_groups']['age_days']['[0,365)'];r=a['target_ratios'];p=a['prior_T_over_C']
        med=lambda arm:'／'.join(fmt(p50(r[arm][k])) for k in VIEWS)
        lines.append(f"|{name}|{a['rows']:,}／{pct(a['row_fraction'])}|{pct(a['date_weight_mass_fraction'])}|{fmt(p50(p['unweighted']))}／{fmt(p50(p['original_date_weight']))}|{med('C')}|{med('T')}|{fmt(r['C'][VIEWS[2]]['ESS'],1)}／{fmt(r['T'][VIEWS[2]]['ESS'],1)}|")
    lines+=['','## query：保存預測反算，沒有 query 真價','','query 使用共用的 full-fit prior 與 F。ratio=保存 native 總價/(F64×1e6)，是浮點反算，不能稱原始 NN ratio bytes；它不是 observed target ratio。不同期／fit/query 也不是同一批房屋。','','|窗口|原 NN ratio 中位|C ratio 中位|T ratio 中位|T/C 中位|T/原 NN 中位|T/C p1／p10／p90／p99|T/C 最小／最大|','|---|---:|---:|---:|---:|---:|---|---|']
    for name,s in stages.items():
        q=s['query_all']['distributions'];d=q['native_T_over_C']['unweighted'];z=d['quantiles']
        vals=[p50(q[k]['unweighted']) for k in ('recovered_ratio_v22','recovered_ratio_C','recovered_ratio_T','native_T_over_C','native_T_over_original')]
        lines.append('|'+name+'|'+'|'.join(fmt(x) for x in vals)+'|'+'／'.join(fmt(z[k]) for k in ('0.01','0.1','0.9','0.99'))+'|'+fmt(d['min'])+'／'+fmt(d['max'])+'|')
    lines+=['','十二個相對 forecast 月份全部列出 T/C 中位（每月第 11 日起算）；並非按誤差選月。','','|月份 0–11|development|confirmation|auxiliary|','|---|---:|---:|---:|']
    for month in range(12):
        lines.append('|'+str(month)+'|'+'|'.join(fmt(p50(stages[s]['query_groups']['relative_forecast_month'][str(month)]['distributions']['native_T_over_C']['unweighted'])) for s in STAGES)+'|')
    lines+=['','|query 建案群|development T/C 中位／列數|confirmation T/C 中位／列數|auxiliary T/C 中位／列數|','|---|---|---|---|']
    for group in ('seen','new_nonmissing','missing'):
        row=[]
        for s in STAGES:
            g=stages[s]['query_groups']['project_status'][group]
            row.append(fmt(p50(g['distributions']['native_T_over_C']['unweighted']))+f"／{g['rows']:,}")
        lines.append('|'+group+'|'+'|'.join(row)+'|')
    lines+=['','seen/new 以原完整 outer-fit project membership 定義，未以 eligible 裁切集合改寫；空 missing 群保留 null。query 五層支持的全部分布在 JSON，沒有刪除極端值。','','## C→T 支持層級：完整 25 種轉移','','0=城市＋行政區＋交易類型＋建物型態，1=城市＋交易類型＋建物型態，2=交易類型＋建物型態，3=交易類型，4=global。表中為列數／原 w 質量；0 列也保留。數字是原始參考交易支持，不是獨立建案數或可靠率。','','|C→T|development 列／w 質量|confirmation 列／w 質量|auxiliary 列／w 質量|','|---|---|---|---|']
    for a in range(5):
        for b in range(5):
            key=f'{a}->{b}';vals=[]
            for s in STAGES:
                g=stages[s]['fit_groups']['support_transition'][key];vals.append(f"{g['rows']:,}／{pct(g['date_weight_mass_fraction'])}")
            lines.append(f'|{a}→{b}|'+'|'.join(vals)+'|')
    lines+=['','R60／prefix 原始 date_support 沒有保存 weight sum-of-squares 或 ESS，本次 reference ESS 如實留 null，不由 raw rows 推測；上表 ratio-loss wF 的 ESS 則由 R101 已完成統計提供。','','## 來源、耗時與限制','','|步驟|秒|','|---|---:|']
    for s in STAGES:lines.append(f"|{s} 獨立診斷程序|{result['execution_seconds'][s]:.3f}|")
    lines.append(f"|父流程（已含三期，不重複加總）|{result['main_elapsed_seconds']:.3f}|")
    lines+=['','[診斷程式](../src/round101_prior_diagnostics.py)、[統計定義](../src/round101_statistics.py)、[彙整來源](../src/summarize_round101.py)與[摘要執行收據](../outputs/round101_summary/execution.json)可追溯。原 main、逐期結果與來源 SHA 再核對；本彙整不重讀逐筆資料。',
        '',
        '本輪說明 R100 的 prior／target ratio／有效 loss 權重與輸出分布確有差異，不證明何者造成最近窗口退步，也不證明過擬合或 Private 表現。query 不用真價，因此本頁沒有新增準確度結論；R100 已知完整回測及不採用理由仍見 [R100摘要](round100_summary.md)。',
        '',
        '先驗時間截斷、支持層級、ratio 目標與權重同時改變，encoder 與 query prior 仍為 full outer-fit；不能把它當逐列完整歷史回測。後續模型需另固定單一假設／對照／窗口；本輪没有新 NN／prior build／CSV／Kaggle 操作，也不依 query 誤差找補價係數。']
    need(len(lines)<=200,'Report exceeds fixed 200-line budget')
    return '\n'.join(lines)+'\n'

def run():
    tick=time.monotonic();need(not OUT.exists() and not REPORT.exists(),'Preserve old summary outputs')
    mainpath=ROOT/'outputs/round101/results.json';execpath=ROOT/'outputs/round101/execution.json'
    main=locked(mainpath);execution=locked(execpath)
    need(main['status']=='three_isolated_diagnostics_completed' and main['stage_order']==list(STAGES)
        and execution['exit_code']==0 and execution['completed_stages']==list(STAGES),'All three stages must complete first')
    inputs={rel(p):sha(p) for p in [mainpath,execpath]};data={};timings={}
    for stage in STAGES:
        records=main['stages'][stage]
        for key in ('results','execution'):
            p=ROOT/records[key]['path'];need(sha(p)==records[key]['sha256'],'Main authority differs');inputs[rel(p)]=sha(p)
        raw=locked(ROOT/records['results']['path']);ex=locked(ROOT/records['execution']['path'])
        need(ex['exit_code']==0 and raw['status']=='descriptive_diagnostics_completed' and raw['stage']==stage,'Stage terminal mismatch')
        need(raw['protected_before']==raw['protected_after']==main['protected_sha256'],'Protected identity differs')
        data[stage]=compact(raw);timings[stage]=ex['elapsed_seconds']
    OUT.mkdir();failure=None
    try:
        result={'schema_version':1,'status':'R101_aggregate_summary_completed','stage_order':list(STAGES),
            'stages':data,'main_elapsed_seconds':execution['elapsed_seconds'],'execution_seconds':timings,
            'input_sha256':inputs,'source_sha256':sha(__file__),'only_saved_aggregate_JSON_read':True,
            'new_models':0,'prior_rebuilds':0,'new_MAE':0,'new_bootstrap':0,'new_CSV':False,
            'query_labels_read':False,'selected_submission':'v22','Private_score':None,
            'interpretation':'Descriptive distribution differences only; no causal or Private guarantee.'}
        text=render(result)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(text)
        write(OUT/'summary.json',result)
        for name,digest in inputs.items():need(sha(ROOT/name)==digest,'Input changed during summary')
        write(OUT/'receipt.json',{'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':sha(__file__),'input_sha256':inputs,'report':{'path':rel(REPORT),'sha256':sha(REPORT),'lines':len(text.splitlines())},
            'summary':{'path':rel(OUT/'summary.json'),'sha256':sha(OUT/'summary.json')},'elapsed_seconds':time.monotonic()-tick,
            'only_aggregate_JSON':True,'real_array_label_model_access':False,'new_fit_predict_score':False})
    except Exception as error:failure={'type':type(error).__name__,'message':str(error)};raise
    finally:write(OUT/'execution.json',{'exit_code':int(failure is not None),'failure':failure,'automatic_retry':False,
        'elapsed_seconds':time.monotonic()-tick,'source_sha256':sha(__file__),'real_array_label_model_access':False,'new_fit_predict_score':False})
    print(json.dumps({'status':'completed','report_lines':len(text.splitlines()),'elapsed_seconds':time.monotonic()-tick,
        'report_sha256':sha(REPORT),'summary_sha256':sha(OUT/'summary.json')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();run()
