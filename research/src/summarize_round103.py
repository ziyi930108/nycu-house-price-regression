"""Render R103 saved aggregate diagnostics only; never load row arrays or labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/round103_summary'; REPORT=ROOT/'reports/round103_report.md'
STAGES=('development','confirmation','auxiliary'); PAIRS=('T-v22','T-C')
QUANTITIES=('native_term','cancellation_term','base_delta','spatial_delta','clipping_term','median_delta')
ALLOWED={'outputs/round103/results.json','outputs/round103/execution.json'}|{f'outputs/round103/{s}/results.json' for s in STAGES}


def need(ok,message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x',encoding='utf8',newline='\n') as f: json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


def collect(inputs):
    def read(name,expected=None):
        need(name in ALLOWED,'Aggregate-only JSON whitelist');p=ROOT/name;h=sha(p);need(expected is None or h==expected,'Result SHA changed')
        inputs[name]=h;return json.loads(p.read_text(encoding='utf-8-sig'))
    root=read('outputs/round103/results.json');ex=read('outputs/round103/execution.json')
    need(ex['exit_code']==0 and ex['failure'] is None and root['status']=='three_window_complementarity_diagnostics_completed','Incomplete R103')
    need(root['stage_order']==list(STAGES) and root['new_models']==root['new_predictions']==root['new_bootstrap']==0 and not root['new_CSV'],'Scope differs')
    need(root['selected_submission']=='v22' and root['Private_score'] is None,'Selection identity differs')
    stages={};highlights={}
    for s in STAGES:
        rec=root['stages'][s]['results'];r=read(rec['path'],rec['sha256']);a=r['analysis']
        need(r['status']=='completed_saved_prediction_diagnostics' and r['MAE_replay']['all_30_MAE_values_within_1e_6'],'Incomplete stage replay')
        need(a['rows']==r['query_rows'] and not r['query_target_values_persisted'],'Row scope differs')
        for p in PAIRS:
            c=a['comparisons'][p];need(len(c['median_routes'])==36 and len(c['error_sign_transitions'])==9,'Missing diagnostic cells')
            for cells in (c['median_routes'],c['error_sign_transitions']):
                need(sum(x['rows'] for x in cells)==r['query_rows'],'Incomplete diagnostic partition')
                for q in QUANTITIES:
                    for measure in ('ordinary_contribution','weighted_contribution'):
                        need(abs(sum(x['quantities'][q][measure] for x in cells)-c['overall']['quantities'][q][measure])<1e-5,'Saved aggregate sum differs')
        stages[s]=r;c=a['comparisons']['T-v22']
        selected={'same_outside': [x for x in c['median_routes'] if x['control_state']==x['T_state'] and x['T_state'] in ('below','above')],
                  'worsening_blocked': [x for x in c['error_sign_transitions'] if x['spatial_change']=='worsened' and x['median_change'] in ('equal','improved')]}
        highlights[s]={k:{'rows':sum(x['rows'] for x in cells),'contributions':{q:{m:sum(x['quantities'][q][m] for x in cells) for m in ('ordinary_contribution','weighted_contribution')} for q in QUANTITIES}} for k,cells in selected.items()}
    need(highlights['development']['same_outside']['rows']==49017,'Development saved routing count differs')
    return {'status':'aggregate_summary_completed','round':103,'stages':stages,'highlights':highlights,
        'diagnostic_source_sha256':root['source_sha256'],'input_sha256':inputs,
        'diagnostic_timing':{'main_result_seconds':root['elapsed_seconds'],'main_execution_seconds':ex['elapsed_seconds'],'stage_seconds_included_in_main':{s:stages[s]['elapsed_seconds'] for s in STAGES}},
        'summary_reads_row_arrays_or_labels':False,'summary_new_MAE_or_bootstrap':False,
        'selected_submission':'v22','Private_score':None,'new_CSV':False}


def render(r):
    fmt=lambda v:'null' if v is None else f'{v:,.2f}'
    def pair(cell,q,kind='contribution'):
        v=cell['quantities'][q];return fmt(v['ordinary_'+kind])+' / '+fmt(v['weighted_'+kind])
    def cp(h,q):return fmt(h['contributions'][q]['ordinary_contribution'])+' / '+fmt(h['contributions'][q]['weighted_contribution'])
    lines=['# R103：原生改善為何未完整保留到組合','',
        '固定診斷顯示：最近期的base反轉與抵銷量下降相符；development許多N分支變動被固定median截掉。這是代數分解與事後描述，不是因果效果；median也阻擋了不少退步，不能因此主張移除它。唯一提交仍v22，Private未知，無新模型／預測／CSV。','',
        '以下金額均為元；每格「普通 / 固定seen-new加權」。差值為T減對照，負值較好。C是R102的145欄正零控制，並非R100 C。', '',
        '## 固定base的誤差抵銷','',
        '`B=.75*(.5*A+.5*P)+.25*L`；`K=.375|eA|+.375|eP|+.25|eL|−|eB|`。原生項為 **.375×原生MAE差**，不是完整原生MAE差；抵銷項為 **−ΔK**，正值表示抵銷減少。兩項相加恰為base差。','',
        '|窗口|比較|原生項 .375ΔMAE|抵銷项 −ΔK|base差|','|---|---|---:|---:|---:|']
    for s in STAGES:
        for p in PAIRS:
            c=r['stages'][s]['analysis']['comparisons'][p]['overall'];lines.append('|'+s+'|'+p+'|'+'|'.join(pair(c,q) for q in QUANTITIES[:3])+'|')
    lines+=['','## 固定median的截取影響','',
        '`lo=min(H,U), hi=max(H,U), M=clip(N,lo,hi)`；`J=|M−y|−|N−y|`。`ΔMAE_M=ΔMAE_N+ΔJ`。J差是兩個模型受截取影響的差，不是移除median的實驗結果。','',
        '|窗口|比較|spatial N差|截取項 ΔJ|完整median差|','|---|---|---:|---:|---:|']
    for s in STAGES:
        for p in PAIRS:
            c=r['stages'][s]['analysis']['comparisons'][p]['overall'];lines.append('|'+s+'|'+p+'|'+'|'.join(pair(c,q) for q in QUANTITIES[3:])+'|')
    h=r['highlights']['development']['same_outside'];lines+=['',f"development有 **{h['rows']:,}筆**同時落在相同外側（below→below或above→above）；其spatial收益對全窗為{cp(h,'spatial_delta')}，截取項{cp(h,'clipping_term')}將之抵銷，median差為0。這是淨值，不代表這些列逐筆全都改善。",'',
        '|窗口|spatial變差但median不變／改善的列數|spatial全窗貢獻|截取全窗貢獻|median全窗貢獻|','|---|---:|---:|---:|---:|']
    for s in STAGES:
        h=r['highlights'][s]['worsening_blocked'];lines.append(f"|{s}|{h['rows']:,}|{cp(h,'spatial_delta')}|{cp(h,'clipping_term')}|{cp(h,'median_delta')}|")
    lines+=['','上表保留median阻擋退步的反例；不能只報被截掉的改善。','',
        '## 最近期分群：新案及2–5千萬元','',
        '普通全窗貢獻＝群列數×群內平均÷全窗列數；加權貢獻＝群內Σ(w×差)，群內加權均值再除群權重質量。空群／零質量均值為null，貢獻為0。不同分群家族會重疊，status與價帶不能加總在一起。','',
        '|家族／群|筆數|原生項全窗貢獻|抵銷項全窗貢獻|base全窗貢獻|base群內均值|median全窗貢獻|median群內均值|','|---|---:|---:|---:|---:|---:|---:|---:|']
    groups=r['stages']['confirmation']['analysis']['comparisons']['T-v22']['groups']
    for c in groups:
        if c['family'] in ('status','price_band'):
            cells=[pair(c,q) for q in QUANTITIES[:3]]+[pair(c,'base_delta','mean'),pair(c,'median_delta'),pair(c,'median_delta','mean')]
            lines.append('|'+c['family']+'/'+c['name']+f"|{c['rows']:,}|"+'|'.join(cells)+'|')
    lines+=['','這些保存分解定位不利貢獻；新案與20–50m是重疊的兩種觀察角度，不能當兩個獨立原因，也不能依驗證真價建立推論分流。所有城市、季度、價帶、seen/new/missing與T−C完整結果都在文末JSON。','',
        '## 完整路由與改善符號表（T−v22）','',
        '六狀態：below/above為區間外；lower_equal/upper_equal僅lo<hi時；inside為嚴格內部；both_equal為lo=hi=N。36格及空格全列。表內數值均為**對全窗貢獻**，不是群內MAE。']
    for s in STAGES:
        c=r['stages'][s]['analysis']['comparisons']['T-v22'];lines+=['',f'### {s}：36路由','','|舊→T|筆數|spatial差|J差|median差|','|---|---:|---:|---:|---:|']
        for x in c['median_routes']:lines.append('|'+x['control_state']+'→'+x['T_state']+f"|{x['rows']:,}|"+'|'.join(pair(x,q) for q in QUANTITIES[3:])+'|')
        lines+=['',f'### {s}：9個誤差改善符號格','','|spatial變化→median變化|筆數|spatial差|J差|median差|','|---|---:|---:|---:|---:|']
        for x in c['error_sign_transitions']:lines.append('|'+x['spatial_change']+'→'+x['median_change']+f"|{x['rows']:,}|"+'|'.join(pair(x,q) for q in QUANTITIES[3:])+'|')
    t=r['diagnostic_timing'];lines+=['','## 範圍、時間及下一步','',
        f"R103診斷主程序{t['main_execution_seconds']:.3f}秒（結果內記{t['main_result_seconds']:.3f}秒）；各stage時間已含其中，不重複相加。此摘要另計時於收據。診斷曾載保存向量及反序列化整份raw價格表後抽query，沒有保存逐筆真價；**摘要本身只讀聚合JSON**。沒有重新模型推論、新fit或bootstrap。",
        '這支持另研究合法fit側的base-aware NN目標；但development的median截取問題仍可能存在。任何新目標都需另固定計畫與合法out-of-time老師，不能拿這批query調權重，也不承諾新版本一定更好。重用窗口、單seed及地理／全fit狀態限制保留；無因果或Private勝率宣稱。','',
        '[固定計畫](round103_plan.md) · [聚合摘要JSON](../outputs/round103_summary/summary.json) · [完整主結果](../outputs/round103/results.json) · [摘要來源](../src/summarize_round103.py)','']
    for s in STAGES:lines.append(f'- [{s}完整結果（含T−C、各arm九誤差方向格及五層MAE）](../outputs/round103/{s}/results.json)')
    lines+=['',f"主結果SHA：`{r['input_sha256']['outputs/round103/results.json']}`；摘要來源SHA：`{r['source_sha256']}`。其餘來源與結果SHA保存在摘要及收據。",'']
    return '\n'.join(lines)


def run():
    need(not OUT.exists() and not REPORT.exists(),'Preserve existing outputs');OUT.mkdir();tick=time.monotonic();inputs={};code=1;failure=None;source=sha(__file__)
    try:
        r=collect(inputs);r.update(source_sha256=source,created_at_utc=datetime.now(timezone.utc).isoformat());write(OUT/'summary.json',r)
        with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(render(r))
        need(source==sha(__file__) and all(sha(ROOT/p)==h for p,h in inputs.items()),'Source/result changed')
        artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in (REPORT,OUT/'summary.json')}
        write(OUT/'receipt.json',{'status':'completed','input_sha256':inputs,'source_sha256':source,'artifact_sha256':artifacts,'row_arrays_or_labels_read':False,'new_MAE_or_bootstrap':False,'elapsed_seconds':time.monotonic()-tick});code=0
        print(json.dumps({'status':'completed','artifacts':artifacts,'elapsed_seconds':time.monotonic()-tick}))
    except Exception:failure=traceback.format_exc();raise
    finally:write(OUT/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':source,'input_sha256':inputs,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False,'row_arrays_or_labels_read':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('--run required')
    run()
