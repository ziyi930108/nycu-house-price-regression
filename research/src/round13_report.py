"""Render fixed R13 results from saved JSON; never load arrays, fit, or rescore."""
from __future__ import annotations
import argparse, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round13'
LABELS={'development':'開發期','confirmation':'確認期','auxiliary':'輔助期'}
CONTROL='v3_control'; CANDIDATE='city_alias_geography_blend'

def need(value,message):
    if not value: raise RuntimeError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def number(x):return '—' if x is None else f'{x:,.2f}'

def delta(a,b):return None if a in (None,0) or b is None else 100*(b-a)/a

def percent(x):return '—' if x is None else f'{x:+.4f}%'

def generate():
    captured={}
    def capture(p,digest=None):
        p=Path(p).resolve()
        need(p.is_relative_to(ROOT) and p.is_file(),'Missing/unsafe source')
        got=sha(p); name=p.relative_to(ROOT).as_posix()
        need(digest is None or digest==got,'Changed source: '+name)
        need(name not in captured or captured[name]==got,'Input changed during report')
        captured[name]=got
        return got
    def locked(p):
        capture(p); receipt=p.with_suffix('.sha256'); capture(receipt)
        need(receipt.read_text(encoding='ascii').strip()==sha(p),'Receipt differs')
        return read(p)
    def mapping(items):
        for name,digest in items.items():capture(ROOT/name,digest)
    plan=locked(OUT/'candidate_plan.json')
    frozen=locked(OUT/'frozen_candidate.json')
    accepted=locked(OUT/'accepted_candidate.json')
    need(plan['round']==accepted['round']==13,'Wrong round')
    need(accepted['plan_sha256']==sha(OUT/'candidate_plan.json')
         and accepted['frozen_candidate_sha256']==sha(OUT/'frozen_candidate.json'),'Wrong finalization lineage')
    need(accepted['candidate']==CANDIDATE and accepted['new_submission_created'] is False,'Wrong release state')
    need(accepted['source_sha256']==plan['source_sha256'],'Different source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    stages=[]
    for stage,title in LABELS.items():
        spec=plan['split_contract'][stage]
        item={'stage':stage,'title':title,'dates':plan['splits'][stage],
              'fit_rows':spec['fit_rows'],'query_rows':spec['query_rows'],
              'status':accepted['stage_status'][stage],'metrics':None,'decision':None,'model':None,'independent_review':None}
        if stage in accepted['stage_result_sha256']:
            path=OUT/'evaluation'/stage/'results.json'
            capture(path,accepted['stage_result_sha256'][stage]); result=locked(path)
            need(result['stage']==stage and result['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong result')
            need(result['source_sha256']==plan['source_sha256'] and result['thresholds']==plan['thresholds']
                 and result['formulas']==plan['formulas'],'Result protocol differs')
            need(item['status']=='price_evaluated' and result['fit_rows']==item['fit_rows']
                 and result['query_rows']==item['query_rows'],'Wrong evaluated stage counts/status')
            need(set(result['metrics'])=={CONTROL,CANDIDATE},'Unexpected candidates')
            for metric in result['metrics'].values():
                for name in ('overall_mae','weighted_mae'):
                    need(math.isfinite(metric[name]) and metric[name]>=0,'Invalid MAE')
                for family in ('strata','forecast_quarters'):
                    need(sum(g['rows'] for g in metric[family].values())==item['query_rows'],'Wrong group totals')
                    for group in metric[family].values():
                        need((group['rows']==0 and group['mae'] is None) or (group['rows']>0 and group['mae'] is not None and math.isfinite(group['mae'])),'Invalid group MAE')
            mapping(result['input_artifact_sha256']); mapping(result['output_artifact_sha256'])
            model=locked(OUT/'models'/stage/'metadata.json')
            for name,digest in model['artifact_sha256'].items():capture(OUT/'models'/stage/name,digest)
            audit=locked(OUT/'audit'/(stage+'_replay.json'))
            need(audit['status']=='passed' and audit['stage']==stage and audit['result_sha256']==sha(path),'Missing matching independent replay')
            need(audit['independent_metrics']==result['metrics'] and audit['independent_decision']==result['decision'],'Audit mismatch')
            mapping(audit['source_input_sha256'])
            item.update(metrics=result['metrics'],decision=result['decision'],
                model={key:model[key] for key in ['iterations','feature_count','categorical_feature_count','sample_weight','target_formula','inverse_formula','raw_z_diagnostics','runner_elapsed_seconds']},
                independent_review={'proof_path':(OUT/'audit'/(stage+'_replay.json')).relative_to(ROOT).as_posix(),
                    'proof_sha256':sha(OUT/'audit'/(stage+'_replay.json')),'replays':audit['replays'],'raw_z_bitwise_equal':audit['saved_raw_z_bitwise_equal']})
        else:
            need(stage in accepted['unevaluated_stages'] and item['status']=='not_executed_after_development_rejection','Unsupported unexecuted reason')
            need(not(OUT/'models'/stage).exists() and not(OUT/'evaluation'/stage).exists(),'Unrecorded stage outputs')
        stages.append(item)
    completed=[s for s in stages if s['metrics'] is not None]
    expected_accept=len(completed)==3 and all(s['decision']['passed'] for s in completed)
    need(accepted['accepted'] is expected_accept,'Acceptance inconsistent')
    if not frozen['passed']:need(len(completed)==1 and completed[0]['stage']=='development','Later stage after rejection')
    capture(Path(__file__))
    for name in ['reports/round13_plan.md','reports/round13_city_alias_feasibility.md','reports/round13_official_name_evidence.md','reports/round13_alias_inventory.md','reports/round13_pipeline_review.md']:
        capture(ROOT/name)
    payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'round':13,'accepted':accepted['accepted'],
        'stopping_reason':accepted['stopping_reason'],'deployment_decision':accepted['deployment_decision'],
        'new_submission_created':False,'stages':stages,'source_input_sha256':captured,
        'historically_untouched':False,'public_score_used':False,'private_score_known':False,
        'report_only_reads_saved_JSON_and_hashes_binary_inputs':True}
    decision='三年度固定回測通過，尚待另行最終訓練與發布' if accepted['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines=['# 第十三輪結果：新增城市共享欄','',f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。','',
        '本報告只讀既有 JSON 結果與驗證收據，並核對來源雜湊；沒有重新訓練、讀取預測陣列或重算 MAE。','',
        '## 實際處理與唯一候選','',
        '- 沿用原始去除 3 列嵌入表頭、train 的 529 列完整去重及 canonical 索引；歷史完整列去重包含價格。本輪没有額外刪除高價或難預測資料。',
        '- 原 87 欄加既有 8 地理欄的值、型別與順序保持不變，末尾新增 1 個類別欄，共 96 欄／24 類別。缺值及歷史地圖規則不變，地理 state 只由 fit X 建立。',
        '- 唯一新欄是 r13_address_city_canonical：原 address_city 完整等於桃園縣才映為桃園市，其餘原樣。所有舊欄、下層行政區、reference／分組複合鍵不改；不做全域字串替換或13區映射。',
        '- 保持原始總價 y_TWD/1e6 目標與 max(z×1e6,0) 還原，原 2,999 輪、seed 42、uniform weights；不使用 eval_set 或早停。raw_z.npy 的 z 是百萬元模型輸出，不是對數。',
        '- 替換 v3 的 37.5% 地理 CatBoost 槽；另 37.5% 舊 raw、25% 舊 unit 及既有 reference trend 一次保持原樣。沒有網格搜尋、價格校正或事後調整融合比例。',
        '- 訓練不使用 query labels；共享 loader 實際會載入含 target 的快取，不能聲稱從未反序列化驗證價格。所有 MAE 都是最終未取整新臺幣價格的誤差。','',
        '## 固定年度結果','',
        'MAE 單位為新臺幣元；變化為正表示退步。未執行不是零誤差或通過。','',
        '| 窗口 | Query 日期 | Fit／query | v3 MAE | 候選 MAE | 變化 | v3 加權 MAE | 候選加權 MAE | 變化 | 結果 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in stages:
        common=f"| {s['title']} | {'～'.join(s['dates'])} | {s['fit_rows']:,}／{s['query_rows']:,} |"
        if s['metrics'] is None:lines.append(common+' — | — | — | — | — | — | 開發期未通過，未執行 |')
        else:
            a,b=s['metrics'][CONTROL],s['metrics'][CANDIDATE]
            lines.append(common+f" {number(a['overall_mae'])} | {number(b['overall_mae'])} | {percent(delta(a['overall_mae'],b['overall_mae']))} | {number(a['weighted_mae'])} | {number(b['weighted_mae'])} | {percent(delta(a['weighted_mae'],b['weighted_mae']))} | {'通過' if s['decision']['passed'] else '未通過'} |")
    for s in completed:
        lines+=['',f"## {s['title']}分組與季度",'', '| 群組 | 筆數 | v3 MAE | 候選 MAE | 變化 |','|---|---:|---:|---:|---:|']
        a,b=s['metrics'][CONTROL],s['metrics'][CANDIDATE]
        for family in ['strata','forecast_quarters']:
            for key,item in a[family].items():
                other=b[family][key]
                lines.append(f"| {key} | {item['rows']:,} | {number(item['mae'])} | {number(other['mae'])} | {percent(delta(item['mae'],other['mae']))} |")
        lines+=['','固定條件：','']
        for key,value in s['decision']['checks'].items():lines.append(f"- `{key}`：{'通過' if value else '未通過'}。")
        d=s['model']['raw_z_diagnostics']
        lines+=['',f"模型保存後，已重新載入並重播全部 {s['query_rows']:,} 列；另有獨立總價逆轉換、v3／候選融合、全部評分及門檻核對。原始 z<0 為 {d['negative_count']:,} 列，z=0 為 {d['zero_count']:,} 列；這些是診斷，未用於調參。",'']
    lines+=['## 過擬合與移轉限制','',
        '先固定唯一候選、公式、三個年度與接受門檻。開發期整體及加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期及輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權可退步至多 3%，兩期同樣有大群組限制。',
        '', '這些年度已被多輪使用，並非全新盲測。工程門檻不能消除跨輪選型偏差，也不是 Private 成績保證。名稱共享的語意正確不保證MAE改善；增加一欄並重訓也可能改變未命中別名的列，不能把結果視為僅被映射列的因果效果。',
        '', '唯一已知 Kaggle 回饋為 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。本輪不依 Public 調參，不查內政部或任何外部成交答案。',
        '', '## 完整紀錄與重現','',
        '- [名稱來源證據](round13_official_name_evidence.md)、[純X名稱盤點](round13_alias_inventory.md)。',
        '- [工作紀錄](round13_work_log.md)、[事前方法規約](round13_plan.md)、[獨立流程審查](round13_pipeline_review.md)。',
        '- [機器可讀報告](round13_results.json)、[不可覆寫計畫](../outputs/round13/candidate_plan.json)、[最終判定](../outputs/round13/accepted_candidate.json)。',
        '- 各期原始結果保存完整月份／城市描述；上述分組表沒有重設接受條件。独立核對 status=passed 只表示紀錄可重播，並非模型被採用。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。已發行檔及舊轮來源維持原樣。','']
    text='\n'.join(lines)
    dests=[ROOT/'reports/round13_report.md',ROOT/'reports/round13_results.json',ROOT/'reports/round13_results.sha256']
    need(not any(p.exists() for p in dests),'Refusing to overwrite report')
    encoded=(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    contents=[text.encode('utf-8'),encoded,(hashlib.sha256(encoded).hexdigest()+'\n').encode('ascii')]
    for name,digest in captured.items():need(sha(ROOT/name)==digest,'Source changed while reporting')
    staging=ROOT/'reports'/('.round13-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest,content in zip(dests,contents):(staging/dest.name).write_bytes(content)
    for dest in dests:need(not dest.exists(),'Concurrent report exists')
    for dest in dests:(staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status':'report_written','accepted':accepted['accepted'],'report_sha256':sha(dests[0]),'results_sha256':sha(dests[1])}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['generate']); parser.parse_args(); generate()
