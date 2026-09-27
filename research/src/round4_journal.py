"""Append-only narrative records for optimization round four."""
from pathlib import Path
from datetime import datetime,timezone
import json
from prepare_data import ROOT,sha256
OUT=ROOT/'outputs/round4'
def record(step,category,action,reason,result,safeguards,artifacts=()):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'work_log.json'
    doc=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'round':4,'events':[]}
    if any(x['step']==step for x in doc['events']):return
    doc['events'].append({'step':step,'logged_at_utc':datetime.now(timezone.utc).isoformat(),'category':category,
       'action':action,'reason':reason,'result':result,'safeguards':safeguards,'artifacts':list(artifacts)})
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 第四輪優化工作紀錄','','持續優化房價預測；保留所有採用與淘汰的嘗試。每項嘗試都依事前門檻決定是否採用，不預先宣稱一定勝過 v2。',
       '僅使用使用者提供的本地資料；不查詢內政部成交答案；Public 不作調參目標；Private 仍未知。','']
    for x in doc['events']:
        lines += [f"## {x['step']}｜{x['category']}",'',f"記錄時間（UTC）：{x['logged_at_utc']}",'']
        for key,title in [('action','動作'),('reason','原因'),('result','結果'),('safeguards','避免過擬合與資料保護')]:
            lines += [f"**{title}：** {x[key]}",'']
        lines += [f'- [{p}](../{p})' for p in x['artifacts']]+['']
    (ROOT/'reports/round4_work_log.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    previous=ROOT/'runs/round3_experiments/snapshot_manifest.json'
    assert sha256(previous)=='96cee2c7977d9317f70e5e64b41cbf057db8fad529019a2e76fb37d1606a202f'
    assert sha256(ROOT/'outputs/submission_v2.csv')=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
    record('001','持續優化與基準確認','開始第四輪，重新讀取第三輪報告、目前推薦版本及已封存證據。',
      '使用者要求持續改進，且需保留防過擬合與完整資料處理紀錄。',
      '第三輪三個新候選都未達預定門檻；仍以 v2 為對照。先平行稽核地址特徵、模型嘗試歷史及另一個更早年度的驗證切分。',
      '不修改既有 v1／v2 模型、CSV、第三輪封存或原始資料；不把重複使用的歷史回測稱為全新測試。',
      ['reports/round3_results.json','runs/round3_experiments/snapshot_manifest.json','outputs/submission_v2.csv'])
    print('round four journal initialized; prior snapshot and submission hashes unchanged')
