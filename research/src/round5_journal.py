"""Append-only narrative records for optimization round five."""
from pathlib import Path
from datetime import datetime,timezone
import json
from prepare_data import ROOT,sha256
OUT=ROOT/'outputs/round5'
def record(step,category,action,reason,result,safeguards,artifacts=()):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'work_log.json'
    doc=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'round':5,'events':[]}
    if any(x['step']==step for x in doc['events']):return
    doc['events'].append({'step':step,'logged_at_utc':datetime.now(timezone.utc).isoformat(),'category':category,
       'action':action,'reason':reason,'result':result,'safeguards':safeguards,'artifacts':list(artifacts)})
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 第五輪優化工作紀錄','','持續優化房價預測；保留所有採用與淘汰的嘗試。每項嘗試都依事前門檻決定是否採用，不預先宣稱一定勝過 v2。',
       '僅使用使用者提供的本地資料；不查詢內政部成交答案；Public 不作調參目標；Private 仍未知。','']
    for x in doc['events']:
        lines += [f"## {x['step']}｜{x['category']}",'',f"記錄時間（UTC）：{x['logged_at_utc']}",'']
        for key,title in [('action','動作'),('reason','原因'),('result','結果'),('safeguards','避免過擬合與資料保護')]:
            lines += [f"**{title}：** {x[key]}",'']
        lines += [f'- [{p}](../{p})' for p in x['artifacts']]+['']
    (ROOT/'reports/round5_work_log.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    assert sha256(ROOT/'runs/round4_experiments/snapshot_manifest.json')=='5113e311de8d793181e95a342033ad7777a1c1df696c75fef62b66c2001f11b0'
    assert sha256(ROOT/'outputs/submission_v2.csv')=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
    record('001','研究與單一候選決定','檢查第四輪開發結果、歷史單價模型與兩類新假設；只選固定整建案識別欄遮罩進入第五輪。',
      '第四輪所有新方法的未見建案 MAE 都退步；需要檢驗模型是否過度依賴建案識別，而非追加大量參數搜尋。',
      '選定固定 SHA256 50% 具名建案遮罩、seed 42、2,999 輪與固定 50% raw 槽位融合。其他研究構想未訓練、未納入比較。',
      '沒有新 Public／Private 分數；既有開發年度曾反覆使用，不能稱為新的盲測。遮罩是否改善仍待回測。',
      ['outputs/round5/research_decision.json','reports/round5_research_features.md','reports/round5_research_objective.md','reports/round5_validation_plan.md'])
    record('002','合成檢查與資料邊界','檢查整案遮罩一致、未知與缺名建案處理、JSON 重載、四欄以外資料不变及不使用目標值。',
      '避免為了防止過擬合而引入新的資料洩漏或錯誤分組。',
      '8 列合成檢查通過，兩位子智能體唯讀審查未發現阻擋問題；沒有訓練或評分真實模型。',
      '每個年度重新僅由其 Fit X 建立狀態；Query 不修改字典或遮罩旗標；原 X 仍供舊模型、v2 reference 與評分使用。',
      ['outputs/round5/audit/mask_synthetic_preflight.json','src/round5_project_mask.py'])
    print('round five journal initialized; fourth-round snapshot and v2 CSV unchanged')
