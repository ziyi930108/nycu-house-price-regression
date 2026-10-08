"""Preserve prior handoff and record the user's explicit research resumption."""
from pathlib import Path
import json
import shutil

root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent
before=out/'before';before.mkdir(exist_ok=False)
for name in ('HANDOFF.md','HANDOFF_LOG.md','AGENTS.md'):
    shutil.copy2(root/name,before/name)
feedback={'date_local':'2026-10-08','timezone':'Asia/Taipei',
 'user_request':'我要你繼續優化模型，我還想要繼續嘗試繳交',
 'research_resumed':True,'prior_R106_stop_superseded_for_new_experiments':True,
 'v22_preserved':True,'new_candidates_allowed':True,'automatic_Kaggle_upload_or_selection':False,
 'private_score':None,'experiments':['R107 weighted SmoothL1 prior NN','R108 single CTR unit CatBoost'],
 'plan':'reports/round107_108_plan.md','stop_at_remaining_percent':1,
 'credits_purchase_reset_authorized':False}
(out/'user_request.json').write_text(json.dumps(feedback,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=root/'HANDOFF.md';text=p.read_text(encoding='utf-8-sig')
text=text.replace('最後實質更新：2026-10-03；狀態核對：2026-10-03','最後實質更新：2026-10-08；狀態核對：2026-10-08',1)
lines=text.splitlines()
for i,line in enumerate(lines):
    if line.startswith('| 唯一提交版本 |'):
        lines[i]='| 已選定提交基準 | **v22保留**；2026-10-08使用者重新授權模型研究及新嘗試提交，尚未選定新最終版本。[原選定](outputs/v22_submission_focus_20260928/user_selection.json)、[恢復要求](outputs/research_resume_20261008/user_request.json)，Private未知 |'
    elif line.startswith('| 目前工作 |'):
        lines[i]='| 目前工作 | 2026-10-08恢復模型研究，事前固定R107日期加權SmoothL1 prior NN與R108單類別CTR單價CatBoost；正在前置審查，尚未有新分數或CSV。原v22簡報、旁白與197aaa2程式交付保留；無舊背景訓練 |'
    elif line.startswith('| 研究狀態 |'):
        lines[i]='| 研究狀態 | R106已完成且未採用；其停止界線被2026-10-08最新模型研究要求撤銷。新輪計畫見[R107／R108](reports/round107_108_plan.md)，兩個候選互相獨立，單GPU序列執行 |'
text='\n'.join(lines)+'\n';p.write_text(text,encoding='utf-8')
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:
    f.write('\n2026-10-08使用者明確要求「我要你繼續優化模型，我還想要繼續嘗試繳交」，重新授權R106之後的模型研究與新試交候選；取代停止研究界線。保留v22與原簡報，研究候選不自動取代最終選定，不代為Kaggle上傳／勾選；維持資料來源禁令、過擬合／時間隔離、保存與1%額度界線。\n')
with (root/'HANDOFF_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 2026-10-08：使用者重新授權模型研究與試交\n\n'
      '- 最新要求明確恢復R106後模型研究，允許新嘗試提交；v22原選定與簡報保留，新最終選擇未定，Private未知。原HANDOFF／日誌／AGENTS先保存before，最新要求另存outputs/research_resume_20261008/user_request.json。\n'
      '- 實讀交接、近輪報告、固定管線與來源，程序列舉未見python／ffmpeg／git活工作；兩個唯讀子智能體盤點已有反證，避免重複停車比例、Huber、時間prior與全域stacking。\n'
      '- 事前固定R107日期weighted prior NN×beta=.1 SmoothL1，R108 R79 max_ctr_complexity2→1；各3fit、全部三窗sealed後才評分。探索試交條件與較嚴強推薦條件分開事先列明，未按新結果放寬。計畫reports/round107_108_plan.md。\n'
      '- 本筆寫入時尚未真fit／評分／產新CSV。待前置獨立審查與來源凍結後由主線協調單GPU依序跑；不使用外部成交答案、不更動Kaggle或舊不可變證據。初始額度used8／剩92，無本輪reset／購買／付費API。\n')
print('resumption recorded; original handoff preserved')
