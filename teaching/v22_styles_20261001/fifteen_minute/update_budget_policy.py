from pathlib import Path
from datetime import datetime,timezone
import json,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
policy={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'user_instruction':'可以使用到0%，但是不可以消耗到點數','supersedes':'stop at 1%','stop_remaining_percent':0,'credits_authorized':False,'purchases_or_resets_authorized':False,'last_observed_used_percent':99,'last_observed_remaining_percent':1,'available_credits_reported':True,'account_credit_usage_disabled_verified':False,'agent_can_set_billing_hard_cap':False,'long_task_resumed':False,'pending_question':'是否已在帳號設定停用額外點數使用？','official_reference':'https://learn.chatgpt.com/docs/pricing'}
(HERE/'budget_policy_0pct_no_credits.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
p=ROOT/'AGENTS.md';s=p.read_text(encoding='utf-8')
s=re.sub(r'2026-10-01使用者新增額度界線：.*?不要為監看另建自動化。', '2026-10-01使用者最新額度界線：「可以使用到0%，但是不可以消耗到點數」。此要求取代先前1%停止門檻。工作各階段前後以get_usage_limits查詢即時Codex額度（remaining=100−usedPercent）；到0%停止新增工作、結束所派子工作並依明確授權暫停goal。不消耗點數、不購買、不重置、不建額度監看自動化。工具為唯讀快照且可能延遲，不能承諾精準到0%或宣稱已設扣點硬上限。目前剩1%且帳號有點數，但未確認點數使用已停用；在此條件確認前，不重啟可能跨界的長任務。', s, count=1,flags=re.S)
s=s.replace('額度1%已觸發暫停，接續需使用者明確恢復。','先前於1%暫停是歷史狀態；最新可用到0%但禁止消耗點數，須遵守上方最新界線。')
p.write_text(s,encoding='utf-8')
p=ROOT/'HANDOFF.md';s=p.read_text(encoding='utf-8')
s=re.sub(r'^最後實質更新：.*$', '最後實質更新：2026-10-01（Asia/Taipei，額度門檻改為0%，禁止扣點；尚未確認帳號停用點數）。目前工作區：C:/NYCU_ML。',s,flags=re.M)
s=re.sub(r'^\| 當前工作 \|.*$', '| 當前工作 | 最新允許方案額度用到0%，禁止消耗點數。目前剩1%，點數使用是否停用未確認，未重啟長任務。15分鐘版仍僅有製作準備；[接續位置](outputs/v22_15min_20261001/README.md) |',s,flags=re.M)
s=re.sub(r'^8\. 2026-10-01使用者明確要求剩餘額度1%時停止。.*$', '8. 使用者最新要求「可以使用到0%，但是不可以消耗到點數」，取代先前1%停止門檻。階段前後查額度，0%停止，不購買、不重置、不用點數。唯讀工具不能設定扣點硬上限，快照也可能延遲。目前剩1%，帳號有點數，是否停用額外點數尚未確認；已詢問使用者，未重啟可能跨界的長任務。先前暫停與pause_status.json保留為歷史。[最新界線紀錄](outputs/v22_15min_20261001/budget_policy_0pct_no_credits.json)。',s,flags=re.M)
p.write_text(s,encoding='utf-8')
p=HERE/'README.md';s=p.read_text(encoding='utf-8');i=s.index('\n');s=s[:i+1]+'\n**後續更新：使用者已允許方案額度用到0%，但禁止使用點數。** 目前仍剩1%，工具不能設定扣點硬上限，已詢問帳號是否停用點數；確認前未重啟長任務。下方1%中止為當時歷史，最新規則見[額度紀錄](budget_policy_0pct_no_credits.json)。\n'+s[i+1:];p.write_text(s,encoding='utf-8')
with (ROOT/'HANDOFF_LOG.md').open('a',encoding='utf-8') as f:f.write('''

## 2026-10-01：額度界線改為0%，禁止消耗點數

- 使用者最新明確要求「可以使用到0%，但是不可以消耗到點數」，取代1%停止門檻。更新AGENTS／HANDOFF與15分鐘版接續README，原pause_status與原日誌保留為歷史。
- 實查每週used99%、remaining1%，ordinaryUsageAllowed為true且帳號有點數。get_usage_limits只有讀取，沒有可用設定扣點硬上限的工具；未更動帳號、購買、重置或消耗點數的設定。官方https://learn.chatgpt.com/docs/pricing說明額度耗盡後可用credits繼續，但此帳號是否已停用點數無法由快照證明。
- 已以非同步文字問題詢問是否已在帳號設定停用額外點數；確認前未重啟可能跨界的長任務、子智能體、PPTX製作或模型研究。15分鐘版仍未完成，原成品保持。這是预算界線更新與狀態保存，不是15分鐘版進度。更新後刷新交接ZIP，實際身分以outputs/handoff/latest.json為準。
''')
print(json.dumps({'policy_updated':True,'long_task_resumed':False}))
