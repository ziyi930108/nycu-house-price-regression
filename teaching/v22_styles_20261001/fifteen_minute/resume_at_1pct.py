from pathlib import Path
from datetime import datetime,timezone
import json,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
(HERE/'resume_at_1pct.json').write_text(json.dumps({'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'user_instruction':'你到0%停止就好','remaining_percent_at_resume':1,'credits_authorized':False,'reset_authorized':False,'resumed_work':'15-minute script preparation','stop_rule':'Check limits between stages and stop new work at observed remaining 0%. No purchase/reset/credit-use action.'},ensure_ascii=False,indent=2),encoding='utf-8')
p=ROOT/'AGENTS.md';s=p.read_text(encoding='utf-8');s=s.replace('目前剩1%且帳號有點數，但未確認點數使用已停用；在此條件確認前，不重啟可能跨界的長任務。','使用者隨後明確指示「你到0%停止就好」，因此剩1%時可接續工作，不再把確認停用點數當先決條件；各階段查額度，到0%停止，不主動操作點數或重置。');p.write_text(s,encoding='utf-8')
p=ROOT/'HANDOFF.md';s=p.read_text(encoding='utf-8');s=re.sub(r'^最後實質更新：.*$', '最後實質更新：2026-10-01（Asia/Taipei，依使用者新指示於1%接續15分鐘版，0%停止）。目前工作區：C:/NYCU_ML。',s,flags=re.M)
s=re.sub(r'^\| 當前工作 \|.*$', '| 當前工作 | 使用者明確指示「你到0%停止就好」，剩1%時已接續15分鐘改稿；不再要求先確認停用點數。新版尚未完成；[接續位置](outputs/v22_15min_20261001/README.md) |',s,flags=re.M)
s=s.replace('目前剩1%，帳號有點數，是否停用額外點數尚未確認；已詢問使用者，未重啟可能跨界的長任務。','使用者隨後明確指示「你到0%停止就好」，目前剩1%時接續工作，不再以確認停用點數為先決條件。各階段查詢，觀察到0%即停止新增工作。')
p.write_text(s,encoding='utf-8')
p=HERE/'README.md';s=p.read_text(encoding='utf-8');s=s.replace('# 15分鐘版：依額度界線中止，尚未完成','# 15分鐘版：已依最新指示接續，尚未完成',1)
s=s.replace('目前仍剩1%，工具不能設定扣點硬上限，已詢問帳號是否停用點數；確認前未重啟長任務。','使用者再明確指示「你到0%停止就好」，已在剩1%時接續改稿，不再以確認停用點數為先決條件。各階段查詢，到0%立即停止新增工作。')
s=s.replace('先等使用者明確恢復並查額度。','使用者已明確允許剩1%時接續並到0%停止。')
p.write_text(s,encoding='utf-8')
with (ROOT/'HANDOFF_LOG.md').open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-01：依明確0%停止指示恢復15分鐘改稿\n\n- 使用者再指示「你到0%停止就好」。當下used99%、remaining1%，不再以確認停用點數為工作前提。恢復被中止的十五分鐘稿代理，主線在各階段查額度；禁止點數／購買／重置要求保持。goal工具仍回傳先前paused，工具無法設active，未虛構已改狀態。\n- 更新AGENTS／HANDOFF與接續README，新增resume_at_1pct.json，保留原pause／policy歷史。此時新稿及PPTX未完成，後續結果另記。\n')
