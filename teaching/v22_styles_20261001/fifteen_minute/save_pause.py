"""Save actual interrupted state; do not build or train anything."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
files={}
for pattern in ('*.py','*.json','build/*.mjs','content/*.json'):
 for p in HERE.glob(pattern):
  if p.name!='pause_status.json':files[p.relative_to(HERE).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
has_proposal=(HERE/'content/proposed_scripts.json').exists()
status={'saved_at_utc':datetime.now(timezone.utc).isoformat(),'reason':'User explicitly required stopping when weekly remaining quota reaches 1%.','used_percent':99,'remaining_percent':1,'active_child':'fifteen_minute_script','child_status':'interrupted','proposal_exists':has_proposal,'approved_script_exists':(HERE/'content/approved_scripts.json').exists(),'new_decks_built':False,'artifact_authoring_marker_executed':False,'new_html_built':False,'new_github_publication':False,'files':files}
(HERE/'pause_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
readme=f'''# 15分鐘版：依額度界線中止，尚未完成

2026-10-01使用者回覆「15分鐘上下，不要超過太多或是太少」。主講目標改成約15分鐘，仍保留至少12分鐘、完整流程和每步理由，不含附錄與問答。原4,981字／規劃17:20講稿保存，不將它當成已符合新時長。

額度工具回報used99%、remaining1%，依使用者要求停止新增工作。子智能體fifteen_minute_script已中止，其餘子工作已完成；停止時未見Python程序。

## 實際完成

- 保存[user_timing_request.json](user_timing_request.json)。
- 準備[setup_sources.py](setup_sources.py)、其生成的[prepare_content.py](prepare_content.py)、兩份build MJS、[preserve_package.py](preserve_package.py)、[run_build.py](run_build.py)，建立runtime node_modules junction。這些是待執行來源，尚未從新稿生成narrative／PPTX。
- 原文提案存在：{has_proposal}；核准稿存在：{status['approved_script_exists']}。有檔不等於完成審核，真實清單與SHA見[pause_status.json](pause_status.json)。
- Presentations技能及相關模板／實作／finalization已讀，尚未執行本輪authoring marker，也未呼叫Artifact製作。不得把prepared來源當完成成品。

## 接續順序

先等使用者明確恢復並查額度。讀HANDOFF與此檔，重新確認所有來源及程序，保留原件。

1. 完成20頁約4,050–4,200純漢字提案，再逐頁核對方法、理由與限制。以275字／分約15分鐘只是配速估計，必須本人計時。不要只改seconds欄位。
2. 核准稿另存content/approved_scripts.json。檢查prepare_content.py的新meta，原r2數字、主講/附錄、OOF與Private界線保持。其4050–4250檢查上界是待審設定，不自動代表所有長度均合格。
3. 調用Presentations skill，第一次authoring前只執行一次edit marker、expected3。再prepare_content與run_build，保留所有原檔並依finalizer核驗。原流程只把Artifact寫的notes1–20移入原OPC，不改圖表工作表及畫面；仍須核對23頁notes、其餘parts及69張新舊PNG。
4. 更新新版本的排練HTML資料、配速與目標提示，不能繼續使用4,981字舊稿。同時保存舊排練記錄及原HTML。
5. 再整理3新版PPTX、3不變PDF、講稿/問答/口述提示、答辯卡PDF/文字及新版HTML的單一上台包。先前單一包尚未製作。projection/README仍連舊r2及16:45，接續時修正導覽，歷史收據保持。
6. 更新GitHub、HANDOFF／LOG與exporter，執行新交接ZIP檢查。不要回寫immutable研究決策。

最後已完成且已發布的是17:20口語稿、三款簡報、四頁答辯卡與20頁離線排練工具，GitHub main c27f92318fec556bc54b5c7e4ce1a97fd7e04d48。本轮未發布新GitHub成品。v22唯一提交，Public1,998,568.5706元、Private未知，R106停止，沒有模型訓練、新CSV或Kaggle動作。
'''
(HERE/'README.md').write_text(readme,encoding='utf-8')
p=ROOT/'HANDOFF.md';s=p.read_text(encoding='utf-8')
s=s.replace('（Asia/Taipei，離線排練工具與發布完成）','（Asia/Taipei，剩餘1%依使用者要求暫停；15分鐘版待完成）',1)
s=re.sub(r'^\| 當前工作 \|.*$', '| 當前工作 | **額度remaining1%，依使用者要求停止並暫停。** 新要求主講約15分鐘，僅準備改稿來源，未完成新稿／PPTX／HTML；[接續位置](outputs/v22_15min_20261001/README.md) |',s,flags=re.M)
s=re.sub(r'^\| 最新非模型產物 \|.*$', '| 最新非模型產物 | 已發布的三款口語PPTX仍為4,981字／規劃17:20，另有答辯卡與排練HTML。15分鐘版尚未製作，不能把舊版配速當新要求已達成 |',s,flags=re.M)
s=s.replace('答辯卡發布階段工具查得used98%、remaining2%，只是查詢當下的帳號共用狀態。','最新工具查得used99%、remaining1%，已觸發使用者停止界線。子工作中止，保存接續記錄後暫停goal；未重置或購買。')
needle='6. 最新上台簡報以選定的v22為唯一模型，面向零基礎，觀眾不需了解內部版本沿革。'
assert needle in s;s=s.replace(needle,needle+' **2026-10-01使用者最新補充：「15分鐘上下，不要超過太多或是太少」。約15分鐘成為目前主講目標，新精簡版尚未完成。**',1)
p.write_text(s,encoding='utf-8')
p=ROOT/'AGENTS.md';s=p.read_text(encoding='utf-8')
if '15分鐘上下，不要超過太多或是太少' not in s:
 s+='\n2026-10-01最新時間補充：「15分鐘上下，不要超過太多或是太少」。主講目標約15分鐘，仍保留至少12分鐘與完整流程理由、不依賴問答補時。原4,981字／17:20稿保持為舊版，新精簡稿需另存且實際改寫；不要只改秒數或聲稱本人已排練。額度1%已觸發暫停，接續需使用者明確恢復。\n';p.write_text(s,encoding='utf-8')
with (ROOT/'HANDOFF_LOG.md').open('a',encoding='utf-8') as f:f.write(f'''

## 2026-10-01：使用者指定約15分鐘，額度1%中止並交接

- 前轮完成離線排練工具、GitHub c27f92318fec556bc54b5c7e4ce1a97fd7e04d48及5,911檔交接包，屬實質進度。本輪先審查下載使用：代理核12個必要原成品存在，指出projection README仍連舊r2／16:45。單一上台包尚未建置，也未修改GitHub。
- 使用者回覆「15分鐘上下，不要超過太多或是太少」，目標由原17:20改約15分鐘。新證據存outputs/v22_15min_20261001/user_timing_request.json。委派20頁精簡到約4050–4200漢字，保留方法理由與必要限制。代理在完成前被中止，proposal存在={has_proposal}，核准稿存在={status['approved_script_exists']}。
- 實際只完成setup_sources及生成的prepare_content／build MJS／preserve_package／run_build來源、目錄與node_modules junction；未執行本輪authoring marker、未製作新PPTX／narrative／HTML、未訓練、未發布。不要將準備完成當作15分鐘版完成；來源仍需審查及執行验证。Presentations技能相關資料已讀，來源A13畫面已看。
- 最新額度used99%、remaining1%，依使用者明確界線立刻停止新增工作，interrupt十五分鐘改稿子智能體，其他子智能體均完成；唯讀確認無Python執行程序。保存README、pause_status、實際檔案SHA、HANDOFF／AGENTS／本日誌及exporter後匯出文件交接包，再將既有goal設paused。未重置／購買／自動化。
- 待使用者恢復後：完成並核准15分鐘稿，更新3款PPTX notes與排練HTML，核23頁notes與69PNG，整理新上台包並更新GitHub。原v22／Public1,998,568.5706／未知Private／R106停止保持。最新有效成品仍是17:20口語版，15分鐘與本人時長皆未驗證。交接ZIP身分以outputs/handoff/latest.json為準，不稱完整重訓備份。
''')
print(json.dumps(status,ensure_ascii=False))
