"""Finish only the four R100 navigation documents and preserve narrow QA receipts."""
from pathlib import Path
import hashlib,json,re,urllib.parse
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'deliverables/v22_github'
SNAP=REPO/'evidence/outputs/research_snapshot_r100_20260929'
FINAL=SNAP/'final'
NAV=['README.md','HANDOFF.md','research/README.md','docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

NEW_PAGE='''# R100：時間前置且排除同建案折的價格先驗

2026-09-29。六次 NN 訓練、三期完整預測、獨立評分與保存結果稽核均已完成。**T 未通過原固定研究門檻，不採用；v22 仍是唯一提交，Private 未知。** 使用者回報的 v22 Public MAE 為 1,998,568.5706 元，以下是本機年度回測，不能混用。

本頁於發布前核驗，基底為 c34842980f8abc4bcc15dc9d5940fe2d50ac118f；當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存發布收據。[上一輪 R99](RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)及其失敗反證保持。

## 為什麼做這個比較

原先的價格先驗採建案分組五折 OOF：每列的先驗不使用同一 held fold，但可使用外層 fit 內較晚日期。這**不是外層 query 洩漏**。本輪的單一假設是：若 fit 列的先驗也只用其日曆季以前交易，能否改善較晚房價的回歸？[固定計畫](../evidence/reports/round100_plan.md)在訓練前記錄兩組、日期邊界、固定配置與採用限制。

對每一日曆季，參考資料須同時滿足有效日期、日期嚴格早於季首、排除該列原建案 held fold。五個 held fold 各至少有 30 筆參考時，整季才納入 loss。無效日期不納入；本次三期實際均為零筆。沿用原完整 fit 的五折群身分，不在裁切後重分折。

|訓練組|價格先驗|其他條件|
|---|---|---|
|C：機制對照|原 R60 group-OOF factor，僅取相同 eligible 列|與 T 相同 num／cat／y／原 float32 日期權重，不重新正規化|
|T：研究候選|同一 loss 列改為日曆季前置且排除同折的 factor|原 R63 trainer、seed 42、30 epochs、batch 2048 及架構保持|

development／confirmation／auxiliary 的 loss 列分別為 240,260／321,494／134,749，每期比原 fit 少 56 筆 warmup。採相同樣本對照，避免把刪除早期樣本與先驗時間方向混在一起。兩組仍使用原完整 outer-fit encoder、完整 fit 的 query prior 與同窗 reference／spatial；**並非每一個輸入都改成逐列當時可知**。沒有重新搜尋季長、比例或 seed。

每組只替換先驗倍率 NN；保留原面積 NN、LightGBM 與另外兩個完整分支。新 NN 分支 base 為 .75×(.5×areaNN＋.5×priorNN)＋.25×LGB，再套原 reference／spatial，最後與 H／U 取 median。三期共六個 NN，0 個新樹模型。全部 query 預測及模型重載封存後，獨立 scorer 才讀 query 真價。[來源與實測摘要](../evidence/reports/round100_summary.md)保留完整細節。

## 完整模型結果

單位為元；差值越負越好。C 是機制對照，不能在看完結果後直接升格為另一個提交候選。

|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 加權差|
|---|---:|---:|---:|---:|---:|---:|
|development|1,407,387.07|1,405,993.40|1,402,289.26|−5,097.81|−3,704.14|−4,948.79|
|confirmation|1,383,228.41|1,382,132.85|1,393,291.76|+10,063.35|+11,158.91|+11,056.78|
|auxiliary|2,309,352.36|2,302,648.68|2,297,675.88|−11,676.48|−4,972.80|−10,303.47|

最近 confirmation 整體退步 0.7275%，未達預先要求至少改善 0.5%。三期 overall／weighted 同時改善的條件亦未過；auxiliary 的嘉義市 731 筆退步 129,464.62 元（6.1285%），觸及固定大群 >5% 限制。T 相對 v24 的三期差為 −5,051.46／+20,095.84／+22,266.74 元，完整反證不刪除。[正式評分報告](../evidence/outputs/round100/evaluation/report.md)與[聚合結果 JSON](../evidence/outputs/round100/evaluation/results.json)為數值依據。

兩個主要比較 T−v22、T−C，各做建案與連續三個月 block 的 2,000 次配對重抽樣，固定 seed。共 24 個區間：4 個全負、14 個跨零、6 個全正。T−v22 在 auxiliary 的四個區間全負；confirmation 的兩個月份區間全正。T−C 在 confirmation 四個區間全正；其餘八個跨零。這支持收益隨年份改變，不能轉述成 Private 勝率，也不能說所有群都無效。

## 有利結果與反證一起看

- T 的原生 prior NN 三期都比原 NN 差，MAE 增加約 157,352／340,775／108,849 元；完整 median 卻在兩期改善。層間差不能直接相加成因果貢獻。
- 新建案 T−v22 三期為 −10,743.61／+16,682.97／−12,556.13 元；最近窗仍退。已見建案則為 +5,081.61／+1,318.26／−6,404.29 元。
- confirmation 的 2,000–5,000 萬價帶對總差貢獻 +8,769.34 元；1 億以上只貢獻 +674.95 元，退步並非少數豪宅造成。
- auxiliary 的 1 億以上貢獻 −2,037.88 元，其餘價格合計 −9,638.60 元；改善也不限於極端高價。
- C 比 v22 的三期點估計都略低，仍只是同樣本機制對照。不能由此斷言少 56 筆本身有效，也不能事後把 C 當作已通過完整採用審查的新版本。

以上為完成後描述，不依 query 真價挑選補價群。年度窗口反覆使用有選型偏差，沒有新的 test 真價或新的盲測。

## 實測成本與評分前工程修正

|步驟|實測秒數|計時範圍|
|---|---:|---|
|六次 NN fit|118.203|已包含於 main|
|三期 temporal prior 生成|160.876|已包含於 main|
|main|391.938|完整前置、六 NN、重載與封存；不能再加上前兩列|
|独立 scorer v2|68.985|main 完成後的來源核對、評分及配對區間|
|保存結果稽核|77.078|內部核驗；外層程序另計 80.109 秒|
|聚合摘要|0.172|只整理已保存 JSON，沒有新評分或圖表|

原 scorer 的前置草稿曾把純 X 稽核順序要求成模型順序，真評分前修正並保留 36→38 項成功合成檢查與來源。模型真訓練期間又發現純 X 稽核計畫 SHA 與科學計畫 SHA 應分別驗證：原碼保持，另建 scorer v2。新契約沿純 X result→request→audit plan 與 scientific plan 核對；31 個其他函數 AST 保持，MAE、bootstrap、gate、公式均未改。

[v2 契約](../evidence/reports/round100_evaluation_contract_v2.md)、[窄差異](../evidence/outputs/round100_evaluator_v2_revision/engineering_diff.patch)、[真實 metadata 檢查](../evidence/outputs/round100_evaluator_v2_revision/actual_metadata_check.json)與[修訂收據](../evidence/outputs/round100_evaluator_v2_revision/final_receipt.json)完整保留。v2 42 項人工檢查成功，真 metadata 前置核對通過後才評分。這是評分前工程契約修正，沒有真實模型失敗／重試，也沒有利用 query 結果改科學計畫。

[完成稽核](../evidence/reports/round100_independent_review.md)核對 1,827 個唯一 SHA、9 個保護檔、705 個完整 prefix state 與 24 個既存區間；不重新載標籤、求 prior、評分或 bootstrap。稽核者亦參與純 X 政策／稽核且審閱 helper；helper、driver、scorer 由其他作者負責。這是跨作者保存證據核驗，不稱完全獨立重作。

## 程式與交接界線

|入口|內容|
|---|---|
|[契約](../research/src/round100_contract.py)／[計畫](../evidence/reports/round100_plan.md)|固定兩組、資料讀取與評分參數|
|[fit 標籤擷取](../research/src/round100_prepare_fit_labels.py)／[純 X 稽核](../research/src/round100_temporal_prior_x_audit.py)|canonical／raw mapping、日期支持、只輸出各期 fit 標籤|
|[prior helper](../research/src/round100_temporal_prior.py)／[主 driver](../research/src/round100_temporal_prior_experiment.py)|季首與 held-fold 排除、C／T 串行、全向量封存|
|[原 scorer](../research/src/round100_temporal_prior_evaluate.py)／[scorer v2](../research/src/round100_temporal_prior_evaluate_v2.py)|原稿與修正版分開保留，實際評分使用 v2|
|[保存結果稽核](../research/src/round100_independent_review.py)／[摘要](../research/src/summarize_round100.py)|既存證據核對與聚合整理|
|[前置複製清單](../evidence/outputs/research_snapshot_r100_20260929/preparation/copy_manifest.json)／[完成複製清單](../evidence/outputs/research_snapshot_r100_20260929/final/copy_manifest.json)|來源原 bytes 與 SHA，不回寫歷史收據|

本包不含交易表、fit 標籤、逐筆向量、NN 模型、NPZ、完整 prefix state 或含各 prefix 價格的 temporal_prior_evidence。705 個狀態及價格證據僅在原工作區，身分由 stage seal 的 artifact SHA 保留。因此原來源報告部分相對連結仍依賴原工作區，文件快照不等於完整可重訓備份或乾淨環境重現。

[有限下一方向](../evidence/reports/round100_next_direction_inventory.md)只提出先另固定 fit 側的 old/new prior 比值、支持層級、ratio／bias 分布診斷；尚未執行、沒有新模型計畫，也不再把 temporal prior 當成未做過的方法。不因本頁自行掃季長、seed 或融合比例。正式 v22 REPORT／PDF／PPTX、379 支歷史來源與既有提交保持。
'''

def edit_navigation():
 p=REPO/'README.md';s=p.read_text(encoding='utf8')
 start=s.index('## 最新時間前推組合研究');end=s.index('## 簡報與專案更新',start)
 s=s[:start]+'''## 最新時間前置先驗研究

[R100 時間前置 prior](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)已完成六次 NN、三期完整模型與獨立評分。使用相同 loss 樣本比較原 group-OOF 先驗 C 與「季首以前且排除同建案折」的先驗 T。T 相對 v22 的 development／confirmation／auxiliary MAE 差為 **−5,097.81／+10,063.35／−11,676.48 元**；最近窗退 0.7275%，auxiliary 嘉義市亦觸及固定群退步限制，研究門檻未過。C 是機制對照，沒有事後升格成另一個提交候選。

24 個 T−v22／T−C 配對區間為 4 負、14 跨零、6 正；不能轉述成 Private 勝率。六次 NN fit 共 118.203 秒，已包含於 main 391.938 秒；scorer 68.985 秒，保存結果稽核內部 77.078 秒（外層 80.109 秒）。評分前的兩份計畫 SHA 契約修正另存 scorer v2，沒有重試模型或更改科學評估參數。唯一提交 v22、未知 Private、正式報告與簡報保持。本頁於發布前核驗，當前發布身分以 Git main 歷史及 SOURCE_MANIFEST 為準。

此前 [R99 四欄組合](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)、[R98 三欄組合](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)都未過採用门檻。[R97 錯開切點](docs/RESEARCH_UPDATES_20260929_OFFSET.md)重建 5 樹＋2 NN，新窗 v22 較低而原三年 v24 較低，反證保持。[Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)、[R93](docs/RESEARCH_UPDATES_20260929_R93.md)、[R89／R91／R92](docs/RESEARCH_UPDATES_20260929.md)、[R85–R88](docs/RESEARCH_UPDATES_20260928.md)亦保留。這些窗口反覆使用與重疊，不是新盲測。

'''+s[end:];p.write_text(s,encoding='utf8',newline='\n')

 p=REPO/'HANDOFF.md';s=p.read_text(encoding='utf8').replace('R85–R99','R85–R100')
 s=re.sub(r'^\| 最新研究 \|.*$', '| 最新研究 | R100 時間前置 prior：6 NN、三期完整評分；T−v22 為 −5,097.81／+10,063.35／−11,676.48 元，未過門檻，無新 CSV |',s,flags=re.M)
 s=re.sub(r'^\| 本批發布 \|.*$', '| 本批發布 | 本頁於發布前核驗，base c3484298…c118f；當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存收據 |',s,flags=re.M)
 s=re.sub(r'^\| 後續界線 \|.*$', '| 後續界線 | R100 六 NN、評分與保存結果稽核完成；下一方向僅提出 fit 側診斷，未執行、未另立新模型計畫 |',s,flags=re.M)
 s=s.replace('- [最新 R99 含median四欄組合]', '- [最新 R100 時間前置 prior](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)：兩組相同 loss 列、三期完整模型、24區間、工程修正與依賴界線。\n- [前批 R99 含median四欄組合]')
 start=s.index('[R99完成頁]');end=s.index('使用者已選定v22，最新紀錄',start)
 s=s[:start]+'''[R100 完成頁](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)是最新入口。原 group-OOF 並非外層 query 洩漏；本輪固定比較相同 eligible loss 列的原先驗 C 與日曆季前置、同 held fold 排除的先驗 T。保留原 R63 trainer、日期權重原值、完整 outer-fit encoder／query prior 與同窗 R/S。每期少 56 warmup 列，沒有重新分折。三期六個 NN 完成封存／重載後才由 scorer 讀 query 真價。

T 相對 v22，development／confirmation／auxiliary 整體差為 −5,097.81／+10,063.35／−11,676.48 元，加權方向相同。最近窗退 0.7275%；auxiliary 嘉義市 731 筆退 6.1285%，原固定門檻未過。相對同樣本 C 為 −3,704.14／+11,158.91／−4,972.80 元。C 三期點估計較原 v22 低，仍只是機制對照，不能事後改選。T 原生 prior NN 三期均退，完整 median 兩期改善，層差不能相加成因果。

24 個主要配對區間中 4 負、14 跨零、6 正；T−v22 的 auxiliary 四個區間全負，而 T−C 的 confirmation 四個全正。窗口多次重用有選型偏差，區間不是 Private 勝率。近期退步不限豪宅，auxiliary 收益也不限豪宅；完整價格／城市／新建案反證見[聚合摘要](evidence/reports/round100_summary.md)。

六次 NN fit 合計 118.203 秒、prior 生成 160.876 秒，均包含於 main 391.938 秒；scorer 68.985 秒；保存結果稽核內部 77.078 秒（外層程序 80.109 秒）。稽核通過 1,827 唯一 SHA、9 保護檔、705 prefix state 與 24 保存區間。稽核者參與純 X 政策／稽核且審閱 helper，其他核心模組由不同作者負責，不能稱完全獨立重作。

原 scorer 的純 X 計畫／科學計畫 SHA 接點在評分前發現，凍結原碼保留，另建 v2 窄修。42 人工檢查與真 metadata 前置核對後正式評分；MAE、配對、gate、公式保持，没有真模型失敗或重試。[v2 契約](evidence/reports/round100_evaluation_contract_v2.md)與原稿／diff 收據均保留。

[R99](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)結束時的 temporal prior 提案現已另立 R100 完成；歷史提案不回寫。[本輪有限下一方向](evidence/reports/round100_next_direction_inventory.md)只提 fit 側 prior／支持層級／ratio／bias 分布診斷，尚未執行或選定新模型，不掃季長、seed 或融合比例。[R98](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)、[R97](docs/RESEARCH_UPDATES_20260929_OFFSET.md)及[Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)完整反證保留。

本批以 c34842980f8abc4bcc15dc9d5940fe2d50ac118f 為基礎；本頁於發布前核驗，當前發布身分以 Git main 歷史與 SOURCE_MANIFEST 為準，根工作區另存發布收據。原 prep／歷史收據、379 source manifest、正式 REPORT／PDF／PPTX 與 v22 保持。交易表、模型、標籤、逐筆陣列、705 完整 prefix state 及含每個 prefix 價格的 temporal_prior_evidence 留在本機；只有其身分在 stage seal 保留。這是文件／程式快照，不是完整可重訓備份，沒有背景自動研究承諾。

'''+s[end:];p.write_text(s,encoding='utf8',newline='\n')

 p=REPO/'research/README.md';s=p.read_text(encoding='utf8').replace('R85–R99','R85–R100').replace('## 最新追加：R99','## 歷史追加：R99')
 s=s.replace('僅只讀假說盤點，未立完整計畫／實作／fit，不能把group OOF說成外層query洩漏。','在 R99 結束時僅是只讀假說；後續另立 R100 完成如下，歷史時序保留。不能把 group OOF 說成外層 query 洩漏。')
 s+='''## 最新追加：R100 時間前置且排除同折的 prior

[R100 完成頁](../docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)保存 C／T 同樣本比較、三期完整四向量、24 配對區間及分群反證。新增六個 prior NN，原 R63 trainer／架構／30 epochs／日期權重不改。T 相對 v22 為 −5,097.81／+10,063.35／−11,676.48 元，未過門檻，不新增 CSV；C 是機制對照。

|程式|用途|
|---|---|
|[contract](src/round100_contract.py)／[fit 標籤擷取](src/round100_prepare_fit_labels.py)|固定科學／評分參數、canonical／raw 索引、每期只輸出 fit 標籤|
|[純 X 稽核](src/round100_temporal_prior_x_audit.py)|日期與原五折支持，不讀價格；其 audit plan 與科學 plan 分別绑定|
|[prior helper](src/round100_temporal_prior.py)|日曆季以前且排除同 held fold，五折至少 30 筆，保存完整 prefix 狀態身分|
|[driver](src/round100_temporal_prior_experiment.py)|原控制回播、C／T 串行、六 NN 與全 query 重載／封存；不評分|
|[原 scorer](src/round100_temporal_prior_evaluate.py)／[v2 scorer](src/round100_temporal_prior_evaluate_v2.py)|保留原稿；v2 窄修兩種 plan 身分接點，科學評分／配對參數相同|
|[原 R63 trainer](src/round63_recency_weighted_prior_train.py)|兩組共同重用，9 NPZ 鍵、目標與原始日期權重保持|
|[保存結果稽核](src/round100_independent_review.py)／[摘要](src/summarize_round100.py)|1,827 唯一 SHA／705 state／24 區間；聚合整理無新 score|

本輪是 fit loss 的 prior 時間方向比較，query prior、outer-fit encoder 與 R/S 保持；不是所有輸入逐列 as-of。C/T 使用同樣 warmup 遮罩，原 raw 日期權重不 renormalize。最近窗 T−C 退 11,158.91 元；不把 C 事後另列提交，也不忽略另外两期完整模型收益。

[前置 41 檔](../evidence/outputs/research_snapshot_r100_20260929/preparation/copy_manifest.json)與[完成複製清單](../evidence/outputs/research_snapshot_r100_20260929/final/copy_manifest.json)分開保留。價格標籤／模型／NPZ／逐筆向量／705 完整 prefix state／含各 prefix 價格的 temporal_prior_evidence 不納入；stage seal 保留其 SHA。原 driver 仍依賴私有工作區，快照不等於乾淨環境重訓。379 歷史清單與正式交付不改，當前包身分以 Git main／SOURCE_MANIFEST 為準。

[下一方向盤點](../evidence/reports/round100_next_direction_inventory.md)只提出先另固定 fit 側表示與支持診斷，未執行，也沒有新增模型計畫。
''';p.write_text(s,encoding='utf8',newline='\n')
 p=REPO/NAV[3]
 assert not p.exists();p.write_text(NEW_PAGE,encoding='utf8',newline='\n')

def links(p):
 text=p.read_text(encoding='utf8')
 return [x.strip('<>') for x in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',text)]

def main():
 assert FINAL.is_dir() and not (FINAL/'document_changes.json').exists()
 edit_navigation()
 changes=[]
 for n in NAV:
  before=FINAL/'navigation_before'/n;p=REPO/n
  changes.append({'path':n,'before_sha256':sha(before) if before.exists() else None,'after_sha256':sha(p),'bytes':p.stat().st_size})
 save(FINAL/'document_changes.json',{'status':'completed','files':changes,'publication_claim':'verified before publication; current identity Git main/SOURCE_MANIFEST'})
 checks=[]
 for n in NAV:
  p=REPO/n
  for href in links(p):
   if re.match(r'^[a-z]+:|^#',href,re.I):continue
   q=(p.parent/urllib.parse.unquote(href.split('#')[0])).resolve()
   checks.append({'document':n,'href':href,'exists':q.exists()})
 assert all(r['exists'] for r in checks),[r for r in checks if not r['exists']]
 prep=read(SNAP/'preparation/copy_manifest.json');final=read(FINAL/'copy_manifest.json')
 records=prep['files']+final['files'];assert len({r['destination'] for r in records})==len(records)
 for r in records:
  assert sha(ROOT/r['source'])==r['sha256']==sha(REPO/r['destination']),r['source']
 baseline=read(SNAP/'preparation/baseline.json')['existing_files'];changed=[]
 for n,digest in baseline.items():
  assert (REPO/n).exists(),n
  if sha(REPO/n)!=digest:changed.append(n)
 assert set(changed)==set(NAV[:3]),changed
 # Original source/report links stay byte-exact. Classify dependencies without rewriting them.
 evidence_links=[]
 for r in records:
  if not r['source'].endswith('.md'):continue
  p=REPO/r['destination'];original=ROOT/r['source']
  for href in links(p):
   if re.match(r'^[a-z]+:|^#',href,re.I):continue
   hrefpath=urllib.parse.unquote(href.split('#')[0]);q=(p.parent/hrefpath).resolve()
   if q.exists():kind='repository_direct'
   else:
    oq=(original.parent/hrefpath).resolve()
    try:rel=oq.relative_to(ROOT).as_posix()
    except ValueError:rel=None
    mapped=REPO/('research/'+rel if rel and rel.startswith('src/') else 'evidence/'+rel) if rel else None
    if mapped and mapped.exists():kind='repository_source_mapping'
    elif oq.exists():kind='original_workspace_only'
    else:kind='unresolved_original_link'
   evidence_links.append({'document':r['destination'],'href':href,'classification':kind})
 source_copy=REPO/'evidence/outputs/research_snapshot_r100_20260929/finish_navigation.py'
 assert not source_copy.exists();source_copy.write_bytes(Path(__file__).read_bytes())
 save(FINAL/'link_verification.json',{'status':'passed_navigation','navigation_count':len(NAV),'navigation_links':checks,
  'original_evidence_links':evidence_links,'dependency_note':'Original report bytes preserved; private full states, temporal_prior_evidence, arrays/models and other original-workspace inputs are not included.'})
 save(FINAL/'navigation_script_identity.json',{'source':Path(__file__).relative_to(ROOT).as_posix(),'destination':source_copy.relative_to(REPO).as_posix(),'sha256':sha(source_copy),'bytes':source_copy.stat().st_size})
 evidence_counts={k:sum(r['classification']==k for r in evidence_links) for k in ['repository_direct','repository_source_mapping','original_workspace_only','unresolved_original_link']}
 result={'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'published_base':'c34842980f8abc4bcc15dc9d5940fe2d50ac118f',
  'baseline_files':len(baseline),'baseline_changed_only':changed,'other_baseline_unchanged':len(baseline)-len(changed),
  'historical_379_sources_and_manifest_unchanged':True,'formal_REPORT_PDF_PPTX_unchanged':True,'SOURCE_MANIFEST_unchanged':True,
  'preparation_original_copies':prep['copied_count'],'final_original_copies':final['copied_count'],
  'total_original_copies':len(records),'total_original_copied_bytes':final['total_copied_bytes'],
  'navigation_changes':changes,'navigation_local_links_checked':len(checks),'all_navigation_links_resolve':True,
  'original_evidence_link_classifications':evidence_counts,'all_original_copies_exact':True,
  'extra_new_navigation_script_copy':{'sha256':sha(source_copy),'bytes':source_copy.stat().st_size},
  'privacy_excluded':['705 complete prefix states','price-bearing temporal_prior_evidence','raw transaction data','labels','arrays','models','private/runs'],
  'ML_fit_or_score_performed':False,'package_commit_push_performed':False}
 save(FINAL/'final_local_verification.json',result)
 print(json.dumps({k:result[k] for k in ['status','total_original_copies','total_original_copied_bytes','navigation_local_links_checked','original_evidence_link_classifications']},ensure_ascii=False))
 print('final_receipt_sha256='+sha(FINAL/'final_local_verification.json'))
if __name__=='__main__':main()
