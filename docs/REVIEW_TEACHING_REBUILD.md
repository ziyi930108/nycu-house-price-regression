# 重建目前4,065字審查版簡報

[重建入口](../scripts/rebuild_review_teaching.py)使用repo內保存的三款4,051字版PPTX，加上目前4,065字主講稿與備忘稿計畫，重做三款23頁審查版PPTX。只更新第12、13頁備忘稿，投影片、原生表格、圖表及嵌入工作簿沿用原件。20頁主講規劃14:45，另3頁附錄；字數只計主講正文，不含提示或設計理由，也不是本人已計時的證明。

這項工作不讀取房價模型或交易資料、不進行訓練或推論，不改v22提交CSV。它不生成PDF、HTML、講解影片或ZIP。舊版[口語簡報入口](../scripts/rebuild_spoken_teaching.py)及4,051字版驗證保留，各自代表不同稿本。

## 輸入與防護

- 原投影片在`teaching/v22_styles_20261001/fifteen_minute/deliverables/`。
- 目前稿本在`teaching/v22_review_20261002/04_程式與證據/本次內容/`；同層`本次製作程式/`的原四個builder保持不變。
- 入口固定核對來源PPTX、目前PPTX、兩份稿本、原builder及兩份製作收據的SHA256。這是固定版本重建；若要改稿，應另建版本及對應證據，不直接取消SHA檢查。
- 輸出目錄必須完全不存在，且不能與repo、Node/Python執行環境或Presentations技能目錄重疊；既有成功與失敗目錄都不覆寫。
- `--check-only`只讀取並檢查輸入，將JSON印到終端，不建立目錄、PPTX或渲染圖，也不執行authoring marker。
- 真正建置結束後重查全部輸入與所讀runtime檔案，三PPTX及69張PNG必須與保存參考bytes完全相同。差異會被視為失敗，不放寬比對。

## 使用方式

需要Python 3.10以上與Codex提供的Node、Python、`@oai/artifact-tool`及Presentations技能。先呼叫`load_workspace_dependencies`取得本機絕對路徑；下例是本專案受測機器的路徑，換電腦須替換成實際回傳值。不要另裝替代artifact套件，或修改bundled runtime。這個流程不用LibreOffice；若後續確實需要，僅可使用`load_workspace_dependencies`取得的bundled絕對路徑，不可改用桌面安裝。

在repo根目錄開PowerShell，先核對輸入：

```powershell
$reviewPython = 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$reviewNode = 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
$reviewModules = 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
$reviewSkill = 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations'
$reviewOutput = [IO.Path]::GetFullPath((Join-Path (Get-Location).Path '../v22_review_rebuilt_01'))
$reviewArgs = @(
  '-B', '-X', 'utf8', './scripts/rebuild_review_teaching.py',
  '--output', $reviewOutput,
  '--node', $reviewNode,
  '--python', $reviewPython,
  '--node-modules', $reviewModules,
  '--presentation-skill', $reviewSkill
)
& $reviewPython @reviewArgs --check-only
```

如果要實際產生PPTX，先閱讀該Presentations技能要求，在技能目錄獨立執行一次edit marker。此入口及原builder不會替呼叫者重複執行marker。以下marker命令採本機已確認的絕對路徑，不用shell變數展開：

```powershell
Push-Location 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations'
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' container_tools/mark_artifact_operation_started.mjs --operation-kind edit --expected-output-count 3 --output-format pptx
Pop-Location
```

確認marker成功後，在同一個PowerShell及repo目錄執行：

```powershell
& $reviewPython @reviewArgs
```

成功會在新目錄產生`deliverables/`的三PPTX、`build/A`至`build/C`各23張渲染圖及檢查紀錄。最外層`rebuild_receipt.json`記錄與目前稿本的完整比對，底層`build/execution.json`記錄原builder的九個步驟。失敗時保留`entry_failure.json`及stdout/stderr，修正原因後使用全新輸出目錄。

這能驗證同一已配置環境中的固定簡報重建；不代表新電腦安裝成功、桌面PowerPoint已試播、本人已排練、HTML或影片已重建，更不代表房價模型已完整重訓。

## 這次實際核驗結果

2026-10-02已從repo來源實跑上述新入口。wrapper量測建置及驗證47.149秒；三個輸出PPTX與目前4,065字稿的審查版整檔SHA256相同，69張PNG也與原4,051字及目前4,065字版的保存參考相同。主講稿改動位於備忘稿，因此投影畫面保持；三款各第12、13頁另實際看圖核對。原生表格、圖表及嵌入工作簿保持，沒有新增ML fit。

[驗證摘要](../evidence/review_teaching_rebuild_20261002/verification_summary.json)及[建置收據](../evidence/review_teaching_rebuild_20261002/rebuild_receipt.json)列出來源、輸出與時間；[防護實測](../evidence/review_teaching_rebuild_20261002/guards.json)確認拒絕既有目錄、repo/runtime重疊、缺檔及來源SHA變更。這次只重建三個PPTX及其檢查圖，沒有重建HTML、PDF或影片。
