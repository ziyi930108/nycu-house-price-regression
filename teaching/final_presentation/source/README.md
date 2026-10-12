# 簡報與講稿重建

本目錄只包含教學簡報的內容與程式，不會訓練房價模型或修改提交CSV。

## 必要來源

- `build_deck.mjs`：24頁簡報、21頁主講與3頁附錄，另輸出口語講稿。
- `readable_ppt_content_20261012.json`：逐頁內容、講稿、理由與來源對照。
- `training_curve.json`：30輪、兩條曲線的60個原始總價MAE，單位新臺幣元。
- `finalize_deck.mjs`：檢查PPTX、建立圖表資料活頁簿，並渲染全部24頁。

圖表活頁簿採Excel能保留的15位有效數字，與原值最大差小於0.000001元。原始JSON保持原數值。曲線是另行固定設定的單一神經網路診斷，不是完整融合，也不是原提交當時的逐輪軌跡。

## 執行環境

使用Codex提供的bundled Node、`@oai/artifact-tool`、Python及Presentations技能finalizer。字型為Microsoft JhengHei與Consolas。這不是一般npm套件的全新環境安裝說明。

目前本機Node：`C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe`

目前Python：`C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`

程式以source目錄的上一層為工作根目錄。先用技能的`mark_artifact_operation_started.mjs`記錄一次編輯，再執行：

```powershell
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' './source/build_deck.mjs'
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' './source/finalize_deck.mjs' r1
```

產物留在`build/`，正式PPTX為`build/final/r1.pptx`，完整講稿為`build/deliverables/完整口語講稿.md`。再次finalize請使用新編號，例如r2，既有final不覆寫。每次仍須視覺確認全部渲染圖。這些程式不會覆寫資料夾根目錄的已交付PPTX。

主講4118個純漢字，275字／分估14.97分鐘，250–300字／分估16.47–13.73分鐘。附錄與問答不計。這是字數估算，尚未本人排練計時。
