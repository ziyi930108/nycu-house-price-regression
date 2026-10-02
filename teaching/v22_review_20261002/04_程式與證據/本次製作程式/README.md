# 審查版備忘稿建置

GitHub接手請先用[repo重建入口](../../../../docs/REVIEW_TEACHING_REBUILD.md)。下列絕對路徑命令保留作10月1日原始執行紀錄，不要在新電腦直接照抄。

這四個程式以原約15分鐘版PPTX為來源，只更新`notes_plan.json`明列的主講頁備忘稿。保留23頁投影片、所有其他OPC parts、原生表格／圖表及嵌入工作簿；不使用python-pptx、不呼叫LibreOffice。

主線先完成`work/content/narrative.json`與`notes_plan.json`。後者須含`narrative_sha256`及三個`styles`（A、B、C）；每筆須含`style`、`source_file`、`source_sha256`、23筆`notes`、排序不重複的`changed_slide_numbers`。每頁script必須完整出現於對應note一次。未列入修改的notes須與來源完全相同。

先以相同參數加`--check-only`檢查。真正authoring前，呼叫者須依Presentations技能在skill目錄**獨立且僅一次**執行marker（`--operation-kind edit --expected-output-count 3 --output-format pptx`）；runner不重複呼叫marker。

```powershell
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/NYCU_ML/outputs/v22_review_pack_20261001_work/build_source/run_build.py' `
  --source-dir 'C:/NYCU_ML/outputs/v22_15min_20261001/deliverables' `
  --work-dir 'C:/NYCU_ML/outputs/v22_review_pack_20261001_work' `
  --runtime-node 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' `
  --runtime-python 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' `
  --runtime-node-modules 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules' `
  --skill-dir 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations' `
  --reference-receipt 'C:/NYCU_ML/outputs/v22_15min_20261001/delivery_receipt.json'
```

`build/`必須不存在；任何既有最終PPTX也會被拒絕覆寫。輸出為`deliverables/*_審查版.pptx`及`build/{A,B,C}/renders/`共69張PNG，收據在`build/execution.json`。可選`--reference-receipt`會核對三款全部PNG bytes與來源收據相同。失敗保存log與failure.json，另設新work目錄再試，不移除原證據。

這個builder不產生PDF／排練HTML／ZIP，也不代表已完成真人排練或桌面PowerPoint試播。若後續另需LibreOffice，只可用`load_workspace_dependencies`回傳的bundled絕對路徑；目前流程不需要也未提供此路徑，禁止改用桌面安裝。
