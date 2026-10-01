# 重建約15分鐘版簡報與排練頁

在repo根目錄使用[rebuild_spoken_teaching.py](../scripts/rebuild_spoken_teaching.py)，明確指定`--edition fifteen-minute`。不指定edition時保持舊17:20口語版，避免把舊流程悄悄換成另一份稿。

```powershell
python scripts/rebuild_spoken_teaching.py --edition fifteen-minute --output C:/work/v22_fifteen_new --node C:/runtime/node/bin/node.exe --python C:/runtime/python/python.exe --node-modules C:/runtime/node/node_modules --presentation-skill C:/skills/presentations --check-only
```

請將runtime及skill路徑換成實際位置。輸出目錄必須不存在，且位於repo、runtime及skill之外。移除`--check-only`才會建立檔案。預設三款，可加`--styles A`或`--styles B C`。`--repository`可選另一份clone；預設使用入口所在repo。前置檢查不建立輸出，不能替代實際建置。

它從repo讀13項專案輸入，包括原r2三款基底、新版notes計畫、講稿、製作配方與排練模板；不用原C:/NYCU_ML的交易資料、模型或outputs。先核SOURCE_MANIFEST，再核對新版4,051字／885秒、23頁順序及附錄界線、notes對應講稿、收據版本與HTML精確重現。所有路徑替換只發生在新輸出副本，原來源不改。

產物包含三款PPTX、content講稿／問答／提示、離線排練HTML、渲染與執行收據。PDF及答辯卡可沿用[已交付連結](../teaching/v22_styles_20261001/fifteen_minute/README.md)，此命令不另外重建13檔上台ZIP。HTML以原模板及narrative SHA重建，保留原始換行；本輪結果與已交付HTML逐bytes相同，排練記錄的來源識別也相同。

已從GitHub main 94922b362e4b728409bc12e70bfa40bda856e202的verified clone，以新增入口實際生成三款及HTML，含外層46.446秒。[驗證](../teaching/v22_styles_20261001/fifteen_minute/rebuild_evidence/verification.json)、[完整建置收據](../teaching/v22_styles_20261001/fifteen_minute/rebuild_evidence/rebuild_receipt.json)、[13項輸入及環境](../teaching/v22_styles_20261001/fifteen_minute/rebuild_evidence/preflight.json)、[八项拒絕案例](../teaching/v22_styles_20261001/fifteen_minute/rebuild_evidence/guard_verification.json)與[唯讀審查](../teaching/v22_styles_20261001/fifteen_minute/rebuild_evidence/review.json)保存證據。

本輪三PPTX全bytes相同、69PNG全bytes相同、23頁notes按實際關聯一致。Artifact只編輯20份主講notes，放回原OPC後用既有finalizer核驗，原表格、圖表工作表及其他parts保持。不同runtime下PPTX封裝bytes可能不同，該欄位會如實回報；渲染、notes或保留parts不符會停止，不能直接放寬門檻。

驗證環境是既有Windows、Codex bundled Node v24.19.0／Python3.12.14、@oai/artifact-tool及Presentations技能工具，需Microsoft JhengHei、Consolas，暖色款加PMingLiU。尚未全新安裝、其他作業系統或桌面PowerPoint測試。Codex編輯前依Presentations技能另執行一次edit marker，runner不重複呼叫；如需LibreOffice只能用bundled版本，不用電腦上另裝的版本。

主講仍為4,051字／規劃14:45，尚未本人計時；本輪只補重建入口。模型、Public與未知Private均保持，沒有模型訓練、CSV或Kaggle操作。此為簡報重建，不是房價模型从零重訓。
