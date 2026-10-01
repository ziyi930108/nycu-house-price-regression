# 從儲存庫重建口語版簡報

本命令預設仍為舊17:20稿。最新約15分鐘版請看[新增版別操作](FIFTEEN_MINUTE_REBUILD.md)，加`--edition fifteen-minute`。下方原實測收據及舊稿身份保持。

[rebuild_spoken_teaching.py](../scripts/rebuild_spoken_teaching.py)以repo原r2三款PPTX、口語講稿計畫及保存製作來源，在全新輸出目錄同步備忘稿。全部專案輸入來自repo，不需原C:/NYCU_ML、交易Excel或房價模型。

已使用從私人GitHub main 7035e492e1164be5783c449c67b7e6435147a551下載的11項輸入，以新增runner實際生成三款；39.552秒含外層執行，無重訓。三份PPTX與已交付口語版逐bytes相同，69張PNG相同，23頁notes按實際slide關聯順序核對。原工作表、媒體及所有非主講notes的OPC部分逐bytes保持。[實際驗證](../teaching/v22_styles_20261001/spoken/rebuild_evidence/verification.json)、[建置收據](../teaching/v22_styles_20261001/spoken/rebuild_evidence/rebuild_receipt.json)、[輸入與環境](../teaching/v22_styles_20261001/spoken/rebuild_evidence/preflight.json)。

## 使用方式

這是在既有Windows、Codex bundled Node v24.19.0／Python3.12.14與字型環境的驗證。需要完整@oai/artifact-tool與Presentations技能container_tools，以及Microsoft JhengHei、Consolas，暖色款另需PMingLiU。未測試全新安裝、其他系統或桌面PowerPoint。不要以使用者電腦上另裝的LibreOffice取代bundled工具。

在repo根目錄執行，下列runtime路徑請改成實際位置。輸出必須尚不存在，且位於repo、runtime與技能資料夾以外：

```powershell
python scripts/rebuild_spoken_teaching.py --output C:/work/v22_spoken_new --node C:/runtime/node/bin/node.exe --python C:/runtime/python/python.exe --node-modules C:/runtime/node/node_modules --presentation-skill C:/skills/presentations --check-only
```

移除`--check-only`才生成。可加`--styles A`或`--styles B C`，不加即三款。`--repository`可指定另一份clone，預設腳本所在repo。Codex中可由load_workspace_dependencies查bundled runtime；agent實際編輯PPTX前另依Presentations技能執行一次edit marker，runner不替agent重複呼叫marker。

前置檢查核來源SHA、路徑、runtime檔案、版本及原投影片與notes關聯；不建立輸出。它不能取代完整建置、字型或渲染檢查。

## 輸出及保存

輸出含deliverables內新PPTX、content講稿與問答、build內路徑適配來源／日誌／渲染，以及preflight.json與rebuild_receipt.json。從repo讀取原文件，只在新目錄改寫副本路徑。利用既有node_modules junction／symlink，不安裝依賴。

Artifact編輯後先保留其候選，再將20份主講notes XML放回原PPTX套件，保留原圖表內嵌工作表。最終檔必須通過同一套finalizer，非notes部分精確保持，23頁備忘稿及全部渲染須與保存版本相同，否則留failure.json並失敗停止。不同runtime可能產生像素差；需檢查輸出，不應直接放寬成功條件。

輸出已存在、與repo重疊、runtime缺件、輸入被改動，都在建置前拒絕；本次另測check-only無輸出，五項檢查通過。建置失敗後保留候選與日誌，修正後改用新目錄，不覆寫舊產物。

此命令重建凍結口語版。未來修稿需新增版本、重新同步notes計畫及核验，而非移除SHA保護。舊[rebuild_teaching.py](TEACHING_REBUILD.md)繼續重建r2。口語版主講4,981字、規劃17:20，仍需真人計時；v22模型、Public及未知Private保持。
