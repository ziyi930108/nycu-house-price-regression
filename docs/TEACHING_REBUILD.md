# 從GitHub來源重建三款簡報

新增[scripts/rebuild_teaching.py](../scripts/rebuild_teaching.py)，可把已保存的v22簡報來源在全新目錄重新生成。只需要本儲存庫的講稿、製作來源、彙總結果及情境照片；不需原始交易Excel、模型權重、逐筆預測或原C:/NYCU_ML研究outputs。

已用從私人GitHub重新clone的79f9a415688009fbe4d4b39f3d662aaa52cb1c28內容實測三款，每款23頁。全部69張PNG與已審原r2逐位元相同，文字、完整notes、圖表數值、媒體及頁面尺寸一致。PPTX整檔SHA不同，因此不宣稱PPTX位元重現，也不取代已發布的原r2成品。[驗證收據](../teaching/v22_styles_20261001/rebuild_evidence/verification.json)保存每款新舊SHA與檢查範圍。

## 環境與執行

這次實測為同一台Windows的既有Node v24.19.0／Python3.12.14、Codex bundled @oai/artifact-tool及Presentations技能。必須保有完整runtime與技能container_tools相依，不能只複製兩個驗證器。字型Microsoft JhengHei、Consolas需安裝，C另需PMingLiU。未測試全新runtime安裝、其他系統或桌面PowerPoint。

將下列範例中的runtime路徑換成實際位置，於repo根目錄執行。輸出目錄必須尚不存在；可以位於repo外的其他可寫位置。若在Codex中操作，可由load_workspace_dependencies取得bundled Node、Python與node_modules，再取得已安裝Presentations技能路徑。

```powershell
python scripts/rebuild_teaching.py --output C:/work/v22_slides_new --node C:/runtime/node/bin/node.exe --python C:/runtime/python/python.exe --node-modules C:/runtime/node/node_modules --presentation-skill C:/skills/presentations --check-only
```

`--check-only`核對來源SHA、路徑、關鍵runtime檔案與版本，不建立輸出；它不代表全部套件／字型或版面已實際執行。確認後移除`--check-only`才正式製作。同一命令可加`--styles A`或`--styles B C`只產生選定風格。`--repository`可指定另一份clone，預設為腳本所在repo。

程式在新目錄建立content、build和deliverables；僅於複製的六份MJS改寫資料／runtime路徑，原來源逐檔SHA前後核對。Node modules以junction／symlink連接既有runtime，未複製或下載套件。輸出不允許與來源、runtime目錄重疊，也不覆寫已存在資料夾。

## 輸出與失敗處理

- deliverables/：新PPTX。
- content/：凍結講稿、敘事及問答。
- build/A、B、C：路徑配置後的來源、執行stdout／stderr、驗證收據、23頁渲染。
- preflight.json：真正讀取的repo輸入SHA、runtime版本／路徑及runner身分。
- rebuild_receipt.json：完成步驟、實測耗時、新PPTX與render SHA及原版比較。
- failure.json：開始建置後若有錯誤，保留當時步驟与日誌；修正後使用另一個新輸出目錄。

不同runtime可能產生不同渲染，請查看比較結果及新畫面，不能只看程序exit0就稱外觀一致。[本次建置收據](../teaching/v22_styles_20261001/rebuild_evidence/rebuild_receipt.json)及[實際輸入](../teaching/v22_styles_20261001/rebuild_evidence/preflight.json)是既有環境的實測記錄，包含當時本機路徑，不是新使用者應照抄的路徑。

此入口重建簡報，沒有訓練或載入房價模型。模型推論請看[重現說明](REPRODUCIBILITY.md)。v22唯一提交、Public1,998,568.5706元、Private未知及16:45主講規劃保持；使用者尚未選最終風格。
