# v22 零基礎報告製作來源

此目錄只製作說明文件，不訓練、不推論或修改價格。正式成品是工作區 deliverables/ 的同名 PDF 和 Markdown。

## 重建

以 report_content.json 為內容來源，build_report.py 生成兩種格式。需要 reportlab 和 pypdf；本次使用 Codex bundled Python。字體預設 Windows 微軟正黑體；可用 --font 和 --bold-font 指定可用的 TrueType 字體。

```powershell
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 C:/NYCU_ML/outputs/v22_beginner_20261001/report/build_report.py --output-dir C:/NYCU_ML/deliverables
```

builder 的頁數與文字檢查不代表版面核驗。每次內容變更後，重新渲染並檢查有變動的頁面：

```powershell
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe' -r 110 -png 'C:/NYCU_ML/deliverables/臺灣房價預測_v22_零基礎報告.pdf' 'C:/NYCU_ML/outputs/v22_beginner_20261001/report/page'
```

## 核验證據

- build_receipt.json：生成結果、頁數、來源與成品身分；此檔保留 builder 的 pending_visual_review 狀態。
- visual_qa.json：最終14頁視檢結果，最終PDF/MD SHA及每頁PNG SHA，為版面驗證入口。
- source_evidence.json：29份方法、程式、地理與重現證據的身分；原v22 CSV未變。
- build_history.json：保留首稿來源表溢出15頁的失敗與後續修复，沒有把失敗改成成功。
- page-01.png 至 page-14.png：本次 final PDF 的110dpi render，只作內部QA。

原始v22程式、模型、提交與舊報告均未修改。程式包不是全部模型重訓備份；Private仍未知。
