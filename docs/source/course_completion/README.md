# 正式報告的文件重建來源

build_report.py讀取同目錄的reference_report.docx、supplement.json及assets，插入7章補充內容，輸出19章正式報告與Markdown。參考Word的SHA在程式內固定，原標題、姓名、學號、樣式、頁尾、表格與單一NN曲線沿用；新增章節只改document.xml及其圖片關係。這是文件重建，沒有模型訓練或提交CSV修改。

本資料夾可獨立搬移。直接執行build_report.py時，預設輸出到本資料夾的父目錄。make_supplement.py可從公開匿名統計摘要重寫supplement.json；build_flowchart.py可在Windows微軟正黑體存在時重畫連接方塊圖PNG/SVG。其餘科學圖沿用主執行緒已核對的ReportLab輸出，原始資料與逐筆價格不在本資料夾。

若放在repo的docs/source/course_completion，可由repo根目錄使用：

```text
python docs/source/course_completion/build_report.py --output-dir docs --markdown-path REPORT.md --source-map-path docs/REPORT_SOURCES.md --qa-dir build/report_course
```

--output-dir決定正式DOCX位置；--markdown-path決定文字版位置；--source-map-path只設定來源對照連結，來源對照檔須另外保留；--qa-dir決定build.json與文字核對檔位置。所有參數都接受相對或絕對路徑。Markdown圖片連結按輸出檔位置自動重算。

本次已用Codex bundled Python/python-docx在獨立來源副本與自訂CLI位置實跑。DOCX ZIP容器時間戳可能不同，各package part的內容一致。原生Word16匯出PDF後以官方render_docx.py與bundled Poppler產生19張PNG，全部已目視；render_review.py是此Windows工作區的QA適配器，內含本機runtime/skill路徑，搬機時需依新環境重設。Windows bundle沒有LibreOffice，未使用桌面LibreOffice替代。

此文件builder驗證不涵蓋全新電腦、其他Office重排或六模型重訓。保存六模型的完整推論另有兩套同機環境的實際通過收據；文件重建與模型推論是不同驗證範圍。

本次新增的最後一章文字由build_report.py內精確段落mapping產生，引用assets/reproducibility_receipt.json及原訓練來源盤點；make_supplement.py不改該章。重建不讀模型／課程交易或封存逐筆預測。20檔可攜推論source另提供，並非本文件builder會執行模型推論。
