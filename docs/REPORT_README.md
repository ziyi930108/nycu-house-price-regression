# 正式報告與可讀文字版

[開啟正式Word報告](台灣房屋銷售價格預測挑戰賽_正式報告.docx)；[閱讀Markdown文字版](../REPORT.md)；[來源對照](REPORT_SOURCES.md)。

沿用原正式報告的A4頁面、標題、姓名、學號與表格外觀，補入完整程式流程、核心寫法、城市價格分布、完整融合驗證密度圖、價格帶與尾端誤差、三個大誤差交易案例，以及對應改進。保留既有方法、歷史不利證據與重現限制。未恢復已刪的摘要結果段。

Word共19頁、19章、16個原生表格、5張圖。完整融合的密度圖使用全部133,697筆時間驗證交易；單一網路曲線仍清楚標示為另行固定30輪診斷。Private未知。

重建入口是source/build_report.py；先執行source/make_supplement.py可從同目錄保存摘要重寫新增內容，再執行builder。builder以source/reference_report.docx的固定SHA複本插入新章，不需讀取交易資料或模型。source/assets含文件所需PNG、流程SVG及公開匿名摘要。文件重建不會訓練模型，也不產生或修改提交CSV。

builder提供 `--output-dir`（DOCX位置）、`--markdown-path`（文字版完整路徑）、`--source-map-path`（來源對照位置）及 `--qa-dir`（收據位置）。圖片連結會按Markdown檔的位置自動重算。省略參數時，輸出到source的父目錄，與本資料夾相同。

版面核對採Microsoft Word 16隱藏實例原生PDF匯出，再由官方render_docx.py配合bundled Poppler產生PNG；全部19頁已目視檢查。不同Office版本仍可能略有重排。本資料夾不含完整六模型重訓備份。

本版只在最後一章補上已完成的六模型推論驗證與原訓練來源補件：同機既有與全新獨立套件環境，各重新載入保存模型／轉換狀態，13個完整向量逐位元相同、47,073筆CSV bytes相同。沒有新增fit，亦未更換硬體。原R107僅新fit一NN與五個保存向量的歷史說明保留；完整raw→六模型重訓尚未驗證。原課程補齊版保持不變。
