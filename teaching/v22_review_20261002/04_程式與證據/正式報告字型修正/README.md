# 正式報告負號字形修正

[本包正式PDF](../../03_正式報告/臺灣房價預測_v22_正式報告.pdf)是原報告的衍生修正版。第10頁「最新年度 -1,999.31 元」原負號顯示成方框，現在改為可顯示的ASCII減號；數值與模型方法沒有變。

11頁全部重新渲染，僅第10頁有像素差異，其餘10頁RGB像素完全相同。第10頁已檢視，負號正常、沒有裁切或重疊；逐頁正文核對符合宣告修補。[最終收據](final_receipt.json)、[來源與生成記錄](build_receipt.json)、[機器核对](verification.json)可供查證。機器核對當時的待看圖標記屬先前階段，最終以final_receipt為準。

`REPORT_font_fixed.txt`內容仍為Markdown格式，為保留字元身分而以文字來源存放。裡面的相對證據路徑以repo根目錄為基準；要點來源連結，請閱讀[根REPORT.md](../../../../REPORT.md)。這份txt是生成輸入，不是獨立的證據導覽頁。

修正版builder沿用原A4版型、字型與表格，只新增Unicode負號正規化和拒絕覆寫。生成依賴已裝有ReportLab與pypdf的Python，以及Windows Microsoft JhengHei字型。從本資料夾執行：

```powershell
$pdfPython = Read-Host '具備ReportLab與pypdf的Python完整路徑'
& $pdfPython -B -X utf8 ./build_report_pdf_font_fixed.py --input ./REPORT_font_fixed.txt --output ./new-output/report_font_fixed.pdf --font-dir C:/Windows/Fonts
if ($LASTEXITCODE -ne 0) { throw 'PDF生成失敗，請保留新輸出與終端訊息。' }
```

輸出PDF和同名build.json都必須不存在。再次生成的PDF時間欄位可能不同，不承諾整檔SHA相同。原實測只用既有環境，不是新電腦安裝測試，也沒有重跑機器學習。原封存PDF、REPORT.md、原builder均保留；修正版另存在本目錄，原封存檔不覆寫。


另已使用修正版builder直接讀原程式包REPORT.md，新的11頁全文、繪圖stream及字型資源與本修正版完全相同，詳見[原來源重建核對](original_source_verification.json)。因此也可將上例`--input`指定為原程式包的REPORT.md；原檔不必改寫。這次比較沒有重渲全報告，也未重跑模型。
