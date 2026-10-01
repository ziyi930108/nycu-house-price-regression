# v22 上台資料包

三款呈現相同的房價回歸流程，各23頁，請挑一款使用，不需連播三份。使用者尚未選定風格；r2代表簡報修訂，模型仍為v22。

| 風格 | 視覺特色 | 可編輯版 | 固定畫面備份 |
| --- | --- | --- | --- |
| A 白底學術 | 白底藍色重點，表格清楚、風格簡潔 | [PPTX](../deliverables/臺灣房價預測_v22_白底學術_r2.pptx) | [PDF](臺灣房價預測_v22_白底學術_投影備份.pdf) |
| B 深色科技 | 深藍底搭配青綠，重點數字突出 | [PPTX](../deliverables/臺灣房價預測_v22_深色科技_r2.pptx) | [PDF](臺灣房價預測_v22_深色科技_投影備份.pdf) |
| C 暖色雜誌 | 米色、赭色與明體標題，封面加入城市照片 | [PPTX](../deliverables/臺灣房價預測_v22_暖色雜誌_r2.pptx) | [PDF](臺灣房價預測_v22_暖色雜誌_投影備份.pdf) |

## 報告怎麼安排

第1至20頁是主講，從回歸、MAE、資料清理與特徵，講到時間回測、六個模型、三條分支、成交參考與空間修正、中位數、結果及限制。第21至23頁分別是公式、參數程式與名詞附錄，不計入主講時長。

[完整講稿](../content/完整講稿.md)的主講部分有4,789個純漢字，規劃配速16分45秒，按每分鐘250至300字估計約15.96至19.16分鐘。這是文字與配速估算，尚未真人計時。先完整排練第1至20頁，確認至少12分鐘，不靠附錄或問答補時；[排練與問答](../content/排練與問答.md)另附16題準備。

PPTX包含可編輯文字、表格與圖表，23頁皆有講者備忘稿。PDF由已核對的投影片PNG逐頁製成，共23頁，作為固定版面的投影備份；每頁是1600×900的整張圖片，文字不能直接搜尋或編輯，也不包含PPTX講者備忘稿。放大清晰度受原圖解析度限制，PDF書籤可跳到指定頁。

## 上台前

1. 解壓縮整包，在實際報告電腦開啟所選PPTX及對應PDF，測試全螢幕與翻頁。
2. 依講稿計時排練，保留說明模型為何如此設計的時間。
3. 第13頁價格是教學假設，不是test答案；第17頁199.86萬元是Public平均絕對誤差，Private仍未知。
4. 第20頁完成主講，附錄留作補充說明。

PDF已用Poppler全部轉成圖片檢視；尚未在桌面PowerPoint或實體投影機測試。此包是簡報成品與講稿，不含完整模型、交易資料或重訓環境。

## PDF 製作來源與重建

本目錄是三款r2的投影備份，不改PPTX、模型或CSV。原23頁PDF各含23個頁面書籤；所有69個內嵌RGB圖與原圖像素完全相同。Poppler重新顯示的像素有差異，未宣稱逐像素相同；主線與子智能體以九張contact sheets檢視69頁，另放大六張重要頁，未見阻擋閱讀的缺陷。

[生成與全頁比較收據](delivery_receipt.json)、[視覺核對](visual_review.json)、[本機九檔上台包收據](package_receipt.json)保存原製作時點。收據中的C:/NYCU_ML是原工作區路徑；上台ZIP保留本機，沒有加入repo。此處的相对連結是repo入口。

先依[簡報重建說明](../../../docs/TEACHING_REBUILD.md)在新目錄完成三款及69張PNG，再執行本目錄的export_slide_pdfs.py。需要Python、ReportLab、pypdf、Pillow及Poppler；本次既有環境版本分別ReportLab 4.4.9、pypdf 6.10.0、Pillow 12.3.0，未測全新安裝。

~~~powershell
python -X utf8 teaching/v22_styles_20261001/projection/export_slide_pdfs.py --reference teaching/v22_styles_20261001 --render-root C:/path/to/rebuilt/build --output C:/path/to/new_pdf_output --pdftoppm C:/path/to/pdftoppm.exe
~~~

render-root必須是三款已完成重建的build目錄。程式先核對凍結PPTX及69張PNG SHA，再寫入不存在且不與來源重疊的新目錄；不需要讀交易資料或模型。PDF生成、內嵌圖片與全頁Poppler檢查已實跑，以上clone路徑的命令尚未另行重跑PDF，不將說明冒稱第二次執行。
