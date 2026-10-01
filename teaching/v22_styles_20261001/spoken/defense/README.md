# 上台答辯提示卡

2026-10-01新增[四頁可列印PDF](臺灣房價預測_v22_答辯提示卡.pdf)與[文字版](臺灣房價預測_v22_答辯提示卡.md)。涵蓋原16題，每題先給可直接說的一兩句，再附追問補充與投影片頁碼。答案共1,603漢字，問答不計入至少12分鐘的主講。

適合先練MAE、valid／時間回測、OOF、Private與重現五題，再練其餘題目。PDF是A4，每頁四題，正文13.2pt、補充11.7pt，有可搜尋文字與16書籤。已逐頁看過四張Poppler渲染，內容抽取核對16題兩層答案，未實體列印或真人試答。原完整問答、三款PPTX和主講稿不改。

短答保留重要限制：OOF按建案而非時間，也非六模型全做五折；Private未知；只完成保存模型推論，未完成全新安裝與從零重訓。方法仍為v22，Public 1,998,568.5706元，R106停止，沒有新模型或CSV。

依據：[原排練問答](../content/排練與問答.md)、[完整口語稿](../content/完整口語講稿.md)及[研究報告](../../../../REPORT.md)。提案／核准JSON、內容查核、[視覺審閱](visual_review.json)及建置收據保存；建置收據的pending狀態屬產生當時，以其後視覺審閱為準。

## 重建PDF

使用Python的reportlab、pypdf，以及Poppler pdftoppm。字型需Microsoft JhengHei及粗體。以下路徑換成實際位置，輸出目錄必須尚不存在；builder預設讀同目錄questions_approved.json，也可用--content指定它。此命令已在原目錄實跑，未另從全新環境安裝測試。

```powershell
python teaching/v22_styles_20261001/spoken/defense/build_cards.py --output C:/work/v22_defense_new --font C:/Windows/Fonts/msjh.ttc --bold-font C:/Windows/Fonts/msjhbd.ttc --pdftoppm C:/runtime/poppler/bin/pdftoppm.exe
```

需先在Codex用PDF技能時，依技能執行一次create marker；builder不重複發marker。程式以固定四頁排版，長度超出會拒絕，全部問題／答案須可完整抽取。完成後仍要看四張渲染，不以文字檢查替代視覺核對。沒有房价模型相依。
