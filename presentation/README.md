# v22 完整流程上台簡報

[PowerPoint](臺灣房價預測_正式報告_v22.pptx)、[完整逐頁講稿](講稿與問答.md)、[排練與問答指南](排練與問答指南.md)及[流程與程式對照](流程與程式對照.md)。

22頁＝20頁主講＋2頁附錄。主講配速規劃 **14分30秒**；正文4,143個純漢字，按每分鐘250–300字估計約13.81–16.57分鐘，不計附錄、問答、來源、英文數字或停頓。這是語速假設，尚未真人排練；實際上台前請完整計時一次。排練指南附各段累積時間及八題常見問答。

主線涵蓋回歸與MAE、資料角色、重複資料清理、特徵與排除欄位、時間驗證、六模型與三分支、單價與面積權重、近期權重、分組OOF價格先驗、成交參考與空間修正、中位數、結果與過擬合限制、最終訓練、提交CSV及保存模型重現。每步均交代問題、處理及理由；示意數字明確標示，設計理由不等於已完成獨立消融證明。主講只說明選定的v22；研究對照與不利證據保留於正式報告附錄，Private仍未知。

4張圖表、10個表格均為可編輯PowerPoint物件，22頁內建備忘稿。最終PPTX已重新匯入並逐頁視覺檢查，未在原生PowerPoint實際開啟驗證。

## 製作來源與時長核對

source/保存製作與finalizer JavaScript、圖表數據、narrative.json、內容大綱及封面圖片。腳本保留本機Codex ArtifactTool runtime路徑，供追溯和後續改稿，不宣稱在全新Node環境可直接重建。模型推論另見根目錄README。

在儲存庫根目錄可用標準Python重算講稿字數（不執行模型）：

~~~powershell
python scripts/verify_talk_timing.py --input presentation/source/narrative.json --output evidence/presentation/timing_receipt.json
~~~

核驗來源為evidence/presentation/內的final.validation.json、content_review.md、delivery_receipt.json及timing_receipt.json。14:30是逐頁配速，不是實測報告時間。

封面：Markus Winkler，Taipei，Unsplash。圖片僅作住宅情境，不代表競賽個別交易。
https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs
https://unsplash.com/license

[私人GitHub專案](https://github.com/ziyi930108/nycu-house-price-regression)保留程式與報告；程式包沒有模型權重與交易資料。
