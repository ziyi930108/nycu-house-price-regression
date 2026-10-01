# v22 約15分鐘上台版

使用者最新要求「15分鐘上下，不要超過太多或是太少」。本版20頁主講共4,051純漢字，比舊口語版減少930字，逐頁規劃14分45秒。按275字／分估計14.73分鐘；250–300字／分為13.50–16.20分鐘，尚未本人計時。至少12分鐘，不靠附錄、問答或等待補時間。

| 風格 | 新PowerPoint | 相同畫面的PDF備份 |
| --- | --- | --- |
| 白底學術 | [簡報](deliverables/臺灣房價預測_v22_白底學術_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_白底學術_投影備份.pdf) |
| 深色科技 | [簡報](deliverables/臺灣房價預測_v22_深色科技_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_深色科技_投影備份.pdf) |
| 暖色雜誌 | [簡報](deliverables/臺灣房價預測_v22_暖色雜誌_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_暖色雜誌_投影備份.pdf) |

三款都是同一個v22，選一款報告即可。20頁主講notes更新，3頁附錄保持。69張渲染與原r2相同、其餘OPC parts與圖表內嵌工作表保持原bytes。PDF沒有講者備忘稿。

先讀[完整口語講稿](content/完整口語講稿.md)和[每頁提示](content/上台口述提示.md)，再下載[離線排練HTML](deliverables/v22-15min-rehearsal.html)到電腦用瀏覽器開啟。GitHub原始碼頁不直接執行HTML。新版紀錄鍵與舊稿分開，20頁新稿與885秒計畫已核對。14–16分鐘只是工具的練習提示範圍，不是老師另定的硬門檻。工具沒有錄音或自動判斷內容完整。

[16題問答](content/排練與問答.md)與[四頁答辯卡](../spoken/defense/README.md)供補充，不算主講。[交付收據](delivery_receipt.json)、[文字核准](content/editor_review.json)、[瀏覽器檢查](browser_verification.json)和[審查及測試修正](review_and_attempts.json)說明驗證範圍。瀏覽器檢查使用模擬狀態，不能當成本人排練；尚未桌面PowerPoint或現場試播。

來源保存供追溯，這批prepare/build/verify腳本依原C:/NYCU_ML目錄與既有runtime，不是可直接從clone執行的新通用入口。既有scripts/rebuild_spoken_teaching.py仍重建舊4,981字口語版，不能宣稱已重建本版。完整13檔上台ZIP保留在原工作區outputs/v22_15min_20261001，未加入repo；repo已提供三款簡報、新稿、HTML及既有PDF/答辯卡連結。

Public MAE仍為1,998,568.5706元，Private未知。沒有新模型訓練、CSV或Kaggle操作，R106研究保持停止。此為文件與程式快照，不是完整重訓備份。
