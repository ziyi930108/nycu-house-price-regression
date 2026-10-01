# v22 白話口語講稿版

2026-10-01依使用者要求改寫20頁主講。三款畫面及3頁附錄保持，60份主講備忘稿同步新稿。

- [A白底學術](deliverables/臺灣房價預測_v22_白底學術_口語講稿版.pptx)
- [B深色科技](deliverables/臺灣房價預測_v22_深色科技_口語講稿版.pptx)
- [C暖色雜誌](deliverables/臺灣房價預測_v22_暖色雜誌_口語講稿版.pptx)
- [完整口語講稿](content/完整口語講稿.md)、[每頁口述提示](content/上台口述提示.md)、[排練與16題問答](content/排練與問答.md)
- [原PDF投影備份](../projection/README.md)：畫面相同，PDF沒有講者備忘稿。

4,981純漢字，250–300字／分鐘估計16.60–19.92分鐘，逐頁規劃17:20；附錄與問答不計入，尚未真人计時。每次先講一個意思，再接術語或例子，分段方便換氣。第17頁精確Public分數留在畫面，口述接近兩百萬元，說清平均誤差及Private未知。使用者尚未選視覺風格。

## 修改與證據

[內容收據](content/revision_receipt.json)與[交付收據](delivery_receipt.json)保存字數、原件SHA及核查。三款69頁PNG與原r2完全相同，表格、圖表、內嵌工作表及媒體保持。20頁主講notes更新，3頁附錄與16題問答答案保持。原稿與提案另存，不覆寫；author_notes.md字數4,969是提案階段，最終approved_scripts.json為4,981字。

第一次A轉存遺失內嵌工作表，被原驗證拒絕，失敗收據保留。修復只複製Artifact產生的20份notes XML回原套件，其餘原件逐byte保持，沒有改成圖表圖片或重新產生工作表。三款都重新通過相同門檻，沒有放寬檢查。

## 程式範圍

本目錄來源是原C:/NYCU_ML製作快照，含既有路徑及拒絕覆寫保護，不能在此副本直接執行。需要原r2資料與Codex bundled Node／Python、@oai/artifact-tool、Presentations技能工具。新增[口語版重建入口](../../../docs/SPOKEN_TEACHING_REBUILD.md)，可指定runtime／repo，在新目錄生成三款口語版；從已發布clone的11項輸入實测，3PPTX及69張PNG完全相同。以下保存的歷史source仍依原布局，請執行新入口。原[簡報重建入口](../../../docs/TEACHING_REBUILD.md)仍產生r2。完整渲染、十檔上台ZIP、失敗候選及發布收據留在原工作區。

本次沒有新ML fit、CSV或Kaggle操作，Public仍1,998,568.5706元、Private未知，v22唯一提交且R106保持停止。已驗證保存模型推論，未驗證全新安裝／從零重訓；本包不是完整重訓備份。尚未桌面PowerPoint／投影／真人計時。
