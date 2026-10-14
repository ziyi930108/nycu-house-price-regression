# 報告製作來源

此資料夾保留正式報告的文件重建程式、必要參考文件、五張圖表及彙總數據。重建不需要原始交易表或模型權重，也不會訓練模型或重新估價。

在儲存庫根目錄執行，先安裝Python套件lxml，再選擇尚不存在報告檔案的輸出資料夾：

```sh
python -m pip install lxml
python report/source/build_report.py --output-dir ../house-report-rebuild
```

程式以參考Word與Markdown建立正式稿，移除不列入本文的公開排行榜小節，保留其他內容與13張表、5張圖。參考檔SHA檢查與拒絕覆寫保護保留，輸出已有正式報告時會停止。`reference_report.docx`與`reference_report.md`是程式使用的重建輸入；閱讀正式內容請開啟上層Word、PDF或`REPORT.md`。

重建輸出包括Word、Markdown與建立收據；不會重新產生PDF。正式PDF沿用已審閱的Word原生匯出檔。其他Office版本或字型可能重新分頁。這是文件重建，不代表已從原始交易表重訓全部六個模型。

`assets/city_statistics.json`與`assets/validation_summary.json`只提供城市及驗證結果的彙總數據。後者省略逐筆案例列表與本機核對路徑，保留原摘要SHA供追溯。五張圖表PNG沿用正式報告來源。未複製個人電腦專用的渲染程式、QA草稿或內部修訂日誌。
