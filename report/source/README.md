# 報告製作來源

現行正式稿為2026-10-19定稿。已核對的Word快照保存在[final_20261019](final_20261019/source.json)，以明確SHA檢查保護；上層Word與PDF均直接複製已審閱版本，沒有重新排版或重產PDF。

在儲存庫根目錄執行以下命令，輸出目錄必須尚不存在。匯出程式只用Python標準函式庫：

```sh
python report/source/final_20261019/export_report.py --output-dir ../house-report-export
```

程式先核對定稿Word與五張圖的SHA，再複製Word，依Word本文順序匯出Markdown與13張表、5張圖，並保存匯出收據。圖片使用原圖，表內換行保留為HTML換行；`<NA>`等文字以HTML實體呈現，避免Markdown把它當成標籤。輸出已有目錄時會停止。這是文件匯出，不會訓練模型、執行房價推論或重新產生PDF。

正式PDF沿用同一Word的Microsoft Word原生匯出，原18頁視覺審閱證據摘要及來源雜湊記在[source.json](final_20261019/source.json)。本次只有原檔複製與文字匯出，沒有宣稱重新渲染或重新目視。其他Office版本或字型可能重新分頁。

原`build_report.py`、`reference_report.docx`與`reference_report.md`保留為2026-10-14歷史製作來源，包含原SHA及拒絕覆寫保護，沒有改成接受新來源。該舊程式只刪除6.4公開排行榜小節，不能產出現行定稿；現行入口以上面的`final_20261019/export_report.py`為準。

`assets/city_statistics.json`與`assets/validation_summary.json`只提供城市及驗證結果的彙總數據。後者省略逐筆案例列表與本機核對路徑，保留原摘要SHA供追溯。五張PNG的雜湊與現行Word內嵌圖片完全相同。公開內容不含原始交易表、模型權重、提交CSV或逐筆預測清單；沒有複製本機渲染程式及QA草稿。
