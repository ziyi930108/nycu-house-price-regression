# 正式報告現況更新與重建

2026-10-02。[12頁正式PDF](臺灣房價預測_v22_正式報告_20261002.pdf)與[root REPORT.md](../REPORT.md)現在包含同機新套件環境的推論實測。摘要與結論不再把這項工作列為未驗證；8.2保留初次測試，8.3另記新環境的76.930302秒及CSV一致性。沒有以單次時間差宣稱提速，也沒有把六個流程向量當六份獨立泛化證據。

資料處理、模型、年度分數、Public、不利群組及附錄歷史比較保持；Private未知、另一台電腦／地理重建／從零重訓未驗證。舊11頁PDF與來源保留在原teaching及docs位置，作當時紀錄；繳交請用上方12頁PDF。三款主簡報、講稿和影片均不改。

## 重建PDF

已有ReportLab與pypdf的Windows環境、微軟正黑體字型。使用同一Python安裝與執行，例如先把`$reportPython`設成你準備的報告環境Python絕對路徑；本次沿用既有報告套件，未驗證新機安裝。

```powershell
& $reportPython scripts/build_report_pdf_current.py --input REPORT.md --output runs/report_new/正式報告.pdf
```

輸出及同名.build.json都必須不存在。新版builder保留負號修正，另把連續章節／小節標題與第一張表格保持同頁。PDF生成後須渲染檢查；[本次收據](../evidence/formal_report_update_20261002/final_receipt.json)記載主線逐頁看12張圖，另有[內容審查](../evidence/formal_report_update_20261002/content_review.json)及[生成收據](../evidence/formal_report_update_20261002/build.json)。內容審查不冒稱獨立視看。

[新GPU實測說明](CLEAN_GPU_REPLAY_20261002.md)與原重現收據不回寫。文件重建沒有載模型、跑推論或訓練，也沒有重建其他媒體。
