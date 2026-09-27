# 資料與模型取得

由原Kaggle課程取得train-v2.xlsx、valid-v2.xlsx、test-reindex-test-v2.2.xlsx及test-reindex-test-sample-submission-v2.csv。公開程式包不包含交易資料；其公開再散布權限未確認。

保存模型推論只需test和sample，SHA見config/v22.json。sample價格僅為格式範例，不用於模型計算。重訓另需train、valid及完整地理前置資料。

使用者限制為：禁止到內政部不動產成交案件實際資訊資料供應系統對答案，其他都可以。價格標籤與參考成交均來自課程資料。

## 地理資料

既有道路特徵取自OpenStreetMap／Geofabrik年度快照。© OpenStreetMap contributors。保存來源聲明見[evidence/data_external/geography/ATTRIBUTION.md](evidence/data_external/geography/ATTRIBUTION.md)，快照收據位於同目錄。道路代表點是粗略位置，並非精確門牌。

樹模型與NN使用R6地理特徵，最後空間修正使用R39地理特徵。本次沒有重新下載道路或取得成交答案。

## 保存模型及狀態

artifacts/manifest.json列出17檔、大小、SHA256及用途，共1,447,418,242 bytes。包含六模型、類別映射、NN encoder／quantile knots／價格先驗、參考成交及空間狀態、兩套地理與feature schema。

價格先驗與參考狀態衍生自課程價格，所以和模型一起保留於本機private/artifacts/，不放公開ZIP。保持原相對路徑，勿替換其他輪次狀態。取得合法副本後可移轉新目錄，不需要舊prediction.npy。

只載入經manifest核對的可信保存pickle。原工作區可用scripts/collect_artifacts.py複製；來源與目的SHA皆檢查，遇到不同內容不覆寫。

Public使用test約30%，Private使用其餘約70%；另附valid不是這個分割。
