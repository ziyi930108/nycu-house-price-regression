# 第六輪：離線地理特徵

**固定回測已接受，release 收據與新 CSV SHA 已核對；推薦本輪 submission_v3.csv。新提交的 Kaggle Public／Private 尚未測得。**

現行推薦為 [本輪新提交 submission_v3.csv](outputs/submission_v3.csv)。已核對 release 收據、接受決策與 47,073 列中繼資料及 CSV SHA；本報告不重新計算或解析價格。新提交的 Kaggle Public／Private 尚未測得。

## 結果與紀錄

- [完整報告與各組／季度結果](reports/optimization_round6_report.md)
- [機器可讀結果及收據來源](reports/round6_results.json)
- [資料處理紀錄](reports/round6_preprocessing_record.md)
- [操作紀錄](reports/round6_work_log.md)
- [驗證規約](reports/round6_validation_plan.md)
- [不可變正式計畫](outputs/round6/candidate_plan.json)

| 階段 | 已保存狀態 |
|---|---|
| development | 已評估；開發通過 |
| confirmation | 已評估；保存的門檻通過 |
| auxiliary | 已評估；保存的門檻通過 |

## 固定方法與報告重現

唯一新增 GPU raw CatBoost 使用 87 原欄＋8 地理欄，95 欄／23 類別；2,999 輪、seed 42、4 threads、總價除以 1e6、均等權重，沒有 eval_set 或 early stopping。固定融合為原 raw 37.5%、地理 raw 37.5%、舊單價 LightGBM 還原總價 25%，再套原 87 欄 reference 一次 trend，沒有搜尋參數、格網或比例。

離線 OSM／Geofabrik 不含外部成交價格。development／confirmation／auxiliary／正式 test 各指定 2022／2023／2021／2024 快照；實際 PBF 時間以 manifest 為準。道路／平面交點是粗略位置。

在專案根目錄重生三份報告（Python 標準函式庫，不載入模型或資料陣列）：

```powershell
.\.venv\Scripts\python.exe src\round6_report.py
```

程式核對 plan、evaluation、freeze、acceptance 與地理 manifest 的 JSON 收據及身分後覆寫報告；沒有評估 JSON 的年度明列未執行，acceptance 缺失時保持尚待驗證。完整實驗另依固定計畫於獨立工作副本重建，不在既有封存輸出上重訓。

計畫 SHA256：`606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8`。v2 Public MAE 2,883,250.4138 元僅為歷史提交紀錄；第六輪 Public、Private 未知。

歷史標籤已曾使用，不能稱這些窗口完全未見；固定單候選不能消除以前的選型偏差。只轉述已保存決策，Private 改善未獲證明。沒有到內政部成交資料對答案。
