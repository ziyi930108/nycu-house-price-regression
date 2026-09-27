# 專案接續規則

先讀 README.md、HANDOFF.md、REPORT.md、docs/REPRODUCIBILITY.md，並依使用者最新要求接續，不因文件列了研究方向便自行啟動訓練。

- 專案是房屋總價回歸，以元的MAE評分，越低越好。
- 嚴禁到內政部不動產成交案件實際資訊資料供應系統核對競賽答案。用課程資料fit的價格先驗與參考成交必須與query隔離。
- v22是本次交付主要候選；Private未知，最終版本未定。Public來自保存截圖回報。歷史資料夾內的「最佳」「待執行」可能是當時狀態。
- 保留不利證據：v24在三期整體回測優於v22；歷史窗口反覆用於選型；部分高價群退步。
- 保存模型推論已全量實測，不等同從零重訓或全新環境安裝已通過。
- 不覆寫舊模型、CSV、SHA及失敗收據，不按test未知真價或ID人為調價。
- private/、data/、runs/不可加入Git。從原工作區取得必要artifact時，按manifest核對。
- 修改程式後執行相應測試；修改數值計算須重跑相關推論。文件改動不需重新訓練。
- 有實質進度更新HANDOFF、README、REPORT/簡報及相應收據，按照docs/PUBLISHING.md刷新SOURCE_MANIFEST。
- 私人repo發布授權只涵蓋本專案交付。改為公開、發布資料與模型或Kaggle最終勾選需遵從使用者明確指示。
