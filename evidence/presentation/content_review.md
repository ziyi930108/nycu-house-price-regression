# v22 單一提交模型內容獨立審查

審查日期：2026-09-28（Asia/Taipei）。本輪只做內容 QA，不修改簡報／報告、模型、CSV、selector，不啟動訓練。

## 審查範圍與結果

- 已閱讀新 build_deck.mjs、source_data.json、逐頁講稿、REPORT.md，並直接解析最終 PPTX 的 15 頁 XML 文字與四張圖表快取值。
- 15 頁＝13 頁主講＋2 頁附錄，講稿主講合計 545 秒。
- 最終 PPTX：outputs/presentation_v22_focused_20260928/deliverables/臺灣房價預測_單一模型正式報告_v22_定稿.pptx
- PPTX SHA256：a12a2396e8e2230231fac493ce27af287b8630b13950b00f4100c2e725849a5e
- source_data.json SHA256：077580234e9faea9bea480f2b0eb076db78f1813293a1effb028800f584bf90f
- REPORT.md SHA256：ee613c7cd550c7cc0c79a1bcc9f8aa5b20d20ed2adc5da66e524ff9b7fb9989e

## 內容核驗通過

1. 觀眾可見主文及逐頁講稿的口述段落不含 v20、v24、R79 比較敘事；研究來源路徑保留原始編號，不要求觀眾理解版本沿革。v22 只作選定檔案身分。
2. MAE 定義為平均絕對總價誤差，單位元；Public 1,998,568.5706 元約為 199.86 萬元，未冒稱準確率或每筆固定誤差。
3. Public 約 30%、Private 約 70% 明示切自 test，與另附 valid 不同；Private 未知。
4. 歷史三期數值核對為 [2309352.35769657, 1407387.070805849, 1383228.4054499655] 元，圖表換成 230.94、140.74、138.32 萬元。不同年度為不同交易，頁面及講稿都不把下降解釋成逐年改版進步，也不混算 Public。
5. 1200／1320／1900 萬元與中位數 1320 萬明確標為假設例子；未當作模型成績或真實交易。共同偏誤限制有保留。
6. 第 6 頁權重說明限定單價 CatBoost。審查曾指出原題目只有「單價訓練」易含混，製作代理已改為「單價 CatBoost 提高近期成交權重」，最終 PPTX 再核對通過。沒有將此日期×面積權重宣稱為六模型共用。
7. 模型數量為三個 CatBoost、一個共用 LightGBM、兩個 NN，共六模型、三分支；附錄先還原元、逐分支 R/S 後 median，公式與實作一致。
8. 三期 fit/query 日期與列數正確，回測不混 valid；歷史窗口反覆參與選型、不是新盲測，bootstrap 不保證 Private。
9. 47,073 筆最大差 0／完整 CSV SHA 一致是保存模型推論重現；84.74 秒指同機既有環境入口執行，不是訓練時間。全新安裝、從零重訓、其他硬體及地理重建未驗證。
10. README、AGENTS、HANDOFF、config、最新 user_selection.json 在選定身分上一致：使用者只提交 v22、Private 未知、未代操作 Kaggle 勾選。REPORT 主文以方法敘事，內部研究比較與最強反證保留於附錄，沒有宣稱已知 Private 最佳。

## 發布整合仍需同步

審查當下，repo README.md 第 15、71 行及 HANDOFF.md 第 17 行仍記錄上一份 16 頁簡報（13＋3）；新定稿是 15 頁（13＋2）。root 已獲通知並負責在發布前更新入口、連結與快照。這是整合待辦，不是本內容審查代為修改。545 秒不變。

本審查是內容／數字／狀態一致性核驗，不代替製作代理的逐頁影像排版 QA，也未在原生 PowerPoint 開啟驗證。

## 最終交付複核追加

核驗時間 UTC：2026-09-27T17:36:02.963577+00:00。以上保留的是前一修訂版審查過程；以下為唯一最終交付身分與整合結果。

- 唯一交付簡報：outputs/presentation_v22_focused_20260928/deliverables/臺灣房價預測_v22_上台報告.pptx
- 最終 PPTX SHA256：8f930bc12dadcff0a399b5e4c3f58930f0d906d8f4313454efeb1fc3ec561e81
- 最終 source_data.json SHA256：d6156c4069ec57fbacb14af7a9391ed13e32e6d8bfe6bf72c66b0e2cd18adbdc
- 直接比對前一修訂版與最終版的 15 頁 XML 可見文字，全部相同；主文仍沒有 v20、v24、R79。第 6 頁明示單價 CatBoost 的權重設定。
- 第 2 頁來源已正確指向 REPORT.md 第 1 節；source_data.json 與 PPTX 內建備忘稿兩處皆已核對。
- repo README 第 15、71 行與 HANDOFF 第 17 行已更新為 15 頁／13 主講＋2 附錄，主講 545 秒。前段頁數整合待辦已解決。
- repo 發布用 presentation/臺灣房價預測_正式報告_v22.pptx 與最終 PPTX 的 SHA 一致：True。

最終內容核验通過。本輪不更動模型或提交數值，不增加敘事修改要求。
