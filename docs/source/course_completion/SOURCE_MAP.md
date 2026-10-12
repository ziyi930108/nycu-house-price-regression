# 章節證據對照

正文引用的數據均來自課程資料或保存的預測與收據。價格案例只來自已知真價的時間驗證資料，沒有test真價，也沒有以本次分析改動模型或提交價格。

| 引用 | 來源 | 對照內容 |
| --- | --- | --- |
| [1] | [資料集合與去重](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/model_common.py)、[特徵解析](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/house_features.py) | 3列表頭、529筆去重、跨集合28筆去重、特徵轉換 |
| [2] | [神經網路編碼](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/round47_nn_encode.py)、[參考單價](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/round60_recency_price_prior.py)、[固定訓練](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/round107_weighted_smooth_train.py) | fit限定、分組OOF、日期權重、SmoothL1、固定30輪 |
| [3] | [三年度結果](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/evidence/outputs/round107/results.json)、[額外起點](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/evidence/outputs/round113/results.json) | 各期MAE、分群退步、重抽樣與歷史不利證據 |
| [4] | 原Word內嵌的[單一網路曲線](assets/training_curve.png) | 原圖bytes保持；另行固定診斷，不是原提交軌跡或完整融合 |
| [5] | 已選定提交、使用者公開分數回報的既有收據與本次截圖及登入Kaggle頁面核對 | Public 1,995,836.2136元，原由使用者回報，本次已核對平台頁面；尚未核對平台上傳檔案bytes，Private未知 |
| [6] | [最終預測組合](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/research/src/round107_final_candidate.py)、[發行收據](https://github.com/ziyi930108/nycu-house-price-regression/blob/main/evidence/outputs/round107/final/release.json) | 僅新訓練一個參考單價網路，其餘五模型沿用保存向量；CSV身分與重現界線 |
| [7] | [城市統計](assets/city_statistics.json) | 清理後train 455,255筆，七城市圖、均值、中位數、四分位與全部23個標籤類別 |
| [8] | [完整驗證公開摘要](assets/validation_report_summary.json)、[分析核對](assets/analysis_receipt.json)、[繪圖核對](assets/plot_rendering.json) | 封存列序、時間範圍、133,697筆完整融合誤差、分群、尾端貢獻、匿名案例與64×64密度圖 |

本次分析使用經SHA核對的原始資料與特徵快取；年度fit/query列序與原封存收據一致，兩者互斥，時間切點吻合，重新計算MAE與既有保存結果一致。目前Excel的容器bytes不同於歷史，但主執行緒另核對13,217,736格內容與順序，未發現內容差異；報告分析的實際輸入仍是封存快取。

各價格帶採左含右不含；首帶未滿500萬元，最後一帶為5,000萬元以上。案例A/B/C分別對應絕對誤差排序第1/3/4名，不是隨機樣本。新建案按當期fit是否出現同城市、行政區與建案名稱定義，與屋齡不同。原因解釋是待驗證假設。

同案130筆／同型店面0筆是描述性盤點，不等於實際reference支持量。現有參考已按建物類型與主要用途分組；下一步是核對各層實際支持，測試既有分組內相近面積／車位條件與支持不足回退，並在訓練前固定門檻。這些步驟尚未執行，不把既有用途分組當作新增功能。

上方GitHub為既有私人儲存庫，需存取權限。此文件包不含原始交易表、私有模型、逐筆驗證價格或提交CSV；保存摘要與文件重建來源不等於全部六模型可在乾淨環境重訓。
