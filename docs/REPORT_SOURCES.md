# 章節證據對照

報告中的[1]至[6]對應下列來源。程式與證據檔保留歷史檔名以便追查；正文聚焦最終提交方法。

| 報告引用 | 來源 | 對照重點 |
| --- | --- | --- |
| [1] | [資料讀取](../research/src/prepare_data.py)、[欄位與去重](../research/src/model_common.py)、[文字解析](../research/src/house_features.py) | 3列表頭、529筆完全去重、28筆跨集合去重及文字轉換 |
| [2] | [方法與程式明細](../evidence/reports/v25_method_and_code_20261010.md)、[神經網路編碼](../research/src/round47_nn_encode.py)、[日期權重與參考單價](../research/src/round60_recency_price_prior.py)、[固定SmoothL1訓練](../research/src/round107_weighted_smooth_train.py) | 六模型、fit限定、OOF、先驗、損失及固定設定 |
| [3] | [三年度結果](../evidence/outputs/round107/results.json)、[額外起點](../evidence/outputs/round113/results.json)、[比較及反證](../evidence/reports/round107_108_report.md) | MAE、分群、門檻與bootstrap界線 |
| [4] | [曲線方法](../evidence/reports/v25_training_curve_20261010.md)、[60點原始數值](../evidence/outputs/v25_training_curve_20261010/curve.json)、[保存預測核對](../evidence/outputs/v25_training_curve_20261010/curve_vector_review.json) | 另行固定30輪單一NN診斷，未挑輪數 |
| [5] | [使用者選定](../evidence/selection/user_selected_v25_20261010.json)、[使用者回報Public](../evidence/kaggle_feedback/public_v25_user_report_20261010.json) | Public回報與未知Private |
| [6] | [最終設定](../evidence/outputs/round107/final/model/training.json)、[發行核對](../evidence/outputs/round107/final/release.json)、[重現界線](V25_REPRODUCIBILITY_20261010.md) | 一次新NN fit、保存向量重播、CSV身分與尚未完成範圍 |

封存原文的連結依當時工作區路徑保存，部分不能在縮減後repo直接開啟；上表直接連結本repo實際保留的檔案。課程交易表、模型權重及逐筆價格未公開。
