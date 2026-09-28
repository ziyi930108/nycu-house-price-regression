# R88 固定CPU消融：無局部支持時回到各分支base

日期：2026-09-28。主執行緒授權固定第二類後處理假設。本計畫在R88任何候選計分前保存；本研究源於已見誤差盤點，adaptive_after_error_audit=true，不冒稱盲測。

## 唯一政策改動

保留v22所有原模型與三分支。對每個分支逐筆判定：

    mask = (reference_diagnostics.source == 'base_fallback') & (~spatial_diagnostics.applied)
    candidate_branch = where(mask, saved_unadjusted_branch_base, original_v22_branch_spatial)
    R88 = median(candidate_NN, candidate_log, candidate_unit)

不挑交易類型、真價、ID、城市、日期或修正比例。mask只取同窗fit建立的reference/spatial與query特徵，不使用query標籤。只對無reference局部支持且未套spatial的列撤銷共享全域時間趨勢及相關修正；其餘價格逐位不變。未修正base仍含各原模型既有時間特徵／權重，不能說完全不使用時間資訊。

動機：已見描述性盤點指出reference fallback仍會套全域趨勢，可能使無局部支持的物件偏移。最強反論：普漲期這個趨勢可能是必要外推；撤銷會低估一般新交易。此政策涵蓋所有符合條件的交易，不只盤點中較不利的廠辦。

## 固定三期與控制

沿原canonical train、R79固定development／confirmation／auxiliary三個年度，不加入valid。fit/query索引與日期、R6/R39地理及fit-only reference/spatial state沿原SHA核對；不重新fit任何物件。

每期逐分支重播predict_reference及spatial.predict，與保存的base/reference/spatial、完整diagnostic及mask精確核對，先復現原v22三分支median，再計算唯一R88。unit取R79；NN取R71 seed42；log取R46。新政策是純函數，人工合成全mask／無mask／混合mask與輸入不變檢查需通過。

三期全部完成，不依早期分數換mask或中止不利結果。v22為主要對照，v24次要；另保存通用診斷所需v11/v12/v5。完整組合評分為TWD MAE，沿原新舊建案固定加權、季度、城市及五價帶與test組成支持估計。固定建案／三月循環月份區塊bootstrap各2000次、seed20260925；兩對照各12區間全部保留。

## 保存、界線與決策

0次新fit、0次原生模型load/predict、0次新state build、不讀test資料、不產CSV、不操作Kaggle／selector。只載合法保存的歷史分支向量與reference/spatial狀態、回測X/y。保存計畫、input/source SHA、各分支mask／候選／診斷、全量結果與真實CPU秒數。原模型、CSV與舊decision保持不變；失敗不覆蓋。

新輸出outputs/round88，來源src/round88_experiment.py。候選完成後，報告完整MAE與群組代價、支持與反證、兩對照配對區間，寫decision_review.json；沒有自動採用或final export。年度窗口反覆用過，bootstrap未校正研究者選型，不能稱Private勝率。Private未知。

本計畫沒有設定或搜尋任何比例、閾值、價格帽或query真價分流。
