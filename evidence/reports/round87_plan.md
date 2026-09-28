# R87：固定等權混合兩種面積表示的原生預測

2026-09-28。這是看過 R85 三期結果後立項的適應性研究。R85 中間、最新期 native MAE改善，但最終median三期退步；本輪固定混合不同表示，檢查互補是否能在完整管線留下收益，不聲稱獨立確認、盲測或防止選型偏差。

唯一候選，三期全部執行：

    mean_native = .5 * R79_native_TWD + .5 * R85_native_TWD
    base = .375 * R79_native_TWD + .375 * R85_native_TWD + .25 * old_LGB_unit_TWD
    ref = same_period_reference(base, trend)
    unit = same_period_spatial(base, ref)
    R87 = median(saved_R71_NN, saved_R46_log, unit)

真正計算base固定使用上面.375/.375/.25的左至右相加順序。mean_native只供單層診斷，不能以.75乘均值重新改運算順序來主張逐位等價。不平均已修正的reference/spatial分支，也不在median之後混合。

不搜尋比例、面積門檻、seed、價格群或城市規則。不新增fit、不載入或predict CatBoost/NN模型，不讀test標籤、不對test推論、不產CSV、不改selector、registry或v22選定。沿用同一期已保存的native TWD、fit/query canonical索引與reference/spatial狀態，各期先重播v22 base/reference/spatial/median全向量及完整metrics。

原R79/R85同窗索引、fit/query特徵、日期隔離、保存R63日期權重和source SHA須核對。train labels只用檢查原fit身分和最後評分，不估計新state。保留所有原始程式／CSV／研究證據SHA與user_selection.json。

主要對照v22/R79，強次對照v24/R83；另保留v11/v12/v5供通用報告重現。固定輸出三期完整總價MAE／加權／新舊案／季度／城市、native/base/reference/spatial指標、真價五帶和固定test組成估計。generic診斷只沿用既有test-X聚合權重，不讀test價格或新增test逐筆預測。配對比較各2000次建案群集及三月循環区塊，沿用seed20260925，對v22與v24的所有區間均呈現。

研究窗口已多輪重用，此候選還由R85結果啟發，選型偏差更須明示。配對bootstrap不涵蓋此適應性選擇、重訓隨機性和未來移轉，不能解讀為Private勝率。Private仍未知。成功／失敗都保存，是否推進另由整體、群組代價和不確定性審查，不使用Public搜尋參數。

執行前先保存本文和程式，再凍結SHA。輸出全部在outputs/round87，禁止覆寫既有輪次。本輪無昂貴GPU訓練；主執行緒仍單獨協調其他研究的GPU。
