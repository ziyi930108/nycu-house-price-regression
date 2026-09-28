# R82 後續方向的有限來源盤點

本檔只記錄本機程式與已完成報告的盤點。尚未選定下一個實驗，未產生新預測、訓練、分數或提交檔。R82 的採用決定另行保存，不能把以下想法當成已證明的改良。

R82 在三個歷史窗口的 unit reference／spatial 分支都比同一 base 的 trend 差；新建案的最終 MAE 也三期都差。這使「直接移除額外全域趨勢」缺乏支持，但不證明原趨勢係數最佳，也不能推論所有 reference 變動都無效。

## 已核對的來源

- `src/round2_neighborhood.py`：confidence 由來源層級、距離、局部離散程度決定；不是 base 與 local price 分歧的函數。ratio clip 保留原 .65–1.5 界限。
- `src/round8_spatial_reference.py`：額外空間殘差乘 `(1-confidence)`，因此更動 reference confidence 也會改變空間補充的有效權重。
- `src/round45_regional.py`、R45 計畫：改區域趨勢，並非目前 unit-only confidence 縮減。
- R5 計畫：遮罩部分建案識別欄、重新訓練，原 reference 不改；不是信任權重調整。
- R25 計畫的 anchored 指地址解析，與價格 anchor 消融不同。
- R67 計畫：向 NN 新增 prior 支持列數特徵，原 reference／spatial 不改；不是 reference confidence gate。
- `src/round2_lightgbm.py` 有早期 reference 與預測的融合，不應宣稱一切參考價格縮減均未曾嘗試。

本次以 confidence、reference weight、ratio bounds、乘法縮減等字串盤點保存來源與研究計畫，未見目前單價分支以固定倍率直接縮減 reference confidence 的已完成候選。這是有界搜尋，不是對未保存工作或所有等價數學形式的完整證明。

## 尚未選定的假設

可考慮保留原未來趨勢、只改局部參考的信任強度；這與 R82 去掉 base 趨勢不同。若採用倍率 rho，令 Bf 是原 future base、P 是原 trend reference，則 `P_rho=(1-rho)*Bf+rho*P`，對應 `c_rho=rho*c`。若將新 confidence 傳給既有 spatial，空間殘差也會變成 `(1-rho*c)*alpha*(G-Bf)`；不能聲稱 spatial 有效權重完全沒變，更不能只改價格而把診斷說成已同步修改。

反對理由：現有 reference 明顯降低單價 base 的 MAE。R81 相對舊 base 的改善被部分抵銷，不代表 reference 本身有害；縮減可能失去有用的同案／同路資訊。任何固定倍率也是一次研究選擇，不能以「沒有網格搜索」宣稱消除過擬合。若未來執行，須先固定一個配方、直接對照與完整三期檢查，並如實保留共同歷史窗口的選型偏差。

目前也在另行盤點 unit CatBoost 的 seed 穩定性；此檔沒有在上述方向間作選擇。

## 單價 CatBoost 種子盤點已完成

獨立來源盤點未找到 R19／R79 單價 CatBoost 的 seed137、2026 或雙種子平均候選；root 另核對原 helper 的固定 random_seed=42 檢查。R24 已做 raw 目標、uniform 權重的 42／137／2026 三種子平均，development 約持平，當時依舊規則停止後兩期；這不是目前面積乘日期權重的 unit 分支。R69 則是 prior NN 的配對種子驗證，不是單價樹模型。不能把這些研究說成從未做過 seed ensemble，也不能把它們當成当前 unit ensemble 已驗證。

證據：[R19 helper](../src/round19_unit_geo.py)、[R79 helper](../src/round79_recency_unit.py)、[R24 報告](round24_report.md)、[R69 報告](round69_report.md)。

一個尚未立項的候選是固定保存 R79 seed42，三期各新增一次 seed137，其餘訓練設定與保存日期權重不變；兩支先各自轉為非負 TWD，再等權平均，沿 v23 純 native 單價 base 的同一 reference／spatial 與固定 NN／log median。主對照需為 v23，v22 作選版參考；不能看單支結果後挑種子、改權重或追加種子。

最強反對理由：尚無證據說種子方差是主要誤差來源。同資料、面積轉換、日期權重及時間外推的共同偏差不會因平均消失，外層 median 也可能抵消原生分支改善。兩種子比較不足以證明 Private 穩定性，還會增加三次 fit 的運算成本。此處只是待評估方向，尚未選為下一實驗，也未產生新預測／分數。
