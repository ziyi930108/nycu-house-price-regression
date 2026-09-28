# R90新切點NN準備介面：實作與核驗

日期：2026-09-29（Asia/Taipei）。這是之後新切點可用的程式基礎設施；本輪沒有真正新切點fit、沒有查詢標籤、沒有讀test，也沒有因R89被否決而啟動六模型補測。

## API與索引契約

程式：[src/round90_offset_nn.py](../src/round90_offset_nn.py)。

`prepare_nn(Xfit, yfit, Gtreefit, Xq, Gtreeq, source_indices, out)`只用傳入fit資料重建encoder／ramp／價格先驗，保存兩個NPZ和JSON-safe prepared dict。X為原87欄，geo為原R6八欄，合計95欄、23個類別。Gtree不是R39的spatial geometry。

**source_indices必須傳OffsetData.fit_indices，即原canonical train矩陣中的位置。** 不可傳`fit_source_cache_indices`，後者僅用於從raw cache抽取對齊的y；也不可傳reset後的`0..n_fit-1`。R60對缺建案、缺地址／道路列的singleton key使用這個canonical位置，傳错會改變OOF fold。prepare保存完整原vector、values SHA和`source_indices_contract`；主driver仍須把此SHA與固定split manifest核對。helper無法只從整數vector自行判斷其真實來源軸。

以下僅是未執行的呼叫範例：

```python
prepared = nn.prepare_nn(
    data.Xfit, yfit, data.tree_fit,
    data.Xquery, data.tree_query,
    data.fit_indices,  # canonical train positions
    "outputs/round90/nn_preparation",
)
assert prepared["input_sha256"]["source_indices"] == nn.values_sha(data.fit_indices)
# 只有主執行緒取得單張GPU排程並確定本輪研究後才呼叫：
# result = nn.train_nn(prepared, "outputs/round90/nn_training")
```

prepared中主要路徑為`area_npz`、`prior_npz`、`saved_date_weights`、`encoder_state`、`ramp_state`、`prior_state`、`prior_evidence`、`train_oof_unit_prior`、`query_unit_prior`、`fold_ids`、`source_indices`及`date_weights_metadata`。皆為workspace-relative path；完整逐檔SHA、每array shape/dtype/value SHA、原frame／fit y／canonical index身分同時保存。回傳dict為detached JSON copy。

## 原v22兩NN合約

| 元件 | 新切點state與輸入 | 目標與損失 | 原trainer |
| --- | --- | --- | --- |
| N_area | 新R47 fit-only encoder；新R54數值quantile ramps；原144欄完整保留為prefix | positive unit × float32(A/100)形成百萬元總價；uniform MAE | R48 |
| N_prior | 同一新R47 encoder的原144數值欄；新R60 grouped OOF/full-fit unit prior | positive ratio × float32(A×prior/1e6)形成百萬元總價；fit-date weighted MAE | R63 |

N_area沒有date sample weights，不能誤換成R74模型。N_prior訓練先驗排除完整held project/fallback group；query使用相同fit窗口全部標籤建立的full prior。這是五折分組OOF，**不是時間OOF**。query不參與encoder/ramp vocabulary／stats，也不提供label或先驗日期權重。

date weights先按原R60 fit-only365日公式在float64計算，再cast float32，不二次正規化；無效日期維持1。area與prior factor皆依原順序在float64完成公式再cast float32。Trainer反轉總價時依保存的float32 factor轉回float64乘原生輸出，不能另用double面積重算替換。

兩trainer均固定seed42、30epochs、batch2048及原R47架構／optimizer。helper鎖定原R47/R48/R52/R54/R60/R63 source SHA；configuration以原R47 literal加固定元件設定構成，已和實際R48／R63 import後完整CONFIG逐項比對一致，沒有呼叫setup、初始化模型或fit。

## 執行與保存邊界

`train_nn(prepared,out)`是唯一會啟動GPU的介面，必須由主driver明確呼叫。它使用`.venv_torch/Scripts/python.exe`串行執行原R48與R63；沒有自動重試、選epoch或併行GPU。新out須空白，既有產物不覆寫；失敗保存stdout與execution receipt。

預期完成後回傳`N_area`、`N_prior`兩個float64 TWD vector及receipt。原trainer本身會保存model.pt，再完整重載全query，要求逐位相同；wrapper再核對該收據、全CONFIG、source/NPZ/factor/weight SHA、30個epoch、row/schema、正有限TWD向量，保存模型／預測hash。這項完整模型重載是**實際train_nn執行時才會完成的檢查**，本次CPU合成檢查沒有假稱它已執行。

## 已做的核驗及限制

[合成收據](round90_nn_selfcheck.json)保存11項檢查：360筆合成fit、7筆合成query、noncontiguous canonical indices，兩次synthetic prepare及一個held-fold標籤擾動。包括query日期／面積／類別擾動不改fit states/weights/OOF、held-fold labels不進自身prior、invalid date精確1、unknown category歸零、ramp prefix、NPZ／JSON重載及輸入不變。合成area寬1137、prior寬144；area寬度由fit knots決定，真正資料不保證1137。

最終測得selfcheck內部0.562秒、外層0.578秒，第一次prepare0.219秒；first prepare包含在總時間，不能相加。一次工程問題為結果JSON序列化遇到numpy.bool_，數值checks已通過；只將check值轉為Python bool後修正，沒有更改數學規則。記錄保留在收據。

未完成的部分：真正cutoff資料prep、兩次native NN訓練與其全query model.pt重載、和其他四native模型整合、全新環境重訓。此helper不決定新候選、不計算MAE、不寫Kaggle CSV。

## 凍結身分

- helper SHA-256：`b473b246925dffe3d084c909679892f45e33f19ed0a6c18c100cfc836bc35e06`
- selfcheck receipt SHA-256：`9f59e162c91eba40128d6cb8a5a66d1f322c0633404eddc00039f6028dd6fb30`
- R89結果另見[分群趨勢稽核](round89_group_trend_audit.md)；R89未通過，不做全量fit／CSV。
