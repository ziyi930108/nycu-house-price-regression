# R93 後續可行性：以欄位 self-attention 替換 prior NN

2026-09-29（Asia/Taipei）。這是只讀盤點與一個固定候選的建議，**不是採用決定**。接獲本任務時 R93 三期 GPU 回測仍在執行，本報告不判定它的成敗。未實作模型、安裝套件、啟動訓練或讀取新 query 標籤；外部閱讀限 ML 原始論文與官方程式，未接觸房價來源。v22 唯一提交及未知 Private 保持。

## 1. 既有實驗沒有涵蓋真正的欄位 self-attention

核對 R47–78 的 91 個 source 檔、保存計畫／報告及實際 trainer；另外搜尋整個 `src/*.py` 的 `MultiheadAttention`、`scaled_dot_product_attention`、`FTTransformer`、`TransformerEncoder`、self-attention，沒有命中。這是已保存程式範圍的結論，不證明未保存實驗不存在。

|保存方法|實際架構與差異|
|---|---|
|R47、R48、R52、R60、R63|[EmbeddingMLP](../src/round47_nn_train.py)：類別 lookup embedding 與數值串接，512／256／128 線性層、LayerNorm、ReLU；目標／prior／日期 loss 的變化沒有增加 Q/K/V 注意力。|
|R54、R56、R59；R62、R64、R65、R67、R72、R73、R74|數值 ramp、平滑／log-ratio 損失、prior／支持數／道路輸入或日期權重；仍屬上述 MLP 家族。|
|R57、R58|[TabM-style 實作](../src/round57_tabm_train.py)是 32 成員共享矩陣與逐成員乘法尺度的 BatchEnsemble，R58另加 ramp；不是讓欄位 token 互相 attention。|
|R49–51、R53、R55、R61、R66、R68–71、R75–78|主要是保存分支融合、消融、種子、配對或發布；沒有另建 FT-Transformer。|

因此，值得考慮的是**學習同一筆交易內不同欄位的條件交互**，例如用途改變面積／樓層的重要性；這是待驗機制，不是由既有誤差證明的原因。FT-Transformer 把每個數值／類別映射成向量，以 Transformer 處理，再由 CLS 預測；原論文也明示沒有對所有表格任務普遍最好的方法。[NeurIPS 原論文](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)

## 2. 唯一候選：小型 FT-style prior-ratio 回歸

先沿用 [R63 prior NN](../src/round63_recency_weighted_prior_train.py)的輸入與價格還原，只替換模型架構。建議正式計畫一次固定下列規格，不搜尋深度、token 寬度、seed 或比例。

- **輸入不重編。** 重用每期 [R47 encoder](../src/round47_nn_encode.py)及 R63 `prepared.npz`：`numeric[B,144]` 為 72 個 fit 統計標準化值＋72 個缺值旗標，`categorical[B,23]` 為 int64；詞表只留 fit 至少5筆的值，未知／稀有碼0。不把 R54的數值折線擴充（總通道依窗口為599／599／590） 或 R65 的額外 prior 輸入混進來。
- **168 個 token。** 144 個數值各用 `x_j*w_j+b_j`；23 個類別各用64維 lookup＋欄位 bias；再加一個可學習 CLS，得到 `[B,168,64]`。這是欄位 attention，不跨成交列，也不以 query 當鄰居。
- **固定 backbone。** 2 blocks、4 heads、每 head16維；PreNorm，第一個 attention 前省略首個 LayerNorm；兩層都做完整 token attention，不做壓縮。FFN 為64→256→ReGLU128→64；attention／FFN dropout .1、residual dropout 0，category dropout沿 .1。CLS 的 LayerNorm→ReLU→Linear→Softplus 輸出單一正 ratio。Tokenizer／CLS採官方64維初始化口徑，Q/K/V bias為0；head bias沿 fit ratio median 的 inverseSoftplus，沒有 query 初始化。
- **固定優化。** 沿 R63 seed42、AdamW lr .001／weight decay .0001、cosine 30 epochs 至1e-5，最後 epoch，float32／matmul high；不 warm-start、不 early stop。為控制記憶體，microbatch256、累積至原2048列才一步 optimizer；最後不足2048的組以實際列數除 loss。每個 microbatch 的加權絕對誤差先求和，再除該 optimizer 組總列數，不能各 microbatch 平均後再等權。推論固定256列一批。

線性數值 token、類別 bias、PreNorm首層例外及 ReGLU可對照[官方程式](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py)。以上小尺寸、既有編碼、完整末層 attention、Softplus頭與固定訓練方式是本題改編，**不是原論文完整復刻或官方預設**。目前只查閱上游，尚未下載／固定 revision；真正實作前應保存引用 revision、授權與 source SHA，不能依浮動 main 重現。由於 embedding 寬度、容量及 dropout 結構也改變，成功只能支持這套架構，不能將差額全歸因於 attention 單一機制。

## 3. OOF、日期權重與其他分支保持同窗

令 `f=A*P/1e6`。fit 的 P 必須重用同窗 [R60 五折建案 group-OOF prior](../src/round60_recency_price_prior.py)，每列不使用所屬 held fold 的價格；query 的 P 由該期 full-fit prior 預測。這個 OOF 是分案而非時間 OOF，但所有資料都在外層 cutoff 以前。不得用 fit 自身 full-fit prior替代 OOF，也不得套後期 encoder／prior／地理狀態。

訓練：`mean(saved_date32 * abs(Softplus(z)*saved_fit_factor32 - y_million32))`。日期權重沿 fit-only365日規則與原保存 float32，無每 batch 正規化，不再乘面積或 R93 建案權重。推論沿既有 float64還原：`float64(ratio)*float64(saved_query_factor32)*1e6`，factor只乘一次、沒有另加價格 cap。原 encoder 補值／尺度與詞表是 pure-X；OOF／full prior 使用 fit 價格，不能統稱無標籤前處理。

v22 的 NN 槽實際來自 [R71 seed42](round71_plan.md)。固定 F＝保存 R54 area NN，T＝新 attention prior NN，U＝原 unit LGB：`new_NN = spatial(reference(.75*(.5*F+.5*T)+.25*U))`。其餘完整 R46 log 分支、R79 unit 分支及外層三支 median 保留；R/S仍各一次，不替換整個 v22 或把 R93 一起加入。[選定模型綁定](../outputs/round79/plan.json)

最低驗證為原三期各一次新 NN fit，先凍結所有配置及來源。完整重播舊 R63 native→R71 NN→v22 median，核對 fi/qi、95欄encoder輸入、144／23陣列、OOF fold／prior／日期 SHA。先做合成 shape／finite／loss及累積梯度核對、改 query 不改 fit state、改 held fold y不改其 OOF、eval列順序／batch分割不改語义；再保存與重載全 query raw ratio和TWD。Query價格只進 driver評分，trainer輸入不含它；attention不得跨 batch列運作。

## 4. 成本與不值得試的反證

**已量測的舊成本**：保存 R63三期 fit 分別19.250／26.968／10.218秒，PyTorch allocated峰值301.08／405.52／231.67 MiB，來自各期 `outputs/round63/{stage}/model/training.json`。本次只讀GPU容量為 RTX4060Ti、16,380MiB；沒有測新架構 step time或峰值。舊數字不能當 attention 預估耗時。

**分析估計**：batch256、168tokens、4heads，一個 float32 attention矩陣為110.25MiB；一個 `[256,168,64]` tensor為10.5MiB。兩層反向还需 logits／probabilities／梯度、QKV／FFN、optimizer及資料，不能把110MiB當總需求。直接沿舊2048 batch會使單個矩陣升至882MiB；不應假定顯卡16GiB就一定足夠。按完整兩層估算，僅 dense＋QK/AV約2,099萬 MAC／列，三期696,671 fit列×30epoch約4.39×10^14 forward MAC，尚未計 backward／softmax等；這是公式估算，不是benchmark或ETA。先在GPU空閒時合成 smoke 量測峰值／step，才推估三期耗時；不可因 OOM 靜默改寬度或自動重試不同候選。

最強反證包括：

1. [R57](round57_report.md)與[R58](round58_report.md)已示範更複雜 NN 不保證有利：相對當時 v15三期皆退，R58亦沒有救回跨期表現；這不否定 attention，卻否定「新架構必勝」。
2. 新類別64維 lookup比原最多32維更大，可能加強建案記憶；數值缺值旗標分成獨立 token也增加運算。共用學習參數不等於已控制身份過擬合。
3. prior factor與大面積仍能放大 ratio錯誤，未解決OOF／full-prior分布差或極端總價。attention不是robust loss，更沒有未來趨勢外推保證。
4. 原論文回歸用MSE、驗證早停並調參，且明示注意力對欄位數有平方成本；我們是固定 weighted MAE／30epochs，不能移植論文改善幅度。保留原 .001 lr可清楚界定實驗，但也可能不是FT適合的優化設定。[論文方法與限制](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)

**建議有條件值得做一次，尚不值得採用。** 必须先接受實測資源成本並完成正式freeze，三期 overall及原weighted全勝v22、最新overall至少0.5%，另保留新案／高價／季度代價、native到median差異與兩對照24區間；有利才考慮R90新切點重建。未過即保留本固定配置失敗，不當場搜尋attention大小／比例。窗口反覆選型非盲測，bootstrap不代表Private勝率；此報告不授權final／新CSV或Kaggle操作。
