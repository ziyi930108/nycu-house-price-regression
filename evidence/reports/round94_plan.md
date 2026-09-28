# R94 固定計畫：欄位 Attention prior-ratio 回歸

2026-09-29（Asia/Taipei）。在任何本輪真實模型 fit／query 評分前固定。R93 已完成且未過門檻；這是新的單一架構候選，不把 R93 權重一起納入。唯一提交仍為 v22；Private 未知。

## 動機與比較範圍

前批改變樹目標、深度、建案權重及時間修正，尚未得到一致跨年改善。R94檢查欄位token之間的條件交互能否改善prior NN及完整集成；這是待驗假設，不是已由誤差確立的原因。原R57/R58 TabM-style三期失敗反證保留。更大的類別embedding亦可能增加建案記憶，不能宣稱新架構必勝。

主對照為完整v22，次對照為完整v24。新架構同時改embedding維度、結構與dropout位置，結果只能支持或否定這套固定配置，不作純attention因果宣稱。舊控制沿同窗保存模型／向量，不同步重訓，GPU固定seed不保證位元相同。

## 單一固定模型

- 引用官方 yandex-research/rtdl-revisiting-models commit e3ed46cac38568785289d8fa16b8cfa585bde27e 的數值／類別token與Transformer結構。上游source、license、初始化契約另存outputs/round94_upstream/及review；實作前核清楚，不安裝浮動套件。這是本題小型FT-style改編，不是論文完整復刻。
- 144個數值各作 x_j*w_j+b_j；23個類別各64維lookup＋欄位bias；加一個可學習CLS，168token、d64、4heads、2blocks。Tokenizer/CLS沿官方uniform ±1/sqrt(64)，QKV與attention輸出bias沿官方初始化；其餘按固定上游契約實作。
- 第一block完整self-attention；第二block僅CLS當query、所有token作key/value，第二block殘差及FFN也只計算CLS。這是正式計畫對可行性稿「兩層完整attention」的明確變更，於任何真fit前固定。上游雖最後Q僅CLS，但residual broadcast後仍算所有FFN；本地另裁切無需輸出的token運算，是改編，需同權重eval等價測例，不稱訓練dropout亂數或位元相同。不回寫歷史可行性稿。
- PreNorm但第一attention前省略LayerNorm；FFN 64→256→ReGLU128→64。attention與FFN dropout .1，residual dropout 0，類別置未知碼dropout .1。CLS LayerNorm→ReLU→Linear→Softplus，單一正ratio。
- 最終head bias初始化為 fit median(y_million/price_factor_million)的inverseSoftplus，未用query價格。官方線性初始化的具體值記錄於upstream review；實作source在真fit前作SHA凍結。

## 固定輸入與損失

三期分別沿用原R63 prepared.npz的九個成員與其同窗fit-only R47 encoder、R60五折建案OOF/full-fit prior、R63日期權重。每期indices/frame/欄位/cardinalities/encoded arrays/factors/date/fold/prior state必須對照原保存SHA與來源核對。舊prepared不就地改寫，亦不用後期state回測。

numeric float32 144欄＝72尺度化值＋72缺值旗標；categorical int64 23欄。fit labels train_y_million、fit及query factor均float32。fit因子A×P_OOF/1e6，query因子A×P_full_fit/1e6；prior含fit價格資訊，不能說是純X。OOF分案不是時間OOF，但只在外層fit之內。trainer契約不接受query標籤。

loss = mean(saved_date32 * abs(Softplus(z)*saved_fit_factor32 - y_million32))。不再乘面積、密度或batch正規化，不clip ratio/price。推論用float64(ratio)*float64(saved_query_factor32)*1e6，因子只乘一次。

seed42、30epochs最後一輪、AdamW lr.001、wd.0001（全部參數，沿R63；不使用官方排除tokenizer/norm/bias的分組API），CosineAnnealingLR eta_min1e-5、float32/matmul high、threads4。microbatch256、累積成原2048列一個optimizer step；最後不足組以實際組列數除總加權L1，每microbatch sum/group_size再backward，不等權平均大小不同的microbatch。每epoch固定來源相同seed的randperm；dropout亂數序列与舊MLP不同，不能稱只有架構一個位元差。

推論batch256、model.eval。正常保存重載用同batch/full-query逐位一致；分批／列順序語義合成檢查可有浮點誤差，需報出容差與最大差，不把近似稱逐位相同。

## 完整回歸組合

F＝保存R54 area NN；T＝新R94 attention prior NN；L＝保存unit LightGBM，全部為總價元：
B=.75*(.5*F+.5*T)+.25*L
N_new=spatial(reference(B))
R94=median(N_new, saved_R46_log_spatial, saved_R79_unit_spatial)

R/S沿同窗原state與規則各一次，不重新fit或改融合比例。必須先重播原R63 native→R71 seed42 NN base/reference/spatial→v22 median全向量與原完整指標；不錯用R63早期混raw的舊base。保存native ratio/TWD、native mean、base、reference、NN spatial、final及必要diagnostics。

## 實作、資源與失敗邊界

先CPU合成shape/finite/正輸出、加權loss/gradient、微批累積（關閉dropout的等價性）、每列獨立而非跨成交attention、eval保存重載與query擾動不改fitstate。盡量沿用已驗證OOF helper，不以合成測例冒充實際fit來源核對。

GPU空閒時只做合成资源smoke，cardinalities可讀同窗metadata但不讀真價格或fit真交易；固定microbatch256與168tokens、optimizer累積2048、至少warmup2組＋量測3組，記錄torch/CUDA/GPU、參數數量、allocated/reserved峰值與每組秒數。由步數估算三期成本，標為估算而非已耗時。若OOM/nonfinite或估算三期>90分鐘，停止並保留失敗／成本，不静默縮模型、改batch或換lr繼續試。此資源篩選不是預測表現篩選。

通過source核對、合成與獨立preflight後，主線單張GPU串行跑development→confirmation→auxiliary各1fit，不因首期分數退步跳过後兩期；真fit共3次，無seed/hyperparameter sweep。工程錯誤保留failed輸出與新修正版路徑，不覆寫保存失敗，不以分數選是否修復。

## 採用門檻、診斷與下一步

三期overall及原固定weighted皆低於v22，且最新overall至少改善0.5%，才可進入更深入候選審查；另完整看新／既有建案、城市、季度、五價帶（事後真價分群僅診斷）、testmix與native到median。保存對v22及v24三期×project/month×overall/weighted的24配對區間（沿原2000次/seed）；區間不作Private勝率。

反覆使用年度窗口存在選型偏差，不是新盲測。若過門檻仍需R90固定新切點重建原六模型、encoder/prior/reference/spatial與新候選，原已prepare X不等於已完成新窗驗證。任何本轮不自动全量fit、test預測、CSV、改v22選定或Kaggle操作；失敗保留研究。

## 保存與交接

固定計畫、上游revision/license、source SHA、preflight、合成/成本收據、每期模型/state/向量/完整結果與失敗另存，舊CSV/selector/報告/簡報保持。完成或中斷先更新HANDOFF/HANDOFF_LOG與新文件ZIP。程式快照不等於乾淨環境端到端重訓。
