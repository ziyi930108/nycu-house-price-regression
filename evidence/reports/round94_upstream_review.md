# R94：FT-style 上游來源固定與實作邊界審查

日期：2026-09-29（Asia/Taipei）。只查官方來源、保存少量原檔及核對程式；未 import 上游或 Torch、未建立模型／trainer、未安裝套件、未跑合成或真實訓練，也未讀 query labels、存取任何房價來源。既有模型、計畫、CSV及[可行性歷史稿](attention_regression_feasibility_after_r93.md)均未修改。

## 固定來源與可追溯性

官方 repository 為 [yandex-research/rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)。本次以 GitHub API 解析後固定 commit **`e3ed46cac38568785289d8fa16b8cfa585bde27e`**，commit date `2024-11-12T21:04:04Z`，訊息 `Minor update`；後續所有下載都使用此 commit 的 URL，不再讀浮動 main。

僅保存六個小檔，共 **77,915 bytes**：根 README／LICENSE、package README／LICENSE／pyproject／單檔 Python source。沒有 clone整庫、下載資料集、圖片或 notebook。[下載 manifest](../outputs/round94_upstream/manifest.json)保留固定 URL、Git blob SHA-1、SHA-256與大小；每個原檔均由下載 bytes 重算 `SHA1(blob header + bytes)`，與固定 commit tree 指向的 blob相同。commit／根tree／package tree的原始API回應亦保留。這是檔案身分證據，不是模型執行測試。

|原檔|SHA-256|
|---|---|
|[package Python](../outputs/round94_upstream/upstream/package/rtdl_revisiting_models.py)|`ca7cef3b44cd54b1aeb51f2cc733bef01fbfe3eff00b5c2bd28ca4b4db6b01b2`|
|[package LICENSE](../outputs/round94_upstream/upstream/package/LICENSE)|`35381b68041ebc3408e1b76ab24ce5eed1f7ea93afd62d109ffdabf1b0daa324`|
|[根 LICENSE](../outputs/round94_upstream/upstream/LICENSE)|`bd39fcc730a79d256b14fcdccbfc8ac9f0c7fad4a2e8d01fc513def67980fe35`|
|manifest|`699800b3ceb2c12c6dccee1931cd27a1c0eb53ba94e7977f5cdfa019963cbc8f`|

[package metadata](../outputs/round94_upstream/upstream/package/pyproject.toml)指定 Python≥3.8、唯一執行套件相依 `torch>=1.8,<3`；source版本為0.0.2。本機既有Torch source為2.13.0+cu126，範圍相容不代表已實測完整API。若R94採本地少量改編，不需要導入上游套件或任何額外dependency。

## 授權與保存方式

根與package皆為 **Apache-2.0**；根版權為2021 Yandex LLC，package為2023 Yandex LLC，所以兩份LICENSE bytes不同，不能合併後宣稱完全相同。package 的授權應伴隨取用的package source。固定tree的根與package層未列NOTICE，本次没有自行杜撰上游NOTICE。

依保存的[授權第4節](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/LICENSE)，後續若將取用／改編程式納入GitHub或交接包，需附完整license、保留相關版權／歸屬，修改檔需標示已修改。可在新trainer檔頭標明來源commit與本題差異，並把package LICENSE及來源manifest一併納入發布；不要修改目前保存的上游原檔來冒充本地版本。這次僅保存來源，尚未發布R94程式。

## 架構、初始化與官方 API 的準確邊界

下列行號均對應保存的原始Python檔（共872行），不是網頁抽取後的行號。方法範圍另存[只讀查核收據](../outputs/round94_upstream/source_review_checks.json)。

|部分|固定source的實際行為及R94含義|
|---|---|
|數值 tokenizer，L161–209|每欄一對可學習向量w/b，輸出 `x[...,None]*w+b`；兩者均 `U(-d^-0.5,d^-0.5)`，d64時±.125。沒有共享數值投影、quantile處理或自動缺值補值。|
|類別 tokenizer，L212–289|每個類別欄獨立lookup，統一d64；每欄另有bias，皆±.125均勻初始化。沒有padding_idx，碼0也是可學習embedding；原R47未知碼規則及category dropout須由本地流程提供。|
|CLS，L675–689、L846–870|一個64維可學習向量，初始化±.125，沿batch展開；實際拼接順序是CLS、144數值、23類別，即168tokens。沒有额外position embedding，欄位w/b／獨立lookup提供欄位身分。|
|Q/K/V與Out，L362–424|四個獨立 `nn.Linear(64,64)`，不是直接使用PyTorch `nn.MultiheadAttention` 的packed權重。weight沿Linear預設；**四者bias都歸零，包含W_out**。head4，每head16，QK除以sqrt(16)=4。|
|attention，L435–465|先Q/K/V，對key軸softmax，再對probability作dropout，乘V後concat及W_out；batch列不互相attention。原碼显式建立logits/probabilities，沒有SDPA/FlashAttention。未壓縮時不要傳Linformer參數。|
|PreNorm／residual，L594–640|只有block0的attention前LayerNorm缺省；每個block的FFN前LayerNorm都存在，block1 attention前也存在。LayerNorm gamma1／beta0、eps1e-5。殘差保持未normalized的輸入；attention及FFN各自有residual dropout。|
|FFN，L468–476、L604–618|`Linear(64,256) → ReGLU → Dropout → Linear(128,64)`；ReGLU是把前半a／後半b拆開後 `a*relu(b)`，不是GELU，也不能交換兩半。沒有手動FFN weight／bias reset，沿Linear預設。|
|head，L632–640、L668–672|取CLS→LayerNorm→ReLU→Linear(d,1)，官方輸出 `[B,1]`、無正值activation。R94額外Softplus／squeeze／prior還原及fit median bias屬本題邏輯，不在官方API內。|

本機Torch只讀source確認Linear預設為 `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`的weight／bias。故QKV/Out、FFN首層、head的weight邊界±.125；FFN第二層weight／bias邊界±1/sqrt128。Attention四個bias再覆寫0；head bias若由本題fit median覆寫，要在建立後只改該項。`source_review_checks.json`保存本機linear.py、normalization.py與version.py雜湊，未import它們。相同seed不代表與官方逐位相同：建構順序、`nn.Embedding`先預設初始化再覆寫所消耗的亂數、微批dropout及runtime均會影響。

若單純對接官方constructor，對應參數為 `n_cont_features=144, cat_cardinalities=當期fit詞表, d_out=1, n_blocks=2, d_block=64, attention_n_heads=4, attention_dropout=.1, ffn_d_hidden=128, ffn_d_hidden_multiplier=None, ffn_activation='ReGLU', ffn_dropout=.1, residual_dropout=0`；不傳 `n_tokens` 或 compression。`ffn_d_hidden`指定值時multiplier必須None。官方 `get_default_kwargs(2)`卻是d128、8heads、attention .15、FFN hidden約170、FFN dropout .05，**不能调用該便利方法後誤以為得到本題規格**。此處是API對照，沒有呼叫constructor。

## 正式計畫已選的兩項差異，以及末層的易錯細節

主線於本審查期間明確決定：

1. **最後block僅CLS作Q，K/V保持全部token**；前一block完整attention。這取代可行性稿的兩層全token attention，正式計畫需明列，歷史可行性稿保持原文。
2. **維持R63全參數AdamW lr .001／weight decay .0001**，不使用官方 `make_parameter_groups`／`make_default_optimizer`。官方便利optimizer是lr1e-4／wd1e-5；分組方法把CLS／所有tokenizer、block normalization及所有bias設wd0。其規則並非所有LayerNorm weight都豁免：output head LayerNorm weight未由block搜尋涵蓋。R94刻意全參數decay，不能稱完全依官方optimizer。

**必須在新計畫／trainer中說明的細節**：固定upstream L656的確令末層Q只有CLS，但L653保存的residual仍有168tokens；L658把 `[B,1,64]` attention輸出廣播加到 `[B,168,64]`，接著FFN還算所有tokens，到L668才取CLS。因此「只有CLS作Q」不等於「末層residual與FFN也只算CLS」。若本地另把residual裁成CLS並僅計算其FFN，可省去不使用的輸出，需標成本地計算裁切並先做同權重、eval／無dropout的CLS輸出與梯度等價驗證；train dropout亂數消耗仍可能不同，不能承諾同seed／逐位訓練一致。主線已在[正式R94計畫](round94_plan.md)明確選定該進一步裁切：末層residual及FFN也只計算CLS。此決定在真fit前固定；新trainer必須完成上述同權重eval輸出／梯度等價檢查。

同樣不能把同形狀的 `nn.TransformerEncoderLayer`直接視為上述模型：預設activation、FFN hidden、norm位置、首個norm及初始化都可能不同。若使用fused SDPA取代显式attention，也要事前固定backend及dropout语义，不能看分數後切換。本次規格沒有因這些便利API而自動更動。

## 建議凍結與合成查核邊界

正式source可只依本地既有Torch重新實作小型模型；需保留上游commit／license／SHA及差異列表。合成至少核對：168×64 token與欄位bias、码0能更新、attention只混欄不混列、首層norm例外、ReGLU順序、四個attention bias0、最後Q長1/KV長168、明示最後residual／FFN形狀、全參數optimizer無遺漏／重複及一致wd、累積2048的loss按實際列數計、eval重載與列置換／微批分割一致。比對不同计算形狀時容許事前固定浮點容差，不能宣稱重訓決定性。

以上都是**待新trainer執行的必要檢查，不是已通過測試**。本次只完成原檔下載、Git blob／SHA比對與AST／文字審查；沒有估測或宣稱任何R94改善、fit時間或GPU峰值，也沒有改动原OOF prior／日期權重／v22融合。
