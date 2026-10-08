# R115：兩種排除政策的純X支持暴露稽核

2026-10-08。在任何本輪真實X統計前固定。這是診斷，**不是回歸模型、Depthwise方法、價格先驗重建或訓練授權**。程式準備與人工CPU自查完成後，須由root獨立核對來源／readset並給出passed preflight，才可freeze及執行真資料稽核。沒有MAE、query y、模型／CSV或自動採用。

## 唯一問題與固定窗口

原R60以project/fallback群hash五折，fit先驗排除held整fold，query用完整fit。要查的是：相較排除transaction fold，整案排除是否改變原四層cell的可用支持、同案暴露及project集中度。支持變化不等於價格中位數變化或預測改善。

四origin固定順序同R114：development（2022-08-11至2023-08-10，240316/81234）、confirmation（2023-08-11至2024-08-10，321550/133697）、auxiliary（2021-08-11至2022-08-10，134805/105511）、offset_2023_02_11（2023-02-11至2024-02-10，272117/107169）。四者一律執行，不按前三窗結果刪掉第四窗。

group fold必須逐列重播原cluster key與`sha256('42|'+key)`前8bytes little-endian mod5，與既有獨立fold.npy exact。project鍵使用city/district/name；缺名依原address、road/section、canonical row回退，代理群不是實體建案認證。

transaction fold唯一規則：對各origin的原fit列順序，重新建立`np.random.default_rng(42)`，只抽一次`permutation(n)`；令`folds[permutation] = np.arange(n) % 5`。各fold大小差至多1，沒有其他種子、分層、分折搜尋或query參與。前份方向memo提到的SHA排序不採用，以本規約為準。

## 只准讀什麼

三August只讀既有`data_cache/train_features.pkl`（無target特徵）、`outputs/round6/geography/development/train_source_cache_indices.npy`（raw-cache到canonical的純索引）、各`outputs/round60/{stage}/fit_indices.npy`／`query_indices.npy`／`prior_fold_ids.npy`，以及`outputs/round79/{stage}/model/date_weights.npy`。權重是獨立保存的原R63 float32日期權重；R79 metadata中value SHA、原R63 preparation與canonical indices須一致。

offset只讀`outputs/round90/preparation/Xfit.pkl`、`Xquery.pkl`、fit/query canonical indices及`outputs/round97/nn_prepared/fold_ids.npy`、`source_indices.npy`、`date_weights_float32.npy`。以原R90 target-free manifest與R97 prepared metadata綁定byte/value SHA、列數及來源axis。不得用reset-fit-row或raw-cache索引替換canonical axis。

只讀少量既有權威JSON與feature_schema作路徑／SHA／dtype／shape／日期權重投影；JSON可能記其他artifact身分，但只將上述target-free檔納readset。**不得開含y的任何NPZ、raw價格pickle、prior價格state/evidence向量、reference/spatial價格pickle、模型或保存預測。** 不因missing weights/folds而抽取NPZ、重算權重、改用其他origin或跳窗；記明阻擋。權威 metadata與readset的實際精確集合由source的`specification()`產生，root預審須完全一致，不能接受額外價格路徑。

真資料讀取前後核readset／source／plan SHA；每stage核fit/query index sorted/unique/disjoint及原列數、outer日期界線、target-free欄位與fold重播。對整份純X快取的反序列化如實揭露，不能聲稱只有fit/query列進過記憶體；只選定四origin的列參與統計。

## 固定支持與集中度公式

原R60 cell依序為city/district/type/building_type、city/type/building_type、type/building_type、type；cell raw count≥30才可用，否則往粗層，最後global（所有reference，沒有額外30門檻）。不新增project或road價格cell、不乘1.25、不裁count。

fit每row分別排除自身group fold或transaction fold，計四層reference raw count及首個合格層0..3／global4。query只用同一完整fit reference，兩政策共用query統計，不把query自身計入支持。保留完整5×5層級轉移，包括空格，並列原float32日期權重的fit質量。

每個reference cell及global以原project/fallback代理群g計：raw交易筆數n、群數G、最大單群raw筆數占比。額外以**已保存完整outer-fit日期權重**轉float64，令`M_g=sum_{j in reference cell/group g} w_j`，`project_mass_ESS=(sum_g M_g)^2 / sum_g M_g^2`，最大單群date-mass占比`max(M_g)/sum_g M_g`。這不是逐fold重新計算R60 prior的日期權重，不是獨立樣本數、價格可靠率或先驗置信度；missing代理另標明，不把30交易說成30獨立案。沒有現存權重不得猜ESS。

另計held/query row的同cluster peers在完整reference與selected cell內的筆數／占比；named-project與missing/fallback分開描述。group-policy fit的同cluster peer reference數必須全部為0，transaction-policy與query不預設為0。query seen/new/missing沿原city/district/project會員定義，missing不併入seen。query分布另按城市及origin-relative四季全列，不依結果刪群。

只輸出聚合分布（min/mean/max與固定linear quantiles 0/.1/.25/.5/.75/.9/.99/1）、層級／轉移筆數／日期質量、各層所有cell及supported cell的集中度分布、fold大小、SHA及來源限制。不得輸出row ID、原列索引清單、project／地址／cell鍵名稱或逐筆向量；保留SHA供身分核對。城市名稱及seen/new/quarter標籤是允許的聚合群名。

## 固定判讀與反證

本輪沒有模型改善門檻或自動fit gate。逐origin完整列出是否存在層級改變、同案暴露、集中度與支持差異；不能事後挑城市或閾值。若只在已有大量多案支持的同層cell多出近似比例，沒有足夠證據追加fit。即使存在gap，純X也不能證明先驗價格不同、更準或Private更好，仍需root另立一項模型假說及價格實驗規約。

唯一可能後續是假設成立後，原R63 ratio NN只將fit prior分折改為transaction-fold，query full-fit、144/23與原架構／loss／weight／融合保持；本R115不實作、授權或選擇它。不混R114 signed head、不做雙context混訓、新scalar、支持收縮／dropout或比例搜尋。相同X同r卻乘兩個F通常對應不相容ratio目標，不能稱雙context免費修復。

最強反證保留：R67支持欄最新改善、其他兩窗退步，且早期query支持P99 106366對fit6705，支持差已發現而非新發現；R100時間先驗native三窗退步，完整最新相對matched C多11158.91元；R65及R102 native改善未穩定留到完整組合。query新案占歷史多數，同案暴露更像seen不代表適合new。row-fold排除本row也不保證排除同合約複本／同批交易依賴，不得稱獨立OOF或逐筆as-of。

四窗重疊且被多輪研究使用，非四票獨立盲測。pure-X不知道價格中位數、y/F、誤差、Private組成或未來效果。沒有Public搜尋、未知test答案或外部成交價格。

## CLI與保存界線

新source為`src/round115_prior_support_x_audit.py`。`--self-check`只用人工CPU資料；`--describe`僅metadata與存在性／SHA契約，不反序列化真X／array；`--freeze`須既有獨立`outputs/research_round115_20261008/preflight_review.json` passed且source＋plan／完整input SHA／規則exact；`--run`只接受該不可變request。準備階段不呼叫真freeze或run。

輸出限定`outputs/round115/`，新request／started／四stage aggregates／results／failure均拒覆寫並帶SHA。工程失敗保留，沒有自動重試或替代資料。只新CPU診斷，不fit、GPU、score、bootstrap、Kaggle、CSV或修改舊模型／source／v22／簡報。實際耗時須跑完才記，不借用歷史秒數。
