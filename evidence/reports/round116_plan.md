# R116：單價樹改為固定Depthwise成長政策

2026-10-08。在R114新模型fit／分數之前固定。R115另為純X支持診斷，與本輪不合併。本輪只測一個樹成長政策，不搜尋深度、葉數、leaf最低筆數、seed或融合比例。

## 假說與明確反證

原R79單價CatBoost採SymmetricTree，同一層共享切分條件；Depthwise在同層每個葉選擇自己的條件。假說是不同建物／地區可能需要不同切分；沒有假設較自由就一定較好。官方文件確認兩者支援CPU／GPU，Depthwise只能匯出JSON／CBM且不支援PredictionDiff，本專案使用CBM與一般predict，不依賴PredictionDiff。[CatBoost官方成長政策與最低葉筆數文件](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy)。

同樣depth8最多256葉，不代表相同假說容量、分裂數或正則化；非對稱葉條件更自由，可能增加對少數案的擬合。保持min_data_in_leaf未明設、官方預設1，不改成30或其他搜尋值；實際模型原生欄位／各樹葉數須保存，不能說原生派生參數全部不變。R27 raw/uniform depth10在development退步0.8988%、新案退步1.4659%，只跑第一窗即停止；不是單價Depthwise的直接試驗，但構成容量風險。R92單價depth6未通過、R112 Ordered單價三窗退步及大群惡化，直接反對『換樹政策必然改善』。R84增加2999→4499樹與seed組合的歷史反證也應保留。

## 唯一固定模型改動

沿原R79：單價target、原有效面積及inverse、365日日期權重乘面積、95欄／23類別、depth8、2999trees、lr .065、seed42、MAE、Plain、max_ctr_complexity2、implicit ctr_target_border_count1、l2_leaf_reg6、Bernoulli subsample .85、random_strength .5、one_hot_max_size8、border_count128、GPU0／gpu_ram_part .75／thread_count4，無eval_set、early stopping或epoch／tree挑選。唯一新增passed parameter grow_policy='Depthwise'，原grow_policy是implicit SymmetricTree；新meta須分清added與changed names。

fit state和legacy R19 feature state是特徵來源證據，不是本輪樹成長參數。所有可從native get_all_params取回的passed parameters按既有float rtol1e-7／其餘精確契約核對，actual grow_policy必須Depthwise、boosting Plain、depth8、2999trees、feature order／23類別及全部leaf counts須核。每樹葉數正整數且≤256，記錄其完整向量SHA、min／max及分布。native派生欄完整保存，缺失runtime-only欄明列，不能默默略過。全query raw及還原TWD保存後CBM重載皆exact；inverse只套一次、沿原非負政策，沒有新增cap、reference或spatial放在helper內。

## 固定四窗、完整組合與判定

與R114相同四origin，一律執行：development（2022-08-11至2023-08-10，240316／81234）、confirmation（2023-08-11至2024-08-10，321550／133697）、auxiliary（2021-08-11至2022-08-10，134805／105511）、offset_2023_02_11（2023-02-11至2024-02-10，272117／107169）。各切點自己的純X／95欄geography、dateweight、fit-only labels及reference／spatial，不能用後期或全量狀態。原v22／v24四分支和最終median按起點精確重播。

只替換v22的unit42 native槽；base=.75×新native+.25×原LightGBM，固定R/S各套一次，最後與原NN／H逐筆median。沒有合併新NN、改v24配方或Public搜尋。四個native模型／metadata／raw＋TWD／五層向量與source/input/protected均直接封存，四fit成功退出後另評分才讀query價格。工程／資源失敗保留，不自動重試或降參數／改CPU。

沿R114固定判定：三August強推薦要求整體與固定加權均改善、最新整體≥0.5%、無季度或≥500列群退步>5%；探索要求最新兩指標不更差、每窗兩指標退步≤0.25%、至少兩窗整體改善、無上述群退步。兩者都另須offset兩指標退步≤0.25%、無季度／≥500列群退步>5%。完整保存城市／月份／相對季度／新舊案／五真價帶／誤差尾部、五層與v24次對照，不只報平均。

Project與3月循環月份各2000次、seed20260925，三August stage index0／1／2，offset index0；是重疊已用資料上的條件不確定性，不是獨立四票、未校正反覆選型、不是Private勝率。只有combined探索門檻與獨立postfit都通過，才另立一次全量fit計畫／獨立final preflight，預留v31；本回測本身不產CSV、fullfit或Kaggle操作。

## 界線

v22原選定、v25探索檔、簡報及旁白保持，Private未知。固定seed不保證GPU bitwise重訓；保存推論核驗不是乾淨環境端到端重訓。禁止內政部或外部成交答案、test ID調價、未知真價路由。所有科學規約和source於第一fit前凍結、獨立預審必填；剩餘Codex≤20%即保存停止，不重置／購買／使用額外點數。
