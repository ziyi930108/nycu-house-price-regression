# R111／R112：固定殘差網路與單價 Ordered boosting

2026-10-08。使用者要求持續模型優化至Codex剩餘20%；停止規則取代1%，不重置／購買／使用額外點數。v22選定與原簡報保留，新候選不自動上傳或勾選。此規約在兩方法任何新房價fit／評分前固定。

## 方法與理由

**R111：先驗倍率的固定殘差MLP。** 原R63九陣列、144數值／23類別、各切點fit-only encoder、建案群五折OOF prior、日期權重、30epoch、batch2048、seed42、AdamW lr=.001／wd=.0001、cosine最低1e-5不變。只替換prior NN的固定架構：原類別embedding維度與逐欄0.1 unknown dropout；concat數值與embedding→Linear256；兩個residual block，每個為LayerNorm256→Linear512→ReLU→Dropout.1→Linear256→Dropout.1，再加identity；最後LayerNorm256→ReLU→Linear1→Softplus。loss仍為`mean(saved_date_weight*abs(ratio*saved_price_factor_million-y_million))`，bias仍依fit median(y/F)初始化，沒有R107平滑loss或R109參數平均。

這是對[表格ResNet研究](https://arxiv.org/abs/2106.11959)的本題適配，以殘差路徑改善訊號傳遞為假設。LayerNorm、embedding、倍率頭及固定訓練均沿本題，並非官方架構完整復刻。寬度／層數／參數量／亂數消耗也改變，不能將效果純歸因skip，也不能保證比原MLP更好。保留R57/R58共享成員及R95attention架構失敗反證；原unknown embedding已有訓練，不聲稱修補未訓練unknown。

**R112：原R79單價/date×面積權重的Ordered boosting。** 95欄／23類別、unit target與還原總價、date×area Pool權重、2999樹、depth8、seed42、max_ctr_complexity2及其餘完整R79參數保持。唯一傳入參數變更`boosting_type: Plain→Ordered`，不帶入R108單欄CTR或R110 target8。沒有eval_set、earlystop或額外價格裁切。保存CBM全query重載exact，驗原生Ordered／GPU／完整recipe與特徵順序。

R17已在raw總價／均一權重模型真試Ordered，development完整MAE退步6.085%、新案7.435%、Q2 9.205%，後兩窗未做。此為強反證；本輪只檢查不同target與date×area權重下的交互假說，不能宣稱Ordered全新或必優。GPU官方文件說明資源較高，原R17成功不能保證此次較大fit成功。任何OOM／非有限／合規失敗保留現場，不按新成績改參數或自動重試。

## 資料、對照與時序

固定沿原三年度：development fit240316/query81234（2022-08-11至2023-08-10）、confirmation fit321550/query133697（2023-08-11至2024-08-10）、auxiliary fit134805/query105511（2021-08-11至2022-08-10）。每個窗口自己的encoder/prior/reference/spatial與切點前地圖；不建立新時間切點。六次fit由主線單GPU依序完成，全部原生模型、metadata及五層query向量直接封存後才讀query價格統一評分。trainer/helper不接受query y。

R111只替換原prior NN，原area NN/LGB/H/U及R/S不變。R112只替換unit CatBoost，原NN分支/H/LGB及R/S不變；兩者不互相混用。原v22整鏈精確重播，v24保留為較強離線次對照；保存對v22及v24全部結果，不省略反證。主對照為保存模型而非同期匹配重訓，不聲稱seed42消除GPU波動。

## 固定判定

沿R107–110同一規則，所有總價以未取整TWD比較。強推薦：三窗整體及固定新舊案加權均改善，最新整體改善≥0.5%，季度或≥500列群沒有>5%退步。探索試交：最新整體及加權均不退步，六窗口指標各退步≤0.25%，至少兩窗整體改善，沒有上述>5%退步。不得看新分數後放寬，未過不為湊版本出CSV。

保留各城市／月份／季度／新舊案／五真價帶、五層branch指標；按原建案及3個月區塊各2000次、seed20260925配對bootstrap。三年度已多輪重用，選型偏差未由bootstrap修正；區間不是Private勝率。新CSV若過探索仍須另立全量規約與獨立發行核對，暫定R111→v28、R112→v29，兩者現在均不存在，不改先前R109→v26/R110→v27的未發行歷史。

不取得內政部或外部成交答案、不按test ID或未知真價改價、不按Public搜參數。source/input/protected SHA凍結，拒覆寫，保留失敗。保存可追溯與重載成功不等於從零重訓或新環境端到端重現；Private未知。
