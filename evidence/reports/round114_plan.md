# R114：固定先驗加上有正負值的金額殘差

2026-10-08。在R111／R112及R113新query評分前固定。本輪是一個新參數化假說，不搜尋head初始化、loss係數、epochs或組合比例。此計畫不代表已完成source、預審或訓練；實際執行須另通過獨立預審與全部來源凍結。

## 為什麼提出

原R63 prior NN輸出正倍率，再乘OOF／full-fit參考總價F，以日期加權總價MAE訓練。對head preactivation z，除不可微點外，梯度含F×sigmoid(z)；這是此參數化的事實，不是證明其有害。改用金額殘差的線性head時，總價F+h的MAE直接head梯度為w×sign，數值尺度與正值約束都不同。要測的是完整時間外組合是否更穩定，不能只看native NN變好就宣稱成功。

保留原R47 MLP隱層及23類別embedding、144數值欄、九array、獨立unknown類別dropout、seed42、30epochs、batch2048、AdamW lr .001／weight_decay .0001、cosine eta_min 1e-5，以及原R63保存float32日期權重，不重算或批次正規化。只把Softplus倍率head改成signed金額linear head，原head weight隨機初始化沿用；bias固定為fit的median(y_million−F_million)。隨機weight不歸零，因此初始預測不保證恰好等於prior。這同時改變參數化、bias、正值約束和梯度，不能稱為純loss對照或優化步幅匹配。

fit總價為F_million＋signed_residual_million，loss固定mean(w×abs(total_million−y_million))，訓練不clip；query時保存未clip signed total與native=max(total,0)兩份float64 TWD，清楚記錄負值筆數、最低值與比例。只這一次既定非負處理，沒有價格上限、ratio cap、校正或依ID特例。先驗F的fit仍是原建案group OOF，query仍是該外層fit自己的full-fit prior；沒有改成時間OOF。

## 最強反證

R91已有加法prior CatBoost：三窗中的中間與早期整體MAE分別退步約47,037／37,857元，最新約改善4,395元；中間及最新億元群分別退步約13,695,997／4,427,483元。因此不能把本假說宣傳為大額房價解方，改成NN也不自動移除這個反證。R104教師offset NN中間及最新整體退步約39,238／11,007元；其ratio、145欄、子集及教師loss與本輪不同，仍提醒offset處理未必有效。R102的native三窗改善，完整組合中間約退步934元、最新約退步7,219元，直接反對只看native選模型。

## 固定四個時間起點與流程

原三個8月窗口順序仍development（2022-08-11至2023-08-10；fit240316/query81234）、confirmation（2023-08-11至2024-08-10；321550/133697）、auxiliary（2021-08-11至2022-08-10；134805/105511），另加R97保存的offset_2023_02_11（2023-02-11至2024-02-10；272117/107169）。四者一律做，共四個NN；不是看完前三窗才決定第四窗。

使用每個起點自己的已驗encoder、OOF／full-fit prior、dateweight、其他native、reference與spatial。只能替換v22的N_prior槽；原N_area、LightGBM、H、U、組合比例與median均保持，R/S各套一次。各起點的原v22、v24與原NN槽完整回放。四個新模型、metadata、未clip native與五層預測均直接封存並檢查成功退出，之後才另行讀query價格。fit-only來源、canonical/raw位置、全部九array／encoder／先驗／dateweight／R-S狀態與來源都在fit前凍結，沒有eval_set或epoch挑選。單GPU串行，工程失敗保留，不自動重試。

## 固定判定

三個8月窗口沿用R107–R112的兩套門檻：強推薦要求三窗整體與加權均改善、最新整體至少改善0.5%、沒有季度或≥500列群退步>5%；探索要求最新整體與加權均不更差、每窗兩指標退步≤0.25%、至少兩窗整體改善、沒有上述群退步。兩者都另要求2月起點整體與加權退步≤0.25%、沒有季度或≥500列群退步>5%。只在兩指標都降低的起點稱一致點估計改善，容忍內退步仍須列出。

沿各origin原來的城市／月份／相對季度／新舊案／五真價帶與誤差尾部，以及五層TWD MAE。對v22與固定v24都呈現完整反證。每起點project與3月循環區塊各2000次，seed20260925，原三窗stage index 0/1/2，offset index0（沿R97），抽樣方法與門檻不據結果更動。不把窗口當四個獨立樣本，也不把bootstrap稱為Private勝率。

只有探索全條件及獨立postfit均通過，才可另立單次全量fit計畫與獨立final preflight，候選編號預留v30。尚未通過時不得產生該CSV；本四窗研究本身沒有全量fit或CSV。全量階段若獲准仍只換一個prior NN、沿原467221 fit／47073 query，不合併R107／R111／R112，不碰v22、v25或原簡報。

## 限制與停止

四窗口都重疊且已參與多輪研究，沒有新盲測；原保存對照不是同期matched refit；固定seed不含GPU重訓變異。OOF不是時間OOF，native重播不等於從原始資料乾淨環境端到端重訓。沒有內政部或外部成交答案、test ID調價、Public搜尋或未知Private資訊。Kaggle上傳／勾選仍由明確指示處理。

持續工作依使用者最新門檻：Codex remaining≤20%即停止新增工作並保存交接，不自行重置、購買或使用額外點數。新source／輸出保留失敗與SHA、拒覆寫；v22仍保留原選定身分。
