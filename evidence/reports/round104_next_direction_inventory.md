# R104 後的有限來源盤點：原生 prior NN 等權平均

2026-09-29。僅以 `rg` 與已完成的 source／report 盤點；未讀逐筆陣列、標籤或模型，未 fit、predict、score，也未建立下一輪計畫或新提交。唯一提交仍 v22，Private 未知。

## 尚未做過的精確組合

截至 R104，未找到「v22 原 R63 prior NN 與 R102T 的 145 欄 prior NN，在 native TWD 層固定各半，再經原 base／reference／spatial／median」的等價已完成實驗。[R102 driver](../src/round102_prior_input_experiment.py) 是分別以新 C／T 取代原 prior 槽；[R103](round103_report.md) 只做誤差診斷；[R104](round104_plan.md) 則重訓固定 base 目標，沒有對這一對 native 做平均。

相近研究不能混稱已測過：

- [R71](round71_report.md) 平均 R63 prior NN 與 R54 area NN；舊 R65／v18 的 145 欄 group-OOF prior 模型也不是 R102T 的 temporal-prior 訓練。
- [R87](round87_plan.md) 平均 R79 與 R85 的 CatBoost 原生預測，兩者使用不同面積目標。
- [R98](round98_summary_v2.md)／[R99](round99_summary.md) 在已修正的 N/H/U／median 後學全域 L1 權重，層級、模型及係數來源皆不同。

## 假設及最強反證

可考慮的唯一未執行假設是：固定 50/50 平均基準與新表示，是否能降低單一配方的預測波動／偏移，保留部分新 native 的收益。這是**方差降低假設**，沒有多 seed 重訓或方差估計，不能先宣稱方差已下降。

[R102 與 R103](round102_next_direction.md) 是最直接的反證：新 native 三窗 MAE 都較低，但 confirmation 在 base 已退步 57,060.64 元；其原生項改善 23,837.92 元，被 80,898.56 元的抵銷損失超過。development 另有 49,017 筆落在相同 median 外側，該群 spatial 的淨改善被截掉。固定等權也可能破壞既有誤差互補，不能只憑兩個 native 都有優點推論最終改善。

[R87 結果](round87_report.md) 也顯示等權不是保證：兩期略好，但最新期完整 MAE 增加 8,628.64 元。它是一般互補假設的反例，不是這一對 prior NN 的等價否證。MAE 的凸性只保證 native 均值的 MAE 不超過兩端 MAE 的平均，並不保證勝過 v22，亦不保證 reference／spatial／median 後仍有收益。

## 若另立固定測試，必須保留的界線

以下只是來源盤點後的實作界線，**尚未固定新實驗計畫或執行**。若後續立項，只允許一個事前固定 50/50 候選：`M=.5*P_v22+.5*P_R102T`，再按 `B=.75*(.5*A+.5*M)+.25*L` 的運算順序，重用同窗原 reference／spatial 狀態，最後與固定 H/U 取 median。不能平均已修正分支後冒稱同一方法。

必須先將原 v22 與完整 R102T 兩個控制的 native／base／reference／spatial／median 全五層向量與指標重播吻合，核對同窗來源、canonical 索引及價格單位；所有三窗完成，原 R/S 狀態保持。沿固定整體／加權、最近期改善門檻、季度與大群退步限制、全群診斷及配對區間，不自動替换 v22。

不能邊看 query 改 alpha、追加 .25／.75、分城市／價格／日期挑權重，或靠 Public 搜比例。這個想法已受 R102–R104 結果啟發，使用的是反覆研究過的窗口；即使事前只固定一次 50/50，也仍是適應性研究，不是新盲測，配對區間不是 Private 勝率。
