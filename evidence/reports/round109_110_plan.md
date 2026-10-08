# R109／R110：兩個固定方法研究

2026-10-08。在R107/R108已完成的結果之外，另立兩個獨立方法。這是累積選型的新輪，不是未見盲測；不合併107、108或按Public改參數。v22及其簡報保留，Private未知。

## 單一變因與理由

- **R109：固定末10epoch參數平均。** 原R63日期加權MAE、30epoch、seed42、所有架構／prior／optimizer／schedule保持。每個epoch21–30結束後按固定順序將state以CPU float64累加，等權除10後轉float32。使用同軌跡，避免跨seed神經元排列問題；原網路為LayerNorm，沒有BatchNorm running statistics。它不是標準SWA，不改尾段學習率、不另挑起點。可能降低末段波動，也可能混入未充分收斂的權重。每窗一次fit，同時保存epoch30及平均模型與各自重載預測。沒有把R107的SmoothL1帶入本輪。
- **R110：單價類別統計的全域8個目標邊界。** 原R79單價/date×面積權重與所有參數保持，max_ctr_complexity仍為原2；只加`ctr_target_border_count=8`。原R23全域8人工模型曾fit成功，但因native descriptor=1的舊驗收停止，未房價fit；本輪是同設定在目前R79單價模型的首次房價測試，不稱全新發明。GPU不支援simple/combinations_ctr內嵌TargetBorderCount=8，使用全域參數；其保存global8及實際結構驗收，不要求descriptor=8或全部8閾值必須入樹。更多目標切分可能刻畫價格分布，也可能提高噪音記憶／成本。

## 固定執行及判定

沿R107/R108完全相同三個年度窗口、canonical索引、fit-only價格與相同切點的encoder/prior/reference/spatial/地理。各3次fit共6次，單GPU依序；全部封存後才評分。原v22組合完整重播。R109只替換prior NN，R110只替換unit CatBoost；兩候選獨立，不新增融合比例。R109額外的epoch30同軌跡向量作機制對照，仍以v22為主要工程判定。

保存全部整體、固定新舊加權、城市／季度／月份／新舊建案／五價格帶與五層診斷。按原建案／3個月區塊各2000次、seed20260925配對區間；不是Private勝率，不校正累積選型。原v24為固定次對照，不省略更強離線反證。

強推薦條件沿前輪：三窗整體及加權均改善，最新整體改善≥0.5%，季度或≥500列群沒有>5%退步。探索提交條件也完全沿前輪：最新整體及加權均不退步、六個窗口指標各退步≤0.25%、至少兩窗整體改善、沒有上述>5%退步。未過不為湊版本出CSV。通過亦不自動替換v22；若通過探索條件，另立全量規約後才fit並發行新試交。

每窗資料／source於fit前凍結，query y只評分時讀；saved-model全query重載必須exact。R109同軌跡epoch30與保存R63差異不得偽稱損失改動；固定seed不保證GPU重訓bytes相同。保留失敗與所有原檔，不查成交來源、不改Kaggle勾選、每階段查額度並在剩餘≤1%保存停止，不用額外點數或reset。
