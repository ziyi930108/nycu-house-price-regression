# R86 固定實驗：把既有分組 OOF 價格先驗提供給單價 CatBoost

日期：2026-09-28。第二個獨立假設；不是將R85平方根尺度與新特徵混合。主對照仍為原v22，不依Public搜尋。

## 固定處理與理由

保持R79目標y/(A*1e4)、權重A/100×保存日期權重、原95欄/23類別、2999樹、depth8、.065、seed42及其他参数。只加一欄log_recency_oof_unit_prior=log1p(prior_TWD_per_m2/1e4)，共96欄，類別仍23。log1p只作用於先驗特徵，不改回歸目標，不fit標準化、不裁價、不調比例。

動機：原單價CatBoost由地區及房屋欄位學價，既有R60的近期加權分組單價提供較平滑的區域價格尺度，可能幫助樹學地理交互作用。過去NN已使用相關prior表示（R65），所以不是首次使用價格先驗；本輪測把這項訊息交給單價CatBoost。也可能與CatBoost類別統計或NN重複、降低分支差異，必须檢查完整組合。

訓練欄位取同窗R60 train_oof_unit_prior.npy，五折按建案分組、held fold只用其他fold的fit標籤及日期。query取同一fit的fullstate prior，完全不讀query標籤。OOF不是時間OOF，不能說每筆只見其交易日以前的成交。

## 驗證與對照

三期、stage順序、canonical train、日期、R6/R39地理、NN/log/LGB/reference/spatial與R85計畫相同，仍只對照v22/R79及v24/R83。unitbase=.75*new_TWD+.25*原LGB；逐分支R/S後median。每期只新增1次GPU CatBoost fit，無eval_set/early stopping、不按第一期挑seed或取消其他期。

在GPU fit之前用fit資料重新計算R60 OOF先驗，核對saved OOF向量、fold、fullfitstate及query prior完全相同；這是同方法再現核驗，不依query價格fit。保存source/file/value SHA與索引。R63的保存pricefactor必須能由A×R60prior/1e6精確轉float32對回prepared值。若不一致先停止，不能用方便但錯誤的fullfit prior取代訓練OOF。

報告整體/固定加權MAE、季度/城市/新舊建案、五價帶、testmix與兩種配對區間（2000次，seed20260925）。歷史窗口多次使用，不是fresh blind validation；bootstrap不校正選型偏差，不是Private勝率。原GPU控制未同期重訓，微小差異可能含重訓變異。

## 保存與判定

使用新outputs/round86、新helper與driver。R85保持原已凍結程式及產物，不修改途中配方。v22、既有CSV、models、selector和歷史decision皆保護SHA。先完成3期，再綜合效果大小、代價與不確定性，決定是否全量fit為新候選。不預先承諾會更好、不自動代選Kaggle。若兩個研究都未提供足夠證據，明確保留失敗與v22，不製造較差CSV湊版本。
