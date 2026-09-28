# R98 時間外 L1 組合：獨立驗證契約審查

2026-09-29。審查 [固定計畫](round98_plan.md)，SHA `394a65459349e479bb3e8a487b60516156705469fa51bfbe75fd8697bae3642b`。本頁只讀 source、小型 JSON 與既有報告，核對文件／source SHA；沒有載入價格、預測或索引陣列，沒有 model load、fit、predict、score 或 bootstrap。模型 helper、數值合成及正式執行尚待各自的前置收據，本文不是執行通過證明。

## 結論

固定方向在資料時間上可成立，沒有必須否決的科學契約問題。只改 v22 最後一層，使用 N/H/U 三支各自已完成 reference/spatial 的歷史 forward 預測，三個 stage 各學一組非負且和為 1 的 L1 權重，不比較挑選權重。先前提醒的跨 stage 標籤角色重疊已明載計畫：每個 stage 獨立程序只拿自己的合法 meta 標籤，全部封存後才另啟 scorer。

這是較晚三窗探索。offset 與另兩窗重疊、三窗訓練母體相依，且配方源於已看過的回測；不符合原三個 8 月窗口完整採用門檻，不是盲測，Private 未知。即使六項點指標與分群門檻全過，也只支持補做更早 forward block，不能直接全量 fit、產新 CSV 或改 v22 選定。

## 合法 forward 鏈與精確來源

| R98 query | meta 預測的 origin／合法日期 | meta 列數 | query 列數 |
|---|---|---:|---:|
| development：2022-08-11～2023-08-10 | auxiliary 模型 fit <2021-08-11；其 query 2021-08-11～2022-08-10 | 105,511 | 81,234 |
| offset：2023-02-11～2024-02-10 | auxiliary 全段，加 development 模型 fit <2022-08-11 所預測且日期 <2023-02-11 的前綴 | 105,511＋尚待純 X 核出的前綴數 | 107,169 |
| confirmation：2023-08-11～2024-08-10 | auxiliary＋development 全段，兩組均保留各自原 origin | 186,745 | 133,697 |

原三期 s 的完整分支：N=`outputs/round71/s/seed42/nn_spatial.npy`；H=`outputs/round46/s/recency_log_spatial.npy`；U=`outputs/round79/s/unit_spatial.npy`。相同 s 的控制為 `outputs/round79/s/predictions.npy`，索引為 `outputs/round79/s/{fit,query}_indices.npy`。只用 seed42 的 N、R79 的 U；不能換成 R74、R83、未修正 base 或 native。

offset query 三欄是 `outputs/round97/reference_spatial/{N,H,U22}/spatial.npy`，控制 `outputs/round97/v22_predictions.npy`；使用 `outputs/round97/{fit,query}_indices.npy`。它們只作 offset query，不能把 R97 模型回推或其 query 前半部答案加入本次 meta-fit。R97 seal 中的 `branches`、`predictions`、`arrays` 與 `artifact_sha256` 是實際欄位，不能假設舊 R79 schema 通用。

已讀 [R79 driver](../src/round79_experiment.py) 的 split／index／frame／reference residual 身分檢查；[R71](../src/round71_missing_cell.py) 使用 R54 area＋R63 seed42 prior、同窗 reference/spatial；[R46](../src/round46_experiment.py) 也是同窗 fit-only log 分支。較底層 [R3 baseline](../src/round3_backtest.py) 與 [R6 auxiliary](../src/round6_runner.py) 明確只在 fit Pool 訓練，沒有 query eval_set；R3/R6 reference 只吃 Xfit/yfit。這支持 row-level forward 譜系，不能消除其後反覆用這些 query 選模型的研究者偏差。

authority 是 [R79 plan](../outputs/round79/plan.json) SHA `7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f`、[R79 results](../outputs/round79/results.json) SHA `7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6`，以及 [R97 seal](../outputs/round97/sealed_manifest.json) SHA `429e5405652b4e9d54308d6f09d6722ab540694b13f92e2ad8577277b85893cb`。本次沒有重讀價格向量驗 SHA；正式 preparer 必須按這些 authority 核 bytes，並核三欄 canonical query index 一致，不能只比長度。

## 洩漏與索引檢查

1. 對每個 stage，meta 日期都嚴格早於該 cutoff、canonical meta 與 query index 不相交、meta index 不重複。offset 前綴只能按 `trade_day_index` 與固定 cutoff 取列，所有三欄與 y 使用同一 mask。不得按真價、殘差、城市或所學權重裁樣本。
2. canonical index 和原 `data_cache/train.pkl` 的 row index 不同。價格擷取須透過 `outputs/round6/geography/development/train_source_cache_indices.npy` 的原 mapping；R97 也保存對應 raw-cache index。權威 hash、映射、順序、日期與來源 origin 要同時保存。不能以 reset 後 0..n-1 配對。
3. development query 的部分／全部價格分別是 offset／confirmation 的合法 meta 標籤；offset query 前半也可能是 confirmation 的合法 meta 標籤。只能保證「各 stage fit 不接收自己的 query y」，不能宣稱三期全 seal 前整個工作流程未讀任何評估列價格。每 stage 專用檔與子程序將此界線落實，main 不提前看 scorer 指標。
4. preparer 反序列化整張 raw pickle 的事實需保留；它不算 query 統計、不把本 stage query y 傳入該 stage driver。scorer 在所有 stage seal 與 exit0、完整 query reload、控制 median exact、權重公式和 source/input SHA 驗證後才評分。
5. query 的 seen/new_nonmissing/missing 仍依其底層原模型 fit X 判斷，不能用較小的 meta-fit 建案目錄。日期與分群都是固定特徵，不得以價格決定推論路徑。

## L1 dual 與工程數值契約

以 P 的三欄依 N/H/U 固定順序、y 同除 1e6。原問題是 `min_w sum(abs(Pw-y))`，`w>=0, sum(w)=1`。MAE 的除以列數是正的常數，不影響解；不可另加 group/date weight。

由 `abs(r)=max_{|u|<=1} u*r`，等價 dual 為 `max y.T@u+v`，限制 `P.T@u+v<=0`、`u∈[-1,1]`、v 自由。換成 solver 最小化 `[-y,-1]@[u,v]` 正確。三個不等式的非負 Lagrange multipliers 即 w；SciPy 最小化不等式 marginals 的相反數是這三個權重，v 的 stationarity 給出 sum(w)=1。因此 parent 固定的 dual 形式與凸 L1 目標等價。

固定 highs-ds、presolve、單執行緒、steepest-devex、300 秒、maxiter 1e6、primal/dual tolerance 1e-8，仍需由 helper 的 parameters／source／實際 runtime 和合成收據綁定。多解保留固定 solver＋固定 row order 的返回解，不能比較其他解的 query 成績；不能 clip、normalize 或靜默改 solver。保存 solver status/message/iterations、原 marginals、sum/min weights、primal objective、dual objective、gap 與可行性／適當 KKT 容差診斷；失敗保留而不帶著不合規權重進 scorer。JSON reload 後用同一 float64 公式產生完整 query，必須逐位與已保存輸出相同。這些是尚待 helper 檢查的實作條件，本頁未執行 LP。

## 評分、反證與有限升級條件

預固定每窗的 overall 與既有 weighted 都低於 v22，且任何季度或至少 500 列固定分群不得 >5% 退步。scorer 的群集合應明列沿 R97：strata、forecast_quarters、calendar months、relative forecast months、cities、五個真價帶；小群也完整報告，只是不觸發 ≥500 門檻。零支持群記 null，零 baseline MAE 不做除零，退步判定直接比較 candidate >1.05*control。

paired 沿既有 project／三個月循環 month 方法，每族 2,000 次，seed sequence `[20260925, stage_index, family_index]`，stage index development=0、offset=1、confirmation=2。offset 不再是 R97 的 index0，須另存新抽樣收據，不能混用舊 CI。共六份 replicates、十二個區間全保留；不能將跨零、勝出比例或三窗重疊結果稱 Private 勝率。群權重仍 seen 0.36617593949822613、新案 0.6338240605017739、missing 0。

convex 組合限制全域外插，但不保留逐列 median 對極端分支的抵抗性；高度相關分支可能使係數不穩定，單一大型市場／高價尾端可能主導等列權 L1。meta-fit 誤差下降不是 query 改善證据。原三個 8 月窗 v24 的有利反證、R97 新窗偏向 v22 的結果不應被本輪偷偷改成權重選擇資料。保留全部三窗和每支單模指標、失敗與不利群，不依第一窗調配置。

## 與舊混合研究及後續範圍

[R2 blend](../src/round2_blend.py) 在同窗選粗網格、base 還曾用該 holdout early stopping；不是本輪「較早完整模型 forward 預測學權重，較晚窗評估」的相同實驗。[既有有限盤點](regression_direction_after_r97_inventory.md) 與 [R84 驗證審核](validation_design_audit_r84.md) 已揭露重用資料限制。

本輪 0 native fit／load，最多 3 次 CPU meta-fit；尚無真實求解耗時，不把 R97 的七模型工時當本輪 ETA。完成後仍不產 CSV、不改正式 v22／PPTX／PDF。若探索全過，先另固定早於 auxiliary 的 forward block 與地理時間權威，再補至少 4 樹＋2 NN 及全部 learned states；本次不是已完成該補驗證或乾淨環境端到端重訓。
