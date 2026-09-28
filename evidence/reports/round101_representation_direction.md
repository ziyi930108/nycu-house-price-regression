# R101：時間先驗表示的有限來源盤點

2026-09-29。只讀既有 source、report 與小型 metadata；本次沒有讀真實陣列／價格，沒有建立 prior、fit、predict、score 或 bootstrap。本頁不是 R102 固定計畫；先等 R101 分布診斷，再由主線決定是否立項。沒有更改根交接、舊來源、v22 選定或 CSV，Private 未知。

## 歷史已做什麼

在本次有限查閱範圍內，尚未找到「R100 quarter-prefix factor，加上同源 temporal log-prior 輸入」的完整實驗；不能將這項有限查閱說成鑑識了全部歷史。

|既有方向|實際差別與反證|
|---|---|
|[R65 source](../src/round65_nn_experiment.py)、[報告](round65_report.md)|第124–154行將保存 R60 group-OOF／full-fit 單價取 log，以 fit OOF mean/std 標準化，144→145欄，factor 不變。不是時間 OOF。原生 NN 三窗改善約10.8–19.1萬元，但完整組合最新與最舊退步；原生收益不能代替完整收益。|
|[R71 source](../src/round71_missing_cell.py)、[報告](round71_report.md)|固定 A144/B145 × old/new base 比較，不能只用 R65 舊 base 解釋所有反證。去掉 oldraw 的新 base 下，最新窗口 A144 三種子均勝 B145；三期九組 Bnew−Anew 為3負／6正。當時固定 unit 完整分支亦不是目前 R79，不能直接移植舊最終分數到 v22。|
|[R67 source](../src/round67_nn_experiment.py)、[報告](round67_report.md)|在 R65 的145欄上，再加 selected cell 的 log1p(raw支持列數)，146欄；仍為原 group-OOF，最新改善而另外兩期退。不是本次時間前置 log-prior 組合。|
|[R72 source](../src/round72_nn_experiment.py)、[計畫](round72_plan.md)|在 A144 上加道路 prior 與原 R60 prior 的 log 差，明確不含 R65 絕對 log-prior；不是時間前置。跨窗完整收益不一致。|
|[R73 source](../src/round73_null_road_experiment.py)、[報告](round73_report.md)|已有相同145欄的正零控制先例：只把道路第145欄歸零。與 R72 比較原生、完整模型方向仍可能不同；source可推論相同初始化／隨機呼叫形狀，原始初始權重與 RNG trace 未保存，不能聲稱無噪音因果識別。|

[R100 driver](../src/round100_temporal_prior_experiment.py)的 `masked_arrays`（第60行）保留原 numeric 輸入，只改 T 的 train factor；[R63 trainer](../src/round63_recency_weighted_prior_train.py)第190行先以 numeric/categorical 輸出 ratio，再乘 factor。[R100摘要](round100_summary.md)中 T 原生 NN 三窗均退約157,352／340,775／108,849元；完整模型兩窗改善經過 R/S/median，不能據此認為原生時間先驗已成功。

## 唯一可否證假設與必要對照

假設限定為：**在 R100T 的時間先驗 factor 與 loss 列完全固定時，讓 NN 明確讀到該列先驗數值，能否比同寬零欄更好地校正 ratio，並保留到完整模型。** 這是表示假設；先驗與 ratio 也可能從原 X 部分推知，未證明原問題不可識別，更未證明缺少此欄是 R100 退步原因。

若主線之後固定實驗，最小 matched 比較是兩臂同145欄：候選追加 temporal log-prior，機制控制追加 float32 正零。候選尺度可精確沿 R65：從保存 float64 temporal prior 取 log，以 eligible fit 的 mean/std(ddof=0)標準化，std恰零才取1；query 使用原 full-fit R60 prior 並只套同一尺度。不得從 float32 factor 反推 prior，也不得把 nonchronological OOF prior 放進 temporal factor 的候選欄。

兩臂原144欄、另外7個NPZ陣列、eligible順序、factor、未重正規化日期權重、初始 ratio bias、R63 trainer、seed42／30epochs／batch2048均相同。每期兩 fit、共六 fit；正零控制不取得事後選型資格。保存 R100T144／C 與 v22/v24 可作背景對照，但145對144不是資訊效果的充分隔離。

原生與完整模型需分開呈現；若只經 median 改善而原生不改善，不支持 ratio 表示機制。所有窗口重用、單 seed、CUDA重訓變異仍在。fit 全體計算的標準化尺度，以及保留的完整 outer-fit encoder/query prior/R/S，都不能稱每筆全部處理皆為 as-of。沒有建議搜尋季長、支持門檻、尺度、種子或混合比例。

## R102 最小整合建議：尚未實作

以新檔另存 R100 driver 的窄改版；以下是 source/API 建議，尚無 R102 模型來源、規約或真執行。

|可重用部分|必須改動或核對|
|---|---|
|R100 `save_npz`、`train_arm` 與三個獨立 stage subprocess|沿未改 R63 trainer；`numeric_count == 144`改為145，保留九鍵、全 query reload、參數／權重／factor及head bias核對。新 round／OUT／request／plan／source closure 身分，原檔不改。|
|保存 `outputs/round100/{stage}/T/prepared.npz` 與 stage seal|直接載該 stage 九陣列，不再從原 R63 NPZ mask 第二次。先核 R100 root/stage execution、seal、prepared 的 authority SHA及全部array identity；自身stage白名單須改為此路徑。|
|保存 `temporal_train_prior.npy`、`eligible_positions`與 R60 query prior|以 root→stage→artifact SHA鏈確認正確 temporal prior 及 kept 順序，再與 train factor 構造一致性核對；query prior仍同窗 full-fit。只追加第145欄，保存 fit尺度／新增欄SHA／原144欄SHA／另7陣列不變證據。|
|R65 log標準化數學、R73正零控制規則|新的純陣列prepare函式取 saved T九陣列＋兩個 prior，回兩臂九陣列＋scale metadata；兩臂 factor 此次必須相同。刪除 R100 `masked_arrays` 的重mask／重建factor路徑。|
|R100原控制、`chain(native,name)`、固定R54/LGB/H/U、R/S state不變|沿 `.75*(.5*area+.5*priorNN)+.25*LGB`、各支R/S及最後median。沿用完整 v22控制重播；新增兩臂 native→base→reference→spatial→median及診斷落盤。|
|R100封存／原v2獨立scorer模式|arm名稱、numeric145、唯一差異欄、來源 R100 plan 身分與新 plan 分開；不再聲稱新建 prefix states 或新 prior。全部六 fit 封存後才外部評分；scorer需按新角色重新綁定，不能把 R100C 誤當正零控制。|

應移除 `build_temporal_train_prior`／state_sink及重新建立705份prefix state的呼叫，保存其原 authority 身分即可。新的 fit 不需要再讀完整float64 fit_y或重新抽標籤：其訓練標籤已在該stage保存的 T NPZ，沿SHA／canonical／eligible契約追溯；query價格仍只准外部 scorer。保留原始 full outer-fit indices供seen/new評分，不用eligible loss列定義seen。

成本參考僅為已測值：R100六NN fit合118.203秒，prior準備160.876秒，均含於main391.938秒。若重用prior可省去該次重建工作，但新145欄實耗尚未測，不承諾確切時間；不為本盤點再跑模型。

## 評分器的最小重用方式

[R100 v2 scorer](../src/round100_temporal_prior_evaluate_v2.py)可以只作純評分工具來源，新 evaluator 負責新的 seal／prelabel gate、讀取邊界及結果身分，避免複製整份舊 authority。只讀核過以下介面，未執行：

- `descriptive_metrics(y, prediction, fit, query, start)`（第226行）可直接沿用，同固定價帶、seen/new權重、季度與尾部。`fit`必須原 full outer-fit X，不能傳 eligible NN loss子集。
- `comparisons(metrics, control_name, candidate_name)`（第284行）可處理新role名稱並保存完整點差。但`used_for_research_gate`只在舊名稱`T/v22`為真，不能照搬成新決策；新門檻另由新計畫的小函式判斷。
- `paired(y, predictions, fit, query, query_indices, stage, control_name)`（第261行）的數學與種子可重用：同三窗stage index0/1/2、family index0/1、seed20260925、2000次；但candidate硬編`T`，回傳`CAUTION`仍描述R100C的loss-row對照。最小薄轉接可只在局部字典把新候選映射成`T`，回傳後明確改回新candidate／difference／caution，不修改舊module globals。若不採別名，直接呼叫它匯入的`methods.group_labels/point_values/cluster_keys/relative_months/bootstrap_projects/bootstrap_months/summarize`即可，不需再實作抽樣演算法。
- `select_query_y(raw, raw_indices)`（第219行）可在新gate完成後沿用；全六fit／三stage封存、預測與公式／標準化state／來源SHA／全query重載等須先全部核對，再唯一開raw價格表。保存R100T144與C完整點分數可另核回其原結果，不能重新命名成新的同寬控制。
- 不直接呼叫舊`preflight/check_header/check_arms/run/render/parameters`：它們含R100固定plan、144欄、兩臂factor差異與舊C/T解讀。也不原樣搬`research_gate`的舊role標記。新source closure必須從新evaluator入口計算，包含實際import的原v2及其原contract，保留舊身分、不把它們換成R102 plan。

新scorer只需小型gate與流程／角色輸出，純metrics和bootstrap沿原來源即可；所有R102細節仍待R101結果與主線事前固定，不在本盤點啟動。

## 本次讀取來源 SHA256

以下為本次讀取後核對的來源身份；Markdown相對連結以上方表格為準。只雜湊文字來源，沒有載入真實陣列或模型。

|檔案|SHA256|
|---|---|
|src/round65_nn_experiment.py|3854f5170170f1d0bdfc3707c8e0e04832a430cdbcd780fb7753bef18c96898c|
|reports/round65_report.md|aeb5246a38165b15b0fb0adcc5a748f8ec601f993db450655a4b62adebafd4f1|
|src/round71_missing_cell.py|6fad31c31699d973779031bd4c93e3b846455f6aabc1f4f95e1e92eff7dc60fd|
|reports/round71_report.md|805301837829adda345ce875a214b988a965b3367cee685cb201e3ab12331f46|
|src/round73_null_road_experiment.py|a8561516415fb292eedc6d26a84fac248224d01e90333d0d79ac1a3e03f378a9|
|reports/round73_report.md|b8cc52de91cf6176be4b3d31387061723c2124be719760fb2a4865e0a3abe8aa|
|src/round67_nn_experiment.py|d04b974b749d660f9b5c8c1213f2c34b34fb98a1ff4e4d4c4958fa3d8451b69b|
|reports/round67_report.md|36a6c21aef092098ee13cafeb4651a0e2c84c5ebcdd3a32754320894b9589596|
|src/round72_nn_experiment.py|0513d2246ec9f5572f6eab398f4657ff20a5e0833f5c960f541b71abec14cf93|
|reports/round72_plan.md|7f47f5a25b3c849fde471094c9a4b01d88caf8910f4479429cfbeaf6fc4ad609|
|src/round100_temporal_prior_experiment.py|8d91db83751c6b55e72d0e07a3d0674458af0c2235896e4d04f6c826549b26bf|
|src/round100_contract.py|f6483f2e1e92f8286ea7aed081080c287888b1b6c9f4a3a519964f63ffdd77f5|
|src/round100_temporal_prior.py|6860ed466829f6848140f5fff105b04d5b1d6e31173702bc1348ba29b8937be8|
|reports/round100_summary.md|cc3d6ae624d367a9c8476afacdd968c30b3753230b99581b08274648d92ef2e5|
|src/round63_recency_weighted_prior_train.py|26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a|
|src/round60_recency_price_prior.py|4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f|
|src/round100_temporal_prior_evaluate_v2.py|c12e69764470b9f47a33f1fc1dbae0063a31aff1bfe5568f487d48a08f8792bf|
