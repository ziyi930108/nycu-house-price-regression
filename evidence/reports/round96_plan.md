# R96 固定 CPU 對照：局部 reference confidence 折半

2026-09-29。主線在 R95 Attention 三期仍執行時授權這個獨立假設；R95結果不作選rho或分支依據。本計畫先於R96真資料候選prediction／評分保存。研究源於已看過的歷史報告，adaptive_after_error_audit=true；不是新盲測，不保證改善。v22仍是使用者唯一提交，Private未知。

## 假設與唯一變更

局部reference的c由層級、距離與MAD工程公式決定，沒有被校準為「局部價比原模型可靠的機率」。原生模型改善可能未被完整組合保留，但層間MAE差不是因果證據。固定降低這個局部信任程度，保留時間外推與全部原fit狀態；只做rho=0.5，不搜尋比率、seed、type/use、城市、ID或真價分流。

對v22的NN、log、unit三支分別使用自己的原base B、原trend reference輸出P與診斷c：

```text
future_days = minimum(maximum(nan_to_num(query_day - fit_anchor, nan=0), 0), 366)
Bfuture = B * exp(saved_beta[3] * future_days / 365.25)
c_new = 0.5 * c
P_new[c > 0] = Bfuture[c > 0] + 0.5 * (P[c > 0] - Bfuture[c > 0])
P_new[c == 0] = P[c == 0]  # copy原bytes，避免無效浮點重算
U_new = original_spatial.predict(state, Xquery, geoquery, B, P_new, new_diag)
final_R96 = median(U_new_NN, U_new_log, U_new_unit)
```

new_diag只更換confidence，source／nearest_distance／reference_dispersion／index原樣copy。spatial因此使用(1−0.5c)*alpha，其餘eligibility／G／beta／anchor／半徑／支持數／clip不變；這不是「只改reference、spatial有效強度完全不變」。固定公式在已非負的原P上收縮，不重新從局部價格推算P；與(1−0.5c)*Bfuture+0.5c*local_price數學等價的說明，不當成浮點逐位相同保證。原生輸出／base不乘新c。

## 原API與邊界

原R2輸出P=max((1−c)*Bfuture+c*local_price,0)。local_price的clip基準為max(B,1)，但Bfuture必須用原B，不能用max(B,1)或先套reference的價格；B=0也不能錯改成1。原NaN日期令future_days=0，+inf→366、−inf→0；保留原nan_to_num的順序及預設infinity行為。原reference仍可能對缺日期以fit填補後找到局部參考；不可自行把缺日期c歸零。原S另要求日期有限，故缺日期不套S。

原reference診斷reset為RangeIndex；真driver將Xquery依保存canonical索引取出後reset_index，再嚴格驗diagnostics.index相同，不能無聲按標籤重排。base/reference必為同序float64有限非負向量；c=float64且0≤c≤1，source是原固定來源集合。base_fallback應c=0；有來源但c數值下溢為0亦copy原P。c=0原P須與Bfuture數值相等，且新reference和最終S輸出都與原對照逐位相同，即使原S有套用也不能改。c>0但剛好P=Bfuture時，reference可不變而新的spatial份額仍可能改價。

舊R2／R8 source、模型parameters/state皆不就地更動，不monkeypatch舊常數、不移除契約。新純helper只由原state讀取beta/anchor，無fit、新殘差或新模型state。helper收reference_model、Xquery、原B/P/diagnostics，回newP/newdiag/JSON報告；不接任何query標籤，也不載真資料檔。

## 同窗控制、合成與保存

主對照v22、次對照v24；沿原R79 development／confirmation／auxiliary的canonical索引、日期、原R6/R39地理與原reference/spatial來源SHA。原三支unit=R79、NN=R71 seed42、log=R46，不混R85–R95模型。三期皆完成，不依首期分數換假設。

每期先逐支重播原R/S/完整diagnostics與保存向量，再重現v22 median；需exact，失敗即停止保留。候選只執行三支新reference收縮和原S，0次新native fit／原生模型load/predict／reference-spatial build；不讀test資料、不新增test向量或CSV。真driver可讀queryy只作評分；不能把整個流程宣稱從未反序列化query標籤。helper人工合成不讀任何真X/y。

合成至少驗：原future計算及366/NaN/±inf邊界、B=0/小於1、c=0精確不變、rho=1原控制（僅內部測例，不是第二候選）、c=.5/1固定算式、wrong dtype/shape/index/state/NaN/負值拒絕、輸入不變、排列等變、pickle重載、真R2和R8在人工state下的weak+spatial／高層來源／fallback+spatial／面積不合格／缺日期／零base。S的source/applied/eligible/G不變；新confidence与effective權重一致，c=0完整輸出exact。

新產物：src/round96_reference_shrink.py、src/round96_experiment.py、reports/round96_helper_selfcheck.json（失敗另attempt保留）、outputs/round96/。記source/input/plan SHA、Bfuture/c與候選三支向量、完整診斷與mean/zero/changed counts、重載精確證據及實測秒數。舊CSV/selector/user_selection/protected reports核前後SHA；helper測試可做人工資料R/S，不冒稱零合成predict。

## 事前評估與停止

固定TWD MAE和原新舊案加權；整體、新舊案、城市、季度、五真價帶、每支base→reference→spatial→median全部保留，真價分帶只診斷。沿原建案與三月循環月份區塊bootstrap、2000次／seed20260925，v22/v24共24區間；不是Private勝率，也未校正累積選型偏差。

必要門檻：三期overall及原fixedweighted都優於v22，且最新overall至少改善0.5%；仍需保留尾部／新案／季度反證，通過不是自動採用。未通不改rho／branch／群政策重試，不final／CSV。通過才考慮已固定新cutoff的穩定性檢查；新cutoff v22/v24/R96共同重建需7 native（5樹＋2原NN），R96 wrapper本身0fit，不沿Attention的8fit數。新cutoff仍非盲測。

## 最強反證與歷史區別

原reference明顯降低多數base MAE；折半可能丟失可靠同案／同路支持，並放大粗地理spatial補充及尾部誤差。c=0保持原趨勢，因此本方法不修無支持廠辦的共同trend，也不修巨大面積外推。R88新案三期退说明「較少修正」不必較好；R85原生改善未傳到median亦不能直接指認R/S有害。

R82改unit trend→anchor（去額外base趨勢）；R88只在base_fallback且無S時回base；R89重估type/use斜率並改殘差與新S state。R96保留trend/state而全三支只縮c，與三者不同。R82／R84已有信任強度提案、R2早期亦有reference融合，故不是首次提出參考混合；有限source/report盤點未見這個固定完整v22政策已被執行。[方向審查](regression_direction_after_attention_audit.md)、[R82盤點](round82_followup_audit.md)、[R84公式](reference_blending_audit_r84.md)。
