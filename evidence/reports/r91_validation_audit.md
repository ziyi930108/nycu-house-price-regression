# R91 獨立來源、數學與 API 預檢

日期：2026-09-29（Asia/Taipei）。審查者：子智能體 /root/r85_validation_audit。這份文件記錄訓練前已完成的只讀審查，並在主執行緒開始 R91 後落盤；未更動凍結 source、未執行真 fit、未讀取新候選的 query 分數。

**結論：未發現阻擋固定三期實驗的程式或數學問題。** 已先通知主執行緒；主執行緒回報 R91 由 PID 27104 串行啟動。這不是實驗通過、模型採用或 Private 改善的結論，真實 fit、保存重載及資料契約仍須由執行收據證明。

## 凍結身分

| 內容 | 落盤 bytes SHA256 |
|---|---|
| [固定計畫](round91_plan.md) | 7cd2c36668326975db32142853c3846fd286fcbba37148e70df68def4c81eb1e |
| [實驗 driver](../src/round91_experiment.py) | f590c97467906bada203dd10475b567770aca05891b83ae39ac768401b658636 |
| [signed residual helper](../src/round91_prior_residual.py) | c52339be9301bfd4fd4fd2eb7a85647fb10769726af9e15266bbada68e507cbf |

以上三個 SHA 在落本文件前以 read_bytes 再核對一致。原先口頭傳遞計畫 SHA 0c52443b134a3248af149fb0a7b7e83307d52959df6ec9a6b81f60f35da4a2e4 是記憶體 LF 字串的 hash；Windows write_text 產生 CRLF 後，落盤身分為上表。主執行緒確認未變更計畫內容、假設或門檻，也未在該差異期間 fit。這是檔案位元組表示的核對修正，不能寫成看結果改計畫或失敗模型 fit。

## 數學與訓練目標

沿用原 effective_area：有效正 building_area，否則有效正 parking_area，否則 1。fit offset 為 B = A × P_OOF，單位 TWD；P_OOF 是同期 fit 資料產生的分組 OOF 單價先驗。

新 target 為 t = (y − B) / 1,000,000，可為負、零或正。Pool 權重僅為 R63 已保存 float32 日期權重逐值轉 float64，沒有乘面積、沒有重新正規化。對未投影預測 z，有：

date_weight × |t − z| = date_weight × |y − (B + 1,000,000 × z)| / 1,000,000。

query 使用同一 fit 所建立 full-fit prior，先還原 A_query × P_query + 1,000,000 × z，再投影至非負總價。helper 明確保留負的 z，不能先把殘差截成零。對齊、長度、有限值、正 prior、float32 日期權重及 overflow 均有檢查。

只有 target／相應權重／還原公式改變；既有 95 欄數值、dtype、順序、index 及 23 個類別欄均沿用 R19，沒有把 prior 或 offset 加到模型特徵。CatBoost 仍為固定 2,999 樹、depth 8、learning_rate 0.065、seed 42 及原 native 參數；沒有 eval_set 或 query early stopping。

## OOF 來源與 fit/query 隔離

1. driver 綁 R79、R83、R60 既有 plan/results/source SHA，要求 R60 與 R79 split_contract 相同，並核原三期 fi/qi 的 dtype、shape、完整內容及日期界線、互斥與列序。
2. 原 X 與 tree geo frame hash、fit y hash、原 R63 prepared fit million labels、日期權重值及日期旗標來源皆核對。R63 日期規則固定為既有 365 日設定，不以新誤差調整。
3. 每期以 price_prior.fit_oof_prior(Xfit, yfit, fi) 完整重播 R60；OOF 值、fold ID、full-fit state 及證據均須與保存狀態一致。query prior 必須等於 predict_prior(saved_fit_state, Xquery)。fit/query 的 float32 A × P / 1e6 factor 也須與 R63 prepared 完全一致。
4. fit_model API 僅接 fit X/y/geo/date/OOF prior；predict_model API 接 query X/geo/prior，不接 query y。fit 使用 OOF offset，不能以 full-fit prior 回填 fit，不能用 in-sample v22 預測當 offset。
5. 既有原始 loader 會反序列化完整 train 標籤向量；這不是純 X loader。driver 從中取 yfit 傳入 prior／model，而 yall[qi] 僅送往 score，不進 helper、權重、prior 或 target。與 R90 嚴格不載價格的純 X preparation 是不同範圍。

R60 是 group OOF，並不是 chronological OOF；同一 fit 窗口較後的交易仍可能在其他 fold 的 reference 中。價格先驗只能降低指定分组的同群標籤直洩，不等於消除所有建案關聯或時間分布差異。OOF prior 與 full-fit query prior 的支持量不同，存在 offset 分布落差。

## 公平對照與完整預測鏈

- 固定 .75 × 新 residual CatBoost 總價 + .25 × 原同窗 LightGBM，未另調比例。
- 原 NN／log 分支沿用同窗保存預測；unit 分支以自己的新 base 重新執行固定 reference、spatial correction，沿用合法 fit state，不新增價格狀態或從 query 回饋修正。
- driver 先核原 v22 的 base → reference → spatial → 三分支 median 完整向量一致；也核 v11/v12/v5/v22/v24 完整 score dictionaries 一致。新結果保留控制組，符合既有 generic 與 paired 診斷格式。
- 原 reference/spatial state 的 fit/date/source、residual 及 geo 身分均核對；spatial fallback 與 reference 完整一致。
- 三期預先固定且全部執行，無依第一期分數停止或改配方。v22 為主對照、v24 為次對照；原本事前三期 overall/weighted 皆改善及最新 overall 至少 0.5% 的門檻保持。

## 保存重載與 metadata

helper wrapper 保留原 R19 的 pure-X legacy_feature_state，但把新的 signed target、offset、date-only 權重與 inverse 身分放在外層；legacy 訓練欄位只作來源紀錄，不沿用舊單價還原。wrapper 檢查 feature hash／規則、95／23、native 參數及舊 inverse 混用。

driver 每期保存真 CatBoost 與 JSON state，重新載入後要求全部 query raw signed z 及還原 TWD 逐 bytes 一致，並核對公開 predict_model 與手動 inverse 完全一致。metadata 分別保存原 y、signed transformed target、OOF prior、offset、date/pool weight 雜湊；另外記 stage、fit/query indices、原 frame hashes、source/provenance、fit 時間與 reload 結果。不能把合成重載測例代替這些真模型執行檢查。

## 已有合成驗證與審查邊界

[helper 自測收據](round91_helper_selfcheck.json) 是作者 /root/r85_error_audit 實際執行的首輪 41 項純合成檢查，exit 0、0.140 秒，helper SHA 與本文件一致；本次獨立審查沒有為寫報告重跑。

涵蓋 signed target、date-only Pool 真參數路徑、負殘差 offset 還原順序、95 欄／23 類別、錯序／非有限／overflow 拒絕、legacy state 精確性、FakeModel＋JSON state 重載、舊 unit inverse 拒絕、native seed／feature capacity 檢查及不污染輸入。自測只有一次 FakeNative fit，沒有真資料、真模型/config load 或真 fit。

獨立審查讀完 plan、driver、helper production API 及合成測例，核相關舊 source 行為與小 metadata、AST／SHA；未啟動新訓練、未碰 test、未改 CSV 或正式 v22 產物。實驗仍使用反覆看過的年度窗口，並非新盲測；native GPU seed 不保證完全 deterministic。即使通過保存模型重載，也不等於從零重訓完全一致，更不代表未知 Private 已改善。

## R90 接續狀態

另項授權的 [R90 純 X 準備報告](round90_preparation_report.md) 已完成固定 2023-02-11 新切點資料準備與完整 roundtrip，沒有價格讀取、encoder/prior/reference/spatial fit、native 訓練或分數。是否為 R91 重建六個 native 模型，應依 R91 固定門檻與主執行緒後续決策；這份预檢不授權或宣稱新切點模型驗證完成。
