# R105：兩個 prior NN 原生總價的固定等權平均

2026-09-29。R104完整評估及有限來源盤點完成後、讀取本輪逐筆預測或真價前固定。只一候選，不搜alpha、不根據Public或query改係數；唯一提交仍v22，Private未知。反覆使用既有窗口，是適應性研究，不是新盲測。

## 假設与原有反證

R102T原生NN三窗MAE比v22原NN低，但完整流程兩退一進。可測固定等權是否减少單一表示偏移，同時保留原模型的互補性。這是待驗的平均／方差降低假設；沒有多seed重訓或方差估計，不能聲稱已證明variance reduction。R103指出native好不等於base好，median會消去改善也會擋退步；R104直接base loss三窗base相對同列C皆退。舊R87其他模型固定等權也有最新期退步，不能假設平均必贏。

## 唯一候選與兩個端點

每期取已封存R102的原v22 prior NN原生總價P與R102T 145欄prior NN原生總價Q，float64 TWD，嚴格同canonical query列：

`M = .5*P + .5*Q`

`B = .75*(.5*A + .5*M) + .25*L`

A、L、H、U以及reference R、spatial S皆用原同cutoff的固定來源，候選完整預測仍`median(S(R(B)), H, U)`。運算順序保持上式；不平均已修正預測冒稱同方法，不裁價、不按地區／日期／ID／價格分群改權重。

兩個控制明確命名：v22為原完整流程；C為原R102T完整端點，不是R104的loss control或原R102C。各期先把P及Q分別經相同base/R/S/median，native/base/reference/spatial/median五層向量皆需和各端點原保存向量bitwise一致。同窗來源hash、四canonical/raw索引與日期全核對；原reference/spatial pickle狀態、fit/query X與輸入文件在呼叫前後不變。

不新fit、不讀NPZ或native模型、不新建prior/encoder/reference/spatial，不接觸test。使用已保存native向量與已fit的R/S狀態套用預測；R/S屬已學狀態，不能說全程沒有模型狀態載入。本輪cutoff不變，既有兩端和R/S均該cut以前fit；不是用後期模型回測早期。相同修正狀態可能未對新M最適，這是本候選固定設計的限制，不能稱重訓全部最佳流程。

## 執行與時間切分

依原順序development（2022-08-11～2023-08-10，240,316 fit／81,234 query）、confirmation（2023-08-11～2024-08-10，321,550／133,697）、auxiliary（2021-08-11～2022-08-10，134,805／105,511）。不改日期／canonical去重／fit見過專案定義。所有價格標籤來自課程，禁止內政部對答案；沿原2023非價格地理，早期窗口歷史可用性偏差仍在，不稱fully historically-as-of。

先完成source、必要人工公式與metadata介面檢查，再凍結計畫、driver/evaluator source、文件身份、9保護檔及評分參數。driver三窗全部保存／精確重載且execution exit0後，獨立scorer才反序列化raw價格pickle並按原canonical→raw mapping抽取各query y。pickle會整表deserialize，不能說query bytes從未接觸；不保存新query標籤。保存T五層預測／診斷、兩端身份、來源／索引／狀態SHA、執行日志與實測耗時；不覆寫舊結果或失敗，自動重試為false。

## 固定評分與選型

兩端五層指標與原R102結果逐欄重播exact；T的native/base/reference/spatial/median都做原overall／fixed seen-new weighted與全部城市／月份／day-quarter／五真價帶／尾部。seen/new使用完整outer fit，而不是其他子集。固定seen=.36617593949822613、new_nonmissing=.6338240605017739、missing=0。

T-v22與T-C各三窗×project/month×overall/weighted，共24個paired CI；每種2000 replicates，seed20260925，stage索引dev0/conf1/aux2、family project0/month1，連續月份block3、4blocks。保存所有replicate arrays與正反例，區間不是Private勝率。沒有根據結果追加其他alpha、seed、bootstrap或子群候選。

Primary沿原門檻：三窗完整overall/weighted都比v22好、confirmation overall至少改善0.5%、所有季度或至少500列固定群無超過5%退步。另列端點補充：T比C完整overall/weighted三窗均改善，與primary分開呈現；它不是因果機制證明，不要求native勝兩端，不能用native凸性界線冒稱完整流程改善。兩項皆過仍只構成研究候選證據，不自動替換v22或全量訓練／上傳Kaggle。

不生成新CSV，不變更正式v22報告／簡報／選定紀錄。程式、失敗與摘要加入交接；模型方向若未過，保留最強反證並另立下一假設，不用本輪query搜比例。
