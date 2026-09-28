# R100 temporal prior helper 來源審查

2026-09-29。結論：在下列來源與已讀 driver callback 契約下，未發現阻擋問題。這是跨作者來源審查；審查者同時是純 X policy／audit 作者，並非全流程完全獨立。helper 由 root、driver 由另一代理實作。本次沒有載入真實 X／y、建立真 prior、訓練、預測或評分，也沒有重跑合成測試。

- [helper](../src/round100_temporal_prior.py) 在完整 outer-fit 列與原 canonical indices 上產生 cluster/fold，之後才取 eligible mask；對保存 folds 和 mask 做 dtype、shape、bytes 相等核對。只靠數值無法辨識錯配的 y，因此原始列身分仍由 driver 的來源／frame／index SHA 契約負責。
- [純 X policy](../src/round100_temporal_prior_x_audit.py) 按日曆季分組，整季五折 reference 各至少30列才保留。helper 的 reference 明確為 valid date、day < quarter start、fold != held；同季、未來與整個 held 建案折均排除，另核 reference／held 群無交集。無效旗標須有限；暖身列沒有虛造 factor，最後不完整季度不另裁列。
- [R60](../src/round60_recency_price_prior.py) 的 `_build`／`_predict` 直接重用，四層鍵由完整 fit X 只建立一次。每個 prefix 以自身合法 reference 定日期 anchor、365日權重、原30列 cell 門檻與 weighted-median global fallback。未建立新的 query state；full-fit query prior 仍由 driver 重用。
- 回傳 prior 是 kept 原序的正有限 float64，kept positions 為 int64。每個非空 quarter/fold 記 reference／held canonical identities、reference y／unit／三日期欄 SHA、state content SHA、支持與命中層級，且 JSON state roundtrip 預測逐位相同。
- `state_sink` 是明示可選 callback。已核 [driver](../src/round100_temporal_prior_experiment.py) 的 callback 用新 quarter/fold 路徑、exclusive JSON 寫入與 file SHA 回傳，不修改 state；因此這個實際接點會保存全部使用到的 prefix states。純 helper 未傳 callback 時則明示未保存，不宣稱完整 state 備份。callback 的一般安全性不是 helper 自行驗證的保證。

已讀 [root 的19項合成收據](round100_helper_selfcheck.json)：passed、0.188秒，source SHA 與已審 helper `6860ed466829f6848140f5fff105b04d5b1d6e31173702bc1348ba29b8937be8` 相符。包含季邊界、暖身、最後partial quarter、state callback／JSON、同季／未來／held價格不變、R60精確重用、列序與拒絕錯誤輸入；列序反轉測例使用 rtol=1e-14，不能宣稱所有列置換都逐位相同。

這些檢查支持實作符合固定規約，不能證明較低 MAE、舊 grouped OOF 有外層洩漏、或 Private 更好。真實執行／全部封存／評分仍由 root 控制；v22 唯一提交不變。來源 SHA 與讀取邊界見[機器收據](round100_helper_source_review.json)。
