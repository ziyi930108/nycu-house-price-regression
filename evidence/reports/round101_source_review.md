# R101 有限來源交叉審查

2026-09-29。結論：已讀固定計畫、統計 helper、stage driver 與 metadata-only parent，未發現阻擋問題。審查者不是本輪程式作者，但曾編寫 R100 純 X audit 並核驗保存結果，不聲稱全研究完全獨立。

- `src/round101_statistics.py` SHA：`7b55dd9d189b1c51ac4de933b24c954b96debe714d5006b0e2964933774d6d30`。
- `src/round101_prior_diagnostics.py` SHA：`22708cefde4ab7fda769954bd6d3564728e2f517eb5e40c97dddedeb40346308`。
- 另讀 [固定計畫](round101_plan.md)、[父程序](../src/round101_launch.py)、保存 metadata 與 execution；三期實際 exit 0、父程序64.359秒。本審查沒有重跑程序或人工測試。

核對一般 linear 分位數、加權 inverse-CDF、population variance、ESS 與全 scope weight-mass 分母；空群不刪除，weighted median 不冒稱等同普通 median。固定日曆季、365／730日 age、25種支持轉移、query12月份與完整 outer-fit seen/new/missing 分組符合計畫。

driver 限定各 stage 的來源 readset，父程序只讀 metadata。NPZ先驗精確九鍵，再逐鍵讀取，C/T八共享陣列與原C切片精確相同；實際 float32 y/factor 提升至float64計ratio與初始化，原日期權重不重新正規化。舊OOF支持只重算純X raw counts；T及query支持只讀保存cell rows，不重建價格prior。query ratio來自保存native價格反算，沒有query真價或新model inference。

限制：本次只读 source／aggregate metadata，沒有載入任何真實陣列、fit/query y、NPZ或模型；沒有重算描述統計、prior、MAE或bootstrap。沒有做完整端到端重現，來源審查也不能證明分布差异导致退步、未来改善或Private表现。原保存reference摘要未含ESS，程式明示null，未由支持列數虛構ESS。v22仍為唯一提交。
