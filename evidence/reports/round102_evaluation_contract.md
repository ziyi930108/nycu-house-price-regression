# R102 評分前契約

此 scorer 是新 145 欄 Czero／Tlog 實驗的隔離評估入口。C 與 T 同用 R100T 的既定 loss 列、價格 factor、標籤與日期權重；C 不是 R100 的舊 mask 對照。原 R100 科學計畫與本輪 R102 計畫分別核 SHA。

只有六個 NN fit、三個 stage、全 query 重載與封存及主程序 exit 0 全部通過，才反序列化原始價格表並按既定 canonical→raw 索引抽取 query y。前置核驗會讀保存的預測向量與純 X；NPZ／模型只核檔案 SHA，不反序列化。

核驗 145 欄、原 144 欄 prefix 證書、七個共享 array 的身分、C 正零欄、T 的 fit-only log 標準化、初始化與 trainer 來源；所有原 v22／v24 向量及完整分支公式亦需一致。新資料轉換的逐值證據由已凍 driver 的保存與重載提供，scorer 不宣稱自行重建 NPZ 特徵。

Primary gate 對 v22；mechanism gate 另要求三期原生 T<C 且三期完整 overall／weighted T<C。T-v22、T-C 固定共 24 個 project／month 條件 CI；C-v22、T-v24 僅 point comparisons。反覆使用窗口與單 seed 限制保留，不能解讀為 Private 勝率，唯一提交仍 v22。

CLI：`--sealed-manifest PATH --evaluation-request PATH --out NEWDIR`。request 必含 `schema_version=1`、新 `plan` record、`parameters`、`source_sha256`、`protected_sha256`、`launch_request` record、固定 `R100_seal` record。來源在主線 freeze 後不再修改。

[Scorer source](../src/round102_prior_input_evaluate.py) · [28 項人工自查](round102_evaluator_selfcheck.json) · [實際 metadata 核對](round102_evaluator_metadata_preflight.json) · [固定計畫](round102_plan.md)。本文件只描述訓練前合約與人工／metadata 檢查，尚未執行本輪真實評分。
