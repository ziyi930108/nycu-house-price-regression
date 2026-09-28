# R102 有限跨作者來源審查

日期：2026-09-29。結論：在本次只讀範圍內，未發現阻擋執行的問題。這是執行前的來源與契約審查，不是模型有效性或執行結果的認證。

審查者為 `r85_error_audit`，不是 R102 driver 或 scorer 作者；曾參與 R100 pure-X 政策／稽核與 helper 審查，因此不宣稱整項研究完全獨立。

|固定來源|SHA-256|
|---|---|
|[R102 計畫](round102_plan.md)|`ee59dba0da8abb79c88cb6e1b73a6417a91944de784f3c7f77e61755a68bc355`|
|[driver](../src/round102_prior_input_experiment.py)|`c1ad2f6070a7ead6adf5eaf9cb289703d507f64609f65610714f3b9e4a2c4edf`|
|[scorer](../src/round102_prior_input_evaluate.py)|`48a05e6eb2596e20a7a9182e6ed04d9feb0ada8b49e2c779142444836f4642fe`|

審查涵蓋 driver 全文及最終新增的原 R100 source／保護檔核對、同窗口資料讀取限制；scorer 全文及沿用 R100 v2 的公式、比較、門檻和配對抽樣函式。未修改任何來源。

- 兩組沿用同一窗口已保存的 R100 T 九陣列與 eligible 列序；原 144 欄保持，C 追加 float32 正零，T 追加以 eligible fit 的 log-prior 平均與母體標準差轉換之第 145 欄。query 使用同一組 fit 統計量。其餘七陣列、factor、原日期權重、標籤及初始化 bias 一致；不重新建 prior 或正規化權重。
- driver 僅開啟該窗口 R100 T 的 NPZ；原 R100 計畫與新 R102 計畫身分分開驗證。六個 NN fit、三個 stage 及完整 query 重載成功後，scorer 才反序列化 raw price；query 標籤依 canonical-to-raw 映射選取。原 v22／v24 完整預測與 v22 分支重播、固定 base 公式及三分支 median 都有核對。
- Primary 門檻為 T 對 v22 的三期整體及固定加權 MAE 均改善、confirmation 至少改善 0.5%，且無固定季度／足夠樣本群超過 5% 的退步。Mechanism 門檻另要求 T 對新 C 的 native MAE 三期改善，且完整模型整體及加權 MAE 三期改善；不能用後處理收益替代 native 改善。新 C 是同寬度正零對照，不是舊 R100 的 mask 對照。
- 配對區間只做 T−v22 與 T−C：三期 × project／month × overall／weighted × 兩對照，共 24 CI。每組 2,000 次，沿用 `SeedSequence([20260925, stage_index, family_index])`；C−v22 與 T−v24 只報點估计。條件 bootstrap 不代表 Private 勝率；反覆使用窗口也不是盲測。

限制：本次僅讀 source／計畫並核 SHA，未載入真實陣列、NPZ、標籤或模型，未 import／執行 driver 或 scorer，未新增訓練、預測、評分或 bootstrap。scorer 對 NPZ／模型採雜湊及保存 metadata 身分驗證，並不自行重新計算第 145 欄；其構造依據是已審查的固定 driver。真實執行結果、聚合解讀和產物仍需完成後另行核對。v22 唯一提交選擇與未知 Private 保持不變。
