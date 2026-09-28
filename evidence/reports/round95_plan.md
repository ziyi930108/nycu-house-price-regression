# R95：R94 Attention 實驗的工程恢復

2026-09-29。這不是新的科學假設或調參候選。沿[原R94固定計畫](round94_plan.md)與原已通過47CPU／v2GPU合成的trainer_v2、CONFIG、三期順序、損失、30epochs、seed、採用門檻，全部保持。

## 為何另開編號／目錄

R94 main在7.813秒、尚未啟動任何真fit時停止：原R48 encoder_state.json沒有獨立.sha256旁檔，但driver錯用locked reader。檔案實際存在，其SHA保存在不可變R48 results的stage artifact_sha256中，原driver也已用這個權威SHA逐檔核過。預先source審查漏抓這個實際讀取契約，保留原審查，不回寫成當時已知。

原outputs/round94、src/round94_experiment.py、launcher/request、execution/traceback及兩個已保存indices完整保留，沒有移走或刪除來繞過拒覆寫守則。新src/round95_experiment.py與outputs/round95、新launcher承接同一設定；R94無模型、無score、無query採用結果，故恢復並非依評分選參數。

## 唯一行為修正

讀R48 encoder JSON時改為 read_json(artifact(48,stage,path))。artifact先用原R48 result中的SHA核實，再讀JSON；不新增或偽造歷史.sha256，不刪任何身分檢查。root另查三期全部locked動態路徑，只有這3份R48 encoder無sidecar，其他18份都有。

其餘更動只是round/candidate/OUT/REQUEST命名與recovery metadata。trainer仍src/round94_attention_prior_train_v2.py，科學DOCUMENT仍reports/round94_plan.md，因此機器text_plan_sha256及trainer.plan_sha256仍c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a；本恢復文件另外列入啟動SHA。

## 執行與證據

[程式差異](../outputs/round95_launcher_20260929/driver_engineering_diff.patch)、[R94失敗收據](../outputs/round94/execution.json)、[R94原source審查](r94_validation_audit.md)。新三期實際進度以outputs/round95與新launcher為準；本文件產生時尚未啟動R95真fit。

資源估算仍是v2合成的約31.18分鐘（三期fit、排除資料/重載/評分），不重跑未變的模型合成測試來製造新證據。v22唯一提交、Public/Private與所有既有模型／CSV不變，不會自动全量／產CSV／操作Kaggle。
