# R94 獨立 source preflight

2026-09-29（Asia/Taipei），審查者 /root/r85_validation_audit。**最終 trainer v2、driver及其收據沒有尚待修正的source／API blocker，可由主線依固定計畫串行執行三期。** 真實資料重播、模型fit及全query重載仍須以正式執行收據確認，不在此預稱完成。

時序：本代理先以訊息回報最終source無blocker。主線回報已於2026-09-28T17:46:53.326Z啟動PID2976，當時launch request如實標記書面報告待到；**本書面在啟動後落地，不能稱書面審查先於launch**。主線已凍結21檔，後續本代理未改任何模型／driver source。

本次僅讀source／小型metadata／保存收據，核AST、來源SHA及authority鍵／路徑；未開NPZ／NPY／pickle／model，未讀逐筆價格、query labels或test，0 GPU／fit／predict／score。沒有修改別人source、R93快照或舊模型。

## 最終身分

|檔案|SHA-256|
|---|---|
|[reports/round94_plan.md](../reports/round94_plan.md)|c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a|
|[src/round94_experiment.py](../src/round94_experiment.py)|ca1823069d1310409bfb7a8a04c09177ed2ca5e15496f66a99b769ac70948563|
|[src/round94_attention_prior_train_v2.py](../src/round94_attention_prior_train_v2.py)|ae197952be777f3e9e2fdb66458887d2a3885fd736b75dd3677eeae43cfeef83|
|[reports/round94_trainer_selfcheck_v2.json](../reports/round94_trainer_selfcheck_v2.json)|bf2ba66b41ff5932988428243dad183c62cbec23bf49ae0ad2cfc42f191a1050|
|[outputs/round94_smoke_v2_20260929/smoke.json](../outputs/round94_smoke_v2_20260929/smoke.json)|55279c54c0e25d55b1a004ca2d2bcf68890ed86fb03d40f37bf0b37bd37ece3f|
|[outputs/round94_smoke_v2_20260929/execution.json](../outputs/round94_smoke_v2_20260929/execution.json)|299d74e9ba4919bb4a011aed4c2d74a34ad1d660bd23ffe5d1a5cdb8c09839a9|

詳細契約沿[上游審查](round94_upstream_review.md)與[整合盤點](round94_integration_inventory.md)；本次機械核對見[receipt](r94_validation_audit_checks.json)。p48與p79歷史source先逐檔核驗且拒衝突，避免當前SHA覆蓋歷史身分；三期driver所需artifact authority鍵與路徑亦已檢查存在，本次未載真陣列。

## 已審查的核心邊界

- 每期fit前核canonical fi/qi、cutoff、frame；NN編碼使用原R6 geo，R39 geo僅供spatial。R47 encoder從fit X重播全state及兩側編碼；R60以fit y和canonical fi重播OOF／folds／full-fit state／evidence，再還原兩側factor32及R63日期權重。沒有用raw cache index冒充canonical，也沒有用後期state回測。
- trainer v2先驗NPZ九鍵再讀成員，沒有query y介面；head只用fit ratio median，30epoch最後一輪，無early stop。driver共用loader會載canonical y，query y僅供score；不能把此說成整個程序沒載query價格。group OOF不是時間OOF，prior含fit標籤。
- 原控制重播R54＋R63 native→R71 seed42 mean/base/reference/spatial與diagnostics，再與R46 log＋R79 unit組median精確比對v22。沒有誤取R63含raw的舊base或R40 unit。候選只換prior native，固定 .75*(.5*area+.5*prior)+.25*LGB，原R/S各一次、原三支median與完整metrics介面保持。
- tokenizer168×64、4heads／2blocks、首attention省norm、ReGLU及四投影bias0符合來源。末層CLS query配全KV，residual取未normalized原CLS；另有獨立upstream broadcast residual參考路徑，同權重eval輸出／全參數gradient核對，不拿同forward自比。
- 加權L1為date32*abs(ratio*factor32−y32)的列均值；真train_group每microbatch sum除actual optimizer group列數，尾組不除固定2048或權重和。513列256+256+1測例驗真正函數gradient與step。還原float64(ratio)*float64(factor32)*1e6，factor僅一次，不clip。全參數單組AdamW與固定schedule不變。

## 測試、成本與已修正工程問題

v2 CPU47項passed、1.250秒；同權重eval輸出最大差1.11e−16、gradient8.88e−16，沿已聲明float64容差。不同batch／排列最大差5.96e−8（rtol2e−6／atol2e−7），不冒稱不同shape逐位相同。相同batch CPU保存重載ratio/TWD逐位相同；真模型全query重載尚待正式fit。

主線v2 GPU smoke2 warmup＋3 timed groups，每2048列均值0.182333秒，10,260 steps推估三期fit1,870.74秒，低於預定5400秒；allocated峰值565,207,040 bytes、reserved746,586,112 bytes，exit0。這是合成估算，不是真fit耗時；不含全部真prepared陣列常駐、來源重播／重載／診斷。v1收據保留，但最終driver只接受v2相符source／CPU／smoke。

訓練前兩項修正均不改模型配置，也不是看候選分數後修法：

1. 原driver預建model目錄與trainer拒覆寫衝突，兩邊也寫同名execution。最終driver將開始／stdout／外層execution放stage/trainer_process；trainer獨占model目錄，stdout亦不再進模型artifact hash。
2. v1先materialize NPZ才驗鍵；v2提前九鍵驗證，只讀指定成員。v1原檔及smoke保持，[工程diff／收據](../outputs/round94_launcher_20260929/trainer_v2_engineering.json)另存。最早CPU double累加器測例失敗已保留，修為numeric dtype；真float32公式不變。

舊控制沒有同期重訓；架構／embedding維度／dropout位置同時改，不能把結果單獨歸因attention。相同seed與randperm API不代表不同架構逐位同序列。年度窗口反覆選型，不是新盲測；bootstrap不是Private勝率。三期門檻與失敗保留不變；即使通過仍需R90新切點重建全部合法狀態。Private未知，v22唯一提交保持。
