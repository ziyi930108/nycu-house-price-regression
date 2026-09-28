# R97 評估工具與標籤隔離契約

日期：2026-09-29。依[事前固定計畫](round97_plan.md)實作；這份文件是程式介面與人工測試紀錄，不是新窗口的實際評分結果。新窗口為 fit < 2023-02-11、query 2023-02-11～2024-02-10（含），分別 272,117／107,169 列。唯一已選提交維持 v22，Private 未知。

## 執行介面與封存

```powershell
.venv/Scripts/python.exe -B -X utf8 src/round97_offset_evaluate.py `
  --sealed-manifest outputs/round97/sealed_manifest.json `
  --evaluation-request outputs/round97_launcher_20260929/evaluation_request.json `
  --out outputs/round97/evaluation
```

`evaluate(sealed_manifest, out, evaluation_request)` 接受上述三個路徑。輸出目錄必須不存在，失敗保留現場及 execution.json，不覆寫、不自動重試。匯入模組不會評分或讀取資料；只有明確執行此介面才會載入已封存預測。

由於主訓練可以先開始，評估來源另行凍結，不要求模型 driver 的 source map 包含後完成的 evaluator。獨立 evaluation request 必須含以下精確欄位，並附同名 `.sha256`：

```text
status = "evaluation_frozen_before_query_score"
parameters = evaluator.parameters()
source_sha256 = evaluator.source_hashes()
plan_sha256 = SHA256(reports/round97_plan.md)
```

科學評估設定在原計畫中已先固定；另行凍結的是評估程式實作。source_hashes 回傳 10 個實際本地 import 依賴的 SHA。模型 driver 的來源與其產物仍按原封存驗證，沒有移除舊 helper 的驗證。

評估程式依序驗證：

1. seal schema 1、正確起迄日期、5 tree + 2 NN 全部完成、兩個完整預測與四分支三層皆存在；driver 明示未使用 query 標籤或產生分數。
2. 全 query 模型／state 重載一致、輸入未變、保護檔前後一致、提交仍 v22 且沒有新增 CSV。driver execution.json 必須 exit 0。
3. 新 evaluation request、計畫、driver seal、所有宣告的 source／input／artifact 檔案 SHA。讀取模型檔的位元組計算 SHA 不等於載入模型或執行推論。
4. R90 純 X preparation 的固定 manifest SHA `400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7`、原始 schema、日期分割、行數、frame 值及 dtype、四個索引向量。
5. canonical 索引透過原始完整 mapping 精確對到 raw-cache 索引。每支 native／base／reference／spatial／完整預測均須為有序、有限、非負 float64 TWD 向量，檔案與數值 SHA 皆一致。
6. 四個 base 公式及兩個完整 median 必須逐位重現。U24 是兩支 unit tree 平均，沒有 LightGBM 比例；不能把單支 native 冒稱完整 v22/v24。

通過後先保存 `prelabel_gate.json`，才呼叫 query label loader。loader 重新核 raw train SHA，`pd.read_pickle` 會反序列化整張來源表；之後僅以 **raw cache 的 `query_source_cache_indices`** 擷取 `總價元`。不能用 canonical query_indices 或重設後的 query 行號擷取真價。評估不保存逐筆價格或 ID，只保存所用 query y 的數值 SHA。原 fit 標籤隔離程序的整表反序列化事實仍依原計畫揭露，不能聲稱整個專案從未碰過 query 價格。

## 固定指標

共評估 21 個已封存向量：2 個完整版本、7 個 native，以及 N／H／U22／U24 各自 base、reference、spatial 共 12 個。全部用同一組 query 價格與列序。

- TWD 整體 MAE，以及原固定新舊建案權重：seen 0.36617593949822613、new_nonmissing 0.6338240605017739、missing 0。沒有重新讀 test-X 估權重。
- 新／舊／缺名建案、所有城市、日曆月份；四相對季度沿原 scorer 的 0–91、92–183、184–275、276 天至末日，並額外呈現 12 個以每月 11 日為邊界的相對月份。
- 五個真價帶：<1,000 萬、1,000–2,000 萬、2,000–5,000 萬、5,000 萬–1 億、>=1 億。下界包含、上界不含，完整分割 query。保存列數、MAE、平均帶方向誤差、低估比例及絕對誤差占比。這些是事後診斷，不能用來推論分流。
- 尾部：絕對誤差中位數／90／95／99 百分位、最大值、總誤差、最大 10 列及最大 ceil(1%×列數) 列的誤差總和與占比。各模型按自己的誤差排序，並非同一組交易；不提供逐筆 ID。總誤差为零時占比記 null。
- 完整 v24−v22 差值；同時整體與固定加權下降才標記本窗口點估計一致支持。每個季度或至少 500 列分群退步超過 5% 都列入不利清單，不因其他指標改善而刪除。

原 `round3_evaluate.score` 與 `v3_cluster_stability` 的分群／抽樣函式直接沿用，未呼叫它們讀取舊窗口或 test 的 entrypoint。

## 配對抽樣

只有完整 v24 對完整 v22 的固定比較：project 與 3 月循環月份區塊各 2,000 次；seed 20260925，單一窗口 local index **0**，family index 0／1，`default_rng(SeedSequence([20260925,0,family_index]))`。

project 在 seen／new／missing 各層抽完整名稱／地理群；缺名有原地址、道路、單列備援。群名不代表已認證的實體建案。month 每次抽 4 段連續 3 月，保留原循環邊界假設。兩模型始終配對；正權重層缺席時 weighted 保留 NaN 並記無效次數，不重抽。報告兩指標共 **4 個** 2.5／97.5 百分位區間，完整 `(2000,4)` arrays 另存並驗保存重載一致。

這是對已保存預測、已使用且重疊的歷史資料的條件性診斷，不是新盲測、Private 勝率、p 值或修正多重選型後的檢定；不涵蓋重訓與未來市場分布移轉。工具沒有選型器、fit、模型載入、predict、test 匯出或 GPU 呼叫。新增時間切點不能抹除先前累積的選型偏差。

## 人工測試與產物

[final CPU 收據](round97_evaluator_selfcheck.json)與 [attempt3](round97_evaluator_selfcheck_attempt3.json)位元組相同：**43 項通過、0.110 秒**，僅合成資料，不讀真實價格／預測。包含未封存先拒絕、獨立 request 來源／參數拒絕、raw-cache 位置而非 DataFrame 標籤取價、日期邊界、手算 MAE／固定權重、missing 零權重、價帶邊界、尾部零誤差、固定差值的兩種 paired 抽樣、相同模型、缺層不重抽、四 base／median 公式、錯誤 SHA／dtype 拒絕等。

[attempt1](round97_evaluator_selfcheck_attempt1.json)原有 36 項已通過。增加封存測例時，[attempt2](round97_evaluator_selfcheck_attempt2_import_failure.json)曾因多一個括號在 import 前發生 SyntaxError；未讀資料、未訓練、未評分。修正後完整重跑 43 項，不隱藏失敗。[原失敗 source 保存與 SHA 證據](../outputs/round97_evaluator_prefreeze/manifest.json)保留了該版本；是反向還原單一括號後對失敗收據 SHA 精確驗證的位元組副本。

凍結 source：`src/round97_offset_evaluate.py`，SHA256 `fbe0b0f841fce7f13afa6e499b6f88498ae061c816ab381fc89a5357eb87d0d8`。final 收據 SHA256 `08c24fd18f12dca3652d71b07b67c1b640a9c98f13200628a417fda0f7e3ce47`。此文件未宣稱實際評估已完成；之後由 root 在主流程封存且 exit 0 後執行。

成功輸出目錄內含 `results.json`、`report.md`、兩份 bootstrap replicates、`prelabel_gate.json` 和 `execution.json`；JSON 有 `.sha256`。結果保存完整向量指標、配對抽樣支持、所有來源與輸入 SHA、執行時間、query y 身分及限制。原 HANDOFF／GitHub 整理由 root 統一處理。
