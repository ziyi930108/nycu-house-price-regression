# R90 純 X 新窗口資料準備完成

日期：2026-09-29（Asia/Taipei）。主執行緒在 R90 helper 凍結及只讀審查後，另外授權只執行一次純 X preparation。此紀錄晚於 [先前未執行審查](round90_preflight_review.md)。

**真實新窗口資料準備已完成，未訓練、未建立encoder/prior/reference/spatial價格state、未讀任何價格表或test。** 因此這不是完整v22/R89新窗口驗證，沒有新分數。

## 範圍與結果

固定 fit < 2023-02-11：272,117列；query 2023-02-11～2024-02-10：107,169列。保留canonical train身分，新fi/qi互斥、日期完整。2023原R6 geography及R39舊兩partition依canonical列重映射，全部所需列有覆蓋；R39只在允許的原缺位置補上，fit新增1,966列、query新增1,099列，R6已有位置逐值保持。

資料helper SHA保持 4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1。整份 [preparation manifest](../outputs/round90/preparation/manifest.json) SHA為 400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7；其來源、frame、dtype、列序、兩類geo與原始cache位置皆有雜湊。原始價格表SHA只繼承記錄，不開啟／反序列化或重新hash價格表。

| 索引 | 列數 | ordered int64 content SHA |
|---|---:|---|
| fit_indices | 272,117 | 64e7eadeb39fa96abe8f3f2903aa5dd6632de52a1a4d4f1037ab0c02e7a027f3 |
| query_indices | 107,169 | 7dd254c2ae8fed176ec2f0740620d24dbbad39f175cb1bc50f80f13c8b354c45 |
| fit_source_cache_indices | 272,117 | beb1a53d8c6a575a346d0446532093d6ceaed261759318630c4e84e56ae15e62 |
| query_source_cache_indices | 107,169 | 9e762d7cc8840d629adfbff5741ecc114473be435c36c706a8e294c65d61db1c |

## 執行與驗證

- 開始收據於child執行之前保存：[launcher started](../outputs/round90/preparation_launch/started.json)，具體命令與固定source SHA；沒有自動重試。
- helper執行exit0，實測9.500秒：[preparation execution](../outputs/round90/preparation/execution.json)。
- launcher外層exit0，10.234秒：[launcher execution](../outputs/round90/preparation_launch/execution.json)。包含子程序啟動，不能與helper內時間相加。
- 使用load_prepared核對全部10項artifact與manifest：4個indices整個向量的dtype/shape/bytes/content SHA、6個X/geo完整frame的值SHA/dtype/columns/index皆相符。
- 另對4向量與6frames完整in-memory pickle roundtrip驗證，array逐bytes、DataFrame逐值exact，全數通過；4.266秒：[roundtrip verification](../outputs/round90/preparation_launch/roundtrip_verification.json)。沒有重跑prepare或重新載入原始來源表。
- 不曾修改helper來通過真資料檢查；本次沒有失敗或修復重跑。較早12項純合成檢查見 [selfcheck](round90_data_selfcheck.json)。

## 接續界線

API：load_prepared(out, expected_manifest_sha) 回傳已驗證的純 X、geo與索引。未來若另決定新窗口訓練，driver仍須依fit_source_cache_indices另行抽取fit y並綁原raw_train SHA，建立全部fit-only encoder/prior/weights、六native及reference/spatial，再封存預測後評分。現在只有資料前置完成。

原模型／CSV／state保留；未讀query價格、未查外部答案、未執行Kaggle或GitHub操作。此準備不能稱為新盲測或從頭訓練驗證成功。
