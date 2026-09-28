# 本批第三方來源與本地改編

Attention研究引用 [Yandex rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models/tree/e3ed46cac38568785289d8fa16b8cfa585bde27e)，固定commit e3ed46cac38568785289d8fa16b8cfa585bde27e。上游package版權為Copyright 2023 Yandex LLC，依Apache-2.0授權；[完整LICENSE](../../docs/third_party/rtdl_LICENSE.txt)由package原檔逐bytes複製，並保留[package原位置](../../evidence/outputs/round94_upstream/upstream/package/LICENSE)。根LICENSE版權2021、package版權2023，兩者原bytes分別保留，沒有合併冒稱同檔。

保存的[原始Python](../../evidence/outputs/round94_upstream/upstream/package/rtdl_revisiting_models.py)、[上游根README](../../evidence/outputs/round94_upstream/upstream/README.md)、[package README](../../evidence/outputs/round94_upstream/upstream/package/README.md)、[metadata](../../evidence/outputs/round94_upstream/upstream/package/pyproject.toml)及[下載manifest](../../evidence/outputs/round94_upstream/manifest.json)保留原bytes。這是六個明確檔案的來源保存，不是完整upstream repository鏡像；上游README其他相對連結可能指向未納入的原repository內容，請使用固定commit網址查閱。

[本地trainer](../src/round94_attention_prior_train_v2.py)已在檔頭標示改編與上游身分：縮小固定模型、最後CLS residual／FFN裁切、類別dropout、正prior-ratio head、加權總價L1、微批累積、固定末輪與全參數AdamW。它不import上游套件，不能稱官方論文完整復刻。裁切版與上游broadcast參考路徑的同權重eval／gradient合成驗證另見[來源審查](../../evidence/reports/round94_upstream_review.md)及[CPU收據](../../evidence/reports/round94_trainer_selfcheck_v2.json)。

保存原版不代表未修改的本地實作；原始檔與改編檔分開，原版LICENSE與著作權歸屬保持。未發現固定tree根與package層NOTICE，沒有杜撰NOTICE。本機本批尚未發布，後續發布需連同這份歸屬說明、完整license與source manifest。
