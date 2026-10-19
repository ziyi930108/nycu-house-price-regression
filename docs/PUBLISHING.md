# v25 來源核對

SOURCE_MANIFEST.json 明列目前追蹤檔案的 bytes 與 SHA256。research/source_manifest.json 只列 v25 所需的保留來源，內容與原工作區來源保持相同；原完整清單可查 Git 歷史。

執行 `python scripts/verify_source.py` 核對清單及私有格式排除。只將明確檔案加入 Git，不以整個工作區掃描方式加入 ignored 檔案。不改寫實驗收據，也不將 source 核對冒稱模型端到端重現。


2026-10-19現行交付檔案、頁數與核對範圍見[發行摘要](RELEASE_20261019.json)。Word／PDF／PPT由已核定稿原bytes同步；舊製作工具保持原來源，不將舊版重建能力套用到新稿。
