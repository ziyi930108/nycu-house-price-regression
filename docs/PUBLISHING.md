# v25 來源核對

SOURCE_MANIFEST.json 明列目前追蹤檔案的 bytes 與 SHA256。research/source_manifest.json 只列 v25 所需的保留來源，內容與原工作區來源保持相同；原完整清單可查 Git 歷史。

執行 `python scripts/verify_source.py` 核對清單及私有格式排除。只將明確檔案加入 Git，不以整個工作區掃描方式加入 ignored 檔案。不改寫實驗收據，也不將 source 核對冒稱模型端到端重現。


前次完整文件同步見[歷史發行摘要](RELEASE_20261019.json)。目前公開簡報另見[公開版說明](RELEASE_20261019_PPT_ONLY.json)；只提供PPT，講稿、逐頁預覽與製作配套保存在本機。公開PPT已清除講者備忘稿，投影片畫面保持。這次只更新目前分支，沒有改寫既有Git歷史。
