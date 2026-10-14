# 報告製作來源

目前來源與參考模板集中在 [report/source](../../report/source/README.md)，套件見 [requirements-report.txt](../../requirements-report.txt)。

舊命令 `python docs/source/build_report.py` 保留作為相容入口，會呼叫最新來源，輸出至尚不存在成品的 `build/report_current` 資料夾；也可用 `--output-dir` 指定新的資料夾。程式拒絕覆寫已存在的報告，不會修改目前交付Word或訓練模型。完整閱讀與重建說明見 [正式報告](../../report/README.md)。
