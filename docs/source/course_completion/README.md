# 報告重建來源

由repo根目錄執行 `python docs/source/build_report.py`，會重建 docs 中的正式Word與根目錄REPORT.md。先保存自己編輯的副本。需要 requirement.txt 中的文件套件；字型沿用本機Microsoft JhengHei。

`reference_report.docx` 保存使用者已確認的排版及原文，`supplement.json` 與 assets 提供新增章節、流程圖、公開統計和匿名案例。[來源對照](SOURCE_MAP.md)。

這是文件重建，使用已保存的公開摘要與圖片，不會訓練模型。Word16匯出與本機逐頁檢查已完成；不同Office版本與新電腦安裝尚未驗證。
