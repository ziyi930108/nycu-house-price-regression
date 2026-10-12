# 報告製作來源

`build_report.py`以python-docx產生可編輯Word，曲線取自`../assets/training_curve.png`。由repo根目錄執行`python docs/source/build_report.py`，會重新產生docs中的正式報告；請先保存自行編輯的副本。

本次使用Codex bundled Python與python-docx，Word16輸出12頁，再以bundled Poppler逐頁核對。排版在其他Word版本仍可能略有差異。這是文件重建入口，不會訓練模型，也未宣稱新電腦安裝或全部六模型重現。
