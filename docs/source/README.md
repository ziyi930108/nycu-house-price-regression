# 報告製作來源

現行18頁報告與來源說明集中在[report/source](../../report/source/README.md)，成品請由[正式報告入口](../../report/README.md)閱讀。套件清單見[requirements-report.txt](../../requirements-report.txt)。

本資料夾build_report.py是歷史相容入口，仍轉呼叫report/source/build_report.py。該舊builder只重建2026-10-14階段的報告，不能將其輸出標為目前2026-10-19定稿；舊程式及reference SHA保持原bytes。新版的明確參考與匯出方式見現行來源說明。

舊版重複Word已由此目錄移出目前追蹤，歷史版本保留於Git。最新Word、PDF與線上文字僅由report/提供。
