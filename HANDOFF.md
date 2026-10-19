# 最終交付交接

2026-10-19：使用者授權繼續整理既有GitHub儲存庫，首頁不列排行榜成績。只以v25繳交，模型研究已停止；本輪工作為更新交付入口與同步已定稿文件。入口為 [README](README.md)，程式 [src](src/README.md)，設定 [config](config/README.md)，操作細節 [USAGE](docs/USAGE.md)。此版文件與檔案清單對齊2026-10-19定稿，發布身分以儲存庫提交紀錄為準。

本輪 [report](report/README.md) 採用2026-10-19定稿的18頁Word、同版PDF及線上文字，八章一附錄；報告正文沿用已確認定稿。公開簡報只提供 [17頁PPT](teaching/final_presentation/台灣房屋銷售價格預測挑戰賽_簡報.pptx)，保留既有投影片畫面與刪除批註，清除講者備忘稿。完整講稿、逐頁預覽與簡報製作配套留在本機，不再納入目前公開分支；首頁標示「簡報」。本次沒有改寫既有Git歷史。

主套件入口requirement.txt引用25項固定ML套件，神經網路另用requirements-torch.txt引用11項固定套件。預測包裝讀取config/predict.json，沿用原封存程式與全部核對規則；research、evidence與封存推論來源保持原bytes。

已實測保存六模型的完整推論；從原始資料重訓全部六模型仍未完成整套驗證，範圍見 [原始來源](inference/v25/training/README.md)。單一NN曲線不代表完整融合訓練軌跡，反覆驗證及分群不利結果保留。首頁不列成績，不代表刪除報告的結果分析或改寫歷史證據。

此repo已公開，使用者指定以此取代Classroom交付；教師實際收件未獨立確認。本輪不操作Kaggle，不改選定CSV或訓練新模型；原始交易資料、模型權重、逐筆預測與執行環境不公開。
