# 最終交付交接

2026-10-19：使用者授權繼續整理既有GitHub儲存庫，首頁不列排行榜成績。只以v25繳交，模型研究已停止；本輪工作為更新交付入口與同步已定稿文件。入口為 [README](README.md)，程式 [src](src/README.md)，設定 [config](config/README.md)，操作細節 [USAGE](docs/USAGE.md)。此版文件與檔案清單對齊2026-10-19定稿，發布身分以儲存庫提交紀錄為準。

本輪 [report](report/README.md) 採用2026-10-19定稿的18頁Word、同版PDF及線上文字，八章一附錄；報告正文沿用已確認定稿。簡報入口為 [teaching/final_presentation](teaching/final_presentation/README.md)，採用最新17頁頁尾精簡版與完整口語講稿，保留使用者所有刪除批註及最後總結／致謝。舊稿保存在Git歷史及作者原工作區。

主套件入口requirement.txt引用25項固定ML套件，神經網路另用requirements-torch.txt引用11項固定套件。預測包裝讀取config/predict.json，沿用原封存程式與全部核對規則；research、evidence與封存推論來源保持原bytes。

已實測保存六模型的完整推論；從原始資料重訓全部六模型仍未完成整套驗證，範圍見 [原始來源](inference/v25/training/README.md)。單一NN曲線不代表完整融合訓練軌跡，反覆驗證及分群不利結果保留。首頁不列成績，不代表刪除報告的結果分析或改寫歷史證據。

此repo已公開，使用者指定以此取代Classroom交付；教師實際收件未獨立確認。本輪不操作Kaggle，不改選定CSV或訓練新模型；原始交易資料、模型權重、逐筆預測與執行環境不公開。
