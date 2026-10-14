# 最終交付交接

2026-10-14：使用者授權依README、程式、設定、requirement.txt、報告五項整理GitHub，並同步最新正式報告。入口為 [README](README.md)，程式 [src](src/README.md)，設定 [config](config/README.md)，操作細節 [USAGE](docs/USAGE.md)。

報告改用 [report](report/README.md) 的18頁Word與PDF、八章一附錄；中文大章節、1.2縣市分析、清理精簡及前次批註保持。原報告與來源保存在Git歷史及作者原工作區。簡報保持20頁版本，未隨本輪改寫。

主套件入口requirement.txt引用25項固定ML套件，神經網路另用requirements-torch.txt引用11項固定套件。預測包裝讀取config/predict.json，沿用原封存程式與全部核對規則；research、evidence與封存推論來源保持原bytes。

Public MAE為1,995,836.2136元，Private未知。已實測保存六模型的完整推論；從原始資料重訓全部六模型仍未完成整套驗證，範圍見 [原始來源](inference/v25/training/README.md)。單一NN曲線不代表完整融合訓練軌跡，反覆驗證及分群不利結果保留。

此repo已公開，使用者指定以此取代Classroom交付；教師實際收件未獨立確認。Kaggle提交說明、上傳與最終勾選仍依原指示延後。未改選定CSV或訓練新模型。
