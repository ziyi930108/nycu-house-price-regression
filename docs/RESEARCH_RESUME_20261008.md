# 2026-10-08 恢復研究：R107–R110

使用者重新授權改良模型及新試交。v22與原簡報保留；v22已回報Public MAE為1,998,568.5706元，Private未知。
以下為反覆使用歷史三窗的離線MAE；新候選Public／Private均未知，沒有代替使用者上傳或勾選Kaggle。

|方法與固定理由|development ΔMAE|confirmation ΔMAE|auxiliary ΔMAE|探索gate|強gate|產物終態|
|---|---:|---:|---:|---|---|---|
|[R107](../evidence/outputs/round107/results.json)：日期加權 prior NN：固定10萬元 SmoothL1，減少微小殘差附近梯度跳動|-1,415.88|-1,525.54|+883.66|True|False|v25試交候選，未選定|
|[R108](../evidence/outputs/round108/results.json)：單價 CatBoost：組合 CTR 複雜度2→1，檢查降低類別交互容量是否有益|+11,741.41|+8,747.72|-710.30|False|False|探索未通過，未發行：Fixed exploratory gates failed; latest MAE and Taipei group deteriorated.|
|[R109](../evidence/outputs/round109/results.json)：原 R63 loss：固定 epoch21–30 參數等權平均，檢查尾段波動；非標準SWA|-246.14|+190.36|+1,661.54|False|False|探索未通過，未發行：Fixed exploratory gates failed; latest and early MAE deteriorated.|
|[R110](../evidence/outputs/round110/results.json)：原 R79 單價：僅 global CTR target borders=8，檢查更多價格分布統計|+23,815.59|+11,073.32|+46,538.22|False|False|探索未通過，未發行：Fixed exploratory gates failed; all three windows and Keelung groups deteriorated.|

ΔMAE單位為新臺幣元，負值表示優於保存v22。完整整體／新舊加權、分群、五層、v24反證與配對區間在原results及報告；本頁不省略失敗而改稱所有方法成功。

探索gate允許每窗指標至多0.25%退步，要求最新兩指標不退步、至少兩窗整體改善且無季度／大型群超5%退步；強gate要求三窗整體及加權均改善、最新整體至少0.5%，同樣限制分群退步。通過探索不等於強推薦或Private較佳。

流程固定先提出假設與切點，再核canonical索引／九鍵NN輸入／fit-only權重與OOF價格先驗、各切點狀態，序列fit、封存模型與query向量，之後才讀query真價評分。各研究獨立，未把不同輪變因合併，也未依Public搜尋參數。

新提交格式：47,073列，欄名`編號,總價元`；依原sample ID順序，總價於最後一次取整並至少1元。程式／證據包僅保留提交SHA及格式收據，沒有CSV、逐筆價格、NPZ／NPY、模型、encoder／prior／reference狀態或課程原表。

限制：重用驗證窗口仍有累積選型偏差；條件bootstrap不是Private勝率；GPU固定seed不保證重訓逐bytes相同。來源可追溯及保存重載驗證，不等於新電腦從零端到端重訓已完成。未到內政部成交系統對答案。

[本輪精確複製清單](../evidence/outputs/research_package_resume_20261008/copy_manifest.json)；原379歷史manifest及所有舊收據保持。README／research README／HANDOFF的舊「停止研究」說法屬其原日期，請以本頁及本輪request／release為新增現況。
