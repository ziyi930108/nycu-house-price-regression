# 2026-10-08 恢復研究：R111–R112

使用者重新授權改良模型及新試交。v22與原簡報保留；v22已回報Public MAE為1,998,568.5706元，Private未知。
本輪停止線為Codex剩餘20%，不購買點數或自行重置；新研究不取代v22選定。
以下為反覆使用歷史三窗的離線MAE；新候選Public／Private均未知，沒有代替使用者上傳或勾選Kaggle。

|方法與固定理由|development ΔMAE|confirmation ΔMAE|auxiliary ΔMAE|探索gate|強gate|產物終態|
|---|---:|---:|---:|---|---|---|
|[R111](../evidence/outputs/round111/results.json)：prior NN固定residual MLP：保持日期總價MAE／prior，檢查內部skip架構|+2,017.37|+5,733.24|-6,192.97|False|False|原三窗探索未通過，未發行：Original three-window exploration gate failed: latest MAE and weighted MAE worsened; only one window improved.|
|[R112](../evidence/outputs/round112/results.json)：R79單價/date×area：僅Plain→Ordered，檢查target/weight交互；保留R17失敗反證|+4,288.21|+16,317.73|+8,689.27|False|False|原三窗探索未通過，未發行：Original three-window exploration gate failed: all three overall and weighted metrics worsened, with large-group regressions above five percent.|

ΔMAE單位為新臺幣元，負值表示優於保存v22。完整整體／新舊加權、分群、五層、v24反證與配對區間在原results及報告；本頁不省略失敗而改稱所有方法成功。

探索gate允許每窗指標至多0.25%退步，要求最新兩指標不退步、至少兩窗整體改善且無季度／大型群超5%退步；強gate要求三窗整體及加權均改善、最新整體至少0.5%，同樣限制分群退步。通過探索不等於強推薦或Private較佳。

流程固定先提出假設與切點，再核canonical索引／九鍵NN輸入／fit-only權重與OOF價格先驗、各切點狀態，序列fit、封存模型與query向量，之後才讀query真價評分。各研究獨立，未把不同輪變因合併，也未依Public搜尋參數。

新提交格式：47,073列，欄名`編號,總價元`；依原sample ID順序，總價於最後一次取整並至少1元。程式／證據包僅保留提交SHA及格式收據，沒有CSV、逐筆價格、NPZ／NPY、模型、encoder／prior／reference狀態或課程原表。

限制：重用驗證窗口仍有累積選型偏差；條件bootstrap不是Private勝率；GPU固定seed不保證重訓逐bytes相同。來源可追溯及保存重載驗證，不等於新電腦從零端到端重訓已完成。未到內政部成交系統對答案。

R113是固定R97非8月起點的額外穩定性檢查（R107/R111/R112各一次；2NN+1CB），重疊歷史仍非盲測。新全量候選需同時通過三窗與R113的獨立審查；原三窗過而R113失敗明列extra_gate_failed_no_release，不冒稱原gate失敗。既有v25只新增後續證據、不回寫原release。

[本輪精確複製清單](../evidence/outputs/research_package_continue_20261008/copy_manifest.json)；原379歷史manifest及所有舊收據保持。README／research README／HANDOFF的舊「停止研究」說法屬其原日期，請以本頁及本輪request／release為新增現況。
