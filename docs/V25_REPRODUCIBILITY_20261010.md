# v25 程式與重現界線｜2026-10-10

v25只重新訓練日期加權prior NN，把L1改為固定10萬元SmoothL1，保留六模型／三分支配方。程式與證據已保存，沒有宣稱新環境從原始Excel一鍵重訓完成。

| 程式 | 作用與執行範圍 |
| --- | --- |
| [round107_weighted_smooth_train.py](../research/src/round107_weighted_smooth_train.py) | 讀原R63九鍵prepared NPZ，固定30epoch／seed42訓練prior NN；完成新模型47,073列保存重載全值相同 |
| [round107_108_research.py](../research/src/round107_108_research.py) | 三時間窗、各切點fit/query狀態核對、訓練封存後評分；含另一個R108對照，沒有將兩者合併 |
| [round107_final_candidate.py](../research/src/round107_final_candidate.py) | 全量context與保存v22向量鏈重播；把新prior替入原NN分支，再做R/S與median |
| [round107_full_release.py](../research/src/round107_full_release.py) | 凍結前置、單次全量fit與分開CSV發行；依賴完整原工作區與歷史SHA／私有prepared/state，不是可搬移的一鍵入口 |
| [verify_selected_csv.py](../scripts/verify_selected_csv.py) | 標準函式庫驗證選定CSV完整SHA、47,073列／唯一ID／正整數，可另核sample列序；不推論、不評MAE、不操作Kaggle |

最後兩個角色不能混淆：CSV核對只確認檔案身分，不能取代模型推論。

在repo根目錄核對原工作區CSV（CSV/sample必須由本人合法取得）：

~~~powershell
python scripts/verify_selected_csv.py --csv C:/NYCU_ML/outputs/submission_v25_trial.csv --selection evidence/selection/user_selected_v25_20261010.json --sample <原始sample的路徑>
~~~

`--sample`可省略；省略時仍核對已選CSV完整SHA及格式，但不另讀sample。命令沒有輸出或覆寫價格。實際v25CSV／原sample已由本機核對全列順序；[獨立證據](../evidence/selection/v25_independent_review_20261010.json)列明範圍。

原[final request](../evidence/outputs/round107/final/request.json)、[seal](../evidence/outputs/round107/final/seal.json)、[training](../evidence/outputs/round107/final/model/training.json)和[release](../evidence/outputs/round107/final/release.json)保持不改。此批全量只有新prior NN一次fit；其餘五個native用保存向量組合重播，不能冒稱重新跑六模型。

舊版專用推論入口與其 artifact 清單已移出主分支。尚無經驗證、可從 repo 直接載六模型重做 v25 的單一 production CLI；v25 新電腦／乾淨環境完整推論、地理重建與全部模型從零重訓未完成。

後續若補可攜v25推論入口，需另立新artifact清單與實測，不解除舊SHA或拒絕覆寫防護。Private未知，不能以這些重現檢查認證Private最佳。
