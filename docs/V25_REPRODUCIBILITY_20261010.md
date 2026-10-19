# v25 程式與重現界線

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
python scripts/verify_selected_csv.py --csv C:/NYCU_ML/lab1/outputs/submission_v25_trial.csv --selection evidence/selection/user_selected_v25_20261010.json --sample <原始sample的路徑>
~~~

`--sample`可省略；省略時仍核對已選CSV完整SHA及格式，但不另讀sample。命令沒有輸出或覆寫價格。實際v25CSV／原sample已由本機核對全列順序；[獨立證據](../evidence/selection/v25_independent_review_20261010.json)列明範圍。

原[final request](../evidence/outputs/round107/final/request.json)、[seal](../evidence/outputs/round107/final/seal.json)、[training](../evidence/outputs/round107/final/model/training.json)和[release](../evidence/outputs/round107/final/release.json)保持不改。此批全量只有新prior NN一次fit；其餘五個native用保存向量組合重播，不能冒稱重新跑六模型。

後續補入[可攜六模型推論入口](../inference/v25/README.md)，保留已實測20檔source/config/manifest原bytes。同機既有與全新獨立套件環境均從保存的六模型、encoder、prior、reference與spatial狀態重新推論，兩次實際exit0；13個完整向量逐位元相同，47,073筆CSV bytes完全相同。推論程序不讀保存答案向量；結束後才由分開程序比較。[公開收據](../inference/v25/evidence/reproducibility_receipt.json)列明來源、比較範圍、私有證據SHA及實測成本，不含逐筆資料。

[訓練來源補件](../inference/v25/training/README.md)另補22支原程式及2份config/schema，保留歷史guard、固定路徑與私有prepared/state/保存向量依賴。完整raw→六模型重訓尚未執行驗證；[可選地理依賴](../inference/v25/training/GEOGRAPHY.md)為歷史版本證據，未做fresh地理安裝或重建。同機新套件環境不是新硬體／作業系統；不同裝置、版本及CPU／GPU逐bytes等價未驗。R2 reference中間向量未直接輸出，沒有直接逐值相同的證據。

原R107不利門檻與反覆驗證選型限制保留。推論重現確認保存狀態的輸出一致，不能據此推論未知交易的預測誤差；選定CSV保持。
