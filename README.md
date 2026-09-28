# 臺灣房屋總價預測 v22

使用課程交易資料、道路地理特徵與六個回歸模型預測房屋總價。使用者已選定 **v22 為唯一提交版本**；Public MAE **1,998,568.5706 元**，Private 尚未知。選定版本不代表已證明 Private 最佳；[選定紀錄](evidence/selection/user_selected_v22_20260928.json)保留日期、原話與 CSV 身分。

- [新專案接手說明](HANDOFF.md)：現況、已完成驗證與下一步。
- [正式研究報告](REPORT.md)：資料處理、公式、年度回測、Public、限制與取捨。
- [正式 PDF](docs/臺灣房價預測_v22_正式報告.pdf)：適合列印與繳交。
- [重現紀錄](docs/REPRODUCIBILITY.md)：實際測試範圍與收據。
- [訓練程式導覽](research/README.md)：保留原始訓練來源及跨輪依賴。
- [模型及狀態清單](artifacts/manifest.json)：17 項必要私有檔案，約 1.45 GB。
- [資料說明](DATA.md)：課程檔案、道路來源及取得方式。

## 最新時間前推組合研究

[R99 把 median 納入可選基線](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)已完成。保留原 N／H／U 三支修正總價，多加逐筆中位數 M，再以較早 forward 資料學非負四權重。相對 v22 的三窗 MAE 差為 **+10,449.82／−6,281.53／+2,821.52 元**，完整研究門檻未過；相對 [R98](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)則為 +1,651.18／−2,026.51／−2,369.15 元，保留有利與不利證據。12 個相對 v22 的配對區間全部跨零，不能稱全面顯著退步或 Private 勝率。

三次 CPU meta-fit 呼叫合計 2.781 秒，包含於 main 6.782 秒；獨立評分 30.156 秒，保存結果稽核 1.250 秒／253 唯一 SHA／12 保存區間。新增 0 次原生樹／NN fit，沒有新 CSV。M 與舊三欄解在 meta 上可行，並不保證較晚 query 較好；唯一提交 v22、未知 Private、正式報告與簡報保持。本頁於發布前核驗，當前發布身分以 Git main 歷史及 SOURCE_MANIFEST 為準。

此前 [R97 錯開切點](docs/RESEARCH_UPDATES_20260929_OFFSET.md)已重建 5 樹＋2 NN：新窗 v22／v24 MAE 1,336,410.67／1,340,328.14 元，原三年則 v24 較低，反證保持。[Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)、[R93](docs/RESEARCH_UPDATES_20260929_R93.md)、[R89／R91／R92](docs/RESEARCH_UPDATES_20260929.md)、[R85–R88](docs/RESEARCH_UPDATES_20260928.md)亦保留未採用結果。這些窗口反覆使用與重疊，不是新盲測。

## 簡報與專案更新

[22頁正式簡報](presentation/臺灣房價預測_正式報告_v22.pptx)與[逐頁講稿](presentation/講稿與問答.md)涵蓋完整流程及每步理由，20頁主講＋2頁附錄。主講配速14分30秒；4,143個純漢字按每分鐘250–300字估計約13.81–16.57分鐘，不計附錄或問答，仍需實際計時排練。另附[排練與問答指南](presentation/排練與問答指南.md)及[流程與程式對照](presentation/流程與程式對照.md)。主講只說明使用者選定的v22。後續發布方式見[更新規則](docs/PUBLISHING.md)。本機已核對程式身分與提交格式。自動檢查設定以範例保留，尚未啟用GitHub Actions。

## 快速開始

需要 Python 3.12、課程原始 test 與 sample，以及保存模型／狀態。公開程式包不含交易表、模型或價格先驗。

~~~powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-inference.txt
.venv\Scripts\python -m pip install -r requirements-torch.txt
~~~

以上是安裝說明，尚未在全新環境完成安裝驗證。實測使用既有兩個 Python 3.12.14 環境：ML 環境與 Torch 2.13.0+cu126 環境。精確重現原 NN 需要對應 CUDA 與 GPU 運算設定，CPU 與其他硬體不保證輸出相同。

若有原研究工作區，可逐檔核對 SHA256 並複製必要 artifact：

~~~powershell
python scripts/collect_artifacts.py --workspace C:/NYCU_ML
~~~

將課程的兩個原始檔案放到 data/。程式核對檔案雜湊及 ID，因為保存地理特徵對應這份 test 的列序。這個入口用於固定競賽資料重現，並非任意新房屋資料的估價 API。

~~~powershell
python src/predict_v22.py --test data/test-reindex-test-v2.2.xlsx --sample data/test-reindex-test-sample-submission-v2.csv --output-dir runs/v22_reproduction --device cuda:0
~~~

若 Torch 在另一環境，加 --nn-python C:/path/to/torch-env/Scripts/python.exe。output-dir 必須是新目錄。

程式讀 Excel、重建87欄特徵、套保存encoder／prior，真正載入六模型，逐分支套參考成交與空間修正，再取中位數。輸出 CSV、未取整預測、三分支向量及 receipt.json；不讀舊預測作為輸入。

最後比對 CSV 與原 v22 的 SHA256；不一致會回報非零結束碼。--device cpu 可用於數值差異診斷，實測不能精確重現原GPU結果。這種差異不是test MAE。

## 方法

~~~python
B_unit = .75*C_unit + .25*L_unit
B_nn   = .75*(.5*N_area + .5*N_prior) + .25*L_unit
B_log  = .75*(.5*C_raw  + .5*C_log)   + .25*L_unit
v22 = median(S(B_nn, R(B_nn)), S(B_log, R(B_log)), S(B_unit, R(B_unit)))
~~~

各模型預測都已還原為總價元。R為fit成交參考與趨勢，S為空間修正。最後一次取整、最低1元。單價CatBoost的訓練採面積權重乘上365日半衰期日期權重。

## 驗證與目錄

~~~powershell
python -m unittest discover -s tests -v
~~~

| 位置 | 內容 |
| --- | --- |
| src/ | 可攜推論入口及必要模組 |
| config/、artifacts/ | 固定配方與檔案身分 |
| research/ | 原始訓練程式與導覽 |
| evidence/ | 報告來源及實測收據 |
| docs/ | 重現說明與正式 PDF |
| presentation/ | 22頁簡報、完整講稿、排練指南與製作來源 |
| private/、data/、runs/ | 本機使用，Git與公開ZIP排除 |

保存模型推論已實測；從零重訓、地理重建、全新環境安裝尚未完成端到端驗證。固定seed也不保證GPU重訓位元相同。

GitHub儲存庫為[ziyi930108/nycu-house-price-regression](https://github.com/ziyi930108/nycu-house-price-regression)，設為私人。程式尚未選擇開源授權；課程資料再散布權限亦未確認。保留.gitignore，勿將private/與data/納入版本控制。
