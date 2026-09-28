# 臺灣房屋總價預測 v22

使用課程交易資料、道路地理特徵與六個回歸模型預測房屋總價。使用者已選定 **v22 為唯一提交版本**；Public MAE **1,998,568.5706 元**，Private 尚未知。選定版本不代表已證明 Private 最佳；[選定紀錄](evidence/selection/user_selected_v22_20260928.json)保留日期、原話與 CSV 身分。

- [新專案接手說明](HANDOFF.md)：現況、已完成驗證與下一步。
- [正式研究報告](REPORT.md)：資料處理、公式、年度回測、Public、限制與取捨。
- [正式 PDF](docs/臺灣房價預測_v22_正式報告.pdf)：適合列印與繳交。
- [重現紀錄](docs/REPRODUCIBILITY.md)：實際測試範圍與收據。
- [訓練程式導覽](research/README.md)：保留原始訓練來源及跨輪依賴。
- [模型及狀態清單](artifacts/manifest.json)：17 項必要私有檔案，約 1.45 GB。
- [資料說明](DATA.md)：課程檔案、道路來源及取得方式。

## 最新時間前置先驗研究

[R100 時間前置 prior](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)已完成六次 NN、三期完整模型與獨立評分。使用相同 loss 樣本比較原 group-OOF 先驗 C 與「季首以前且排除同建案折」的先驗 T。T 相對 v22 的 development／confirmation／auxiliary MAE 差為 **−5,097.81／+10,063.35／−11,676.48 元**；最近窗退 0.7275%，auxiliary 嘉義市亦觸及固定群退步限制，研究門檻未過。C 是機制對照，沒有事後升格成另一個提交候選。

24 個 T−v22／T−C 配對區間為 4 負、14 跨零、6 正；不能轉述成 Private 勝率。六次 NN fit 共 118.203 秒，已包含於 main 391.938 秒；scorer 68.985 秒，保存結果稽核內部 77.078 秒（外層 80.109 秒）。評分前的兩份計畫 SHA 契約修正另存 scorer v2，沒有重試模型或更改科學評估參數。唯一提交 v22、未知 Private、正式報告與簡報保持。本頁於發布前核驗，當前發布身分以 Git main 歷史及 SOURCE_MANIFEST 為準。

此前 [R99 四欄組合](docs/RESEARCH_UPDATES_20260929_MEDIAN_STACK.md)、[R98 三欄組合](docs/RESEARCH_UPDATES_20260929_TEMPORAL_STACK.md)都未過採用门檻。[R97 錯開切點](docs/RESEARCH_UPDATES_20260929_OFFSET.md)重建 5 樹＋2 NN，新窗 v22 較低而原三年 v24 較低，反證保持。[Attention／R96](docs/RESEARCH_UPDATES_20260929_ATTENTION.md)、[R93](docs/RESEARCH_UPDATES_20260929_R93.md)、[R89／R91／R92](docs/RESEARCH_UPDATES_20260929.md)、[R85–R88](docs/RESEARCH_UPDATES_20260928.md)亦保留。這些窗口反覆使用與重疊，不是新盲測。

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
