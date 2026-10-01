# 臺灣房屋總價預測 v22

最新[約15分鐘上台版](teaching/v22_styles_20261001/fifteen_minute/README.md)已完成三款PPTX、4,051字白話稿及離線排練工具，主講規劃14:45，尚未本人計時。原模型v22、Public及未知Private保持。

新增[離線上台排練頁](teaching/v22_styles_20261001/spoken/rehearsal/README.md)：20頁講稿、提示、分頁計時與卡住筆記。實測時長仍需本人排練。

另附[四頁上台答辯提示卡](teaching/v22_styles_20261001/spoken/defense/README.md)：16題短答、追問補充與對照頁碼。原簡報及主講稿保持，問答不計主講時長。

先前[白話口語講稿版](teaching/v22_styles_20261001/spoken/README.md)已同步三款PPTX備忘稿。20頁主講4,981漢字，估計16.60–19.92分鐘、規劃17:20，未真人計時；畫面保持r2。

[原PDF投影備份](teaching/v22_styles_20261001/projection/README.md)與新口語版畫面相同，PDF無講者備忘稿。[風格指南](teaching/v22_styles_20261001/readability/README.md)供選擇參考，尚未代使用者選風格。

使用課程交易資料、道路地理特徵與六個回歸模型預測房屋總價。使用者已選定 **v22 為唯一提交版本**；Public MAE **1,998,568.5706 元**，Private 尚未知。選定版本不代表已證明 Private 最佳；[選定紀錄](evidence/selection/user_selected_v22_20260928.json)保留日期、原話與 CSV 身分。

- [新專案接手說明](HANDOFF.md)：現況、已完成驗證與下一步。
- [正式研究報告](REPORT.md)：資料處理、公式、年度回測、Public、限制與取捨。
- [正式 PDF](docs/臺灣房價預測_v22_正式報告.pdf)：適合列印與繳交。
- [重現紀錄](docs/REPRODUCIBILITY.md)：實際測試範圍與收據。
- [訓練程式導覽](research/README.md)：保留原始訓練來源及跨輪依賴。
- [模型及狀態清單](artifacts/manifest.json)：17 項必要私有檔案，約 1.45 GB。
- [資料說明](DATA.md)：課程檔案、道路來源及取得方式。

## 最新簡報與講稿 2026-10-01

[先前口語版三款與講稿](teaching/v22_styles_20261001/spoken/README.md)都是20頁主講＋3頁附錄，以同一v22完整流程及設計理由為主線。[完整口語稿](teaching/v22_styles_20261001/spoken/content/完整口語講稿.md)及[口述提示](teaching/v22_styles_20261001/spoken/content/上台口述提示.md)配合使用。主講字數估時不包含問答，仍需實際排練。

[原r2](teaching/v22_styles_20261001/README.md)及[原零基礎教材](teaching/v22_20261001/README.md)保留。既有[重建入口](docs/TEACHING_REBUILD.md)產生原r2，新增[口語版重建入口](docs/SPOKEN_TEACHING_REBUILD.md)，可直接讀repo，在新目錄同步備忘稿；三款檔案與69渲染已重建相同。原14頁報告方法適用，原配速屬歷史稿。

## 最新聯動dropout研究

[R106完成頁](docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md)：R106六次NN與三窗評分完成；完整T−v22 MAE為 **-63.26／+2,439.67／+422.80 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**

C為原R63設定重訓，T只改四欄mask聯動；C−v22偏移、24CI及反證全部保留。正式報告／簡報聚焦v22。研究於本輪完成後暫停，不安排下一輪。

## 先前完整流程簡報存檔

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
| teaching/ | 新三款23頁簡報、完整講稿、原零基礎報告與製作來源 |
| presentation/ | 原22頁正式簡報與來源，保留歷史版本 |
| private/、data/、runs/ | 本機使用，Git與公開ZIP排除 |

保存模型推論已實測；從零重訓、地理重建、全新環境安裝尚未完成端到端驗證。固定seed也不保證GPU重訓位元相同。

GitHub儲存庫為[ziyi930108/nycu-house-price-regression](https://github.com/ziyi930108/nycu-house-price-regression)，設為私人。程式尚未選擇開源授權；課程資料再散布權限亦未確認。保留.gitignore，勿將private/與data/納入版本控制。
