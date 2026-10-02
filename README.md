# 臺灣房屋總價預測 v22

[最新答辯卡與問答](teaching/v22_defense_20261002/README.md)已更新第15題：同機新套件環境的保存模型推論已完成，新電腦與從零重訓仍未驗證。原r2封存材料中「全新環境安裝尚未驗證」是當時進度，答辯請用此更新；主講、PPTX及影片保持。

新增[上台前練習](teaching/v22_practice_20261002/index.html)：用同一筆假設交易串起六模型與三分支；新計時工具分開保存分頁練習與完整主講，記錄暫停和回看，避免把單頁累計當完整報告。請完整下載repo後開啟。此工具沿用4,065字正文，不增加簡報頁數或改動影片；原審查包的舊計時頁保留，現在練習請用新入口。

工具README中的重建命令只產生計時單頁；新目錄沒有其「回到上台練習入口」需要的`index.html`，但單頁可直接計時。需要完整導覽時，先完整複製`teaching/v22_practice_20261002/`九個檔案，再將重建輸出指定為副本內尚不存在的`排練計時_重建.html`，不要覆寫原件；同層入口、自測頁與說明便能保留。

最新[約15分鐘審查版](teaching/v22_review_20261002/README.md)已整理三款PPTX、4,065字口語稿、离線排練頁、15:00合成旁白影片與字型修正正式PDF。先完整下載repo，再開[審查頁](teaching/v22_review_20261002/index.html)。主講規劃14:45，尚未本人計時，合成影片不能代替真人排練。

[最新PPTX重建入口](docs/REVIEW_TEACHING_REBUILD.md)與[影片重組入口](teaching/v22_review_20261002/04_程式與證據/影片製作程式/README.md)各自說明實測範圍。原4,051字稿及其[舊重建命令](docs/FIFTEEN_MINUTE_REBUILD.md)保留作歷史，不能當成本版4,065字稿。

使用課程交易資料、道路地理特徵與六個回歸模型預測房屋總價。使用者已選定 **v22 為唯一提交版本**；Public MAE **1,998,568.5706 元**，Private 尚未知。選定版本不代表已證明 Private 最佳；[選定紀錄](evidence/selection/user_selected_v22_20260928.json)保留日期、原話與 CSV 身分。

- [新專案接手說明](HANDOFF.md)：現況、已完成驗證與下一步。
- [正式研究報告](REPORT.md)：資料處理、公式、年度回測、Public、限制與取捨。
- [正式 PDF](teaching/v22_review_20261002/03_正式報告/臺灣房價預測_v22_正式報告.pdf)：適合列印與繳交。
- [重現紀錄](docs/REPRODUCIBILITY.md)：實際測試範圍與收據。
- [訓練程式導覽](research/README.md)：保留原始訓練來源及跨輪依賴。
- [模型及狀態清單](artifacts/manifest.json)：17 項必要私有檔案，約 1.45 GB。
- [資料說明](DATA.md)：課程檔案、道路來源及取得方式。

## 上台用哪一份

手機直式影片中的公式與字幕會縮小，暫停控制列也可能遮住字幕。可先開[逐頁口語講稿](teaching/v22_review_20261002/index.html#script)，再展開目前頁；公式建議用電腦看，或嘗試手機橫向全螢幕。原審查包與影片保持，這段指引不代表已驗證所有裝置。

[最新三款與講稿](teaching/v22_review_20261002/README.md)主講同一個v22流程及設計理由，不要求觀眾理解內部版本沿革。4,065字只計口述正文，不重念提示與理由，附錄和問答不計時。

[前版4,051字](teaching/v22_styles_20261001/fifteen_minute/README.md)、[歷史4,981字口語版](teaching/v22_styles_20261001/spoken/README.md)及[原r2](teaching/v22_styles_20261001/README.md)均保留。各自重建入口只對應自己的版本。三款視覺相同主線，尚待使用者選擇與實際排練。

## 最新聯動dropout研究

[R106完成頁](docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md)：R106六次NN與三窗評分完成；完整T−v22 MAE為 **-63.26／+2,439.67／+422.80 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**

C為原R63設定重訓，T只改四欄mask聯動；C−v22偏移、24CI及反證全部保留。正式報告／簡報聚焦v22。研究於本輪完成後暫停，不安排下一輪。

## 先前完整流程簡報存檔

[22頁正式簡報](presentation/臺灣房價預測_正式報告_v22.pptx)與[逐頁講稿](presentation/講稿與問答.md)涵蓋完整流程及每步理由，20頁主講＋2頁附錄。主講配速14分30秒；4,143個純漢字按每分鐘250–300字估計約13.81–16.57分鐘，不計附錄或問答，仍需實際計時排練。另附[排練與問答指南](presentation/排練與問答指南.md)及[流程與程式對照](presentation/流程與程式對照.md)。主講只說明使用者選定的v22。後續發布方式見[更新規則](docs/PUBLISHING.md)。本機已核對程式身分與提交格式。自動檢查設定以範例保留，尚未啟用GitHub Actions。

## 快速開始（Windows／Python3.12受測路線）

需要 Python 3.12、課程原始 test 與 sample，以及保存模型／狀態。公開程式包不含交易表、模型或價格先驗。

~~~powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
python -m venv .venv-nn
.venv-nn\Scripts\python.exe -I -m pip --isolated install --only-binary=:all: --index-url https://pypi.org/simple -r config/requirements-torch-cu126-win-py312.txt
~~~

2026-10-02已在同一台Windows／Python3.12.14 base新建ML與GPU套件環境，完成保存六模型、47,073筆推論，CSV及六個比較向量逐bytes等於原v22。主程式76.930秒；[新環境重現證據與限制](docs/CLEAN_GPU_REPLAY_20261002.md)。這是新venv，不是新電腦／新OS或從零重訓。

`.venv`沿用上一階段受測的兩份requirements，其中一般PyPI Torch為CPU版；兩NN實際由`.venv-nn`的**2.13.0+cu126**執行。新的GPU清單固定官方wheel及SHA，僅適用Windows x64／CPython3.12，約2.6GB。原requirements保持；[上一階段CPU安裝紀錄](docs/ENVIRONMENT_INSTALL_20261002.md)仍是獨立歷史證據。

若有原研究工作區，可逐檔核對 SHA256 並複製必要 artifact：

~~~powershell
.venv\Scripts\python.exe scripts/collect_artifacts.py --workspace C:/NYCU_ML
~~~

將課程的兩個原始檔案放到 data/。程式核對檔案雜湊及 ID，因為保存地理特徵對應這份 test 的列序。這個入口用於固定競賽資料重現，並非任意新房屋資料的估價 API。

~~~powershell
# 請在本repo的新PowerShell視窗執行；只清除此視窗的Python環境變數。
@([Environment]::GetEnvironmentVariables('Process').Keys) | Where-Object { $_ -like 'PYTHON*' } | ForEach-Object { [Environment]::SetEnvironmentVariable($_, $null, 'Process') }
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
.venv\Scripts\python.exe -I -B -X utf8 src/predict_v22.py --test data/test-reindex-test-v2.2.xlsx --sample data/test-reindex-test-sample-submission-v2.csv --output-dir runs/v22_reproduction --nn-python .venv-nn/Scripts/python.exe --device cuda:0
~~~

上方兩環境分工對應本次實測。父程式使用-I；原NN子程序不自動繼承-I，所以本次另清除PYTHON相關變數並關閉user site，不能稱整條流程皆為-I。這不是全新驅動安裝教學；受測GPU為RTX4060Ti，其他GPU／驅動尚未驗證。output-dir必須是新目錄，不會自動覆寫既有結果。

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
.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

| 位置 | 內容 |
| --- | --- |
| src/ | 可攜推論入口及必要模組 |
| config/、artifacts/ | 固定配方與檔案身分 |
| research/ | 原始訓練程式與導覽 |
| evidence/ | 報告來源及實測收據 |
| docs/ | 重現說明與正式 PDF |
| teaching/ | 最新審查簡報、講稿、影片、修正版報告與來源；另保留歷史教材 |
| presentation/ | 原22頁正式簡報與來源，保留歷史版本 |
| private/、data/、runs/ | 本機使用，Git與公開ZIP排除 |

保存模型推論已在原環境及同機新套件環境實測；從零重訓、地理重建及新電腦端到端驗證仍未完成。固定seed也不保證GPU重訓位元相同。

GitHub儲存庫為[ziyi930108/nycu-house-price-regression](https://github.com/ziyi930108/nycu-house-price-regression)，設為私人。程式尚未選擇開源授權；課程資料再散布權限亦未確認。保留.gitignore，勿將private/與data/納入版本控制。
