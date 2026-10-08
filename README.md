## 2026-10-08 追加研究：完整保留未採用方法

R111 殘差架構 NN 與 R112 Ordered 單價樹完成原三窗六 fit，兩者均未通過既定探索／強門檻，不產 v28／v29。額外 February 起點另完成 R107／R111／R112 三 fit：R107 整體改善約303元，R111改善約2,825元，R112退步約4,787元；額外起點改善不翻改原三窗失敗。

[本批結果與採用界線](docs/RESEARCH_CONTINUE_20261008.md) · [原三窗完整報告](evidence/reports/round111_112_report.md) · [額外起點完整報告](evidence/reports/round113_report.md) · [原三窗研究程式](research/src/round111_112_research.py) · [額外起點研究程式](research/src/round113_offset_research.py)

v25仍是小幅改善的探索候選，Public／Private未知；v22原選定、Public 1,998,568.5706元與上台材料保持。本次只發布完成的111–113程式與聚合證據，不包含未完成後續研究、提交CSV、交易表、模型或逐筆預測。重複歷史窗口存在選型偏差，並非Private保證。使用者最新停止線為Codex剩餘20%，不重置或使用額外點數。

以下所有原日期正文逐bytes保留，以其日期理解歷史狀態。

---

## 2026-10-08 研究恢復與新試交

使用者已重新授權模型優化與嘗試新提交。四方法、十二次回測及一次全量 prior NN fit 完成；v25為新增探索候選，最新年度離線 MAE 改善約0.1103%，早期略退步，Public／Private尚未知。R108／R109／R110未通過探索門檻，沒有v26／v27。原v22選定、Public 1,998,568.5706與約15分鐘簡報保留，Private未知。

[本輪方法、結果與限制](docs/RESEARCH_RESUME_20261008.md) · [R107／R108詳細報告](evidence/reports/round107_108_report.md) · [R109／R110詳細報告](evidence/reports/round109_110_report.md) · [v25訓練程式](research/src/round107_weighted_smooth_train.py) · [全量與CSV發行程式](research/src/round107_full_release.py)

CSV、交易資料及模型留在完整原工作區；此repo保存來源與彙總證據，乾淨環境端到端重訓尚未完成。以下原日期正文完整保留，歷史停止研究／唯一版本等敘述以當時狀態理解。

---

# 臺灣房屋總價預測 v22

完整下載或clone後，開啟[統一審查頁](teaching/v22_review_20261002/index.html)：三款PPTX、4,065字口語稿、15:00合成旁白影片、正式PDF，以及目前使用的答辯卡與練習工具，都從這裡進入。GitHub網頁通常只顯示HTML原碼，請保留完整目錄後用瀏覽器開啟。

主講目標約15分鐘、至少12分鐘，尚未本人排練；影片不能代替真人計時。[練習說明](teaching/v22_practice_20261002/README.md)區分完整主講和分頁練習；[最新答辯](teaching/v22_defense_20261002/README.md)說清同機新套件推論已重現、新電腦與從零重訓尚未驗證。

[三款PPTX重建](docs/REVIEW_TEACHING_REBUILD.md)與[影片重組](teaching/v22_review_20261002/04_程式與證據/影片製作程式/README.md)各自列出實測範圍。舊稿和製作紀錄保留，但目前入口已直接連到新工具；這次只調整導覽與說明，主稿、媒體及模型保持。

使用課程交易資料、道路地理特徵與六個回歸模型預測房屋總價。使用者已選定 **v22 為唯一提交版本**；Public MAE **1,998,568.5706 元**，Private 尚未知。選定版本不代表已證明 Private 最佳；[選定紀錄](evidence/selection/user_selected_v22_20260928.json)保留日期、原話與 CSV 身分。

- [新專案接手說明](HANDOFF.md)：現況、已完成驗證與下一步。
- [正式研究報告](REPORT.md)：資料處理、公式、年度回測、Public、限制與取捨。
- [正式 PDF](docs/臺灣房價預測_v22_正式報告_20261002.pdf)：12頁，含同機新套件環境實測，適合列印與繳交；[更新與重建](docs/REPORT_UPDATE_20261002.md)。
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

2026-10-03起會先檢查 `--nn-python` 及所選裝置；錯誤設定在模型計算及輸出建立之前退出。可用 `src/nn_predict.py --check-environment --device cuda:0` 單獨檢查。[改動、測試及相同CSV證據](docs/CLI_PREFLIGHT_20261003.md)。

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
