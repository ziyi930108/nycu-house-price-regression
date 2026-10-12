# v25 六模型推論與重現證據

已在原 Windows 主機的既有獨立 ML／NN 環境，以及同機新建立的兩個獨立套件 venv，各載入保存的六個模型與完整 encoder、prior、reference、spatial fit 狀態，重新執行六個 native forward、三支融合及最後中位數輸出。推論程序實際 exit code 為 0；推論完成後才由分開的驗證程序讀取封存向量比對。

全部 47,073 筆的 13 個完整向量數值逐位元相同，涵蓋六個 native、三個 base、三個 reference／spatial 後向量及未取整中位數。新 CSV 的全部 ID 順序、整數預測與檔案 bytes 也完全相同，SHA256 為 `36231cd17a6cbeafefeb62a7989d207fa20d6c5f7aaf99857e4d9addd25597c1`。[彙總收據](evidence/reproducibility_receipt.json)記錄來源身分、比較範圍及實測成本。這是預測重現檢查，並非未知 test 真價 MAE；Private 分數仍未知。

同機全新 ML／NN 套件 venv 已用 [25／11 個固定官方 wheel](dependencies/historical_lock_provenance.json) 安裝，來源、SHA、版本、實際匯入路徑及依賴檢查均通過。[安裝摘要](evidence/fresh_package_installation.json)記錄真正新建環境的安裝證據。其後完整六模型推論與獨立全值比較亦已通過：新套件環境 outer 實測71.855秒，既有環境77.262秒，兩次均13個向量逐位元相同、CSV bytes完全相同。沒有更換硬體或作業系統，也未完成六模型從頭重訓。

## 來源與私有依賴

[source](source/closure.json)完整保留已實測的 20 檔 source/config/manifest 原始 bytes 與相對結構，closure SHA256 為 `a781bc2d3c08464c69e591d60ff7deb344aac946c55bdd405ce41da884f0dc30`。其內 README 與 closure 的 draft 標記是執行前封存原文，為保存雜湊不回寫；目前執行狀態以本頁及獨立彙總收據為準。沿用來源中的 v22 註解代表工程來源，目前執行方法為 v25／R107 prior NN。

此目錄不含權重、課程資料、交易 ID 值、逐筆價格或 query NPZ。[17 項私有依賴清單](source/artifacts/manifest.json)只公開相對路徑、角色、大小與 SHA。合法取得原保存模型／fit 狀態者須在外部私人根目錄保留清單列出的相對結構，並另取得原課程 test Excel 與 sample；入口會先核對其完整 SHA，再產生合法 query 特徵。Price prior/reference/spatial 含合法 fit 價格或 residual，屬保存的 fitted state，不是未知 test 真價。生成的 query NPZ 僅在此次推論程序內產生，不接受封存預測向量充當輸入。

在儲存庫根目錄只讀查看入口（不載入 ML 套件或模型）：

```powershell
python -I -B -X utf8 inference/v25/source/src/predict_v25.py --describe
```

正式推論需使用兩個獨立 Python 環境，且指定尚不存在的私人輸出資料夾。以下占位值須換為合法取得的實際路徑；此指令會載入模型，應先完成環境與來源審查。

```powershell
<ML_PYTHON> -I -B -X utf8 inference/v25/source/src/predict_v25.py `
  --artifacts <PRIVATE_ARTIFACT_ROOT> `
  --test <ORIGINAL_COURSE_TEST.xlsx> --sample <ORIGINAL_COURSE_SAMPLE.csv> `
  --nn-python <TORCH_PYTHON> --device cuda:0 `
  --expected-closure-sha a781bc2d3c08464c69e591d60ff7deb344aac946c55bdd405ce41da884f0dc30 `
  --output-dir <NEW_PRIVATE_OUTPUT_DIRECTORY>
```

依賴鎖針對 Windows CPython 3.12／CUDA 12.6。已實測設定為 NN batch 4096、4 執行緒與 matmul high。未驗證其他裝置、CPU／GPU 逐 bytes 等價或不同版本；R2 reference 中間向量未直接輸出，不能宣稱該中間層直接逐值相同。新輸出若不符合選定 CSV SHA，入口會保存失敗證據並以非零狀態結束，不修改選定檔案或自動修價。

原 R107 不利驗證及反覆使用驗證窗口的選型限制保留。新研究 R123–127 不包含在這份交付；研究結果不自動改變使用者選定的 v25。

## 兩套固定依賴環境

以下在儲存庫根目錄建立兩個尚不存在的 Windows x64／CPython 3.12 venv。已驗證的 base Python 為3.12.14；鎖檔含官方 wheel URL 與 SHA，pip 只安裝列明的25／11項，不自動解析其他依賴。實際新環境安裝已完成，不代表其他硬體也已驗證。

```powershell
py -3.12 -m venv .venv-v25-ml
py -3.12 -m venv .venv-v25-nn
.venv-v25-ml/Scripts/python.exe -m pip install --no-deps --require-hashes -r inference/v25/dependencies/requirements-fresh-ml.lock.txt
.venv-v25-nn/Scripts/python.exe -m pip install --no-deps --require-hashes -r inference/v25/dependencies/requirements-fresh-nn.lock.txt
.venv-v25-ml/Scripts/python.exe -m pip check
.venv-v25-nn/Scripts/python.exe -m pip check
```

一般 requirements 中的 `torch==2.13.0` 不等於已驗證的 `2.13.0+cu126` wheel。原推論設定、17個artifact、test與sample的SHA及4096 batch均固定；不能自行換依賴後宣稱沿用本次逐位元相同結論。

## 原始訓練程式補齊

另依只讀盤點補入22支原始訓練／資料準備來源與2份config/schema，保留原bytes、歷史guard及指定路徑。[六模型入口與命令證據](training/README.md)區分來源可取得與尚未完成的raw→六模型重訓；沒有因來源補齊宣稱已完成重訓。[可選地理建置依賴](training/GEOGRAPHY.md)另列歷史Shapely／PyProj／Osmium版本，未做fresh地理安裝／重建，保存模型推論不需要重建地理資料。
