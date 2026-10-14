# 安裝與執行指南

## 1. 工作與前置條件

| 工作 | 需要準備 | 入口 |
| --- | --- | --- |
| 閱讀報告與程式 | 下載儲存庫即可 | [正式報告](../report/README.md)、[程式導覽](../src/README.md) |
| 檢查設定及來源 | Python 3.12，無需模型與ML套件 | `python src/predict.py --describe`、`python scripts/verify_source.py` |
| 由保存模型產生預測 | 兩套環境、原測試資料、範例檔、保存模型與轉換狀態 | `src/predict.py` |
| 閱讀或接續原始訓練 | 原始train／valid、地理資料與各階段前置狀態 | [六模型訓練來源](../inference/v25/training/README.md) |
| 重新製作報告 | 文件套件及已保存的報告來源 | [報告資料夾](../report/README.md) |

原始訓練程式保留當時的路徑與資料核對條件，尚未完成從下載此儲存庫到六模型全部重訓的整套測試。請先閱讀各模型前置條件；不得用保存的預測向量代替重新訓練。

## 2. 安裝兩套環境

在儲存庫根目錄依 [README安裝命令](../README.md#安裝環境)建立兩個尚不存在的Windows x64／Python 3.12環境。已驗證的Python版本為3.12.14，GPU使用CUDA 12.6。

`requirement.txt`讀取主環境25個固定套件；`requirements-torch.txt`讀取神經網路環境11個固定套件，其中PyTorch為`2.13.0+cu126`。清單保留已驗證wheel的位置與SHA256。`--require-hashes`核對下載檔案，`--no-deps`避免額外安裝未列入版本，最後以`pip check`檢查套件相依性。

`requirements-inference.txt`是主要套件版本摘要，完整固定安裝以`requirement.txt`為準。`requirements-report.txt`用於文件製作；地理從頭建置的歷史依賴另見 [GEOGRAPHY](../inference/v25/training/GEOGRAPHY.md)，保存模型推論不用重建地理資料。

## 3. 準備私人輸入

- 原測試檔 `test-reindex-test-v2.2.xlsx`。
- 原範例提交檔 `test-reindex-test-sample-submission-v2.csv`。
- [模型與轉換狀態清單](../inference/v25/source/artifacts/manifest.json)中的17項檔案，保留各檔案在私人根目錄下的相對位置。

模型、編碼器、參考價格、空間修正與地理資料要對應同一套已保存狀態。程式會核對原輸入與狀態的SHA256，不能任意替換。清單只公開檔案角色與指紋，不含資料內容。訓練資料整理見 [DATA.md](../DATA.md)。

## 4. 填入設定

```powershell
Copy-Item config/predict.json config/predict.local.json
```

編輯`config/predict.local.json`，依 [設定說明](../config/README.md)填入私人檔案位置。相對路徑統一從儲存庫根目錄計算。`nn_python`指向`.venv-nn/Scripts/python.exe`；主程式使用`.venv-ml/Scripts/python.exe`。

輸出資料夾必須尚未存在；本機設定與輸出不要加入Git。

## 5. 先檢查，再執行

```powershell
python src/predict.py --config config/predict.local.json --describe
.venv-ml/Scripts/python.exe src/predict.py --config config/predict.local.json --check-inputs
.venv-ml/Scripts/python.exe src/predict.py --config config/predict.local.json
```

`--describe`顯示設定及封存程式資訊，不載入模型。`--check-inputs`檢查執行前的檔案與路徑；通過不代表已產生預測，正式推論仍會核對來源、資料與模型。

正式執行載入四個樹模型與兩個神經網路，重新建立測試特徵、三路估價及中位數。輸出資料夾包含：

| 檔案 | 用途 |
| --- | --- |
| `submission_v25_reproduced.csv` | 47,073筆總價，順序與原範例檔一致 |
| `receipt.json` | 執行環境、模型數量、輸出指紋與比對結果 |
| `preflight.json` | 本次來源、輸入與環境檢查 |
| 各模型與三路的`.npy`／`.npz` | 本次推論中間資料，留在私人輸出資料夾 |

CSV應符合固定選定檔案的指紋；若不符，程式保存結果並以失敗狀態結束，供檢查環境或數值差異，不會自動改價或上傳Kaggle。完整既有證據見 [推論紀錄](../inference/v25/README.md)。
