# 台灣房屋銷售價格預測挑戰賽（HW1）

姓名：廖子頤　學號：315581048  
[Kaggle 個人頁面](https://www.kaggle.com/ziyi315581048)

根據房屋的地點、面積、樓層、類型與交易時間，預測每筆交易的成交總價，單位為新臺幣元。評估指標為平均絕對誤差（MAE），數值越小，代表平均預測誤差越低。

**正式報告：[PDF](report/台灣房屋銷售價格預測挑戰賽_正式報告.pdf) · [Word](report/台灣房屋銷售價格預測挑戰賽_正式報告.docx) · [線上文字](report/REPORT.md)**

## 方法與製作流程

1. **整理資料**：排除誤植表頭與完全重複的交易，處理日期、面積與文字欄位。
2. **建立特徵**：將房屋條件轉成模型可使用的數值與類別，所有轉換規則由當期訓練資料建立。
3. **訓練模型**：使用 CatBoost、LightGBM 與神經網路，共六個模型，分別學習總價、單價及參考單價的調整倍率等表示。
4. **整合估價**：將各模型輸出換算成總價，組成三路估價，套用成交參考與空間修正，再取三路總價的中位數。
5. **檢查與輸出**：以較早交易訓練、較晚交易驗證，計算總價 MAE；正式預測依範例檔的編號與順序輸出 CSV。

## 檔案位置

| 項目 | 入口 | 內容 |
| --- | --- | --- |
| 專案說明 | 本頁 | 目標、流程、安裝與操作順序 |
| 程式 | [src/](src/README.md) | 預測入口及清理、特徵、六模型訓練、驗證與融合程式導覽 |
| 設定 | [config/](config/README.md) | 預測路徑設定及模型參數、時間切分與融合設定的來源 |
| 套件需求 | [requirement.txt](requirement.txt) | 主環境的固定套件；神經網路環境另行安裝 |
| 正式報告 | [report/](report/README.md) | 最新18頁、八章與一份附錄，含Word、PDF與製作來源 |

原始訓練來源放在 `research/`，已驗證的六模型推論實作放在 `inference/`，相關紀錄放在 `evidence/`。這些是上述入口所使用的來源與依賴。[上台簡報與講稿](teaching/final_presentation/README.md)另附於簡報資料夾。

## 安裝環境

已驗證環境為 **Windows x64、Python 3.12.14、CUDA 12.6 與 NVIDIA GPU**。主程式與神經網路使用兩個獨立環境；以下命令在儲存庫根目錄執行：

```powershell
py -3.12 -m venv .venv-ml
py -3.12 -m venv .venv-nn
.venv-ml/Scripts/python.exe -m pip install --no-deps --require-hashes -r requirement.txt
.venv-nn/Scripts/python.exe -m pip install --no-deps --require-hashes -r requirements-torch.txt
.venv-ml/Scripts/python.exe -m pip check
.venv-nn/Scripts/python.exe -m pip check
```

套件清單使用已驗證的固定下載位置與檔案指紋。閱讀報告不需安裝；重新製作Word才需要 [requirements-report.txt](requirements-report.txt)。其他作業系統或GPU環境尚未驗證。[完整環境與資料準備](docs/USAGE.md)

## 執行順序

先查看設定與程式來源，這一步不需交易資料或模型：

```powershell
python src/predict.py --config config/predict.json --describe
python scripts/verify_source.py
```

實際預測前，須備妥原測試資料、範例提交檔，以及已保存的六模型與必要轉換狀態。依 [config說明](config/README.md)複製設定為自己的本機設定，填入檔案路徑，再執行：

```powershell
.venv-ml/Scripts/python.exe src/predict.py --config config/predict.local.json --check-inputs
.venv-ml/Scripts/python.exe src/predict.py --config config/predict.local.json
```

輸出會寫入設定指定的新資料夾，包含預測CSV與執行紀錄。原始交易資料、模型權重及逐筆預測保存在本機，未包含於此公開儲存庫。詳細輸入、輸出檔名與操作說明見 [執行指南](docs/USAGE.md)。

## 結果與驗證範圍

Public MAE為 **1,995,836.2136元**，Private成績未知。歷史驗證有改善也有退步，反覆使用相同驗證期間可能造成選型偏差；完整結果與限制見正式報告。

已實測載入保存的六模型與完整轉換狀態重新預測，47,073筆CSV與選定答案完全一致。[推論驗證紀錄](inference/v25/evidence/reproducibility_receipt.json)記錄環境與比較範圍；本次新增設定入口也完成[完整預測核對](docs/VALIDATION.md)。從原始訓練資料重新訓練全部六模型的完整流程仍未驗證；[原始訓練程式與前置條件](inference/v25/training/README.md)已列明，不能直接當成一鍵重訓命令。
