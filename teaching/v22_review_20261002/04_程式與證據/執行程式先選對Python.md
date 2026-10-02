# 執行程式時，固定使用同一個Python

安裝套件、收集artifact與推論須使用同一套ML Python。repo README已明確使用`.venv/Scripts/python.exe`；以下說明如何指定既有環境，避免裸`python`指向系統的另一套Python。

先完整下載或clone儲存庫，在repo根目錄打開PowerShell。保留原README的依賴與artifact要求，所有安裝、收集、推論及測試命令都明確使用同一個ML環境執行檔。**下列是操作指引，並未在全新環境重新安裝或訓練。**

若已按原README建立`.venv`，先確認路徑：

```powershell
$v22Python = (Resolve-Path -LiteralPath '.venv/Scripts/python.exe').Path
& $v22Python -c 'import sys; print(sys.executable); print(sys.version)'
```

如果要使用已有ML環境，將`$v22Python`改成該環境的Python完整路徑，不要指向Node、FFmpeg或其他程式。不要只複製虛擬環境資料夾就假設換電腦可用。原實測為Python3.12.14，使用ML與Torch兩套既有環境，版本及硬體詳見程式包`docs/REPRODUCIBILITY.md`。

原README的相關命令應一致寫成：

```powershell
# 要安裝時，安裝進指定的ML環境；新安裝尚未端到端驗證。
& $v22Python -m pip install -r requirements-inference.txt

# 僅在有原研究工作區時，依manifest複製17項私有artifact。
& $v22Python scripts/collect_artifacts.py --workspace C:/NYCU_ML

# 格式測試；通過不等於完成模型推論或重訓。
& $v22Python -m unittest discover -s tests -v
```

`--workspace`是你真正保存資料、模型與狀態的原研究目錄；換電腦需要自行移轉相應檔案，repo不含約1.45GB私有artifact。不要拿舊預測向量代替缺少的模型或自行放寬SHA檢查。

原始test與sample要按程式包DATA/README放到`data/`。若NN使用另一套Torch環境，推論時明確指定：

```powershell
$v22NnPython = Read-Host 'Torch環境的Python完整路徑'
if (-not (Test-Path -LiteralPath $v22NnPython -PathType Leaf)) { throw '找不到Torch Python。' }
& $v22Python src/predict_v22.py --test data/test-reindex-test-v2.2.xlsx --sample data/test-reindex-test-sample-submission-v2.csv --output-dir runs/v22_reproduction_new --device cuda:0 --nn-python $v22NnPython
if ($LASTEXITCODE -ne 0) { throw '推論未通過，請保留輸出與日誌後查錯。' }
```

ML與NN用同一個環境時可省略`--nn-python`，但須先備妥相容的Torch/CUDA。輸出目錄必须是尚不存在的新目錄。這個入口只對固定競賽test的已保存狀態重現，不能當任意房屋估價API。

原保存模型推論已在受測GPU環境完成47,073筆，CSV與v22一致；本補充沒有重跑推論。CPU或其他硬體的數值差異不是真實MAE，也不能據此判斷Private好壞。GPU重訓與乾淨電腦安裝仍未驗證。
