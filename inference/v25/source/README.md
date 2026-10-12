# v25 六模型完整推論草稿

這是獨立草稿，尚未真載入模型或產生價格；未更新 GitHub、舊入口、私有模型鏡像或選定 CSV。只接受課程原始 test Excel、sample 與固定 17 項 artifact。既有 encoder、price prior、R2 reference、R39 spatial state 全數保留；六個 native 模型、三分支與最終 median 都由本次新推論計算。

只讀描述（標準函式庫，不 import Torch／CatBoost／pandas）：

```powershell
python -I -B -X utf8 source/src/predict_v25.py --describe
```

根目錄 `closure.json` 凍結全部 source/config/manifest 與依賴規格。正式執行必須由審查者提供其 SHA，不接受外部已準備 NPZ、保存預測路徑或任意模型路徑；所有模型及 fitted state 先核清單 SHA 才反序列化。會建立新資料夾，拒絕任何既有輸出，也拒絕把輸出放在 source 之內。NN child 僅讀這次即時產生的 query-only 陣列，checkpoint 完整 configuration／cardinalities／numeric_count 都必須符合凍結 spec。

root 審查通過後可用以下格式執行，現在勿執行；GPU 與套件環境先由 root 協調。所有占位值均須換成真實路徑／審查 SHA：

```powershell
<ML_PYTHON> -I -B -X utf8 source/src/predict_v25.py `
  --artifacts C:/NYCU_ML `
  --test <ORIGINAL_COURSE_TEST.xlsx> `
  --sample <ORIGINAL_COURSE_SAMPLE.csv> `
  --nn-python <TORCH_PYTHON> --device cuda:0 `
  --expected-closure-sha <REVIEWED_CLOSURE_SHA256> `
  --output-dir <NEW_PRIVATE_OUTPUT_DIRECTORY>
```

目前固定需求沿用已驗證 v22 的 ML 套件及 Windows CPython 3.12 CUDA 12.6 NN wheel 規格。新環境尚未安裝或驗證；CPU 介面保留，但不承諾與 GPU 逐 bytes 一致。新跑輸出只寫指定的新私有資料夾，包含 query-only NPZ、六個 native 向量、分支、median、新 CSV 及收據；這些逐筆產物不放 repo／文件交接。程式不讀、不改原選定 CSV，只以凍結 SHA 比較新 CSV。與原選定 SHA 不符時保留完整輸出並以失敗結束，不能自動修價。

`src/tree_runtime.py`、`nn_prepare.py`、特徵與編碼程式、五個 closure helper 直接沿用 10 月 3 日完整 v22 推論成功的來源。`src/nn_predict.py` 只增加角色 spec／checkpoint SHA 與完整 configuration 檢查，網路算式及批次不變。新 `predict_v25.py`／`runtime_guard.py` 改用 R107 prior NN，保留原三分支算式，增加 source、原始輸入／列序、query 陣列身分、獨立輸出及前後 SHA 保護。
