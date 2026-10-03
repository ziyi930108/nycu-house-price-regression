# 2026-10-03：推論前先檢查神經網路環境

指定錯誤的 `--nn-python`，或使用沒有可用 CUDA 的環境時，舊入口會先算四個樹模型、寫入神經網路輸入，最後才失敗。新版會先用指定的 Python 檢查 Torch／NumPy 和所選裝置，檢查成功才核對資料與 artifact、建立輸出並計算模型。預測公式、保存模型和 v22 提交選擇不變。

## 實際行為

- 既有輸出目錄直接拒絕，保留所有舊檔；不先啟動 NN 檢查。
- NN Python 必須是存在的檔案；以絕對路徑啟動，保留符號連結，以免切換到虛擬環境外的 Python。
- 子程序沿用正式 NN 執行器的 imports；以一個元素檢查所選裝置初始化及回傳 NumPy。檢查不讀模型、資料或保存預測，不寫成品，逾60秒退出。
- 檢查失敗會給非零結束碼、帶出原始依賴或 CUDA 錯誤，且不建立輸出目錄。不會自動改用 CPU。`--device cpu` 是明確選項，但受測 CPU 數值不能精確重播原 GPU CSV。
- 父程式和實際 NN 使用相同 interpreter 路徑；子程序沒有新增 `-I`，仍依 README 清除 PYTHON 環境變數。檢查通過不是後續無故障保證；模型、資料內容及最後 CSV SHA 繼續各自檢查。

可單獨確認 NN 環境（不需要私有模型）：

```powershell
.venv-nn\Scripts\python.exe src/nn_predict.py --check-environment --device cuda:0
```

## 驗證與限制

受控測試先在舊程式重現：不存在 Python、缺 Torch、CUDA 不可用三種情況都先到達樹模型 stub。修正後共10項單元測試，9項通過；本機不能建立檔案符號連結，因此該項跳過，沒有宣稱已完成 Linux 或符號連結 venv 實測。[原失敗紀錄](../evidence/cli_preflight_20261003/baseline_tests.txt)與[最後測試紀錄](../evidence/cli_preflight_20261003/tests.txt)都保留。

另用真實執行器驗證不存在路徑、CPU Torch 環境要求 CUDA 會在樹模型訊息前退出，沒有輸出目錄；同一 CPU 環境的明確 CPU probe 成功。[實際拒絕收據](../evidence/cli_preflight_20261003/negative_checks.json)。缺 Torch／無法啟動／timeout 的測試使用 stub，沒有新建缺套件環境。

最終原碼在同機既有 ML／GPU venv 重跑六個保存模型、47,073筆，主程式 **81.837492秒**，CSV及六個比較向量（兩NN、三修正分支、最終未取整總價）逐bytes與原版相同。17項 artifact 的SHA保持；沒有重訓、新安裝或新的Public／Private分數。CPU/GPU差異不能解讀成test MAE。

CSV SHA256：`6490e2da78d4bf8173a27444de673a020152878f78e7ebec54d4fdc9bddfae35`。詳見[最終來源與完整收據](../evidence/cli_preflight_20261003/final_verification.json)。首次修改後已重跑成功，但獨立審查發現 `resolve()` 可能改變符號連結型 venv；改成 `absolute()` 後才進行上述最後重跑。兩次來源與收據分開保留，[第一輪收據](../evidence/cli_preflight_20261003/first_verification.json)不回寫。

研究維持停止、v22唯一提交、Private未知。既有簡報、4,065字稿、原旁白、影片與12頁正式PDF保持；接手頁的過時11頁文字同步修正。
