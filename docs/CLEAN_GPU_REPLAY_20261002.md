# 同機新GPU套件環境重現v22

2026-10-02。重新安裝套件後，保存六模型仍生成與原v22完全相同的提交；這是程式交付的重現驗證，不是新模型版本或Private評分。

## 實際結果

沿用同日新建的ML套件venv，另建不繼承system site的NN venv。新NN環境由[官方CUDA12.6索引](https://download.pytorch.org/whl/cu126/torch/)取得Windows x64／CPython3.12的Torch2.13.0+cu126，固定wheel SHA；NumPy2.3.5。原Torch環境與上一階段CPU環境均保留。

| 項目 | 結果 |
| --- | --- |
| 輸入 | 原test Excel、sample的ID／schema、17項保存模型／地理／fit狀態 |
| 執行 | 六模型，47,073筆，fit 0；沒有用保存預測或test真價當模型輸入 |
| 主程式 | 76.930302秒；外層子程序78.939654秒，兩者計時範圍不同 |
| 浮點比對 | 最終未取整、N_area、N_prior、三條修正分支，共6向量逐bytes同 |
| CSV | 734,491bytes，SHA256 `6490e2da78d4bf8173a27444de673a020152878f78e7ebec54d4fdc9bddfae35` |
| 未取整SHA | `93ac0a1366dd4806a90e0007360b9f72a8e57415413edfb0bedca6d4781c9d0d` |
| GPU | RTX4060Ti、driver616.64，Torch組建CUDA12.6；本機既有驅動未重裝 |

兩個NN加三分支與最後融合是一條計算流程的不同產物，六個相同向量不是六份獨立泛化證據。零預測差也不是test MAE，更不表示Private誤差為零。唯一提交仍v22，Public MAE 1,998,568.5706元，Private未知。

## 重做方式與依賴

使用[README的快速開始](../README.md)。`.venv`是ML環境，`.venv-nn`是GPU環境；`--nn-python`明確指定後者。ML沿用前階段兩requirements的受測安裝組合，其中CPU版Torch不負責本次NN推論。

[GPU安裝清單](../config/requirements-torch-cu126-win-py312.txt)的兩項規格與本次pip命令完全相同，僅把實際命令參數整理成-r檔案；未為此再次下載／安裝。GPU wheel約2.6GB，SHA `380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2`。不是跨平台或全部transitive套件的hash lock；Windows以外、不同Python標籤請勿沿用此wheel。原requirements-torch.txt及原模型清單不改。

安裝457.647秒，首次GPU import／device probe56.993秒，只是本機量測，不保證別台時間。GPU安裝來源含官方PyTorch wheel與PyPI傳遞套件；ML環境freeze在開跑前重新讀取，與前階段相同。

## 隔離、程序及失败紀錄

新兩venv都不繼承system site。父程式-I不會傳給原NN子程序；本次在外層清除所有PYTHON開頭變數，再設PYTHONNOUSERSITE=1／PYTHONDONTWRITEBYTECODE=1。NN以無旗標方式啟動，沒有改原nn_predict.py；分開probe實讀isolated=0、no_user_site=1，套件來自新NN venv。這個probe不是實際NN逐import追蹤，腳本所在目錄不同。

第一個開跑檢查因nvidia-smi非空停止：本WDDM把已觀察的Windows／瀏覽器桌面程式列為C+G。第二次Get-Process名稱查詢回exit1，stdout只有控制器兩程序；缺少pythonw是原因推斷，原stderr受抑制，不能說日誌明寫原因。兩次都未建立predict命令或載模型。之後保留原失敗，允許已逐項看過的桌面程序名稱、拒絕未知GPU程序與其他Python工作，才完成唯一一次真正模型推論。一般桌面圖形工作仍存在，不宣稱獨占GPU。

## 可查證檔案

- [重現彙總](../evidence/clean_gpu_20261002/reproduction_receipt.json)、[原入口收據](../evidence/clean_gpu_20261002/inference.json)、[獨立審查](../evidence/clean_gpu_20261002/independent_review.json)。
- [完成後的向量比較](../evidence/clean_gpu_20261002/comparison.json)、[面積NN](../evidence/clean_gpu_20261002/N_area.json)、[先驗NN](../evidence/clean_gpu_20261002/N_prior.json)。比較在推論完成後另程序執行。
- [官方索引記錄](../evidence/clean_gpu_20261002/official_index.json)、[安裝wheel SHA](../evidence/clean_gpu_20261002/wheel_verified.json)、[GPU套件清單](../evidence/clean_gpu_20261002/gpu_packages.txt)、[硬體](../evidence/clean_gpu_20261002/gpu_hardware.txt)。
- [GPU環境probe](../evidence/clean_gpu_20261002/gpu_probe.json)、[probe程式](../evidence/clean_gpu_20261002/gpu_probe.py)、[實際主程序命令](../evidence/clean_gpu_20261002/command.json)、[環境控制](../evidence/clean_gpu_20261002/environment_controls.json)。

彙總含本機完整證據的SHA；GitHub只附本頁連結的部分檔案，沒有venv、2.6GB wheel、模型、CSV或預測向量。完整日誌與兩次停止紀錄在主工作區`outputs/v22_clean_gpu_20261002/`。初版收據保留，只把第二次停止原因改寫成推斷，數值與計算不變。

這次沿用既有作業系統、Python base、GPU與驅動，以及保存地理／模型。**沒有從train/valid重訓、重建原OSM、測試另一台電腦或量到Private分數。** 原84.7364秒推論收據、上一階段CPU安裝收據及所有原件保留；各自描述當時範圍，不回寫成新結果。
