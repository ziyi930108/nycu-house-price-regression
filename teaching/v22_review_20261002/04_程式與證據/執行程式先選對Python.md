# 執行程式時使用明確的Python

請從[repo README的Windows／Python3.12受測路線](../../../README.md)依序操作。安裝、收集artifact及主推論使用明確的ML Python；兩個NN另由`--nn-python`指定GPU環境。這份補充只交代分工，實際命令集中在README，避免兩份指令版本不同。

同機新建ML與GPU套件環境的保存模型推論已驗證，見[CUDA重現範圍](../../../docs/CLEAN_GPU_REPLAY_20261002.md)。一般PyPI安裝得到的Torch CPU組建，與受測的`2.13.0+cu126`不同，不能直接當成同一條GPU路線。原requirements、wheel身分與實際版本都有紀錄。

這次共用原作業系統、Python主程式、GPU與驅動。新電腦、其他硬體、從原始train/valid重訓及地理重建仍未驗證；直接複製虛擬環境資料夾不保證可攜。

推論需要原test、sample與17項保存模型／狀態，見[資料與artifact用途](../../../DATA.md)。repo不附全部私有資料；不要用舊預測代替缺模型，也不要放寬SHA檢查。輸出使用尚不存在的新目錄，失敗時保留日誌再核對重現說明。

閱讀簡報、PDF、影片與練習工具不需要安裝ML環境。此處沒有啟動任何安裝、模型訓練或推論。
