# 同機新環境安裝核驗

2026-10-02。在原Windows電腦，以已存在的Python3.12.14 base另建venv，不繼承系統或user site套件。原ML／Torch環境、兩份requirements、模型、CSV與選定版本均未改。這次補上「程式交給別人前，套件是否真的能安裝及載入」的證據。

## 實際完成

| 項目 | 本次結果 |
| --- | --- |
| 初始隔離 | 新venv內八項主要套件均不存在；system site關閉、user site關閉、使用-I |
| 安裝 | 原requirements-inference.txt及requirements-torch.txt，官方PyPI，僅wheel，35個套件輪檔 |
| 依賴 | ML安裝後及Torch安裝後的pip check均通過 |
| 載入 | 八套件、cKDTree、threadpoolctl、CatBoost及torch.nn成功；2,130個已載入第三方模組及直接套件metadata都落在新venv |
| 程式入口 | predict_v22.py --help、nn_predict.py --help成功；未傳資料／模型 |
| 格式 | 原四項unittest通過；只測提交格式、防錯與取整，不測房價預測 |
| PyTorch | distribution記2.13.0；runtime實讀2.13.0+cpu；torch.version.cuda為null |
| 模型／預測 | 載入0、fit 0、推論0，沒有GPU運算 |

原精確重現用Torch2.13.0+cu126／CUDA12.6；本次CPU wheel不同。`torch==2.13.0`只限制public版本，沒有要求CUDA組建。README的`cuda:0`推論必須另備相容GPU環境；不要直接在此CPU環境執行並期待原CSV。未驗證新的CUDA安裝命令，所以此處不把未測命令當保證。

## 證據與時間

- [整體收據](../evidence/environment_20261002/environment_receipt.json)：各命令退出碼、時間、環境與保護檔SHA。
- [35個wheel來源與SHA](../evidence/environment_20261002/installed_wheels.json)：本次解析結果，不是跨平台lock；全部來自files.pythonhosted.org。
- [安裝後36項清單](../evidence/environment_20261002/installed_packages.txt)：含venv自帶pip25.0.1；freeze的Torch字串不顯示runtime的+cpu，須併看下項。
- [實際import版本及路徑](../evidence/environment_20261002/imports.json)、[Python隔離狀態](../evidence/environment_20261002/packages.json)、[venv設定](../evidence/environment_20261002/tested_pyvenv_config.txt)。
- [格式測試](../evidence/environment_20261002/format_tests.txt)、[主入口help](../evidence/environment_20261002/prediction_help.txt)、[NN入口help](../evidence/environment_20261002/neural_help.txt)。
- [import核對程式](../evidence/environment_20261002/import_check.py)、[環境讀取程式](../evidence/environment_20261002/inspect_environment.py)：可用受測venv的Python加-I -B執行；只讀套件，沒有模型推論。收據中的工作區絕對路徑是本次來源，不是其他電腦的安裝路徑。

ML解析35.273秒、安裝84.521秒；Torch解析22.357秒、安裝75.645秒；首批import62.294秒，兩help各1.618／1.601秒，格式測試子程序0.502秒（unittest測試本身0.013秒）。這些是逐命令量測，不是總工時。解析階段已下載wheel至新的工作區cache，安裝時重用此cache；不是每階段都重新下載。

第一次sandbox網路被WinError10013拒絕，[原stderr保留](../evidence/environment_20261002/first_network_failure.txt)；隨後正常使用者權限連官方PyPI成功。不能把第一次的「no matching distribution」當套件版本不存在。

## 驗證界線

這是**同機、同base Python的新套件環境**，不是新電腦／新OS／驅動安裝。未驗證此環境的模型載入、47,073筆推論、CSV一致、CUDA運算、原始地理重建或全量重訓，也未測PPTX／PDF／影片的依賴。原84.7364秒推論收據及歷史summary原文保持，不能把舊結果轉移到新環境。

此程式包附摘要、診斷source及部分原始輸出；完整pip解析報告、命令／stderr／來源SHA留在主工作區`outputs/v22_clean_environment_20261002/`。receipt的evidence_sha256含這些本機檔案，不代表它們全在GitHub副本。venv與cache不納入GitHub或文件ZIP。

唯一提交仍v22，Public MAE 1,998,568.5706元，Private未知。沒有新增CSV或操作Kaggle。
