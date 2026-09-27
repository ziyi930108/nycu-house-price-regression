# 可重現範圍

測試問題是「保存v22模型與狀態能否重新生成原提交」，不能以舊預測向量重組代替模型推論。

## 環境

Windows，Python3.12.14。ML環境：NumPy2.3.5、pandas3.0.1、CatBoost1.2.10、LightGBM4.7.0、SciPy1.18.1、scikit-learn1.9.1、openpyxl3.1.5。

NN環境：Torch2.13.0+cu126、CUDA12.6、NVIDIA GeForce RTX4060Ti、batch4096、4threads、float32 matmul precision high。實測採兩個既有環境，--nn-python指定Torch；未驗證全新單一環境安裝。

## 核驗

1. 原始Excel重建87特徵、核對frame hash，sample全量ID與順序。
2. 17項artifact獨立複製且逐檔核對SHA，載六個模型。
3. 從encoder／prior重建兩個NN共8個query陣列，與原編碼全部位元一致。
4. 六模型真正推論47,073筆，受測GPU輸出與原native完全一致。
5. 重算reference/spatial與median，最後一次取整。
6. 比對CSV完整SHA。總入口收據見evidence/reproduction。

本次新增訓練0次，沒有改原模型、原CSV或selector。保存模型推論、向量重組、模型重新訓練是不同驗證。

## CPU差異

全量面積NN的CPU對原GPU，平均絕對預測差565.275492元、最大748,585.011808元、47,010筆取整值不同。這不是test真實MAE，不能推定哪個裝置較準。精確重播應用受測GPU設定，其他硬體／套件不保證相同。

## 未驗證

- 全新套件安裝、其他作業系統或GPU。
- 從原始OSM快照重建地理。
- 從清理train/valid、OOF至全部模型重訓的可攜全流程。
- 任意新資料的地理對齊與泛化。
- Private分數、平台實際上傳bytes。

test固定SHA檢查避免拿錯列序配保存地理狀態。新模型或配方需另建版本和清單，不覆寫v22身分。有實質進度時同步更新原專案HANDOFF.md、HANDOFF_LOG.md及交付報告與收據。

## 最終結果

隔離模式單一入口完成，47,073筆未取整預測位元一致；CSV SHA256完全等於原v22。實測84.7364秒，4項格式測試通過。詳見[摘要](../evidence/reproduction/summary.json)與[收據](../evidence/reproduction/integration_verified.json)。第一次僅LF／CRLF差異導致全檔核驗失敗，已修正並整條重跑；失敗收據保留且摘要標記舊status欄誤導，未刪除。
