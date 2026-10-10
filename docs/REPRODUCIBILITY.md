# 版本界線更新｜2026-10-10

目前使用者選定v25，專屬範圍見[V25_REPRODUCIBILITY_20261010.md](V25_REPRODUCIBILITY_20261010.md)。下方所有六模型與新環境實測屬v22，不可移作v25認證。

下方為保留原 bytes 的歷史內容；選定、分數及重現範圍請以前述 2026-10-10 說明為準。

---

# 可重現範圍

測試問題是「保存v22模型與狀態能否重新生成原提交」，不能以舊預測向量重組代替模型推論。

## 原保存模型推論的受測環境

Windows，Python3.12.14。ML環境：NumPy2.3.5、pandas3.0.1、CatBoost1.2.10、LightGBM4.7.0、SciPy1.18.1、scikit-learn1.9.1、openpyxl3.1.5。

NN環境：Torch2.13.0+cu126、CUDA12.6、NVIDIA GeForce RTX4060Ti、batch4096、4threads、float32 matmul precision high。該次實測採兩個既有環境，--nn-python指定Torch；84.7364秒的收據仍屬這兩個環境。

## 2026-10-02同機新venv安裝

另以同一Python3.12.14 base建立不繼承系統套件的新venv，原兩份requirements安裝、pip check、八套件import、兩CLI的--help與四項格式測試通過。35個wheel來源／SHA、2,130個已載入第三方模組路徑與命令收據已保存；詳見[安裝核驗](ENVIRONMENT_INSTALL_20261002.md)。本次取得的是Torch2.13.0+cpu、compiled CUDA為null，與原+cu126不同。未在新環境載模型、推論或重訓，也未測GPU運算；不把兩批證據串成同一次端到端重現。

## 後續同機新CUDA環境重現

新增[實測路線與收據](CLEAN_GPU_REPLAY_20261002.md)：保留前階段CPU新venv作ML，另建Torch2.13.0+cu126／CUDA12.6的NN新venv。真正載入六模型、47,073筆，主程式76.930秒；CSV與六個比較向量全部逐bytes相同。原模型／source／requirements不改，fit0。父-I與子程序環境控制分開記錄，未宣稱全程-I、獨占GPU、新OS或全量重訓。

## 原保存模型推論核驗

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

- 其他電腦／作業系統／GPU或驅動組合；本機新套件環境的保存模型重播已另完成。
- 從原始OSM快照重建地理。
- 從清理train/valid、OOF至全部模型重訓的可攜全流程。
- 任意新資料的地理對齊與泛化。
- Private分數、平台實際上傳bytes。

test固定SHA檢查避免拿錯列序配保存地理狀態。新模型或配方需另建版本和清單，不覆寫v22身分。有實質進度時同步更新原專案HANDOFF.md、HANDOFF_LOG.md及交付報告與收據。

## 原兩個既有環境的推論結果

隔離模式單一入口完成，47,073筆未取整預測位元一致；CSV SHA256完全等於原v22。實測84.7364秒，4項格式測試通過。詳見[摘要](../evidence/reproduction/summary.json)與[收據](../evidence/reproduction/integration_verified.json)。第一次僅LF／CRLF差異導致全檔核驗失敗，已修正並整條重跑；失敗收據保留且摘要標記舊status欄誤導，未刪除。
