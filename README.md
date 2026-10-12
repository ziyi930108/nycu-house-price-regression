# 台灣房屋銷售價格預測挑戰賽

姓名：廖子頤　學號：315581048

本儲存庫提供最終提交方法的程式、報告、簡報及必要證據。

1. [Word正式報告](docs/台灣房屋銷售價格預測挑戰賽_正式報告.docx)：19頁，包含程式流程、清理與文字轉換、城市統計、完整融合驗證圖及誤差案例。
2. [線上完整報告](REPORT.md)與[章節證據對照](docs/REPORT_SOURCES.md)。
3. [上台簡報與講稿](teaching/final_presentation/README.md)：20頁皆主講，無附錄；4,222純漢字，275字／分估15:21，尚未本人計時。
4. [六模型可攜推論](inference/v25/README.md)、[原始訓練來源與命令證據](inference/v25/training/README.md)、[程式與重現界線](docs/V25_REPRODUCIBILITY_20261010.md)。
5. [選定與分數證據](docs/V25_SELECTION_20261010.md)。

Public MAE **1,995,836.2136元**，使用者截圖與登入頁面已核對；Private未知。六個原生模型的原 metadata 各記467,221筆 fit，測試47,073筆。歷史時間驗證反覆用於選型，跨期與分群效果不一致，不利證據保留。

## 執行與依賴

目前推論設定見[已驗證入口的設定](inference/v25/source/config/v25.json)；原訓練設定與欄位規格見[原設定快照](research/config/original_v1_final_config.json)及[原欄位規格](research/config/original_feature_schema.json)。`python scripts/verify_source.py`核對完整來源快照。報告與簡報可依各自source說明重建，這不等於模型重訓。

課程指定的[requirement.txt](requirement.txt)仍載入三份一般套件清單。已驗證的六模型推論使用兩套獨立 ML／NN 環境，請依[固定25／11個官方wheel的hash-lock安裝指引](inference/v25/README.md#兩套固定依賴環境)；一般清單中的`torch==2.13.0`不能替代已驗證的`2.13.0+cu126` wheel與其SHA。[可選地理建置依賴](inference/v25/training/GEOGRAPHY.md)另列歷史版本證據，尚未驗證全新地理環境或raw地理重建。

價格標籤來自課程資料；禁止到內政部實價登錄系統核對答案。GitHub不包含交易表、模型權重、逐筆預測或提交CSV。原R107全量階段只新訓練參考單價網路，其餘五個沿用保存預測；其原收據與限制不回寫。後續已從保存的六模型及完整fit狀態，在同機既有與全新獨立套件環境各完成六個fresh forward。兩次13個完整向量與47,073筆CSV都完全一致，沒有新增fit；[公開彙總收據](inference/v25/evidence/reproducibility_receipt.json)列明範圍與成本。

另補22支原訓練／準備程式和2份設定／欄位規格，保持原bytes與歷史guard。從原始train／valid重訓全部六模型的端到端驗證仍未完成；同機新套件環境不等於新硬體驗證。原driver仍依賴歷史路徑、私有prepared/state或保存向量，來源齊全不能當成一鍵重訓已成功。

模型與選定CSV保持。單一NN逐輪曲線與完整融合結果分開說明；舊文件與來源保留於本機及Git歷史。新研究R123–127不包含在此交付。GitHub Classroom由使用者處理；Kaggle提交說明與最終勾選依使用者指示暫不修改。
