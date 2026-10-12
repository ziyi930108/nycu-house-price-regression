# 最終交付交接

入口見[README](README.md)。Word19頁；簡報20頁皆主講，4,222純漢字，275字／分估15:21，尚未本人計時。完整程式流程、城市統計、融合驗證圖、大誤差案例與對應改進及先前刪除要求保留。

Public 1,995,836.2136元有使用者截圖與登入頁面支持；Private未知，平台上傳bytes尚未核對。模型與選定CSV保持。單一NN曲線不是完整融合訓練軌跡，城市與案例是模型凍結後分析，原因是假說。原152個research/evidence保持逐bytes相同。

[六模型推論入口](inference/v25/README.md)完整保留20檔已實測source closure。在同機既有與全新獨立ML／NN套件環境，從保存模型與完整fit狀態各重新執行六個native forward；13個完整向量與47,073筆CSV均完全相同，兩次實際exit0。公開收據只含[彙總與身分](inference/v25/evidence/reproducibility_receipt.json)，不含逐筆資料。這是推論重現，不是未知test價格評分，也沒有新增fit或更換硬體。

[原始訓練來源](inference/v25/training/README.md)補22支程式及2份config/schema；原driver的歷史guard、固定路徑、私有prepared/state與保存向量依賴保留。從原始train／valid完整重訓全部六模型尚未驗證；[可選地理依賴](inference/v25/training/GEOGRAPHY.md)只有歷史版本證據，未做fresh地理安裝／重建。原R107只新fit一個NN的收據不因後續推論成功而回寫。新研究R123–127不發布。

Kaggle僅獲准改姓名與GitHub個人連結；提交說明、上傳與最終勾選明確延後。GitHub Classroom由使用者處理。接續教學使用短段與算例，不宣稱已講過就等於熟練。
