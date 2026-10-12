# 簡報重建來源

本資料夾對應16頁製作流程簡報，全部為主講、無附錄。完整口語講稿 4,298 純漢字，275字／分估約15:38，250–300字／分估14.33至17.19分鐘，未本人計時。

以 `build_presentation.mjs` 讀取同目錄 `content.json`、`training_curve.json` 及 `assets/house_illustration.png`，產生可編輯候選；再以 `finalize_presentation.mjs` 完成套件檢查與逐頁渲染。需可用的 Node.js、@oai/artifact-tool、Python 及 Presentations 技能工具，環境路徑可由檔內環境變數調整。此來源重建簡報，不會訓練房價模型。

```powershell
node ./build_presentation.mjs C:/path/to/new-build
node ./finalize_presentation.mjs C:/path/to/new-build r1
```

輸出目錄須使用新路徑。圖表維持原生可編輯資料，finalizer補入對應Excel資料表。曲線原始60個MAE值沿用保存資料，含單一網路診斷限制，沒有用圖中最低點挑選輪數。房屋插圖為既有生成示意圖，來源記錄在 assets/house_illustration.json。

實際完成的重建與視覺核對以本次交付收據為準。不同Office版本可能略有重排，未宣稱原生PowerPoint或全新電腦安裝已測試。模型／CSV未改，Private未知。

本次僅重寫第9、10頁與備忘稿，先說日期加權的原因與影響，再以相似交易參考價與總價算例說明。保留第7頁價格目標對照及其餘14頁；7張原生表、3原生圖和原60點曲線保持。查表分類與建案分組隔開答案是不同步驟，分組OOF不等同時間OOF。
