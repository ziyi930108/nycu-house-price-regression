# 簡報與講稿的文件重建

本資料夾的來源檔與房屋插圖只重建教學簡報。它們不會重新訓練六個房價模型，不會改提交價格，也不操作Kaggle。

## 來源內容

- `build_presentation.mjs`：建立22頁可編輯PPTX草稿。
- `finalize_presentation.mjs`：核對PPTX結構、原生圖表與活頁簿，再渲染22頁。
- `content.json`：22頁的標題、講稿、補充理由與來源對照。19頁主講正文為3758個純漢字。
- `training_curve.json`：固定30輪、兩條線，共60個原始總價MAE數值，單位新臺幣元。
- `README.md`：本說明。
- `assets/house_illustration.png`：透明背景房屋示意插圖，放在第2頁右下。
- `assets/house_illustration.json`：內建image_gen模式、完整生成提示詞、雜湊與位置；插圖並非實際交易物件。

程式從自身的source資料夾讀取內容與曲線，不依賴舊版媒體或工作區outputs。曲線屬於另行固定設定的單一神經網路診斷，不是完整融合，也不是原提交當時的逐輪紀錄。Excel活頁簿使用15位有效數字，與原始JSON差異小於0.000001元，原始JSON數值不改。

## 必要執行環境

需要Codex提供的bundled Node、`@oai/artifact-tool`、Python及Presentations技能finalizer，還需Microsoft JhengHei與Consolas字型。這是同一套文件執行環境的重建方式，沒有宣稱全新電腦或任意硬體可直接安裝重現。

可覆蓋的環境變數：

- `RUNTIME_NODE_MODULES`：bundled Node packages的絕對路徑。
- `RUNTIME_PYTHON`：bundled Python執行檔。
- `PRESENTATIONS_SKILL_DIR`：Presentations技能目錄，內含container_tools。

Node執行檔由呼叫者指定。程式內預設路徑對應目前的Codex bundle；換電腦時請先取得新機的bundle位置，再設定以上變數。

## 重建步驟

在本資料夾的上一層執行以下PowerShell命令。第一個位置參數可指定新的輸出目錄，若省略則使用source旁的`rebuild/`。finalize的第二個位置參數為輸出編號，例如r1。

```powershell
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' './source/build_presentation.mjs' 'C:/NYCU_ML/outputs/my_presentation_rebuild'
& 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' './source/finalize_presentation.mjs' 'C:/NYCU_ML/outputs/my_presentation_rebuild' r1
```

產物包含`candidate.pptx`、`final/r1.pptx`、`renders_r1/slide-01.png`至`slide-22.png`及核對收據。再次finalize請換編號，既有final檔不覆寫。這些產物不會覆寫交付資料夾根目錄的PPTX。

Codex進行新的文件編輯任務時，仍應依Presentations技能先記錄一次artifact operation，最後逐張確認渲染圖。自動核對只驗證結構與資料，不能代替人眼檢閱或PowerPoint操作驗證。

## 講稿與時間

完整口語講稿放在source的上一層。此次刪除原第4頁「四份資料的用途」及原第5頁「表頭排除與完全去重」，後續頁碼減2。封面先前刪除及第2頁房屋插圖保持，其餘22頁的口語正文、補充理由與備忘稿維持原文，只重編章號及配速表。主講第1至19頁共3758純漢字，275字／分估13分40秒；250–300字／分約12分32秒至15分02秒。附錄第20至22頁與問答不計；尚未本人計時排練。
