# v22 上台答辯更新

2026-10-02。上台練習請用本資料夾的[四頁答辯卡PDF](臺灣房價預測_v22_答辯提示卡.pdf)、[可編輯文字版](臺灣房價預測_v22_答辯提示卡.md)與[排練及問答](排練與問答.md)。維持原16題，只更新第15題。

已在**同一台電腦的新套件環境**，由原始測試表和保存狀態重跑六模型，47,073筆預測與提交CSV均與原v22相同。這是保存模型推論重現。原始資料重訓全部模型、重建地理資料，以及新電腦／其他硬體仍未驗證。

原r2審查頁、程式先讀我和問答中的「全新環境安裝尚未驗證」描述的是當時進度；如今請以上述範圍回答。第19頁主講說「全新電腦尚未驗證」仍成立，所以本次不改主講稿、PPTX或15分鐘影片。問答不計主講時間。

逐頁技術來源實際在原審查包的`04_程式與證據/本次內容/narrative.json`，完整口語稿供口述。GitHub對應路徑為`teaching/v22_review_20261002/04_程式與證據/本次內容/narrative.json`。最新重現證據在repo的`docs/CLEAN_GPU_REPLAY_20261002.md`及`evidence/clean_gpu_20261002/`。

Public MAE仍為1,998,568.5706元，Private未知。沒有改模型、預測、提交選擇或Kaggle操作；預測相同不代表誤差為零。

## 重建答辯卡

保留`sources`內兩檔，在具備ReportLab、pypdf的Python與Poppler環境執行以下命令。把Python、字型、pdftoppm換成自己的絕對路徑；輸出資料夾必須尚不存在。此程式只做四頁PDF與文字稿，不會製作簡報或執行模型。Microsoft JhengHei字型沒有放入套件，其他字型可能改變換行，需另看渲染。

```powershell
& 'C:/path/to/python.exe' sources/build_cards.py --output 'C:/path/to/new-output' --font 'C:/Windows/Fonts/msjh.ttc' --bold-font 'C:/Windows/Fonts/msjhbd.ttc' --pdftoppm 'C:/path/to/pdftoppm.exe'
```

這個命令是需要替換路徑的範例。本次在既有本機PDF工具環境實跑；不是全新文件工具安裝驗證。`build_receipt.json`記錄實際身分、版面與生成時間，`independent_review.json`記錄內容核對，檔案清單見`manifest.json`。
