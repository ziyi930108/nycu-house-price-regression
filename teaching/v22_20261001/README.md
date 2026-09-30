# v22 零基礎簡報與報告

本套教材說明同一個已選定的v22模型。先解釋回歸、特徵、學習與推論，再說明資料整理、時間回測、單價權重與六模型組合。每個設計都有白話理由、實際做法及證據界線。

## 閱讀與上台順序

1. 先讀[零基礎報告](deliverables/臺灣房價預測_v22_零基礎報告.pdf)第2–5頁，認識回歸、MAE和資料角色。完整14頁另有公式、程式位置與重現指令。
2. 打開[23頁簡報](deliverables/臺灣房價預測_v22_零基礎簡報.pptx)，第1–20頁主講，第21–23頁作技術附錄。
3. 配合[完整講稿](deliverables/臺灣房價預測_v22_完整講稿.md)逐頁練習，完整口述稿也在PPTX的備忘稿中。
4. 用[排練與16題問答](deliverables/臺灣房價預測_v22_排練與問答.md)檢查容易混淆的概念。

主講4,658純漢字，每分鐘250–300字估算15.53–18.63分鐘，逐頁規劃16分20秒。未經本人計時彩排，不依靠問答、附錄或停頓補足12分鐘。

可編輯正文：[報告Markdown](deliverables/臺灣房價預測_v22_零基礎報告.md)。Public MAE仍為1,998,568.5706元，Private未知；教材改版沒有改CSV或重新訓練。

## 程式與製作來源

- content/narrative.json：23頁敘事與完整口述。
- build/build_deck.mjs、finalize_deck.mjs：使用Codex bundled Node與artifact-tool製作及驗證PowerPoint。
- report/report_content.json、build_report.py：可重建PDF與Markdown，需reportlab、pypdf及微軟正黑體，參考report/README.md。
- build/verify_content.py與qa：講稿字數、notes完整性、來源及實際渲染核對。

製作程式目前依原C:/NYCU_ML工作區與指定Codex bundled runtime，連結source引用原工作區；不宣稱下載GitHub後任何環境都能直接重建。模型推論請從專案README和docs/REPRODUCIBILITY.md進入，另須原始test與17項保存artifact。程式文件包不是完整重訓備份。

封面情境照片：Markus Winkler，Taipei，Unsplash。https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs 。不代表課程中的個別交易，來源也保留於簡報備忘稿。
