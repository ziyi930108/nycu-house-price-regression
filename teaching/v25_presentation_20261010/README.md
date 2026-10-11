# v25 上台簡報與實際訓練曲線

[完整PPTX](deliverables/臺灣房價預測_v25_含實際訓練曲線_r2.pptx) · [完整口語講稿](deliverables/完整口語講稿.md) · [單頁曲線PPTX](../v25_learning_curve_slide_20261010/deliverables/實際訓練與驗證曲線_v25.pptx) · [曲線白話報告](../../evidence/reports/v25_training_curve_20261010.md)

24頁包含21頁主講與3頁附錄，第19頁是自己的實測曲線。正文4,217漢字，275字／分估15.33分鐘；尚未真人計時，附錄／問答不補主講時長。完整流程、每步理由、失敗與過擬合限制均保留。

這次固定v25先驗NN配方另跑一次30輪診斷，先封存全部快照再讀時間驗證價格。兩線使用相同快照、關閉dropout、不加權總價MAE，圖的橫軸為固定架構的epoch。這不是原v25訓練軌跡、六模型融合曲線或Private表現，也沒有依曲線重選epoch或修改原CSV。

圖表與表格是可編輯的PowerPoint原生物件，曲線workbook保存60個聚合MAE點，僅顯示四捨五入到1元，原始精度在[JSON](../../evidence/outputs/v25_training_curve_20261010/curve.json)。[渲染圖](build/A/renders_r2/slide-19.png)與[審查範圍](delivery_review.json)供閱讀；未在原生PowerPoint內開啟驗證。

[投影片builder](source/build_deck.mjs)、[finalizer](source/finalize_deck.mjs)、[內容JSON](source/narrative.json)保存原工作區來源。執行需要原C:/NYCU_ML完整工作區與指定bundled runtime；不是已驗證的GitHub-only可攜重建入口，也不會從這些聚合圖表重訓完整房價模型。原始資料、30個模型快照、逐筆價格與標籤不納GitHub。

舊版簡報、PDF與影片已移出主分支，歷史留存於 Git 與原工作區。新v25簡報已更新模型、loss、公開分數與程式界線，尚未製作新的配音影片。
