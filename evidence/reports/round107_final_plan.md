# R107全量探索提交規約（v25）

2026-10-08。R107三個重用年度窗口全部評分後，按訓練前已固定的探索條件通過。最新窗口整體MAE改善0.1103%，早期窗口退步883.66元；未達強推薦門檻。按建案抽樣的三個整體區間都跨0，且最新與早期仍比v24差。這次全量fit只為使用者授權的新試交，不能稱Private最佳或已取代v22。

全量只新增**一次**R107 prior NN fit：使用原R63 final九鍵prepared.npz，467221 fit／47073 test，固定weighted SmoothL1 beta=.1百萬元、30epoch、seed42及原其餘參數，不重訓其他分支、encoder或prior。不做新的loss、比例、epoch或seed選擇，不與R108合併。

先用原final fit資料、保存原生預測、同一reference與spatial重播v22完整鏈；逐bytes核對未取整向量和已選CSV的整數總價。這是保存向量組合重播，不冒稱重新推論原六模型。新R107原生輸出先還原總價，替換prior NN：`.75*(.5*saved_area+.5*new_prior)+.25*saved_LGB`，僅做一次原R及S，與原完整log/unit分支取中位數。所有來源、計畫、資料和既有受保護產物在fit前凍結，完成後再次核對；保存重載需全query逐bytes相同。

新輸出`outputs/submission_v25_trial.csv`，拒絕覆寫。ID取原sample的47073列與順序；只有「編號,總價元」兩欄。未取整向量一次`maximum(rint(p),1).astype(int64)`。以Artifact Tool的typed range寫入、核對全部列後以標準CSV編碼匯出UTF-8 BOM；獨立Python csv reader再核對每一個ID/整數值、重複、列數、欄位、範圍與SHA。預測向量差不是MAE，不用test未知真價評分，不依test ID改價。

fit與發行各另存request/seal/release、來源閉包、模型、預測、CSV、實測時間及失敗。Public／Private保持未知，不動Kaggle、不改v22原選定、不改原簡報。完整source保留；已保存可追溯不等於乾淨環境端到端重訓完成。額度剩餘≤1%停止新增工作，不自行重置或使用額外點數。
