# R117 工程恢復 v2：補齊動態父來源，原科學規約不改

2026-10-08。原正式freeze exit0、9.891秒；原train session34164 exit1、84.125秒。development新transaction prior與30epoch NN已完成，但driver在check_metadata以request來源清單取src/round52_nn_train.py時KeyError。該父來源由trainer動態使用，原AST靜態closure未涵蓋。流程未完成stage seal／五層，未跑其餘三origin，未讀新query價格／評分／CSV。原CPU65case用人工完整lineage，沒有驗證實際sources()必含所有動態父檔，是測試缺口；前審通過不代表真流程已完成。

原reports/round117_plan.md、三source、request、failed execution、model、prepared、log與全部雜湊原樣保留。另檔src/round117_research_v2.py／outputs/round117_v2／outputs/research_round117_v2_20261008；不解除舊拒覆寫、修改原request或以手填SHA冒充舊closure已完整。v2明確把src/round52_nn_train.py及其相依納完整closure，凍結前驗證trainer實際動態父lineage皆在映射內，missing binding必須CPU拒絕。

這是明確工程恢復，不是原launcher自動retry。原R117唯一科學改動、四cutoff、fold seed42／原R60估計器、八array不變、ratio NN／初始bias公式／epochs／loss／組合／gates全部不改。為讓新request／metadata／seal簡單且一致，v2新目錄**全部四prior＋四NN重新執行**，不copy原部分模型／prior冒稱新fit。原1個已訓練但未封存native列為aborted attempt，成功批次仍最多四新native；需如實另列總計5次actual NN fit，不能隱藏重跑成本。

v2須重新正式CPU checks、完整source／input exact獨立preflight、原四context CPU回放綁定，再由主線freeze／串行GPU。獨立代理不得審自己的新source就稱獨立；主線／另一代理須審最小diff與新實際closure guards。先核目前無live R117程序。四prior/model／五層全部seal且trainexit0，才另evaluate讀query價格，之後獨立postfit。不沿用原preflight SHA假裝v2已審；可引用原context證據但必須明示來源未獨立新跑且所有原context相依一致。

失敗不是模型效果差，不能產MAE、採用判定或v32。若再次有工程問題，保留失敗另診斷，不自動改科學設定／重啟。v22、v25、上台資料與Kaggle不變；Private未知；額度剩餘≤20%保存並停止，不用重置或點數。
