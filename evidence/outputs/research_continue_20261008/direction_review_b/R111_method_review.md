# R111 固定殘差先驗網路：作者實作與 CPU 自查

2026-10-08。此檔是新 trainer 的作者說明，並非獨立來源審查、房價訓練結果或 Private 證據。正式研究規約以 `reports/round111_112_plan.md` 為準。

## 已完成的程式

`src/round111_residual_prior_train.py`，SHA256 `318aae73904ceb8c02fdf42107bd80db4388d35717cdd123e5085f8ec298f0db`。

- `check_configuration()`（第74行）核對原 R47／R63 source SHA；再呼叫 R63 的完整設定檢查，包含原 R52 SHA。保存完整原設定、實際設定、固定架構覆寫與逐欄差異，沒有修改舊模組 globals。
- `PreNormResidualBlock`（第87行）為 `x + Dropout(.1)(Linear256(Dropout(.1)(ReLU(Linear512(LayerNorm256(x))))))`。
- `TabularResidualPrior`（第105行）沿用23欄 embedding 維度公式與獨立0.1 category dropout；144 numeric 與 embeddings 串接後投影256，經兩個固定 block，最後 LayerNorm→ReLU→Linear1→Softplus。
- `validate_arrays()`（第135行）只接受原九個陣列，固定144 numeric／23 categories、float32 numeric/target/factor/weight及int64 category/cardinality；檢查列序對齊、有限、正價格／因子／權重、類別範圍。query y 不在可接受鍵集合。
- `checkpoint_payload()`／`restore_checkpoint()`（第178／186行）保存實際類別、完整設定與 state_dict；重載前核對鍵、型別、形狀、有限值及原來源身分，避免 `load_state_dict` 靜默將錯誤dtype轉型。
- `run()`（第206行）使用原R63的日期加權總價MAE、正值倍率、fit median(y/F) bias、30 epochs、batch2048、seed42、AdamW lr.001／wd.0001與cosine最低1e-5。沒有參數平均、SmoothL1、早停或按query選epoch。

儲存 `run.json`、`model.pt`、`training.json`、`predictions_ratio.npy`、`predictions_twd.npy`。正式執行會驗證全query ratio及TWD重載bytes exact，兩個NPY保存後identity exact，並重核 source/input SHA；此次尚未執行這個真資料路徑。既有同名產物拒絕覆寫。除倍率×保存價格因子×1e6外，不額外取整或裁價。

## 已做與未做的驗證

最新記錄在同目錄 `R111_author_selfcheck_v2.json`（附 `.sha256`）；先前 `R111_author_selfcheck.json` 保留。記憶體compile與 `--self-check` 40項均通過，最新量測自查2.969秒。只用synthetic CPU tensors，0房價資料讀取、0 model fit、0 optimizer step、0 GPU呼叫、0新CSV。CPU generator直接設seed，沒有呼叫CUDA setup。

自查包括實際shape／正值／有限值、embedding維度、兩個固定block、無BatchNorm狀態、同批其他query列改變不影響本列、列重排等變、零殘差分支identity、常數殘差相加、category mask公式及eval不dropout、加權MAE精確值與梯度、unknown embedding梯度、全部參數有限梯度、倍率到TWD只乘一次、九陣列拒絕錯誤、記憶體checkpoint ratio/TWD精確重載、錯設定及dtype拒絕、舊global設定未變。列重排只用CPU算術容差；保存／重載向量檢查為bytes exact，兩者不能混稱。

首輪作者自查曾因使用 inference-mode 產生的 index 參與 autograd indexing 而失敗；只修正自查為 no_grad 後重跑通過。不是模型房價fit失敗，也沒有按結果修改架構。

最後作者檢查將 NPZ exact 九鍵集合檢查前移到讀取任何 NPZ values 前；因此即使 query_y 取代一個原鍵而保持九鍵，亦會在讀值前拒絕。原本只先檢查九鍵數量、再於 validate 拒絕錯鍵，存在過晚拒絕的邊界問題；此修正不改模型或訓練配方。

尚未進行真fit、GPU容量／耗時量測、三年度MAE評分、全真query模型重載、新CSV或Kaggle操作。獨立 preflight 與主線調度仍為正式fit前必要條件。

## 理由、反證與適用限制

假設是殘差路徑有助固定訓練條件下的訊號傳遞，並保留本題倍率與時間加權目標。這是表格ResNet的本題適配，LayerNorm／embedding／倍率頭均依本專案；不是官方架構完整復刻。寬度、容量、參數初始化與亂數消耗一起改變，不能把可能改善純歸因skip，也不能聲稱相同seed意味相同初值或跨GPU逐位元重訓。

本次唯讀搜尋未發現R1–110既有hidden residual-block trainer。R57／R58的32成員共享參數網路不是同一結構，但它們三年度皆未勝原對照，顯示較複雜網路不保證改善；R95 attention亦為不同架構。原R47的0.1 category dropout早已訓練unknown embedding，本次沒有「修復未訓練unknown」的主張。R59已試prior numeric ramps，亦不應包裝成尚未探索的改良。

fit-only encoder／五折建案OOF prior／日期權重的來源與切點由主線 frozen request 驗證；trainer只檢查數值契約，不能自行證明 provenance。各年度只替換prior NN，原area NN／LGB／log／unit／reference／spatial與五層重播保持；不得借用較晚切點狀態。使用保存v22為主對照並保留v24強反證，六次候選fit封存後統一評分；反覆使用年度的選型偏差仍存在，bootstrap不是Private勝率。
