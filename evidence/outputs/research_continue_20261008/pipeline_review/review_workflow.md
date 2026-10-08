# R111 起接續研究：獨立流程審查與待驗事項

2026-10-08。這是讀取既有 source、plan、已完成審查證據後的流程建議，**不是 R111／R112 的 preflight 通過證明**，也未進行新 fit、GPU 推論或載入未評分 query 真價。新候選固定為主執行緒指定的 R111 ResNet prior/date-weighted MAE，以及 R112 原 R79 單價/date×area 權重的 Ordered boosting。

## 最小可重用流程

1. 新 source、新 plan、新輸出目錄。原 R107–110 source、原始模型／資料／CSV／失敗證據保留。新 source import 原流程時使用原 immutable 版本；避免改共享 CONFIG 或舊 module globals。source closure 採 `round100_contract.source_hashes`，另顯式綁定 plan、動態載入的 constructor／builder、正式 self-check receipts 與獨立 preflight。AST local-import closure 不能當成完整套件／環境鎖定。
2. 沿相同三個年度 canonical windows，逐窗綁定 fit/query 索引、fit-only labels、九個 R63 NPZ keys、OOF/full-fit prior、原 R79 日期權重值 SHA、tree geography、encoder/reference/spatial 狀態與固定 v22／v24 預測來源。新 cutoffs 必須重建全部 fit state，不能拿後期狀態回測。raw label 檔可先 hash，封存前不得 deserialize query y。
3. preflight 完成且 source 停止變動後，主執行緒先正式執行 CPU self-check，再寫不可覆寫 request。單張 GPU 由主執行緒依序 fit；兩候選獨立，不混入 R107 SmoothL1、R108 CTR complexity 或 R110 target borders。全部六個模型及 query 預測封存完成前不評分。
4. 每窗必須封存 actual native checkpoint、metadata、必要 encoder/feature_state、native 預測與 base/reference/spatial/median 五層。根 seal 前重新核全部 child artifact；evaluate 讀 y 前再核 source、inputs、protected、stage、metadata、model 及 layers。`path/sha256` file record 與含 dtype/shape/values_sha256 的 array record 分別驗證，不以兩種 dict 直接相等替代。
5. saved-model 必須全部 query rows 重載預測 exact；CatBoost 只允許事前聲明的有限浮點 native serialization tolerance，不能以此放寬價格預測。評分只在全部模型封存後一次進行。execution receipts 必須保存真實子程序 exit code、時間與失敗，不把 wrapper exit 0 當成內部成功。
6. 獨立 postfit 審查在正式 evaluate exit 0 後執行，核完整鏈、實際 native 配置、分群 arithmetic、manual gates 與 v24 反證。六次 fit 都結束才讀 y；CPU 審查失敗若是審查工具 schema 錯誤，另存修復版與新 attempt，不能改寫原證據或算成模型 fit failure。
7. 未過事前 exploratory gate 不全量 fit／出 CSV。通過後另立 fullfit plan、preflight、source freeze，再作一個全量 fit 與 CSV release review。研究候選不自動替換使用者選定 v22，也不自動上傳／勾選 Kaggle。Private 始終未知。

## 固定對照與門檻

- 主對照：保存原 v22 完整組合，先重播原 NN／unit 與 reference/spatial/median 鏈，MAE 對回原紀錄。
- 次對照：事前固定原 R83/v24，同時保存點估計與全部分群反證；不得只報相對 v22 的改善。
- 強推薦：三窗整體與固定新舊加權 MAE 均嚴格改善；最新整體至少改善 0.5%；季度或至少 500 列群無大於 5% 退步。
- 探索提交：最新整體與加權皆不退步；六個窗口指標各退步不超過 0.25%；至少兩窗整體改善；無上述分群退步。
- 分群保留 strata、forecast quarters、months、cities、relative forecast months、price bands，不省略價格帶或最差城市。門檻判斷用未四捨五入 float；strict improvement 與 no-worse 分開。
- 三窗口反覆使用，不能當新盲測。建案／月份配對 bootstrap 是保存向量下的條件不確定性，不能稱 Private 勝率或累積選型校正。

## 已知封存與 schema 問題

- R107 stage 原先沒有直接綁 checkpoint；舊 postfit 只補記審查時 checkpoint SHA。此限制保留，不回寫舊封存稱已事前綁定。
- R109 stage 已直接綁 mean/epoch30 兩 checkpoint；新批應沿用這一點。R109 未保留十個個別 epoch states，postfit 無法獨立重新相加；其訓練演算法自查與終態身分驗證有不同證明範圍。
- 舊 R109/110 evaluate 沒有直接驗 `model_metadata` record，新批需補；根 seal 建立前也應遍歷 child artifacts，避免只收 stage.json SHA。
- raw categorical schema 是 21 欄；unit model 經 geography augmentation 為 23 欄。feature_state 的原 X 檢查必須傳 21 欄原 schema，actual model 才驗 23 欄，避免上一輪 auditor 的 schema 誤用。
- R79 feature_state 內 area-only／原參數欄是 historical provenance，不能拿它宣稱當次 Ordered fit 已驗收；當次 metadata、Pool weight identity 與保存 CBM native parameters 才是實際配置證據。
- protected set 包含動態所有 `submission_v*.csv`；新批 freeze 需等前批 release 結束，避免合法新增 CSV 被判 frozen protected set 變動。
- 舊 publication packager 綁舊 2,788-member manifest；新一次 GitHub packaging 需以當前 manifest 另立規約與審查，不能直接重用舊 publication base。文件快照不等於乾淨環境端到端重訓備份。

## R111／R112 專屬待驗

R111：逐項列出 ResNet blocks、width、normalization、activation、embedding/category dropout、residual branch dropout、head，以及與 R63 的 linked architecture changes。固定 30 epochs、seed、optimizer、schedule、九 NPZ／F／date weights；loss 保持 `mean(w * abs(ratio * F - y_million))`，不改成普通 ratio MAE。checkpoint 要保存 constructor/config/numeric count/cardinalities／state_dict；CPU synthetic forward、weighted-loss 與保存重載 contract checks，不啟動真 fit。若使用 BatchNorm，必須明確驗保存 running buffers 並保持 query inference eval mode；不能套用舊 LayerNorm 無 buffers 假設。

R112：只將原 R79 `boosting_type` 改 Ordered，其餘 depth8／2999／seed42／lr0.065／MAE／max_ctr_complexity2 及 features/unit/date×area weights 保持。當次 CBM native boosting_type、tree count、target/inverse、原始／augmented feature order 和 date weight 值 SHA 都驗；不得只核 request 或 feature_state provenance。CPU parser/self-check 能證明合約，不能冒稱已在真實價格 fit 成功；若真 fit unsupported，保留失敗，不私自改 loss／Plain／sampling。

## NN loss 與 price_factor 核對

原 R63 正 ratio 乘 `F=effective_area * fit-label unit prior / 1e6`，訓練在 million TWD 以日期加權 MAE；推論由 float32 ratio 轉 float64 後乘保存 F32 與 1e6。F 的 fit 建構是 group-fold OOF，query 是 outer full-fit prior，並非逐日 chronological。代數 `w|Fr-y| = wF|r-y/F|`（F>0）說明普通 ratio loss 會變更目標；此代數不能聲稱兩種 float32 計算順序逐位相同。

R64 已試 log-ratio/log-space MAE，R109 已試尾十 epoch 參數平均，均未過探索門檻；新 ResNet 可以只改 backbone 並保留真正總價目標，避免重複上述失敗。原 bias 仍取普通 ratio 中位數；wF-weighted median 是常數 ratio 的 weighted-total-MAE 最適初始化，但目前只屬未執行的機制假設，不納入已固定 R111／R112、不進行 grid 或追加 fit。

本輪停止界線依直接使用者最新要求及 `../user_request.json` 為剩餘 20%；取代舊 plan 的 1% 文字。不使用額外點數、付費 API 或自行 reset。
