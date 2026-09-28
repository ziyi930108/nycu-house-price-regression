# R96 固定 reference 折半：獨立 preflight

2026-09-29（Asia/Taipei），審查者 `/root/r85_direction_audit`。**最終 source／API／實際 metadata 契約核對通過，沒有剩餘 preflight blocker，可由主線凍結並啟動。** 本文件先於 R96 真資料評估落地；不表示已執行三期、已改善或應採用。

範圍為只讀 source、小型 JSON、artifact 路徑與雜湊；沒有載入真 X／y／query 價格／test／模型／預測陣列，沒有真 predict、score、fit 或 GPU 呼叫。沒有重跑作者的合成測試，也沒有改別人的 source。機器與只讀檢查分列於 [checks JSON](r96_validation_audit.json)，SHA `048793450a7b451c3076beaa52e6f700fb4ba04359f4ab223a4b442c5fa7de78`。

## 最終版本與合成身分

| 檔案 | SHA-256 |
|---|---|
| [固定計畫](round96_plan.md) | `7178be401e3927f4ec26164159d2b515aef97e3a2f0158c3e19ee50d1c08a071` |
| [helper](../src/round96_reference_shrink.py) | `65fd4cddf98c686fe53054baefe8ab8ec1bf00822e25589a0ad513669e41d894` |
| [driver](../src/round96_experiment.py) | `5ec0a6682bf3153fe56d274d96f60d4917c5410acf85857147e53e0196d948f2` |
| [完成後 review](../src/round96_review.py) | `2434a0504b8953a7b10003cd560a3631dd3ed1978689c9a038ffc65547189b79` |
| [39 項合成收據](round96_helper_selfcheck.json) | `8824ec6b9829228357c62dc8b936451c1515d850d198e90637531bece4455701` |

三支新 source 的 AST／compile 通過。合成收據的 source、六檔依賴閉包、plan、literal PARAMETERS 完整相符；status passed、exit0、39 checks，作者實測 **0.202944 秒**。人工資料下實際 R2 predict 1 次、R8 build 1 次、R8 predict 6 次；0 model fit／GPU／真資料讀取。不能把這份測試寫成「連人工 R/S 都沒有執行」。

作者保留三份 attempt：首次要求 source dtype 必須 object，錯拒 Pandas StringDtype；改驗非缺且合法 source 值並保留原 dtype，第二次 37 項通過；補兩項 `c>0、P==Bfuture` 的空間耦合測例後第三次 39 項通過。三份保存 SHA 皆核實。模型規則、rho、真資料分數沒有參與這個修正。

## 數學、浮點與 R/S 介面

公開 `shrink_reference(reference_model, Xquery, base_TWD, original_reference_TWD, original_reference_diagnostics)` 不接受 y 或可調 rho，回傳新 float64 P、deep-copy diagnostics、JSON report；`future_base` 公開供 driver 保存證據。沒有模型重訓、state 建置或原常數 monkeypatch。

- `Bfuture` 按原 R2 逐操作順序計算：原 B × exp(beta[3]×capped days/365.25)。NaN→0 天、+inf→366 天、−inf→0 天。使用原 B，**不使用 max(B,1)**；max(B,1) 是原 local_price 內部規則，原 P 已套 max(...,0)。缺日期原 reference 仍可能有 c>0，不額外抹掉它；R8 自己的有限日期條件保留。
- 唯一數值定義為 `Pnew[c>0]=Bfuture+.5*(P−Bfuture)`，`cnew=.5*c`；c=0 直接 copy 原 P bytes，並核原 P 與 Bfuture 相等。沒有再裁價。和重新代入 local_price 的混合式可作數學解釋，**不宣稱浮點逐位相同**。
- diagnostics 的 source／nearest_distance／dispersion／index／dtype 原樣保留，只換 confidence。來源集合、c 範圍、shape/dtype、非負有限價格、原 reference 格式都驗證；不無聲重排 index。
- driver 將新 P 和新 c 一起傳原 R8，raw base、state、geo 不改。空間的有效剩餘權重由 `(1−c)*alpha` 變成 `(1−.5c)*alpha`；G、beta、anchor、eligibility、鄰居與 alpha 不變。這是耦合修正，不能宣稱只改 reference 而空間強度完全不變。
- c=0 且原 S 有套用的列也要求完整輸出 bitwise 原樣；c>0 但 P 恰等 Bfuture 時仍傳新 c，不能用價格是否改變當政策 mask。合成明確覆蓋兩者及 invalid date、零／小於一 base、未命中空間、rho=1 內部控制、排列、reload、輸入不變。

## 本次抓到並已修的真 metadata 契約

草稿一度對所有歷史 plan 讀 `splits` 與 `input_sha256`。**R71 plan 沒有這兩欄**；本審查的兩個 metadata-only 程式依序實際遇到 `KeyError: 'splits'` 和 `KeyError: 'input_sha256'`，已直接回報 driver 作者與主線。沒有執行 R96 主程式或候選評分，沒有據分數修程式。

最終修正保留所有 `split_contract` 比對；R46／R48／R79 另核 `splits`，R71 日期依實際 `results.stages[*].dates` 逐期對照。一般 input 讀前三份的 `input_sha256`，R71 讀實際 `inherited_data_sha256`，再做衝突與 SHA 檢查。沒有增添假的舊 JSON 欄位或旁檔。

修正後本審查實際核對：

| 檢查 | 已驗內容 |
|---|---|
| **15 份 locked JSON** | 5 份 results＋4 份 plans＋三期各 metadata／state_summary；檔案與實際 `.sha256` 旁檔存在，內容 SHA 全匹配 |
| **75 個 artifact 路徑／權威鍵** | 三期所有原索引及三分支 base/reference/spatial／兩組 diagnostics；檔案存在且在各 results 的 artifact map，未反序列化 arrays |
| **51 個歷史 source** | 合併 source map 無衝突，每份原 source 實際 SHA 匹配 |
| **307 個合併 inherited input 鍵** | 正確讀兩種 plan schema，重複键 digest 無衝突；此 preflight 未以讀大資料檔替代 driver 啟動前的全量 SHA 核驗 |
| 三期日期／列數／切分 | 五份歷史 results 的 dates、fit/query rows 與固定 R79 contract 相同 |

這次只使用確實有旁檔的 15 份 JSON，沒有重犯 R94 對無旁檔 encoder 使用 locked reader 的錯誤。上列是本次獨立核验計數，不能和其他輪次的側檔數混用。

## Driver、保存與完成後 review

driver 先驗凍結 request／helper receipt，再核歷史來源與 artifact SHA，才載同窗資料。三分支來源固定為 NN=R71 seed42、log=R46、unit=R79；原 base→reference→spatial 全向量及兩組 diagnostics 逐位重播，原 v22 median 與 v22/v24/v11/v12/v5 完整 metrics 校對後，才呼叫候選 helper。不混 R95 attention、不平均已修正分支。

候選三支分開傳原 R8；c=0 分支／共同 c=0 的最終 median 均核原價格不變。未受修改的 R8 diagnostics 欄位全等，新 c 與有效剩餘權重有檢查；保存每支 Bfuture、reference、spatial、diagnostics 與完整 median。所有新 npy 即時全向量 roundtrip，候選 diagnostics pickle roundtrip；fit state memory digest、X／geo frame 值檢前後不變。以上是已讀的實作保護，**真資料通過與否要看稍後 execution**，不能在此先寫完成。

driver 三期總計預期原 reference predict 9 次、原／新 spatial predict 18 次、shrink 9 次；載入三份 reference state 與三份 spatial state，**0 native model load/predict、0 fit、0新價格state build**。shared loader 會反序列化 canonical y；只有 scorer 使用 query y，不能把整個 main 稱完全未讀標籤。

完成後 review 的 `status='completed'`、`branch_metrics[branch][layer]`、`original_branch_metrics`、`new_fits/new_NN_fits` 與 protected before/after 介面和 driver 相符；generic／paired 使用標準完整 metrics、predictions.npy、索引及 artifact map。review 等三期及 generic／兩 paired 完成，保留 v22/v24 共 24 區間；門檻為三期 overall／weighted 全改善且最新≥0.5%，不自动選型／CSV。

固定rho、單期／三期標籤都不進 helper。方法是已看歷史誤差後提出的適應性研究，窗口重用、共同偏差和粗地理空間放大風險仍在；bootstrap 不是 Private 勝率。通過門檻才考慮新切點，完整 v22/v24/R96 需 7 native（5 樹＋2 原 NN），R96 自身不增加 fit。此處不啟動新窗口、不改 v22 唯一提交選定。
