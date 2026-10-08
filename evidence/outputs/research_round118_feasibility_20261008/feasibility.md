# R118：四起點的時間外條件融合可行性盤點

2026-10-08。這是日期／索引及既有流程盤點，沒有選新模型或新 cutoff。**四起點的完整 meta 資料目前尚未齊備，但補一個短於一年的完整 forward 區塊在資料與地理來源上有條件可行。**「完整」指六個 native 模型、編碼／prior 及 reference／spatial 都依該區塊的 prefix 建立，不是一定滿十二個月。

原 R98／R99 的真正 forward-OOF 全域融合已做過，不能寫成尚未嘗試。R99 的 meta 目標改善約 1.5–2.6 萬元，但下一期三窗仍兩窗退步；這是條件融合最強的既有反證。增加權重模型容量與補一段歷史資料都不能保證改善，更不代表未知 Private 的勝率。R2 的 holdout early stopping 限制屬其早期流程；原 v22 的 R3 baseline 明載 `fit_uses_eval_set=False`／`iterations_selected_from_confirmation=False`，R79 fit 也沒有 eval set 或 early stopping，不能把 R2 問題移植至這些 native。

## 真正完成的讀取與檢查

先做 source／aggregate metadata 探查，再把精確二進位 readset 寫入 [planned_readset.json](planned_readset.json)，才載入本輪日期／索引。共綁 77 個既有檔案，前後 SHA 全相同；16 個分支／v22 預測檔僅串流 SHA，未反序列化。15 個 standalone 日期／索引陣列依原 file／dtype／shape／values SHA 檢查。純 X pickle 必然暫時物化原 87 欄，隨即只保留 `trade_day_index`、兩個日期旗標；未使用其餘欄位，未開價格 pickle。這項物化範圍不能寫成檔案層只讀三欄。

原 canonical mapping 沿用 455,784→455,255，沒有重做含價格去重；有效日期 455,247、異常 8、超界旗標 5。最早有效日期 2008-06-26，最晚 2024-08-10。原三期／offset 的全部 fit/query 日期遮罩均精確重播；原 R98 三組 meta 索引和日期亦逐位元相同。[機器結果](feasibility.json)；[實際 readset](readset.json)。第一次程式內量測 2.937 秒。

沒有價格／逐筆預測／模型／private prior/reference/spatial state 載入、NPZ 讀取、fit、GPU、MAE、bootstrap、新訓練 arrays、CSV 或 Kaggle 動作。舊 aggregate 報告含已公布歷史分數，不是本輪新計分。PBF 和地理 cache 未反序列化，地理時間與 cache 身分依原 metadata；不能冒稱本輪重解析 PBF 或全量幾何逐列重播。

## 四個 origin 的現有 meta 覆蓋

| Origin | Fit／query 列數 | 現有合法完整 forward meta | Meta 日期 | 缺口 |
|---|---:|---:|---|---|
| auxiliary，2021-08-11 | 134,805／105,511 | 0 | 無 | 缺較早 prefix 訓練的完整六模型＋R/S forward 區塊 |
| development，2022-08-11 | 240,316／81,234 | 105,511 | 2021-08-11～2022-08-10 | 已有 auxiliary 區塊，不能當獨立盲測 |
| offset，2023-02-11 | 272,117／107,169 | 137,312 | 2021-08-11～2023-02-10 | auxiliary 全部＋development 前 31,801 列 |
| confirmation，2023-08-11 | 321,550／133,697 | 186,745 | 2021-08-11～2023-08-10 | auxiliary＋development 全部 |

以上 meta rows 均屬該 outer fit、與其 query 互斥，且日期嚴格早於 origin。權威為 R98 artifact inventory／prepared manifest，原 R79/R71/R46 分支雜湊；offset 為 R97 sealed manifest／R90 pure-X manifest。本輪額外 hash 全部 N/H/U＋v22 保存向量，沒有再算 median 或載入值。

offset query 與 development、confirmation 各重疊 **49,433／57,736** 列。confirmation 前合法的 offset 前段 49,433 列早已在 development meta 裡，因此多拼 offset 不會新增唯一 meta rows，反而可能重複計權。各晚期 fit 包含早期 query，市場與訓練母體相依；已反覆研究這些窗口，補新 seed／bootstrap 都不會還原盲測。

## 較早年度資料量與地理限制

下表只是列出所有非空的 8/11～次年 8/10 日期 band，**不選 cutoff**，也不表示已有完整模型 forward 預測。2008-08-11～2009-08-10 為零列，程式的非空列表略過它；由前後 prefix 相同可核其為空，這不是排除不利窗口。第一列僅部分歷史且 prefix 為零，不能視為可訓練的完整 band。

| 日期 band 起日 | Prefix 列數 | Band 列數 |
|---|---:|---:|
| 2007-08-11（部分） | 0 | 1 |
| 2008-08-11（空） | 1 | 0 |
| 2009-08-11 | 1 | 1 |
| 2010-08-11 | 2 | 243 |
| 2011-08-11 | 245 | 4,509 |
| 2012-08-11 | 4,754 | 23,148 |
| 2013-08-11 | 27,902 | 17,129 |
| 2014-08-11 | 45,031 | 10,202 |
| 2015-08-11 | 55,233 | 5,299 |
| 2016-08-11 | 60,532 | 5,934 |
| 2017-08-11 | 66,466 | 12,591 |
| 2018-08-11 | 79,057 | 16,558 |
| 2019-08-11 | 95,615 | 21,349 |
| 2020-08-11 | 116,964 | 17,841 |

現有地圖 audit 的 PBF replication header 為 2021-01-01T21:42:03Z、2022-01-01T21:21:26Z、2023-01-01T21:21:53Z、2024-01-01T21:21:15Z。其 2026 年 audit／下載建立時間與歷史 snapshot 時點不同；header 也不是道路開通日期。已列目錄與原地理來源 metadata 中未見 2020 或更早快照，這是有界盤點，非全磁碟不存在的鑑識證明。所有上表早期 August origins 都早於 2021 快照，不能直接借它重建當時 v22 地理。

### 主線追加的短 forward band 可行性

Root 明確追加兩個日期候選，只核 prefix/query 至 2021-08-10，不自動選其中一個。[追加事前 readset](supplemental_request.json)；[追加結果](partial_band_feasibility.json)，程式內量測 1.467 秒，13 個既有 inputs 前後 SHA 不變。

| 未選定起日 | Prefix／query 列數 | 台灣午夜對應 UTC origin | 2021 地圖早於 origin？ |
|---|---:|---|---|
| 2021-01-02 | 126,926／7,879 | 2021-01-01 16:00Z | **否**：header 晚 20,523 秒（5:42:03） |
| 2021-01-03（最早合法台灣午夜，最後追加） | 126,957／7,848 | 2021-01-02 16:00Z | 是：origin 晚於 header 65,877 秒 |
| 2021-02-11 | 128,614／6,191 | 2021-02-10 16:00Z | 是 |

不能只看日期字串「1/1 比 1/2 早」：原地理契約的 origin 是台灣當地午夜。2021-01-02 在這個既定慣例下不合格；本盤點沒有替換它的 forecast 時間或自行選另一日。

主線最後另要求從 timestamp 推得最早合法的台灣午夜 2021-01-03，只追加日期覆蓋，未選新實驗。[最後追加 request](earliest_boundary_request.json)／[結果](earliest_boundary_feasibility.json)。它與 2021-02-11 一樣，fi/qi 聯集精確等於原 auxiliary fit；五個既有 inputs 前後 SHA 相同。三個候選共約 6,191～7,879 meta rows，**小樣本可能由少數高價尾部主導，seen/new 案比例、每案有效支持及尾部組成都未在這次只讀日期盤點量測**。形式上能建立完整模型，不等於有強而穩健的驗證；本輪不能讀價格替它挑較好的切點。

兩組 fi/qi 都互斥，且聯集精確等於原 auxiliary fit 134,805 列。原 R6 2021 `train.pkl` 是完整 canonical 455,255 列純幾何；原 R39 2021 `fit.pkl` 覆蓋這 134,805 列，所以可在**另立的新 pure-X adapter**中按 canonical 索引／auxfi 的逆映射切片，不需要拿較晚價格狀態。這裡只核 cache 存在、metadata SHA／row-axis 宣告與 source；尚未對 cache binary 重新 hash 或切片逐位元驗證。

R6 `encode_geo_features` 是逐列固定的 float32／grid／status 轉換，無 fit 統計、y 或 query vocabulary。R39 fallback 的唯一性取完整同年度 city-road index；`classify(pair, rows, records)` 的 rows 只影響統計，座標／唯一性由 records 決定，非後期 query 頻數學得。R39 幾何 cache 與含標籤殘差的 R8 spatial **state** 角色不同：前者可在來源／對齊驗收後重用，後者必須重建。

## 若日後補區塊，哪些需要重建

| 相依 | 現有可用配方／資料 | 新 origin 必做 |
|---|---|---|
| canonical 原 X、mapping、日期解析 | 原 87 欄與保存位置 | 沿原 mapping 切新 fi/qi、完整日期遮罩與互斥 SHA；不再以 raw 價格去重 |
| 固定 2021 R6／R39 純幾何 | 短 band 的日期可滿足來源先於 origin；完整 cache 的 row-axis metadata 已存 | 新 adapter 核 metadata／binary／canonical 切片；保留 tree 與 spatial 地理角色，不能以後期 state 改名 |
| C_raw、L_unit | R3/R6 frozen 2999／3999 配方 | 新 prefix fit；LGB category maps 也重新建立，無 query eval |
| C_log、C_unit | R46／R79 固定 95 欄／23 cats／2999；單價乘 date×area | 新 prefix fit、特徵 state、新 prefix 日期權重；不能切舊較晚 weights |
| N_area | R47 encoder、R54 knots、R48 trainer | 新 prefix numeric stats／vocab／ramps 與 30 epochs；ramp 寬度由新 knots 決定 |
| N_prior | R60 group-OOF/full-fit prior、R63 ratio weighted-MAE trainer | 新 prefix fit-label prior／dateweights／30 epochs；group-OOF prior 不是完整模型 forward 預測 |
| reference／spatial | R2 `fit_final_reference`、R8 `build` | 新 prefix 標籤建立 hedonic／residual／支持狀態；不得借原 auxiliary 或較晚價格 state |
| 三分支完整 v22 | U=.75C_unit+.25L；N=.75(.5N_area+.5N_prior)+.25L；H=.75(.5C_raw+.5C_log)+.25L | 各 base 接同源 R/S，保存 native/base/reference/spatial/median；全 query 重載精確後封存 |

現有 R90/R97 已證明可新 origin 隔離（原 offset 的舊缺口已解決）；R97 多訓 unit seed137 是另一候選，重建 v22 只需原六 native，不是七個。大多舊 driver 硬綁三期／offset 日期、SHA／OUT／控制與拒覆寫，不能改 globals／stage 名直接冒充新 origin；新資料 adapter／driver／規約和所有身份須另檔。現在未建立它們，也沒有乾淨環境端到端重訓證明。

## 每筆 meta row 的條件特徵應屬於哪個 prefix

| 特徵範圍 | 正確來源／可否由純 X 重建 | 本輪限制 |
|---|---|---|
| 交易月份、面積／類型、備註等原欄 | 該 row 原 X；無價格 | 本輪只實讀日期，未建其餘條件特徵 |
| seen/new、案／路名／類別支持次數、純 X 日期加權支持／ESS | **產生該 row 預測的 prefix**；例如 auxiliary meta rows 只能用 <2021-08-11 的 X | 不能用 development／confirmation outer fit 事後把 row 變成 seen；另固定 key／正規化／權重來源才可重建 |
| unknown category、numeric range／missing／標準化、ramp 位置 | 同一 prefix fit 的 R47/LGB/R54 狀態；純 X 可以重新 fit | 不得以「沒有 y」為由沿用較晚統計；可用 source 重建不等於本輪已驗值 |
| 地理有無位置、extent／status／grid | row 本身＋該 forward origin 以前固定 snapshot；不是 house 精確位置 | 新短 band 要依同2021 snapshot／行軸驗收，不能用後期地圖補位 |
| prior 支持層／count | 若定義純 key/date count，可由自己 prefix X 重建；prior 價格值／篩選則依 fit labels | 不可拿 group-fold heldout 的支持當完整 forward model 的 query 支持；不能重用後期 prior state |
| reference／spatial confidence、殘差支援／價格趨勢 | 某些 key count 是純 X；實際 eligibility／fill／hedonic／殘差／confidence 依 prefix labels 與完整算法 | 需重建並保存該 prefix 的合法 query diagnostics；不能拿純 count 冒稱完整 confidence |
| 分支分歧／尺度化不一致 | 僅保存的合法 forward N/H/U 向量；逐 row 不需 query y | 本輪只核向量檔 SHA，沒有載入數值或選公式；新早期 band 必先完成六 native＋R/S |

因此「只讀 outer fit 的 X」不等於條件特徵時間正確：同一 outer meta 矩陣可能來自兩個不同的模型 prefix，各 row 必須帶明確的 prediction-producing origin／fi 身份。當前 R98 meta source-block／source-order／位置 SHA 能追溯這個角色；尚未保存所提全部 prefix-relative gate features，需另立固定規約和純 X 特徵契約。

## 成本與有界建議

只引用舊 R89 相依盤點中的已量測範圍：C_raw baseline elapsed 109.984–153.221 秒、LGB 155.375–504.623 秒（均含原 driver 的其他工作）；C_log helper 109.547–135.156 秒；C_unit 外層 151.250–190.125 秒；N_area／N_prior 純 fit 10.344–24.093／10.218–26.968 秒。原 R60 prepare 5.063–11.453 秒、R54 ramp prepare 2.812–5.297 秒，scope 各異。這些不能相加冒充新 band ETA；新 adapter、encoder、cache 驗收、R/S、全部 reload、診斷與條件權重學習成本尚未量測。6,191 筆短 band 的統計容量／市場代表性也不能由列數保證。

**僅一項建議：可再立一個固定的完整 forward-meta 補齊規約，再決定是否有資格研究低容量條件融合。** 2021-02-11 的短 band 在日期／snapshot／cache 行軸來源上可行，這是可行性證據，尚未選它作訓練 cutoff。規約至少要先鎖一個 origin、原六模型＋R/S、完整來源／時間角色、meta 特徵、容量與對照／採用門檻；獨立 preflight 後才可建立新 fit-label inputs 與訓練。若不補最早 auxiliary 的完整 forward 區塊，就不具備原四起點都同等完整的條件融合驗證，不能只報較晚三窗。

作者另完成 30 項人工 CPU 日期邊界、拒讀價格／預測／private state、source 無 fit/predict/load-model、R39 row-frequency 不改幾何判定的檢查。這是作者 self-check，不冒稱獨立審查。封包工具第一次因 JSON／MD 共用 stem 的 `.sha256` 命名撞名而拒覆寫；保留失敗來源／收據，後續 finalize 使用 `feasibility.md.sha256`，沒有改舊結果或重跑日期分析。

v22 仍為使用者選定版本；v25 仍是探索候選，原簡報／旁白保持，Private 未知。這份盤點不選新提交、未預留 CSV 版本、未更改任何既有 source／plan／receipt；共享 HANDOFF 由主線更新。
