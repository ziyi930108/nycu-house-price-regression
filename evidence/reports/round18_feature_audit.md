# R18 特徵解析與資料品質：有界純 X 稽核

本次未找到有廣泛實際支持、又尚未處理的主要房屋原始欄位缺口。確認仍存在少量樓層語法錯值；較常見的問題是面積差與車位零面積的解讀條件，而非沒有提供面積／車位資訊。以下只保留兩項未實作的品質假設，**不將它們加入本輪模型，不據此宣稱 MAE 或未來泛化會改善**。R18 的四個識別欄消融由另一份方法審查與事前規約獨立處理。

## 實際接觸範圍

只載入 `data_cache/train_features.pkl`、`data_cache/test_features.pkl`，分別為 **455,784／47,073 列、87 欄、21 類別**，並斷言沒有編號、總價、y 或 target 欄。訓練 X 已去掉 3 列內嵌表頭，尚未套用原流程的 529 筆含目標去重；本次不重建去重，也不把這個列數當成年度 fit 列數。兩份快取 SHA 在單次盤點前後一致。

新統計只依這兩份 X；沒有載入 train.pkl、valid.pkl、valid 特徵快取、原始 Excel、模型、預測或價格／MAE 檔案。既有 feature schema 與方法文件僅作來源和歷史對照，包含其中的舊欄位摘要，不重新接觸 valid 資料。沒有外部查詢、內政部資料或任何成交答案。test X 用於研究可用性，因此不能稱這個特徵研究完全沒看過測試分布；本對話已知歷次模型摘要，但沒有用分數搜尋本次文字規則。

本次是一次限定的快取讀取與聚合，加上 7 個純合成樓層字串；沒有新增稽核 framework、共享 source、特徵快取或模型。

## 1. 可證實的樓層錯值很少，不能擴大為普遍資訊遺失

現行 [house_features.py](../src/house_features.py) 只在「數字—數字」間把連字號改成範圍符號；若前端已有「層／樓」，第二個數字前的連字號仍被當成負號。[R16 數值審查](round16_numeric_review.md) 已指出此缺陷，本次確認它尚未修入凍結的原 87 欄。

| 固定檢查 | train | test | 可支持的解讀 |
|---|---:|---:|---|
| 數字＋層／樓＋連字號＋數字 | 0 | 5 | test 全為 `1層-2層`，目前 min=-2、max=1、mean=-0.5、basement=1；明示地上範圍應為 1、2、1.5、0。 |
| 以「地下／B」起始、後續省略此前綴的完整數字列舉 | 0 | 0 | 限定文法未命中；只有合成例顯示潛在錯誤，不能當成現有資料大問題。 |
| 以「地下／B」起始、後端省略此前綴的完整範圍 | 0 | 0 | 同上；不是證明所有可能地下層字串都正確。 |
| 原樓層含「屋頂／突出／頂樓」 | 0 | 0 | 屋頂層數混入一般樓層的合成風險，在兩集沒有實際支持。 |
| 注音「ㄧ」或全形數字 | 1 | 0 | 唯一為 `ㄧ至四層`，目前只抽到 4；不得據此無限制改寫字元。 |
| 原樓層文字非缺失、floor_mean 缺失 | 30,522 | 2,979 | 其中 30,323／2,976 為全／全部／全棟等整棟描述，其餘多為見登記事項／執照、車位或未指定數字的地下層。 |
| floor_count > 1 | 263 | 41 | 多樓層有既有特徵；不是整類遺漏。 |
| relative_floor > 1 | 27 | 6 | 僅為一致性警訊，不能由此裁定移轉樓層或總樓層何者錯。 |

新增合成核對中，`地下一、二層` 目前成為 [-1, 2]；`地下1至3層` 成為跨地下至地上範圍；`二十層、屋頂突出物一層` 把屋頂層數 1 混入一般樓層。這些是 parser 的文法邊界，**不是觀察到的真實列**。作為對照，`地下一層至地下三層` 正確解析成 -3 至 -1，`一層、地下一層` 的 mean=0 是合理平均，不代表存在第零層。R3 已修正跨地下／地上範圍的 count 不含零層；R16 另記錄範圍 mean 仍只用兩端的限制，本次沒有擴大修正。

完整 `transfer_floor_raw`、全棟／地下旗標、total_floors、unit_floor、相對樓層原本就有。不能把整棟文字沒有唯一數值樓層，或棟戶沒有樓層標記，一律當成漏解析。

## 2. 面積差與車位資訊需要語意條件，不能冒充已知淨室內面積

現行 `indoor_area = building_area - parking_area` 是差值，沒有扣除其他共有面積的資料或權威口徑，名稱不能作為淨室內面積已被驗證的證據。以下交叉計數補足 R16 的解讀，沒有估補或裁切任何面積。

| 固定 X 條件 | train | test | 類型分解／邊界 |
|---|---:|---:|---|
| has_parking=1 且 parking_area=0 | 9,735 | 903 | 約占各集 2.14%／1.92%；房地加車位為 9,702／903，其他 train 33 列。零面積可能未分列，不能自動解讀成沒有停車面積。 |
| has_parking=1 且 parking_count=0 | 50 | 0 | train 中 48 列房地、2 列房地加車位；可能是來源口徑不一致，不自行把 count 改成 1。 |
| building_area>0 且 indoor_area<=0 | 315 | 7 | **300／7 是單獨車位交易**；另 15／0 是房地加車位。不能把全部列當成住宅面積錯誤。 |
| building_area>0 且 building_count=0 | 916 | 44 | train 為房地加車位 635、車位 146、房地 135；test 為房地加車位 36、房地 8。不擅自指定正確棟數。 |
| 有車位但 parking_type 缺失 | 230 | 0 | has_parking 也由面積／筆數決定，類型缺失不等於停車訊號整列遺失。 |

三種原始面積、各自零值／負值旗標、transaction_type、parking_type、parking_count、has_parking、parking_fraction 等已有輸入；這些條件大多可由現有樹模型組合。加品質狀態至多是明示交互條件，不能宣稱新收集到有效面積。

既有單價模型用正建物面積，否則正車位面積，否則 1 作有效面積，並有對應 MAE 權重及逆轉換；**不以 indoor_area 作分母**。這點沿 [既有數值審查](round16_numeric_review.md) 與 [歷史方法審查](round16_prior_unit_methods.md)，本次不讀價格或重新評估。不能把單價尺度或面積權重重新命名為未試過的方法。

## 其餘欄位：已有處理與未確認的邊界

- **棟／戶別：**完整 building_and_unit 原本就是類別。R16 已把固定的明示 code-floor／棟幢字串拆成 unit_code_prefix；補充盤點的 train 33,789／test 4,696 是缺 block 列獲得字串碼，不是新的樓層真值或物理戶型鍵。本次不重跑、修改或重提同一前綴方法。[支援盤點](round16_prefix_support.md)、[固定規約](round16_plan.md) 已說明連字號 suffix 不解讀、不同來源相同碼不保證同種實體。
- **地址：**address_district 與 supplied district 不相同有 16,807／912 列，這只是原字串不等，不能全部算解析錯誤。R3 已有只信 supplied 前綴證據的修正器，R4 已試完整區名／第二道路／道路對／案名前綴支持；R6 地理另有严格道路解析與粗位置。R3 既有分組對照中，實際 reference 成員與層級變動為 0；本次不重做有價格的 reference 分析，也不把重新命名當成新訊號。[R3 處理紀錄](round3_preprocessing_record.md)、[R4 處理紀錄](round4_preprocessing_record.md)。
- **日期／屋齡：**本次日期無效 train 8、test 0，與 schema 一致；現行民國／西元日期、1990–2030 合理性旗標與以 2000-01-01 為零點的 trade_day_index 已存在。建築完成年月全空、管理組織常數是 [R7 原欄審查](round7_feature_audit.md) 的既有證據，本次未重讀原始欄；不能憑空補屋齡，也不把交易年份當屋齡。
- **格局／備註：**房廳衛、隔間、面積比例和八個備註旗標已有對應。rooms=0 有 22,560／2,782 列，其中 living_rooms>0 為 2,061／208；這是格局／申報不確定性，不能無證據當成缺值或自動補房數。本次沒讀備註原文，沒有依錯誤選詞或把可能含價格的全文加進模型。
- **95 欄的範圍：**既有 87 欄加 R6 的 8 地理欄為 95／23。道路或平面交點代表位置不是住宅門牌或建築內部座標；本次沒有發現可從已讀 X 補出的可信朝向、樓棟位置或車位實體幾何。

## 最多兩項後續假設；本輪不實作

1. **優先保留面積語意品質狀態作有限假設。**若日後另立規約，可事先固定「有車位但未分列面積」「獨立車位」「住宅面積差非正」等可解釋條件，研究是否明示這些條件比任由模型組合更穩定。不要補標準車位面積、把負差值裁為零、修改單價分母或 reference；也不要把條件互相重疊的列當成不同錯誤累加。現有特徵冗餘與單獨車位混入足以讓此假設無效，未構成採用證據。
2. **把樓層文法修復定位為資料品質工作。**預先區分帶單位的範圍分隔與真正負號，保留 raw 並定義多層摘要；先以合成及逐列變更差異核對。現有實際支持僅 test 5 列連字號與 train 1 列字形，不能期待它解決廣泛新建案泛化，更不能為增加覆蓋而加入本次未見的猜測文法。

這兩項不是本輪追加候選，不與已選識別欄消融混合。若未來真做價格實驗，仍須先凍結一個明確差異與同一年度驗證規則；不可用本次 test X 命中數當採用門檻，不可依反覆使用的年度成績事後修文法。歷史窗口已用於多輪研究，任何後續比較都不是新的盲測，也不能保證 Private 成績。

## 可重現口徑與來源身分

一次 pandas 讀兩份 features；用表列布林式計列数，再按 transaction_type 聚合。樓層令 `C=[0-9 + house_features._CN]`（正規表示式字元集合），固定查數字後的層／樓連字號、以地下／大寫 B 起始的完整數字列舉／範圍，以及屋頂／突出／頂樓／注音ㄧ／全形數字；不是任意模糊搜尋。數值缺失以 `isna()`，類別缺失以既有 `__MISSING__`。合成直接呼叫當前 `_floor_tuple`，不建立模型。只輸出上述聚合與樓層短字串，沒有地址、案名、戶號明細。

以下是本次原文或局部查閱的程式／schema／方法文件，以及唯一兩份被載入的資料。文件的既有統計與本次新計數在內文分開標示；雜湊只讀位元，不代表另載其連結中的資料。

| 路徑 | SHA-256 |
|---|---|
| src/house_features.py | `108e42d4429961a10a704f85bd66791599d904e6e96381c4c2f6dd4275316388` |
| src/round3_features.py | `b4a904b59f5e4a4f44c5cd6d7a36ffc9b94f340f8483db56ee56c93c082d95cd` |
| src/round4_augmented.py | `05d47c444cfe1de656c0174579dff25eed92ef98ac6e16ac5cceeb86cea5c837` |
| src/round6_geo.py | `df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0` |
| reports/feature_schema.json | `18d1a4c98c7645387480fee7bbac532ea44e6ca9487e5a11e535eb43419f80aa` |
| reports/round3_preprocessing_record.md | `49ecca52be944282158dd5192f85030966f10efb343c6ac02c901e4254c92a3a` |
| reports/round4_location_opportunity.md | `1727fb58a1b19a3429b91c3502035df279b56123e2e1febb60f13152b4bd4968` |
| reports/round4_preprocessing_record.md | `0d8f16a459f6ab0fec9fa928ceeefa8ca3e4a235a07bd8496c1eaba214d19ddb` |
| reports/round7_feature_audit.md | `6e7387608e1762233724b10c3e1f7e88a81e73a8ddd37fba82095ae45b245dc4` |
| reports/round16_numeric_review.md | `526a6dd34145ef431461226cde1cc12e98d9c96d1a47aae33e0a562ddb8ede4f` |
| reports/round16_prior_unit_methods.md | `2d2d4b544c02abda6fa56aeafc0e09fcac5aa68eb01f59eee824b2c112c839b7` |
| reports/round16_prefix_support.md | `7e92892c064d08b5c4c32ee7f8b8b33cd0a54d1c5c6e50b9226ddcb1450451fe` |
| reports/round16_plan.md | `1ad540cf19e7ad774b5ea01057771dcda1967fe85fedfff85fecedbb292ff659` |
| data_cache/train_features.pkl | `7ad92824e7410ae86a79e11c17199f3136651f50908b3ac06b1949734ac4a7a7` |
| data_cache/test_features.pkl | `5c6c1b220923bbc405e531c31b25595717b63b49afcffb2c3c438facaf9691d4` |

本文件為唯一新增產物；所有原 source、快取、模型、分數與既有報告保持不變。

### 執行程式保存紀錄

依主流程後續要求，將本次已成功執行的 stdin Python 內容原樣保存為 [round18_feature_audit.py](../src/round18_feature_audit.py)，SHA-256 `05bcaae2b03c2ed8c593cb3f72c056ef34ce0c5f9c31740a8e7de9038ae80ce7`。這是**執行完成後保存 recipe**，不是聲稱執行前已有這份檔案或已凍結候選；保存時只做 AST 語法檢查，沒有再次載入 X 或重跑盤點。原文「唯一新增產物／沒有新增 framework」中的產物範圍因此更新為本 MD 加這一份原樣程式，沒有增加新規則或新統計。

從專案根目錄重現：`.venv\Scripts\python.exe -X utf8 src\round18_feature_audit.py`。程式只將同一份純 X 計數 JSON 印到標準輸出，不寫資料、模型或報告；本次沒有另存 JSON 檔。程式的讀取路徑固定為上表两份特徵快取，另 import 已綁 SHA 的 house_features；資料前後雜湊一致才完成。
