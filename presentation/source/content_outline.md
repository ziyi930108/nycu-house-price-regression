# 房屋總價預測完整流程簡報大綱

本次只說明已選定提交模型。20 頁主講、2 頁附錄，不用內部版本比較要求觀眾理解研究歷史。每頁包含實際步驟與設計理由，並保留證據能支持到的範圍。

## 篇幅與配速

- 主講正文純漢字：4,143 字。另計英文詞及數字字串為 4,161 等價單位，後者不拿來支撐最低時間。
- 規劃時間：870 秒，約 14:30。
- 純漢字 300 字／分鐘：約 13 分 49 秒。250 字／分鐘：約 16 分 34 秒。目標 14–15 分鐘建議以約 280–295 字／分鐘排練。
- 不計投影片標題、重點列、引用、附錄、Q&A 或停頓。不以播放動畫或刻意停頓拉長。這是字數配速估計，尚非本人實際上台耗時。
- 凍結來源：narrative.json，SHA256 `5cb3136ee446128a96ce1f81b5ddbb4b77e555b420e67a3717d7075fb5a633e2`。

## 主講頁序

| 頁 | 主題 | 口述漢字 | 規劃秒數 | 累計時間 | 此頁為何必要 |
| --- | --- | ---: | ---: | --- | --- |
| 01 | 臺灣房屋總價預測 | 152 | 30 | 0:30 | 先給聽眾明確任務、選定狀態與敘事路線，避免要求理解內部版本歷史。 |
| 02 | 任務與評估指標 | 167 | 35 | 1:05 | 用兩筆明確示例說明回歸、單位與絕對值，避免把MAE當百分比或讓正負誤差抵銷。 |
| 03 | 從原始資料到提交檔 | 233 | 50 | 1:55 | 先交代流程與各階段目的，後續技術頁能對應到整體位置。 |
| 04 | 四份檔案的角色 | 173 | 35 | 2:30 | 明確區別valid與Kaggle test分割，避免將sample價格誤視為標籤。 |
| 05 | 清理重複與格式問題 | 228 | 50 | 3:20 | 解釋每一步刪除的是格式或重複資訊，而非依未知答案移除困難樣本。 |
| 06 | 把文字欄位轉成特徵 | 186 | 40 | 4:00 | 透過具體解析例子說明特徵工程，不把87欄當作抽象數量展示。 |
| 07 | 欄位取捨與地理資訊 | 187 | 40 | 4:40 | 解釋洩漏風險、不可用欄位與非價格地理來源，限定95欄只適用該模型。 |
| 08 | 用過去資料預測下一年 | 221 | 45 | 5:25 | 講清楚時間切分及狀態洩漏，避免把切索引誤當完整隔離。 |
| 09 | 六個模型與三條價格分支 | 215 | 45 | 6:10 | 解釋六模型計數、分支層級與log動機，避免把互補性寫成未做的消融結論。 |
| 10 | 單價表示與面積權重 | 241 | 50 | 7:00 | 以同單價誤差、不同面積示例說明權重原因，並保留固定fallback的語義。 |
| 11 | 單價 CatBoost 的近期成交權重 | 234 | 50 | 7:50 | 說明日期權重的假設、半衰期、正規化及限定範圍，避免聲稱必然有效。 |
| 12 | 神經網路與分組價格先驗 | 252 | 55 | 8:45 | 解釋embedding與prior的作用及OOF避免的洩漏類型，保留非時間OOF限制。 |
| 13 | 成交參考與空間修正 | 242 | 50 | 9:35 | 說明模型與局部參考分工、支持條件及回退，避免把流程描述成外部查價。 |
| 14 | 逐筆取三分支中位數 | 216 | 45 | 10:20 | 用明示為示意的數值解釋median，保留共同偏差與不可顛倒的計算順序。 |
| 15 | 三期年度回測結果 | 189 | 40 | 11:00 | 呈現唯一提交模型跨期結果並避免把跨資料集差異當成進步因果。 |
| 16 | Kaggle Public 與 Private | 167 | 35 | 11:35 | 清楚報單一提交模型成績與未知範圍，避免榜單版本比較或Private承諾。 |
| 17 | 泛化與高價尾部的限制 | 197 | 40 | 12:15 | 完整保留選型偏差、尾部風險與反證，不要求觀眾理解其他版本編號。 |
| 18 | 最終訓練與提交檔 | 218 | 45 | 13:00 | 區分歷史最終fit和目前推論，說明格式核對是交付的一部分。 |
| 19 | 保存模型的推論重現 | 240 | 50 | 13:50 | 說明真正重新推論與保存向量重播差異，防止誤把85秒當訓練或宣稱下載即可全鏈重訓。 |
| 20 | 結論 | 185 | 40 | 14:30 | 以完整方法與可追查交付作結，不改回版本排名或宣稱Private最佳。 |

## 投影片重點與依據

### 01 臺灣房屋總價預測

主講。

- 近期成交加權與多模型回歸
- 已選定提交模型 v22

來源：REPORT.md；Cover photograph: Markus Winkler, Taipei, Taiwan. https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs；Photo used under the Unsplash License as contextual imagery only.

### 02 任務與評估指標

主講。

- 輸出每筆交易的總價元
- MAE 是平均絕對金額誤差
- 越低越好

來源：REPORT.md

### 03 從原始資料到提交檔

主講。

- 讀檔並確認資料角色
- 清理與特徵轉換
- 時間回測與模型學習
- 全量訓練、推論及提交核對

來源：REPORT.md；src/predict_v22.py

### 04 四份檔案的角色

主講。

- train 455,787 列
- valid 11,994 列
- test 47,073 列
- sample 僅提供提交格式

來源：REPORT.md；research/src/prepare_data.py；research/src/model_common.py

### 05 清理重複與格式問題

主講。

- 移除 3 列混入的英文表頭
- 完整交易去重，排除編號、包含價格
- 最終 fit 467,221 筆
- test 全數保留

來源：REPORT.md；research/src/model_common.py；research/src/prepare_data.py

### 06 把文字欄位轉成特徵

主講。

- 民國日期轉成時間特徵
- 中文樓層保留地下與整棟資訊
- 地址拆出縣市、區域及道路
- 87 欄基礎特徵

來源：src/house_features.py；REPORT.md

### 07 欄位取捨與地理資訊

主講。

- 編號只用於對齊，總價只作標籤
- 排除完整備註與解約情形
- 不可用或常數欄位不納入
- 單價 CatBoost 使用 95 欄，23 欄類別

來源：src/house_features.py；REPORT.md；evidence/data_external/geography/ATTRIBUTION.md

### 08 用過去資料預測下一年

主講。

- 三個年度時間切分
- 模型與先驗狀態依當期 fit 建立
- query 價格只用來評分
- 歷史窗口已多次使用

來源：REPORT.md；evidence/outputs/round79/results.json

### 09 六個模型與三條價格分支

主講。

- 三個 CatBoost、一個 LightGBM、兩個 NN
- 單價、對數、神經網路三條分支
- LightGBM 由三條分支共用
- 不同表示是設計動機，仍需實測

來源：REPORT.md；src/predict_v22.py；research/src/round12_log_geo.py

### 10 單價表示與面積權重

主講。

- 目標 y ÷（A × 10,000）
- 預測乘回 A × 10,000 得到元
- 面積權重 A ÷ 100 對齊總價誤差
- A 使用固定有效面積規則

來源：REPORT.md；research/src/round19_unit_geo.py；research/src/round79_recency_unit.py

### 11 單價 CatBoost 的近期成交權重

主講。

- 365 天半衰期
- 相差一年，較早日期的相對權重減半
- 合法日期正規化，無效日期權重 1
- 單價 CatBoost 權重＝面積 × 日期

來源：REPORT.md；research/src/round60_recency_price_prior.py；research/src/round79_recency_unit.py

### 12 神經網路與分組價格先驗

主講。

- 類別 embedding 搭配數值編碼
- 面積 NN 與先驗倍率 NN
- 價格先驗採五折建案分組 OOF
- 分組 OOF 不等於時間 OOF

來源：REPORT.md；research/src/round47_nn_train.py；research/src/round60_recency_price_prior.py；src/nn_prepare.py

### 13 成交參考與空間修正

主講。

- 可比成交只來自 fit
- 先按建案、地址及道路尋找支持
- 參考較弱時才補充空間資訊
- 支持不足時回退，不查外部成交價

來源：research/src/round2_neighborhood.py；research/src/round8_spatial_reference.py；src/tree_runtime.py；REPORT.md

### 14 逐筆取三分支中位數

主講。

- 所有分支先還原成總價元
- 各分支完成修正後再取中位數
- 降低單一極端輸出的影響
- 共同偏差仍可能保留

來源：src/predict_v22.py；REPORT.md

### 15 三期年度回測結果

主講。

- 2021–2022：230.94 萬元
- 2022–2023：140.74 萬元
- 2023–2024：138.32 萬元
- 不同年度是不同評分集合

來源：REPORT.md；evidence/outputs/round79/results.json

### 16 Kaggle Public 與 Private

主講。

- 已回報 Public MAE：1,998,568.5706 元
- 約 199.86 萬元
- Public 約 30%，Private 約 70%
- Private 分數未知

來源：REPORT.md；evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json

### 17 泛化與高價尾部的限制

主講。

- 窗口重用可能累積選型偏差
- 整體 MAE 會掩蓋價格群退步
- 中位數不能消除共同偏差
- 保存反證，不按未知真價改價

來源：REPORT.md；evidence/reports/round79_report.md；evidence/reports/round83_delivery.md

### 18 最終訓練與提交檔

主講。

- 467,221 筆有標籤資料建立最終保存狀態
- test 47,073 筆全部產生預測
- 編號、總價元兩欄，按 sample 順序
- 最後只取整一次，核對格式與雜湊

來源：REPORT.md；research/src/model_common.py；src/submission_io.py；src/predict_v22.py

### 19 保存模型的推論重現

主講。

- 原始 test 加保存模型、地理及 fit 狀態
- 真正重跑四個樹模型與兩個 NN
- 47,073 筆完全一致，最大差 0 元
- 84.74 秒是推論時間，從零重訓未驗證

來源：src/predict_v22.py；evidence/reproduction/integration_verified.json；docs/REPRODUCIBILITY.md

### 20 結論

主講。

- 清楚定義資料與可用資訊
- 把單價學習對齊總價誤差
- 不同價格表示經固定流程組合
- 結果可追查，未知與限制保留

來源：REPORT.md；README.md

### 21 附錄 A 完整模型公式

附錄，不計主講時間。

- 所有預測先還原為總價元
- 各分支完成 R 與 S，再逐筆 median
- 精確係數供問答與程式核對

來源：src/predict_v22.py；REPORT.md

### 22 附錄 B 參數與重現條件

附錄，不計主講時間。

- 單價 CatBoost：MAE、2,999 樹、深度 8
- NN：512／256／128，30 epochs
- 受測 NN：RTX 4060 Ti、CUDA 12.6
- 完整環境與限制見 REPRODUCIBILITY

來源：REPORT.md；docs/REPRODUCIBILITY.md；evidence/reproduction/integration_verified.json

## 敘事界線

- 模型家族互補、log 尺度、近期權重及中位數皆先說設計動機。沒有逐模組隔離消融就不說已證明每個元件更好。
- 基礎特徵不讀標籤，但價格先驗與成交參考使用當期 fit 的已知價格；待預測交易的答案不參與建構。
- 三期年度回測不混 valid；最終全量 fit 才併入 valid。這句不擴張成整個研究從未用 valid 做過任何評估。
- 單價 CatBoost 的面積×日期權重有明確範圍，其他模型各有設定。先驗五折按建案分組，不是時間 OOF。
- 六模型先還原成元，各分支依序做成交參考及空間修正，最後三分支 median，再一次取整。
- 47,073 筆相同證明保存模型的推論重現，84.74 秒只屬受測同機推論；不宣稱新環境安裝或全鏈重訓已完成。
- MAE 分別標明是歷史 query、Kaggle Public 或未知 Private。示例交易不當作實際競賽樣本。

## 交接給簡報製作

製作程式只讀 narrative.json 的 slides。頁面 number/title/key_points 用於版面，script 為完整口述正文，seconds 僅用排練規劃。visual 的數值與 is_illustrative 標記是內容依據，不能另行更換示例造成講稿不一致。sources 放在每頁備忘稿。所有主講頁的 script 都應保留在成品 Notes 與逐頁講稿中。

本檔是內容製作大綱。上台提示與不計主講時間的問答見 speaker_coaching.md。
