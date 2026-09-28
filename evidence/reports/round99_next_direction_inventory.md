# R99 後有限方向盤點

2026-09-29。僅來源閱讀，非新計畫、實作或採用決定；未載價格／逐筆向量，未 fit、預測或評分。

**僅保留一個待核假設：prior 的時間前置生成。** 本次搜尋 src／reports 的 temporal、chronological、rolling、時間 OOF 等字樣，抽查 R52／60／63／65／67／72／86／91／94–95 規約及 R90 準備程式；此有限範圍未找到已執行的時間前置 prior，不宣稱鑑識全部歷史產物。

[R60 helper](../src/round60_recency_price_prior.py) 以 `reference=~held` 建立建案五折 prior，明記 `chronological_oof=False`：其他折可含較 held row 晚的 fit 成交；query 則只用截止日前完整 fit state。[R90](../src/round90_offset_nn.py) 仍沿用。[R63](round63_plan.md) 改日期損失權重，[R65](round65_plan.md)／[R72](round72_plan.md) 增加欄位，[R91](round91_plan.md) 改殘差目標，均未改這項時間生成規則。這不是外層 query 標籤洩漏。

可否證假說：固定時間區塊的訓練列，只取區塊起點前、且排除原 held 建案折的價格生成 prior，使 ratio NN 學習時也面對「用較早成交估下一段」的基準；query 仍用同窗 full-fit prior。保留原層級、面積、近期中位數規則及總價 L1，其他分支與最終 median 不改；不是再搜尋融合比例。它只能檢驗時間方向，不會自動消除四折／全 fit 的支持量差。

**反證與缺口優先。** [R60](round60_report.md) 既有近期 prior 已改善三期；[R67](round67_report.md) 支持量欄只改善最近期，較早 query 支持量 P99 為106,366、fit為6,705，但粗層回退也可能造成此差，不能歸因時間或據此保證收益。時間截斷會減少參考案，可能增加回退及尾部不穩定。最早資料沒有更早標籤，暖身段、無效日期、空 reference 與區塊邊界尚未固定；若需讓早期列退出 NN 損失，也同時改變訓練樣本，不能稱純單因素改善。

因此只值得先另立契約、以純 X 日期／分組核支持可行性，再決定是否實驗。若立項，預期需三窗重建 prior、各新增一個 prior NN fit；這是工作量推估，沒有量測新耗時。先固定一種區塊與冷啟動政策，完整跑完三窗，不按首窗調參；須核原生到完整模型、全部正反分群與未知 Private 的界線。

本文件未授權下一輪，也未修改 R99 凍結檔、v22 唯一提交或 CSV。尚無足夠證據預期它一定優於 v22。
