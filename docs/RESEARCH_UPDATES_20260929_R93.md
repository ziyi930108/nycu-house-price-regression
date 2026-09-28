# 回歸改良研究：2026-09-29，R93 建案密度權重

R93 三期訓練、完整分群與配對診斷已完成。**沒有達到事前採用門檻，因此保留研究，不替換 v22、不全量 fit、不新增 Kaggle CSV。** 這不表示它在所有群都無效，也不表示已證明全面顯著退步。

使用者唯一提交仍為 v22，已回報 Public MAE 1,998,568.5706 元，Private 未知。R93 的本機整理以已發布的656143a3a55c3f8444546f8011338349ca8f1d66為基礎；本頁建立時未執行新 commit、push 或 ZIP。發布身分以後續 Git 歷史與發布收據為準。

## 固定假設與實際改動

大型建案包含許多相似成交列，可能在模型切分與葉節點估計中占較大比重。本輪檢查：只用 fit 建案出現次數折減其相對權重，能否改善新案與跨年度泛化。R5 早已提出未實測的密度折減備案，本輪是延伸實驗，不能稱為首次提出。

原 w0=(A/100)×date64。對同期 fit 的「城市|行政區|完整案名」計數 n，具名 q=1/sqrt(n)，缺名 q=1；再用 c=sum(w0)/sum(w0×q) 形成 w1=w0×q×c，保留原 Pool 總權重尺度。缺名及單例也會乘 c，最終權重並非不變；這不是 q/mean(q)。

唯一更動是 fit 的相對樣本權重。原95欄／23類別、unit target y/(A×10000)、depth8、2999樹、lr .065、seed42及其他參數保持。query 不重算建案密度，預測也不乘 q 或 c。完整組合仍為75%新單價 native總價＋25%原 LightGBM，套原 reference/spatial後，與保存 NN／log 分支取中位數。

## 三期完整組合 MAE

以下是本機歷史回測，單位新臺幣元、越低越好。差額=R93−v22，負值較好；不是 Kaggle Private 分數。

| query年度 | v22 MAE | R93 MAE | 整體差額 | 原固定加權差額 |
|---|---:|---:|---:|---:|
| 2021-08-11～2022-08-10 | 2,309,352.36 | 2,298,811.82 | -10,540.53 | -8,031.26 |
| 2022-08-11～2023-08-10 | 1,407,387.07 | 1,408,053.97 | +666.90 | +598.05 |
| 2023-08-11～2024-08-10 | 1,383,228.41 | 1,387,196.68 | +3,968.28 | +4,875.51 |

中間、最新的整體及加權點估計退步，最舊改善。最新改善率為−0.2869%（即退步約0.287%）；「三期整體及固定加權全部改善、最新整體至少改善0.5%」三項固定條件皆未過。沒有因指標不理想而改門檻、搜尋權重強度或只挑有利窗口。[不可變決策](../evidence/outputs/round93/decision_review.json)保存精確數值。

## 配對區間應如何解讀

每個對照都保留三期×建案／月份區塊×整體／加權，共12個區間；每項2,000次、固定seed。對 v22 的結果是 **2個偏改善、10個跨零、0個偏退步**；兩個負區間都是最舊年度的月份抽樣，分別為整體與加權。中間、最新的八個區間全部跨零，不能把正的點估計說成已證明顯著變差。

對 v24 的12個區間為0個偏改善、5個跨零、7個偏退步，原有較強的本機次對照也保留。未通過事前門檻是本輪不採用的理由；區間跨零既不是證明等效，也不支持 Private 必勝。窗口已反覆研究、區間彼此相關，這些 bootstrap 未校正多輪選型、GPU重訓或未來市場移轉，不是 Private 勝率。

[對v22的全部區間](../evidence/reports/round93_paired_vs_v22.md)與[對v24的全部區間](../evidence/reports/round93_paired_vs_v24.md)保留完整端點。

## 哪些地方有利，哪些地方付出代價

- **已見建案**三期都小幅改善（約4,036／4,018／905元）。原本希望改善的**新案**，中間、最新反而退步約3,275／10,014元，最舊改善約12,148元。故不能只由「降低大案票重」推論新案一定更好。
- **5,000萬至1億**價帶三期皆改善，但**億元以上**三期都退步。最舊年度低於1億的四個價帶及四季度全部改善，最舊收益不能說成只來自少數豪宅。
- 中間年度 native/reference/spatial 指標較好，最外層 median 卻略差；最舊 native/spatial 較差，但 median 較好。最終選擇具非線性，各層差值不能直接相加作因果解釋。

上述價帶只作事後診斷，未知 test 真價不能用來部署逐筆選模。完整城市／季度、價格帶、新舊案與各層結果見[完整報告](../evidence/reports/round93_report.md)、[診斷](../evidence/reports/round93_diagnostics.md)及[保存結果](../evidence/outputs/round93/results.json)。

## fit 權重盤點：不是所有樣本都更平均

在模型評分外另做了不使用價格陣列的 fit 分布盤點。大建案按次數折減後，**最大單列權重占比三期都上升**，權重有效列數均下降：

| fit期別 | 最大單列占比：原→新 | 有效權重列數：原→新 | 缺名資料 Pool占比：原→新 |
|---|---:|---:|---:|
| 中間年度 | 0.169%→0.659% | 73,989→14,564 | 0.199%→1.278% |
| 最新年度 | 0.092%→0.623% | 114,802→15,827 | 0.084%→0.567% |
| 最舊年度 | 0.147%→0.208% | 36,838→14,943 | 1.367%→9.894% |

有效權重列數=(sum(w))²/sum(w²)，只描述權重集中度，不能當作獨立樣本數、資訊量或泛化保證。中間／最新期具名資料仍占新 Pool 的98.72%／99.43%，並非缺名舊資料取得大部分權重；原面積、日期與小案權重的交互也會改變集中度及年份質量。

[fit分布報告](../evidence/reports/round93_fit_distribution.md)與[彙總JSON](../evidence/outputs/round93_fit_distribution/results.json)保存具名／缺名、全部年份及交叉表。不匯出案名、地址、逐筆權重或價格。

## 來源核验與實測時間

每期 fit 前先由driver獨立重建 counts/q/c/w，核對 [fit預檢](../evidence/outputs/round93_fit_audit/results.json)完整值SHA、原日期權重、canonical索引、日期、特徵及地理來源。原 v22 base→reference→spatial→median 向量與指標完整重播。三個新 CBM 保存後，全 query raw 和還原 TWD 重載逐位相同。

53項合成檢查（0.328秒、2次 FakeModel、0真fit）與正式訓練分開。首次合成斷言查錯 legacy metadata層的 KeyError 保留，修正讀取位置而未改權重公式。所有 main、generic與paired流程均exit0；合成測例不是三次真fit或真CBM核驗的替代證據。

| 完成項目 | 次數／時間 |
|---|---:|
| 新 CatBoost native fit | 3次，共473.484秒 |
| 完整主流程 | 549.828秒 |
| generic分群診斷 | 26.906秒 |
| fit計數／權重預檢 | 4.031秒 |
| fit權重分布盤點 | 5.781秒 |

fit已包含於主流程，不能重複相加；診斷可能並行，也不能累加成聊天牆鐘時間。未量測整段開發對話耗時。

資料邊界需如實描述：fit audit的 train_features pickle先整表反序列化再取fit；只取prepared NPZ的train_sample_weight成員，不讀價格陣列。主訓練driver另載完整canonical y，但只有fit y傳入訓練，query y僅用score。不能宣稱所有query bytes從未被載入。固定seed的GPU不保證位元決定性，舊控制未同時重訓。

## R90與Attention都不是已完成的新模型

R90仍只有固定2023-02-11切點的純X資料準備（272,117 fit／107,169 query）與data/tree/NN helper；新窗口encoder、OOF prior、reference/spatial及六個native未fit，沒有該窗口MAE。

下一個欄位 self-attention 方向只有[文獻與來源可行性盤點](../evidence/reports/attention_regression_feasibility_after_r93.md)。報告探索替換prior NN架構的單一候選與成本，**尚未實作、未跑合成GPU smoke、未fit、沒有預測或分數**，也未凍結上游revision。本頁不將建議當作採用決定，不用舊MLP耗時冒充Attention benchmark。可行性盤點接獲任務時R93仍在執行，是歷史時序；本頁以現在完成的R93結果為準。

任何下一輪都需另固定實作／来源／成本邊界，沿合法fit/query狀態評估；不能把架構較新當成保證較準。此批結束不表示整個專案不再研究，後續另批保存。

## 程式與證據入口

- [固定計畫](../evidence/reports/round93_plan.md)、[早期方向盤點](../evidence/reports/next_regression_direction_after_r92.md)、[訓練前獨立審核](../evidence/reports/r93_validation_audit.md)。
- [權重／模型helper](../research/src/round93_project_density_unit.py)、[三期driver](../research/src/round93_experiment.py)、[報告／決策程式](../research/src/round93_review.py)。
- [完成後獨立核驗](../evidence/reports/round93_independent_review.md)：另讀3個CBM、核398個主鏈檔案SHA與全部24個保存區間端點，沒有重新fit或predict。
- [合成收據](../evidence/reports/round93_helper_selfcheck.json)、[main execution](../evidence/outputs/round93/execution.json)、[完整結果](../evidence/outputs/round93/results.json)、[採用審查](../evidence/outputs/round93/decision_review.json)。
- [本批複製清單](../evidence/outputs/research_snapshot_r93_20260929/copy_manifest.json)。
- [前批R89／R91／R92](RESEARCH_UPDATES_20260929.md)、[更早R85–R88](RESEARCH_UPDATES_20260928.md)。

本批只保留原bytes程式、報告、彙總JSON與SHA，排除資料、native模型、private／pickle state、NPY／NPZ、逐筆價格及CSV。379支歷史source_manifest、原v22、正式REPORT/PDF與22頁簡報不改。evidence中的凍結報告沿原工作區相對路徑保存；../src對應research/src，未包含的私有資料仍需原工作區。程式可追溯與保存重載不等於全新環境從原始資料端到端重訓。
