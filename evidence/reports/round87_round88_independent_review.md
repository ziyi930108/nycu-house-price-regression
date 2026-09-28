# R87／R88 獨立複核

日期：2026-09-28。本審查同意兩輪保留研究、不進final fit／test推論／CSV。v22仍為使用者選定版本，Private未知；這不是證明v22必定贏Private。

## 範圍與結論

讀取兩輪固定計畫與完整driver、三期結果、generic診斷、兩組配對結果；另外核對R87報告／decision及R88合成policy檢查。沒有新fit、模型load、reference/spatial呼叫或候選價格生成。R88另只讀保存向量／mask核對九個分支的政策與非mask位元不變。

| 輪次 | 中間年度 ΔMAE | 最新年度 ΔMAE | 最舊年度 ΔMAE | 建議 |
| --- | ---: | ---: | ---: | --- |
| R87 | -735.06 | +8,628.64 | -9,817.53 | 不進全量fit |
| R88 | +10,977.52 | +7,675.02 | +16,617.14 | 不進全量fit |

差值是完整median的候選−v22，以新臺幣元計，負數改善。不是原生單支差或Public分數。

## R87：不能用三期兩勝代替泛化證據

- 程式按固定左至右順序計算0.375×R79 native TWD + 0.375×R85 native TWD + 0.25×原LGB，再重算同窗reference/spatial，最後median固定NN/log。沒有平均已修正分支或事後搜索比例。
- 同窗索引、fit/query日期、標籤／特徵／日期權重SHA、fit-only狀態及v22各層重播都設有檢查。R79/R85原生向量已是TWD，因此沒有混用單價與總價。
- 中間年度改善735.06元很小，最新年度反而退步8,628.64元；最新新案76,096列增加11,089.78元、已見案57,601列增加5,377.26元，遠期Q4增加37,511.01元。這是對「可以取代v22」的重要反證。
- 最新≥1億的319列MAE增加5,601,171.13元，對全期差貢獻+13,364.35元。中低價三帶雖都改善，仍被尾部抵銷；不得依未知真價分流。
- 最舊期整體改善9,817.53元，≥1億171列獨自貢獻−9,997.52元，其餘四帶合計+179.99元。這個結果不能代表一般交易全面進步，也不等於任意一筆極端值造成全部收益。
- 固定test組成估計對v22在中間／最新期增加24,711.60／14,866.78元，最舊改善10,838.00元。這是有歷史支持cell的重加權估計，不是test或Private真實MAE。
- R87作者報告與decision的主要結論、所有24區間、分群和執行秒數核對一致。其不進final建議合理；本審查没有另最佳化或採用。

## R88：政策正確實作，但全面撤銷無支持列的趨勢未改善整體

政策必須且實際使用：source等於base_fallback，且spatial.applied為False。不是兩條件任一成立，也不是按廠辦、城市、價格或ID挑選。

| 分支 | 回退來源 | 原分支來源 |
| --- | --- | --- |
| unit | R79各期base.npy | R79 unit_spatial.npy |
| NN | R71各期seed42/base.npy | R71 seed42/nn_spatial.npy |
| log | R46各期base_recency_log.npy | R46 recency_log_spatial.npy |

回退的是各分支原有blend base，並非某支native。原base仍含模型的時間特徵／日期權重，不能說完全取消時間資訊。driver先重播三分支reference、spatial與完整diagnostics並精確匹配，再套policy；query標籤只進評分。合成檢查包含全mask、無reference-mask、已套spatial、混合四案例與輸入不變。

本審查另外重新讀取保存的base、original、candidate、mask及兩類diagnostics，只做相等比較：9組mask全等於AND規則；mask內candidate全等於base；mask外candidate與original的bytes全部相同；三期最終非union列與原v22 bytes相同。沒有產生任何新候選價格。

| 期別 | 三分支共同mask列 | 最終價格改變列 | query總列 |
| --- | ---: | ---: | ---: |
| 中間 | 10,617 | 10,617 | 81,234 |
| 最新 | 11,641 | 11,641 | 133,697 |
| 最舊 | 37,337 | 37,337 | 105,511 |

三分支在本資料的mask實際相同，但實作仍分別用自己的diagnostics，不把這個觀察寫死成共用mask。

| 期別 | 分支 | mask內原MAE | mask內回退MAE | 對該分支全期MAE差 |
| --- | --- | ---: | ---: | ---: |
| 中間 | log | 2,359,618.57 | 2,470,212.61 | +14,454.26 |
| 中間 | nn | 2,273,550.38 | 2,218,780.23 | -7,158.27 |
| 中間 | unit | 2,137,627.17 | 2,240,376.69 | +13,429.00 |
| 最新 | log | 2,772,580.72 | 2,836,078.83 | +5,528.78 |
| 最新 | nn | 2,107,347.89 | 2,140,638.65 | +2,898.63 |
| 最新 | unit | 1,972,192.54 | 2,095,988.23 | +10,778.89 |
| 最舊 | log | 2,356,828.29 | 2,405,170.35 | +17,106.72 |
| 最舊 | nn | 2,298,342.57 | 2,334,976.58 | +12,963.62 |
| 最舊 | unit | 2,105,516.26 | 2,145,348.01 | +14,095.19 |

保留有利反證：中間期NN回退分支本身改善7,158.27元，但log／unit退步，最終median仍退步10,977.52元。中間／最新已見案分别改善5,199.17／3,548.00元，新案卻增加19,949.57／16,170.31元；最舊新舊案都退步。最新Q1/Q2及中間Q1有改善，不能說每一群都較差。

真價分帶中，中間≥1億182列改善1,101,240.07元，最新5,000萬–1億1,725列改善64,799.15元，最舊≥1億171列改善404,226.87元；但1,000–2,000萬與2,000–5,000萬兩個主體價帶三期全退步。這與「局部撤銷趨勢可能有利，但普遍新案需要外推」相容，不是已證明完整市場因果機制。

R88整體、固定加權與固定test組成三期均較v22差，因此不支持擴大成新提交。此消融僅否定本次固定AND政策的整體採用，不表示所有時間修正都必須保留，也沒有授權繼續依分群搜索例外。

## 兩輪各24個配對區間已完整查看

| 輪次 | 對照 | 全負 | 跨零 | 全正 |
| --- | --- | ---: | ---: | ---: |
| R87 | v22 | 0 | 12 | 0 |
| R87 | v24 | 0 | 5 | 7 |
| R88 | v22 | 0 | 6 | 6 |
| R88 | v24 | 0 | 4 | 8 |

R87對v22的12項全部跨零；對v24最新期4項全正、最舊期3項全正，其餘跨零。部分全正下界僅數十元，不應把邊界讀成強而普遍的證明。
R88對v22：中間month兩項全正、project兩項跨零；最新4項跨零；最舊4項全正。對v24：中間4項跨零，最新／最舊8項全正。這支持保留不利風險，但不是宣稱每一期都有顯著退步。

全部採原固定2,000次、seed20260925、建案群集與三個月循環區塊，valid均2,000、invalid均0。沒有重新抽樣或挑一種CI。兩輪均源於已看過的殘差／R85結果，屬適應性研究；窗口已反覆選型，bootstrap沒有校正這些選擇，也不涵蓋重訓及未來市場移轉，不能當Private勝率。

## 執行與範圍界線

R87主流程外層47.063秒，R88外層57.734秒，均exit0。兩輪各0fit、0原生模型load/predict；R87有6次reference與6次spatial呼叫（對照＋候選），R88各9次（3分支×3期原流程重播）。不是整段訓練耗時。

獨立檢查没有更改任何driver、計畫、結果、模型、CSV、selector或root handoff；只新增本報告。R88完整作者報告仍由實驗作者整理，本報告的數字直接引用已完成且有SHA的main、generic與paired結果，不冒稱已複核尚未提供的作者報告。

## 核對來源身分

下列SHA為已完成結果與實際執行來源；未在作者修改preflight途中提前凍結。其餘向量身分見各期artifact清單。本次新報告不取代原結果。

| 檔案 | SHA256 |
| --- | --- |
| src/round87_experiment.py | 25d5c02bbb20fcc692cccb4caada1123ac448b3e83ae4571abd0b39fd644fc93 |
| reports/round87_plan.md | 73efc2c8f6611c002cac83bad431018e41a22bbfc78396d421b13292f18c1de4 |
| outputs/round87/results.json | 97149b56e4c8955b9e653a2cd59e3f624a7c002697687fc1cde0648b09e64d21 |
| reports/round87_diagnostics.json | 5258b62be65d22131f890df57ed1c6cb021d26382756c984992f9714e945f571 |
| outputs/round87/paired_vs_v22/results.json | 44402d668221976873f6f3ac4548fa100ed33962d23dc98e5c2d14842a2ec5c8 |
| outputs/round87/paired_vs_v24/results.json | 416f42a6d1382ba9ce675609ed6d9a7ffff3dff241a1a90c4bcea116a38e1fca |
| src/round88_experiment.py | a005f51c04d5308fb22ea77c3c24f3129be4e03987f81d1b7cfec1f75cc1da9f |
| reports/round88_plan.md | a835546cfa5a6476f02de6db580a21844411b851f3a9a1da1dd12188760ed668 |
| outputs/round88/results.json | 67a2b84b566575f524b8bf781dcf4f4d35f3e4bd1442a7b2ebb9fe404a6de6ce |
| reports/round88_diagnostics.json | 5be4fc08cf9f1c769074e12badbfd852bd3d6ff2ff8714b30627fd44fe8eb858 |
| outputs/round88/paired_vs_v22/results.json | c21646ab04d426a98c184e5f4413eb3b74a226ad3d186f182000fe3014775baf |
| outputs/round88/paired_vs_v24/results.json | 8a685fbe9a0225b2f1119623d9bfee91b021e8938c4231ac96171d3bdaa84499 |

[R87作者報告](round87_report.md) · [R87主結果](../outputs/round87/results.json) · [R88固定計畫](round88_plan.md) · [R88主結果](../outputs/round88/results.json) · [R88完整generic診斷](round88_diagnostics.json)
