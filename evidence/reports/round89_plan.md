# R89 固定計畫：按建物型態／用途收縮時間斜率，reference與spatial一致重建

日期：2026-09-28。接續活躍目標「繼續製作」。本計畫在R89候選評分前落地，規則在fit支持盤點結果回報前固定。不是全新盲測；主對照v22/R79、次對照v24/R83。保存所有失敗、原模型、CSV及選定紀錄。

## 假設與舊先例

R85–R88未提供穩定替代方案，直接撤銷無支持列的全域時間外推也沒有改善。R3/R45已研究按城市收縮的future-baseline趨勢，但局部reference正規化及R8 spatial仍使用原global beta。R89改用建物型態×主要用途，且讓fit殘差、local query hedonic、future base與spatial同用群時間係數，檢查不同物件趨勢異質性。不是首次研究分群時間趨勢，也不將舊架構稱為程式錯誤。

## 唯一固定估計規則

沿同一期原reference的finite residual選定列、近4年、有建案名，每reference project key至少8列。所有欄位與價格均來自該期fit。先保留project交易日期跨度至少180日的全部列。群是building_type與main_use的原字串tuple，不按城市或query結果再分群；缺任一類別的群回退。

每群需至少1000列、30個city|district|project獨立案、且所選列日期總跨度至少730日。這是識別時間斜率的固定支持條件；不足／未知群使用原global beta。

在每個reference project內，將4個hedonic欄及log(y/building_area)減去案內平均。保留原global beta的前三係數，對完整global殘差：
    response = centered_log_unit - centered_hedonic @ global_beta
    predictor = centered_transaction_years
以一維Huber IRLS估delta。6輪、log residual threshold=.10、gram ridge=1e-5。每案初始總權重1（每列1/案列數），每輪權重為此初始權重乘Huber因子，不只第一輪使用。

raw_gamma = global_time_beta + fitted_delta
clipped_gamma = clip(raw_gamma, -.08, .20)
alpha = min(.5, independent_projects/(independent_projects+200))
effective_gamma = (1-alpha)*global_time_beta + alpha*clipped_gamma

沒有搜尋門檻、比例、正則、年度或類別例外。只fit時間delta，不重估其他三個斜率；每案等權是新估計設計，與舊R3逐列等權不同。

## 一致重建及推論

對所有finite原reference列（不只估計斜率的recent/support子集），按其群更新：
    residual_new = residual_old - hedonic_time * (effective_gamma - global_time_beta)
原參考行／近鄰結構／fill／距離尺度不變，但用新殘差重新算local center、MAD、confidence；以新殘差重新build spatial，每期一份新空間state。每reference group和每spatial group的fit rows必須屬單一type/use，直接從X檢查，不拆字串猜測。

新wrapper有獨立family/format/source identity，global beta只作fallback。query按type/use分組，reference/spatial view套同一gamma及新殘差、重算全部diagnostics，依原列序散回。不是將新beta塞到舊殘差／spatial狀態。所有三分支一起使用相同新trend狀態，原NN/log/unit各自base保持v22原配方。

為保留fallback位元相同，helper先用原global reference/spatial完整重播query，只有supported且delta非零群以新結果覆蓋，其餘保留原結果。實際推論不讀舊預測檔充當模型。global重播需與保存三支及最終v22完全匹配。

原local hedonic/G使用query原日期與fit fill；只有future-baseline的日期差封頂366天。這個既有不對稱保留，不混入新的cap改動。spatial缺日期仍不套用。

## 驗證與保存

沿R79 development、confirmation、auxiliary順序三期，全部做完才判斷。各期fit/query索引、日期界線、課程labels、X/geo、原reference/spatial與分支身份先以原SHA核對。只給fit_model該期fit標籤；predict_model不接query labels。沒有新CatBoost/NN fit或GPU工作；有三次CPU群趨勢wrapper fit與三次new spatial build，群內係數fit數另列。

保存新wrapper/model、metadata、source、indices、各branch reference/spatial/diagnostics及最終未取整TWD。每期存檔重載，全部query／三branch輸出及diagnostics精確匹配。合成檢查包含正負趨勢、無效支持與未知群、收縮與clip、案內中心化／等權、殘差與query一致、輸入不變、fallback精確、分組還原列序等。必要失敗修復須在真實評分前完成；中途失敗保留新目錄與收據，不移除舊SHA或覆寫保護。

輸出與既有generic/paired格式相容。報告overall、固定加權、季度、城市、新舊案、真價五帶與test組成估計；後者仍不是test MAE。兩對照各建案／月份區塊2000次、seed20260925，共24區間全保存，不能當Private勝率。

## 採用界線

三期整體及固定加權須同時優於v22，最新期整體改善至少0.5%，才值得進一步考慮全量fit/test候選；這是本輪事前研究門檻，不是Private保證。即使通過，仍須檢查新案／高價帶重大退步與區間，不自動發布CSV或改Kaggle選定。若未通過，完整保留反證與原v22，不在同一轮搜尋例外、群別比例或新的支持門檻。

價格標籤只用課程資料，禁止去內政部系統對答案，不使用未知test真價、test ID人工調價或Public搜參數。舊窗口已重複選型，這項新假設也是受既有誤差啟發；bootstrap不校正這些選擇或未来市場變化。新切點需所有受影響fit狀態按新邊界重建。
