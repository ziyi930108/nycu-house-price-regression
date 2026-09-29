# R103：固定組合的誤差抵銷與中位數路由診斷

2026-09-29。在讀取本輪逐筆預測／query真價前固定。R102已完成並發布；上輪屬實質進展，得到原生NN三窗改善、完整組合兩退一進的證據。本轮不產生模型候選或新融合係數，Private未知，唯一提交仍v22。

## 問題與邊界

R102 T相對v22，最近確認期在base已由原生改善轉成退步；development到median才轉退。固定檢查這兩個轉折，以區分「原生誤差改善」與「既有組合的誤差抵銷改變」。分析是對反覆使用的舊query做事後描述，不能當新盲測、因果處理效果或Private勝率。

只比較T−v22和T−R102正零C，三窗全部報。不改權重、不重訓、不重新推論／prior／R/S、不搜比例、seed或門檻，不計新bootstrap、不查外部成交答案、不讀test/valid。不按真價改預測，也不保存真價逐筆檔。

## 固定輸入與身分

取R102 root／stage seals所綁定的同窗保存向量：area、unit LGB、H、U及v22/C/T的native、base、reference、spatial和median。重播原base與median公式並核exact；只載NPY，不載模型或prepared NPZ。原root seal、評分JSON、各stage seals、所需array/feature/raw/cache-index文件和source SHA在執行前寫入request；未使用的模型／705 prior state不重讀或全面核驗。

使用原canonical mapping、各stage fit/query indices、source-cache indices與pure-X，核對完整outer-fit／query日期、筆數、排序與映射。query y依原評分器 `raw_train.iloc[canonical_mapping[query_indices]]['總價元']` 取得，值SHA必須等於R102評分所存身份。raw pickle實際會反序列化整表，僅選出的query y供診斷；不冒稱只讀query bytes。seen/new/missing依完整outer fit，而不是eligible loss子集。

三個窗口：development 2022-08-11–2023-08-10，81,234筆；confirmation 2023-08-11–2024-08-10，133,697筆；auxiliary 2021-08-11–2022-08-10，105,511筆。既有模型切點不改，沒有將較晚模型回測。

## 診斷一：base誤差抵銷的恆等分解

保持原運算順序 `B=.75*(.5*A+.5*P)+.25*L`；誤差e=prediction−truth。

每列抵銷量 `K=.375*abs(eA)+.375*abs(eP)+.25*abs(eL)−abs(eB)`。

在A/L/y皆固定時，`delta_abs_base=.375*delta_abs_native−delta_K`，除float64捨入外是代數恆等式。核最大殘差<=1e-5元；K容許>=−1e-5，不能clip後偷改恆等式。分別報原生改善項、抵銷改變項及base結果；層間差不能自行相加當因果貢獻。

另按P與固定剩餘加權誤差 `.375*eA+.25*eL` 的負／零／正方向，保存每arm九個格子的筆數與base誤差貢獻。零採精確float64零，不另調epsilon。這些格子只作描述，不導出逐筆分流規則。

## 診斷二：median是固定H/U區間內的截取

令lo=min(H,U)、hi=max(H,U)、N=spatial，核 `median(N,H,U)=clip(N,lo,hi)` exact。

狀態固定六種：below、lower_equal（lo<hi且N=lo）、inside、upper_equal（lo<hi且N=hi）、above、both_equal（lo=hi=N）。各對照到T的36種轉移全部列出，空格保留零筆/null，不只選改善群。

每列 `J=abs(clip(N,lo,hi)−y)−abs(N−y)`；`delta_abs_median=delta_abs_spatial+delta_J`，最大殘差<=1e-5元。每轉移報spatial差、截取影響差與median差，均列對全窗的贡献；分组總和須回到全窗。另固定按spatial誤差差的改善／相同／退步 × median誤差差的改善／相同／退步九格報筆數及貢獻。沒有尋找最佳N/H/U權重。

## 全部預定群與摘要

全窗、seen/new_nonmissing/missing、全部城市（空值獨立）、四relative季度、五真價帶（<10m、10–20m、20–50m、50–100m、>=100m）全部保存上述總分解；全部36 median轉移及九格符號表在全窗保存。價格群依已知驗證y，僅用於診斷。保留aux嘉義市／屏東縣與confirmation 20–50m等既有反證，不依结果丟群。

同時報普通平均與固定seen/new加權。全窗每seen列权重 .36617593949822613/n_seen、new列 .6338240605017739/n_new、missing为0；各群列对全窗加權贡献与群內正質量加權平均，質量零回null。不能把群內平均當全窗贡献。固定分割的群／路由貢獻須加回全窗，兩分解恆等式亦核同時成立。保存原生/base/reference/spatial/median MAE，与R102 aggregate以<=1e-6元核對；新算只是描述性重播，不新增測試集。

先以人工數據驗證恆等式、邊界／H=U、零權重／空群及列序／來源拒絕；source與request凍結後才真跑。用最小必要來源核查，不重做已完成訓練、bootstrap或全面模型核驗。失敗保留source/錯誤與輸出，不自動覆寫重試。

## 後續決策

若最近期base退步由抵銷減少解釋，下一個模型方向應檢查fit側合法的組合目標／互補訓練；不能在本次query上找權重。若development改善被固定median截取消失，記錄具體路由贡献，再考慮另立fit-side模型假設；不能從此診斷直接批准替換v22。任何新fit需另先固定計畫、合法out-of-time狀態與跨窗評估。

保留所有來源與結果、實測耗時，更新HANDOFF與HLOG／新文件ZIP。GitHub只同步程式與聚合說明；本輪新增逐筆預測／model／CSV均為0。
