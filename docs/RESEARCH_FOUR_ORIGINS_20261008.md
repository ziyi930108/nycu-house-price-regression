# 2026-10-08 四起點後續研究：R114–R117

v22仍為使用者選定提交；已回報Public MAE為1,998,568.5706元，Private未知。新研究／v25探索候選不自動改Kaggle勾選。
本批與先前R111／R112六fit＋R113三fit分開。成功批次共12 native fits：R114四NN＋R116四CatBoost＋R117 v2四NN，後者另四次fit-only先驗重建。原R117還有一次先驗／NN完成但封存前工程失敗的aborted attempt；因此本批實際13次native fit、5次新prior重建，沒有隱藏重跑成本。R115是四起點純X診斷，沒有fit或MAE。
以下是重用歷史窗口的離線TWD MAE差；負值較保存v22好，新候選Public／Private均未知。

|固定方法／理由|dev ΔMAE|latest ΔMAE|aux ΔMAE|Feb ΔMAE|探索／強combined|真實終態|
|---|---:|---:|---:|---:|---|---|
|[R114](../evidence/outputs/round114/results.json)：先驗金額＋signed NN修正；weighted總價MAE，推論一次非負投影|+3,300.76|+5,182.91|-5,589.82|+260.75|False／False|combined門檻失敗，不產CSV：Combined exploration failed: latest overall and weighted MAE worsened, only auxiliary improved, and a large Chiayi group regressed above five percent.|
|[R116](../evidence/outputs/round116/results.json)：單價CatBoost只加入Depthwise；Plain/date×area保持|-8,852.44|+20,177.67|-12,226.44|+12,602.16|False／False|combined門檻失敗，不產CSV：Combined exploration failed: latest and February overall and weighted MAE worsened with multiple large-group regressions above five percent.|
|[R117](../evidence/outputs/round117_v2/results.json)：fit先驗改固定交易五折；原Softplus倍率NN／其他八array／query先驗保持|-1,461.88|+1,820.78|-5,105.09|-3,498.62|False／False|combined門檻失敗，不產CSV：Combined exploration failed: latest overall and weighted MAE worsened despite three other origins improving; original fixed latest-no-worse condition failed.|

原三August窗與固定Feb起點分開評估，兩套combined均要求對應三窗gate AND 額外origin gate。探索容許每窗至多0.25%退步且限制大群／季度退步；強gate另要求全面改善與最新至少0.5%。不因結果更改門檻。
R115支持／ESS與同案暴露的診斷只提供一次受限fold比較的理由，不是價格改善證據。R117交易fold允許同案peer，因此即使OOF排除自身，也不是group隔離或chronological OOF；query仍使用原全fit先驗。
R114／R116原診斷內false只描述封存時的預測投影／count統計；事後raw／unprojected MAE實際使用query真價，僅於四origin全部封存及成功退出後。另附明確語意binding，不改原source、results、gates或舊收據。

|研究|真命令訓練／run秒數|另評分秒數|獨立審查秒數來源|
|---|---:|---:|---|
|R114|286.812|170.828|[postfit](../evidence/outputs/research_round114_20261008/postfit_review.json) 175.609|
|R115|95.937|無MAE|[postfit](../evidence/outputs/research_round115_20261008/postfit_review.json) 3.016|
|R116|601.844|471.828|[postfit](../evidence/outputs/research_round116_20261008/postfit_review.json) 249.110|
|R117|416.140|190.547|[postfit](../evidence/outputs/research_round117_v2_20261008/postfit_review.json) 295.000|

原R117 aborted train命令exit1、84.125秒；無stage／全批seal／MAE，不能當模型效果失敗或v2成功fit。修復只補動態父來源閉包，四prior＋四NN於新目錄重跑；原source/request/artifacts/failed receipts保留。
沒有把wrapper秒數、driver秒數或各fit相加成新的ETA。

[R118可行性盤點](../evidence/outputs/research_round118_feasibility_20261008/feasibility.md)：77個精確inputs、30作者人工checks。另附root獨立有界核驗：重跑30人工reader traps、四meta遮罩／ID hash／重疊與三日期UTC邊界。只讀日期／索引／metadata及保存向量opaque SHA；零fit／MAE／價格或逐筆預測載入。原manifest內pending旗標保留為當時快照，以新增dated root review補充；沒有冒稱全量地理binary replay。
既有早期地圖時間邊界可支持2021-01-03起點（prefix126,957／query7,848），但完整forward block仍須另建4樹＋2NN及各prefix encoder／prior／reference／spatial。可建構不等於強驗證；這份盤點沒有選融合模型或自動訓練，之後工作須另立規約。

限制：反覆使用同窗有選型偏差，bootstrap不是Private勝率，GPU seed不保證逐bytes重訓。保存重載與來源完整，不等於新電腦端到端重訓已完成。未去內政部成交系統對答案。
程式／證據包排除課程原表、CSV、逐筆價格／向量、模型、encoder／prior／reference狀態。僅有本機不公開產物的SHA引用。既有379歷史、13production與所有舊媒體／收據保持。
[本批複製manifest](../evidence/outputs/research_package_four_origins_20261008/copy_manifest.json)；本工具不Git操作、不上傳／勾選Kaggle，不使用額外點數或重置。
