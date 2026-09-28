# R92獨立實作審查

日期：2026-09-29（Asia/Taipei）。範圍為固定計畫、helper、driver、合成收據及有限歷史盤點；未自行fit、未載入CatBoost模型、未產生預測或讀test。核對時`outputs/round92`尚不存在。

**結論：未發現阻擋正式三期實驗的問題。** 本結論是實作合約審查通過，沒有宣稱depth6已改善MAE，也沒有把FakeModel測項當真實CBM容量證據。實際模型的容量及全query重載仍須由正式driver完成。

## 不是重做原R15候選

| 歷史方向 | 已測內容及結果 | 與R92差異 |
| --- | --- | --- |
| R15 | 原95欄、depth8→6、raw y/1e6、uniform Pool。開發期v3融合MAE 1,502,390.31→1,553,514.34，退步3.4028%；確認／輔助期未跑 | 替换37.5%地理raw槽且只有原reference。不是R79 unit target、date×area權重與當前三分支median |
| R19／R79 | 原單價target與面積inverse均使用depth8；R79加入保存R63日期權重 | 這次才將此近期加權unit元件固定depth6 |
| R84 | 相同unit表示與depth8，seed42／137各把樹數2999→4499；三期未穩定通過 | 未檢驗depth6，且兩seed平均與本次單seed42不同 |

本次透過reports的depth/shallow檢索、R15／R19／R79／R84 source與report有限盤點，未找到既有「R79 date×area unit depth6」完成紀錄。不將此有限盤點說成所有歷史參數空間的窮盡證明。R15的原始不利證據保留，淺樹可能欠擬合，不能預先宣稱有利。

## Helper與數學合約

- `parameters()`複製原R79／R19參數，只把depth8改6，參數鍵集合不变並斷言唯一差異。2999樹、lr .065、seed42、l2=6、subsample=.85、random_strength=.5、其他設定維持。
- `effective_area`、`transform_target`、`inverse_transform`直接沿用原R19。target為`y/(A*1e4)`，inverse為原非負`z*A*1e4`；沒有prior offset或额外特徵。
- `sample_weights`以原R19 `A/100`乘保存R63 float32日期weight轉float64；不重算、不正規化，拒絕錯dtype、非正／非有限及Series索引不符。
- Pool實際收到相同95欄／23類別、unit label、area×date weight，只在constructor depth為6。無eval_set、早停、query label或query weight API。
- wrapper是獨立R92身分；原R19純X legacy state完整嵌入不改寫，歷史depth8／weight欄明載只屬feature構造與舊訓練來源。wrapper的actual parameters／weight identities另記，driver核它與同窗原R19 state相等。
- 新`check_model`直接查原生model，未呼叫舊depth8 checker：2999 trees、depth6、SymmetricTree、Plain、MAE、seed42；全部95 feature names與23 cat positions；每棵葉數為正整數且≤64，保存完整leaf vector SHA。其餘實際核心整數參數及浮點參數核對；浮點使用既定1e-7相對容差以容許CatBoost儲存表示。

## Driver與資料邊界

`fit_model(X,y,G,cats,w)`／`predict_model(model,state,X,G,cats)`及driver的`raw_predict`接口相容。driver先核原source／input SHA、凍結selfcheck receipt、fit/query canonical索引、日期區間、X／geo frame身分、fit target、R63 NPZ float32日期權重與date flags／day值SHA。新Pool weights另逐位核對原`A/100*w.astype(float64)`。

原v22完整base→reference→spatial→median先重播；新支仍固定`.75*native + .25*原LGB`，同窗原reference/spatial各一次，再和保存NN/log分支median。没有因變淺而移除LGB、另調融合、換地理或重建reference/spatial。輸出保留v11/v12/v5/v22/v24/R92完整metrics供通用診斷。

driver保存CBM和新wrapper，使用新checker重新載入，核全部query raw及TWD逐位相同；公開helper預測亦和顯式inverse相同。這些檢查寫在正式執行路徑上，但此獨立審查沒有冒稱尚未執行的CBM重載已通過。真query價格可由共用loader讀入，僅用於對照與候選評分，不進fit接口；不能描述成整個driver從未載入query價格。

## 已核驗的合成與來源

[selfcheck receipt](round92_helper_selfcheck.json)為31項、0.187秒、exit0、1次FakeModel fit、0次真fit。已獨立核其source SHA全部對上目前helper／舊source；原R79 plan包含的舊source SHA亦未改。只讀解析driver/helper語法通過，沒有重跑selfcheck或啟動GPU。

31項涵蓋有效面積fallback、target/inverse、unit weighted-L1與date weighted總價L1等價、Pool確實傳label/weight、只有depth改動、95/23欄與legacy state保留、錯depth／seed／lr／iterations／L2／leaf／categories／names拒絕、query不收y、輸入不變與模擬JSON／FakeModel重載。

GPU同期depth8控制未重訓；既有控制與新候選差異包含重訓非確定性。三期是已重用窗口，不能當新盲測；31個工程檢查也不證明泛化或Private更好。三期整體／加權、最新≥0.5%改善、分群／季度／高價與24個配對區間均保留正式結果後判斷，不於本審查提前採用。

## 審查身分

- [固定計畫](round92_plan.md)：`8b1a7419fb3536eca1179c202f58984c0941f6b5f79d0714adc7b3b1c3ec10f7`
- [driver](../src/round92_experiment.py)：`d84cb2acb42f188a049768ad55f1277cfa3ee6ab5305b4856b0f14c826817f43`
- [helper](../src/round92_shallow_unit.py)：`21bbf98551de72aa3d03da367f8b46abd8b39746bbdfafe5e081fcb03786fd07`
- [selfcheck](round92_helper_selfcheck.json)：`590261e49d90a245bf07135a4f2ba9a248033ffbdff6c5979e39a8be2297b7ca`
- [R79 helper](../src/round79_recency_unit.py)：`9a124322a139e5621c28d26a6e7ab95385c73507f017a2765d9eaaa0eef39cd2`
- [R19 helper](../src/round19_unit_geo.py)：`fac832e6c81871833a136d05759c7f0926fb749d2320a9c39b581e3986dc75cf`

歷史依據：[R15](round15_report.md)、[R19](round19_report.md)、[R84](round84_report.md)。本報告不改任何source、舊模型、提交CSV、selector或正式使用者文件。
