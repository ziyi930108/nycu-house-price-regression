# R92 固定計畫：降低近期加權單價樹的交互深度

日期：2026-09-29（Asia/Taipei）。R91的中間期已完成、未過門檻，其餘兩期仍按原計畫執行。R92獨立測容量，不把R91目標或R89趨勢併入；不沿著Public分數調參。v22唯一提交與簡報保持。

## 假設與控制

R84在相同depth8增加樹數沒有穩定收益。既有raw總價模型曾試depth6，不等於目前近期加權单價表示的depth6已驗證。R92檢查較淺單棵樹是否減少稀疏建案／地區類別的高階交互，提升新案與跨年度泛化；相反也可能欠擬合，這是待驗假設。

唯一參數變動：R79 unit CatBoost depth 8→6。固定2999 iterations、lr.065、seed42、l2=6、subsample=.85、random_strength=.5與其餘全部原参数。95特徵／23類別、原面積unit target y/(A*1e4)、weight=A/100*原保存R63 float32 date weights轉float64、原inverse與正值投影保持。不重算日期權重、不改半衰期、無OOF新特徵、不加入prior offset。

仍.75*新unit native總價+.25*原LGB，同窗原reference/spatial，再與保存NN/log分支median。不能順手拿掉LGB或另改median。原模型／程式／SHA與選定CSV全部保留，新source/state/model/output各用新檔。

## 可核驗的實作

R92獨立wrapper/family明載新depth；內含純X的legacy R19 feature state供完整相同比較，其depth8及unit歷史欄只屬原特徵建構來源。自己的check_model驗實際depth6、2999樹、每棵最多64 leaves、95 native names／23 cat positions，其他參數完全符合固定值。不能去改舊check_model或移除舊SHA拒覆寫規則。

先作bounded synthetic/FakePool：Pool target／area×dateweight確實不變、params只有depth不同、95欄保持、錯depth/leaf/category拒絕、query API無label。真fit前凍結source與計畫／selfcheck SHA，主線按單GPU串行；不得與R91昂貴fit並行。

## 三期驗證及門檻

完整development、confirmation、auxiliary全部執行。原fit/query索引、日期、y、X/geo、原reference/spatial、日期權重完整SHA／列序核驗；v22 base→reference→spatial→median重播相同。每模型保存、全query raw與TWD重載逐位相同，公開helper推論與手算inverse一致。主對照v22，次對照v24。

固定報告整體、加權、新舊案、所有城市／季度、真價五帶、尾部、已有test-X組成估計，以及native/base/reference/spatial/median每層。24個paired區間：两對照×三期×建案／月份区塊×overall／weighted，2000次seed20260925。不按真價或test ID部署事後規則。

採用研究門檻同本批：三期整體及固定加權全優於v22，最新整體改善至少0.5%；再審新案／高價／季度重大代價及全部區間。通過後才考慮固定R90新切點重建檢查，不自动全量、CSV或Kaggle改勾選。不通過保留研究，不在同一輪掃depth／tree／learning-rate／seed。

原控制未同期GPU重訓，固定seed也非位元決定性；差異不能全當depth的純因果效果。這是受前面結果啟發的適應性研究，多輪重用窗口不是盲測；bootstrap不校正選型偏差或估計Private勝率。只用課程fit價格，禁止到內政部系統對答案。所有失敗與不利分群保存。
