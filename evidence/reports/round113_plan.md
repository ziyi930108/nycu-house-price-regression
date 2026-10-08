# R113：三個固定方法的非8月起點穩定性檢查

2026-10-08。在R111／R112六次模型全部評分前固定；目前尚未讀該批新query分數。此輪不提出第四個新模型、不改R111／R112原三窗門檻，也不重新選參數；檢驗原R107、R111、R112在同一個已存在的R97時間起點能否保持。三者一律執行，不按前批分數決定是否做。

## 固定方法與切點

直接使用已凍結R107 weighted SmoothL1 beta=.1 million prior NN、R111固定兩block residual prior NN、R112原R79 unit/date×面積的Ordered helper，三個source保持原科學配置、seed與epochs。共新增2個NN及1個CatBoost，主線單GPU依序，不重建或微調其他native，不混合新方法。

唯一時間起點固定2023-02-11，query至2024-02-10（含）；fit272117、query107169。重用R97在該起點真正fit的encoder、144/23九陣列、group OOF/full priors、date weights、其他native、reference與spatial，以及R90當時地圖／純X資料。不能使用較晚或全量模型回推，也不能修改原c.SPLITS或把此窗口假名成development。模型driver只讀已保存的fit_y；原隔離擷取器曾反序列化整raw價格表的歷史限制保留。

先核R90 preparation manifest和R97 sealed_manifest及每個依賴SHA、canonical/raw索引／日期／fit目標／prepared identity；將R97自己的v22與v24五層／四分支和最終median精確重播。R107/R111僅替換N_prior槽，R112僅替換unit42在v22的unit槽；R/S固定且各分支只各套一次。v24是原保存配方的次對照，不把候選塞進v24再調組合。三個新模型、metadata與五層向量直接封存並獨立核對後才讀query價格。

## 固定評估與額外發行檢查

沿R97的TWD整體／固定新舊案加權、城市／月份／相對季度／新舊案／五真價帶、五層及誤差尾部。對v22與v24均完整呈現。依原建案與3月循環月份區塊各2000次、seed20260925／stage index0；不因新結果改抽法或挑群。真價帶僅診斷，不作test路由。

額外穩定性條件在新分數之前固定：整體及加權各不得比同origin v22退步超過0.25%，任一季度或≥500列群不得退步>5%。同時兩指標都降低才稱此origin點估計一致支持；容忍範圍內仍有退步須明列，不能稱改善。這是額外反證檢查，不是Private概率或新的三比一投票。

R111／R112若要新全量發行，除原三窗探索門檻、獨立審查外，還須通過本額外穩定性條件與其獨立審查；未通過不發行v28/v29。這是發行前更嚴格的核對，不回寫先前不可變decision／release。既有v25保留原技術發行身分，此輪只新增R107的後續證據／建議，不改其價格或宣稱原來看過此origin。

## 限制與執行

R97 origin早已供v22/v24評分，且與三個8月query窗口重疊；新增模型的同origin得分尚未知，資料仍非盲測、窗口不能當獨立樣本加總。OOF是建案分組而非時間OOF。保存對照不是同期matched重訓；GPU固定seed不保證bitwise，saved-model全query reload仍須exact。Bootstrap未校正累積選型、重訓或未來shift，不能稱Private勝率。保存可追溯不是乾淨環境完整重訓。

嚴禁內政部或外部成交答案、test ID調價、Public搜參數。所有source/input/protected於新fit前凍結，拒覆寫，工程失敗保留且不自動重跑。原v22選定、簡報、旁白及Kaggle保持；Codex剩餘≤20%保存交接並暫停，不重置／購買／使用額外點數。R113沒有新CSV或finalfit。
