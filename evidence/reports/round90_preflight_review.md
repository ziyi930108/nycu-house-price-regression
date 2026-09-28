# R90 資料／四樹 helper：預備審查

記錄日期：2026-09-29（Asia/Taipei）。執行與審查者：子智能體 /root/r85_validation_audit。

**R90 尚未執行真實 preparation、模型 fit、推論或評分。** R89 未通過後，主執行緒決定保留 R90 為錯開時間窗口的可用基礎程式，不為該候選啟動六 native 重建。

## 資料 helper 已完成

[src/round90_offset_data.py](../src/round90_offset_data.py) 固定 fit < 2023-02-11，query 2023-02-11～2024-02-10。API：

- load_inputs() → OffsetData：只讀來源已綁 SHA 的純 X、索引、地理與小 metadata；不讀價格表。
- prepare(out) → OffsetData：預設 outputs/round90/preparation；拒絕既有目錄，保存純 X／索引／manifest／execution。
- load_prepared(out, expected_manifest_sha) → OffsetData：要求呼叫者已綁 manifest SHA，核 artifact、frame、dtype、日期、fit/query 分離及 R6/R39 規則。
- self_check()：合成邊界與索引映射檢查，沒有真資料讀取。

回傳 fit_indices、query_indices、fit_source_cache_indices、query_source_cache_indices、Xfit、Xquery、tree_fit、tree_query、spatial_fit、spatial_query、categorical_columns、contract。fit y 應由未來主 driver 另用 fit_source_cache_indices 抽取；contract.inherited_provenance 保存原 raw_train SHA，但 helper 不開啟或重hash價格表。

來源 R79 plan、R6/R39 manifest、2023 PBF extraction audit 綁定原 SHA；R39 manifest 與 R79 內嵌記錄一致。R6 全 canonical geography 依新索引重切；R39 舊 fit/query geography 以其 canonical 索引聯集重映射。實際 PBF header 時間需早於新 Feb origin。已有完整地理鏈是繼承證據，helper 明載不重hash全部 PBF/鏈、也不開 test/valid。

資料 helper SHA：4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1。

## 已實際完成的檢查

AST 通過。12項合成檢查實際 exit0，檢查精確名單見 [round90_data_selfcheck.json](round90_data_selfcheck.json)：包含起日／末日邊界、無效日期、canonical與raw位置區別、輸入不變、價格欄拒絕、重複／越界／缺少／重疊索引拒絕，以及跨舊role地理映射和dtype保留。為保存收據沒有重跑，也未編造耗時。

另只讀小 metadata 核對4份既有 sidecar、R39嵌入R79身分、pure-X旗標、schema來源與matching/index audit SHA。這不能替代未執行的真資料逐列重映射驗證。

## 四樹 helper 只讀結果

[src/round90_offset_trees.py](../src/round90_offset_trees.py) 本次審查 SHA：666e9fc61d22565aae2f738767bfef471e12679d4e1c0bbe86cf366afa540110。

未發現目標／權重／API／fit-query隔離的必修錯：

- raw：原87欄、y/1e6、固定2999棵與原參數。
- LightGBM：fit-only category maps，y/(A×1e4)，factor/1e6=A/100，固定3999棵；未知 query category 不新增映射。
- log：使用原 R46 fit_model，新 fit 日期計權，無 query y。
- unit：使用原 R79 fit_model，接收已綁定的float32日期權重，原面積×日期配方。
- 四支各保存模型並核全部 query 的總價重載一致；兩geographic CatBoost另保存 feature state。無eval_set、early stopping或query y參數。

source_hashes() 記錄的是呼叫時目前檔案身分，本身不等於固定實驗plan核驗。未來新driver仍必須在fit之前核：新fi/qi、fit y行序／SHA、date32來源和值SHA、資料contract、來源／參數與預先凍結plan相等。helper文件已明示caller-verified window；這些不能靠此只讀審查冒稱完成。

## 實作邊界

原 source、SHA、CSV、模型與 state 未修改。沒有新 origin 列數／模型成本／分數，沒有查外部答案。完整依賴與舊實测成本見 [錯開窗口盤點](r89_offset_validation_inventory.md)。R91 屬另一個預先固定的新目標表示候選，不可把 R90 的未執行程式列為已完成壓力測試。
