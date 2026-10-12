# 原始地理資料建置的可選依賴

此清單適用於六個基底模型前置的 raw→地理資料建置程式。保存模型推論使用的 ML／NN lock 已涵蓋各自依賴；本補充處理新增地理來源程式所需的 Shapely、PyProj、Osmium，不要求推論流程重新取得地理資料。

建議在獨立的 Windows x64／CPython 3.12 地理環境，先安裝既有 ML 依賴，再安裝[地理依賴](../../../requirements-geography.txt)。不要把這套環境與 NN／CUDA 環境混裝。以下命令在 repo 根目錄執行，只是安裝介面，這次沒有實際執行：

```powershell
py -3.12 -m venv .venv-geo
.venv-geo/Scripts/python.exe -m pip install -r requirements-inference.txt
.venv-geo/Scripts/python.exe -m pip install -r requirements-geography.txt
```

NumPy 2.3.5、Pandas 3.0.1、Shapely 2.1.2、PyProj 3.7.2、Osmium 4.3.1 有原 `road_index_2024_annual_v3/audit.json` 的 `versions` 證據；後三項亦與本機 `.venv-geo` 的 distribution metadata 一致。`geo_osm_extract.py:63` 與 `geo_parent_coverage_v2.py:69` 明確要求 Osmium 4.3.1。OpenPyXL 3.1.5 沿用既有 ML 清單，供合法課程 Excel/cache 準備；不聲稱原地理環境已安裝它。證據路徑與原檔 SHA 見 [provenance](geography_dependency_provenance.json)。

可沿來源閱讀的原始入口包括：

- [geo_acquire_annual.py](../../../research/src/geo_acquire_annual.py)：`--year`／`--verify-only`，既有非價格地理快照取得與核對。
- [geo_osm_extract.py](../../../research/src/geo_osm_extract.py)：`--pbf`／`--output`，本機快照抽取；原檔拒絕既存輸出。
- [geo_parent_coverage_v2.py](../../../research/src/geo_parent_coverage_v2.py) 與 [geo_road_index_v3.py](../../../research/src/geo_road_index_v3.py)：父區域覆蓋證據與道路索引。
- [geo_address_audit.py](../../../research/src/geo_address_audit.py) 與 [geo_match_queries_v2.py](../../../research/src/geo_match_queries_v2.py)：合法課程地址解析、`--index`／`--parsed`／`--output` 對應。
- [round6_prepare_geo.py](../../../research/src/round6_prepare_geo.py)：`--stage`／`--matches`／`--index`／`--extraction`，對齊原始模型列；[round39_prepare_geo.py](../../../research/src/round39_prepare_geo.py) 保留原 `--run` 入口與歷史 guard。

這些是原 source CLI 介面，不是已在新 clone 成功執行的命令。仍需合法非價格地理快照、父區域／抽取／索引／對應 audit、課程 raw 資料、私有合法 cache、完整 source 身分與歷史路徑等前置資料。原 ROOT、SHA、拒覆寫與來源檢查都保留；單有套件清單不會自動解決這些依賴。

本補包只保存版本與來源證據，未讀模型、NPZ 或逐筆交易，未安裝套件、匯入地理模組、重建地理資料、訓練或評分。歷史安裝紀錄不等於 fresh 地理環境驗證，也不等於六模型完整重訓完成；Private 仍未知。
