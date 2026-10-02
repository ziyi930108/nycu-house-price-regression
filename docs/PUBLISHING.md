# GitHub 交付與後續更新

儲存庫：[ziyi930108/nycu-house-price-regression](https://github.com/ziyi930108/nycu-house-price-regression)，私人。程式、正式報告與簡報供專案成員閱讀，交易資料、模型權重與價格先驗不包含於 repository。

## 修改後檢查

1. 保留舊模型、原始CSV、歷史實驗收據。新實驗使用新目錄。
2. 更新REPORT、README、簡報與相應證據，清楚區分實測與未知。
3. 如推論程式影響數值，重新執行適當的模型驗證；只改文件不需重訓。
4. 執行格式測試，重新打包SOURCE_MANIFEST，再執行來源核驗。

~~~powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/package_source.py --output ../source_snapshot_NEW.zip
.venv\Scripts\python.exe scripts/verify_source.py
~~~

ZIP檔名每次使用新的名稱。打包工具會更新SOURCE_MANIFEST.json，verify_source.py會核對這份清單與379支歷史程式來源SHA。提交時將修改檔案和新清單一起commit。

.gitattributes使用 -text 保留來源精確bytes，避免Git自動換行讓歷史SHA失效。請不要為了通過驗證而刪除或跳過雜湊檢查；需記錄新的合法產物身分。

## 檢查範圍與自動化設定

本機驗證涵蓋source manifest、歷史來源SHA、私有檔案排除與4項提交格式測試，不載入模型、資料或訓練GPU。

自動檢查範例保留在source-checks.example.yml，尚未啟用GitHub Actions。現有OAuth登入缺workflow scope，GitHub拒絕建立.github/workflows設定；這不影響程式、報告與簡報的發布。日後若要啟用，可在具有workflow權限的登入下把範例移至.github/workflows/，再刷新manifest和提交。自動測試也不能驗證Private分數或替代已記錄的本機全量推論。

完整模型推論仍需DATA.md所列的合法資料與private artifacts。使用者已選定v22為唯一提交版本，Private分數仍未知；未代操作Kaggle最終勾選。

## 原工作區交接

原C:/NYCU_ML持續維護HANDOFF.md與HANDOFF_LOG.md，研究完成時更新狀態及原因，執行src/export_handoff.py產生新的交接快照。GitHub的文件更新與本機交接記錄應保持一致。
