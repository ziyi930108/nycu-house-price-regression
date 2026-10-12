# 簡報重建來源

本資料夾重建20頁主講簡報，無附錄。主講正文4,222純漢字，275字／分估15:21；250–300字／分約14.07–16.89分鐘，未由本人排練計時。章節為資料準備3–9、模型訓練10–13、預測整合14–16、驗證與改進17–20。

## 重建文件

需要Node.js、`@oai/artifact-tool`、Python，以及Presentations技能所附驗證工具。先建候選，再finalize、生成逐頁PNG；用新的輸出目錄與修訂名稱，避免覆寫既有證據。

```powershell
node ./build_presentation.mjs C:/path/to/new-build
node ./finalize_presentation.mjs C:/path/to/new-build r1
```

不傳輸出目錄時預設為此source資料夾旁的`rebuild`。這是文件重建，不會讀取原房價Excel或訓練模型，更不等於乾淨環境完整重訓六個模型。

可覆蓋環境路徑：`RUNTIME_NODE_MODULES`指向含`@oai/artifact-tool`的node_modules；`PRESENTATIONS_SKILL_DIR`指向Presentations技能根目錄；`RUNTIME_PYTHON`指向Python可執行檔。程式預設使用本機Codex隨附runtime。字體沿用已核對參考版的Microsoft JhengHei，以design font policy明示保留設計，不依賴舊交付PPTX路徑。

## 檔案與資料

- `build_presentation.mjs`：可編輯形狀、流程連接線、表格與圖表。
- `finalize_presentation.mjs`：20頁套件、版面及字體檢查，補入5個圖表Excel工作簿，再渲染。
- `content.json`：20頁標題、主講正文、補充理由、來源與配速。
- `training_curve.json`：原30輪、60個MAE值。圖中數字存入Excel時保留15位有效數；原始數值不改。
- `analysis.json`：圖表及匿名案例的顯示值與來源雜湊。
- `city_statistics.json`、`validation_report_summary.json`、`analysis_receipt.json`：已驗證分析摘要及收據，由最終r3分析複製；不包含逐列原始索引與抽樣散點資料。價格帶為左含右不含，首組未滿500萬元，末組5000萬元以上。
- `assets/house_illustration.png`與`.json`：既有房屋示意圖及來源紀錄。
- `reproducibility_receipt.json`：公開匿名的兩環境推論核對摘要，同機既有與全新套件環境各重載六模型，13份47,073列向量及完整CSV一致；不含逐筆預測、編號或權重。

8個原生表格位於5、6、7、8、9、10、14、20頁；5個原生圖表位於6、11、12、17、18頁；第3頁為原生方塊與連接線，僅房屋示意圖為PNG。城市圖明示6個城市，其統計來源涵蓋清理後455,255筆train。完整方法價格帶圖涵蓋2023-08-11至2024-08-10全部133,697筆保留交易；圖表為萬元，JSON保留元與全精度。柱圖Excel數字必要時正規化到15位有效數，未改權威摘要。第17頁單網路曲線是另外的固定設定診斷，不是完整融合曲線或原提交軌跡。

實際已完成本機source建檔、finalize與全20頁渲染／目視檢查；完整收據在同交付的QA說明中。未宣稱原生PowerPoint介面、不同Office版本或全新電腦安裝已測試。Classroom依使用者指示先跳過；本次Kaggle僅授權修改姓名與GitHub，提交和最終勾選延後。本重建未操作外部平台、未改模型與CSV，Private仍未知。

本次僅調整第3、16頁的原全量階段與後續推論重現說明，18頁正文／理由／來源保持。後續兩環境真推論無新fit，保存預測只供推論完成後核對。它證明同機保存狀態推論重現，尚未完成原始資料全六模型重訓、不同硬體／作業系統或其他套件版本測試；直接R2中間層未另匯出，不主張該層直接相等。預測差為0不是測試真價MAE，Private未知。
