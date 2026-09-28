# R100 評分器 v2：前置身分契約修正

2026-09-29。主模型已在訓練，尚未執行本輪真實評分；此次修正不改模型、實驗計畫、評分、抽樣或採用門檻。作者仍為評分器作者，不能稱完全獨立重作。

已凍結的 v1 在 check_pure_X_audit 把純 X 結果的 plan_sha256 和模型科學計畫比較。實際結果綁定的是獨立 [純 X 計畫](round100_x_audit_plan.md)，其 SHA 為 d536d65461487af9ef116eadc7299d8b84c67fa069665a8feac51cc2438fde6b；[原 audit request](../outputs/round100_x_audit_preflight/request.json) 的 scientific_plan_sha256 才綁定 [模型計畫](round100_plan.md) bbb29bcc…2e107。錯誤由 helper 作者在結果稽核前的來源／metadata 閱讀發現，若未修正會在讀取真價前錯誤拒絕；沒有實際 fit 或評分失敗。

另存 [新評分器 v2](../src/round100_temporal_prior_evaluate_v2.py)，SHA c12e69764470b9f47a33f1fc1dbae0063a31aff1bfe5568f487d48a08f8792bf。v2 精確核 result 路徑／SHA、result→request SHA、request→audit plan SHA 與 scientific plan SHA，以及兩份 metadata 的原執行順序、rules、source/input maps。保留 audit 的 auxiliary/development/confirmation 順序；模型與評分仍 development/confirmation/auxiliary。[窄 diff](../outputs/round100_evaluator_v2_revision/engineering_diff.patch) 只涉及前置核對與人工測例，31個其餘函數 AST 不變。[修訂收據](../outputs/round100_evaluator_v2_revision/revision.json) 與原 v1 完整bytes皆保存，原凍結 source／收據／GitHub前置快照未改寫。

[42項人工檢查](round100_evaluator_selfcheck_v2.json) 首次成功，測得 0.343 秒；另做 [真實 metadata 窄檢查](../outputs/round100_evaluator_v2_revision/actual_metadata_check.json)，測得 0.141 秒。後者讀實際 pure-X result/request、兩個計畫、原主／評分 request、三期 fit-label metadata，以及已完成 development 兩臂 training JSON，接口皆相容。沒有載入陣列／真價／模型，也未執行完整 prelabel gate或評分；其餘兩期仍由主線執行，不能把此檢查稱為全部六fit產物驗證。

主線需另凍結 evaluation_request_v2 並以 v2 同一 CLI 執行；不能拿 v1 request 對 v2 source。CLI仍為 --sealed-manifest PATH --evaluation-request PATH --out NEWDIR。原 [評分契約](round100_evaluation_contract.md) 是當時審查範圍的歷史證據，保留其未抓到此身份錯配的事實。v22仍唯一提交，Private未知。
