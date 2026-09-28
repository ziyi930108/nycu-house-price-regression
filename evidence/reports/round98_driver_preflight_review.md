# R98 獨立 driver preflight

結論：就本次已檢查的 source、API、資料隔離與凍結契約，**沒有阻擋執行的問題**。審查者為 /root/r85_validation_audit。driver／helper API 與 pipeline 已在主線啟動前回報無 blocker；最後 request 的獨立機器複核及本書面落檔是在主線發出啟動通知之後完成。本文件不冒稱書面先於 launch，也不代表真實 meta-fit 或評分已成功。

已實核 [research request](../outputs/round98_launcher_20260929/research_request.json) 的 **28 個 source pins**，包含 driver 的 3 檔本地 closure、scorer 的 10 檔 closure，以及 helper／SciPy／HiGHS 共 5 個 source／binary pins；全部與落盤 bytes 一致。[evaluation request](../outputs/round98_launcher_20260929/evaluation_request.json) 的模型與評估參數、plan、preparation 身分，以及 9 份保護檔案 SHA 亦精確相符。兩份 canonical 合成收據均為 passed：helper 42 項、evaluator 42 項。本輪最後 request 複核 exit 0；僅讀程式及 metadata，未載入 array、真實標籤、模型或執行 fit／score。

[driver](../src/round98_temporal_stack.py) 每期只允許該期目錄下精確 11 個 array keys，僅包含合法歷史 meta_y，拒絕 query target 與跨期 path；既存 preparation 已綁定 canonical／raw 索引、日期與 meta/query 不重疊。三期由 [pipeline](../outputs/round98_launcher_20260929/run_pipeline.py) 依序啟動獨立子程序，全部 query predictions 封存後才可另行評分。helper 的參數、save/load API、全 query 重載一致性與不覆寫契約吻合；StackerFitError 會保存 solver metadata／arrays 後失敗退出，不重試或換權重。草稿中失敗證據保存不足與精確 stage readset 的問題，已在本次凍結前修正，並非失敗真實 fit。

最終 driver SHA：87e3ededcfd7520b3fcf730d2c1b962f5e8000cd0b49cd283756c4b2f144ac6d；contract：0bf50c43c4e3df4e506e9542dee35c76a77d984ce6870afb4a32ca3c3923cb62；helper：a1a501bc2c7874fc0d21983ad5a2be63e8d1c085f21d24a096220ff32dbfc864；scorer：6540f7045e8f47c6a0e8339a6680dd59a85839644bb0fcbbef192ea9ac7a23e1。各自的完整證據見 [helper 自查](round98_helper_selfcheck.json)、[evaluator 自查](round98_evaluator_selfcheck.json)、[artifact inventory](round98_artifact_inventory.md) 與 [固定計畫](round98_plan.md)。本文件不再改 source、request 或既存收據；Private 未知，選定提交仍為 v22，R98 尚須依真實執行結果判讀。
