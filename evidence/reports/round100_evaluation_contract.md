# R100 評分器契約與訓練前審查

本文件由評分器作者 /root/r85_validation_audit 撰寫，並只讀審查主 driver；不是評分器的完全獨立重作。2026-09-29，尚未執行真實評分或讀取 query 價格；六次模型訓練由主線另外協調。

[固定計畫](round100_plan.md) SHA bbb29bcc4fb83eaa3fdac9f15ade8221ced3437abacb04e891038ffb3462e107。評分入口為 [round100_temporal_prior_evaluate.py](../src/round100_temporal_prior_evaluate.py)，SHA a86ef89b1216e4dc597b301ef64d8a702d2fd2a44a3f972098b88afebadd9120。CLI：--sealed-manifest PATH --evaluation-request PATH --out NEWDIR。parameters() 直接沿 [共用契約](../src/round100_contract.py)；評分 request 需精確參數、來源閉包、計畫、九個保護檔與主研究 request 身分。JSON 由保存 SHA 身分讀取，不假設歷史文件都有 sidecar。

先驗三窗口、六個 completed NN、全 query 重載、模型／NPZ／來源 SHA、相同訓練列與八個共用陣列身分、原 R63 CONFIG、固定 date weights、full-fit query factor、原始 canonical→raw 映射、完整三支 median 與 v22/v24 保存向量，之後才唯一一次反序列化原始價格 pickle。模型與 NPZ 僅核 bytes，評分器不載入其狀態或 fit labels；pickle 整表反序列化限制有明確記錄。seen/new/missing 依原 outer fit，沒有改用 eligible 訓練子集。

三期依 development／confirmation／auxiliary，各報 T/C/v22/v24 完整與三組 native/base/reference/spatial 指標。T−v22、T−C 各 project/month 2000 次、兩個 MAE，共24區間；seed 與 stage/family 索引按計畫固定。C−v22、T−v24 完整點差保留。門檻包括三期 overall/weighted 改善、confirmation 至少0.5%、無季度或至少500列固定群退超5%；通過也不自動換提交，Private 未知。

[canonical 人工收據](round100_evaluator_selfcheck.json) 為 38 項成功，測得 0.297 秒；涵蓋封存拒絕條件、日期左右邊界、raw/canonical 軸、三支公式、R63 真實 metadata 介面、共同 labels/weights、兩個配對控制與研究門檻。首輪36項亦成功，但交叉審查發現 pure-X audit 執行順序其實為 auxiliary/development/confirmation；原先要求它等於模型順序會錯拒絕。已改為完整集合／數量驗證並保留原 audit 順序，再加入兩項人工檢查；不是實際 fit 或評分失敗。[首輪收據](round100_evaluator_selfcheck_attempt1.json)、[修正紀錄](../outputs/round100_evaluator_prefreeze/audit_order_revision.json) 與兩份來源均保留。

主 driver 的 prior helper 最終來源及凍結 request 由主線在全部元件定稿後檢查；本頁不冒稱未完成的六次訓練、真評分或 request 已核驗。已對 driver 初稿做來源／API只讀核對，包含三期原歷史 artifact 路徑皆有實際 authority；新分支診斷 SHA 缺口已由作者補 stage artifact map。
