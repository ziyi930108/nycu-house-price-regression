# R106：四個身分類別的聯動 dropout

2026-09-29。R106六次NN與三窗評分完成；完整T−v22 MAE為 **-63.26／+2,439.67／+422.80 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**

C依原R63逐欄獨立0.1遮罩重訓；T令案名、完整門牌、棟及戶、棟別共用案名原mask。仍抽完整23欄，其他19欄與eval不變；同144數值欄、原prior、日期權重、初始化及30epochs。新C不要求等於保存v22；六fit全部重載封存後才隔離評分。

|窗口|T−v22整體／加權|T−C整體／加權|C−v22整體／加權|
|---|---:|---:|---:|
|development|-63.26／-64.48|-63.26／-64.48|+0.00／+0.00|
|confirmation|+2,439.67／+2,627.10|+2,439.67／+2,627.10|+0.00／+0.00|
|auxiliary|+422.80／+593.23|+422.80／+593.23|+0.00／+0.00|

Primary：False；機制對照：False。24 CI：0全負／24跨零／0全正。所有五層及分群反證見[完整報告](../evidence/reports/round106_report.md)。
六fit 94.642秒已含main 167.750秒；scorer 51.422秒。launcher 224.157秒含主流程，不能重複相加；非聊天工時。

unknown embedding原已訓練；聯動不遮numeric prior或其他地址，並非真新案模擬。單seed、重用窗口與2023地理可用性限制保留，CI不是Private勝率。Public仍為使用者回報1,998,568.5706元。
依要求停止後續研究，不開R107或背景訓練；再次開始須使用者指示。正式v22報告與簡報保持。

[計畫](../evidence/reports/round106_plan.md) · [結果](../evidence/outputs/round106/evaluation/results.json) · [trainer](../research/src/round106_joint_dropout_train.py) · [driver](../research/src/round106_joint_dropout_experiment.py) · [scorer](../research/src/round106_joint_dropout_evaluate.py) · [複製清單](../evidence/outputs/research_snapshot_r106_20260929/copy_manifest.json)
原碼、人工attempt與聚合收據保留；不包原交易表、逐筆array、模型及完整learned states，部分路徑仍依賴本機工作區，非完整重訓備份。
發布前base bde2a9221ea6beaad2297e97e12900e322b3872c；當前身分以Git main／SOURCE_MANIFEST為準。
