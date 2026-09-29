# R105：固定原生平均

2026-09-29。R105固定原生50/50平均與三期隔離評分已完成；完整T−v22 MAE為 **-317.81／+2,010.70／-6,060.66 元**（development／confirmation／auxiliary）。研究門檻未全通過，不採用；**唯一提交仍v22、Private未知，無新CSV。**

只測一次float64 `M=.5*P+.5*Q`；P是v22原prior NN，Q是R102T。先重播兩端五層exact，再沿原base／reference／spatial／median。C指原R102T完整端點。0新fit、0原生模型或NPZ載入，但有載入已學R/S狀態；三窗全部封存後才獨立評分。

|窗口|T−v22 MAE|T−C MAE|T−v22 weighted|T−C weighted|
|---|---:|---:|---:|---:|
|development|-317.81|-1,252.15|-298.05|-1,242.38|
|confirmation|+2,010.70|-5,208.66|+2,259.80|-5,806.24|
|auxiliary|-6,060.66|+3,559.24|-5,086.64|+3,167.48|

Primary：False；補充端點門檻：False。24 CI為6全負／18跨零／0全正；完整門檻、分群、五層及區間見[報告](../evidence/reports/round105_report.md)。
實測main 54.328秒、評分66.375秒、摘要0.172秒；launcher 125.438秒包含主流程，不可重複加總。這不是聊天工時。

反覆窗口並非盲測；沒有多seed方差估計，平均不保證完整流程受益。2023非價格地理存在歷史可用性限制；bootstrap不是Private勝率。唯一提交v22與使用者回報Public 1,998,568.5706元不變。

[固定計畫](../evidence/reports/round105_plan.md) · [結果JSON](../evidence/outputs/round105/evaluation/results.json) · [driver](../research/src/round105_native_average.py) · [scorer](../research/src/round105_native_average_evaluate.py) · [複製清單](../evidence/outputs/research_snapshot_r105_20260929/copy_manifest.json)

[下一方向盤點](../evidence/reports/round105_next_direction_inventory.md)尚無新計畫或fit。[R104歷史頁](RESEARCH_UPDATES_20260929_BASE_OBJECTIVE.md)保持原bytes；正式報告及簡報不變。快照排除交易表、逐筆向量、模型及learned states，部分原碼路徑依賴本機布局，不是完整重訓備份。
發布前基底：374d488b2d1cfac5833bf685016522ad683d8949；實際身分以Git main／SOURCE_MANIFEST為準。
