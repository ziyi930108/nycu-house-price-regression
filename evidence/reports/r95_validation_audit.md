# R95 工程恢復獨立審查

2026-09-29（Asia/Taipei），審查者 /root/r85_validation_audit。**限定增量審查通過，未見本次恢復的source／reader blocker；可由主線凍結新request後執行原R94固定三期。** 本次未執行driver、GPU或模型，未載資料陣列、query價格或test。

[新driver](../src/round95_experiment.py) SHA：6893ee155b94d3bc5fd218695c903ba29cd34dc210432042615807d3f1c75912。保留[舊driver](../src/round94_experiment.py) SHA ca1823069d1310409bfb7a8a04c09177ed2ca5e15496f66a99b769ac70948563。[增量diff](../outputs/round95_launcher_20260929/driver_engineering_diff.patch)已完整讀取；只有R48 encoder reader、執行編號／路徑／輸出key與恢復metadata變動，沒有改訓練數學或資料切分。新source AST通過。

唯一行為修正為 read_json(artifact(48,s,encoder_state_path))：沿原R48 results的artifact SHA驗證JSON，再讀取；仍須與R47 encoder完整內容及兩個檔案SHA相同。artifact已在正式開始前綁定原authority，沒有新建假sidecar或刪身分guard。

本次已實查 **27個仍使用locked的JSON＋27個sidecar**，不只查JSON存在：共用small()的2個plan＋7個results＝9份；每期R63 preparation、R47 result／encoder、R60 prior preparation／state、spatial summary＝6份，三期18份。每個sidecar實際內容均等於對應JSON的SHA。另3份R48 encoder確實無sidecar，但檔案SHA各與原R48 stage artifact authority相同。[逐檔核對收據](r95_validation_audit_checks.json)保存全部路徑及SHA。

[科學計畫R94](round94_plan.md)仍為 c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a；trainer v2仍為 ae197952be777f3e9e2fdb66458887d2a3885fd736b75dd3677eeae43cfeef83。原47項CPU／v2 GPU smoke與固定CONFIG不變，本次未重跑。R95[恢復計畫](round95_plan.md)另納launch freeze；模型公式、OOF／日期／factor、三期順序、30epoch、seed42與門檻沿原R94。

R94[失敗收據](../outputs/round94/execution.json)為7.813秒exit1，停止於第一期encoder讀取、尚未呼叫真fit。原preflight只核JSON存在及artifact authority，**漏抓read_locked要求獨立sidecar的執行契約**；这是本代理原審查的遺漏，不能把當時「無blocker」改寫成已知。原source、失敗輸出及[舊審查](r94_validation_audit.md)保持原bytes。R95恢復未依query分數選參數，不把這次工程失敗當模型表現不佳。

範圍仍是source／實際檔案reader契約核對；真實encoder／OOF／向量重播、三fit與重載是否成功，須等R95 execution。Private未知、v22唯一提交與既有CSV不變。
