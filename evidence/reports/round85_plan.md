# R85 固定實驗：平方根面積尺度回歸

日期：2026-09-28。使用者最新授權重新改良回歸模型；原v22及上台簡報保留，新研究不代表自動變更最終Kaggle選擇。

## 唯一改動與假設

有效面積A沿用R79：正且有限建物面積，否則車位面積，否則1。固定F=sqrt(A/100)，目標t=(y/1e6)/F，Pool權重F×保存R63日期權重，預測max(z×F×1e6,0)。不掃描面積冪次，不裁掉大面積或高價樣本，不設test價格上限。

動機：原單價表示將總價完全除以面積。平方根尺度是總價與單價之間的固定折衷，可能降低不同面積對逆轉換的放大，同時保留面積關係。這是待驗證假設，不能由原有殘差盤點視為已證明。未截零前 F*d*abs(y/(1e6*F)-z)=d*abs(y-1e6*F*z)/1e6；仍對齊日期加權總價MAE。共同權重尺度與模型實際擬合可能影響正則化，不宣稱只是無效果的代數換算。

沿用95欄/23類別、CatBoost GPU MAE、2999樹、depth8、learning_rate .065、seed42及原全部其他参数。訓練無query eval_set/early stopping。日期權重直接沿該期已保存float32值，不重算或重正規化。

完整候選：base=.75*new_sqrt_TWD+.25*saved_LGB_unit_TWD，按該期保存reference與spatial重算此分支，最後median(savedNN,savedLog,newBranch)。不改其他分支與比例。

## 事前固定驗證

沿R79 canonical train與三期：development 2022-08-11至2023-08-10、confirmation 2023-08-11至2024-08-10、auxiliary 2021-08-11至2022-08-10；fit嚴格早於query起日，不混valid。三期全部跑完，不因第一期結果挑剩餘期別；每期新增一次fit，先不產test預測或CSV。

對照v22/R79為主要、v24/R83為次要。不將舊控制重新訓練，保留GPU訓練變異未隔離的限制；若效果很小，不能聲稱已證明因果改進。合法重用同切分、同SHA的特徵、地理、NN/log、LGB及reference/spatial；核對索引、fit日期、特徵、標籤與保存狀態身分，不能拿最終模型回測。

報告完整組合整體/固定加權MAE、新舊建案/季度/城市、五價帶、固定test組成估計與模型各層。配對沿建案群集與三月循環區塊、2000次、seed20260925，兩對照各12區間；不稱Private勝率。所有歷史窗口曾重用，這不是新盲測。

## 保存與決策

新目錄outputs/round85，新helper/driver；所有旧模型、CSV、selector與歷史決策不動。固定計畫與source/input SHA在fit前保存，每個模型保存CBM/feature_state/實際目標與權重metadata、全query重載一致核驗、日志及真實秒數，失敗不覆蓋。query標籤只交評分函數。

先比較三期效果大小、群組代價及配對不確定性，決定是否值得全量fit成新試用版本；不只依原生單模型或單一期挑選，亦不以某一期必贏作臨時門檻。若有價值，再另寫全量fit與CSV計畫後執行；不能保證Public或Private改善。不查內政部成交答案、不搜尋Public参数、不按ID改價。
