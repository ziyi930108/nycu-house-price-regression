# R104有限結果核對

主線與另一子代理只讀已完成 source／aggregate／training metadata，未重讀逐筆array、未重新score／bootstrap。這是有限一致性審查，不是完全獨立重現。

- root與三期results逐欄相同；八模型計數、六學生同NPZ／十array身分／初始化／原日期權重、30epoch及全query reload均符合固定計畫。
- primary四項與mechanism四項皆false。T−C base三期overall及weighted都退；完整median兩期退、最早一期改善。native MAE不是本輪必要改善門檻，仍全部揭露。
- 共24條件區間，各2000有效／0無效。對v22：4偏改善、2跨零、6偏退；對C：4偏改善、4跨零、4偏退。不是Private勝率。
- development在base已比C退793,761.94元，不能將失敗直接歸咎R/S；median明顯限制退步。auxiliary由spatial比C退14,962.91元，翻成median改善30,621.90元，對兩控制各四CI全負，是不能稱「完全無效」的反證。
- T自己的訓練objective下降不代表泛化改善；C/T loss定義不同，不直接比较兩條數值。只憑這些aggregate不能斷言一定是過擬合。
- 教師與query模型年代／資料量的差異、Softplus及負等價residual是可能因素，尚無因果證據。負等價residual dev/conf/aux為1106／1292／44列，均保留。
- 學生子集只占原outer fit約51.3%／63.6%／13.2%，會影響對v22的解釋，但不能單靠樣本減少解釋T−C，兩者完全同列。

[固定計畫](round104_plan.md)；[完整結果](../outputs/round104/evaluation/results.json)。唯一提交v22，Private未知；沒有新CSV或Kaggle操作。
