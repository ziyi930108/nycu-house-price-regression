# R93 固定計畫：fit建案筆數的平方根折減權重

日期：2026-09-29（Asia/Taipei）。R89、R91、R92三期均未過門檻。下一項只測native單價樹的樣本群權重，原v22唯一提交／報告／CSV保持，Private未知。

## 假設、歷史與最強風險

大型建案提供許多相似成交列，可能在CatBoost切分與leaf估計中集中影響，對fit未見建案較不穩定。試以fit建案樣本量平方根折減其相對票重，檢驗新案與跨期泛化。R5已把密度折減列未採用備案（只折減超過100筆者）；本輪為其未實測延伸，不能說概念首次出現。R8／R89在reference/trend的等案票不等於native Pool已測；R11是query-X domain weighting。

反證與風險：競賽評的是交易列MAE，不是每建案等權。小案、單例及大面積高價單例的相對影響可能升高；不能把「較平均」當必然更正確，也不是直接削弱尾部。既有案名遮罩／去識別與部分權重實驗失敗，均需保留。

## 唯一改動與完整數學式

原R79單價CatBoost的target、95欄／23類別、depth8、2999樹、lr .065、seed42及其餘參數完全保持。A為原有效面積，d為同窗原保存R63 float32日期權重轉float64：

- 原w0=(A/100)*d，target=y/(A*10000)。
- 具名建案g以fit中city|district|project完整key計數n_g；raw q_i=1/sqrt(n_g)。缺案raw q_i=1。
- **固定保存原Pool總權重尺度**：c=sum_fit(w0)/sum_fit(w0*q)，w_i=w0_i*q_i*c。這是fit X/日期/面積所決定的全局一個倍率，不是query參數；不使用價格。
- 全fit sum(w)與sum(w0)以float64容差核對；保存q、c、最終w摘要與值SHA。
- 缺案只在raw q層中性1，仍共同乘c，不能宣稱最終權重完全不變。不是q/mean(q)，也不重新正規化原日期權重。
- 原TWD inverse/max投影保持。投影前新loss等價date*q*c加權總價L1/1e6，已偏離原date-only L1；這是本輪可否泛化的核心待驗部分。

不剪裁q、不設n門檻、不搜尋指數或權重強度。原w0與新w都要求有限正值；空fit拒絕。

## 建案與fit/query邊界

重用round5_project_mask.project_keys(X)，原三欄正規化fillna('__MISSING__')、str/strip、'', 'nan', 'None', '<NA>', 'NaT'為缺值，按案名缺失定缺案；city/district缺失但案名有效時仍是具名key，沿用原口徑。不額外推定「無」等字串為缺案。正規化後三欄包含'|'即拒絕，避免分隔碰撞；不得默默改群。

計數僅使用當期fit，不能加入query或test的出現次數。保存每列counts/q/weights與來源SHA，state存摘要／規則／hash；query推論不需要重新計數，也不接收label/weights。Synthetic必須證明更換query rows不改fit weights，改fit target不改weights；另檢查缺值、同案不同city/district、索引保留、pipeseparator拒絕、counts正確及Pool真收到新權重。

## 完整系統與三期流程

新native只替換v22單價槽：.75*new_native_TWD+.25*saved_LGB，沿同窗原reference/spatial，再與固定NN/log分支median。reference/spatial的票重不變；不併入R89趨勢、R91offset、R92depth6或額外新特徵。

先完成helper合成／獨立math/API核對，再凍結計畫／source與selfcheck SHA，單GPU三期development、confirmation、auxiliary全部跑；不因第一期差而省略其餘期。主driver完整來源／fit-query日期與索引／原95X與geo／原日期權重重播核對，確認fit端group數與新weights；舊v22 base/reference/spatial/median和完整score精確重播。各新模型保存、全query raw/TWD重載逐位一致，公開predict API與explicit inverse一致。主driver雖載canonical y，只將當期fit y傳入fit，query y只用score；不可說從未deserialize query y。

三期固定overall、原加權、新舊案、全部城市／季度、五真價帶、test-X組成估計、native/base/reference/spatial/median層指標。兩對照v22/v24，24 paired區間按既有2000次seed20260925建案／月份overall/weighted方案，不將真價帶診斷部署成test選模。

門檻與本批相同：三期overall及原固定weighted全優於v22，最新overall至少改善0.5%，再審新案／尾部／季度重大代價與區間。過門檻才考慮R90新切點六模型完整重建驗證，不自動全量fit/CSV/Kaggle勾選；未過保留失敗，不事後改成只看新案或等建案MAE。

## 限制與保存

窗口已多輪選型、假設由已見結果引導，不是盲測。bootstrap不校正選型偏差，不是Private勝率。GPU固定seed非位元決定性，舊控制未同期重訓，不能把微小差全部當純因果權重效果。

僅課程fit價格，不查內政部答案，不按Public搜參數或test ID改價。保存原模型、CSV、source、失敗、sha guard；新產物用outputs/round93，拒絕覆寫。helper/source可追溯與保存重載不等於全新環境端到端重訓。
