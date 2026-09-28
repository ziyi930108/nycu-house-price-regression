# R92之後的單一研究方向：fit-only建案樣本量折減

2026-09-29（Asia/Taipei）。這是只讀歷史盤點與下一候選建議，不是模型結果或已凍結的R93實作。本次未訓練、沒有計算新候選向量／MAE，沒有讀取新query標籤或test答案；沒有修改任何舊source、state、CSV、selector或root交接檔。

**建議下一個有界實驗：在v22的R79單價CatBoost訓練權重，增加只由fit X建案筆數形成的平方根折減；測試跨建案穩定性。** 不動目標／95欄／預測還原／reference／spatial／外層median，不同時換seed、深度、epoch或融合比例。它不是保證改善尾部的修正，更不是事後按高價或ID調價。

重要歷史更正：**「建案密度權重」不是本次首次想到。R5已把它列第二備案，但保存決定明示未選入候選；未找到它在native CatBoost Pool實際訓練的完成紀錄。** 本次可稱未實測的後續方向，不能包裝成全新概念。主線接獲盤點後已表示可另立R93固定計畫，本報告本身不授權或啟動fit。

## 盤點範圍與R92時序

搜尋範圍是工作區已保存的round計畫／報告、早期optimization_round報告、model_selection_explanation及對應模型helper中target／sample_weight／project-key實作；特別追查「建案密度」「每案等票」「群組sample_weight」「平方根樣本數」「尾部」「遮罩」「domain weighting」。涵蓋R1–R92方法沿革，但不是逐個模型重新推論、不是所有檔案的逐行形式驗證，也不涵蓋未保存run。

|方法族|保存輪次與內容|對這次方向的意義|
|---|---|---|
|原始回歸、特徵與時間|R1–4：raw/unit/structural、近期權重、深度／LGB平滑、地址／案名前綴、時間正規化、tail routing|已經試过一般容量、位置與尾部切换，不能再只換名稱|
|識別依賴|R5整案遮罩、R17 Ordered、R18刪四識別欄、R42遮罩接地理模型|改特徵／boosting，均不是本次native建案筆數權重；R5另有未執行密度備案|
|地理與表示|R6–9、R13/16/21/25/26/30、R39/41：geo、POI、城市／前綴／網格／道路對／車位比值|沒有找到此次權重；有些停在純X或相容性，不能說全都完成fit|
|樹家族／容量／CTR|R14/15/20/23/24/27/31/32、R83/84、R92|與LGB/XGB、深度／步長／CTR／seed／樹數不同；不是再掃深度|
|目標／價格尺度|R12/19/22/44/46/79、R85/86/91：log、unit、Huber、價格權重、日期權重、sqrt面積、prior特徵／offset|已有多種表示；本次保持R79表示，只改各fit案的相對曝光|
|參考成交與空間|R8/33–39、R45、R82、R88/89|R8等案中位数与R89每案IRLS票只作用reference／trend，不是native Pool權重|
|NN與先驗|R47–67：raw/area、seeds、ratio prior、ramps、TabM-style、recency、SmoothL1、log-ratio、prior/support欄|沒有找到native案量折減；既有先驗按group OOF，不等於每案等權训练|
|組合與消融|R28、R35–43、R49–55、R66/68–78、R80/81、R87|固定均值／median、移除舊支、seed與層級診斷；不把新權重說成融合比例搜尋|
|新增時間窗基礎設施|R90|驗證工具，不是已完成新切點泛化證明|

收到本任務時，主線回報R92仍在跑、前兩期沒有穩定改善，因此本方向不以「R92全輪已否決」作前提。盤點期間於本地00:35查檔，發現[主流程收據](../outputs/round92/execution.json)已落地：UTC 16:30:36（本地00:30:36）exit0，478.015秒。本次沒有讀取R92三期完整metrics、generic或paired，也未對其作採用審查；完成訓練不等於完整審查結案。文件不把早先正在執行的狀態倒寫成早已完成。

## 與R5舊備案的確切差別

[R5研究文件](round5_research_features.md)第3節提出：具名fit案q=min(1,sqrt(100/n_project))，缺案q=1，最後除全fit均值；原raw總價／87欄模型及當時融合。文件末節明示主線只選整案遮罩，密度權重不列候選；[R5研究決定](../outputs/round5/research_decision.json)的unselected_research亦列Project-density weighting。

目前提議是在R79的95欄、單價目標及面積×日期權重上，對**所有具名有效案**使用q=n_project^(-1/2)，沒有「100筆以下不折減」平台。這是比舊備案更全面的群筆數折減，不可冒稱只是把已成功方案搬過來，也不可把未執行舊備案當成失敗實驗。是否值得做，應由一個固定候選的三期結果回答。

只在已保存source／plan中未發現該native权重完成實驗；有限搜尋不能證明任何未保存嘗試都不存在。

## 為何仍值得一個固定試驗

1. **誤差主體是普通交易、新案與跨期漂移。** [R85誤差盤點](r85_error_audit.md)顯示100–200m²占三期總絕對誤差62.24%／63.06%／67.64%；真價≥1億僅占3.95%／5.02%／3.13%。繼續只壓少數巨大預測，很難解決主要誤差，也可能破壞真實大額交易。
2. **交易列很多，不代表獨立建案很多。** [既有建案區塊診斷](v3_cluster_stability_report.md)的新案群為1,944／2,075／3,539個名稱地理群，但依列質量的有效群數約326.58／383.30／555.45。這不是統計獨立樣本數，只说明列數高度集中。R5純X預覽中504個至少100筆的具名案占具名fit列53.26%，亦提示集中；該預覽240,689列未套正式canonical去重，**不可直接當R93正式fit支持數或沿用其索引**。
3. **現有area×date權重未直接控制建案曝光量。** 大案若有很多相近房型，可能在有限樹的split／leaf優化中反覆主導共享結構。平方根折減讓同案樣本的重要性隨數量次線性增長，保留完整價格和有用案名，與永久刪欄、遮罩身份不同。這是待驗機理，不是已證明建案過擬合。
4. **只變動一支既有模型，成本與洩漏面可控。** 不需要新的價格prior、nested OOF模型、外部地圖或query-X密度估計；三次native fit即可比較完整流程。它優先研究穩定性，不宣稱直接解決高價尾部。

## 建議固定的數學契約

令d_i為原保存float32日期權重轉float64，A_i為原有效面積；u_i=y_i/(A_i×1e4)。在**同一fit子集**依city／district／完整案名取得n_g：

```text
q_i = 1 / sqrt(n_g)        # 有效具名fit案；n_g只數當期fit列
q_i = 1                   # 缺案或明定無效案，預正規化中性值
w0_i = (A_i/100) * float64(saved_date_weight_float32_i)
kappa = sum_fit(w0) / sum_fit(w0 * q)
c_i = kappa * q_i
w_i = w0_i * c_i
native_TWD = max(model_output * A_query * 1e4, 0)
```

主線已在[R93事前計畫](round93_plan.md)固定**保留原Pool總權重尺度**：以fit的area×原保存日期權重求kappa，令sum(w)在float64容差內等於sum(w0)。這不是q/mean(q)，不重算或重新正規化日期權重，也不令整個Pool均值1。保留總尺度不等於保留群質量、有效樣本數或優化動態。

缺案「中性1」只指正規化前q=1；正規化後缺案的列乘數c_i=kappa通常大於1，最終Pool權重並非完全等於原權重。原invalid date仍保有日期因子1，也不代表乘上area及c後總weight=1。這兩者都需分別保存與報告。

投影前，新的加權單價損失等價於：

```text
w_i * abs(u_i - z_i)
= (d_i * c_i / 1e6) * abs(y_i - A_i*1e4*z_i)
```

所以新的統計目標是date×建案因子加權的總價L1，**不再與原date-only總價L1相同**，更不是無權重的競賽交易列MAE。對面積／日期相同的案，n個相近樣本的總權重約由n改為sqrt(n)，不是每案完全等票。跨案日期與面積不同時，不能直接宣稱每案總權重正比sqrt(n)或都相同。

模型參數、target、inverse、95欄及原始資料保持；仍`.75*新native+.25*原LGB`→原同窗reference→spatial→與保存NN/log逐列median。不得乘c到query預測、reference修正或評分MAE；query新案不需估n_g，也不更新fit群字典。

## 可重用的key與missing口徑

最貼近現有流程的是[src/round5_project_mask.py::project_keys](../src/round5_project_mask.py)，它不讀價格，並檢查原87欄輸入；回傳字串key與missing-name旗標，保持原index。

|項目|既有行為／建議|
|---|---|
|三個欄位|address_city、address_district、完整project_name；不要換原district、單純案名或地址道路鍵|
|正規化|fillna('__MISSING__')→astype(str)→strip；把空字串、nan、None、<NA>、NaT字面值轉成__MISSING__|
|原始sentinel|house_features已將NaN/null等原始缺值正規化；新helper應收既有feature X，不重新讀原文並擴張詞典|
|缺案判定|舊project_keys只以正規化後project_name==__MISSING__判missing；city/district缺失仍可有具名key|
|索引|R5保留輸入index；round3_evaluate與v3_cluster_stability同值規則會reset index，使用時不可與未reset frame靜默對齊|
|不採用的fallback|v3_cluster_stability.cluster_keys會將缺案按地址／道路成群；不符合本次缺案各列q=1，不拿它直接數n_g|
|分隔符|舊key使用字面值`|`連接；freeze前應證明三個正規化欄無歧義分隔碰撞，或合成中拒絕碰撞，不靜默合併tuple不同的案|
|禁止自行加的判別|不把「無」、短案名或未列詞彙臨時當缺失，不做fuzzy merge／跨區別名統一，不以query誤差判定有效案|

「無效案」需在R93計畫精確定義。沿舊key的最小相容解讀是Python缺值／既有missing字面值；**若另把city/district缺失列列為無效、令q=1，這是額外保守規則，必須明載**，不能說舊helper本來已排除。合法非缺失名稱與地理組合也只是資料proxy，不保證唯一實體建案；同名可合併、別名可分裂。

此key只供fit權重；scorer原seen/new、reference與spatial鍵不改。即使日後純X發現解析限制，也不能就地修舊key使v22或其他輪的SHA失效。

## 最強反證與可能失敗方式

- **實際評分按交易列，未必按建案平衡。** 大案的多戶可能正是未來市場的正確列占比，不是重複垃圾資料。折減它們會增加偏差、傷害常見住宅与已見案；不能把評分偷偷改成每案平均使候選看似進步。
- **高價單例可能被相對放大。** 少量案、缺名、缺位置如果保留q=1，經保留全fit Pool總權重的倍率kappa後，比大型住宅案的相對權重更高；極端面積仍乘在Pool權重內，raw／unit極端標籤都未刪除。這可能惡化大額錯誤而非抑制它，故不是尾部robust loss或winsorization。
- **既有權重與識別方法未必有效。** [R11](round11_report.md)的純X重要性權重通過工程支持檢查，development整體仍退步3.8077%、新案退步4.5945%；[R18](round18_report.md)刪識別欄後新案也退步1.3706%。這些是不同機理，但足以反駁「減少記憶／做權重就會改善新案」的泛稱。
- **大權重集中可轉移到少數稀有案。** 原每列權重、群權重質量、sum(weight²)與群mass ESS都應比較；ESS是描述量，不是獨立樣本數或Private勝率。新的平均c=1不等於Pool均值保持，也不證明變異降低。
- **樹與完整組合會互動。** R85某些native層改善而median退步；[R91](round91_independent_review.md)最新native/base改善，但reference/spatial退步、median再微改善。新權重的單支收益不能代替完整三分支驗證。
- **不能把所有尾部當異常。** R91的20–50m與50–100m三期都改善、最舊億元群亦改善；開發／最新億元群卻退步。R85盤點顯示少數億元群只占部分總誤差，單靠尾部説明不了所有跨期差異。
- **多輪使用窗口仍有選型偏差。** 新方向本身由先前誤差啟發，不是盲選；固定三期、配對bootstrap或新source名稱都不能消除這點。GPU固定seed也非位元重訓確定性，原v22控制未同期重訓。

## 最小必要實驗與停止方式

1. **先freeze一個純X權重契約。** 正式同窗canonical fi、87欄、key與invalid規則、q公式、kappa、原area／保存日期權重相乘順序、參數与source閉包先記錄。純X準備只算fit，無需query X、y或Public；固定三期都算，不因某期集中度不理想而搜尋指數／100筆平台／cap。
2. **先作bounded synthetic與資料身分檢查。** 同案n一致；相同n的不同案相同q；單例／缺名／invalid的q=1；缺名不可合成巨案；兩個同名跨城市／區案分開；空字串及各sentinel一致；pipe碰撞處理明確；排列等變、index精確、JSON重載；新增query不改fit任何state；n≥1、q/c/w正有限；target／95欄／date32 bytes不變；FakePool直接驗w乘積，metadata與實際傳值一致。全體missing的synthetic應得到q=c=1、w等於R79。
3. **只做三次native fit。** 每期一支，2999/depth8/lr.065/seed42等保持R79；新weights不可與R92 depth6、R91 offset或R89群斜率混搭。保存model/state/weights與全query raw/TWD重載一致。三期全部完成，不按首期先挑對自己有利的年份。
4. **驗完整組合與固定門檻。** primary v22、secondary v24；整體／原固定加權、新舊案、四季度、所有城市、原真價五带；原生→base→R→S→median逐層，報告高價尾部對整體差的有號貢獻。真價群僅診斷，不成為query分流或刪列。24個建案／月份配對區間全部保留，不挑跨零或有利抽法。
5. **仍沿本批門檻。** 三期整體及固定加權全優於v22，最新整體至少0.5%，另審新案／高價／季度代價，才考慮主線已固定的新切點重建。若未通過，保存失敗；不把指數換成1/3、加100筆平台、改缺案規則或改正規化口徑後冒稱原假設。這不是授權全量fit或CSV，v22選定不改。

最少新增訓練量為三個CatBoost模型，沒有新NN／prior／reference/spatial fit；CPU權重計數可直接完成，但本次未執行，不填虛構耗時或支持數。真正成本須由新launcher／fit execution實測。若工程preflight失敗，应保存技術原因，不冒稱MAE否決。

## 關鍵來源身分

- R5未執行備案文件：`9f5e849f71340e950065a49f1ff2ae7cc05d80e5e03fa233a609fae8fda23033`。
- R5研究決定：`251035988e77ba795b464bc42c77d76aeb6dcdeb723b0595c686fef87116c95a`。
- R5 project key helper：`78509888884c658510108887a54a99356515ac381569662363db74b4315c56d0`。
- 原scorer key：`e00b2b09d4f0ed8d6fbd547f00ebbb2d8096136a567a56eb85ddacc6ec1f4271`。
- cluster fallback helper：`46622f859660e40bd78fbbd612db53b5a9bd5069f55eb9e7b169e19cc134d1c9`。
- R91獨立審查：`a6a71c85ce432b89ed0ce489d26f3ffa895d703171a79535fcd342dbaf0a3b43`。
- 本次僅確認R92主程序exit0的execution：`c41dc0e9fc6f371438a065546f874745bc5f9aa824fdaa724b91d1eb91206fc9`；不代替尚未在本盤點審閱的全部結果。

本報告只提出一個可否證的穩定性方向；沒有證據可聲稱將優於未知Private，沒有發布新價格、查外部成交答案或改動原v22交付內容。
