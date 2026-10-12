# 六基底模型完整訓練可行性盤點（只讀來源／收據）

課程 PDF 沒有明文要求「一鍵從原始資料重訓六模型」。第15頁要求 GitHub
至少提供 README、程式、設定、requirement.txt 與報告；第19頁的截圖要求
用課程 train／valid 訓練、test 預測、MAE 分析和程式交付 Classroom。
它也沒有指定 main.py、禁用套件、自己寫反向傳播或固定特定切分。
本次重新目視第15、19頁；沿用前次完整27頁的課程審查。

「沒有一鍵要求」不能推論成「完整訓練程式已齊」。在本次 HEAD
`2fcf5cda93c1ff82d63c5b36d5aa2a87a70a57ce`，research/src 只有70支追蹤的
Python；本機checkout實體502支包括忽略與未追蹤檔案。原始六基底入口、
特徵／地理準備、動態 NN trainer 的73支保守 static closure 中，有22支
不在這個 HEAD。它們被 `.git/info/exclude` 忽略，所以「本機看得到」不代表
乾淨 clone 取得得到。兩個原 config/schema 也不在 HEAD。

已將22支原來源與2份原設定／schema逐bytes複製到新的 `source_additions/`，
由 `source_additions_manifest.json` 列來源、repo目的路徑、完整SHA。原repo、
原始程式、SHA拒絕條件、模型、價格state、NPZ、CSV均未改動。這24個補充
檔案是待root審查發布的來源補齊，不是已發布或完成端到端重訓。

## 六模型及實際資料筆數

逐模型入口／命令證據／完整設定／prepared生成依賴見 `six_models.json`。
不能將組合driver當作native trainer：R79 final composition重播向量；真正
C_unit入口是 round79_final_unit。原 L_unit 是 train_final 的
lightgbm_l1_unit63，不能誤換成 R2 final_spatial63。

六個原生 full-fit 收據都記467,221筆；最終推論query為47,073筆：C_raw、L_unit使用
`reports/final_training.json`；C_log用R46 fit_metadata；C_unit用R79 metadata；
N_area與R107 N_prior用各自 training.json。這是逐一讀原收據，沒有只套用
最新canonical筆數。455,784＋11,994＝467,778是未去重相加數；train去529後
455,255＋11,994＝467,249，再去28＝467,221。
C_raw／L_unit原final_training只記fit，沒有query_rows欄位；不能把後續
47,073列推論證據說成該原訓練收據也記錄了query欄位。

沒有收據顯示這五個舊基底直接用467,778筆預訓練。因此目前報告第4章的
清理筆數及最終467,221敘述沒有被這次盤點推翻；不需為假設問題修改文件。
本次沒有讀實際indices／labels或重算去重，不能額外認證六個訓練row-ID
向量彼此逐bytes相同。舊原始metadata、source guard與prepared身分仍是證據。

## 仍未補成完整raw→六模型執行的實際缺口

1. 六個native訓練核心與所需新補充來源可保存，但舊final driver含研究階段
   完成收據、固定歷史SHA、固定路徑及拒絕覆寫。這些是歷史程序，不可刪掉
   guard就稱可攜一鍵訓練。research/src內原程式的ROOT也按原工作區布局計算。
2. R107只訓練N_prior，讀原R63九鍵prepared NPZ。N_area同樣讀R55 prepared；
   必須從原始合法標籤重新建立自己的encoder／量化數值／OOF prior／日期權重，
   不能把保存NPZ、價格prior或查表當成raw輸入生成已完成。
3. R6／R39道路地理需要合法非價格地理快照、地址解析、道路index與matches。
   僅提供程式不會使這些資料自動存在；沒有從空環境重建、重新核對全部列序。
4. R2成交reference、R39 spatial價格state必須用這次合法fit標籤和對齊後native
   預測重新建立。保存舊state與五條向量可支持推論重播，不能支持六模型新fit。
5. 所有raw清理／87欄特徵／95欄geo／NN encoder／OOF prior／六native／三分支
   R/S／median／CSV尚無一個已實跑且不依賴舊價格預測的完整重訓執行鏈。

## 最小後續補齊方案（未執行）

先發布來源補充與原設定，說明它們的入口和前置條件。再於新路徑建立一個
固定配方的乾淨訓練runner，接受課程原四表與已核合法地理來源，不覆寫原
研究資料夾。清理重建cache及87欄特徵，固定full-fit IDs與95欄geo，重建
自己的類別／數值encoder、五折group OOF prior、query fit-only prior與日期權重。

沿原六種價格目標與超參數訓練3個CatBoost、1個LightGBM、2個NN；單GPU串行，
每個native真OS退出、保存重載全query核對後才組合三支。重建R/S價格state，
最後median與整數CSV在新路徑輸出。此為6次native研究模型fit，另含encoder、
OOF／reference／spatial等fit-state建立，不能籠統稱只6個無額外成本步驟。

先做source／製造fixture／完整輸入preflight，再由root批准真full training；
不在盤點工具內訓練。新模型的數值或SHA不同時保存原因與差異，不能改價
追逐原CSV。全部成功後才能寫「已重訓」，也不自動取代已選v25或Kaggle選定。

已完成的v25六模型完整forward屬保存fit狀態的推論重現；另有同機新套件
環境檢查由root協調。本盤點不把它們算成全六模型重訓；Private仍未知。
