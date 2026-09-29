# R106：四個身分類別共享訓練 dropout

2026-09-29。在本輪讀取真NPZ／逐筆預測／真價或fit前固定。依R105後有限盤點，只改一個訓練表示設定，不搜尋欄位、mask率、seed或混合權重。v22仍唯一提交，Private未知；已重複使用窗口，不是新盲測。

## 假設與唯一差異

原v22/R63 prior NN已對23個類別逐欄獨立施加0.1 dropout。C維持這個方式；T對project_name、address_raw、building_and_unit、building_block共用project_name原mask。每次forward仍抽取完整23欄mask，再複製這一欄到其餘三欄，不增加隨機抽樣，不更動其他19欄mask。位置由凍結encoder欄名順序核定，不能猜索引。eval時都不人工遮罩，原EmbeddingMLP結構、初始化、head、hidden dropout與參數量不變，不monkeypatch舊來源。

動機是訓練偶爾同時失去相互替代的身分線索。只談人工事件，四欄全遮機率從0.1^4變為0.1，每欄邊際0.1不變；原unknown embedding早已受訓。數值prior、其餘地址與地理欄不遮，這不等同真新案，也不是去識別。R42固定整案遮罩的新案三窗皆退、R18刪欄開發退，保留相近反證；共享mask亦減少部分遮罩組合多樣性，不能預設更好。

## 原資料與六次對照訓練

直接重用同切點R63九鍵prepared：144数值／23類別、原group-OOF fit prior／full-fit query因子、原日期權重與全部fit列。非R102的145欄或時間prior。C/T逐鍵完全相同、原輸入不寫回；驗prepared／preparation／encoder／原fit/query indices與來源SHA。只允許fit標籤；query真價不進trainer或driver，不讀test。

沿原seed42、30epochs、batch2048、架構[512,256,128]、原AdamW/cosine與weighted total-price L1；fit-global日期權重不重算、不batch renormalize，初始化原median ratio，無clip或early stop。三窗按development、confirmation、auxiliary各C後T，共6NN fit，主線單GPU串行，不自動重試。每支模型全部query保存／模型重載預測exact；同seed不是CUDA重訓逐位確定保證。新C對保存v22的差異完整報告，不强求相等。

沿原年度：development query2022-08-11～2023-08-10，240,316 fit／81,234 query；confirmation 2023-08-11～2024-08-10，321,550／133,697；auxiliary 2021-08-11～2022-08-10，134,805／105,511。四canonical/raw indices與日期核對，同期A/L/H/U及reference/spatial重用；0新tree／encoder／prior／R/S build。2023非價格geo晚於早期origin，非fully historically-as-of。課程價格標籤為來源，禁止內政部對答案。

原v22五層先重播exact；新C/T native總價沿原 B=.75*(.5*A+.5*native)+.25*L、reference、spatial，最後median(N,H,U)，操作順序不變。R/S state與fit/query框架不可變。保存native/base/reference/spatial/median、診斷、模型、配置、欄名／mask與輸入身分、來源與耗時。全部三窗6fit及封存exit0後獨立scorer才反序列化raw價格整表、按原索引抽query y，不另保存標籤。

## 事前評分與判定

原v22五層metrics重播原R102結果exact；新C/T五層全評，完整median是主要決策，保留C-v22重訓偏移。全outer-fit定義seen/new；原overall與固定加權 seen=.36617593949822613、new_nonmissing=.6338240605017739、missing=0，所有城市／月份／day-quarter／五價帶／尾部全列。

Primary沿原三窗完整overall/weighted都勝v22、confirmation overall至少0.5%、所有季度或至少500列固定群无>5%退步。另列C/T機制對照：T完整overall/weighted三窗皆勝C、對C也無季度或≥500群>5%退步。native改善不是門檻，不能隱去native反證；單seed對照不證明跨seed因果或Private效果。兩項皆過只支持後續研究，不自動全量fit/採用/CSV/Kaggle。

T-v22及T-C各三窗×project/month×overall/weighted，共24paired CI，每類2000replicates，seed20260925，stage索引0/1/2、family project0/month1、month block3×4。保存replicates及所有正反例；CI不是Private勝率，不涵蓋累積選型偏差或未來分布。

source／plan／原輸入及9保護SHA先凍結；只必要人工mask/RNG/eval/loss與metadata接口檢查。保留任何失敗原件。若未通過，不改mask率、欄組、seed或query分流追分；正式v22CSV／簡報／選定不變。完成後更新報告、HANDOFF、日誌與文件包，私人GitHub依既有授權同步；快照不稱完整重訓備份。
