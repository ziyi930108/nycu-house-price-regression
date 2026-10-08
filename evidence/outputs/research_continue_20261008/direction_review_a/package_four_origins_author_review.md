# 四起點研究打包器：作者審查，2026-10-08

來源：`src/package_research_four_origins_20261008.py`，SHA256 `a7edf6d9e9d347b7afc3fbb8c59d9be7db24415dae32a7679ce5bec212e5f16a`。這是作者來源／人工 CPU 守衛審查，尚非獨立簽核、正式 package check 或已複製證據。

本次只建立工具，沒有執行 `--check`、`--apply`、`--refresh-manifest`、Git 操作、上傳、模型訓練、價格評分或原始資料／模型反序列化。正式複製由主執行緒在獨立來源審查和真實完成證據俱全後另行執行；R117 v2 真實 postfit 尚未存在時，工具必須拒絕。

## 固定範圍與保護

- 本機 GitHub 交付目錄 `deliverables/v22_github`，基準 HEAD 固定 `f596c4260d10144bdbc963d1c2a57519375a7ba0`，原 SOURCE_MANIFEST 為 3,229 會員、3,230 tracked。工具僅唯讀解析本機 HEAD／index，不呼叫 Git。
- 逐筆驗證全部舊 manifest 雜湊。除三個導航頁之外，3,226 個舊會員必須精確不變，包含先前 3,014 個 protected、379 歷史研究與 13 production source；舊 CSV／媒體／收據保持。原 manifest 本身可由受限刷新更新，舊版另存。
- 新增 R114／R115／R116／原 R117 aborted／R117 v2 的 source、plan、report、人工檢查、正式執行／封存／獨立 postfit 等有界聚合證據。可明確加入完成的 R118 可行性盤點與 root 獨立審查；R119 未納入。
- 沒有遍歷整個 outputs 複製。只複製固定名稱、exact reference 和受限 `--extra-file`；原資料、CSV、NPZ／NPY／PT／CBM／PKL、學習狀態、prior／prepared 私有資料夾與逐筆 JSON 價格／向量均拒絕。私有檔案可串流核 SHA，不能由此成為複製項目。
- 路徑必須 canonical relative POSIX，拒絕 traversal、symlink／junction 祖先；既有未列檔不能覆寫或自動收入 manifest。複製及生成文字均掃描 credential 與逐筆資料。部分複製失敗保留產物、拒絕靜默重跑，沒有宣稱原子交易。

## 真實完成與工程失敗

成功研究的前審、request、四個 stage／native 模型／五層、整批 seal、訓練成功退出、其後獨立 evaluate、postfit、實際 CPU 審查命令及各自 source／input 身分必須互相精確綁定。R114／R116 原 raw MAE 診斷中 false 的語意，另要求 root clarification／獨立 interpretation／binding；false 不得被解讀成 MAE 不讀 query y。

原 R117 的來源缺少動態父檔 R52，development 先驗與 NN 已完成，但在 metadata seal 之前工程失敗。原 source、request、模型、價格向量、失敗命令與 84.125 秒紀錄保留，沒有 seal／分數／CSV。這不是模型效果失敗。

R117 v2 另目錄全部四 prior／NN fresh 執行，科學方法保持；request、結果與獨立 postfit 必須精確記錄原 1 次 aborted native ＋新 4 次、總 5 次 R117 native，以及未載入／重用舊 partial 模型／prior。閉包是 51 Python ＋原科學 plan ＋工程 amendment，27 個舊 partial 檔逐 SHA 綁定。獨立 preflight 最初 .venv_torch 缺 pandas、import 即退出的 0.343 秒失敗與其後 .venv 成功審查也保留，不能混成新的模型 retry。

因此整個本批若實際成功完成 R114／R116／R117 v2，總計 12 個成功 native fits ＋原 1 aborted＝13 個實際 native fits、5 個新 prior builds；R115／R118 零 fit。先前 R111／R112 六 fit 與 R113 三 fit 分開，不把舊批混入本批工作量。

四 origin combined 仍是原三 August 門檻 AND 固定 Feb 額外起點，探索與強門檻分開。combined 失敗只接受具體 `--no-release`；探索通過但尚未全量發行只接受 `--defer-release`。現版不接收完成的未來 full release；若要收入其 final request／model／CSV authoring／獨立 final review，需另經來源審查擴充，不可默放行。v22 的選定與未知 Private 保持。

## 介面與人工檢查

`--self-check` 是純人工 CPU 檢查；`--check` 正式唯讀比對完整舊包與本批完成證據；`--apply` 明確複製；`--refresh-manifest` 只接收前次 copy_manifest 中的新增檔與保留原正文的 dated 導航更新。三個研究都需各自提供 no-release 或 defer 的具體理由。可選 `--postfit-execution ROUND=EXACT_PATH` 只允許該研究命名的實際獨立命令收據。

作者正式 CPU selfcheck 135 項、exit 0、0.109 秒，含缺／錯 postfit、Private／Public 非空、門檻非 AND、工程成本漏計／舊 partial 重用、固定前審巢狀 receipt、模型 artifact 缺失／額外／跨起點、Depthwise native 邊界、逐筆價格／向量／secret／CSV log、非法路徑、來源／導航覆寫及實際命令錯誤守衛。source memory compile 與執行前後 SHA 精確。收據：`package_four_origins_author_selfcheck.json`，SHA256 `6ecc5afabdf62599b7022f3c779eee6e477af3de61514ead78104e66c20ce969`。

先前 attempt1 source／117-check 收據保留為當時快照，不能證明現版。現版仍需第三方獨立來源審查；作者沒有執行真 package check，未證實所有尚未完成的 future schema 會在實際完備批次通過。可能的封裝錯誤須保留首次失敗證據、另版修正，不能改 ML 科學方法以通過打包。

限制：保存重載／雜湊不等於新電腦端到端重訓已完成；重用窗口有選型偏差，bootstrap 不等於 Private 勝率；GPU seed 不保證重新訓練逐 bytes 一致。此工具沒有 Kaggle、網路、額外點數或重置功能。
