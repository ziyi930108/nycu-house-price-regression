"""Build the round-three experiment report after the frozen confirmation run."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from model_common import ROOT
from prepare_data import DATA_DIR, sha256
from round3_journal import record

def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,x):
    (ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    dev=read('outputs/round3/evaluation/development/results.json')
    con=read('outputs/round3/evaluation/confirmation/results.json')
    frozen=read('outputs/round3/frozen_candidates.json')
    assert frozen['selected_candidate']=='v2_control'
    assert not con['selection']['final_accept']
    assert con['evaluated_candidates']==['v2_control']
    source=read('reports/run_manifest.json')['input_sha256']
    actual={name:sha256(DATA_DIR/name) for name in source}
    assert source==actual
    submission=ROOT/'outputs/submission_v2.csv'
    digest=sha256(submission)
    assert digest=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
    assert digest==sha256(ROOT/'runs/v2/outputs/submission_v2.csv')
    import pandas as pd
    import numpy as np
    from prepare_data import ID,TARGET
    csv=pd.read_csv(submission,encoding='utf-8-sig')
    sample=pd.read_csv(DATA_DIR/'test-reindex-test-sample-submission-v2.csv',encoding='utf-8-sig')
    assert list(csv.columns)==[ID,TARGET] and len(csv)==47073
    assert csv[ID].equals(sample[ID]) and csv[TARGET].notna().all() and csv[TARGET].gt(0).all() and np.isfinite(csv[TARGET]).all()
    checks={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':actual,'original_inputs_unchanged':True,
       'recommended_submission':'outputs/submission_v2.csv','submission_sha256':digest,'submission_unchanged_from_frozen_v2':True,
       'rows':len(csv),'columns':list(csv.columns),'sample_id_order_exact':True,'positive_finite_predictions':True,
       'new_submission_written':False,'public_mae':2883250.4138,'private_mae':None}
    write('reports/round3_final_checks.json',checks)
    names={'v2_control':'v2 對照','recency_blend':'近期資料加權混合','regional_trend':'收縮後的地區趨勢','recency_regional':'近期加權＋地區趨勢'}
    base=dev['metrics']['v2_control']
    rows=[]
    for name in names:
        m=dev['metrics'][name]
        rows.append({'candidate':name,'overall_mae':m['overall_mae'],'weighted_mae':m['weighted_mae'],
             'overall_improvement_percent':100*(1-m['overall_mae']/base['overall_mae']),
             'weighted_improvement_percent':100*(1-m['weighted_mae']/base['weighted_mae']),
             'eligible':None if name=='v2_control' else dev['selection']['candidate_checks'][name]['eligible']})
    summary={'round':3,'outcome':'retain_v2','new_submission':None,'recommended_submission':'outputs/submission_v2.csv',
      'public_mae':2883250.4138,'private_mae':None,'new_development_candidates':3,'development':rows,
      'development_selection':dev['selection'],'confirmation_control':con['metrics']['v2_control'],
      'confirmation_note':'Only the control was evaluated; no new candidate qualified, so this is a baseline diagnostic, not evidence of an accepted improvement.',
      'confirmation_historically_untouched':False,'test_weights':dev['target_group_weights'],
      'original_inputs_unchanged':True,'submission_sha256':digest,
      'evidence':{'development':'outputs/round3/evaluation/development/results.json',
       'confirmation':'outputs/round3/evaluation/confirmation/results.json','frozen_candidates':'outputs/round3/frozen_candidates.json'}}
    write('reports/round3_results.json',summary)
    record('008','固定參數重新訓練','在開發年度前的 240,316 筆資料，重新訓練原 CatBoost、原 LightGBM 與近期加權 CatBoost。',
       '用相同時間切分比較，避免沿用看過未來資料的模型。',
       '原 CatBoost 2,999 輪、LightGBM 3,999 輪；近期加權固定 2,800 輪，半衰期 730 天、最低權重 0.1。近期單模型 MAE 1,824,573.08，較原 CatBoost 1,728,642.53 差。',
       '訓練沒有 eval_set、early stopping；所有日期權重只由訓練期決定；完整保存參數、模型、預測、來源與 SHA。',
       ['outputs/round3/baseline/development','outputs/round3/recency/development/metadata.json'])
    details='；'.join(f"{names[r['candidate']]}：整體 MAE {r['overall_mae']:,.2f}、加權 MAE {r['weighted_mae']:,.2f}" for r in rows)
    record('009','統一候選比較與凍結','對三個預定新版本與 v2 對照，計算同一年度、同一批 81,234 筆資料的誤差。',
       '僅保留跨建案組別與預測季度表現穩定的改善。',details+'。三個新候選全未通過；自動選擇 v2_control，已保存不可覆寫的凍結檔與 SHA 收據。',
       '未按結果降低至少改善 1% 的門檻，也未追加係數搜尋；確認年度模型於凍結後才訓練。',
       ['outputs/round3/evaluation/development/comparison.csv','outputs/round3/frozen_candidates.json','outputs/round3/frozen_candidates.sha256'])
    cm=con['metrics']['v2_control']
    record('010','較新年度基準回測','只用較新年度之前的 321,550 筆資料訓練 v2 對照，預測後續 133,697 筆。',
       '新候選全部淘汰後，補足 v2 在另一年度的泛化診斷。',
       f"整體 MAE {cm['overall_mae']:,.2f}，三組加權 MAE {cm['weighted_mae']:,.2f}。此段只評估 v2，沒有重選新候選。",
       '來源、參數、索引及凍結收據通過核對；沒有用本段分數再調參。歷史標籤曾使用，不冒稱完全未見測試。',
       ['outputs/round3/evaluation/confirmation/results.json'])
    record('011','資料處理與失敗嘗試成冊','整理缺失值、重複資料、日期／面積／地址／樓層特徵、目標轉換與全部採用／未採用措施。',
       '讓作業能說明每一步做了什麼，以及為何保留或淘汰。',
       '87 特徵為 21 類別＋66 數值；獨立地址修正器已實作但本輪價格模型未採用；兩種尾端抑制規則都使回測惡化；完整表格與程式保留。',
       '不把實作等同採用、不省略失敗實驗、不連線查詢內政部成交答案。',
       ['reports/round3_preprocessing_record.md','reports/round3_overfitting_controls.md','outputs/round3/audit/tail_audit.json'])
    record('012','最終選擇與檔案檢查','保留原 submission_v2.csv 為本輪建議提交版本，不產生改名但相同的 v3 CSV。',
       '本輪沒有足夠證據支持新模型，應優先守住事先訂好的防過擬合規則。',
       '四個原始輸入 SHA 未變，v2 CSV 與封存檔逐位元相同；47,073 筆、兩欄、ID 順序及正值檢查通過。Public MAE 2,883,250.4138，Private 未知。',
       '只封存第三輪實驗，不改寫前兩輪模型；不把本機結果當作 Private 保證。',
       ['reports/round3_final_checks.json','reports/round3_results.json','outputs/submission_v2.csv'])
    lines=['# 第三輪優化結果與防過擬合紀錄','',
      '**結論：保留 `submission_v2.csv`。本輪三個新候選都未達預先訂好的接受門檻，因此沒有發布 v3，也不需要重交相同檔案。**','',
      '使用者已確認 v2 的 Kaggle Public MAE 為 **2,883,250.4138 元**。Public 約佔測試資料 30%，Private 約 70% 才是最終評分；目前沒有 Private 分數。','',
      '## 為何更換驗證方式','',
      '原 valid 的地理建案組別為已見 10,628 筆、新建案 26 筆、建案缺失 1,340 筆；test 為已見 17,237 筆、新建案 29,836 筆、缺失 0 筆。新建案佔 test **63.38%**，原 valid 表現不能直接代表 test。test 最多超出訓練末日 355 天，因此本輪採用接近一年的時間回測。','',
      '開發段：只用 2022-08-11 之前 240,316 筆訓練，預測 2022-08-11～2023-08-10 的 81,234 筆。確認段：只用 2023-08-11 之前 321,550 筆訓練，預測 2023-08-11～2024-08-10 的 133,697 筆。兩段評分列不重疊；較新模型依 rolling-origin 規則可使用較早年度的歷史成交。','',
      '加權 MAE 以 test 無標籤資料的已見／新建案／缺失組成計算；不使用 test 價格。缺失組權重為 0，沒有樣本的組別誤差記為 null，不假造為 0。','',
      '## 預先固定的候選與結果','',
      '近期加權混合以固定 50% 原 CatBoost＋50% 近期 CatBoost 組成 raw 部分；再沿用 75% raw＋25% 單價 LightGBM。近期權重半衰期固定 730 天。地區版本僅修改未來整體基準的年增率，採四年訓練資料、最低 30 建案／1,000 筆、最多 50% 地區權重，資料不足回退全域趨勢；原本局部參考成交不變。','',
      '| 開發年度版本 | 整體 MAE（元） | 加權 MAE（元） | 整體改善率 | 通過門檻 |',
      '|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {names[r['candidate']]} | {r['overall_mae']:,.2f} | {r['weighted_mae']:,.2f} | {r['overall_improvement_percent']:+.2f}% | {'對照' if r['eligible'] is None else '否'} |")
    lines += ['', '![開發年度整體與季度穩定性](round3_development_stability.png)', '', '圖中負值表示 MAE 降低；左圖虛線為至少改善 1% 的門檻，右圖虛線為單季退步 5% 的上限。僅代表開發年度。', '', '改善率正值代表誤差降低，負值代表退步。比較必須在同一時段進行，不能把此表與不同期間的 Public 或原 valid 分數相減來宣稱進步。','',
      '開發接受條件事先固定為整體、加權 MAE **各改善至少 1%**，每個預測季度及樣本數至少 500 的建案組別退步不超過 5%。三個新候選均未達兩項改善門檻；含近期加權的兩個版本另違反季度退步限制。結果出來後未降低門檻、未追加調參。','',
      '## 較新年度基準診斷','',
      f"開發結果已凍結為 v2_control，後續只訓練及評估 v2 對照：整體 MAE **{cm['overall_mae']:,.2f} 元**，加權 MAE **{cm['weighted_mae']:,.2f} 元**。沒有合格的新候選可進入確認比較，因此此分數是基準診斷，不能稱為新模型通過確認。",'',
      '## 資料處理與未採用嘗試','',
      '- 原始 train 455,787 筆：移除 3 列夾入英文欄名，再移除 529 筆排除編號後、包含價格的完全重複列，留下 455,255 筆。原 valid 11,994 筆、test 47,073 筆。原檔不修改。',
      '- 特徵共 87 個（21 類別、66 數值），編號與價格不作輸入；缺失、異常日期與負面積皆有明確處理／標記。詳細規則見資料處理紀錄。',
      '- 地址解析稽核找到新市區、前鎮區、平鎮區等名稱被舊規則截斷。獨立修正器安全更正 5,919／160／744 筆 train／valid／test，但本輪價格候選仍使用凍結的原特徵，避免同時更動過多因素。',
      '- 嘗試兩種固定尾端規則：大面積逐步靠向 raw，及 bulk／land 改用 raw。兩者都使驗證誤差惡化；沒有為了預測看起來合理而截尾。實際極高價案例中，單價模型有時更準。',
      '- 保存所有候選、逐季度／月份／城市／建案組別的分數、模型、未四捨五入預測、索引、參數與 SHA，並封存本輪實驗。','',
      '## 可重現與限制','',
      '每個年度均重新訓練，固定輪數，不使用評分段的 eval_set 或 early stopping。地區趨勢、缺失填值與日期權重只由該期訓練資料估計。先凍結候選、程式、資料、參數和索引，再訓練較新年度的模型。Public 僅記錄，未用來搜尋係數；完全沒有查詢內政部成交答案。','',
      '這些歷史標籤在前幾輪曾使用，確認段約 43.54% 也屬先前的時間驗證資料，因此不能稱為全新、完全未見的 holdout。回測與 test 仍有城市分布差異；嚴格檢查能降低過擬合風險，不能保證 Private 表現。','',
      '重跑順序：另建工作副本，保留原始資料、快取、既有依賴及 Python 環境，也必須保留 outputs/round3/candidate_plan.json、validation_protocol.json、audit 與 work_log.json。新副本中的 baseline、recency、evaluation、frozen_candidates.json 與 .sha256 應尚未產生。原工作區與封存結果保持原樣；訓練腳本會重寫同名產物，評估程式則拒絕覆寫已保存結果。', '', '```text',
      'python src/round3_backtest.py --split development --component catboost',
      'python src/round3_backtest.py --split development --component lightgbm',
      'python src/round3_backtest.py --split development --component reference',
      'python src/round3_recency.py --split development',
      'python src/round3_evaluate.py --split development --freeze',
      'python src/round3_backtest.py --split confirmation --component catboost',
      'python src/round3_backtest.py --split confirmation --component lightgbm',
      'python src/round3_backtest.py --split confirmation --component reference',
      'python src/round3_evaluate.py --split confirmation',
      'python src/round3_report.py', '```','',
      'GPU 訓練可能存在微小數值差異；本輪已保存實際模型與預測以確保可追溯。v2 提交檔 SHA：`'+digest+'`。','',
      '## 檔案入口','',
      '- [逐步工作紀錄](round3_work_log.md)',
      '- [資料處理紀錄](round3_preprocessing_record.md)',
      '- [防過擬合措施](round3_overfitting_controls.md)',
      '- [開發候選完整比較](../outputs/round3/evaluation/development/comparison.csv)',
      '- [機器可讀結果](round3_results.json)',
      '- [最終資料與提交檔檢查](round3_final_checks.json)',
      '- [維持原 v2 提交檔](../outputs/submission_v2.csv)', '']
    (ROOT/'reports/optimization_round3_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'outcome':'retain_v2','development':rows,'confirmation_control_mae':cm['overall_mae'],'confirmation_control_weighted_mae':cm['weighted_mae'],'submission_unchanged':True},ensure_ascii=True),flush=True)
if __name__=='__main__': main()
