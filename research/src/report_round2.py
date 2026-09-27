"""Build the round-two report from released artifacts and fixed holdout predictions."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from prepare_data import ROOT, sha256
from model_common import load_data, mae


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8-sig'))


def money(value):
    return '未執行' if value is None else f'{value:,.0f}'


def main():
    config=read('models/round2/release_config.json')
    comparison=read('outputs/round2/blending/comparison.json')
    old_cv=read('outputs/round2/blending/baseline_crossfit_reference.json')
    stability=read('outputs/round2/stability/stability_report.json')
    reference=read('outputs/round2/neighborhood/final_reference.json')
    baseline=read('runs/v1/reports/results.json')['selection']
    data=load_data()
    candidates=[]
    catalog={r['name']:r for r in comparison['catalog']}
    for r in comparison['candidates']:
        c={'name':r['name'],'validation_mae':r['validation_mae'],
           'clean_validation_mae':None,'temporal_mae':r['temporal_mae'],
           'seen_project_mae':r['seen_mae'],'unseen_project_mae':r['unseen_mae'],
           'reweighted_temporal_mae':r['reweighted_temporal_mae']}
        source=catalog[c['name']]
        if source.get('valid_predictions'):
            p=np.maximum(np.load(ROOT/source['valid_predictions']),0)
            mask=data['valid_clean_mask']
            c['clean_validation_mae']=mae(data['y_valid'][mask],p[mask])
        c['round']=2 if 'round2' in source['metadata_path'] else 1
        candidates.append(c)
    submissions=[]
    config_sha=sha256(ROOT/'models/round2/release_config.json')
    for v in config['variants']:
        audit=read(f"outputs/submission_{v['name']}_checks.json")
        if audit['config_sha256']!=config_sha: raise ValueError('Submission generated from a different configuration')
        if audit['sha256']!=sha256(ROOT/f"outputs/submission_{v['name']}.csv"): raise ValueError('Submission changed')
        submissions.append(audit)
    chosen=next(v for v in config['variants'] if v['name']=='v2')
    improvements={k:(baseline[k]-chosen['metrics'][k])/baseline[k] for k in
                  ['validation_mae','temporal_mae','reweighted_temporal_mae']}
    proof_path=ROOT/'reports/round2_notebook_checks.json'
    proof=read('reports/round2_notebook_checks.json') if proof_path.is_file() else None
    result={'metric':'MAE','unit':'TWD','kaggle_score':None,'baseline':baseline,
            'candidate_comparison':candidates,'variants':config['variants'],'submissions':submissions,
            'improvement_over_v1':improvements,'strategy_comparison':{
                'baseline_strategies':old_cv['strategies'],'round2_strategies':comparison['strategies'],
                'group_definition':comparison['crossfit_group_definition'],
                'coarse_schemes':comparison['coarse_schemes'],
                'caveat':comparison['crossfit_caveat']},'stability':stability,
            'recommendation':{'version':'v2','reason':'原驗證與時間驗證均改善，模型比混合策略簡單；混合版只在重加權指標有小幅優勢，保留供實際提交比較。'},
            'constraints':{'local_supplied_data_only':True,'no_government_answer_lookup':True,
                           'no_test_price_labels':True},
            'validation':{'original_rows':11994,'clean_rows':11958,'seen_project_rows':10632,'unseen_project_rows':1362,
                'temporal_fit_rows':397041,'temporal_rows':58206,'cutoff':'2024-04-01','end_inclusive':'2024-08-10',
                'temporal_seen_rows':32227,'temporal_unseen_rows':25979,'test_unseen_rate':config['test_unseen_project_rate'],
                'all_reported_validation_scores_from_pre_final_fit_predictions':True},
            'final_training':{'rows':config['training_rows'],'sources':config['training_sources'],
                'neighborhood_reference':reference,'components':config['components']},
            'source_sha256':config['source_sha256'],'release_config_sha256':config_sha,
            'notebook_verification':proof,'generated_at_utc':datetime.now(timezone.utc).isoformat()}
    out=ROOT/'reports'; out.mkdir(exist_ok=True)
    (out/'round2_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/round2/matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['V1','V2 trend','V2 anchor','V2 mix']
    rows=[baseline]+[v['metrics'] for v in config['variants']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#8a97a8','#157f78','#759daf','#bc945a']
    for ax,key,title in zip(axes,['validation_mae','temporal_mae'],['Provided validation','Future-month validation']):
        values=[r[key]/1e6 for r in rows]
        bars=ax.bar(labels,values,color=colors,width=.62)
        ax.bar_label(bars,labels=[f'{r[key]:,.0f}' for r in rows],padding=5,fontsize=10)
        ax.set_ylim(0,max(values)*1.2); ax.set_ylabel('MAE (million TWD; lower is better)')
        ax.set_title(title,pad=16,weight='bold'); ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('Round 2: offline model comparison',fontsize=16,weight='bold')
    fig.savefig(out/'round2_model_comparison.png',dpi=170)
    plt.close(fig)

    table_header='| 版本 | 原驗證 MAE | 去重疊驗證 MAE | 時間驗證 MAE | 未見建案 MAE | 重加權時間 MAE |\n|---|---:|---:|---:|---:|---:|\n'
    def row(label,m):
        keys=['validation_mae','clean_validation_mae','temporal_mae','unseen_project_mae','reweighted_temporal_mae']
        return '| '+label+' | '+' | '.join(money(m.get(k)) for k in keys)+' |\n'
    release_table=table_header+row('v1',baseline)+''.join(row(v['name'],v['metrics']) for v in config['variants'])
    candidate_table=table_header+''.join(row(c['name'],c) for c in candidates if c['round']==2)
    old_global=next(s for s in old_cv['strategies'] if s['strategy']=='global')
    new_global=next(s for s in comparison['strategies'] if s['strategy']=='global')
    old_gated=next(s for s in old_cv['strategies'] if s['strategy']=='project_gated')
    new_gated=next(s for s in comparison['strategies'] if s['strategy']=='project_gated')
    ci=next(s for s in stability['bootstrap_summary'] if s['contrast']=='v1_vs_trend' and s['metric']=='target_mix_mae_improvement')
    csv_table='| 檔案 | 筆數 | 最小／最大預測（元） | SHA-256 |\n|---|---:|---:|---|\n'
    for audit in submissions:
        name=audit['name']; csv_table+=f"| [submission_{name}.csv](../outputs/submission_{name}.csv) | {audit['rows']:,} | {audit['minimum']:,}／{audit['maximum']:,} | `{audit['sha256']}` |\n"
    md=f'''# 房價預測：第二輪優化報告

建議先提交 **[submission_v2.csv](../outputs/submission_v2.csv)**。本輪完成 6 個新候選的時間驗證，另保留參考版與混合版，共 3 份可直接上傳的 CSV。v1 已封存於 `runs/v1`，未覆蓋。

v2 原驗證 MAE 為 **{chosen['metrics']['validation_mae']:,.0f} 元**，比 v1 降低 **{improvements['validation_mae']:.2%}**；時間驗證 MAE 為 **{chosen['metrics']['temporal_mae']:,.0f} 元**，降低 **{improvements['temporal_mae']:.2%}**。所有分數都來自本機驗證；目前沒有 Kaggle 上傳分數。

![各版離線 MAE 比較](round2_model_comparison.png)

## 可提交版本

{release_table}

MAE 越小越好，單位皆為新臺幣元。「去重疊驗證」排除與 train 原始特徵完全重疊的 36 筆，保留 11,958 筆；沒有據此刪除 test 資料。

- **v2**：相似交易參考加上僅由訓練資料估計的時間趨勢，為主要推薦版本。
- **v2_anchor**：仍會依參考交易與待預測交易的日期差調整局部價格，但不另對整體基準加上未來趨勢。保留以檢驗較弱的市場趨勢假設。
- **v2_mix**：75% v2 加 25% 空間特徵 LightGBM。比 v2 的重加權時間 MAE 改善約 0.50%，原驗證與整體時間 MAE 則略差，所以保留為替代提交版本。

## 本輪的六個候選

{candidate_table}

`catboost_recency2y_d8` 對近期訓練資料增加權重；`catboost_depth9` 增加樹深。`lightgbm_unit_smooth31` 增加平滑並減少葉節點；`lightgbm_unit_spatial63` 移除建案名稱、完整地址與棟戶等細粒度識別欄位，保留較粗的位置特徵。兩種相似交易候選分別為 anchor 與 trend。

表內「未執行」表示該候選只完成時間驗證，未完整執行原驗證。候選的原驗證結果與全量重訓模型分開保存，沒有用全量模型回測已參與訓練的標籤。

## 驗證設計與限制

原 valid 有 11,994 筆，其中 10,632 筆屬於訓練資料已見建案，1,362 筆屬於未見建案。測試集交易日期是 2024-08-11 至 2025-07-31，且約 {config['test_unseen_project_rate']:.2%} 屬於 train＋valid 未見建案，原 valid 因此容易高估實際表現。

時間驗證只以 2024-04-01 之前的 397,041 筆訓練，預測 2024-04-01 至 2024-08-10 的 58,206 筆；其中已見建案 32,227 筆，未見建案 25,979 筆。重加權指標按本地無標籤 test 的未見建案比例，重新組合兩組 MAE，僅是分布近似。

混合策略以「縣市＋鄉鎮市區＋建案」分 5 組選權重，舊候選池與新候選池使用相同方法。全域策略的交叉選權重後重加權 MAE，從 **{old_global['crossfit_reweighted_temporal_mae']:,.0f}** 降至 **{new_global['crossfit_reweighted_temporal_mae']:,.0f}**，改善 **{1-new_global['crossfit_reweighted_temporal_mae']/old_global['crossfit_reweighted_temporal_mae']:.2%}**。依已見／未見分開選權重的策略則為 {old_gated['crossfit_reweighted_temporal_mae']:,.0f} → {new_gated['crossfit_reweighted_temporal_mae']:,.0f}，新方案仍比全域方案差，因此沒有採用更複雜的分組混合。

這是**固定候選預測的穩定性診斷**。基礎模型的早停與候選設計已使用此時間驗證集，分組選權重不會消除先前選型偏差；以上交叉選權重數字不能當成完全未碰過資料的測試成績，也不可與 v1 固定權重的 {baseline['reweighted_temporal_mae']:,.0f} 混為同一比較口徑。

另以 2,973 個建案做 500 次成對分組重抽樣。v2 相對 v1 的重加權 MAE 改善約 {ci['point_delta']:,.0f} 元，條件 95% 區間為 {ci['ci95_lower']:,.0f} 至 {ci['ci95_upper']:,.0f} 元；4 月至 8 月的各月整體、已見與未見建案均改善（8 月僅含 1–10 日）。這個區間未校正候選選擇，也不涵蓋未來市場變動與重新訓練的不確定性。重抽樣固定採用四捨五入後的 63.1615% 比例，故與主表可能有不到 1 元的數值差異。

## 相似交易方法與資料使用

基準為 75% CatBoost 總價模型＋25% LightGBM 單價模型。參考價格僅從相應訓練側資料建立，按建案、地址、道路及區域等層級尋找相似交易，以面積、車位、樓層與交易日期差異調整，再依參考資料的支持程度與基準融合。參考資料的分位數篩選、標準化、價格調整係數與最晚日期均只在該次訓練側估計；驗證標籤只用於評分。

原 valid 日期沒有超出訓練資料最晚日期，所以 anchor 與 trend 在原 valid 的預測相同。兩者未來趨勢差異主要由時間驗證檢驗，不能把原 valid 的改善解讀為已驗證未來市場走勢。

定版後才合併 train 與 valid，去除重複後共 **467,221 筆**，重新建立參考與訓練最終模型。全量參考估計的年化對數時間係數約 0.07009，約等於每年 7.26% 的乘數增幅，整體基準的未來外推最多 366 天。這是此資料擬合出的規則，不是外部房市資訊或對未來漲幅的保證。參考版與混合版用來保留不同假設的可比較結果。

全程只使用使用者提供的本地競賽檔案，**沒有到內政部不動產成交案件實際資訊資料供應系統對答案**，也沒有使用外部房價或測試價格標籤。原始 4 個輸入檔案的 SHA-256 會在最終封存時重新核對。

## 提交檔與重現

{csv_table}

三份 CSV 都包含 47,073 筆，欄位依序為「編號」「總價元」，編號與範例提交檔完全同序，價格皆為有限的正整數。四捨五入至元，不額外以人工門檻截斷高價交易。

[Notebook](../solution_round2.ipynb) 預設載入已保存模型，重新讀取測試 Excel、建立特徵並產生三份提交檔；[操作說明](../README_round2.md) 提供重現與設定組裝方式。執行後的 [Notebook 核對紀錄](round2_notebook_checks.json) 保存所有程式格的執行結果與三份 CSV 重跑前後 SHA。[完整數據](round2_results.json) 與 [最終檔案核對表](round2_run_manifest.json) 提供機器可讀紀錄。

所有離線成績都是選定版本的預測證據；實際推薦仍應在你上傳 Kaggle 後，依回報分數核對。
'''
    (out/'optimization_round2_report.md').write_text(md,encoding='utf-8')
    print(json.dumps({'recommended':'v2','variants':len(config['variants']),'new_candidates':sum(c['round']==2 for c in candidates),'improvements':improvements,'notebook_verified':bool(proof and proof.get('verification_passed'))},ensure_ascii=True),flush=True)

if __name__=='__main__':
    main()
