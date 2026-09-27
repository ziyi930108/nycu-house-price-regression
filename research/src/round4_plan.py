"""Write a new immutable, prespecified round-four experiment plan before training."""
from __future__ import annotations
from datetime import datetime,timezone
from round4_common import *
from prepare_data import DATA_DIR
from round4_journal import record

def main():
    from round4_augmented import FEATURE_RULES
    from round4_spatial_raw import PARAMETERS as RAW_SPATIAL_PARAMETERS
    from round4_evaluate import FORMULAS,REQUIRED_COMPONENTS,THRESHOLDS
    from round4_time_normalized import parameters as anchor_parameters
    for folder in ['models/development','evaluation/development']:
        require(not (OUT/folder).exists(),'Cannot predeclare after development artifacts exist')
    data=load_data()
    sources=fit_source_hashes()
    files=[ROOT/'data_cache'/f'{name}{suffix}' for name in ['train','valid','test'] for suffix in ['.pkl','_features.pkl','.json']]
    files += [ROOT/'reports/feature_schema.json',ROOT/'data_cache/valid_clean_mask.npy',ROOT/'data_cache/valid_clean_mask.json',
       ROOT/'runs/v1/models/final_config.json',ROOT/'runs/round3_experiments/snapshot_manifest.json',
       ROOT/'outputs/round2/neighborhood/final_reference.pkl',ROOT/'requirements.txt',ROOT/'requirements-round2.txt']
    files += [DATA_DIR/name for name in read(ROOT/'reports/run_manifest.json')['input_sha256']]
    data_hash={rel(path):sha256(path) for path in files}
    contract={}
    for stage,(start,end) in SPLITS.items():
        fi,qi=temporal_indices(data,start,end)
        contract[stage]={'fit_rows':len(fi),'query_rows':len(qi),
          'fit_indices_content_sha256':hashlib.sha256(np.asarray(fi,dtype='<i8').tobytes()).hexdigest(),
          'query_indices_content_sha256':hashlib.sha256(np.asarray(qi,dtype='<i8').tobytes()).hexdigest()}
    plan_value={'round':4,'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'control':'v2_control','candidate_formulas':FORMULAS,'required_components':REQUIRED_COMPONENTS,'thresholds':THRESHOLDS,'splits':{k:list(v) for k,v in SPLITS.items()},'split_contract':contract,
      'fit_source_sha256':sources,'evaluation_source_sha256':{'src/round4_evaluate.py':sha256(ROOT/'src/round4_evaluate.py')},
      'data_sha256':data_hash,'original_input_sha256':data['metadata']['source_sha256'],
      'candidates':{
       'anchor_blend':{'formula':'.75*(.5*old_catboost+.5*anchor_catboost)+.25*unit_lightgbm','components':['anchor_catboost']},
       'augmented_blend':{'formula':'.75*(.5*old_catboost+.5*augmented_catboost)+.25*unit_lightgbm','components':['augmented_catboost']},
       'spatial_raw_blend':{'formula':'.75*old_catboost+.125*unit_lightgbm+.125*spatial_raw_lightgbm','components':['spatial_raw_lightgbm']},
       'combined':{'formula':'.75*(.5*old_catboost+.25*anchor_catboost+.25*augmented_catboost)+.125*unit_lightgbm+.125*spatial_raw_lightgbm','components':list(COMPONENTS)}},
      'prediction_rule':'Apply the original v2 reference trend once to each candidate baseline; same fit-side reference and 87 original reference features as control.',
      'components':{
       'anchor_catboost':{'parameters':anchor_parameters(),'lookback_years':4,'target':'y/1e6 * G','weight':'1/G',
          'G':'exp(fit_only_beta_time*(fit_anchor_day-trade_day)/365.25)',
          'drop_features':['trade_year','trade_month_index','trade_day_index'],
          'missing_fit_dates':'exclude from this component only','prediction_semantics':'fit-anchor-date total price TWD; no future multiplier within the component'},
       'augmented_catboost':{'parameters':anchor_parameters(),'target':'y/1e6','weight':'uniform',
          'features':'original 87 plus the fixed 10 fit-only geographic/name features','feature_rules':FEATURE_RULES},
       'spatial_raw_lightgbm':{'parameters':RAW_SPATIAL_PARAMETERS,'configuration':'round2 spatial63 parameters fixed4000 trees/n_jobs4, raw target y/1e6 and uniform weights',
          'drop_features':['project_name','address_raw','building_and_unit','building_block'],'target':'y/1e6','weight':'uniform'}},
      'training':{'eval_set':False,'early_stopping':False,'new_development_components':3,'new_development_candidates':4,'query_labels_for_features_or_training':False},
      'development_rules':{'minimum_overall_improvement':.01,'minimum_weighted_improvement':.01,'max_quarter_regression':.05,'large_stratum_min_rows':500,'max_large_stratum_regression':.05},
      'confirmation_rules':{'max_overall_regression':0.0,'max_weighted_regression':0.0,'minimum_improved_quarters':2,'large_stratum_min_rows':500,'max_large_stratum_regression':.05},
      'auxiliary_rules':{'max_overall_regression':.03,'max_weighted_regression':.03,'large_stratum_min_rows':500,'max_large_stratum_regression':.05,'role':'additional veto only, not candidate reselection or coefficient selection'},
      'selection':'Lowest weighted development MAE among eligible candidates; deterministic name tie-break. Freeze one candidate before confirmation or auxiliary fitting/scoring; no reselection. If none qualifies retain v2.',
      'weights':'Test feature-only geographic project seen/new_nonmissing/missing proportions using unchanged round3 address_city/address_district/project_name; augmented geography is not used for scoring groups.',
      'quarter_definition':{'Q1':[0,91],'Q2':[92,183],'Q3':[184,275],'Q4':[276,None]},
      'historically_untouched':False,'public_score_policy':'record only, never coefficient tuning','external_property_answers':False,
      'auxiliary_limitations':'2021-08-11..2022-08-10 check is disjoint from other check windows, but its labels previously appeared in historical training; geographic project keys may not uniquely identify physical developments.'}
    locked_write(OUT/'candidate_plan.json',plan_value)
    record('002','新增驗證年度','加入 2021-08-11～2022-08-10 的完整一年作為凍結候選的額外否決檢查。',
      '減少只依賴同一個開發年度的選型偏差。',
      '早期年度 134,805 筆訓練、105,511 筆檢查，新地理建案 85.70%；另保留原 development 與 confirmation 年度。只選一個候選進入後兩項檢查。',
      '三段評分列不重疊，但歷史標籤曾在舊模型訓練出現；不宣稱完全未見。額外年度整體／加權最多退步 3%，大建案組最多 5%。',
      ['reports/round4_validation_plan.md','outputs/round4/audit/auxiliary_split_design.json'])
    record('003','預先凍結模型嘗試','固定三個新基礎模型與四個融合方案：時間正規化、地址／案名特徵增強、raw 空間 LightGBM，以及合併方案。',
      '改進未見建案的可轉移資訊與樹模型的時間外插；避免重複上一輪無效的調整。',
      '訓練前保存候選公式、特徵規則、參數、日期、原始檔／快取／程式 SHA 與不可覆寫收據。開發兩項 MAE 仍須各改善至少 1%。',
      '固定輪數，不用 eval_set／early stopping；不依看過的 Public 分數挑係數；新特徵詞彙只學自當期訓練 X。',
      ['outputs/round4/candidate_plan.json','reports/round4_model_history.md','reports/round4_location_opportunity.md'])
    print(json.dumps({'plan':'outputs/round4/candidate_plan.json','sha256':sha256(OUT/'candidate_plan.json'),'split_contract':contract},ensure_ascii=True),flush=True)
if __name__=='__main__':main()
