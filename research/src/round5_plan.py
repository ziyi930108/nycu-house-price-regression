"""Freeze the sole fifth-round candidate, including the full scoring source closure."""
from datetime import datetime,timezone
from round5_common import *
from prepare_data import DATA_DIR
from round5_journal import record


def main():
    from round5_project_mask import MASK_RULES,parameters
    from round5_evaluate import FORMULAS,REQUIRED_COMPONENTS,THRESHOLDS,source_hashes
    from round3_evaluate import target_weights,runtime_versions
    for name in ['models','evaluation','baseline','candidate_plan.json','candidate_plan.sha256']:
        require(not (OUT/name).exists(),'Cannot predeclare after experiment artifacts or a plan exist')
    require(sha256(ROOT/'outputs/submission_v2.csv')=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b','The retained submission changed')
    require(sha256(ROOT/'runs/round4_experiments/snapshot_manifest.json')=='5113e311de8d793181e95a342033ad7777a1c1df696c75fef62b66c2001f11b0','Fourth-round snapshot changed')
    # The original preflight remains historical evidence of the initial helper;
    # use the corrected-normalization preflight for the actual training source.
    synthetic=read(OUT/'audit/mask_synthetic_preflight_v2.json')
    require(synthetic['self_check']['status']=='passed' and synthetic['helper_sha256']==sha256(ROOT/'src/round5_project_mask.py'),'Current mask preflight missing')
    require(synthetic['synthetic_key_equality_with_scorer'] and synthetic['synthetic_missing_flag_equality_with_scorer'],'Mask/scorer semantics differ')
    evaluation_preflight=read(OUT/'audit/evaluation_preflight.json')
    require(evaluation_preflight['status']=='passed','Evaluation preflight missing')
    verify_hashes(evaluation_preflight['source_sha256'])
    data=load_data()
    files=[ROOT/'data_cache'/f'{name}{suffix}' for name in ['train','valid','test'] for suffix in ['.pkl','_features.pkl','.json']]
    files += list((ROOT/'data_cache').glob('*features*.json'))
    files += [ROOT/'reports/feature_schema.json',ROOT/'data_cache/valid_clean_mask.npy',ROOT/'data_cache/valid_clean_mask.json',
      ROOT/'runs/v1/models/final_config.json',ROOT/'runs/round3_experiments/snapshot_manifest.json',
      ROOT/'runs/round4_experiments/snapshot_manifest.json',ROOT/'outputs/round4/candidate_plan.json',
      ROOT/'outputs/round2/neighborhood/final_reference.pkl',ROOT/'requirements.txt',ROOT/'requirements-round2.txt']
    files += [DATA_DIR/name for name in read(ROOT/'reports/run_manifest.json')['input_sha256']]
    contract={}
    for stage,(start,end) in SPLITS.items():
        fi,qi=temporal_indices(data,start,end)
        contract[stage]={'fit_rows':len(fi),'query_rows':len(qi),
          'fit_indices_content_sha256':hashlib.sha256(np.asarray(fi,dtype='<i8').tobytes()).hexdigest(),
          'query_indices_content_sha256':hashlib.sha256(np.asarray(qi,dtype='<i8').tobytes()).hexdigest()}
    previous=read_locked(ROOT/'outputs/round4/candidate_plan.json')
    require(contract==previous['split_contract'],'Chronological partitions changed from the previous protocol')
    require(THRESHOLDS==previous['thresholds'],'Acceptance thresholds changed')
    full_X,full_y=full_training_data(data)
    inputs={rel(path):sha256(path) for path in sorted(set(files))}
    audit_files=[OUT/'research_decision.json',OUT/'audit/mask_synthetic_preflight_v2.json',OUT/'audit/evaluation_preflight.json',
       ROOT/'reports/round5_validation_plan.md']
    value={'round':5,'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'control':'v2_control','candidate_formulas':FORMULAS,'required_components':REQUIRED_COMPONENTS,'thresholds':THRESHOLDS,
      'splits':{k:list(v) for k,v in SPLITS.items()},'split_contract':contract,
      'final_contract':{'fit_rows':len(full_X),'query_rows':len(data['X_test'])},
      'fit_source_sha256':fit_source_hashes(),'evaluation_source_sha256':source_hashes(),
      'data_sha256':inputs,'original_input_sha256':data['metadata']['source_sha256'],
      'research_evidence_sha256':{rel(path):sha256(path) for path in audit_files},
      'plan_creator_sha256':sha256(Path(__file__)),'runtime_versions':runtime_versions(),
      'candidates':{'project_mask_blend':{'formula':'.75*(.5*old_catboost+.5*masked_catboost)+.25*unit_lightgbm',
          'components':['project_mask_catboost']}},
      'components':{'project_mask_catboost':{'parameters':parameters(),'mask_rules':MASK_RULES,
          'target':'y/1e6','weight':'uniform','features':'original 87 columns; mask exactly four identifier columns according to fit-only project state'}},
      'training':{'eval_set':False,'early_stopping':False,'new_development_components':1,'new_development_candidates':1,
          'query_labels_for_features_or_training':False,'hyperparameter_sweep':False,'seed_sweep':False,'blend_weight_sweep':False},
      'prediction_rule':'Original unmasked 87-column X is used for old models, geographic scoring groups and the same fit-side v2 reference. Apply reference trend once to each fixed composite baseline.',
      'development_rules':previous['development_rules'],'confirmation_rules':previous['confirmation_rules'],'auxiliary_rules':previous['auxiliary_rules'],
      'selection':'One prespecified candidate. If every development gate passes freeze it; otherwise retain v2 and do not run later-stage candidates or controls. Later stages are vetoes only, never reselection.',
      'target_group_weights':target_weights(data),'quarter_definition':previous['quarter_definition'],
      'historically_untouched':False,'public_score_policy':'record only; never choose coefficients or gates from Public',
      'external_property_answers':False,'prior_round4_manifest_sha256':sha256(ROOT/'runs/round4_experiments/snapshot_manifest.json'),
      'limitations':['Repeated development reuse creates selection bias despite a single new candidate.',
        'Masking can remove useful information and does not prove reduced overfitting.',
        'A 1% gate is not a statistical-significance test; transactions within a project are correlated.',
        'Auxiliary labels were used in historical fitting and are not untouched. No Private improvement is guaranteed.']}
    locked_write(OUT/'candidate_plan.json',value)
    record('004','訓練前凍結','固定一個整案遮罩模型與一個融合候選，保存完整訓練／評分來源、資料、切分索引、參數與公式。',
      '事前限制搜尋範圍，避免看過分數後改遮罩率、seed 或比例；確保後續可核對實際執行條件。',
      '開發 Fit 240,316 筆、Query 81,234 筆。沿用原四季度與三組门檻，整體和加權 MAE 都須改善至少 1%。',
      '固定 2,999 輪、不設 eval_set、不早停；未通過開發不得啟動新候選後續年度或 auxiliary baseline。',
      ['outputs/round5/candidate_plan.json','outputs/round5/audit/evaluation_preflight.json','outputs/round5/audit/mask_distribution.json'])
    print(json.dumps({'plan':'outputs/round5/candidate_plan.json','sha256':sha256(OUT/'candidate_plan.json'),
          'candidates':1,'split_contract':contract,'final_contract':value['final_contract']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
