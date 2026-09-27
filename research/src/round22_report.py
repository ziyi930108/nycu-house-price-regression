"""Render finalized R22 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round22'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'huber_catboost_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                   'actual_learning_rate', 'actual_loss_function', 'actual_eval_metric',
                   'actual_leaf_estimation_method', 'actual_leaf_estimation_iterations', 'actual_boost_from_average')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names', 'added_feature_names',
    'baseline_feature_count', 'original95_preserved',
    'native_objective_contract', 'implicit_native_default_differences', 'raw_z_units',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count', 'native_learning_rate_comparison_rtol',
    'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
FEASIBILITY_PATH = 'outputs/round22_research/huber_feasibility.json'
METHOD_SOURCES_PATH = 'outputs/round22_research/method_sources.json'
REMOVED_FEATURES = []
CHANGED_PARAMETERS = ['loss_function']
ADDED_FEATURES = []
NATIVE_OBJECTIVE_CONTRACT = {'loss_function': 'Huber:delta=1', 'eval_metric': 'MAE',
    'leaf_estimation_method': 'Newton', 'leaf_estimation_iterations': 1, 'boost_from_average': False}
IMPLICIT_NATIVE_POLICY_CHANGES = {'boost_from_average': {'from': True, 'to': False},
    'leaf_estimation_method': {'from': 'Exact', 'to': 'Newton'}}
IMPLICIT_NATIVE_DEFAULT_DIFFERENCES = {key: {'baseline': value['from'], 'candidate': value['to']}
    for key, value in IMPLICIT_NATIVE_POLICY_CHANGES.items()}

TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
NATIVE_LEARNING_RATE_CONTRACT = {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0}
DOCUMENTS = ('reports/round22_plan.md', 'reports/round22_method_review.md',
    'reports/round22_pipeline_review.md')
# The work log is a navigation link; the final archiver binds its closed bytes.


def need(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def number(value): return '—' if value is None else f'{value:,.2f}'
def delta(a, b): return None if a in (None, 0) or b is None else 100 * (b-a) / a
def percent(value): return '—' if value is None else f'{value:+.4f}%'


def validate_metrics(metrics, rows):
    """Check saved summary structure, without calculating errors or MAE."""
    need(set(metrics) == {CONTROL, CANDIDATE}, 'Unexpected candidate inventory')
    for metric in metrics.values():
        need(metric['rows'] == rows, 'Metric row count differs')
        for key in ('overall_mae', 'weighted_mae'):
            need(type(metric[key]) in (float, int) and math.isfinite(metric[key]) and metric[key] >= 0, 'Invalid saved MAE')
        for family, names in [('strata', set(GROUPS)), ('forecast_quarters', {'Q1', 'Q2', 'Q3', 'Q4'})]:
            need(set(metric[family]) == names and sum(g['rows'] for g in metric[family].values()) == rows, 'Saved groups do not partition rows')
            for group in metric[family].values():
                n, value = group['rows'], group['mae']
                need(type(n) is int and n >= 0, 'Invalid saved group count')
                need((n == 0 and value is None) or (n > 0 and type(value) in (float, int) and math.isfinite(value) and value >= 0), 'Invalid saved group MAE')
    for family in ('strata', 'forecast_quarters'):
        need(all(metrics[CONTROL][family][key]['rows'] == metrics[CANDIDATE][family][key]['rows'] for key in metrics[CONTROL][family]), 'Control/candidate group rows differ')


def validate_finalization(accepted, frozen, stages):
    """Verify finalized status consistency, never choose a model from scores."""
    need([s['stage'] for s in stages] == list(LABELS), 'Stage order differs')
    completed = [s for s in stages if s['metrics'] is not None]
    need(completed and completed[0]['stage'] == 'development', 'Finalized development is required')
    need(set(accepted['stage_result_sha256']) == {s['stage'] for s in completed}, 'Saved result inventory differs')
    need(accepted['unevaluated_stages'] == [s['stage'] for s in stages if s['metrics'] is None], 'Unevaluated inventory differs')
    for stage in stages:
        need(stage['status'] == accepted['stage_status'][stage['stage']], 'Stage status differs')
        if stage['metrics'] is None:
            need(stage['status'] == 'not_executed_after_development_rejection'
                 and all(stage[k] is None for k in ('decision', 'model', 'independent_review')), 'Unexecuted stage has fabricated data')
        else:
            need(stage['status'] == 'price_evaluated' and type(stage['decision']['passed']) is bool, 'Invalid evaluated decision')
    dev_pass = completed[0]['decision']['passed']
    need(frozen['passed'] is dev_pass and frozen['later_stages_allowed'] is dev_pass, 'Development freeze differs')
    need(frozen['selected_candidate'] == (CANDIDATE if dev_pass else CONTROL), 'Frozen selection differs')
    need((dev_pass and len(completed) == 3) or (not dev_pass and len(completed) == 1), 'Unauthorized or incomplete later stages')
    expected = len(completed) == 3 and all(s['decision']['passed'] for s in completed)
    need(accepted['accepted'] is expected, 'Acceptance inconsistent with saved decisions')
    need(accepted['deployment_decision'] == ('backtest_accepted_pending_final_release' if expected else 'keep_v3')
         and accepted['new_submission_created'] is False
         and accepted['currently_recommended_submission'] == 'outputs/submission_v3.csv', 'Unsupported release state')
    return completed


def validate_capacity(capacity):
    need(set(capacity) == set(CAPACITY_FIELDS), 'Native capacity evidence fields differ')
    need(capacity['actual_model_depth'] == 8 and capacity['actual_grow_policy'] == 'SymmetricTree'
         and capacity['actual_boosting_type'] == 'Plain'
         and capacity['actual_trees'] == 2999, 'Wrong saved native depth/policy/tree count')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 256, 'Wrong native leaf-cap evidence')
    digest = capacity['tree_leaf_counts_sha256']
    need(isinstance(digest,str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
         'Invalid leaf-count LE-int64 digest')
    validate_objective_contract({key:capacity['actual_'+key] for key in NATIVE_OBJECTIVE_CONTRACT})
    lr=capacity['actual_learning_rate']
    need(type(lr) in (int,float) and math.isfinite(lr)
         and abs(lr-0.065) <= 1e-7*abs(0.065), 'Native learning rate exceeds fixed storage tolerance')


def validate_objective_contract(value):
    need(isinstance(value,dict) and set(value)==set(NATIVE_OBJECTIVE_CONTRACT), 'Native objective contract fields differ')
    for key,expected in NATIVE_OBJECTIVE_CONTRACT.items():
        need(type(value[key]) is type(expected) and value[key]==expected, 'Native objective contract differs: '+key)


def validate_model(model, plan, stage):
    need(model['round']==22 and model['stage']==stage and model['family']=='catboost'
         and model['component']=='geography_huber_catboost' and model['target_mode']=='raw', 'Wrong model identity')
    params,baseline=model['parameters'],model['baseline_parameters']
    need(params==plan['parameters'] and baseline==plan['baseline_parameters']
         and baseline['loss_function']=='MAE' and baseline['eval_metric']=='MAE'
         and set(params)==set(baseline) and params==dict(baseline,loss_function='Huber:delta=1')
         and model['changed_parameter_names']==plan['changed_parameter_names']==CHANGED_PARAMETERS
         and plan['model_hyperparameter_change']=={'loss_function':{'from':'MAE','to':'Huber:delta=1'}},
         'Only the passed loss_function may differ from R6')
    need(params['iterations']==2999 and params['learning_rate']==.065 and params['depth']==8
         and params['boosting_type']=='Plain' and params['random_seed']==42 and params['thread_count']==4
         and params['eval_metric']=='MAE' and params['loss_function']=='Huber:delta=1', 'Wrong fixed Huber recipe')
    need(not any(k in params for k in ('leaf_estimation_method','leaf_estimation_iterations','boost_from_average')),
         'Implicit native defaults must not become additional passed parameters')
    need(model['removed_feature_names']==plan['removed_feature_names']==[]
         and model['added_feature_names']==plan['added_feature_names']==[]
         and plan['feature_rules']['removed_feature_names']==[]
         and plan['feature_rules']['added_feature_names']==[], 'Original 95-feature view required')
    need(model['iterations']==model['actual_iterations']==model['actual_trees']==2999
         and model['feature_count']==95 and model['categorical_feature_count']==23
         and model['native_categorical_feature_count']==23
         and model['baseline_feature_count']==95 and model['original95_preserved'] is True,
         'Original 95/23 schema required')
    need(model['sample_weight']==plan['sample_weight']=='uniform'
         and model['weight_argument_passed_to_pool'] is plan['weight_argument_passed_to_pool'] is False,
         'Raw uniform Pool training required')
    need(model['feature_rules']==plan['feature_rules'], 'Feature rules differ')
    rules=model['feature_rules']
    need(rules['feature_count']==95 and rules['categorical_feature_count']==23
         and rules['baseline_feature_count']==95 and rules['original95_preserved'] is True
         and rules['existing_95_values_dtypes_order_and_index_unchanged'] is True, 'Rule-level preservation/schema differs')
    for item in (model,plan,rules): validate_objective_contract(item['native_objective_contract'])
    need(plan['implicit_native_policy_changes']==IMPLICIT_NATIVE_POLICY_CHANGES
         and model['implicit_native_default_differences']==rules['implicit_native_default_differences']==IMPLICIT_NATIVE_DEFAULT_DIFFERENCES,
         'Loss-induced native policy changes differ')
    need(rules['baseline_loss_function']=='MAE' and rules['loss_function']=='Huber:delta=1'
         and rules['eval_metric']=='MAE' and rules['huber_delta_in_model_units']==1.
         and rules['huber_delta_TWD']==1e6 and rules['raw_z_units']==model['raw_z_units']=='million_TWD'
         and rules['native_defaults_explicitly_passed'] is False
         and rules['same_passed_parameters_does_not_imply_same_native_defaults'] is True, 'Huber objective rules differ')
    need(model['target_formula']==plan['target_formula']==TARGET_FORMULA
         and model['inverse_formula']==plan['inverse_formula']==INVERSE_FORMULA, 'Wrong raw target/inverse scale')
    need(plan['native_learning_rate_contract']==NATIVE_LEARNING_RATE_CONTRACT
         and model['native_learning_rate_comparison_rtol']==rules['native_learning_rate_comparison_rtol']==1e-7
         and model['native_learning_rate_comparison_atol']==rules['native_learning_rate_comparison_atol']==0.,
         'Learning-rate storage tolerance changed')
    need(model['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and rules['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and rules['iterations']==rules['geo_encoder_rules']['iterations']==2999, 'Original geographic encoder provenance differs')
    need(set(model['artifact_sha256'])==MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    fitted=model['fitted_actual_learning_rate'];saved=model['actual_learning_rate']
    need(type(fitted) in (int,float) and math.isfinite(fitted)
         and abs(fitted-.065)<=1e-7*.065 and abs(fitted-saved)<=1e-7*abs(saved), 'Fitted/saved LR representation differs beyond fixed tolerance')
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied',
                'extra_price_cap_applied','reference_or_trend_applied_here','reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: '+key)


def validate_replay_feature_evidence(audit,model):
    need(audit['removed_feature_names']==model['removed_feature_names']==[]
         and audit['added_feature_names']==model['added_feature_names']==[], 'Independent feature declaration differs')
    need(audit['original_95_values_dtypes_order_index_preserved'] is True,
         'Independent original model-view preservation was not confirmed')
    need(audit['changed_parameter_names']==CHANGED_PARAMETERS
         and audit['native_categorical_feature_count']==model['native_categorical_feature_count']==23,
         'Independent parameter or native category declaration differs')
    need(audit['raw_z_units']=='million_TWD' and audit['native_learning_rate_comparison_rtol']==1e-7
         and audit['native_learning_rate_comparison_atol']==0., 'Independent raw units or tolerance differ')
    validate_objective_contract(audit['native_objective_contract'])
    need(audit['native_objective_contract']==model['native_objective_contract']
         and audit['implicit_native_policy_changes']==IMPLICIT_NATIVE_POLICY_CHANGES,
         'Independent native objective policies differ')
    evidence=audit['independent_fit_state_evidence']
    need(evidence['base_feature_count']==95 and evidence['feature_count']==95
         and evidence['categorical_count']==23 and evidence['fit_rows']==model['fit_rows']
         and evidence['removed_feature_names']==[] and evidence['added_feature_names']==[]
         and evidence['fit_state_reconstructed_independently'] is True
         and evidence['fit_model_feature_values_sha256']==model['fit_model_feature_values_sha256'],
         'Independent fit-only feature-state reconstruction differs')
    validate_objective_contract(evidence['native_objective_contract'])
    return copy.deepcopy(evidence)


def research_expectations(source_text):
    """Read frozen literal declarations only, never import the ML driver."""
    found={}
    for node in ast.parse(source_text).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RESEARCH_FILES','RESEARCH_EXPECTED_SHA'):
                    need(target.id not in found,'Duplicate research declaration')
                    found[target.id]=ast.literal_eval(node.value)
    need(set(found)=={'RESEARCH_FILES','RESEARCH_EXPECTED_SHA'},'Missing literal R22 research declarations')
    seeds,expected=found['RESEARCH_FILES'],found['RESEARCH_EXPECTED_SHA']
    need(isinstance(seeds,(tuple,list)) and len(seeds)>0 and len(set(seeds))==len(seeds)
         and isinstance(expected,dict) and set(seeds)==set(expected), 'Research seed and expected map inventories differ')
    return {name:expected[name] for name in seeds}


def validate_research(research,dependencies,declared,feasibility,method_sources):
    """Only saved artificial compatibility and source history; no property data."""
    need(isinstance(research,dict) and research and research==declared, 'Frozen research differs from literal driver map')
    for path in (FEASIBILITY_PATH,METHOD_SOURCES_PATH):
        need(path in research and path[:-5]+'.sha256' in research, 'Research JSON and receipt must be frozen')
    need(all(dependencies.get(k)==v for k,v in research.items()), 'Research outside frozen dependencies')
    expected=dict(research)
    def add(name,digest):
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid research identity')
        need(name not in expected or expected[name]==digest, 'Conflicting compatibility provenance')
        expected[name]=digest
    for name,digest in research.items():add(name,digest)
    need(feasibility['round']==22 and feasibility['status']=='synthetic_GPU_compatibility_passed'
         and feasibility['synthetic_only'] is True and feasibility['synthetic_model_count']==2
         and feasibility['property_MAE_computed'] is False
         and feasibility['property_X_y_models_or_predictions_loaded'] is False, 'Wrong artificial compatibility scope')
    need(feasibility['declared_candidate_loss']=='Huber:delta=1' and feasibility['declared_delta_TWD']==1e6
         and feasibility['candidate_passed_parameter_changes']==CHANGED_PARAMETERS
         and feasibility['feasibility_overrides']=={'iterations':2,'verbose':False}, 'Synthetic objective/scope changed')
    baseline=feasibility['baseline_parameters'];params=feasibility['candidate_parameters']
    need(baseline['loss_function']=='MAE' and baseline['eval_metric']=='MAE' and baseline['iterations']==2999
         and params==dict(baseline,loss_function='Huber:delta=1'), 'Synthetic passed-parameter comparison differs')
    for field in ('source_sha256','input_sha256'):
        for name,digest in feasibility[field].items():add(name,digest)
    need(set(method_sources)=={'source_sha256','evidence_sha256','official_urls'}
         and isinstance(method_sources['official_urls'],list) and method_sources['official_urls']
         and all(isinstance(url,str) and url.startswith('https://catboost.ai/') for url in method_sources['official_urls']),
         'Wrong source-only method inventory')
    for field in ('source_sha256','evidence_sha256'):
        need(isinstance(method_sources[field],dict) and method_sources[field], 'Missing source history')
        for name,digest in method_sources[field].items():add(name,digest)
    models=feasibility['models'];need(set(models)=={'huber_candidate','mae_control'}, 'Exactly two artificial models required')
    for name,item in models.items():
        add(item['model_path'],item['model_sha256'])
        need(item['model_path'].startswith('outputs/round22_research/') and item['model_path'].endswith('.cbm'),
             'Synthetic model outside research namespace')
        need(item['tree_count']==2 and item['feature_count']==95 and item['fit_rows']>0 and item['query_rows']>0
             and item['saved_reload_predictions_exact'] is True, 'Artificial model scope or reload check differs')
        need(len(item['categorical_positions'])==len(set(item['categorical_positions']))==23
             and all(type(k) is int and 0<=k<95 for k in item['categorical_positions']), 'Artificial category schema differs')
        passed=params if name=='huber_candidate' else baseline
        need(item['passed_parameters']==dict(passed,iterations=2,verbose=False), 'Artificial execution overrides differ')
        for phase in ('fitted_native_parameters','saved_native_parameters'):
            native=item[phase]
            if name=='huber_candidate':validate_objective_contract({key:native[key] for key in NATIVE_OBJECTIVE_CONTRACT})
            else:
                need(native['loss_function']=='MAE' and native['eval_metric']=='MAE'
                     and native['leaf_estimation_method']=='Exact' and type(native['leaf_estimation_iterations']) is int
                     and native['leaf_estimation_iterations']==1 and native['boost_from_average'] is True,
                     'Artificial control native defaults differ')
    a,b=(models[name]['saved_native_parameters'] for name in ('mae_control','huber_candidate'))
    need(set(a)==set(b), 'Synthetic saved native parameter inventories differ')
    difference={key:{'control':a[key],'candidate':b[key]} for key in a if a[key]!=b[key]}
    intended={'loss_function':{'control':'MAE','candidate':'Huber:delta=1'},
      **{key:{'control':value['from'],'candidate':value['to']} for key,value in IMPLICIT_NATIVE_POLICY_CHANGES.items()}}
    need(difference==feasibility['native_control_candidate_differences']==intended, 'Observed implicit defaults differ')
    need(research==expected, 'Frozen research closure omits named compatibility/history evidence')
    return {'frozen_research_dependency_count':len(research),'dependency_sha256':copy.deepcopy(research),
      'required_research_seed_paths':list(declared),'research_type':'synthetic_GPU_Huber_compatibility_and_source_history',
      'feasibility_path':FEASIBILITY_PATH,'feasibility_sha256':research[FEASIBILITY_PATH],
      'feasibility_status':feasibility['status'],'synthetic_only':True,'synthetic_model_count':2,
      'synthetic_tree_count_each':2,'declared_delta_TWD':feasibility['declared_delta_TWD'],
      'native_control_candidate_differences':copy.deepcopy(difference),
      'synthetic_model_artifacts':{name:{'path':item['model_path'],'sha256':item['model_sha256']} for name,item in models.items()},
      'limitations':copy.deepcopy(feasibility['limitations']),
      'method_sources_path':METHOD_SOURCES_PATH,'method_sources_sha256':research[METHOD_SOURCES_PATH],
      'official_urls':copy.deepcopy(method_sources['official_urls']),
      'report_recomputed_research_or_price_metrics':False}


def render(payload):
    stages=payload['stages'];completed=[s for s in stages if s['metrics'] is not None]
    decision='三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines=['# 第二十二輪結果：固定 Huber 損失的地理 CatBoost','',
        f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。','',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有读取預測陣列內容、重訓或重算 MAE。','',
        '## 實際處理與唯一候選','',
        '- 沿用移除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。完整列去重依原始欄位（含價格、排除編號）進行，沒有以 87 特徵欄重新去重，也沒有刪除高價或難預測列。',
        '- 原 87 欄加既有 8 地理欄，共 95 欄／23 類別全部保留；不增刪特徵。原95欄的值、型別、列序及欄序不變，原 87 欄仍用於 reference 與評分分組。',
        '- 唯一傳入參數差異為 loss_function：MAE → Huber:delta=1；eval_metric 仍為 MAE。其他原 R6 參數固定：2,999 棵樹、learning_rate=0.065、Plain、depth 8、seed 42、4 threads；changed_parameter_names=[loss_function]、added_feature_names=[]、removed_feature_names=[]。',
        '- delta=1 的單位是百萬元總價模型輸出，對應總價殘差 1,000,000 元。此數值事前固定，沒有從標籤分位數或驗證殘差估計。',
        '- 只有一項傳入參數改變，不代表只有一項原生設定改變。人工相容性測試觀察到 loss 連動的函式庫預設：leaf_estimation_method 由 Exact 變 Newton、boost_from_average 由 true 變 false；leaf_estimation_iterations=1。正式模型亦直接核對這些設定，沒有另外把它們傳入訓練參數。',
        '- 目標仍為 y_TWD/1e6、uniform，Pool 不傳入 sample_weight；預測只以 max(z×1e6,0) 還原總價。無 eval_set、早停、額外價格上限或校正。',
        '- 新 raw 地理 CatBoost 替換 v3 的 37.5% 地理模型槽；37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保留。先還原 TWD 再融合，原 reference trend 只施加一次。',
        '- feature_state 只接受 fit X／geo／categories，不接受 y 或 query。價格模型訓練不使用 query labels；共享 loader 會反序列化含 target 的快取，不能聲稱從未載入驗證價格。test-X 建案比例只用於固定加權評分，沒有用作本模型訓練權重。','',
        '研究依據是保存的兩個人工 GPU 模型（每個僅 2 棵樹）的 Huber／MAE 相容性比較，及本地既有方法 source 審查；完整來源、人工模型位元組 SHA、JSON、收據、日誌和方法文件由計畫綁定。人工標籤不是房價資料，沒有計算房價 MAE。本報告只核保存 JSON 與雜湊，不載入人工 CBM 或重跑測試。小型相容性通過不等於正式 2,999 棵樹的成本或準確度已獲保證。每個回測窗口沿用其原本對應年度的地理資料。','',
        '## 固定年度結果','',
        'MAE 是融合及一次 reference 後、未取整總價的新臺幣誤差；變化為正表示退步。未執行不是零誤差或通過。','',
        '| 窗口 | Query 日期 | Fit／query | v3 MAE | 候選 MAE | 變化 | v3 加權 MAE | 候選加權 MAE | 變化 | 結果 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for stage in stages:
        common = f"| {stage['title']} | {'～'.join(stage['dates'])} | {stage['fit_rows']:,}／{stage['query_rows']:,} |"
        if stage['metrics'] is None:
            lines.append(common + ' — | — | — | — | — | — | 開發期未通過，未執行 |')
        else:
            a, b = stage['metrics'][CONTROL], stage['metrics'][CANDIDATE]
            lines.append(common + f" {number(a['overall_mae'])} | {number(b['overall_mae'])} | {percent(delta(a['overall_mae'], b['overall_mae']))} | {number(a['weighted_mae'])} | {number(b['weighted_mae'])} | {percent(delta(a['weighted_mae'], b['weighted_mae']))} | {'通過' if stage['decision']['passed'] else '未通過'} |")
    for stage in completed:
        lines += ['', f"## {stage['title']}分組與季度", '', '| 群組 | 筆數 | v3 MAE | 候選 MAE | 變化 |', '|---|---:|---:|---:|---:|']
        a, b = stage['metrics'][CONTROL], stage['metrics'][CANDIDATE]
        for family, keys in [('strata', GROUPS), ('forecast_quarters', ('Q1', 'Q2', 'Q3', 'Q4'))]:
            for key in keys:
                item, other = a[family][key], b[family][key]
                lines.append(f"| {key} | {item['rows']:,} | {number(item['mae'])} | {number(other['mae'])} | {percent(delta(item['mae'], other['mae']))} |")
        lines += ['', '固定條件：', '']
        lines += [f"- `{key}`：{'通過' if value else '未通過'}。" for key, value in stage['decision']['checks'].items()]
        diagnostic=stage['model']['raw_z_diagnostics']
        lines+=['',f"已重新載入原生 CBM 與特徵狀態，重播全部 {stage['query_rows']:,} 列；另核對原始 z、總價還原、v3／候選融合及保存的評分與門檻。原始 z<0 為 {diagnostic['negative_count']:,} 列，z=0 為 {diagnostic['zero_count']:,} 列。z 是百萬元總價模型輸出，不是對數或單價；計數只作診斷，未用於調參。",'']
        capacity=stage['independent_review']['native_model_capacity']
        lines += [f"訓練記憶體 learning_rate={stage['model']['fitted_actual_learning_rate']:.12g}；以下原生 learning_rate 為保存 CBM 重載值。兩種表示均依既定容差核對，不強求位元表示相同。", '']
        lines+=[f"原生模型證據：boosting_type={capacity['actual_boosting_type']}、depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹、最大 {capacity['maximum_tree_leaf_count']} 葉（上限 256）、learning_rate={capacity['actual_learning_rate']:.12g}。學習率對 0.065 使用 rtol=1e-7、atol=0 容許儲存誤差；另核對 23 個原生類別欄位置。完整葉數向量的 LE-int64 SHA 保存在機器報告；這份報告本身不重新載入模型。",'']
        lines += [f"原生目標證據：loss_function={capacity['actual_loss_function']}、eval_metric={capacity['actual_eval_metric']}、leaf_estimation_method={capacity['actual_leaf_estimation_method']}、leaf_estimation_iterations={capacity['actual_leaf_estimation_iterations']}、boost_from_average={str(capacity['actual_boost_from_average']).lower()}。這些是直接核對保存模型的原生設定，不只依賴傳入參數宣告。", '']
    lines+=['## 解讀與防過擬合限制','',
        'Huber 在固定 delta 內為二次損失、外部為線性尾端；它改變了訓練目標，最小化者一般不等於 MAE 的條件中位數。不能由名稱推定比 MAE 更抗大型離群值，也不能由人工相容性測試推定 MAE 改善。MAE 仍是保存結果與固定驗收門檻使用的指標。','',
        '這是同一原95欄、raw目標、uniform權重與固定融合下的一個損失函數候選；同時伴隨已揭露的函式庫原生預設變化，不宣稱只隔離一個內部數值效應。沒有依本輪價格結果改 delta、起始值、葉節點估計或門檻。相同seed不保證GPU跨次訓練逐位元一致；保存模型的完整精確重播是另一項驗證。','',
        '既有座標多為道路或交會處代表點，不是住宅或門牌的精確測量；本輪不修正位置精度或改變地理規則。完整特徵保留亦不保證重訓後與v3逐位元相同。','',
        '開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。','',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。1% 是工程門檻，不是統計顯著性；不能消除跨輪選型偏差或保證 Private 成績。沒有依 Public 成績調參。','',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。','',
        '## 完整紀錄','',
        '- [方法與歷史審查](round22_method_review.md)、[事前規約](round22_plan.md)。',
        '- [獨立流程審查](round22_pipeline_review.md)、[工作紀錄](round22_work_log.md)。',
        '- [人工 GPU 相容性紀錄](../outputs/round22_research/huber_feasibility.json)、[歷史方法來源清單](../outputs/round22_research/method_sources.json)。',
        '- [機器可讀報告](round22_results.json)、[凍結計畫](../outputs/round22/candidate_plan.json)、[最終判定](../outputs/round22/accepted_candidate.json)。',
        '- 各期 JSON 保存原始群組／季度摘要與獨立重播證據；proof 的 status=passed 表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。','']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round22_report.md', ROOT/'reports/round22_results.json', ROOT/'reports/round22_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round22-report-*')), 'Partial report staging must be retained and inspected')
    captured = {}
    def capture(path, digest=None):
        path = Path(path).resolve()
        need(path.is_relative_to(ROOT) and path.is_file(), 'Missing/unsafe source')
        actual = sha(path); name = path.relative_to(ROOT).as_posix()
        need(digest is None or digest == actual, 'Changed source: ' + name)
        need(name not in captured or captured[name] == actual, 'Input changed during report')
        captured[name] = actual
        return actual
    def locked(path):
        capture(path); receipt = path.with_suffix('.sha256'); capture(receipt)
        need(receipt.read_text(encoding='ascii').strip() == sha(path), 'Receipt differs')
        return read(path)
    def mapping(values):
        for name, digest in values.items(): capture(ROOT/name, digest)
    plan = locked(OUT/'candidate_plan.json'); frozen = locked(OUT/'frozen_candidate.json'); accepted = locked(OUT/'accepted_candidate.json')
    plan_sha, frozen_sha = sha(OUT/'candidate_plan.json'), sha(OUT/'frozen_candidate.json')
    need(plan['round'] == frozen['round'] == accepted['round'] == 22
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_dependency_sha256'].items()),
         'Research dependencies are outside the frozen input map')
    expected=research_expectations((ROOT/'src/round22_experiment.py').read_text(encoding='utf-8-sig'))
    feasibility=locked(ROOT/FEASIBILITY_PATH);method_sources=locked(ROOT/METHOD_SOURCES_PATH)
    research_summary=validate_research(plan['research_dependency_sha256'],plan['dependency_sha256'],expected,feasibility,method_sources)
    mapping(plan['research_dependency_sha256'])
    need(set(accepted['stage_status']) == set(LABELS), 'Wrong stage status inventory')
    for area in ('models', 'evaluation'):
        folder = OUT/area
        need(folder.is_dir() and {p.name for p in folder.iterdir()} == set(accepted['stage_result_sha256']), 'Unexpected or partial stage artifacts')
    stages = []
    for stage, title in LABELS.items():
        spec = plan['split_contract'][stage]
        item = {'stage': stage, 'title': title, 'dates': plan['splits'][stage], 'fit_rows': spec['fit_rows'],
                'query_rows': spec['query_rows'], 'status': accepted['stage_status'][stage],
                'metrics': None, 'decision': None, 'model': None, 'independent_review': None}
        if stage in accepted['stage_result_sha256']:
            path = OUT/'evaluation'/stage/'results.json'
            capture(path, accepted['stage_result_sha256'][stage]); result = locked(path)
            need(result['round'] == 22 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
            need(result['source_sha256'] == plan['source_sha256'] and result['thresholds'] == plan['thresholds']
                 and result['formulas'] == plan['formulas'] and result['target_group_weights'] == plan['target_group_weights'], 'Saved result protocol differs')
            need(item['status'] == 'price_evaluated' and result['fit_rows'] == item['fit_rows']
                 and result['query_rows'] == item['query_rows'], 'Wrong stage counts/status')
            validate_metrics(result['metrics'], item['query_rows'])
            mapping(result['input_artifact_sha256']); mapping(result['output_artifact_sha256'])
            model = locked(OUT/'models'/stage/'metadata.json')
            validate_model(model, plan, stage)
            need(model['plan_sha256'] == plan_sha and model['source_sha256'] == plan['source_sha256'], 'Model provenance differs')
            for name, digest in model['artifact_sha256'].items(): capture(OUT/'models'/stage/name, digest)
            audit_path = OUT/'audit'/(stage+'_replay.json'); audit = locked(audit_path)
            need(audit['round'] == 22 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            feature_evidence = validate_replay_feature_evidence(audit, model)
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round22_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity'],
                    'removed_feature_names': list(audit['removed_feature_names']),
                    'added_feature_names': list(audit['added_feature_names']),
                    'native_objective_contract': copy.deepcopy(audit['native_objective_contract']),
                    'implicit_native_policy_changes': copy.deepcopy(audit['implicit_native_policy_changes']),
                    'independent_fit_state_evidence': feature_evidence})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 22, 'candidate': CANDIDATE,
        'accepted': accepted['accepted'], 'stopping_reason': accepted['stopping_reason'],
        'deployment_decision': accepted['deployment_decision'], 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv', 'stages': stages,
        'plan_sha256': plan_sha, 'frozen_candidate_sha256': frozen_sha,
        'accepted_candidate_sha256': sha(OUT/'accepted_candidate.json'),
        'source_input_sha256': captured, 'historically_untouched': False, 'public_score_used': False,
        'private_score_known': False, 'research_summary': research_summary,
        'report_only_reads_saved_JSON_and_hashes_binary_inputs': True}
    text = render(payload)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    contents = [text.encode('utf-8'), encoded, (hashlib.sha256(encoded).hexdigest()+'\n').encode('ascii')]
    for name, digest in captured.items(): need(sha(ROOT/name) == digest, 'Source changed while reporting')
    staging = ROOT/'reports'/('.round22-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Complete bounded artificial contract/render checks; no real evidence read."""
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (RuntimeError,KeyError,ValueError,SyntaxError):checks.append(name)
        else:raise AssertionError(name+' not rejected')
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'random_seed':42,'thread_count':4,
              'loss_function':'MAE','eval_metric':'MAE','boosting_type':'Plain','l2_leaf_reg':6}
    params=dict(baseline,loss_function='Huber:delta=1')
    rules={'removed_feature_names':[],'added_feature_names':[],'iterations':2999,'geo_encoder_rules':{'iterations':2999},
           'native_learning_rate_comparison_rtol':1e-7,'native_learning_rate_comparison_atol':0.0,
           'geo_encoder_rule_training_fields_are_baseline_provenance_only':True,
           'feature_count':95,'categorical_feature_count':23,'baseline_feature_count':95,'original95_preserved':True,
           'existing_95_values_dtypes_order_and_index_unchanged':True,
           'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
           'implicit_native_default_differences':copy.deepcopy(IMPLICIT_NATIVE_DEFAULT_DIFFERENCES),
           'baseline_loss_function':'MAE','loss_function':'Huber:delta=1','eval_metric':'MAE',
           'huber_delta_in_model_units':1.,'huber_delta_TWD':1e6,'raw_z_units':'million_TWD',
           'native_defaults_explicitly_passed':False,'same_passed_parameters_does_not_imply_same_native_defaults':True}
    plan={'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
          'changed_parameter_names':list(CHANGED_PARAMETERS),'feature_rules':copy.deepcopy(rules),
          'removed_feature_names':[],'added_feature_names':[],'sample_weight':'uniform','weight_argument_passed_to_pool':False,
          'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
          'native_learning_rate_contract':copy.deepcopy(NATIVE_LEARNING_RATE_CONTRACT),
          'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
          'implicit_native_policy_changes':copy.deepcopy(IMPLICIT_NATIVE_POLICY_CHANGES),
          'model_hyperparameter_change':{'loss_function':{'from':'MAE','to':'Huber:delta=1'}}}
    cap={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
         'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain',
         'actual_learning_rate':.065,**{'actual_'+k:v for k,v in NATIVE_OBJECTIVE_CONTRACT.items()}}
    model={'round':22,'stage':'development','family':'catboost','component':'geography_huber_catboost','target_mode':'raw',
           'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
           'changed_parameter_names':list(CHANGED_PARAMETERS),'removed_feature_names':[],'added_feature_names':[],
           'baseline_feature_count':95,'original95_preserved':True,
           'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
           'implicit_native_default_differences':copy.deepcopy(IMPLICIT_NATIVE_DEFAULT_DIFFERENCES),'raw_z_units':'million_TWD',
           'iterations':2999,'actual_iterations':2999,'feature_count':95,'categorical_feature_count':23,
           'native_categorical_feature_count':23,'sample_weight':'uniform','weight_argument_passed_to_pool':False,
           'fit_rows':7,'fit_model_feature_values_sha256':'c'*64,'feature_rules':copy.deepcopy(rules),
           'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
           'artifact_sha256':dict.fromkeys(MODEL_FILES,'b'*64),'raw_z_diagnostics':{'negative_count':1,'zero_count':0},
           'runner_elapsed_seconds':1.,'fitted_actual_learning_rate':.065,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,'geo_encoder_rule_training_fields_are_baseline_provenance_only':True,**cap}
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied','extra_price_cap_applied',
                'reference_or_trend_applied_here','reference_features_changed'):model[key]=False
    validate_model(model,plan,'development');checks.append('raw95_23_Plain_only_Huber_parameter_change')
    for name,patch in [('nativeOrdered',{'actual_boosting_type':'Ordered'}),('wrong97features',{'feature_count':97}),
                       ('wrong25cats',{'categorical_feature_count':25}),('native22cats',{'native_categorical_feature_count':22}),
                       ('unit_target',{'target_mode':'unit'}),('weighted_Pool',{'weight_argument_passed_to_pool':True}),
                       ('unit_inverse',{'inverse_formula':'max(z * area * 1e4, 0)'}),
                       ('wrong_tree_count',{'actual_trees':5998}),('native_wrong_learning_rate',{'actual_learning_rate':.0325}),
                       ('native_nan_learning_rate',{'actual_learning_rate':float('nan')}),
                       ('changed_rtol',{'native_learning_rate_comparison_rtol':1e-6}),
                       ('changed_atol',{'native_learning_rate_comparison_atol':1e-8}),
                       ('extra_calibration',{'calibration_applied':True}),('native_MAE',{'actual_loss_function':'MAE'}),
                       ('native_wrong_delta',{'actual_loss_function':'Huber:delta=2'}),('native_wrong_metric',{'actual_eval_metric':'RMSE'}),
                       ('native_Exact',{'actual_leaf_estimation_method':'Exact'}),('native_two_leaf_steps',{'actual_leaf_estimation_iterations':2}),
                       ('native_leaf_steps_bool',{'actual_leaf_estimation_iterations':True}),('native_boost_average',{'actual_boost_from_average':True}),
                       ('native_bool_as_zero',{'actual_boost_from_average':0})]:
        bad=copy.deepcopy(model);bad.update(patch)
        rejects(name,lambda m=bad:validate_model(m,plan,'development'))
    for lr in (0.06499999761581421,0.06499999762):
        float32_cap=copy.deepcopy(cap);float32_cap['actual_learning_rate']=lr
        validate_capacity(float32_cap);checks.append('native_storage_LR_accepted_'+str(lr))
    saved_pair=copy.deepcopy(model);saved_pair['actual_learning_rate']=.06499999762
    saved_pair['fitted_actual_learning_rate']=.06499999761581421
    validate_model(saved_pair,plan,'development');checks.append('canonical_saved_and_fitted_LR_pair_accepted')
    bad=copy.deepcopy(model);bad['parameters']['l2_leaf_reg']=7
    bad_plan=copy.deepcopy(plan);bad_plan['parameters']=copy.deepcopy(bad['parameters'])
    rejects('second_parameter_change_rejected',lambda:validate_model(bad,bad_plan,'development'))
    bad=copy.deepcopy(model);bad['changed_parameter_names']=[]
    rejects('parameter_change_declaration_rejected',lambda:validate_model(bad,plan,'development'))
    bad=copy.deepcopy(plan);bad['native_learning_rate_contract']['expected']=.0325
    rejects('plan_native_LR_contract_changed',lambda:validate_model(model,bad,'development'))
    bad=copy.deepcopy(model);bad['geo_encoder_rule_training_fields_are_baseline_provenance_only']=False
    rejects('baseline_geo_rule_provenance_lost',lambda:validate_model(bad,plan,'development'))
    bad=copy.deepcopy(plan);bad['native_objective_contract']['loss_function']='MAE'
    rejects('plan_objective_changed',lambda:validate_model(model,bad,'development'))
    bad=copy.deepcopy(model);bad['implicit_native_default_differences']['leaf_estimation_method']['candidate']='Exact'
    rejects('implicit_policy_not_declared',lambda:validate_model(bad,plan,'development'))
    for key in ('leaf_estimation_method','leaf_estimation_iterations','boost_from_average'):
        bad=copy.deepcopy(model);bad_plan=copy.deepcopy(plan)
        for v in (bad,bad_plan):
            for field in ('parameters','baseline_parameters'):v[field][key]=NATIVE_OBJECTIVE_CONTRACT[key]
        rejects('implicit_policy_became_passed_'+key,lambda m=bad,p=bad_plan:validate_model(m,p,'development'))
    for key,value in [('added_feature_names',['extra']),('removed_feature_names',['project_name']),
                      ('baseline_feature_count',97),('original95_preserved',False)]:
        bad=copy.deepcopy(model);bad[key]=value
        rejects('model_'+key,lambda m=bad:validate_model(m,plan,'development'))
    evidence={'fit_rows':7,'base_feature_count':95,'feature_count':95,'categorical_count':23,'removed_feature_names':[],
              'added_feature_names':[],'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
              'fit_state_reconstructed_independently':True,'fit_model_feature_values_sha256':'c'*64}
    proof={'added_feature_names':[],'removed_feature_names':[],'changed_parameter_names':list(CHANGED_PARAMETERS),
           'original_95_values_dtypes_order_index_preserved':True,'raw_z_units':'million_TWD',
           'native_categorical_feature_count':23,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,'independent_fit_state_evidence':copy.deepcopy(evidence),
           'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
           'implicit_native_policy_changes':copy.deepcopy(IMPLICIT_NATIVE_POLICY_CHANGES)}
    validate_replay_feature_evidence(proof,model);checks.append('independent_original95_fit_only_Huber_evidence_match')
    for key,value in [('raw_z_units','ten_thousand_TWD_per_effective_square_metre'),
                      ('native_categorical_feature_count',25),('native_learning_rate_comparison_rtol',1e-5),
                      ('changed_parameter_names',[]),('original_95_values_dtypes_order_index_preserved',False),
                      ('native_objective_contract',{}),('implicit_native_policy_changes',{})]:
        bad=copy.deepcopy(proof);bad[key]=value
        rejects('proof_'+key,lambda q=bad:validate_replay_feature_evidence(q,model))
    for key,value in [('fit_model_feature_values_sha256','d'*64),('feature_count',97),('categorical_count',25),
                      ('native_objective_contract',{})]:
        bad=copy.deepcopy(proof);bad['independent_fit_state_evidence'][key]=value
        rejects('proof_fitstate_'+key,lambda q=bad:validate_replay_feature_evidence(q,model))
    source_name='src/synthetic_recipe.py';method_name='reports/synthetic_method.md';config_name='synthetic/config.json'
    research={FEASIBILITY_PATH:'a'*64,FEASIBILITY_PATH[:-5]+'.sha256':'b'*64,
              METHOD_SOURCES_PATH:'c'*64,METHOD_SOURCES_PATH[:-5]+'.sha256':'d'*64,
              source_name:'e'*64,method_name:'f'*64,config_name:'1'*64,
              'outputs/round22_research/mae_control.cbm':'2'*64,'outputs/round22_research/huber_candidate.cbm':'3'*64}
    control_native={'loss_function':'MAE','eval_metric':'MAE','leaf_estimation_method':'Exact',
        'leaf_estimation_iterations':1,'boost_from_average':True,'iterations':2,'learning_rate':.06499999762}
    huber_native=dict(control_native,**NATIVE_OBJECTIVE_CONTRACT)
    models={}
    for name,native,passed,digest in [('mae_control',control_native,baseline,'2'*64),('huber_candidate',huber_native,params,'3'*64)]:
        models[name]={'model_path':'outputs/round22_research/'+name+'.cbm','model_sha256':digest,'tree_count':2,
            'feature_count':95,'fit_rows':8,'query_rows':2,'saved_reload_predictions_exact':True,
            'categorical_positions':list(range(21))+[93,94],'passed_parameters':dict(passed,iterations=2,verbose=False),
            'fitted_native_parameters':copy.deepcopy(native),'saved_native_parameters':copy.deepcopy(native)}
    observed={key:{'control':control_native[key],'candidate':huber_native[key]} for key in control_native if control_native[key]!=huber_native[key]}
    feasibility={'round':22,'status':'synthetic_GPU_compatibility_passed','synthetic_only':True,'synthetic_model_count':2,
        'property_MAE_computed':False,'property_X_y_models_or_predictions_loaded':False,
        'declared_candidate_loss':'Huber:delta=1','declared_delta_TWD':1000000,'candidate_passed_parameter_changes':['loss_function'],
        'feasibility_overrides':{'iterations':2,'verbose':False},'baseline_parameters':copy.deepcopy(baseline),
        'candidate_parameters':copy.deepcopy(params),'source_sha256':{source_name:'e'*64},'input_sha256':{config_name:'1'*64},
        'models':models,'native_control_candidate_differences':observed,'limitations':['Artificial compatibility only.']}
    methods={'source_sha256':{source_name:'e'*64},'evidence_sha256':{method_name:'f'*64},
             'official_urls':['https://catboost.ai/docs/en/concepts/loss-functions-regression']}
    source='RESEARCH_FILES = '+repr(tuple(research))+'\nRESEARCH_EXPECTED_SHA = '+repr(research)
    expected=research_expectations(source)
    summary=validate_research(research,research,expected,feasibility,methods)
    ok('synthetic_models_and_source_history_research_closure',summary['frozen_research_dependency_count']==len(research)
       and summary['synthetic_model_count']==2 and not summary['report_recomputed_research_or_price_metrics'])
    rejects('missing_research_dependency',lambda:validate_research(research,{},expected,feasibility,methods))
    bad=copy.deepcopy(expected);bad[source_name]='9'*64
    rejects('driver_map_identity_mismatch',lambda:validate_research(research,research,bad,feasibility,methods))
    for field,value in [('status','pending'),('synthetic_only',False),('synthetic_model_count',1),
                        ('property_MAE_computed',True),('property_X_y_models_or_predictions_loaded',True),
                        ('declared_delta_TWD',2e6),('candidate_passed_parameter_changes',[])]:
        bad=copy.deepcopy(feasibility);bad[field]=value
        rejects('research_'+field,lambda f=bad:validate_research(research,research,expected,f,methods))
    bad=copy.deepcopy(feasibility);bad['source_sha256']['unfrozen.py']='9'*64
    rejects('unfrozen_synthetic_source',lambda:validate_research(research,research,expected,bad,methods))
    bad=copy.deepcopy(feasibility);bad['models']['huber_candidate']['model_sha256']='9'*64
    rejects('synthetic_model_hash_conflict',lambda:validate_research(research,research,expected,bad,methods))
    for field,value in [('tree_count',2999),('saved_reload_predictions_exact',False),('categorical_positions',[0,0])]:
        bad=copy.deepcopy(feasibility);bad['models']['huber_candidate'][field]=value
        rejects('synthetic_model_'+field,lambda f=bad:validate_research(research,research,expected,f,methods))
    bad=copy.deepcopy(feasibility);bad['models']['huber_candidate']['saved_native_parameters']['boost_from_average']=True
    rejects('synthetic_policy_contradiction',lambda:validate_research(research,research,expected,bad,methods))
    bad=copy.deepcopy(methods);bad['evidence_sha256']['unfrozen.md']='9'*64
    rejects('unfrozen_method_evidence',lambda:validate_research(research,research,expected,feasibility,bad))
    bad=copy.deepcopy(methods);bad['status']='completed_pure_X_only'
    rejects('wrong_inherited_inventory_schema',lambda:validate_research(research,research,expected,feasibility,bad))
    for path in (FEASIBILITY_PATH,METHOD_SOURCES_PATH):
        receipt=path[:-5]+'.sha256';bad=copy.deepcopy(research);bad.pop(receipt)
        rejects('missing_receipt_'+receipt,lambda m=bad:validate_research(m,m,m,feasibility,methods))
    rejects('missing_research_declaration',lambda:research_expectations('OTHER = {}'))
    rejects('dynamic_declaration_not_executed',lambda:research_expectations('RESEARCH_FILES=call()\nRESEARCH_EXPECTED_SHA={}'))
    metric={'rows':4,'overall_mae':10.,'weighted_mae':11.,
            'strata':{'seen':{'rows':2,'mae':10.},'new_nonmissing':{'rows':2,'mae':10.},'missing':{'rows':0,'mae':None}},
            'forecast_quarters':{q:{'rows':1,'mae':10.} for q in ('Q1','Q2','Q3','Q4')}}
    metrics={CONTROL:copy.deepcopy(metric),CANDIDATE:copy.deepcopy(metric)}
    validate_metrics(metrics,4);checks.append('all_groups_quarters_and_empty_mae_contract')
    for name,mutate in [('negative_mae',lambda m:m[CANDIDATE].update(overall_mae=-1)),
                        ('nonfinite_mae',lambda m:m[CANDIDATE].update(weighted_mae=float('nan'))),
                        ('empty_as_zero',lambda m:m[CANDIDATE]['strata']['missing'].update(mae=0)),
                        ('wrong_quarter_rows',lambda m:m[CANDIDATE]['forecast_quarters']['Q1'].update(rows=2))]:
        bad=copy.deepcopy(metrics);mutate(bad)
        rejects(name,lambda m=bad:validate_metrics(m,4))
    def stages_for(passes):
        rows=[]
        for i,(stage,title) in enumerate(LABELS.items()):
            present=i==0 or passes[0]
            rows.append({'stage':stage,'title':title,'dates':['start','end'],'fit_rows':7,'query_rows':4,
                'status':'price_evaluated' if present else 'not_executed_after_development_rejection',
                'metrics':copy.deepcopy(metrics) if present else None,
                'decision':{'passed':passes[i],'checks':{'saved_gate':passes[i]}} if present else None,
                'model':copy.deepcopy(model) if present else None,
                'independent_review':{'native_model_capacity':copy.deepcopy(cap)} if present else None})
        return rows
    def authorities(passes,stages):
        frozen={'passed':passes[0],'later_stages_allowed':passes[0],
                'selected_candidate':CANDIDATE if passes[0] else CONTROL}
        accepted={'accepted':all(passes),'deployment_decision':'backtest_accepted_pending_final_release' if all(passes) else 'keep_v3',
                  'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
                  'stage_result_sha256':{x['stage']:'a'*64 for x in stages if x['metrics'] is not None},
                  'stage_status':{x['stage']:x['status'] for x in stages},
                  'unevaluated_stages':[x['stage'] for x in stages if x['metrics'] is None]}
        return accepted,frozen
    for name,passes in [('dev_reject',[False,False,False]),('later_reject',[True,False,True]),('all_pass',[True,True,True])]:
        stages=stages_for(passes);accepted,frozen=authorities(passes,stages)
        validate_finalization(accepted,frozen,stages);checks.append(name+'_finalization')
        text=render({'accepted':all(passes),'stages':stages})
        ok(name+'_status',('尚待另行最終訓練與發布' in text) if all(passes) else ('本輪未採用，維持 submission_v3.csv' in text))
        ok(name+'_raw95_native12_Huber',all(x in text for x in ['boosting_type=Plain','95 欄／23 類別','Huber:delta=1','leaf_estimation_method=Newton','百萬元','uniform'])
           and all(x not in text for x in ['A/100','effective_area','round20','round21','geo_grid_0p005','重跑紀錄','194.935','5,998','0.0325','11 份','四欄消融','91 欄']))
    stages=stages_for([True,True,True]);accepted,frozen=authorities([True,True,True],stages)
    bad=copy.deepcopy(accepted);bad['accepted']=False
    rejects('inconsistent_final_acceptance',lambda:validate_finalization(bad,frozen,stages))
    bad=copy.deepcopy(frozen);bad['later_stages_allowed']=False
    rejects('inconsistent_dev_freeze',lambda:validate_finalization(accepted,bad,stages))
    bad=copy.deepcopy(stages);bad[2].update(metrics=None,decision=None,model=None,independent_review=None,
                                            status='not_executed_after_development_rejection')
    rejects('later_missing_not_completed_veto',lambda:validate_finalization(accepted,frozen,bad))
    text=render({'accepted':False,'stages':stages_for([False,False,False])})
    ok('unexecuted_stages_null_display',text.count('開發期未通過，未執行')==2)
    ok('raw_dedup_and_original87_reference',all(x in text for x in ['含價格、排除編號','原 87 欄仍用於 reference']))
    ok('Huber_objective_and_native_policy_limitations',all(x in text for x in ['最小化者一般不等於 MAE 的條件中位數','不能由名稱推定比 MAE 更抗大型離群值','兩個人工 GPU 模型','每個僅 2 棵樹','沒有計算房價 MAE','leaf_estimation_method 由 Exact 變 Newton','boost_from_average 由 true 變 false']))
    ok('unchanged_geography_and_scale',all(x in text for x in ['不增刪特徵','1,000,000 元','不修正位置精度','保存模型的完整精確重播','沒有依本輪價格結果改 delta']))
    ok('known_public_unknown_new_no_release',all(x in text for x in ['2,883,250.4138','Public 與 Private 皆未知','沒有上傳或發布新提交檔']))
    ok('MAE_percentage_direction',delta(100,101)==1 and delta(100,99)==-1 and delta(0,1) is None
       and percent(None)=='—' and number(None)=='—')
    ok('native12_and_model_summary_fields_complete',len(CAPACITY_FIELDS)==12 and all(k in MODEL_FIELDS for k in cap)
       and set(MODEL_FIELDS)<=set(model))
    return {'status':'passed','synthetic_only':True,'checks':checks,'check_count':len(checks),
        'complete_report_self_check_executed':True,'real_JSON_labels_predictions_or_models_read':False,
        'report_files_written':False,'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['generate','self-check'])
    args=parser.parse_args()
    if args.action=='self-check':print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    else:generate()
