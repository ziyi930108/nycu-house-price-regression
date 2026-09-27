"""Render finalized R21 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round21'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'fine_grid_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                   'actual_learning_rate')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names', 'added_feature_names',
    'baseline_feature_count', 'original95_preserved',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count', 'native_learning_rate_comparison_rtol',
    'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
INVENTORY_PATH = 'outputs/round21_research/grid_audit.json'
REMOVED_FEATURES = []
CHANGED_PARAMETERS = []
ADDED_FEATURES = ['geo_grid_0p005', 'geo_grid_0p005_shift']
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
NATIVE_LEARNING_RATE_CONTRACT = {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0}
DOCUMENTS = ('reports/round21_plan.md', 'reports/round21_method_review.md',
    'reports/round21_pipeline_review.md')
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
    lr=capacity['actual_learning_rate']
    need(type(lr) in (int,float) and math.isfinite(lr)
         and abs(lr-0.065) <= 1e-7*abs(0.065), 'Native learning rate exceeds fixed storage tolerance')


def validate_model(model, plan, stage):
    need(model['round']==21 and model['stage']==stage and model['family']=='catboost'
         and model['component']=='geography_fine_grid_catboost' and model['target_mode']=='raw', 'Wrong model identity')
    params,baseline=model['parameters'],model['baseline_parameters']
    need(params==baseline==plan['parameters']==plan['baseline_parameters']
         and model['changed_parameter_names']==plan['changed_parameter_names']==[]
         and plan['model_hyperparameter_change']=={}, 'All original R6 passed parameters must remain unchanged')
    need(params['iterations']==2999 and params['learning_rate']==.065 and params['depth']==8
         and params['boosting_type']=='Plain' and params['random_seed']==42
         and params['thread_count']==4 and params['loss_function']=='MAE', 'Wrong fixed R6 recipe')
    need(model['removed_feature_names']==plan['removed_feature_names']==[]
         and model['added_feature_names']==plan['added_feature_names']==ADDED_FEATURES
         and plan['feature_rules']['removed_feature_names']==[]
         and plan['feature_rules']['added_feature_names']==ADDED_FEATURES, 'Wrong fixed two-grid feature delta')
    need(model['iterations']==model['actual_iterations']==model['actual_trees']==2999
         and model['feature_count']==97 and model['categorical_feature_count']==25
         and model['native_categorical_feature_count']==25
         and model['baseline_feature_count']==95 and model['original95_preserved'] is True,
         'Original 95 features and final 97/25 schema required')
    need(model['sample_weight']==plan['sample_weight']=='uniform'
         and model['weight_argument_passed_to_pool'] is plan['weight_argument_passed_to_pool'] is False,
         'Raw uniform Pool training required')
    need(model['feature_rules']==plan['feature_rules'], 'Feature rules differ')
    rules=model['feature_rules']
    need(rules['feature_count']==97 and rules['categorical_feature_count']==25
         and rules['baseline_feature_count']==95 and rules['original95_preserved'] is True
         and rules['existing_95_values_dtypes_order_and_index_unchanged'] is True, 'Rule-level preservation/schema differs')
    need(rules['grid_degrees']==.005 and rules['grid_shift_degrees']==.0025
         and rules['coordinate_input']=='existing float32 geography coordinates cast to float64'
         and rules['regular_grid_formula']=='floor(lat_float64 / 0.005):floor(lon_float64 / 0.005)'
         and rules['shifted_grid_formula']=='floor((lat_float64 + 0.0025) / 0.005):floor((lon_float64 + 0.0025) / 0.005)'
         and rules['missing_grid']=='__MISSING__'
         and rules['unknown_query_grid_policy']=='keep computed string; no fit-vocabulary filtering'
         and rules['original_coarse_grid_policy']=='preserve existing geo_grid_002; never rebuild from float32'
         and rules['grid_rounding_epsilon_or_boundary_adjustment'] is False
         and rules['fit_grid_counts_or_vocabulary_filter'] is False, 'Fixed grid rules differ')
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
         and audit['added_feature_names']==model['added_feature_names']==ADDED_FEATURES,
         'Independent feature declaration differs')
    need(audit['original_95_values_dtypes_order_index_preserved'] is True,
         'Independent original model-view preservation was not confirmed')
    need(audit['changed_parameter_names']==[]
         and audit['native_categorical_feature_count']==model['native_categorical_feature_count']==25,
         'Independent parameter or native category declaration differs')
    need(audit['raw_z_units']=='million_TWD' and audit['native_learning_rate_comparison_rtol']==1e-7
         and audit['native_learning_rate_comparison_atol']==0., 'Independent raw units or tolerance differ')
    evidence=audit['independent_fit_state_evidence']
    need(evidence['base_feature_count']==95 and evidence['feature_count']==97
         and evidence['categorical_count']==25 and evidence['fit_rows']==model['fit_rows']
         and evidence['removed_feature_names']==[] and evidence['added_feature_names']==ADDED_FEATURES
         and evidence['fit_state_reconstructed_independently'] is True
         and evidence['fit_model_feature_values_sha256']==model['fit_model_feature_values_sha256'],
         'Independent fit-only feature-state reconstruction differs')
    need(evidence['grids_reconstructed_independently_from_stored_float32'] is True,
         'Fine grids must be independently reconstructed')
    digest=evidence['fit_added_grid_values_sha256']
    need(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
         'Invalid independent added-grid frame digest')
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
    need(set(found)=={'RESEARCH_FILES','RESEARCH_EXPECTED_SHA'},'Missing literal R21 research declarations')
    seeds,expected=found['RESEARCH_FILES'],found['RESEARCH_EXPECTED_SHA']
    need(isinstance(seeds,(tuple,list)) and len(seeds)>0 and len(set(seeds))==len(seeds)
         and isinstance(expected,dict) and set(seeds)==set(expected), 'Research seed and expected map inventories differ')
    return {name:expected[name] for name in seeds}


def validate_research(research,dependencies,declared,inventory):
    """Compare frozen pure-X provenance; never recompute inventory or price scores."""
    need(isinstance(research,dict) and research and research==declared, 'Frozen research differs from literal driver map')
    seeds=list(declared)
    need(INVENTORY_PATH in seeds and INVENTORY_PATH.replace('.json','.sha256') in seeds,
         'Inventory JSON and receipt must be frozen research seeds')
    need(all(dependencies.get(k)==v for k,v in research.items()), 'Research outside frozen dependencies')
    expected={name:research[name] for name in seeds}
    def add(name,digest):
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid research identity')
        need(name not in expected or expected[name]==digest, 'Conflicting pure-X provenance')
        expected[name]=digest
    for name,digest in expected.copy().items(): add(name,digest)
    need(inventory['round']==21 and inventory['status']=='completed_pure_X_only'
         and isinstance(inventory['scope'],dict) and isinstance(inventory['rules'],dict), 'Wrong pure-X inventory contract')
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in inventory[field].items(): add(name,digest)
    for item in inventory['inputs'].values():
        need(type(item['bytes']) is int and item['bytes']>=0, 'Invalid research input size')
        add(item['path'],item['sha256'])
    synthetic=inventory['synthetic_evidence']
    add(synthetic['path'],synthetic['sha256'])
    need(synthetic['path'].endswith('.json') and synthetic['path'][:-5]+'.sha256' in research,
         'Pure-X synthetic evidence receipt omitted')
    need(research==expected, 'Frozen research closure differs from seeds and inventory evidence')
    return {'frozen_research_dependency_count':len(research),'dependency_sha256':copy.deepcopy(research),
      'required_research_seed_paths':list(seeds),'research_type':'pure_X_geographic_grid_support_inventory',
      'inventory_path':INVENTORY_PATH,'inventory_sha256':research[INVENTORY_PATH],
      'inventory_status':inventory['status'],'inventory_scope':copy.deepcopy(inventory['scope']),
      'inventory_alignment':copy.deepcopy(inventory['alignment']),'inventory_rules':copy.deepcopy(inventory['rules']),
      'inventory_limitations':copy.deepcopy(inventory['limitations']),
      'inventory_synthetic_evidence':copy.deepcopy(inventory['synthetic_evidence']),
      'report_recomputed_inventory_or_price_metrics':False}


def render(payload):
    stages=payload['stages'];completed=[s for s in stages if s['metrics'] is not None]
    decision='三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines=['# 第二十一輪結果：固定細網格的地理 CatBoost','',
        f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。','',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有读取預測陣列內容、重訓或重算 MAE。','',
        '## 實際處理與唯一候選','',
        '- 沿用移除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。完整列去重依原始欄位（含價格、排除編號）進行，沒有以 87 特徵欄重新去重，也沒有刪除高價或難預測列。',
        '- 原 87 欄加既有 8 地理欄，共 95 欄／23 類別全部保留；再於末尾新增 geo_grid_0p005、geo_grid_0p005_shift 兩個類別欄，模型合計 97 欄／25 類別。原95欄的值、型別、列序及欄序不變，原 87 欄仍用於 reference 與評分分組。',
        '- 細網格只從既有 geo 的 float32 緯經度轉成 float64；一般格為 floor(lat/0.005):floor(lon/0.005)，平移格為 floor((lat+0.0025)/0.005):floor((lon+0.0025)/0.005)，兩軸皆向正方向平移。以十進位整數與冒號組成類別字串，不加 epsilon、不四捨五入、不搜尋格寬或起點。',
        '- 缺座標時兩欄為 __MISSING__。未曾在 fit 出現的 query 格仍保留計算字串，不用 fit 次數門檻或詞彙表過濾。原 geo_grid_002 的 0.02 度粗網格原封保留，沒有用已轉成 float32 的座標重新建立它。',
        '- 所有原 R6 傳入參數相同：2,999 棵樹、learning_rate=0.065、MAE、Plain、depth 8、seed 42、4 threads；changed_parameter_names=[]、removed_feature_names=[]，僅 added_feature_names 指定上述兩欄。',
        '- 目標仍為 y_TWD/1e6、uniform，Pool 不傳入 sample_weight；預測只以 max(z×1e6,0) 還原總價。無 eval_set、早停、額外價格上限或校正。',
        '- 新 raw 地理 CatBoost 替換 v3 的 37.5% 地理模型槽；37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保留。先還原 TWD 再融合，原 reference trend 只施加一次。',
        '- feature_state 只接受 fit X／geo／categories，不接受 y 或 query。價格模型訓練不使用 query labels；共享 loader 會反序列化含 target 的快取，不能聲稱從未載入驗證價格。test-X 建案比例只用於固定加權評分，沒有用作本模型訓練權重。','',
        '研究依據是計畫綁定的純 X 地理格支援盤點及其 source／JSON／receipt／方法文件；完整來源與輸入 SHA 保存在機器報告，不固定依賴檔案總數，也不沿用前輪研究清單。盤點限指定的地理欄與保存索引，沒有用價格或新模型誤差選格寬。本報告不重做盤點。每個回測窗口沿用其原本對應年度的地理資料。純X支援盤點只描述fit/query網格覆蓋與座標重複，不使用建案身分；同格筆數不能視為獨立建案支持，亦不能由可用率推定價格改善。實際支持分布與限制見下方方法文件及盤點JSON。','',
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
        lines+=[f"原生模型證據：boosting_type={capacity['actual_boosting_type']}、depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹、最大 {capacity['maximum_tree_leaf_count']} 葉（上限 256）、learning_rate={capacity['actual_learning_rate']:.12g}。學習率對 0.065 使用 rtol=1e-7、atol=0 容許儲存誤差；另核對 25 個原生類別欄位置。完整葉數向量的 LE-int64 SHA 保存在機器報告；這份報告本身不重新載入模型。",'']
    lines+=['## 解讀與防過擬合限制','',
        '兩個相近而不同起點的固定網格提供局部位置類別，可能緩和單一格線的邊界切割；但格子更細並不提高原始地理位置精度。座標多為道路或交會處代表點，不是住宅或門牌的精確測量，同路多筆資料可能共用位置。0.005 度是固定角度格寬，不是精確等距或地址誤差上限。','',
        '新增類別可能增加空間細節，也可能增加稀疏性與過擬合；相鄰格仍是不同類別，不自動具有距離平滑性。原0.02度粗格保留作為既有訊號，但不保證回退行為。即使缺座標時兩個新增欄位都是缺值，重訓後的預測仍不保證與v3逐位元相同。','',
        '原95欄、raw目標、uniform權重、原R6全部參數、融合及趨勢規則均不變，沒有依本輪價格結果調格寬、平移量或支援次數門檻。相同seed不保證GPU跨次訓練逐位元一致；保存模型的完整精確重播是另一項驗證。','',
        '開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。','',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。1% 是工程門檻，不是統計顯著性；不能消除跨輪選型偏差或保證 Private 成績。沒有依 Public 成績調參。','',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。','',
        '## 完整紀錄','',
        '- [方法與歷史審查](round21_method_review.md)、[事前規約](round21_plan.md)。',
        '- [獨立流程審查](round21_pipeline_review.md)、[工作紀錄](round21_work_log.md)。',
        '- [純X地理支援盤點](../outputs/round21_research/grid_audit.json)。',
        '- [機器可讀報告](round21_results.json)、[凍結計畫](../outputs/round21/candidate_plan.json)、[最終判定](../outputs/round21/accepted_candidate.json)。',
        '- 各期 JSON 保存原始群組／季度摘要與獨立重播證據；proof 的 status=passed 表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。','']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round21_report.md', ROOT/'reports/round21_results.json', ROOT/'reports/round21_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round21-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 21
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_dependency_sha256'].items()),
         'Research dependencies are outside the frozen input map')
    expected=research_expectations((ROOT/'src/round21_experiment.py').read_text(encoding='utf-8-sig'))
    inventory=locked(ROOT/INVENTORY_PATH)
    research_summary=validate_research(plan['research_dependency_sha256'],plan['dependency_sha256'],expected,inventory)
    mapping(plan['research_dependency_sha256'])
    for item in inventory['inputs'].values():
        need((ROOT/item['path']).stat().st_size==item['bytes'], 'Pure-X inventory input byte size changed')
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
            need(result['round'] == 21 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 21 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            feature_evidence = validate_replay_feature_evidence(audit, model)
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round21_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity'],
                    'removed_feature_names': list(audit['removed_feature_names']),
                    'added_feature_names': list(audit['added_feature_names']),
                    'independent_fit_state_evidence': feature_evidence})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 21, 'candidate': CANDIDATE,
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
    staging = ROOT/'reports'/('.round21-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Complete bounded synthetic contract/render checks; no real evidence read."""
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (RuntimeError,KeyError,ValueError,SyntaxError):checks.append(name)
        else:raise AssertionError(name+' not rejected')
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'random_seed':42,'thread_count':4,
              'loss_function':'MAE','boosting_type':'Plain','l2_leaf_reg':6}
    params=copy.deepcopy(baseline)
    rules={'removed_feature_names':[],'iterations':2999,'geo_encoder_rules':{'iterations':2999},
           'native_learning_rate_comparison_rtol':1e-7,'native_learning_rate_comparison_atol':0.0,
           'geo_encoder_rule_training_fields_are_baseline_provenance_only':True}
    rules.update(feature_count=97,categorical_feature_count=25,baseline_feature_count=95,original95_preserved=True,
        existing_95_values_dtypes_order_and_index_unchanged=True,added_feature_names=list(ADDED_FEATURES),
        grid_degrees=.005,grid_shift_degrees=.0025,coordinate_input='existing float32 geography coordinates cast to float64',
        regular_grid_formula='floor(lat_float64 / 0.005):floor(lon_float64 / 0.005)',
        shifted_grid_formula='floor((lat_float64 + 0.0025) / 0.005):floor((lon_float64 + 0.0025) / 0.005)',
        missing_grid='__MISSING__',unknown_query_grid_policy='keep computed string; no fit-vocabulary filtering',
        original_coarse_grid_policy='preserve existing geo_grid_002; never rebuild from float32',
        grid_rounding_epsilon_or_boundary_adjustment=False,fit_grid_counts_or_vocabulary_filter=False)
    plan={'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
          'changed_parameter_names':list(CHANGED_PARAMETERS),'feature_rules':copy.deepcopy(rules),
          'removed_feature_names':[],'sample_weight':'uniform','weight_argument_passed_to_pool':False,
          'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
          'native_learning_rate_contract':copy.deepcopy(NATIVE_LEARNING_RATE_CONTRACT)}
    cap={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
         'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain',
         'actual_learning_rate':.065}
    model={'round':21,'stage':'development','family':'catboost','component':'geography_fine_grid_catboost','target_mode':'raw',
           'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
           'changed_parameter_names':list(CHANGED_PARAMETERS),'removed_feature_names':[],
           'iterations':2999,'actual_iterations':2999,'feature_count':97,'categorical_feature_count':25,
           'native_categorical_feature_count':25,'sample_weight':'uniform','weight_argument_passed_to_pool':False,
           'fit_rows':7,'fit_model_feature_values_sha256':'c'*64,'feature_rules':copy.deepcopy(rules),
           'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
           'artifact_sha256':dict.fromkeys(MODEL_FILES,'b'*64),'raw_z_diagnostics':{'negative_count':1,'zero_count':0},
           'runner_elapsed_seconds':1.,'fitted_actual_learning_rate':.065,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,
           'geo_encoder_rule_training_fields_are_baseline_provenance_only':True,**cap}
    model.update(added_feature_names=list(ADDED_FEATURES),baseline_feature_count=95,original95_preserved=True)
    plan.update(added_feature_names=list(ADDED_FEATURES),model_hyperparameter_change={})
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied','extra_price_cap_applied',
                'reference_or_trend_applied_here','reference_features_changed'):model[key]=False
    validate_model(model,plan,'development');checks.append('raw97_25_Plain_exact_R6_parameters')
    for name,patch in [('nativeOrdered',{'actual_boosting_type':'Ordered'}),('wrong91features',{'feature_count':91}),
                       ('wrong19cats',{'categorical_feature_count':19}),('native22cats',{'native_categorical_feature_count':22}),
                       ('unit_target',{'target_mode':'unit'}),('weighted_Pool',{'weight_argument_passed_to_pool':True}),
                       ('unit_inverse',{'inverse_formula':'max(z * area * 1e4, 0)'}),
                       ('wrong_tree_count',{'actual_trees':5998}),('native_wrong_learning_rate',{'actual_learning_rate':.0325}),
                       ('native_nan_learning_rate',{'actual_learning_rate':float('nan')}),
                       ('changed_rtol',{'native_learning_rate_comparison_rtol':1e-6}),
                       ('changed_atol',{'native_learning_rate_comparison_atol':1e-8}),
                       ('extra_calibration',{'calibration_applied':True})]:
        bad=copy.deepcopy(model);bad.update(patch)
        rejects(name,lambda m=bad:validate_model(m,plan,'development'))
    float32_cap=copy.deepcopy(cap);float32_cap['actual_learning_rate']=0.06499999761581421
    validate_capacity(float32_cap);checks.append('actual_float32_learning_rate_accepted')
    bad=copy.deepcopy(model);bad['parameters']['l2_leaf_reg']=7
    bad_plan=copy.deepcopy(plan);bad_plan['parameters']=copy.deepcopy(bad['parameters'])
    rejects('third_parameter_change_rejected',lambda:validate_model(bad,bad_plan,'development'))
    bad=copy.deepcopy(model);bad['changed_parameter_names']=['iterations']
    rejects('parameter_change_declaration_rejected',lambda:validate_model(bad,plan,'development'))
    bad=copy.deepcopy(plan);bad['native_learning_rate_contract']['expected']=.0325
    rejects('plan_native_contract_changed',lambda:validate_model(model,bad,'development'))
    bad=copy.deepcopy(model);bad['geo_encoder_rule_training_fields_are_baseline_provenance_only']=False
    rejects('baseline_iterations_misdeclared_as_current',lambda:validate_model(bad,plan,'development'))
    evidence={'fit_rows':7,'base_feature_count':95,'feature_count':97,'categorical_count':25,'removed_feature_names':[],
              'fit_state_reconstructed_independently':True,'fit_model_feature_values_sha256':'c'*64}
    evidence.update(added_feature_names=list(ADDED_FEATURES),fit_added_grid_values_sha256='d'*64,
        grids_reconstructed_independently_from_stored_float32=True)
    proof={'added_feature_names':list(ADDED_FEATURES),'removed_feature_names':[],'changed_parameter_names':list(CHANGED_PARAMETERS),
           'original_95_values_dtypes_order_index_preserved':True,'raw_z_units':'million_TWD',
           'native_categorical_feature_count':25,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,'independent_fit_state_evidence':copy.deepcopy(evidence)}
    validate_replay_feature_evidence(proof,model);checks.append('independent_original95_plus2_fit_only_evidence_match')
    for key,value in [('raw_z_units','ten_thousand_TWD_per_effective_square_metre'),
                      ('native_categorical_feature_count',22),('native_learning_rate_comparison_rtol',1e-5),
                      ('changed_parameter_names',['learning_rate']),('original_95_values_dtypes_order_index_preserved',False)]:
        bad=copy.deepcopy(proof);bad[key]=value
        rejects('proof_'+key,lambda q=bad:validate_replay_feature_evidence(q,model))
    bad=copy.deepcopy(proof);bad['independent_fit_state_evidence']['fit_model_feature_values_sha256']='d'*64
    rejects('independent_fit_hash_mismatch',lambda:validate_replay_feature_evidence(bad,model))
    audit_source='src/round21_grid_audit.py';method='reports/round21_method_review.md'
    prior='outputs/round21_research/method_before_audit.md';synth='outputs/round21_research/self_check.json'
    research={INVENTORY_PATH:'a'*64,INVENTORY_PATH[:-5]+'.sha256':'b'*64,audit_source:'c'*64,
      method:'d'*64,prior:'e'*64,synth:'f'*64,synth[:-5]+'.sha256':'1'*64,
      'synthetic/geo.pkl':'2'*64,'synthetic/fit_indices.npy':'3'*64}
    inventory={'round':21,'status':'completed_pure_X_only','rules':{'grid_degrees':.005},
      'scope':{'pure_X_only':True},'alignment':{'fit_rows':7,'query_rows':4},'limitations':['synthetic limitation'],
      'source_sha256':{audit_source:'c'*64},'evidence_sha256':{prior:'e'*64},
      'synthetic_evidence':{'path':synth,'sha256':'f'*64,'checks':5},
      'inputs':{'geo':{'path':'synthetic/geo.pkl','sha256':'2'*64,'bytes':20},
                'fit_indices':{'path':'synthetic/fit_indices.npy','sha256':'3'*64,'bytes':8}}}
    source='RESEARCH_FILES = '+repr(tuple(research))+'\nRESEARCH_EXPECTED_SHA = '+repr(research)
    expected=research_expectations(source)
    summary=validate_research(research,dict(research),expected,inventory)
    ok('dynamic_pure_X_research_closure',summary['frozen_research_dependency_count']==len(research)
       and summary['inventory_scope']==inventory['scope'] and not summary['report_recomputed_inventory_or_price_metrics'])
    rejects('missing_research_dependency',lambda:validate_research(research,{},expected,inventory))
    bad=copy.deepcopy(expected);bad[audit_source]='9'*64
    rejects('driver_map_identity_mismatch',lambda:validate_research(research,research,bad,inventory))
    bad=copy.deepcopy(inventory);bad['inputs']['geo']['sha256']='9'*64
    rejects('inventory_input_hash_conflict',lambda:validate_research(research,research,expected,bad))
    bad=copy.deepcopy(inventory);bad['inputs']['geo']['bytes']=-1
    rejects('inventory_input_negative_bytes',lambda:validate_research(research,research,expected,bad))
    bad=copy.deepcopy(inventory);bad['source_sha256']['unlisted_source.py']='9'*64
    rejects('unfrozen_inventory_source',lambda:validate_research(research,research,expected,bad))
    bad=copy.deepcopy(inventory);bad['status']='pending'
    rejects('unfinished_inventory',lambda:validate_research(research,research,expected,bad))
    for receipt in (INVENTORY_PATH[:-5]+'.sha256',synth[:-5]+'.sha256'):
        bad=copy.deepcopy(research);bad.pop(receipt)
        rejects('missing_receipt_'+receipt,lambda bad=bad:validate_research(bad,bad,bad,inventory))
    rejects('missing_research_declaration',lambda:research_expectations('OTHER = {}'))
    rejects('dynamic_declaration_not_executed',lambda:research_expectations('RESEARCH_FILES=call()\nRESEARCH_EXPECTED_SHA={}'))
    for key,value in [('added_feature_names',list(reversed(ADDED_FEATURES))),('baseline_feature_count',97),('original95_preserved',False)]:
        bad=copy.deepcopy(model);bad[key]=value
        rejects('model_'+key,lambda bad=bad:validate_model(bad,plan,'development'))
    bad=copy.deepcopy(model);bad['feature_rules']['grid_degrees']=.01
    bad_plan=copy.deepcopy(plan);bad_plan['feature_rules']=copy.deepcopy(bad['feature_rules'])
    rejects('changed_grid_width',lambda:validate_model(bad,bad_plan,'development'))
    for key,value in [('grids_reconstructed_independently_from_stored_float32',False),('fit_added_grid_values_sha256','invalid')]:
        bad=copy.deepcopy(proof);bad['independent_fit_state_evidence'][key]=value
        rejects('proof_'+key,lambda bad=bad:validate_replay_feature_evidence(bad,model))
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
        ok(name+'_raw97_native7_fixed_grids',all(x in text for x in ['boosting_type=Plain','95 欄／23 類別','97 欄／25 類別','geo_grid_0p005_shift','百萬元','uniform'])
           and all(x not in text for x in ['A/100','effective_area','round20','重跑紀錄','194.935','5,998','0.0325','11 份','四欄消融','91 欄']))
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
    ok('grid_precision_and_support_limitations',all(x in text for x in ['不提高原始地理位置精度','同格筆數不能視為獨立建案支持','不保證與v3逐位元相同','不能由可用率推定價格改善']))
    ok('coarse_grid_preserved_positive_shift_no_rounding',all(x in text for x in ['geo_grid_002','0.02 度粗網格原封保留','兩軸皆向正方向平移','不加 epsilon','float32']))
    ok('known_public_unknown_new_no_release',all(x in text for x in ['2,883,250.4138','Public 與 Private 皆未知','沒有上傳或發布新提交檔']))
    ok('MAE_percentage_direction',delta(100,101)==1 and delta(100,99)==-1 and delta(0,1) is None
       and percent(None)=='—' and number(None)=='—')
    ok('native7_and_model_summary_fields_complete',len(CAPACITY_FIELDS)==7 and all(k in MODEL_FIELDS for k in cap)
       and set(MODEL_FIELDS)<=set(model))
    return {'status':'passed','synthetic_only':True,'checks':checks,'check_count':len(checks),
        'complete_report_self_check_executed':True,'real_JSON_labels_predictions_or_models_read':False,
        'report_files_written':False,'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['generate','self-check'])
    args=parser.parse_args()
    if args.action=='self-check':print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    else:generate()
