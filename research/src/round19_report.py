"""Render finalized R19 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round19'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'unit_catboost_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fit_effective_area_sha256', 'fit_sample_weights_sha256', 'fit_factor_sha256',
    'fit_weight_diagnostics', 'native_categorical_feature_count',
    'geo_rule_target_weight_fields_are_provenance_only') + CAPACITY_FIELDS
REMOVED_FEATURES = []
TARGET_FORMULA = 'y_TWD / (effective_area(X) * 1e4)'
INVERSE_FORMULA = 'max(z * (effective_area(X) * 1e4), 0)'
WEIGHT_FORMULA = 'effective_area / 100'
RAW_Z_UNITS = 'ten_thousand_TWD_per_effective_square_metre'
AREA_RULE = 'finite positive building_area; else finite positive parking_area; else 1'
AREA_HASH_FIELDS = ('fit_effective_area_sha256', 'fit_sample_weights_sha256', 'fit_factor_sha256')
RESEARCH_FILES = ('src/round19_area_audit.py', 'outputs/round19_research/area_audit.json',
                  'outputs/round19_research/area_audit.sha256')
DOCUMENTS = ('reports/round19_plan.md', 'reports/round19_area_method_review.md',
    'reports/round19_pipeline_review.md')
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


def validate_model(model, plan, stage):
    need(model['round'] == 19 and model['stage'] == stage and model['family'] == 'catboost'
         and model['component'] == 'geography_unit_catboost' and model['target_mode'] == 'unit', 'Wrong model identity')
    params, baseline = model['parameters'], model['baseline_parameters']
    need(params == baseline == plan['parameters'] == plan['baseline_parameters']
         and params['depth'] == 8 and params['boosting_type'] == 'Plain'
         and model['changed_parameter_names'] == plan['changed_parameter_names'] == [],
         'All passed parameters must equal the original R6 baseline')
    need(model['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES
         and plan['feature_rules']['removed_feature_names'] == REMOVED_FEATURES,
         'R19 cannot remove model features')
    need(params['iterations'] == 2999 and params['random_seed'] == 42 and params['thread_count'] == 4
         and params['loss_function'] == 'MAE', 'Wrong fixed CatBoost recipe')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['feature_count'] == 95 and model['categorical_feature_count'] == 23
         and model['native_categorical_feature_count'] == 23, 'Wrong model size/schema')
    need(model['sample_weight'] == plan['sample_weight'] == WEIGHT_FORMULA
         and model['weight_argument_passed_to_pool'] is plan['weight_argument_passed_to_pool'] is True,
         'Wrong effective-area-weighted Pool training contract')
    need(model['feature_rules'] == plan['feature_rules'], 'Feature rules differ')
    need(model['target_formula'] == plan['target_formula'] == TARGET_FORMULA
         and model['inverse_formula'] == plan['inverse_formula'] == INVERSE_FORMULA, 'Wrong unit target/inverse scale')
    need(model['geo_rule_target_weight_fields_are_provenance_only'] is True,
         'Nested old geography target/weight rules must be provenance only')
    validate_weight_evidence(model)
    need(set(model['artifact_sha256']) == MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    for key in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'calibration_applied',
                'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: ' + key)


def validate_weight_evidence(model):
    for key in AREA_HASH_FIELDS:
        digest=model[key]
        need(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
             'Invalid fit area/weight/factor digest')
    diagnostic=model['fit_weight_diagnostics']
    need(diagnostic['rows']==model['fit_rows'] and diagnostic['effective_area_rule']==AREA_RULE
         and diagnostic['weights_normalized_to_mean_one'] is False, 'Fit area/weight diagnostic scope differs')
    counts=diagnostic['effective_area_source_counts']
    need(set(counts)=={'building_area','parking_area_fallback','default_one_fallback'}
         and all(type(n) is int and n>=0 for n in counts.values())
         and sum(counts.values())==model['fit_rows'], 'Area fallback counts do not partition fit rows')
    for key in ('effective_area','sample_weight','price_factor'):
        item=diagnostic[key]
        need(set(item)=={'min','max','mean'} and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in item.values())
             and item['min']<=item['max'], 'Invalid fit area/weight/factor summary')


def validate_replay_feature_evidence(audit, model):
    need(audit['removed_feature_names'] == model['removed_feature_names'] == [],
         'Independent feature-removal declaration differs')
    need(audit['original_95_values_dtypes_order_index_preserved'] is True,
         'Independent original model-view preservation was not confirmed')
    evidence = audit['independent_fit_state_evidence']
    need(evidence['base_feature_count'] == evidence['feature_count'] == 95
         and evidence['categorical_count'] == model['native_categorical_feature_count'] == 23
         and evidence['fit_rows'] == model['fit_rows'] and evidence['removed_feature_names'] == []
         and evidence['fit_state_reconstructed_independently'] is True
         and evidence['fit_model_feature_values_sha256'] == model['fit_model_feature_values_sha256'],
         'Independent fit-only feature-state reconstruction differs')
    area = audit['independent_area_weight_evidence']
    need(area == evidence['independent_area_weight_evidence'], 'Top-level/nested area proof differs')
    for key in (*AREA_HASH_FIELDS, 'fit_weight_diagnostics', 'target_formula', 'inverse_formula', 'sample_weight'):
        need(area[key] == model[key], 'Independent area/weight proof differs: '+key)
    need(area['raw_z_units'] == RAW_Z_UNITS and audit['raw_z_units'] == RAW_Z_UNITS,
         'Unit output was mislabeled as raw million TWD')
    for key in ('area_weight_factor_recomputed_from_original_X',
                'Pool_weight_usage_verified_from_frozen_source_and_saved_metadata', 'historical_fit_not_reexecuted'):
        need(area[key] is True, 'Area/weight evidence scope differs: '+key)
    return copy.deepcopy(evidence)


def validate_research(research, dependencies, record):
    """Check saved research declarations and every frozen edge; no input-array statistics."""
    need(isinstance(research,dict) and set(RESEARCH_FILES).issubset(research)
         and all(dependencies.get(k)==v for k,v in research.items()), 'Research closure differs')
    for name,digest in research.items():
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid research identity')
    need(record['round']==19 and record['status']=='target_free_effective_area_support_audit'
         and record['feature_values_modified'] is False and record['prices_None_R1_transform_used'] is True,
         'Area research identity/data-use differs')
    scope=record['read_scope']
    need(scope['target_cells_decoded']==0 and scope['train_test_target_free_caches_only'] is True
         and scope['query_test_weights_descriptive_not_training'] is True,
         'Research was not target-free/descriptive')
    for key in ('MAE_computed','candidate_fitted','model_or_prediction_loaded','raw_price_cache_or_Excel_loaded','valid_dataset_loaded'):
        need(scope[key] is False, 'Unexpected area research use: '+key)
    alignment=record['alignment']
    need(alignment['canonical_rows']==455255 and alignment['fit_rows']==240316 and alignment['query_rows']==81234
         and alignment['indices_and_dates_match_released_R6'] is True
         and alignment['original_saved_mapping_reused'] is True
         and alignment['target_dedup_recomputed_here'] is False, 'Research mapping declaration differs')
    expanded=set(RESEARCH_FILES)
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in record[field].items():
            need(research.get(name)==digest, 'Research source/evidence not bound');expanded.add(name)
    for name,item in record['inputs'].items():
        need(name==item['path'] and research.get(name)==item['sha256']
             and type(item['bytes']) is int and item['bytes']>=0, 'Research input identity differs')
        expanded.add(name)
    need(expanded==set(research), 'Frozen research includes missing or undeclared edges')
    return {'frozen_research_dependency_count':len(research), 'dependency_sha256':copy.deepcopy(research),
            'read_scope':copy.deepcopy(scope),'alignment':copy.deepcopy(alignment),
            'research_summary_not_used_as_backtest_fit_rows':True,
            'report_did_not_load_research_arrays_or_recompute_statistics':True}


def render(payload):
    stages = payload['stages']; completed = [s for s in stages if s['metrics'] is not None]
    decision = '三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines = ['# 第十九輪結果：面積加權單價地理 CatBoost', '', f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。', '',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有讀取預測陣列內容、重訓或重算 MAE。', '',
        '## 實際處理與唯一候選', '',
        '- 沿用去除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。歷史完整列去重包含價格；本輪沒有額外刪除高價或難預測列。',
        '- 原 87 欄加既有 8 地理欄，共 95 欄／23 類別全部保留；沒有移除建案、地址、棟戶或 building_block。值、型別、列序及欄序保持不變；沿用同一套年度地理規則。',
        '- CatBoost Pool 使用原 23 個類別欄；feature_state.json 的 nested geo_state 與模型視圖均為 95 欄，只用 fit X／geo，不接受 query 或 target。原 87 欄仍用於 reference 與建案分組；完整列去重依原始資料欄位（含價格、排除編號）進行，沿用凍結索引。',
        '- 固定 CatBoost MAE、2,999 輪、seed 42、4 threads；Plain、depth 8 及全部傳入參數與原 R6 完全相等，changed_parameter_names=[]，removed_feature_names=[]。只改目標及 Pool 權重：A 取有限正 building_area，否則有限正 parking_area，再否則 1；target=y_TWD/(A×1e4)，weight=A/100（不正規化為均值1），還原 max(z×A×1e4,0)。不用 eval_set 或早停，不加價格上限或校正。',
        '- 以本輪 unit CatBoost 還原總價後替換 v3 的 37.5% 地理 raw CatBoost 槽；另 37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保持不變。融合使用 TWD，原 reference trend 只施加一次；不能再乘1e6。model.cbm 保存原生 CatBoost 模型。',
        '- 訓練不使用 query labels；共享 loader 實際會反序列化含 target 的快取，不能聲稱從未載入驗證價格。固定 test-X 建案分組比例僅用於加權評分，與本輪實際傳入 Pool 的面積權重是兩回事。', '',
        '事前有效面積研究以純 X 特徵檢查既有 R1 分母規則、備援及面積尺度，不使用價格決定面積界限，也不把研究全資料盤點當作每期 canonical fit 覆蓋。研究來源／資料／紀錄均按計畫雜湊保存；報告不重算研究統計。各期下列面積備援數只取自該期 fit 的保存 metadata，與獨立重播證據相等。nested 舊 R6 地理規則中的 raw／uniform 欄位只代表原始來源，實際模型以頂層 unit／A/100 契約為準。', '',
        '## 固定年度結果', '',
        'MAE 是融合及一次 reference 後、未取整總價的新臺幣誤差；變化為正表示退步。未執行不是零誤差或通過。', '',
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
        diagnostic = stage['model']['raw_z_diagnostics']
        lines += ['', f"已重新載入原生 CBM 與保存的特徵狀態，重播全部 {stage['query_rows']:,} 列；另核對總價逆轉換、v3／候選融合、保存評分與門檻。原始 z<0 為 {diagnostic['negative_count']:,} 列，z=0 為 {diagnostic['zero_count']:,} 列。z 是每有效平方公尺的萬元模型輸出（10,000 TWD/m²）；A=1 備援不認證真實住宅單價。這些計數僅作診斷，未用於調參。", '']
        fallback = stage['model']['fit_weight_diagnostics']['effective_area_source_counts']
        lines += [f"該期 fit 有效面積來源：building_area {fallback['building_area']:,} 列、parking_area 備援 {fallback['parking_area_fallback']:,} 列、固定 1 備援 {fallback['default_one_fallback']:,} 列。A、A/100 權重、A×1e4 因子的逐列 LE-float64 SHA 與正值統計由獨立重播核對；原生模型亦核對 23 個類別欄的位置。", '']
        capacity = stage['independent_review']['native_model_capacity']
        lines += [f"獨立重播證據核對原生模型：boosting_type={capacity['actual_boosting_type']}、depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹，最大葉節點數 {capacity['maximum_tree_leaf_count']}（上限 256）。完整葉節點數向量的 little-endian int64 SHA 保存在機器報告；本報告只核對已保存證據，不重新載入模型。", '']
    lines += ['## 解讀與防過擬合限制', '',
        '本輪原 95／23、depth 8、Plain、2,999 輪及全部傳入參數固定，但目標參數化與訓練權重改變。原生檢查確認 Plain、SymmetricTree、樹數、每樹最多 256 葉及 23 個類別欄。GPU 訓練可能有非確定性；相同 seed 不保證跨次重訓逐位元相同，保存模型的精確重播是另一項檢查。', '',
        '對未投影 p_raw=A×1e4×z，(A/100)×|y/(A×1e4)−z|=|y−p_raw|/1e6，固定 fit 上加權平均分母亦只是常數。負值還原後沿用非負投影；在 y≥0 時投影不增加絕對誤差。這不表示 raw 與 unit CatBoost 有相同函數空間、切分、類別目標統計或正則化效果，也不保證改善。R1 已做面積加權 unit CatBoost87，本輪差別為固定 R6 地理95分支；不是第一次發現損失等價。大面積會放大單價誤差，停車／土地交易的備援分母不是普遍經濟單價。沒有按這些風險刪除交易、調面積閾值或截尾；也沒有再挑欄位、深度、輪數、權重或校正。', '',
        '先固定唯一候選、公式、年度及門檻。開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。', '',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。固定工程門檻不能消除跨輪選型偏差，也不保證 Private 成績；本輪沒有掃參數、融合權重或依 Public 調參。', '',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。', '',
        '## 完整紀錄', '',
        '- [有效面積方法與純 X 研究](round19_area_method_review.md)、[事前規約](round19_plan.md)。',
        '- [獨立流程審查](round19_pipeline_review.md)、[工作紀錄](round19_work_log.md)。',
        '- [機器可讀報告](round19_results.json)、[凍結計畫](../outputs/round19/candidate_plan.json)、[最終判定](../outputs/round19/accepted_candidate.json)。',
        '- 各期 JSON 保留完整月份／城市描述與獨立重播證据；proof 的 status=passed 只表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。', '']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round19_report.md', ROOT/'reports/round19_results.json', ROOT/'reports/round19_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round19-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 19
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_dependency_sha256'].items()),
         'Research dependencies are outside the frozen input map')
    area_research = locked(ROOT/RESEARCH_FILES[1])
    research_summary = validate_research(plan['research_dependency_sha256'], plan['dependency_sha256'], area_research)
    for item in area_research['inputs'].values():
        capture(ROOT/item['path'], item['sha256'])
        need((ROOT/item['path']).stat().st_size == item['bytes'], 'Research input byte length differs')
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
            need(result['round'] == 19 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 19 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            feature_evidence = validate_replay_feature_evidence(audit, model)
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round19_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity'],
                    'removed_feature_names': list(audit['removed_feature_names']),
                    'independent_fit_state_evidence': feature_evidence,
                    'independent_area_weight_evidence': audit['independent_area_weight_evidence']})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 19, 'candidate': CANDIDATE,
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
    staging = ROOT/'reports'/('.round19-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Complete bounded synthetic contract/render checks; no real files except own source hash."""
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError(name+' not rejected')
    params={'iterations':2999,'depth':8,'random_seed':42,'thread_count':4,'loss_function':'MAE','boosting_type':'Plain'}
    rules={'removed_feature_names':[]}
    plan={'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),'changed_parameter_names':[],
          'feature_rules':rules,'removed_feature_names':[],'sample_weight':WEIGHT_FORMULA,
          'weight_argument_passed_to_pool':True,'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA}
    cap={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
         'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain'}
    diag={'rows':7,'effective_area_rule':AREA_RULE,'weights_normalized_to_mean_one':False,
          'effective_area_source_counts':{'building_area':5,'parking_area_fallback':1,'default_one_fallback':1},
          'effective_area':{'min':1.,'max':100.,'mean':30.},'sample_weight':{'min':.01,'max':1.,'mean':.3},
          'price_factor':{'min':1e4,'max':1e6,'mean':3e5}}
    model={'round':19,'stage':'development','family':'catboost','component':'geography_unit_catboost','target_mode':'unit',
           'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),'changed_parameter_names':[],
           'removed_feature_names':[],'iterations':2999,'actual_iterations':2999,'feature_count':95,
           'categorical_feature_count':23,'native_categorical_feature_count':23,'sample_weight':WEIGHT_FORMULA,
           'fit_rows':7,'fit_model_feature_values_sha256':'c'*64,'weight_argument_passed_to_pool':True,
           'feature_rules':rules,'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
           'artifact_sha256':dict.fromkeys(MODEL_FILES,'b'*64),'raw_z_diagnostics':{'negative_count':1,'zero_count':0},
           'fit_weight_diagnostics':copy.deepcopy(diag),'geo_rule_target_weight_fields_are_provenance_only':True,
           **dict.fromkeys(AREA_HASH_FIELDS,'d'*64),**cap}
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied','extra_price_cap_applied',
                'reference_or_trend_applied_here','reference_features_changed'):model[key]=False
    validate_model(model,plan,'development');checks.append('unit95_23_original_Plain_recipe_valid')
    for name,patch in [('nativeOrdered',{'actual_boosting_type':'Ordered'}),('old91',{'feature_count':91}),
                       ('old19cats',{'categorical_feature_count':19}),('native19cats',{'native_categorical_feature_count':19}),
                       ('rawtarget',{'target_mode':'raw'}),('unweighted',{'weight_argument_passed_to_pool':False}),
                       ('rawinverse',{'inverse_formula':'max(z * 1e6, 0)'}),('changedparams',{'changed_parameter_names':['depth']}),
                       ('removed_identifier',{'removed_feature_names':['project_name']})]:
        bad=copy.deepcopy(model);bad.update(patch)
        rejects(name+'_rejected',lambda m=bad:validate_model(m,plan,'development'))
    for name,key,value in [('count','rows',8),('normalized','weights_normalized_to_mean_one',True)]:
        bad=copy.deepcopy(model);bad['fit_weight_diagnostics'][key]=value
        rejects('weight_diagnostic_'+name,lambda m=bad:validate_model(m,plan,'development'))
    bad=copy.deepcopy(model);bad['fit_weight_diagnostics']['sample_weight']['min']=0
    rejects('zero_weight_diagnostic_rejected',lambda:validate_model(bad,plan,'development'))
    area={k:copy.deepcopy(model[k]) for k in (*AREA_HASH_FIELDS,'fit_weight_diagnostics','target_formula','inverse_formula','sample_weight')}
    area.update(raw_z_units=RAW_Z_UNITS,area_weight_factor_recomputed_from_original_X=True,
        Pool_weight_usage_verified_from_frozen_source_and_saved_metadata=True,historical_fit_not_reexecuted=True)
    evidence={'fit_rows':7,'base_feature_count':95,'feature_count':95,'categorical_count':23,'removed_feature_names':[],
        'fit_state_reconstructed_independently':True,'fit_model_feature_values_sha256':'c'*64,
        'independent_area_weight_evidence':copy.deepcopy(area)}
    proof={'removed_feature_names':[],'original_95_values_dtypes_order_index_preserved':True,'raw_z_units':RAW_Z_UNITS,
           'independent_fit_state_evidence':evidence,'independent_area_weight_evidence':area}
    validate_replay_feature_evidence(proof,model);checks.append('independent_unit_area_weight_and_full95_evidence_match')
    for key,value in [('raw_z_units','million_TWD'),('fit_factor_sha256','e'*64),('historical_fit_not_reexecuted',False)]:
        bad=copy.deepcopy(proof);bad['independent_area_weight_evidence'][key]=value
        rejects('independent_'+key+'_mismatch',lambda q=bad:validate_replay_feature_evidence(q,model))
    research=dict.fromkeys(RESEARCH_FILES,'a'*64);research['synthetic/input.pkl']='b'*64
    research_record={'round':19,'status':'target_free_effective_area_support_audit','feature_values_modified':False,
        'prices_None_R1_transform_used':True,'read_scope':{'target_cells_decoded':0,'train_test_target_free_caches_only':True,
            'query_test_weights_descriptive_not_training':True,'MAE_computed':False,'candidate_fitted':False,
            'model_or_prediction_loaded':False,'raw_price_cache_or_Excel_loaded':False,'valid_dataset_loaded':False},
        'alignment':{'canonical_rows':455255,'fit_rows':240316,'query_rows':81234,
            'indices_and_dates_match_released_R6':True,'original_saved_mapping_reused':True,'target_dedup_recomputed_here':False},
        'source_sha256':{RESEARCH_FILES[0]:'a'*64},'evidence_sha256':{},
        'inputs':{'synthetic/input.pkl':{'path':'synthetic/input.pkl','sha256':'b'*64,'bytes':8}}}
    summary=validate_research(research,dict(research),research_record)
    ok('all_frozen_research_edges_bound',summary['dependency_sha256']==research)
    rejects('missing_research_edge',lambda:validate_research(research,{},research_record))
    bad=copy.deepcopy(research_record);bad['read_scope']['target_cells_decoded']=1
    rejects('research_target_decode_rejected',lambda:validate_research(research,research,bad))
    bad=copy.deepcopy(research_record);bad['inputs']['synthetic/input.pkl']['sha256']='c'*64
    rejects('research_input_identity_rejected',lambda:validate_research(research,research,bad))
    metric={'rows':4,'overall_mae':10.,'weighted_mae':11.,
            'strata':{'seen':{'rows':2,'mae':10.},'new_nonmissing':{'rows':2,'mae':10.},'missing':{'rows':0,'mae':None}},
            'forecast_quarters':{q:{'rows':1,'mae':10.} for q in ('Q1','Q2','Q3','Q4')}}
    metrics={CONTROL:copy.deepcopy(metric),CANDIDATE:copy.deepcopy(metric)}
    validate_metrics(metrics,4);checks.append('all_group_quarter_rows_and_empty_mae_contract')
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
        ok(name+'_unit95_23_no_R18_remnants',all(x in text for x in ['boosting_type=Plain','95 欄／23 類別','10,000 TWD/m²','weight=A/100'])
           and all(x not in text for x in ['91 欄','19 個類別','四欄消融','round18','先於主流程','15 個來源']))
    stages=stages_for([True,True,True]);accepted,frozen=authorities([True,True,True],stages)
    bad=copy.deepcopy(accepted);bad['accepted']=False
    rejects('inconsistent_final_acceptance',lambda:validate_finalization(bad,frozen,stages))
    bad=copy.deepcopy(frozen);bad['later_stages_allowed']=False
    rejects('inconsistent_dev_freeze',lambda:validate_finalization(accepted,bad,stages))
    text=render({'accepted':False,'stages':stages_for([False,False,False])})
    ok('later_unexecuted_null_display',text.count('開發期未通過，未執行')==2)
    ok('loss_equivalence_preclip_and_risk_caveats',all(x in text for x in ['未投影 p_raw','正則化效果','放大單價誤差','不是第一次發現損失等價']))
    ok('raw_dedup_and_fit_only_context_preserved',all(x in text for x in ['含價格、排除編號','原 87 欄仍用於 reference','每期 canonical fit']))
    ok('known_public_and_unknown_future_no_release',all(x in text for x in ['2,883,250.4138','Public 與 Private 皆未知','沒有上傳或發布新提交檔']))
    ok('MAE_percentage_direction',delta(100,101)==1 and delta(100,99)==-1 and delta(0,1) is None
       and percent(None)=='—' and number(None)=='—')
    ok('all_model_summary_fields_present',all(x in MODEL_FIELDS for x in (*AREA_HASH_FIELDS,'fit_weight_diagnostics','native_categorical_feature_count',*CAPACITY_FIELDS)))
    return {'status':'passed','synthetic_only':True,'checks':checks,'check_count':len(checks),
        'complete_report_self_check_executed':True,'real_JSON_labels_predictions_or_models_read':False,
        'report_files_written':False,'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['generate', 'self-check'])
    args = parser.parse_args()
    if args.action == 'self-check': print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else: generate()
