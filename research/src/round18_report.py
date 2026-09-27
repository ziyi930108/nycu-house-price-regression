"""Render finalized R18 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round18'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'structural_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds') + CAPACITY_FIELDS
REMOVED_FEATURES = ['project_name', 'address_raw', 'building_and_unit', 'building_block']
DOCUMENTS = ('reports/round18_plan.md', 'reports/round18_method_audit.md',
    'reports/round18_feature_audit.md', 'reports/round18_error_diagnostic.md',
    'reports/round18_pipeline_review.md')
RESEARCH_FILES = ('outputs/round18_research/error_diagnostic.json',
    'src/round18_error_diagnostic.py', 'src/round18_feature_audit.py',
    'data_cache/train_features.pkl', 'data_cache/test_features.pkl', 'src/house_features.py')
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
    need(model['round'] == 18 and model['stage'] == stage and model['family'] == 'catboost'
         and model['component'] == 'geography_structural_raw_catboost' and model['target_mode'] == 'raw', 'Wrong model identity')
    params, baseline = model['parameters'], model['baseline_parameters']
    need(params == baseline == plan['parameters'] == plan['baseline_parameters']
         and params['depth'] == 8 and params['boosting_type'] == 'Plain'
         and model['changed_parameter_names'] == plan['changed_parameter_names'] == [],
         'All passed parameters must equal the original R6 baseline')
    need(model['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES
         and plan['feature_rules']['removed_feature_names'] == REMOVED_FEATURES,
         'Only the four fixed model identifiers may be removed')
    need(params['iterations'] == 2999 and params['random_seed'] == 42 and params['thread_count'] == 4
         and params['loss_function'] == 'MAE', 'Wrong fixed CatBoost recipe')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['feature_count'] == 91 and model['categorical_feature_count'] == 19, 'Wrong model size/schema')
    need(model['sample_weight'] == 'uniform' and model['weight_argument_passed_to_pool'] is False,
         'Wrong uniform Pool training contract')
    need(model['feature_rules'] == plan['feature_rules'], 'Feature rules differ')
    need(model['target_formula'] == 'y_TWD / 1e6' and model['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong raw target scale')
    need(set(model['artifact_sha256']) == MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    for key in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'calibration_applied',
                'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: ' + key)


def validate_replay_feature_evidence(audit, model):
    need(audit['removed_feature_names'] == model['removed_feature_names'] == REMOVED_FEATURES,
         'Independent removed-feature list differs')
    need(audit['remaining_91_values_dtypes_order_index_preserved'] is True,
         'Independent remaining model-view preservation was not confirmed')
    evidence = audit['independent_fit_state_evidence']
    need(evidence['base_feature_count'] == 95 and evidence['feature_count'] == 91
         and evidence['categorical_count'] == 19 and evidence['fit_rows'] == model['fit_rows']
         and evidence['removed_feature_names'] == REMOVED_FEATURES
         and evidence['fit_state_reconstructed_independently'] is True
         and evidence['fit_model_feature_values_sha256'] == model['fit_model_feature_values_sha256'],
         'Independent fit-only feature-state reconstruction differs')
    return copy.deepcopy(evidence)


def validate_research(diagnostic, research, dependencies):
    """Original diagnostic has no receipt; its bytes are bound by the frozen plan."""
    need(set(RESEARCH_FILES).issubset(research) and len(research) == 19
         and all(dependencies.get(k) == v for k, v in research.items()), 'Research closure differs')
    need(diagnostic['status'] == 'exploratory_diagnostic_not_candidate'
         and diagnostic['stage'] == 'development' and diagnostic['historically_untouched'] is False
         and diagnostic['saved_overall_mae_exact_match'] is True, 'Diagnostic scope differs')
    evidence = diagnostic['source_sha256']
    need(len(evidence) == 15 and all(research.get(k) == v for k, v in evidence.items()),
         'Diagnostic source/input inventory differs')
    scope = diagnostic['read_scope']
    need(scope['target_cells_decoded'] == 81234
         and scope['confirmation_auxiliary_valid_test_targets_decoded'] is False
         and scope['new_predictions_generated'] is False and scope['model_loaded_or_fitted'] is False,
         'Diagnostic data-use declaration differs')
    return {'status': diagnostic['status'], 'stage': diagnostic['stage'],
            'diagnostic_original_receipt_exists': False, 'diagnostic_dependencies': len(evidence),
            'frozen_research_dependencies': len(research), 'read_scope': copy.deepcopy(scope),
            'historically_untouched': False, 'new_candidate_scoring_performed_by_report': False}


def render(payload):
    stages = payload['stages']; completed = [s for s in stages if s['metrics'] is not None]
    decision = '三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines = ['# 第十八輪結果：永久移除四個識別欄的總價地理 CatBoost', '', f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。', '',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有讀取預測陣列內容、重訓或重算 MAE。', '',
        '## 實際處理與唯一候選', '',
        '- 沿用去除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。歷史完整列去重包含價格；本輪沒有額外刪除高價或難預測列。',
        '- 先以原 87 欄加既有 8 地理欄形成 95 欄，再僅在模型視圖永久移除 project_name、address_raw、building_and_unit、building_block，剩 91 欄／19 類別。其餘值、型別、列序及欄序保持不變；沿用同一套地理規則。',
        '- CatBoost Pool 使用保留的 19 個類別欄；feature_state.json 的 nested geo_state 保留原 95 欄結構，模型狀態記錄 91 欄，兩者只用 fit X／geo，不接受 query 或 target。原 87 欄仍用於 reference 與建案分組；完整列去重依原始資料欄位（含價格、排除編號）進行，沿用凍結索引。',
        '- 固定 CatBoost MAE、2,999 輪、seed 42、4 threads；Plain、depth 8 及全部傳入參數與原 R6 完全相等，changed_parameter_names=[]。只改四欄移除。目標 y_TWD/1e6，還原 max(z×1e6,0)。訓練均一權重，Pool 不傳 weight，不用 eval_set 或早停，不做面積權重、額外價格上限或校正。',
        '- 以四欄移除的模型替換 v3 的 37.5% 地理 raw CatBoost 槽；另 37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保持不變，先還原 TWD 再融合，沿用原 reference trend 一次。model.cbm 保存原生 CatBoost 模型。',
        '- 訓練不使用 query labels；共享 loader 實際會反序列化含 target 的快取，不能聲稱從未載入驗證價格。固定 test-X 建案分組比例僅用於加權評分，不是訓練 sample_weight。', '',
        '純 X 特徵稽核只檢查 train/test 特徵快取；另有既有開發期 v3 誤差診斷，讀取 81,234 個該期目標，不能把整份研究稱為無標籤或盲測。診斷沒有讀確認期／輔助期／valid／test 目標，沒有生成新預測或載入／訓練模型。原始診斷 JSON 沒有獨立收據，其 bytes 與 15 個來源／輸入、純 X 稽核配方及輸入合併為 19 個唯一研究依賴，皆由正式計畫綁定；報告沒有補造舊收據。唯一四欄消融的決定先於主流程讀取本輪新增診斷數字，不依診斷再挑欄位或門檻。', '',
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
        lines += ['', f"已重新載入原生 CBM 與保存的特徵狀態，重播全部 {stage['query_rows']:,} 列；另核對總價逆轉換、v3／候選融合、保存評分與門檻。原始 z<0 為 {diagnostic['negative_count']:,} 列，z=0 為 {diagnostic['zero_count']:,} 列。z 是百萬元模型輸出，這些計數僅作診斷，未用於調參。", '']
        capacity = stage['independent_review']['native_model_capacity']
        lines += [f"獨立重播證據核對原生模型：boosting_type={capacity['actual_boosting_type']}、depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹，最大葉節點數 {capacity['maximum_tree_leaf_count']}（上限 256）。完整葉節點數向量的 little-endian int64 SHA 保存在機器報告；本報告只核對已保存證據，不重新載入模型。", '']
    lines += ['## 解讀與防過擬合限制', '',
        '本輪只移除四個固定識別欄，模型由 95／23 縮為 91／19；depth 8、Plain、目標尺度、均一權重、2,999 輪及其餘參數固定。原生檢查確認 Plain、SymmetricTree、樹數與每樹最多 256 葉。資料表示改變可能改變切分與運算成本；相同參數不保證相同耗時。GPU 訓練可能有非確定性，相同 seed 不保證跨次重訓逐位元相同；保存模型的精確重播是另一項檢查。', '',
        '永久刪除四欄可能減少依賴案名與棟戶的類別統計，也會失去已見建案的有效訊號；粗位置與缺地理資料未必能補回。道路、地區、網格及交互仍可能辨識建案，因此不是完全無識別，也不創造新建案價格資訊。R1 的三欄忽略、R4 無地理的 spatial LightGBM、R5 整案半遮罩均屬相近歷史，不能稱移除識別方法第一次出現。本輪沒有重新挑欄位、遮罩率、boosting、深度、輪數、融合權重或價格校正；原生模式／容量合規本身不等於有效。', '',
        '先固定唯一候選、公式、年度及門檻。開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。', '',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。固定工程門檻不能消除跨輪選型偏差，也不保證 Private 成績；本輪沒有掃參數、融合權重或依 Public 調參。', '',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。', '',
        '## 完整紀錄', '',
        '- [方法與歷史比较](round18_method_audit.md)、[純 X 特徵稽核](round18_feature_audit.md)、[開發期誤差診斷](round18_error_diagnostic.md)、[事前規約](round18_plan.md)。',
        '- [獨立流程審查](round18_pipeline_review.md)、[工作紀錄](round18_work_log.md)。',
        '- [機器可讀報告](round18_results.json)、[凍結計畫](../outputs/round18/candidate_plan.json)、[最終判定](../outputs/round18/accepted_candidate.json)。',
        '- 各期 JSON 保留完整月份／城市描述與獨立重播證据；proof 的 status=passed 只表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。', '']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round18_report.md', ROOT/'reports/round18_results.json', ROOT/'reports/round18_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round18-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 18
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_dependency_sha256'].items()),
         'Research dependencies are outside the frozen input map')
    research_summary = validate_research(read(ROOT/RESEARCH_FILES[0]), plan['research_dependency_sha256'], plan['dependency_sha256'])
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
            need(result['round'] == 18 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 18 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            feature_evidence = validate_replay_feature_evidence(audit, model)
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round18_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity'],
                    'removed_feature_names': list(audit['removed_feature_names']),
                    'independent_fit_state_evidence': feature_evidence})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 18, 'candidate': CANDIDATE,
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
    staging = ROOT/'reports'/('.round18-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Bounded R18 metadata/render deltas; no actual JSON/arrays/models read."""
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejects(name,operation):
        try:operation()
        except (RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError(name+' not rejected')
    params={'iterations':2999,'depth':8,'random_seed':42,'thread_count':4,'loss_function':'MAE','boosting_type':'Plain'}
    rules={'removed_feature_names':list(REMOVED_FEATURES)}
    plan={'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),
          'changed_parameter_names':[],'feature_rules':rules,'removed_feature_names':list(REMOVED_FEATURES)}
    cap={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
         'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain'}
    model={'round':18,'stage':'development','family':'catboost','component':'geography_structural_raw_catboost','target_mode':'raw',
           'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),'changed_parameter_names':[],
           'removed_feature_names':list(REMOVED_FEATURES),'iterations':2999,'actual_iterations':2999,
           'feature_count':91,'categorical_feature_count':19,'sample_weight':'uniform','fit_rows':7,
           'fit_model_feature_values_sha256':'c'*64,'weight_argument_passed_to_pool':False,
           'feature_rules':rules,'target_formula':'y_TWD / 1e6','inverse_formula':'max(z * 1e6, 0)',
           'artifact_sha256':dict.fromkeys(MODEL_FILES,'b'*64),'raw_z_diagnostics':{'negative_count':1,'zero_count':0},**cap}
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied','extra_price_cap_applied',
                'reference_or_trend_applied_here','reference_features_changed'):model[key]=False
    validate_model(model,plan,'development');checks.append('raw91_19_Plain_R6_parameters_unchanged')
    for name,patch in [('nativeOrdered',{'actual_boosting_type':'Ordered'}),('old95schema',{'feature_count':95}),
                       ('old23categories',{'categorical_feature_count':23}),('changedparams',{'changed_parameter_names':['depth']}),
                       ('wrong_removal',{'removed_feature_names':REMOVED_FEATURES[:-1]})]:
        bad=copy.deepcopy(model);bad.update(patch)
        rejects(name+'_rejected',lambda m=bad:validate_model(m,plan,'development'))
    badplan=copy.deepcopy(plan);badplan['removed_feature_names']=REMOVED_FEATURES[::-1]
    rejects('plan_removal_order_mismatch',lambda:validate_model(model,badplan,'development'))
    proof={'removed_feature_names':list(REMOVED_FEATURES),'remaining_91_values_dtypes_order_index_preserved':True,
           'independent_fit_state_evidence':{'fit_rows':7,'base_feature_count':95,'feature_count':91,'categorical_count':19,
               'removed_feature_names':list(REMOVED_FEATURES),'fit_state_reconstructed_independently':True,
               'fit_model_feature_values_sha256':'c'*64}}
    validate_replay_feature_evidence(proof,model);checks.append('independent_95_to91_fit_evidence_matches_metadata')
    for key,value in [('feature_count',95),('fit_rows',8),('fit_model_feature_values_sha256','d'*64)]:
        bad=copy.deepcopy(proof);bad['independent_fit_state_evidence'][key]=value
        rejects('proof_'+key+'_mismatch',lambda q=bad:validate_replay_feature_evidence(q,model))
    research=dict.fromkeys(RESEARCH_FILES,'a'*64)
    research.update({f'synthetic/input{i}':'b'*64 for i in range(13)})
    diagnostic={'status':'exploratory_diagnostic_not_candidate','stage':'development','historically_untouched':False,
                'saved_overall_mae_exact_match':True,'source_sha256':dict(list(research.items())[4:]),
                'read_scope':{'target_cells_decoded':81234,'confirmation_auxiliary_valid_test_targets_decoded':False,
                              'new_predictions_generated':False,'model_loaded_or_fitted':False}}
    summary=validate_research(diagnostic,research,dict(research))
    ok('research15_plus6_merge19_no_original_receipt',summary['diagnostic_dependencies']==15
       and summary['frozen_research_dependencies']==19 and not summary['diagnostic_original_receipt_exists'])
    bad=copy.deepcopy(diagnostic);bad['read_scope']['confirmation_auxiliary_valid_test_targets_decoded']=True
    rejects('later_target_research_scope_rejected',lambda:validate_research(bad,research,research))
    metric={'rows':4,'overall_mae':10.,'weighted_mae':11.,
            'strata':{'seen':{'rows':2,'mae':10.},'new_nonmissing':{'rows':2,'mae':10.},'missing':{'rows':0,'mae':None}},
            'forecast_quarters':{q:{'rows':1,'mae':10.} for q in ('Q1','Q2','Q3','Q4')}}
    def stages_for(passes):
        rows=[]
        for i,(stage,title) in enumerate(LABELS.items()):
            present=i==0 or passes[0]
            rows.append({'stage':stage,'title':title,'dates':['start','end'],'fit_rows':7,'query_rows':4,
                'metrics':{CONTROL:copy.deepcopy(metric),CANDIDATE:copy.deepcopy(metric)} if present else None,
                'decision':{'passed':passes[i],'checks':{'saved_gate':passes[i]}} if present else None,
                'model':copy.deepcopy(model) if present else None,
                'independent_review':{'native_model_capacity':copy.deepcopy(cap)} if present else None})
        return rows
    for name,passes in [('dev_reject',[False,False,False]),('later_reject',[True,False,True]),('all_pass',[True,True,True])]:
        text=render({'accepted':all(passes),'stages':stages_for(passes)})
        ok(name+'_status',('尚待另行最終訓練與發布' in text) if all(passes) else ('本輪未採用，維持 submission_v3.csv' in text))
        ok(name+'_native_Plain91_19',all(x in text for x in ['boosting_type=Plain','depth=8','上限 256','91 欄／19 類別','model.cbm'])
           and 'Ordered' not in text and 'round17' not in text)
    text=render({'accepted':False,'stages':stages_for([False,False,False])})
    ok('unexecuted_is_null_not_zero',text.count('開發期未通過，未執行')==2)
    ok('raw_dedup_and_original87_reference_distinguished',all(x in text for x in ['含價格、排除編號','nested geo_state','原 87 欄仍用於 reference']))
    ok('diagnostic_not_falsely_claimed_targetfree',all(x in text for x in ['81,234 個該期目標','不能把整份研究稱為無標籤','沒有補造舊收據','19 個唯一研究依賴']))
    ok('residual_proxies_and_history_honest',all(x in text for x in ['不是完全無識別','R1 的三欄忽略','R5 整案半遮罩']))
    ok('six_native_fields_and_removals_in_modelsummary',len(CAPACITY_FIELDS)==6 and all(x in MODEL_FIELDS for x in CAPACITY_FIELDS)
       and 'removed_feature_names' in MODEL_FIELDS)
    return {'status':'passed','synthetic_only':True,'checks':checks,'check_count':len(checks),
        'real_JSON_labels_predictions_or_models_read':False,'report_files_written':False,'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['generate', 'self-check'])
    args = parser.parse_args()
    if args.action == 'self-check': print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else: generate()
