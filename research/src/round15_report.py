"""Render finalized R15 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round15'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'shallow_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds') + CAPACITY_FIELDS
DOCUMENTS = ('reports/round15_plan.md', 'reports/round15_shallow_geo_feasibility.md',
             'reports/round15_pipeline_review.md')
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
    need(capacity['actual_model_depth'] == 6 and capacity['actual_grow_policy'] == 'SymmetricTree'
         and capacity['actual_trees'] == 2999, 'Wrong saved native depth/policy/tree count')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 64, 'Wrong native leaf-cap evidence')
    digest = capacity['tree_leaf_counts_sha256']
    need(isinstance(digest,str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
         'Invalid leaf-count LE-int64 digest')


def validate_model(model, plan, stage):
    need(model['round'] == 15 and model['stage'] == stage and model['family'] == 'catboost'
         and model['component'] == 'geography_shallow_raw_catboost' and model['target_mode'] == 'raw', 'Wrong model identity')
    params, baseline = model['parameters'], model['baseline_parameters']
    need(params == plan['parameters'] and params['depth'] == 6 and baseline['depth'] == 8
         and set(params) == set(baseline)
         and sorted(k for k in params if params[k] != baseline[k]) == ['depth']
         and model['changed_parameter_names'] == ['depth'], 'Only depth may differ from the R6 baseline')
    need(params['iterations'] == 2999 and params['random_seed'] == 42 and params['thread_count'] == 4
         and params['loss_function'] == 'MAE', 'Wrong fixed CatBoost recipe')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['feature_count'] == 95 and model['categorical_feature_count'] == 23, 'Wrong model size/schema')
    need(model['sample_weight'] == 'uniform' and model['weight_argument_passed_to_pool'] is False,
         'Wrong uniform Pool training contract')
    need(model['feature_rules'] == plan['feature_rules'], 'Feature rules differ')
    need(model['target_formula'] == 'y_TWD / 1e6' and model['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong raw target scale')
    need(set(model['artifact_sha256']) == MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    for key in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'calibration_applied',
                'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: ' + key)


def render(payload):
    stages = payload['stages']; completed = [s for s in stages if s['metrics'] is not None]
    decision = '三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines = ['# 第十五輪結果：深度 6 的總價地理 CatBoost', '', f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。', '',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有讀取預測陣列內容、重訓或重算 MAE。', '',
        '## 實際處理與唯一候選', '',
        '- 沿用去除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。歷史完整列去重包含價格；本輪沒有額外刪除高價或難預測列。',
        '- 原 87 欄加既有 8 地理欄，值、型別、列序及欄序保持不變，共 95 欄／23 類別；沿用同一套地理規則。',
        '- CatBoost Pool 直接使用原 23 個類別欄；feature_state.json 只記錄 fit X 與 fit 地理狀態，不接受 query 或 target。原 87 欄仍用於 reference、建案分組及原去重規則。',
        '- 固定 CatBoost MAE、2,999 輪、seed 42、4 threads；只有 depth 從原 R6 的 8 改為 6，其餘參數完整保留。目標 y_TWD/1e6，還原 max(z×1e6,0)。訓練均一權重，Pool 不傳 weight，不用 eval_set 或早停，不做面積權重、額外價格上限或校正。',
        '- 以深度 6 模型替換 v3 的 37.5% 地理 raw CatBoost 槽；另 37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保持不變，融合後沿用原 reference trend 一次。model.cbm 保存原生 CatBoost 模型。',
        '- 訓練不使用 query labels；共享 loader 實際會反序列化含 target 的快取，不能聲稱從未載入驗證價格。固定 test-X 建案分組比例僅用於加權評分，不是訓練 sample_weight。', '',
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
        lines += [f"獨立重播證據核對原生模型：depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹，最大葉節點數 {capacity['maximum_tree_leaf_count']}（上限 64）。完整葉節點數向量的 little-endian int64 SHA 保存在機器報告；本報告只核對已保存證據，不重新載入模型。", '']
    lines += ['## 解讀與防過擬合限制', '',
        '本輪刻意僅比較 CatBoost 深度 8 與 6，原 95 欄、目標尺度、均一權重、2,999 輪與其餘參數固定。原生模型容量檢查涵蓋 depth、SymmetricTree、樹數與每棵樹最多 64 個葉節點。GPU 訓練可能有非確定性；相同 seed 不代表跨次重訓必定逐位元相同，本輪保存模型的重播一致性與跨次重訓是不同檢查。', '',
        '較淺的樹限制模型容量，也可能造成欠擬合；減少深度不保證降低未來總價 MAE。本輪沒有搜尋其他深度、輪數、融合權重或價格校正，容量檢查通過本身不等於候選有效。', '',
        '先固定唯一候選、公式、年度及門檻。開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。', '',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。固定工程門檻不能消除跨輪選型偏差，也不保證 Private 成績；本輪沒有掃參數、融合權重或依 Public 調參。', '',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。', '',
        '## 完整紀錄', '',
        '- [實作可行性與歷史方法比較](round15_shallow_geo_feasibility.md)、[事前規約](round15_plan.md)。',
        '- [獨立流程審查](round15_pipeline_review.md)、[工作紀錄](round15_work_log.md)。',
        '- [機器可讀報告](round15_results.json)、[凍結計畫](../outputs/round15/candidate_plan.json)、[最終判定](../outputs/round15/accepted_candidate.json)。',
        '- 各期 JSON 保留完整月份／城市描述與獨立重播證据；proof 的 status=passed 只表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。', '']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round15_report.md', ROOT/'reports/round15_results.json', ROOT/'reports/round15_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round15-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 15
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
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
            need(result['round'] == 15 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 15 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round15_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity']})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 15, 'candidate': CANDIDATE,
        'accepted': accepted['accepted'], 'stopping_reason': accepted['stopping_reason'],
        'deployment_decision': accepted['deployment_decision'], 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv', 'stages': stages,
        'plan_sha256': plan_sha, 'frozen_candidate_sha256': frozen_sha,
        'accepted_candidate_sha256': sha(OUT/'accepted_candidate.json'),
        'source_input_sha256': captured, 'historically_untouched': False, 'public_score_used': False,
        'private_score_known': False, 'report_only_reads_saved_JSON_and_hashes_binary_inputs': True}
    text = render(payload)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    contents = [text.encode('utf-8'), encoded, (hashlib.sha256(encoded).hexdigest()+'\n').encode('ascii')]
    for name, digest in captured.items(): need(sha(ROOT/name) == digest, 'Source changed while reporting')
    staging = ROOT/'reports'/('.round15-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Saved-summary-shaped synthetic dictionaries; no actual JSON/model reads."""
    checks = []
    def ok(name, condition):
        need(condition, name); checks.append(name)
    def rejects(name, fn):
        try: fn()
        except (RuntimeError, KeyError): checks.append(name)
        else: raise AssertionError(name+' not rejected')
    metric = {'rows': 4, 'overall_mae': 10., 'weighted_mae': 11.,
        'strata': {'seen': {'rows': 2, 'mae': 10.}, 'new_nonmissing': {'rows': 2, 'mae': 10.}, 'missing': {'rows': 0, 'mae': None}},
        'forecast_quarters': {q: {'rows': 1, 'mae': 10.} for q in ('Q1', 'Q2', 'Q3', 'Q4')}}
    metrics = {CONTROL: copy.deepcopy(metric), CANDIDATE: copy.deepcopy(metric)}
    validate_metrics(metrics, 4); checks.append('saved_metrics_shape_and_null_empty_group')
    bad = copy.deepcopy(metrics); bad[CANDIDATE]['strata']['missing']['mae'] = 0.
    rejects('empty_group_zero_MAE_rejected', lambda: validate_metrics(bad, 4))
    bad = copy.deepcopy(metrics); bad[CONTROL]['overall_mae'] = float('nan')
    rejects('nonfinite_MAE_rejected', lambda: validate_metrics(bad, 4))
    def fixture(passes):
        dev_pass = passes[0]; count = 3 if dev_pass else 1
        stages = []
        for index, (stage, title) in enumerate(LABELS.items()):
            evaluated = index < count
            stages.append({'stage': stage, 'title': title, 'dates': ['start', 'end'], 'fit_rows': 7, 'query_rows': 4,
                'status': 'price_evaluated' if evaluated else 'not_executed_after_development_rejection',
                'metrics': copy.deepcopy(metrics) if evaluated else None,
                'decision': {'passed': passes[index], 'checks': {'saved_gate': passes[index]}} if evaluated else None,
                'model': {'raw_z_diagnostics': {'negative_count': 1, 'zero_count': 0}} if evaluated else None,
                'independent_review': {'proof_sha256': 'synthetic', 'native_model_capacity': {
                    'actual_model_depth':6,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
                    'maximum_tree_leaf_count':64,'tree_leaf_counts_sha256':'a'*64}} if evaluated else None})
        accepted = {'accepted': all(passes), 'stage_status': {s['stage']: s['status'] for s in stages},
            'stage_result_sha256': {s['stage']: 'synthetic' for s in stages if s['metrics'] is not None},
            'unevaluated_stages': [s['stage'] for s in stages if s['metrics'] is None],
            'deployment_decision': 'backtest_accepted_pending_final_release' if all(passes) else 'keep_v3',
            'new_submission_created': False, 'currently_recommended_submission': 'outputs/submission_v3.csv'}
        frozen = {'passed': dev_pass, 'later_stages_allowed': dev_pass, 'selected_candidate': CANDIDATE if dev_pass else CONTROL}
        return accepted, frozen, stages
    for label, passes in [('development_reject', [False, False, False]), ('later_reject', [True, False, True]), ('three_pass_pending_release', [True, True, True])]:
        accepted, frozen, stages = fixture(passes)
        validate_finalization(accepted, frozen, stages); checks.append(label+'_status_supported')
        text = render({'accepted': accepted['accepted'], 'stages': stages})
        ok(label+'_correct_recommendation', ('尚待另行最終訓練與發布' in text) if accepted['accepted'] else ('本輪未採用，維持 submission_v3.csv' in text))
        ok(label+'_raw_CatBoost95_23_depth6', all(word in text for word in ['95 欄／23 類別', '2,999', 'depth=6', 'model.cbm', '上限 64']) and 'f000' not in text)
    accepted, frozen, stages = fixture([False, False, False])
    bad = copy.deepcopy(stages); bad[1]['metrics'] = metrics
    rejects('invented_later_stage_metrics_rejected', lambda: validate_finalization(accepted, frozen, bad))
    bad = copy.deepcopy(accepted); bad['accepted'] = True
    rejects('acceptance_cannot_override_saved_decisions', lambda: validate_finalization(bad, frozen, stages))
    bad = copy.deepcopy(accepted); bad['new_submission_created'] = True
    rejects('no_release_claim_from_backtest', lambda: validate_finalization(bad, frozen, stages))
    text = render({'accepted': False, 'stages': stages})
    ok('unexecuted_phases_explicit_and_not_zero', text.count('開發期未通過，未執行') == 2)
    ok('depth_only_GPU_underfit_reused_window_limits_present', all(s in text for s in ['只有 depth', 'GPU 訓練可能有非確定性', '欠擬合', '並非未接觸盲測', '2,883,250.4138']))
    ok('variation_sign_is_deterioration_positive', delta(100, 101) == 1. and delta(100, 99) == -1. and delta(0, 1) is None)
    cap = {'actual_model_depth':6,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
           'maximum_tree_leaf_count':64,'tree_leaf_counts_sha256':'a'*64}
    validate_capacity(cap); checks.append('saved_native_capacity_supported')
    bad_cap = dict(cap, actual_model_depth=8)
    rejects('wrong_saved_native_depth_rejected', lambda: validate_capacity(bad_cap))
    bad_cap = dict(cap, maximum_tree_leaf_count=256)
    rejects('wrong_saved_native_leaf_cap_rejected', lambda: validate_capacity(bad_cap))
    return {'status': 'passed', 'synthetic_only': True, 'checks': checks, 'check_count': len(checks),
            'real_JSON_labels_predictions_or_models_read': False, 'report_files_written': False, 'script_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['generate', 'self-check'])
    args = parser.parse_args()
    if args.action == 'self-check': print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else: generate()
