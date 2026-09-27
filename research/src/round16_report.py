"""Render finalized R16 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round16'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'prefix_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'prefix_parser_rules', 'prefix_parser_sha256', 'fit_prefix_nonmissing_rows',
    'fit_prefix_unique_values') + CAPACITY_FIELDS
DOCUMENTS = ('reports/round16_plan.md', 'reports/round16_unit_inventory.md',
    'reports/round16_prefix_support.md', 'reports/round16_prior_unit_methods.md',
    'reports/round16_numeric_review.md', 'reports/round16_pipeline_review.md')
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
         and capacity['actual_trees'] == 2999, 'Wrong saved native depth/policy/tree count')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 256, 'Wrong native leaf-cap evidence')
    digest = capacity['tree_leaf_counts_sha256']
    need(isinstance(digest,str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
         'Invalid leaf-count LE-int64 digest')


def validate_model(model, plan, stage):
    need(model['round'] == 16 and model['stage'] == stage and model['family'] == 'catboost'
         and model['component'] == 'geography_prefix_raw_catboost' and model['target_mode'] == 'raw', 'Wrong model identity')
    params, baseline = model['parameters'], model['baseline_parameters']
    need(params == baseline == plan['parameters'] == plan['baseline_parameters']
         and params['depth'] == 8 and model['changed_parameter_names'] == []
         and plan['changed_parameter_names'] == [], 'All original R6 parameters must remain unchanged')
    need(params['iterations'] == 2999 and params['random_seed'] == 42 and params['thread_count'] == 4
         and params['loss_function'] == 'MAE', 'Wrong fixed CatBoost recipe')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['feature_count'] == 96 and model['categorical_feature_count'] == 24, 'Wrong model size/schema')
    need(model['sample_weight'] == 'uniform' and model['weight_argument_passed_to_pool'] is False,
         'Wrong uniform Pool training contract')
    need(model['feature_rules'] == plan['feature_rules'], 'Feature rules differ')
    need(model['prefix_parser_rules'] == plan['feature_rules']['prefix_parser_rules']
         and model['prefix_parser_sha256'] == plan['reviewed_parser_sha256']
         == plan['source_sha256']['src/round16_unit_prefix.py'], 'Prefix parser evidence differs')
    n, u = model['fit_prefix_nonmissing_rows'], model['fit_prefix_unique_values']
    need(type(n) is int and type(u) is int and type(model['fit_rows']) is int
         and model['fit_rows'] == plan['split_contract'][stage]['fit_rows']
         and 0 <= u <= n <= model['fit_rows'] and ((n == 0) == (u == 0)),
         'Invalid fit-only prefix support counts')
    need(model['target_formula'] == 'y_TWD / 1e6' and model['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong raw target scale')
    need(set(model['artifact_sha256']) == MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    for key in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used', 'calibration_applied',
                'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: ' + key)


def validate_prefix_review(audit, model):
    need(audit['prefix_parser_sha256'] == model['prefix_parser_sha256'],
         'Independent prefix parser source differs')
    evidence = audit['independent_fit_state_evidence']
    need(evidence['fit_rows'] == model['fit_rows']
         and evidence['feature_count'] == 96 and evidence['categorical_count'] == 24
         and evidence['fit_state_reconstructed_independently'] is True
         and all(evidence[key] == model[key] for key in
                 ('fit_prefix_nonmissing_rows', 'fit_prefix_unique_values')),
         'Independent fit prefix evidence differs')


def render(payload):
    stages = payload['stages']; completed = [s for s in stages if s['metrics'] is not None]
    decision = '三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines = ['# 第十六輪結果：棟戶文字的共享前綴碼', '', f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。', '',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有讀取預測陣列內容、重訓或重算 MAE。', '',
        '## 實際處理與唯一候選', '',
        '- 沿用去除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。歷史完整列去重包含價格；本輪沒有額外刪除高價或難預測列。',
        '- 保留原 87 欄與既有 8 地理欄的值、型別、列序及欄序，在原 95 欄後只追加 unit_code_prefix，共 96 欄／24 類別。原 building_and_unit 完整文字早已是類別特徵，本輪新增可共享的拆解碼，不是補回遺失文字。',
        '- 新欄由同一份固定 parser 逐列解析；不學 fit/query 詞彙或案內統計。CatBoost Pool 直接使用原 23 個類別加新碼。feature_state.json 只接受 fit X 與 fit 地理，保存 parser 規則、來源及 fit 支援數；原 87 欄仍用於 reference、建案分組及原去重規則。',
        '- 沿用原 R6 depth 8、CatBoost MAE、2,999 輪、seed 42、4 threads，baseline_parameters == parameters、changed_parameter_names=[]。目標 y_TWD/1e6，還原 max(z×1e6,0)。訓練均一權重，Pool 不傳 weight，不用 eval_set、早停、面積權重、額外價格上限或校正。',
        '- 唯一候選 prefix_geography_blend 以新模型替換 v3 的 37.5% 地理 raw CatBoost 槽；另 37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 不變。融合後在原 87 欄上套用一次原 reference trend。model.cbm 保存原生 CatBoost 模型。',
        '- 訓練不使用 query labels；共享 loader 實際會反序列化含 target 的快取，不能聲稱從未載入驗證價格。固定 test-X 建案分組比例僅用於加權評分，不是訓練 sample_weight。', '',
        '## 文字支援證據與解析邊界', '',
        '首版 exact A1-5F 文法因漏掉常見尾端「號」而零命中，不能解讀成資料沒有這種格式。另一次固定補充盤點，在去除嵌入表頭後、尚未排除 529 列完整重複的 train X 455,784 列中，辨識 33,789 列原 block 缺失但有明示樓層前綴碼；test X 47,073 列中為 4,696 列。兩版資料與程式均保留，補充未讀價格或模型。這些是全快取的純 X 支援數，既不是年度 fit 數，也不是降低 MAE 的證據。', '',
        'A1-5F號 與 A1棟5F號 可以同得 A1，但這只是字串碼相同，不代表同棟、同戶型、方位或物理物件；跨建案相同碼可能含义不同。沒有明示 F／樓／層的 A1-5、A1-5號 不猜樓層。新碼不覆蓋既有 block 或 floor。', '',
        '共同排除只涵蓋規約列出的分隔、至／到／~ 等標記及多個棟／幢，不是完整自然語言範圍偵測。例如 A棟1-3樓、A棟3-1號 仍取 A，A棟1至3樓 才因指定標記被拒絕；棟後 suffix 不解讀，輸出不宣稱單一樓層或單一物件。少量既有樓層／面積問題另列數值審查，未混入此候選。', '',
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
        model = stage['model']
        lines += [f"該期 fit 中前綴非缺值 {model['fit_prefix_nonmissing_rows']:,} 列、不同字串碼 {model['fit_prefix_unique_values']:,} 個。它們包含兩種固定文法的所有有效碼，與上面的全快取「補回缺 block 格式」統計口徑不同；不以支援率推論價格改善。", '']
        capacity = stage['independent_review']['native_model_capacity']
        lines += [f"獨立重播證據核對原生模型：depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹，最大葉節點數 {capacity['maximum_tree_leaf_count']}（上限 256）。完整葉數向量的 little-endian int64 SHA 保存在機器報告；本報告只核對已保存證據，不重新載入模型。", '']
    lines += ['## 解讀與防過擬合限制', '',
        '本輪只增加一個共享字串類別，原 95 欄、R6 深度 8、目標尺度、均一權重、2,999 輪及其餘參數完全固定。原生容量檢查涵蓋 depth、SymmetricTree、樹數與每樹最多 256 葉；容量合規本身不表示候選有效。GPU 訓練可能有非確定性，相同 seed 不代表跨次重訓逐位元相同；保存模型的重播一致性是另一項檢查。', '',
        '碼可跨格式共享，也可能在不同建案代表不同意思。增加類別可能無效或增加過擬合；本輪沒有搜尋其他文法、深度、輪數、融合權重或價格校正，不能僅憑 4,696 筆格式支援採用。', '',
        '先固定唯一候選、公式、年度及門檻。開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。', '',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。固定工程門檻不能消除跨輪選型偏差，也不保證 Private 成績；本輪沒有掃參數、融合權重或依 Public 調參。', '',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。', '',
        '## 完整紀錄', '',
        '- [首版格式盤點](round16_unit_inventory.md)、[補充文字支援](round16_prefix_support.md)、[既有方法](round16_prior_unit_methods.md)。',
        '- [事前規約](round16_plan.md)、[數值處理審查](round16_numeric_review.md)、[獨立流程審查](round16_pipeline_review.md)、[工作紀錄](round16_work_log.md)。',
        '- [機器可讀報告](round16_results.json)、[凍結計畫](../outputs/round16/candidate_plan.json)、[最終判定](../outputs/round16/accepted_candidate.json)。',
        '- 各期 JSON 保留完整月份／城市描述與獨立重播證據；proof 的 status=passed 只表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。', '']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round16_report.md', ROOT/'reports/round16_results.json', ROOT/'reports/round16_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round16-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 16
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_artifact_sha256'].items()),
         'Research evidence is not bound by the formal plan')
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
            need(result['round'] == 16 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 16 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            validate_prefix_review(audit, model)
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round16_verify_backtest.py', audit['source_sha256'])
            item.update(metrics=result['metrics'], decision=result['decision'],
                model={key: model[key] for key in MODEL_FIELDS},
                independent_review={'proof_path': audit_path.relative_to(ROOT).as_posix(), 'proof_sha256': sha(audit_path),
                    'replays': audit['replays'], 'raw_z_bitwise_equal': audit['saved_raw_z_bitwise_equal'],
                    'native_model_capacity': audit['native_model_capacity'],
                    'prefix_parser_sha256': audit['prefix_parser_sha256'],
                    'independent_fit_state_evidence': audit['independent_fit_state_evidence']})
            if stage == 'development': need(frozen['result_sha256'] == sha(path), 'Development freeze refers to another result')
        else:
            need(stage in accepted['unevaluated_stages'] and item['status'] == 'not_executed_after_development_rejection', 'Unsupported unexecuted reason')
            need(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(), 'Unrecorded stage outputs')
        stages.append(item)
    validate_finalization(accepted, frozen, stages)
    capture(Path(__file__))
    for name in DOCUMENTS: capture(ROOT/name)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 16, 'candidate': CANDIDATE,
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
    staging = ROOT/'reports'/('.round16-report-'+uuid.uuid4().hex); staging.mkdir()
    for dest, content in zip(destinations, contents): (staging/dest.name).write_bytes(content)
    for dest in destinations: need(not dest.exists(), 'Concurrent report exists')
    for dest in destinations: (staging/dest.name).rename(dest)
    staging.rmdir()
    print(json.dumps({'status': 'report_written', 'accepted': accepted['accepted'],
        'report_sha256': sha(destinations[0]), 'results_sha256': sha(destinations[1])}), flush=True)


def self_check():
    """Bounded R16 delta checks only; saved JSON-shaped synthetic dictionaries."""
    checks = []
    def ok(name, condition):
        need(condition, name); checks.append(name)
    def rejects(name, fn):
        try: fn()
        except (RuntimeError, KeyError): checks.append(name)
        else: raise AssertionError(name+' not rejected')
    params = {'iterations':2999,'depth':8,'random_seed':42,'thread_count':4,'loss_function':'MAE'}
    rules = {'prefix_parser_rules':{'synthetic_rule':'fixed'},'feature_count':96}
    plan = {'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),
        'changed_parameter_names':[],'feature_rules':copy.deepcopy(rules),'reviewed_parser_sha256':'a'*64,
        'source_sha256':{'src/round16_unit_prefix.py':'a'*64},'split_contract':{'development':{'fit_rows':7}}}
    capacity = {'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
        'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64}
    model = {'round':16,'stage':'development','family':'catboost','component':'geography_prefix_raw_catboost',
        'target_mode':'raw','parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),
        'changed_parameter_names':[],'iterations':2999,'actual_iterations':2999,'feature_count':96,
        'categorical_feature_count':24,'sample_weight':'uniform','weight_argument_passed_to_pool':False,
        'feature_rules':copy.deepcopy(rules),'prefix_parser_rules':rules['prefix_parser_rules'],
        'prefix_parser_sha256':'a'*64,'fit_rows':7,'fit_prefix_nonmissing_rows':5,'fit_prefix_unique_values':2,
        'target_formula':'y_TWD / 1e6','inverse_formula':'max(z * 1e6, 0)',
        'artifact_sha256':{k:'b'*64 for k in MODEL_FILES},'raw_z_diagnostics':{'negative_count':1,'zero_count':0},
        **capacity}
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied',
                'extra_price_cap_applied','reference_or_trend_applied_here','reference_features_changed'):
        model[key] = False
    validate_model(model,plan,'development'); checks.append('R16_model_raw96_24_unchanged_parameters_prefix_evidence')
    for name, patch in [('wrong_depth',{'parameters':dict(params,depth=6)}),
                        ('old_schema',{'feature_count':95}),('unexpected_parameter_delta',{'changed_parameter_names':['depth']}),
                        ('wrong_parser_source',{'prefix_parser_sha256':'b'*64}),
                        ('prefix_counts_exceed_fit',{'fit_prefix_nonmissing_rows':8}),
                        ('noninteger_prefix_count',{'fit_prefix_unique_values':2.0}),
                        ('wrong_native_capacity',{'maximum_tree_leaf_count':257})]:
        changed=copy.deepcopy(model); changed.update(patch)
        rejects(name,lambda m=changed:validate_model(m,plan,'development'))
    proof = {'prefix_parser_sha256':'a'*64,'independent_fit_state_evidence':{
        'fit_rows':7,'feature_count':96,'categorical_count':24,
        'fit_state_reconstructed_independently':True,'fit_prefix_nonmissing_rows':5,'fit_prefix_unique_values':2}}
    validate_prefix_review(proof,model); checks.append('independent_prefix_proof_matches_saved_metadata')
    bad_proof=copy.deepcopy(proof); bad_proof['independent_fit_state_evidence']['fit_prefix_unique_values']=3
    rejects('independent_prefix_count_mismatch',lambda:validate_prefix_review(bad_proof,model))
    metric={'rows':4,'overall_mae':10.,'weighted_mae':11.,
        'strata':{'seen':{'rows':2,'mae':10.},'new_nonmissing':{'rows':2,'mae':10.},'missing':{'rows':0,'mae':None}},
        'forecast_quarters':{q:{'rows':1,'mae':10.} for q in ('Q1','Q2','Q3','Q4')}}
    def stages_for(passes):
        result=[]
        for i,(stage,title) in enumerate(LABELS.items()):
            evaluated=i==0 or passes[0]
            result.append({'stage':stage,'title':title,'dates':['start','end'],'fit_rows':7,'query_rows':4,
                'metrics':{CONTROL:copy.deepcopy(metric),CANDIDATE:copy.deepcopy(metric)} if evaluated else None,
                'decision':{'passed':passes[i],'checks':{'saved_gate':passes[i]}} if evaluated else None,
                'model':copy.deepcopy(model) if evaluated else None,
                'independent_review':{'native_model_capacity':copy.deepcopy(capacity)} if evaluated else None})
        return result
    for name,passes in [('dev_reject',[False,False,False]),('later_reject',[True,False,True]),('all_pass',[True,True,True])]:
        text=render({'accepted':all(passes),'stages':stages_for(passes)})
        ok(name+'_saved_status_render', ('尚待另行最終訓練與發布' in text) if all(passes) else '本輪未採用，維持 submission_v3.csv' in text)
        ok(name+'_R16_no_old_depth6_or_LGB_encoding', all(w in text for w in ['96 欄／24 類別','depth=8','上限 256','unit_code_prefix','changed_parameter_names=[]'])
           and '深度 6' not in text and 'f000' not in text)
    text=render({'accepted':False,'stages':stages_for([False,False,False])})
    ok('unexecuted_windows_are_explicit',text.count('開發期未通過，未執行')==2)
    ok('support_numbers_not_fit_or_gain_claim',all(w in text for w in ['33,789','4,696','零命中','不是年度 fit 數','不是降低 MAE']))
    ok('lexical_and_hyphen_limits_explicit',all(w in text for w in ['不是完整自然語言範圍','A棟1-3樓','A棟3-1號','A棟1至3樓','不代表同棟']))
    ok('original95_87_and_score_weights_distinction',all(w in text for w in ['原 95 欄','原 87 欄','不是訓練 sample_weight']))
    ok('reused_windows_GPU_and_Public_limits',all(w in text for w in ['並非未接觸盲測','GPU 訓練可能有非確定性','2,883,250.4138']))
    ok('all_prefix_metadata_preserved',all(k in MODEL_FIELDS for k in ('prefix_parser_rules','prefix_parser_sha256','fit_prefix_nonmissing_rows','fit_prefix_unique_values')))
    return {'status':'passed','synthetic_only':True,'checks':checks,'check_count':len(checks),
        'real_JSON_labels_predictions_or_models_read':False,'report_files_written':False,'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['generate', 'self-check'])
    args = parser.parse_args()
    if args.action == 'self-check': print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else: generate()
