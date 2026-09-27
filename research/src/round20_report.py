"""Render finalized R20 saved JSON/receipts only; no arrays, fitting or rescoring."""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round20'
LABELS = {'development': '開發期', 'confirmation': '確認期', 'auxiliary': '輔助期'}
CONTROL, CANDIDATE = 'v3_control', 'slow_catboost_geography_blend'
GROUPS = ('seen', 'new_nonmissing', 'missing')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy', 'predictions.npy', 'raw_z.npy'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                   'actual_learning_rate')
MODEL_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count', 'native_learning_rate_comparison_rtol',
    'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
REMOVED_FEATURES = []
CHANGED_PARAMETERS = ['iterations', 'learning_rate']
TARGET_FORMULA = 'y_TWD / 1e6'
INVERSE_FORMULA = 'max(z * 1e6, 0)'
NATIVE_LEARNING_RATE_CONTRACT = {'expected': 0.0325, 'rtol': 1e-7, 'atol': 0.0}
PRE_SCORE_RETRY_RECORD = {
    'path': 'outputs/round20_pre_score_failure_01/failure_record.json',
    'sha256': '5b7904b9d8b9d6199fec7f8d76ae137e4d8047f90b0f4d966d2536ebb30acf79',
    'status': 'superseded_pre_score_infrastructure_failure',
    'query_price_evaluation_executed': False,
    'model_hyperparameters_changed_for_retry': False,
}
DOCUMENTS = ('reports/round20_plan.md', 'reports/round20_method_review.md',
    'reports/round20_pipeline_review.md')
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
         and capacity['actual_trees'] == 5998, 'Wrong saved native depth/policy/tree count')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 256, 'Wrong native leaf-cap evidence')
    digest = capacity['tree_leaf_counts_sha256']
    need(isinstance(digest,str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
         'Invalid leaf-count LE-int64 digest')
    lr=capacity['actual_learning_rate']
    need(type(lr) in (int,float) and math.isfinite(lr)
         and abs(lr-0.0325) <= 1e-7*abs(0.0325), 'Native learning rate exceeds fixed storage tolerance')


def validate_model(model, plan, stage):
    need(model['round'] == 20 and model['stage'] == stage and model['family'] == 'catboost'
         and model['component'] == 'geography_slow_catboost' and model['target_mode'] == 'raw', 'Wrong model identity')
    params, baseline = model['parameters'], model['baseline_parameters']
    need(params==plan['parameters'] and baseline==plan['baseline_parameters']
         and set(params)==set(baseline)
         and sorted(k for k in params if params[k]!=baseline[k])==CHANGED_PARAMETERS
         and model['changed_parameter_names']==plan['changed_parameter_names']==CHANGED_PARAMETERS,
         'Only iterations and learning_rate may differ from R6')
    need(baseline['iterations']==2999 and baseline['learning_rate']==0.065
         and baseline['depth']==8 and baseline['boosting_type']=='Plain', 'Wrong R6 baseline')
    need(params['iterations']==5998 and params['learning_rate']==0.0325
         and params['depth']==8 and params['boosting_type']=='Plain'
         and params['random_seed']==42 and params['thread_count']==4
         and params['loss_function']=='MAE', 'Wrong fixed slow CatBoost recipe')
    need(model['removed_feature_names']==plan['removed_feature_names']==REMOVED_FEATURES
         and plan['feature_rules']['removed_feature_names']==REMOVED_FEATURES, 'Original95 features must remain')
    need(model['iterations']==model['actual_iterations']==model['actual_trees']==5998
         and model['feature_count']==95 and model['categorical_feature_count']==23
         and model['native_categorical_feature_count']==23, 'Wrong model size/schema')
    need(model['sample_weight']==plan['sample_weight']=='uniform'
         and model['weight_argument_passed_to_pool'] is plan['weight_argument_passed_to_pool'] is False,
         'Raw uniform Pool training required')
    need(model['feature_rules']==plan['feature_rules'], 'Feature rules differ')
    need(model['target_formula']==plan['target_formula']==TARGET_FORMULA
         and model['inverse_formula']==plan['inverse_formula']==INVERSE_FORMULA, 'Wrong raw target/inverse scale')
    need(plan['native_learning_rate_contract']==NATIVE_LEARNING_RATE_CONTRACT
         and model['native_learning_rate_comparison_rtol']==1e-7
         and model['native_learning_rate_comparison_atol']==0.0
         and model['feature_rules']['native_learning_rate_comparison_rtol']==1e-7
         and model['feature_rules']['native_learning_rate_comparison_atol']==0.0, 'Learning-rate tolerance changed')
    need(model['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['iterations']==5998
         and model['feature_rules']['geo_encoder_rules']['iterations']==2999,
         'Nested geographic baseline iterations must remain provenance only')
    need(set(model['artifact_sha256'])==MODEL_FILES, 'Expected native CBM and fixed six artifacts')
    validate_capacity({key:model[key] for key in CAPACITY_FIELDS})
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied',
                'extra_price_cap_applied','reference_or_trend_applied_here','reference_features_changed'):
        need(model[key] is False, 'Wrong training/prediction policy: '+key)


def validate_replay_feature_evidence(audit, model):
    need(audit['removed_feature_names']==model['removed_feature_names']==[], 'Independent feature declaration differs')
    need(audit['original_95_values_dtypes_order_index_preserved'] is True,
         'Independent original model-view preservation was not confirmed')
    need(audit['changed_parameter_names']==CHANGED_PARAMETERS
         and audit['native_categorical_feature_count']==model['native_categorical_feature_count']==23,
         'Independent parameter or native category declaration differs')
    need(audit['raw_z_units']=='million_TWD'
         and audit['native_learning_rate_comparison_rtol']==1e-7
         and audit['native_learning_rate_comparison_atol']==0.0, 'Independent raw units or learning-rate tolerance differ')
    evidence=audit['independent_fit_state_evidence']
    need(evidence['base_feature_count']==evidence['feature_count']==95
         and evidence['categorical_count']==23 and evidence['fit_rows']==model['fit_rows']
         and evidence['removed_feature_names']==[] and evidence['fit_state_reconstructed_independently'] is True
         and evidence['fit_model_feature_values_sha256']==model['fit_model_feature_values_sha256'],
         'Independent fit-only feature-state reconstruction differs')
    return copy.deepcopy(evidence)


def research_expectations(source_text):
    """Read only the frozen driver's literal map, without importing its ML code."""
    tree=ast.parse(source_text)
    found=[node.value for node in tree.body if isinstance(node,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='RESEARCH_EXPECTED_SHA' for t in node.targets)]
    need(len(found)==1, 'Expected one static historical research map')
    value=ast.literal_eval(found[0])
    need(isinstance(value,dict) and len(value)==11, 'Expected eleven historical evidence identities')
    return value


def validate_research(research, dependencies, expected):
    """Only compare declared identities; never parse historical model scores."""
    need(isinstance(research,dict) and len(research)==11 and research==expected,
         'Historical research map differs from frozen driver expectations')
    need(all(dependencies.get(k)==v for k,v in research.items()), 'Research outside frozen dependencies')
    for name,digest in research.items():
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid historical research identity')
    return {'frozen_research_dependency_count':11,'dependency_sha256':copy.deepcopy(research),
            'historical_source_configuration_metadata_document_evidence_only':True,
            'new_research_JSON_or_receipt_required':False,
            'report_did_not_read_historical_metrics_for_selection':True}


def render(payload):
    stages=payload['stages'];completed=[s for s in stages if s['metrics'] is not None]
    decision='三年度固定回測通過，尚待另行最終訓練與發布' if payload['accepted'] else '本輪未採用，維持 submission_v3.csv'
    lines=['# 第二十輪結果：較小學習率、較多樹的地理 CatBoost','',
        f'**{decision}。** 本流程沒有產生新提交檔或上傳 Kaggle。','',
        '本報告只讀保存的 JSON 與收據，並對依賴檔核對位元組雜湊；沒有讀取預測陣列內容、重訓或重算 MAE。','',
        '## 實際處理與唯一候選','',
        '- 沿用移除 3 列嵌入表頭、train 的 529 列完整去重及原 canonical 索引。完整列去重依原始欄位（含價格、排除編號）進行，沒有以 87 特徵欄重新去重，也沒有刪除高價或難預測列。',
        '- 原 87 欄加既有 8 地理欄，共 95 欄／23 類別全部保留；原值、型別、列序、欄序及年度地理規則不變。feature_state 只接受 fit X／geo；原 87 欄仍用於 reference 與評分分組。',
        '- 相對原 R6，唯一兩個傳入參數變更為 iterations：2,999→5,998；learning_rate：0.065→0.0325。其餘 MAE、Plain、depth 8、seed 42、4 threads 及所有設定不變。changed_parameter_names=[iterations, learning_rate]，removed_feature_names=[]。',
        '- 目標仍為 y_TWD/1e6，uniform，Pool 不傳入 sample_weight；預測只以 max(z×1e6,0) 還原總價。無 eval_set、早停、價格上限或額外校正。',
        '- 用新 raw 地理 CatBoost 替換 v3 的 37.5% 地理模型槽；37.5% 舊 raw CatBoost、25% 舊 unit LightGBM 保留。先還原 TWD 再融合，原 reference trend 只施加一次；model.cbm 保存原生模型。',
        '- 訓練不使用 query labels；共享 loader 會反序列化含 target 的快取，不能聲稱從未載入驗證價格。test-X 建案比例仍只用於固定加權評分，沒有用作本模型訓練權重。','',
        '歷史新穎性證據為事前固定的 11 份 source／config／metadata／文件，沒有新增研究資料集或面積盤點。報告只核對這些證據與計畫的 SHA 身分，不讀舊價格成績選參數。nested R6 地理 encoder 規則中的 iterations=2999 是來源宣告；本輪實際訓練與原生模型的 authority 為頂層 5998 與保存模型證據。','',
        '## 正式評分前的重跑紀錄','',
        '首次實際訓練完成後，在保存模型的原生欄位逐值比較中止；尚未完成全 query 相等核對，也未執行價格評分。已完整保留原模型、計畫、來源與日誌，並將學習率儲存表示的比較改為事前既有容差，其餘原生欄位仍精確相等。候選、超參數、特徵、公式及門檻沒有因此改變。',
        '由於首次訓練的 metadata 與記憶體 query 預測未在中止前保存，不能補造相等證據；因此重新凍結後以相同設定完整重跑。以下表格只呈現後續正式完成的回測，不把未評分的中止當作候選績效。保存模型重載的 actual_learning_rate 為正式原生值，另保留 fitted_actual_learning_rate 作訓練記憶體記錄。',
        '- [評分前中止與重跑原因](../outputs/round20_pre_score_failure_01/failure_record.json)。','',
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
        lines+=[f"原生模型證據：boosting_type={capacity['actual_boosting_type']}、depth={capacity['actual_model_depth']}、grow_policy={capacity['actual_grow_policy']}、{capacity['actual_trees']:,} 棵樹、最大 {capacity['maximum_tree_leaf_count']} 葉（上限 256）、learning_rate={capacity['actual_learning_rate']:.12g}。學習率對 0.0325 使用 rtol=1e-7、atol=0 容許儲存誤差；另核對 23 個原生類別欄位置。完整葉數向量的 LE-int64 SHA 保存在機器報告；這份報告本身不重新載入模型。",'']
    lines+=['## 解讀與防過擬合限制','',
        '2,999×0.065 與 5,998×0.0325 的名義步長乘積同為 194.935，但不表示等價模型或容量不變。較小步長會改變後續殘差與切分；增加樹數會增加總模型容量、訓練及推論成本，不能事先宣稱減少過擬合。這是固定兩個參數的單一候選，不是單一參數因果實驗，成效只依已凍結三期門檻判斷。','',
        '相同 seed 不保證 GPU 跨次訓練逐位元一致；保存模型的精確重播是另一項檢查。原95欄、raw 目標、uniform 權重、類別處理、融合及趨勢規則均不變，没有掃描輪數或學習率，也沒有按本輪結果再調整。','',
        '開發期整體與加權須至少改善 1%，非空季度及 rows≥500 的原三組退步不可超過 5%；通過後才允許確認期、輔助期。確認期整體／加權均不可退步且至少兩季改善；輔助期整體／加權退步上限 3%，兩期同樣保留大群組限制。季度按原相對預測天數切分，沒有重新選切法。','',
        '歷史年度已反覆用於多輪研究，並非未接觸盲測。1% 是工程門檻，不是統計顯著性；不能消除跨輪選型偏差或保證 Private 成績。沒有依 Public 成績調參。','',
        '唯一已知 Kaggle 回饋是 v2 Public MAE 2,883,250.4138；v3／本輪 Public 與 Private 皆未知。沒有查內政部或外部成交答案，沒有上傳或發布新提交檔。','',
        '## 完整紀錄','',
        '- [方法與歷史審查](round20_method_review.md)、[事前規約](round20_plan.md)。',
        '- [獨立流程審查](round20_pipeline_review.md)、[工作紀錄](round20_work_log.md)。',
        '- [機器可讀報告](round20_results.json)、[凍結計畫](../outputs/round20/candidate_plan.json)、[最終判定](../outputs/round20/accepted_candidate.json)。',
        '- 各期 JSON 保存原始群組／季度摘要與獨立重播證據；proof 的 status=passed 表示核對通過，不等同採用候選。',
        '- [目前推薦提交](../outputs/submission_v3.csv)。即使三窗 accepted=true，仍需另行最終訓練與發布，這份報告不建立 CSV。','']
    return '\n'.join(lines)


def generate():
    destinations = [ROOT/'reports/round20_report.md', ROOT/'reports/round20_results.json', ROOT/'reports/round20_results.sha256']
    need(not any(p.exists() for p in destinations), 'Refusing to overwrite report')
    need(not list((ROOT/'reports').glob('.round20-report-*')), 'Partial report staging must be retained and inspected')
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
    need(plan['round'] == frozen['round'] == accepted['round'] == 20
         and plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == plan_sha
         and accepted['frozen_candidate_sha256'] == frozen_sha, 'Wrong finalization lineage')
    need(accepted['source_sha256'] == frozen['source_sha256'] == plan['source_sha256'], 'Different frozen source')
    mapping(plan['source_sha256']); mapping(plan['dependency_sha256'])
    failure_path=ROOT/PRE_SCORE_RETRY_RECORD['path']
    capture(failure_path,PRE_SCORE_RETRY_RECORD['sha256'])
    failure=read(failure_path)
    for key in ('status','query_price_evaluation_executed','model_hyperparameters_changed_for_retry'):
        need(failure[key]==PRE_SCORE_RETRY_RECORD[key], 'Preserved pre-score history differs: '+key)
    need(all(plan['dependency_sha256'].get(k) == v for k, v in plan['research_dependency_sha256'].items()),
         'Research dependencies are outside the frozen input map')
    expected=research_expectations((ROOT/'src/round20_experiment.py').read_text(encoding='utf-8-sig'))
    research_summary=validate_research(plan['research_dependency_sha256'],plan['dependency_sha256'],expected)
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
            need(result['round'] == 20 and result['stage'] == stage and result['plan_sha256'] == plan_sha, 'Wrong saved result')
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
            need(audit['round'] == 20 and audit['status'] == 'passed' and audit['stage'] == stage
                 and audit['plan_sha256'] == plan_sha and audit['result_sha256'] == sha(path), 'Missing matching independent proof')
            need(audit['independent_metrics'] == result['metrics'] and audit['independent_decision'] == result['decision']
                 and audit['raw_z_diagnostics'] == model['raw_z_diagnostics']
                 and audit['saved_raw_z_bitwise_equal'] is True, 'Independent proof differs')
            validate_capacity(audit['native_model_capacity'])
            need(audit['native_model_capacity'] == {key:model[key] for key in CAPACITY_FIELDS},
                 'Independent native capacity differs from model metadata')
            feature_evidence = validate_replay_feature_evidence(audit, model)
            mapping(audit['source_input_sha256'])
            capture(ROOT/'src/round20_verify_backtest.py', audit['source_sha256'])
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
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'round': 20, 'candidate': CANDIDATE,
        'accepted': accepted['accepted'], 'stopping_reason': accepted['stopping_reason'],
        'deployment_decision': accepted['deployment_decision'], 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv', 'stages': stages,
        'plan_sha256': plan_sha, 'frozen_candidate_sha256': frozen_sha,
        'accepted_candidate_sha256': sha(OUT/'accepted_candidate.json'),
        'source_input_sha256': captured, 'historically_untouched': False, 'public_score_used': False,
        'private_score_known': False, 'research_summary': research_summary,
        'pre_score_retry_record': copy.deepcopy(PRE_SCORE_RETRY_RECORD),
        'report_only_reads_saved_JSON_and_hashes_binary_inputs': True}
    text = render(payload)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    contents = [text.encode('utf-8'), encoded, (hashlib.sha256(encoded).hexdigest()+'\n').encode('ascii')]
    for name, digest in captured.items(): need(sha(ROOT/name) == digest, 'Source changed while reporting')
    staging = ROOT/'reports'/('.round20-report-'+uuid.uuid4().hex); staging.mkdir()
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
        except (RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError(name+' not rejected')
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'random_seed':42,'thread_count':4,
              'loss_function':'MAE','boosting_type':'Plain','l2_leaf_reg':6}
    params={**baseline,'iterations':5998,'learning_rate':.0325}
    rules={'removed_feature_names':[],'iterations':5998,'geo_encoder_rules':{'iterations':2999},
           'native_learning_rate_comparison_rtol':1e-7,'native_learning_rate_comparison_atol':0.0,
           'geo_encoder_rule_training_fields_are_baseline_provenance_only':True}
    plan={'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
          'changed_parameter_names':list(CHANGED_PARAMETERS),'feature_rules':copy.deepcopy(rules),
          'removed_feature_names':[],'sample_weight':'uniform','weight_argument_passed_to_pool':False,
          'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
          'native_learning_rate_contract':copy.deepcopy(NATIVE_LEARNING_RATE_CONTRACT)}
    cap={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':5998,
         'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain',
         'actual_learning_rate':.0325}
    model={'round':20,'stage':'development','family':'catboost','component':'geography_slow_catboost','target_mode':'raw',
           'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
           'changed_parameter_names':list(CHANGED_PARAMETERS),'removed_feature_names':[],
           'iterations':5998,'actual_iterations':5998,'feature_count':95,'categorical_feature_count':23,
           'native_categorical_feature_count':23,'sample_weight':'uniform','weight_argument_passed_to_pool':False,
           'fit_rows':7,'fit_model_feature_values_sha256':'c'*64,'feature_rules':copy.deepcopy(rules),
           'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
           'artifact_sha256':dict.fromkeys(MODEL_FILES,'b'*64),'raw_z_diagnostics':{'negative_count':1,'zero_count':0},
           'runner_elapsed_seconds':1.,'fitted_actual_learning_rate':.0325,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,
           'geo_encoder_rule_training_fields_are_baseline_provenance_only':True,**cap}
    for key in ('query_labels_used','fit_uses_eval_set','early_stopping_used','calibration_applied','extra_price_cap_applied',
                'reference_or_trend_applied_here','reference_features_changed'):model[key]=False
    validate_model(model,plan,'development');checks.append('raw95_23_Plain_fixed_two_parameter_recipe')
    for name,patch in [('nativeOrdered',{'actual_boosting_type':'Ordered'}),('wrong91features',{'feature_count':91}),
                       ('wrong19cats',{'categorical_feature_count':19}),('native22cats',{'native_categorical_feature_count':22}),
                       ('unit_target',{'target_mode':'unit'}),('weighted_Pool',{'weight_argument_passed_to_pool':True}),
                       ('unit_inverse',{'inverse_formula':'max(z * area * 1e4, 0)'}),
                       ('old_tree_count',{'actual_trees':2999}),('native_old_learning_rate',{'actual_learning_rate':.065}),
                       ('native_nan_learning_rate',{'actual_learning_rate':float('nan')}),
                       ('changed_rtol',{'native_learning_rate_comparison_rtol':1e-6}),
                       ('changed_atol',{'native_learning_rate_comparison_atol':1e-8}),
                       ('extra_calibration',{'calibration_applied':True})]:
        bad=copy.deepcopy(model);bad.update(patch)
        rejects(name,lambda m=bad:validate_model(m,plan,'development'))
    float32_cap=copy.deepcopy(cap);float32_cap['actual_learning_rate']=0.032499998807907104
    validate_capacity(float32_cap);checks.append('actual_float32_learning_rate_accepted')
    bad=copy.deepcopy(model);bad['parameters']['l2_leaf_reg']=7
    bad_plan=copy.deepcopy(plan);bad_plan['parameters']=copy.deepcopy(bad['parameters'])
    rejects('third_parameter_change_rejected',lambda:validate_model(bad,bad_plan,'development'))
    bad=copy.deepcopy(model);bad['changed_parameter_names']=[]
    rejects('lost_two_change_declaration',lambda:validate_model(bad,plan,'development'))
    bad=copy.deepcopy(plan);bad['native_learning_rate_contract']['expected']=.065
    rejects('plan_native_contract_changed',lambda:validate_model(model,bad,'development'))
    bad=copy.deepcopy(model);bad['geo_encoder_rule_training_fields_are_baseline_provenance_only']=False
    rejects('baseline_iterations_misdeclared_as_current',lambda:validate_model(bad,plan,'development'))
    evidence={'fit_rows':7,'base_feature_count':95,'feature_count':95,'categorical_count':23,'removed_feature_names':[],
              'fit_state_reconstructed_independently':True,'fit_model_feature_values_sha256':'c'*64}
    proof={'removed_feature_names':[],'changed_parameter_names':list(CHANGED_PARAMETERS),
           'original_95_values_dtypes_order_index_preserved':True,'raw_z_units':'million_TWD',
           'native_categorical_feature_count':23,'native_learning_rate_comparison_rtol':1e-7,
           'native_learning_rate_comparison_atol':0.0,'independent_fit_state_evidence':copy.deepcopy(evidence)}
    validate_replay_feature_evidence(proof,model);checks.append('independent_raw95_fit_only_evidence_match')
    for key,value in [('raw_z_units','ten_thousand_TWD_per_effective_square_metre'),
                      ('native_categorical_feature_count',22),('native_learning_rate_comparison_rtol',1e-5),
                      ('changed_parameter_names',[]),('original_95_values_dtypes_order_index_preserved',False)]:
        bad=copy.deepcopy(proof);bad[key]=value
        rejects('proof_'+key,lambda q=bad:validate_replay_feature_evidence(q,model))
    bad=copy.deepcopy(proof);bad['independent_fit_state_evidence']['fit_model_feature_values_sha256']='d'*64
    rejects('independent_fit_hash_mismatch',lambda:validate_replay_feature_evidence(bad,model))
    research={f'synthetic/evidence{i}.json':'a'*64 for i in range(11)}
    expected=research_expectations('RESEARCH_EXPECTED_SHA = '+repr(research))
    summary=validate_research(research,dict(research),expected)
    ok('eleven_static_historical_bindings_no_research_data',summary['frozen_research_dependency_count']==11
       and summary['dependency_sha256']==research and summary['new_research_JSON_or_receipt_required'] is False)
    rejects('missing_research_dependency',lambda:validate_research(research,{},expected))
    bad=copy.deepcopy(expected);bad['synthetic/evidence0.json']='b'*64
    rejects('changed_research_identity',lambda:validate_research(research,research,bad))
    bad=copy.deepcopy(research);bad['extra.json']='a'*64
    rejects('expanded_research_inventory',lambda:validate_research(bad,bad,bad))
    rejects('missing_static_research_declaration',lambda:research_expectations('OTHER = {}'))
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
        ok(name+'_raw95_native7_no_area_remnants',all(x in text for x in ['boosting_type=Plain','95 欄／23 類別','5998','百萬元','uniform'])
           and all(x not in text for x in ['A/100','effective_area','面積加權單價','round19','四欄消融','91 欄']))
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
    ok('step_product_not_equivalence_or_overfit_claim',all(x in text for x in ['194.935','不表示等價模型或容量不變','不能事先宣稱減少過擬合']))
    ok('raw_dedup_and_original87_reference',all(x in text for x in ['含價格、排除編號','原 87 欄仍用於 reference']))
    ok('baseline_encoder_iteration_is_provenance',all(x in text for x in ['iterations=2999','authority 為頂層 5998']))
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
