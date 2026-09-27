"""Render R7 status from saved JSON receipts and docs; never train or score.

Standard library only. Does not read y, arrays, model binaries or CSV values.
A passed backtest is not a formal release; the recommendation stays v3.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORTS = ROOT/'outputs/round7', ROOT/'reports'
CONTROL, CANDIDATE = 'v3_control', 'geography_unit_replacement'
NAMES, STAGES = (CONTROL, CANDIDATE), ('development', 'confirmation', 'auxiliary')
GROUPS, QUARTERS = ('seen', 'new_nonmissing', 'missing'), ('Q1', 'Q2', 'Q3', 'Q4')
LABELS = {CONTROL:'現行 v3', CANDIDATE:'固定地理單價替換'}
GROUP_LABELS = {'seen':'已見建案', 'new_nonmissing':'新且非缺名建案', 'missing':'建案缺名'}
SPLITS = {'development':['2022-08-11','2023-08-10'], 'confirmation':['2023-08-11','2024-08-10'], 'auxiliary':['2021-08-11','2022-08-10']}
COUNTS = {'development':(240316,81234), 'confirmation':(321550,133697), 'auxiliary':(134805,105511)}
FORMULAS = {CONTROL:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once', CANDIDATE:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*geo_unit_lgb; then v2 trend once'}
THRESHOLDS = {'development_min_overall_improvement':.01, 'development_min_weighted_improvement':.01,
 'development_max_quarter_deterioration':.05, 'large_stratum_min_rows':500, 'max_large_stratum_deterioration':.05,
 'confirmation_max_overall_deterioration':0., 'confirmation_max_weighted_deterioration':0.,
 'confirmation_min_improved_quarters':2, 'auxiliary_max_overall_deterioration':.03,
 'auxiliary_max_weighted_deterioration':.03, 'auxiliary_quarters_diagnostic_only':True}
GATE_NAMES = {'development':{'overall_improves_at_least_1pct','weighted_improves_at_least_1pct','quarters_deteriorate_at_most_5pct','large_strata_deteriorate_at_most_5pct'},
 'confirmation':{'overall_within_limit','weighted_within_limit','large_strata_deteriorate_at_most_5pct','at_least_two_quarters_improve'},
 'auxiliary':{'overall_within_limit','weighted_within_limit','large_strata_deteriorate_at_most_5pct'}}
GATE_LABELS = {'overall_improves_at_least_1pct':'整體至少改善 1%', 'weighted_improves_at_least_1pct':'加權至少改善 1%',
 'quarters_deteriorate_at_most_5pct':'每季退步最多 5%', 'large_strata_deteriorate_at_most_5pct':'n≥500 各建案組退步最多 5%',
 'overall_within_limit':'整體符合本階段上限', 'weighted_within_limit':'加權符合本階段上限', 'at_least_two_quarters_improve':'至少兩季改善'}
DOCS = ('reports/round7_validation_plan.md', 'reports/round7_feature_audit.md', 'reports/round7_work_log.md')


def need(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_locked(path, snapshots):
    path = Path(path); receipt = path.with_suffix('.sha256')
    if not path.exists():
        need(not receipt.exists(), 'Orphan receipt: '+rel(receipt))
        snapshots[rel(path)] = snapshots[rel(receipt)] = None
        return None
    need(path.is_file() and receipt.is_file(), 'Missing JSON receipt: '+rel(path))
    digest = sha(path); need(receipt.read_text(encoding='ascii').strip() == digest, 'Receipt differs: '+rel(path))
    value = read(path); snapshots[rel(path)] = digest; snapshots[rel(receipt)] = sha(receipt)
    return value


def improvement(candidate, baseline):
    if candidate is None or baseline is None or baseline == 0: return None
    return 100*(baseline-candidate)/baseline


def check_metric(value):
    need(type(value['rows']) is int and value['rows'] > 0, 'Invalid saved row count')
    for name in ('overall_mae','weighted_mae'):
        need(isinstance(value[name], (int,float)) and math.isfinite(value[name]) and value[name] >= 0, 'Invalid saved MAE')
    for field, keys in (('strata',GROUPS), ('forecast_quarters',QUARTERS)):
        need(set(value[field]) == set(keys), 'Saved metric groups differ')
        need(sum(value[field][key]['rows'] for key in keys) == value['rows'], 'Group row totals differ')
        for key in keys:
            item = value[field][key]; need(type(item['rows']) is int and item['rows'] >= 0, 'Invalid group rows')
            if item['rows'] == 0: need(item['mae'] is None, 'Empty-group MAE must be null')
            else: need(isinstance(item['mae'], (int,float)) and math.isfinite(item['mae']) and item['mae'] >= 0, 'Invalid group MAE')


def check_decision(stage, decision):
    need(set(decision['checks']) == GATE_NAMES[stage] and all(type(v) is bool for v in decision['checks'].values()), 'Saved gate names/types differ')
    need(type(decision['passed']) is bool and decision['passed'] == all(decision['checks'].values()), 'Saved gate flags contradict passed')
    need(decision['selected_candidate'] == (CANDIDATE if decision['passed'] else CONTROL), 'Saved gate selection differs')
    need(decision['candidate_reselection_performed'] is False, 'Unexpected candidate reselection')
    need(len(set(decision['improved_quarters'])) == len(decision['improved_quarters']) and set(decision['improved_quarters']) <= set(QUARTERS), 'Invalid saved improved quarters')
    if stage == 'confirmation':
        need(decision['checks']['at_least_two_quarters_improve'] == (len(decision['improved_quarters']) >= 2), 'Saved quarter-count flag contradicts list')


def outcome(plan, frozen, results, accepted):
    if plan is None: return 'awaiting_plan', '正式計畫尚未凍結；未有本輪完成評估。'
    dev = results['development']
    if dev is None: return 'awaiting_development', '計畫已凍結；development 評估尚待完成。'
    if accepted is not None:
        if accepted['accepted']: return 'backtests_passed_release_pending', '固定三年度回測已接受；尚無本程式驗證的新正式 release／CSV，現行推薦維持 v3。'
        return 'retain_v3', '本輪正式決策：不採用新候選，保留 v3。'
    if not dev['decision']['passed']: return 'development_rejected_finalization_pending', 'Development 未通過；保留 v3，依規則不執行後兩期，正式 finalize 紀錄尚待完成。'
    if frozen is None: return 'development_passed_freeze_pending', 'Development 已通過；候選 freeze 尚待保存，後兩期尚未放行。'
    pending = [s for s in ('confirmation','auxiliary') if results[s] is None]
    if pending: return 'development_passed_validation_pending', 'Development 通過且已凍結；'+ '、'.join(pending)+' 評估待完成，現行推薦維持 v3。'
    return 'awaiting_finalization', '三年度結果已保存；正式接受決策尚待 finalize，現行推薦維持 v3。'


def load_snapshot():
    snapshots = {}; plan = read_locked(OUT/'candidate_plan.json', snapshots)
    frozen = read_locked(OUT/'frozen_candidate.json', snapshots)
    accepted = read_locked(OUT/'accepted_candidate.json', snapshots)
    results = {s:read_locked(OUT/'evaluation'/s/'results.json', snapshots) for s in STAGES}
    models = {s:read_locked(OUT/'models'/s/'metadata.json', snapshots) for s in STAGES}
    docs = {}
    for name in DOCS:
        path = ROOT/name
        if path.is_file():
            docs[name] = sha(path); snapshots[name] = docs[name]
        else: snapshots[name] = None
    if plan is None:
        need(frozen is None and accepted is None and all(v is None for v in results.values()) and all(v is None for v in models.values()), 'Artifacts exist without frozen plan')
        return plan, frozen, accepted, results, models, docs, snapshots
    psha = snapshots[rel(OUT/'candidate_plan.json')]
    need(plan['round'] == 7 and plan['candidate'] == CANDIDATE and plan['control'] == CONTROL, 'Wrong R7 plan identity')
    need(plan['formulas'] == FORMULAS and plan['thresholds'] == THRESHOLDS and plan['splits'] == SPLITS, 'Fixed formulas/gates/windows differ')
    need(plan['feature_count'] == 95 and plan['categorical_count'] == 23 and plan['parameters']['n_estimators'] == 3999 and plan['parameters']['num_leaves'] == 63 and plan['parameters']['n_jobs'] == 4, 'Fixed model configuration differs')
    for flag, expected in [('single_development_candidate',True), ('tuning_or_weight_search',False), ('public_score_used_for_selection',False), ('historically_untouched',False), ('external_property_answers_used',False), ('fit_uses_eval_set',False), ('early_stopping_used',False)]:
        need(plan[flag] is expected, 'Plan policy differs: '+flag)
    need(plan['target_group_weights']['missing'] == 0 and set(plan['target_group_weights']) == set(GROUPS), 'Target group weights differ')
    need(abs(sum(plan['target_group_weights'].values())-1) < 1e-12, 'Target weights do not sum to one')
    for name in DOCS[:2]: need(plan['dependency_sha256'][name] == docs.get(name), 'Frozen research document differs: '+name)
    for stage in STAGES:
        spec = plan['split_contract'][stage]
        need((spec['fit_rows'],spec['query_rows']) == COUNTS[stage], 'Planned stage counts differ')
        model = models[stage]
        if model is not None:
            need(model['round'] == 7 and model['stage'] == stage and model['plan_sha256'] == psha and model['component'] == 'geography_unit_lightgbm', 'Model metadata identity differs')
            need(model['source_sha256'] == plan['source_sha256'] and model['parameters'] == plan['parameters'], 'Model metadata source/parameters differ')
            need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 3999 and model['feature_count'] == 95 and model['categorical_feature_count'] == 23, 'Model metadata shape differs')
            need((model['outer_fit_rows'],model['query_rows']) == COUNTS[stage], 'Model metadata rows differ')
            for flag, expected in [('fit_uses_eval_set',False), ('early_stopping_used',False), ('category_maps_fit_only',True), ('reload_64_rows_verified',True), ('query_labels_used',False), ('direct_component_score_omitted',True)]: need(model[flag] is expected, 'Model policy differs: '+flag)
            need(model['prediction_units'] == 'TWD', 'Prediction units differ')
            need(model['frozen_candidate_sha256'] == (None if stage == 'development' else snapshots[rel(OUT/'frozen_candidate.json')]), 'Model freeze identity differs')
            key = 'outputs/round6/geography/'+stage+'/manifest.json'
            need(model['geo_manifest_sha256'] == plan['stage_dependency_sha256'][stage][key], 'Saved model geography identity differs')
        result = results[stage]
        if result is None: continue
        need(model is not None, 'Evaluation lacks saved model metadata')
        need(result['round'] == 7 and result['stage'] == stage and result['plan_sha256'] == psha, 'Evaluation identity differs')
        need(result['source_sha256'] == plan['source_sha256'] and result['formulas'] == FORMULAS and result['thresholds'] == THRESHOLDS and result['target_group_weights'] == plan['target_group_weights'], 'Evaluation protocol differs')
        need(result['evaluated_candidates'] == list(NAMES) and set(result['metrics']) == set(NAMES), 'Unexpected scored candidate list')
        need((result['fit_rows'],result['query_rows']) == COUNTS[stage], 'Evaluation row counts differ')
        need(result['frozen_candidate_sha256'] == (None if stage == 'development' else snapshots[rel(OUT/'frozen_candidate.json')]), 'Evaluation freeze differs')
        name = rel(OUT/'models'/stage/'metadata.json')
        need(result['input_artifact_sha256'][name] == snapshots[name], 'Evaluation model-metadata SHA differs')
        for flag, expected in [('v3_control_replayed_all_rows',True), ('new_model_replayed_all_rows',True), ('candidate_component_separately_scored',False), ('query_labels_used_for_training',False), ('public_score_used',False), ('historically_untouched',False)]: need(result[flag] is expected, 'Evaluation policy differs: '+flag)
        for value in result['metrics'].values(): check_metric(value); need(value['rows'] == COUNTS[stage][1], 'Scored row count differs')
        for field in ('strata','forecast_quarters'):
            need({k:v['rows'] for k,v in result['metrics'][CONTROL][field].items()} == {k:v['rows'] for k,v in result['metrics'][CANDIDATE][field].items()}, 'Compared groups have different rows')
        check_decision(stage, result['decision'])
    if frozen is not None:
        dev = results['development']; need(dev is not None, 'Freeze lacks development result')
        need(frozen['plan_sha256'] == psha and frozen['result_sha256'] == snapshots[rel(OUT/'evaluation/development/results.json')], 'Freeze identity differs')
        need(frozen['passed'] == dev['decision']['passed'] and frozen['selected_candidate'] == dev['decision']['selected_candidate'] and frozen['later_stages_allowed'] == frozen['passed'], 'Freeze decision differs')
        need(frozen['source_sha256'] == plan['source_sha256'] and frozen['formulas'] == FORMULAS and frozen['thresholds'] == THRESHOLDS and frozen['parameters'] == plan['parameters'], 'Freeze protocol differs')
        for name in [rel(OUT/'models/development/metadata.json'), rel(OUT/'evaluation/development/results.json')]: need(frozen['development_artifact_sha256'][name] == snapshots[name], 'Freeze JSON evidence differs')
    for stage in ('confirmation','auxiliary'):
        if models[stage] is not None or results[stage] is not None: need(frozen is not None and frozen['passed'] is True, 'Later-stage artifact without eligible frozen candidate')
    if accepted is not None:
        need(frozen is not None and results['development'] is not None, 'Acceptance lacks selection evidence')
        need(accepted['round'] == 7 and accepted['candidate'] == CANDIDATE and accepted['plan_sha256'] == psha and accepted['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Acceptance identity differs')
        stages = STAGES if frozen['passed'] else ('development',)
        need(set(accepted['result_sha256']) == set(stages) and accepted['unevaluated_stages'] == [s for s in STAGES if s not in stages], 'Acceptance stages differ')
        for stage in stages: need(results[stage] is not None and accepted['result_sha256'][stage] == snapshots[rel(OUT/'evaluation'/stage/'results.json')], 'Acceptance result identity differs')
        expected = len(stages) == 3 and all(results[s]['decision']['passed'] for s in stages)
        need(type(accepted['accepted']) is bool and accepted['accepted'] == expected, 'Acceptance contradicts saved gates')
        need(accepted['deployment_decision'] == (CANDIDATE if expected else 'keep_v3') and accepted['no_reselection_or_parameter_change'] is True and accepted['new_submission_created'] is False, 'Acceptance deployment policy differs')
        need(accepted['source_sha256'] == plan['source_sha256'], 'Acceptance sources differ')
    return plan, frozen, accepted, results, models, docs, snapshots


def summarize(plan, frozen, accepted, results, models, docs, snapshots):
    status, text = outcome(plan, frozen, results, accepted)
    rejected = results['development'] is not None and not results['development']['decision']['passed']
    stages = {}
    meta_keys = ('component','created_at_utc','fit_started_at_utc','fit_completed_at_utc','outer_fit_rows','query_rows',
      'iterations','actual_iterations','actual_trees','feature_count','categorical_feature_count','parameters',
      'fit_uses_eval_set','early_stopping_used','category_maps_fit_only','reload_64_rows_verified','query_labels_used',
      'direct_component_score_omitted','prediction_units','training_seconds','elapsed_seconds','runner_elapsed_seconds',
      'geo_manifest_sha256','feature_state_content_sha256','source_sha256','artifact_sha256')
    for stage in STAGES:
        value = results[stage]; model = models[stage]
        state = 'evaluated' if value is not None else 'not_required' if rejected and stage != 'development' else 'pending'
        changes = None
        if value is not None:
            base = value['metrics'][CONTROL]; changes = {}
            for name, metric in value['metrics'].items():
                changes[name] = {k:improvement(metric[k],base[k]) for k in ('overall_mae','weighted_mae')}
                for field in ('strata','forecast_quarters'):
                    changes[name][field] = {k:improvement(item['mae'],base[field][k]['mae']) for k,item in metric[field].items()}
        result_path = rel(OUT/'evaluation'/stage/'results.json'); model_path = rel(OUT/'models'/stage/'metadata.json')
        stages[stage] = {'status':state, 'not_executed_reason':'development_rejected' if state == 'not_required' else 'no_saved_evaluation' if state == 'pending' else None,
          'dates':SPLITS[stage], 'fit_rows':COUNTS[stage][0], 'query_rows':COUNTS[stage][1],
          'result_path':result_path, 'result_sha256':snapshots[result_path], 'metadata_path':model_path, 'metadata_sha256':snapshots[model_path],
          'model_metadata':None if model is None else {k:model[k] for k in meta_keys if k in model},
          'metrics':None if value is None else value['metrics'], 'decision':None if value is None else value['decision'], 'relative_improvement_percent':changes}
    selected = frozen['selected_candidate'] if frozen is not None else None if results['development'] is None else results['development']['decision']['selected_candidate']
    return {'round':7, 'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'metric':'MAE', 'unit':'TWD',
      'status':status, 'status_text':text, 'selected_candidate':selected, 'candidate':CANDIDATE, 'control':CONTROL,
      'plan_sha256':snapshots[rel(OUT/'candidate_plan.json')], 'frozen_candidate_sha256':snapshots[rel(OUT/'frozen_candidate.json')],
      'accepted_candidate_sha256':snapshots[rel(OUT/'accepted_candidate.json')], 'acceptance':accepted,
      'acceptance_finalized':accepted is not None, 'backtest_accepted':None if accepted is None else accepted['accepted'],
      'recommendation':'v3', 'recommended_submission':'outputs/submission_v3.csv', 'new_submission':None,
      'new_release_verified':False, 'stages':stages, 'formulas':FORMULAS if plan is None else plan['formulas'],
      'thresholds':THRESHOLDS if plan is None else plan['thresholds'], 'parameters':None if plan is None else plan['parameters'],
      'target_group_weights':None if plan is None else plan['target_group_weights'], 'fixed_configuration_frozen':plan is not None,
      'public_mae':None, 'private_mae':None, 'public_score_used_for_selection':False, 'historically_untouched':False,
      'unknown_score_note':'No new R7 release or Public/Private score is certified by this report; existing v3 remains recommended.',
      'relative_improvement_sign':'positive means improvement versus same-stage v3; empty group or zero baseline is null',
      'relative_improvement_source':'arithmetic on saved aggregate MAEs; no labels or prediction arrays read',
      'reporting_only':True, 'scoring_or_training_performed':False,
      'receipt_verified_inputs':{k:v for k,v in snapshots.items() if v is not None and k.startswith('outputs/round7/')},
      'missing_optional_inputs':[k for k,v in snapshots.items() if v is None and k.endswith('.json')],
      'document_sha256':docs, 'report_script_sha256':sha(__file__),
      'verification_scope':'JSON receipts, saved identity links, source/configuration declarations, saved gate-flag consistency, and frozen documentation SHA. Binary models, NPY, raw data, CSV contents and performance gates are not replayed or recomputed.'}


def amount(value): return '無樣本（null）' if value is None else f'{value:,.2f}'
def percent(value): return '—' if value is None else f'{value:+.3f}%'
def link(path, title, base=REPORTS): return '['+title+']('+Path(os.path.relpath(path,base)).as_posix()+')'


def stage_lines(stage, item):
    lines = ['## '+stage, '', f"固定 query {item['dates'][0]}～{item['dates'][1]}；fit {item['fit_rows']:,} 列、query {item['query_rows']:,} 列。", '']
    if item['model_metadata'] is not None:
        m = item['model_metadata']
        lines += [f"已保存模型 metadata，記錄时间 `{m['created_at_utc']}`；{m['actual_iterations']:,} 輪、95 欄／23 類別，64 列重新載入核對旗標為 true。這是 metadata 證據，本報告不再次載入模型。", '',
          f"耗時依 metadata：LightGBM fit `training_seconds={m.get('training_seconds')}`；component API `elapsed_seconds={m.get('elapsed_seconds')}`；runner `runner_elapsed_seconds={m.get('runner_elapsed_seconds')}`。各口徑分開，不以終端 stdout 代替。", '',
          link(ROOT/item['metadata_path'],'模型 metadata')+f"；SHA256 `{item['metadata_sha256']}`。", '']
    if item['status'] != 'evaluated':
        message = 'Development 未通過，依固定規則不執行此年度。' if item['status'] == 'not_required' else '尚無帶有效收據的完成評估；不推測訓練是否仍在執行。'
        return lines + ['**未有本階段完成評估。** '+message+' metrics／decision 為 null，不寫成 0 分。', '']
    metrics, changes, decision = item['metrics'], item['relative_improvement_percent'], item['decision']
    lines += ['| 方法 | 整體 MAE（元） | 加權 MAE（元） | 整體改善 | 加權改善 |', '|---|---:|---:|---:|---:|']
    for name in NAMES:
        m, d = metrics[name], changes[name]
        lines.append(f"| {LABELS[name]} | {amount(m['overall_mae'])} | {amount(m['weighted_mae'])} | {percent(d['overall_mae'])} | {percent(d['weighted_mae'])} |")
    lines += ['', '下列 gate 直接轉述已保存旗標；本報告不以 MAE 重新判定門檻。', '', '| 固定檢查 | 保存結論 |', '|---|---|']
    for name, passed in decision['checks'].items(): lines.append('| '+GATE_LABELS[name]+' | '+('通過' if passed else '未通過')+' |')
    lines += ['', f"已保存 `passed={str(decision['passed']).lower()}`、`selected_candidate={decision['selected_candidate']}`；改善季度：{'、'.join(decision['improved_quarters']) or '無'}。", '',
      '| 建案組 | 列數 | v3 MAE | 新候選 MAE | 相對改善 |', '|---|---:|---:|---:|---:|']
    for group in GROUPS:
        b, c = metrics[CONTROL]['strata'][group], metrics[CANDIDATE]['strata'][group]
        lines.append(f"| {GROUP_LABELS[group]} | {b['rows']:,} | {amount(b['mae'])} | {amount(c['mae'])} | {percent(changes[CANDIDATE]['strata'][group])} |")
    lines += ['', '| 預測季度 | 列數 | v3 MAE | 新候選 MAE | 相對改善 |', '|---|---:|---:|---:|---:|']
    for quarter in QUARTERS:
        b, c = metrics[CONTROL]['forecast_quarters'][quarter], metrics[CANDIDATE]['forecast_quarters'][quarter]
        lines.append(f"| {quarter} | {b['rows']:,} | {amount(b['mae'])} | {amount(c['mae'])} | {percent(changes[CANDIDATE]['forecast_quarters'][quarter])} |")
    return lines+['',link(ROOT/item['result_path'],'完整保存結果')+f"；SHA256 `{item['result_sha256']}`。",'']


def render_report(report):
    lines = ['# 第七輪地理單價模型：固定候選驗證報告', '', '**'+report['status_text']+'**', '',
      '現行推薦仍是 '+link(ROOT/'outputs/submission_v3.csv','既有 submission_v3.csv')+'。本輪沒有由本程式核驗的新正式 release／CSV；回測接受、全量重訓與正式提交是不同階段。Public／Private 未知，本報告不讀 Kaggle 分數，也不執行上傳。', '',
      f"報告時間：`{report['generated_at_utc']}`；正式 plan SHA：`{report['plan_sha256'] or '尚未產生'}`。", '',
      '本程式只讀 R7 保存 JSON／收據與文件，不 import 訓練模組、不讀 y、模型、預測陣列或 CSV 數值，不訓練或重算逐列 MAE。相對改善由保存的彙總 MAE 計算；JSON 旗標核對不等於再次驗證二進位產物。', '']
    for stage in STAGES: lines += stage_lines(stage,report['stages'][stage])
    lines += ['## 固定資料處理與唯一變更', '',
      '新元件 `geography_unit_lightgbm` 讓既有單價 LightGBM 使用原 87 欄＋原樣 8 欄歷史 OSM 地理特徵，共 95 欄／23 類別。固定 3,999 輪、63 葉、4 threads，其他參數完全沿用舊 unit63 設定；沒有 eval_set、early stopping、參數／seed／格網／比例搜尋。', '',
      '只有 v3 的 25% 原單價 LightGBM 槽位被新元件完整替換；原 raw CatBoost 和 R6 geography CatBoost 各 37.5% 不變。新元件輸出先換算成非負、未取整總價 TWD，再於固定完整融合對原 87 欄 X 與 fit 側 reference 套一次 trend。保存的同窗 R6 geography_blend 已含 trend，作 v3 基準時不再套第二次。', '',
      '| 方法 | 固定公式 |', '|---|---|']
    for name in NAMES: lines.append('| '+LABELS[name]+' | `'+report['formulas'][name]+'` |')
    lines += ['', '有效面積 a 依序取正有限建物面積、正有限車位面積，否則取 1。訓練目標為 `y/(a×10000)`、樣本權重為 `a/100`；每列加權絕對損失滿足 `(a/100)×abs(y/(a×10000)−u)=abs(y−a×10000×u)/1000000`。因此與總價絕對誤差只差固定尺度，整體正規化亦為訓練集固定常數；不是改成偏重小房的無權重單價目標。', '',
      '類別映射只 fit 訓練側 X＋geo；未見類別固定 −1。原欄位、欄序及 dtype 保持，缺失位置不補 0,0。8 欄 OSM、NaN／格網規則與既有 prepared manifest 不改。年度快照固定 development 2022、confirmation 2023、auxiliary 2021；若未來正式 full/test 才使用既定 2024 快照。沒有外部成交答案或內政部對答案。', '',
      '原始欄位純 X 稽核沒有發現有有效資料卻整欄漏接的主要房屋屬性；完成日期全缺、管理組織常數，未把它們虛構成新特徵。這不證明所有文字已被充分利用，也不證明本次地理單價替換一定改善。', '',
      '## 固定門檻與歷史使用限制', '',
      '- Development：相對同窗 v3，整體與固定 test 比例加權 MAE 均至少改善 1%；每季、n≥500 建案組最多退步 5%。失敗即停止本輪，不執行後兩期。',
      '- Confirmation：整體與加權不得退步，至少兩季改善；n≥500 建案組最多退步 5%。',
      '- Auxiliary：只作否決，整體與加權最多退步 3%，n≥500 建案組最多退步 5%；季度只保存診斷。', '',
      '後兩期只驗 freeze 的單一候選，不重新選比例、參數或候選。所有 query y 僅用於正式完整融合評估，不先個別打分新 unit 元件；原 valid 不混入年度 fit，禁止拿見過 query 標籤的全量模型回填年度控制。', '',
      '分組沿用原 address_city|address_district|project_name：seen／new_nonmissing／missing。預測季為起點後 0–91／92–183／184–275／276 天以上；不是曆年季度。缺名空組 MAE 為 null，不填 0；缺名權重 0 亦不代表得到 0 分。', '']
    if report['target_group_weights'] is not None: lines += ['固定 test 三組權重：`'+json.dumps(report['target_group_weights'],ensure_ascii=False,sort_keys=True)+'`。','']
    if report['acceptance'] is not None:
        a=report['acceptance'];lines += [f"正式保存 `accepted={str(a['accepted']).lower()}`、`deployment_decision={a['deployment_decision']}`。即使 accepted=true，本輪仍沒有經本程式核驗的新 release／提交檔，繼續推薦 v3。",'']
    lines += ['三個年度標籤已跨輪反覆使用，不能稱完全未看過或 untouched；同建案列相關，年度 fit／query 亦非完全獨立。1% 是工程門檻，不是統計顯著性，改善不保證未來 Public／Private。固定單候選限制本輪自由度，無法消除跨輪選型偏差。', '',
      '## 追溯與報告重生', '']
    for name,title in [('reports/round7_validation_plan.md','事前驗證計畫'),('reports/round7_feature_audit.md','純 X 原始欄位稽核'),('reports/round7_work_log.md','工作紀錄')]:
        if name in report['document_sha256']:lines.append('- '+link(ROOT/name,title))
    lines += ['- '+link(REPORTS/'round7_results.json','完整機器可讀結果'),'- '+link(ROOT/'README_round7.md','第七輪入口')]
    for name,title in [('candidate_plan.json','正式 plan'),('frozen_candidate.json','development freeze'),('accepted_candidate.json','正式接受決策')]:
        path=OUT/name
        lines.append('- '+(link(path,title) if rel(path) in report['receipt_verified_inputs'] else title+'：尚未產生'))
    lines += ['', '在專案根目錄執行 `.venv\\Scripts\\python.exe src\\round7_report.py` 重生本報告、results 與 README；只整理已保存證據，不調用訓練或評估入口。原始完整參數、來源／依賴 SHA 及 metadata 索引由 results JSON 指向，報告沒有重放其二進位內容。', '']
    return '\n'.join(lines).replace('記錄时间','記錄時間')


def render_readme(report):
    lines=['# 第七輪：固定地理單價 LightGBM', '', '**'+report['status_text']+'**', '',
      '維持推薦 '+link(ROOT/'outputs/submission_v3.csv','既有 submission_v3.csv',ROOT)+'。本輪新正式 release／CSV 尚未經本流程驗證；Public／Private 未知。', '',
      '唯一候選將 v3 的 25% 舊單價 LightGBM 換為相同 95 欄地理版；兩個 CatBoost 各 37.5%，trend 一次。固定 3,999 輪、63 葉及原參數；面積權重維持總價 MAE 目標，沒有 eval_set、early stopping 或權重搜尋。', '',
      '| 固定年度 | 保存狀態 |','|---|---|']
    for stage,item in report['stages'].items():
        text='已完成評估；門檻'+('通過' if item['decision']['passed'] else '未通過') if item['status']=='evaluated' else 'Development 失敗，依規則不執行' if item['status']=='not_required' else '尚無完成評估；分數為 null'
        lines.append('| '+stage+' | '+text+' |')
    lines += ['', '## 結果與紀錄', '']
    for path,title in [(REPORTS/'optimization_round7_report.md','完整方法、分組／季度與 gate'),(REPORTS/'round7_results.json','機器可讀结果與來源 SHA'),(REPORTS/'round7_validation_plan.md','事前驗證計畫'),(REPORTS/'round7_feature_audit.md','原始欄位與純 X 稽核'),(REPORTS/'round7_work_log.md','操作工作紀錄')]:
        lines.append('- '+link(path,title,ROOT))
    lines += ['', '## 報告重生', '', '```powershell', '.\\.venv\\Scripts\\python.exe src\\round7_report.py', '```', '',
      '報告程式只讀保存 JSON／收據與文件，不訓練、不讀 y／預測陣列，不重新評分或上傳。三年度標籤已跨輪使用；固定門檻不是統計顯著性證明，不能保證 Private 改善。沒有查外部成交答案，也沒有用 Public 分數調參。', '']
    return '\n'.join(lines).replace('可讀结果','可讀結果')


def verify_inputs(snapshots):
    for name,digest in snapshots.items():
        path=ROOT/name
        if digest is None: need(not path.exists(), 'Previously absent artifact appeared; rerun report: '+name)
        else: need(path.is_file() and sha(path)==digest, 'Input changed during report rendering: '+name)


def self_check():
    empty={s:None for s in STAGES}; devpass={'decision':{'passed':True}}; devfail={'decision':{'passed':False}}
    cases=[(None,None,empty,None,'awaiting_plan'),({},None,empty,None,'awaiting_development'),
      ({},None,{**empty,'development':devfail},None,'development_rejected_finalization_pending'),
      ({},None,{**empty,'development':devpass},None,'development_passed_freeze_pending'),
      ({},{},{**empty,'development':devpass},None,'development_passed_validation_pending'),
      ({},{},{s:devpass for s in STAGES},None,'awaiting_finalization'),
      ({},{},{s:devpass for s in STAGES},{'accepted':True},'backtests_passed_release_pending'),
      ({},{},{**empty,'development':devfail},{'accepted':False},'retain_v3')]
    for plan,frozen,results,accepted,expected in cases: need(outcome(plan,frozen,results,accepted)[0]==expected,'Synthetic status mismatch')
    need(improvement(90,100)==10 and improvement(110,100)==-10 and improvement(None,100) is None and improvement(10,0) is None,'Synthetic ratio/null mismatch')
    return {'all_checks_passed':True,'synthetic_status_cases':len(cases),'ratio_and_null_cases':4,'model_training_or_scoring':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check-only',action='store_true');args=parser.parse_args()
    proof=self_check()
    if args.self_check_only: print(json.dumps(proof));return
    values=load_snapshot();report=summarize(*values)
    contents={REPORTS/'optimization_round7_report.md':render_report(report),REPORTS/'round7_results.json':json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',ROOT/'README_round7.md':render_readme(report)}
    verify_inputs(values[-1])
    for path,text in contents.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    print(json.dumps({'status':report['status'],'backtest_accepted':report['backtest_accepted'],'recommendation':'v3','outputs':[rel(p) for p in contents]},ensure_ascii=False))


if __name__=='__main__':main()
