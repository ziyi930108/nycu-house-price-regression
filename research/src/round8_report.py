"""Render R8 status from saved JSON receipts and docs; never train or score.

Standard library only. Does not read y, arrays, model binaries or CSV values.
A passed backtest is not a formal release; the recommendation stays v3.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORTS = ROOT/'outputs/round8', ROOT/'reports'
CONTROL, CANDIDATE = 'v3_control', 'project_balanced_spatial_reference'
NAMES, STAGES = (CONTROL, CANDIDATE), ('development', 'confirmation', 'auxiliary')
GROUPS, QUARTERS = ('seen', 'new_nonmissing', 'missing'), ('Q1', 'Q2', 'Q3', 'Q4')
LABELS = {CONTROL:'現行 v3', CANDIDATE:'固定建案平衡地理參考'}
GROUP_LABELS = {'seen':'已見建案', 'new_nonmissing':'新且非缺名建案', 'missing':'建案缺名'}
SPLITS = {'development':['2022-08-11','2023-08-10'], 'confirmation':['2023-08-11','2024-08-10'], 'auxiliary':['2021-08-11','2022-08-10']}
COUNTS = {'development':(240316,81234), 'confirmation':(321550,133697), 'auxiliary':(134805,105511)}
FORMULAS = {CONTROL:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once', CANDIDATE:'p3 + (1-c)*a*(Gclip-Bfuture) only on eligible weak sources; a=.30*exp(-project_MAD/.18); Gclip/Bfuture in [.65,1.5]; no extra trend'}
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
DOCS = ('reports/round8_validation_plan.md', 'reports/round8_design_review.md', 'reports/round8_location_feasibility.md', 'reports/round8_work_log.md')


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


def check_parameters(value):
    # Read the complete dictionary from the receipt-backed plan, not a helper import.
    expected={'format_version':1,'radius_m':2000.0,'maximum_extent_m_each':2000.0,
      'maximum_projects':40,'minimum_projects':5,
      'same_nonmissing_group':['address_city','building_type','main_use'],
      'fit_project_name_required':True,'exclude_query_project':True,
      'fit_area_bounds_m2':[15.0,1500.0],'query_area_bounds_m2':[15.0,1500.0],
      'fit_recent_years':4.0,'days_per_year':365.25,'maximum_weight':0.30,'MAD_scale':0.18,
      'eligible_v2_sources':['district_style','base_fallback'],
      'geographic_price_ratio_bounds':[0.65,1.5],'future_day_cap':366.0,
      'v2_refitted':False,'additional_trend_applied_to_G':False,
      'Dq_plus_Dr_distance_hard_gate':False}
    need(isinstance(value,dict),'Reference parameters must be a dictionary')
    for key,expected_value in expected.items():
        need(value.get(key)==expected_value,'Fixed reference parameter differs: '+key)


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
    need(plan['round'] == 8 and plan['candidate'] == CANDIDATE and plan['control'] == CONTROL, 'Wrong R8 plan identity')
    need(plan['formulas'] == FORMULAS and plan['thresholds'] == THRESHOLDS and plan['splits'] == SPLITS, 'Fixed formulas/gates/windows differ')
    need(plan['new_tree_models_trained']==0 and plan['original_feature_columns']==87 and plan['geography_feature_columns']==8,'Reference input configuration differs')
    check_parameters(plan['parameters'])
    for flag, expected in [('single_development_candidate',True), ('tuning_or_weight_search',False), ('public_score_used_for_selection',False), ('historically_untouched',False), ('external_property_answers_used',False), ('query_labels_passed_to_component',False), ('query_X_used_to_fit_reference',False)]:
        need(plan[flag] is expected, 'Plan policy differs: '+flag)
    need(plan['target_group_weights']['missing'] == 0 and set(plan['target_group_weights']) == set(GROUPS), 'Target group weights differ')
    need(abs(sum(plan['target_group_weights'].values())-1) < 1e-12, 'Target weights do not sum to one')
    for name in DOCS[:3]: need(plan['dependency_sha256'][name] == docs.get(name), 'Frozen research document differs: '+name)
    for stage in STAGES:
        spec = plan['split_contract'][stage]
        need((spec['fit_rows'],spec['query_rows']) == COUNTS[stage], 'Planned stage counts differ')
        model = models[stage]
        if model is not None:
            need(model['round'] == 8 and model['stage'] == stage and model['plan_sha256'] == psha and model['component'] == CANDIDATE, 'Model metadata identity differs')
            need(model['source_sha256'] == plan['source_sha256'] and model['parameters'] == plan['parameters'], 'Model metadata source/parameters differ')
            need(type(model['new_tree_models_trained']) is int and model['new_tree_models_trained']==0,'Unexpected tree training')
            need((model['outer_fit_rows'],model['query_rows']) == COUNTS[stage], 'Model metadata rows differ')
            for flag, expected in [('reload_64_rows_verified',True), ('v3_control_replayed_all_rows',True), ('feature_frames_unchanged',True), ('query_labels_used',False), ('reference_state_uses_query',False)]: need(model[flag] is expected, 'Model policy differs: '+flag)
            need(model['prediction_units'] == 'TWD', 'Prediction units differ')
            need(model['frozen_candidate_sha256'] == (None if stage == 'development' else snapshots[rel(OUT/'frozen_candidate.json')]), 'Model freeze identity differs')
            key = 'outputs/round6/geography/'+stage+'/manifest.json'
            need(model['geo_manifest_sha256'] == plan['stage_dependency_sha256'][stage][key], 'Saved model geography identity differs')
            need(plan['stage_dependency_sha256'][stage].get(model['reference_path'])==model['reference_sha256'],'Saved fit-reference identity differs')
        result = results[stage]
        if result is None: continue
        need(model is not None, 'Evaluation lacks saved model metadata')
        need(result['round'] == 8 and result['stage'] == stage and result['plan_sha256'] == psha, 'Evaluation identity differs')
        need(result['source_sha256'] == plan['source_sha256'] and result['formulas'] == FORMULAS and result['thresholds'] == THRESHOLDS and result['target_group_weights'] == plan['target_group_weights'], 'Evaluation protocol differs')
        need(result['evaluated_candidates'] == list(NAMES) and set(result['metrics']) == set(NAMES), 'Unexpected scored candidate list')
        need((result['fit_rows'],result['query_rows']) == COUNTS[stage], 'Evaluation row counts differ')
        need(result['frozen_candidate_sha256'] == (None if stage == 'development' else snapshots[rel(OUT/'frozen_candidate.json')]), 'Evaluation freeze differs')
        name = rel(OUT/'models'/stage/'metadata.json')
        need(result['input_artifact_sha256'][name] == snapshots[name], 'Evaluation model-metadata SHA differs')
        for flag, expected in [('v3_control_replayed_all_rows',True), ('new_reference_replayed_all_rows',True), ('query_labels_used_for_training',False), ('public_score_used',False), ('historically_untouched',False)]: need(result[flag] is expected, 'Evaluation policy differs: '+flag)
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
        need(accepted['round'] == 8 and accepted['candidate'] == CANDIDATE and accepted['plan_sha256'] == psha and accepted['frozen_candidate_sha256'] == snapshots[rel(OUT/'frozen_candidate.json')], 'Acceptance identity differs')
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
    meta_keys=('component','created_at_utc','outer_fit_rows','query_rows','parameters',
      'reload_64_rows_verified','v3_control_replayed_all_rows','feature_frames_unchanged','query_labels_used',
      'reference_state_uses_query','new_tree_models_trained','prediction_units','runner_elapsed_seconds',
      'reference_path','reference_sha256','geo_manifest_sha256','source_sha256','artifact_sha256',
      'fit_X_sha256','query_X_sha256','fit_geo_sha256','query_geo_sha256')
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
    return {'round':8, 'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'metric':'MAE', 'unit':'TWD',
      'status':status, 'status_text':text, 'selected_candidate':selected, 'candidate':CANDIDATE, 'control':CONTROL,
      'plan_sha256':snapshots[rel(OUT/'candidate_plan.json')], 'frozen_candidate_sha256':snapshots[rel(OUT/'frozen_candidate.json')],
      'accepted_candidate_sha256':snapshots[rel(OUT/'accepted_candidate.json')], 'acceptance':accepted,
      'acceptance_finalized':accepted is not None, 'backtest_accepted':None if accepted is None else accepted['accepted'],
      'recommendation':'v3', 'recommended_submission':'outputs/submission_v3.csv', 'new_submission':None,
      'new_tree_models_trained':0,'original_feature_columns':87,'geography_feature_columns':8,
      'reference_query_feature_columns':95,'feature_count_interpretation':'87 original X plus 8 geographic columns for reference lookup; no new 95-column tree model',
      'new_release_verified':False, 'stages':stages, 'formulas':FORMULAS if plan is None else plan['formulas'],
      'thresholds':THRESHOLDS if plan is None else plan['thresholds'], 'parameters':None if plan is None else plan['parameters'],
      'target_group_weights':None if plan is None else plan['target_group_weights'], 'fixed_configuration_frozen':plan is not None,
      'public_mae':None, 'private_mae':None, 'public_score_used_for_selection':False, 'historically_untouched':False,
      'unknown_score_note':'No new R8 release or Public/Private score is certified by this report; existing v3 remains recommended.',
      'relative_improvement_sign':'positive means improvement versus same-stage v3; empty group or zero baseline is null',
      'relative_improvement_source':'arithmetic on saved aggregate MAEs; no labels or prediction arrays read',
      'reporting_only':True, 'scoring_or_training_performed':False,
      'receipt_verified_inputs':{k:v for k,v in snapshots.items() if v is not None and k.startswith('outputs/round8/')},
      'missing_optional_inputs':[k for k,v in snapshots.items() if v is None and k.endswith('.json')],
      'document_sha256':docs, 'report_script_sha256':sha(__file__),
      'verification_scope':'JSON receipts, saved identity links, source/configuration declarations, saved gate-flag consistency, and frozen documentation SHA. Binary models, NPY, raw data, CSV contents and performance gates are not replayed or recomputed.'}


def amount(value): return '無樣本（null）' if value is None else f'{value:,.2f}'
def percent(value): return '—' if value is None else f'{value:+.3f}%'
def link(path, title, base=REPORTS): return '['+title+']('+Path(os.path.relpath(path,base)).as_posix()+')'


def stage_lines(stage, item):
    lines = ['## '+stage, '', f"固定 query {item['dates'][0]}～{item['dates'][1]}；fit {item['fit_rows']:,} 列、query {item['query_rows']:,} 列。", '']
    if item['model_metadata'] is not None:
        m=item['model_metadata']
        lines += [f"已保存參考資料結構 metadata，時間 `{m['created_at_utc']}`；新樹模型數為 0，原 87 欄 X＋8 欄地理資訊用於查詢。64 列重新載入及既有 v3 全列 replay 旗標均為 true。這是保存 metadata 的證據，本報告不再次載入二進位資料。", '',
          f"runner 耗時依 metadata：`runner_elapsed_seconds={m.get('runner_elapsed_seconds')}`。這是該 runner 計時區段的參考建構、推論及核對耗時，不稱為新樹訓練時間，不以 stdout 代替。", '',
          link(ROOT/item['metadata_path'],'參考 metadata')+f"；SHA256 `{item['metadata_sha256']}`。", '']
    if item['status'] != 'evaluated':
        message = 'Development 未通過，依固定規則不執行此年度。' if item['status'] == 'not_required' else '尚無帶有效收據的完成評估；不推測參考建構或推論是否仍在執行。'
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
    lines=['# 第八輪固定建案平衡地理參考：驗證報告','','**'+report['status_text']+'**','',
      '現行推薦仍是 '+link(ROOT/'outputs/submission_v3.csv','既有 submission_v3.csv')+'。本輪沒有由本程式核驗的新正式 release／CSV；回測接受、完整資料參考與正式發布是不同階段。本輪新 Public／Private 未知，報告不讀 Kaggle 分數也不上傳。','',
      f"報告時間：`{report['generated_at_utc']}`；正式 plan SHA：`{report['plan_sha256'] or '尚未產生'}`。",'',
      '只讀 R8 保存 JSON／收據與文件，僅用標準函式庫；不 import 專案模組、不讀 y、預測陣列、模型／reference 二進位或 CSV 值，不建參考、不訓練、不重算逐列 MAE。相對改善只由保存彙總 MAE 計算，核對 JSON 不等於再次重播二進位產物或重新判定性能門檻。','']
    for stage in STAGES:lines+=stage_lines(stage,report['stages'][stage])
    lines+=['## 固定方法與資料處理','',
      '本輪新樹模型數為 0。重用 v3：原 raw CatBoost 與 R6 geography CatBoost 各 37.5%，原單價 LightGBM 換算總價 25%。沒有採用 R7 地理單價模型。原 87 欄 X 與既有 8 欄地理資料合計 95 欄資訊供參考查詢，不能寫成重新訓練一個 95 欄樹模型。','',
      '只對既有 reference source 為 district_style／base_fallback 的合格列修正，其他來源與不合格／支持不足列逐列保持 v3。原 district_style 本來已跨道路；本輪增量是固定地理範圍及建案平衡，而非首次跨道路。','',
      '半徑固定 2,000 公尺，兩端 extent 各≤2,000；同非缺失 city、building_type、main_use，允許跨 district，不另設同區條件。fit 參考只取同期既有 v2 有限 residual、15～1,500 平方公尺面積、具名建案、原 anchor 前 4×365.25 天至 anchor 的有限日期；query 沿用固定面積資格。排除 query 同 city|district|project 的案。','',
      '先對每案在半徑內的合格實際列取 residual median，再按每案最近合格實際位置距離排序，同距離依 project key 排序。最多 40 案且至少 5 案；案間取未加權 median／MAD，不讓案內交易多寡增加票數，不以建案中心替代實際位置。','',
      '保留原 v2 beta、fill 與 anchor，不重新 fit hedonic 或趨勢。G_raw=area×exp(center+h_query@beta)，再截斷至 [0.65,1.5]×B_future；此尺度與原 v2 的未套 future trend 基底不同。B_future 由原始 v3 blend 套既有最多 366 天趨勢，不能把已含 trend 的 p3 再套。a=0.30×exp(−MAD/0.18)，p8=p3+(1−c)×a×(G−B_future)，G 和 p8 不加第二次 trend。0.30 借用原 street 的數值，不代表可靠程度相同。','',
      '| 方法 | 正式固定公式 |','|---|---|']
    for name in NAMES:lines.append('| '+LABELS[name]+' | `'+report['formulas'][name]+'` |')
    if report['parameters'] is not None:
        lines+=['','完整參數直接來自有收據的正式 plan：','','```json',json.dumps(report['parameters'],ensure_ascii=False,sort_keys=True,indent=2),'```']
    else:lines+=['','正式 plan 尚未凍結，parameters 為 null；以上為選定方案說明，不代表已執行。']
    lines+=['','## 資料契約與限制','',
      '道路或平面交會是粗位置，不是住宅門牌。extent 是幾何範圍，不是住宅定位誤差或信賴界線；單交點 extent=0 也不是誤差=0。不採 distance+兩端 extent 的住宅距離硬限制，不聲稱立體交通連通。無位置不補 0,0。','',
      '歷史地圖固定 development 2022、confirmation 2023、auxiliary 2021，沿用同窗 R6 prepared manifest、原去重與 indices SHA。重現含 target 的原完整列去重可能讀取既有 target，但不以價格建立座標或新增地理規則。禁止向內政部等外部成交答案來源對答案。','',
      '元件只接 fit reference／X／geo 與 query X／geo、既有預測，不接 query y。driver 的 partition 可以載入標籤供正式完整候選 score，因此不宣稱整个流程從未讀標籤。建構時不把 query 標籤傳給元件，不用 query y 挑半徑、支持數、日期或權重。','',
      '## 固定門檻與階段權限','',
      '- Development：對同窗 v3，整體及固定 test 比例加權 MAE 均至少改善 1%；每季及 n≥500 三組最多退步 5%。失敗即停止，不執行後兩期。',
      '- Confirmation：整體及加權不得退步，至少兩季改善，n≥500 三組最多退步 5%。',
      '- Auxiliary：整體及加權最多退步 3%，n≥500 三組最多退步 5%；季度只作診斷。','',
      '後兩期只驗已 freeze 的唯一候選並作否決，不重選半徑、支持數、比例或候選。完整 query 評分，不只對修正子集計分，也不先打分 G 再挑方法。Public 不參與選型。','',
      '三組键保持 address_city|address_district|project_name；跨 district 鄰居不代表改評分鍵。四季為起點後 0–91／92–183／184–275／276 天以上。空組 MAE 為 null，missing 權重 0 不表示缺名預測無誤差。','']
    if report['target_group_weights'] is not None:lines+=['固定 test X 三組權重：`'+json.dumps(report['target_group_weights'],ensure_ascii=False,sort_keys=True)+'`。','']
    if report['acceptance'] is not None:
        a=report['acceptance'];lines+=[f"正式保存 `accepted={str(a['accepted']).lower()}`、`deployment_decision={a['deployment_decision']}`。即使 accepted=true，本程式也沒有驗證新 release／CSV，仍推薦 v3。",'']
    lines+=['三年度標籤已跨輪反覆使用，不能稱完全未見或 untouched。同案列相關、年度 fit 亦重疊；模型對 fit 的過擬合與反覆研究選型偏差不同。1% 是工程門檻，不是統計顯著性；五個案不代表五個完全獨立市場觀測。固定一個候選無法消除跨輪偏差，回測改善不保證新 Public／Private。','','## 證據與報告重生','']
    for name,title in [('reports/round8_validation_plan.md','固定驗證計畫'),('reports/round8_design_review.md','評分前研究審查'),('reports/round8_location_feasibility.md','純 X 地理可行性'),('reports/round8_work_log.md','工作紀錄')]:
        if name in report['document_sha256']:lines.append('- '+link(ROOT/name,title))
    lines+=['- '+link(REPORTS/'round8_results.json','完整機器可讀結果'),'- '+link(ROOT/'README_round8.md','第八輪入口')]
    for name,title in [('candidate_plan.json','正式 plan'),('frozen_candidate.json','development freeze'),('accepted_candidate.json','正式接受決策')]:
        path=OUT/name;lines.append('- '+(link(path,title) if rel(path) in report['receipt_verified_inputs'] else title+'：尚未產生'))
    lines+=['','在專案根目錄執行 `.venv\\Scripts\\python.exe src\\round8_report.py` 重生報告、results 與 README。results 保留來源與 metadata 身分；二進位、陣列與 CSV 的完整性及性能由原 driver／封存流程驗證，本程式不代替它們。','']
    return '\n'.join(lines).replace('整个','整個').replace('三組键','三組鍵')


def render_readme(report):
    lines=['# 第八輪：固定建案平衡地理參考','','**'+report['status_text']+'**','',
      '維持推薦 '+link(ROOT/'outputs/submission_v3.csv','既有 submission_v3.csv',ROOT)+'。本輪沒有經此報告核驗的新正式 release／CSV；新 Public／Private 未知。','',
      '不訓練新樹，重用 v3 原三個元件與比例。原 87 欄＋8 欄 OSM 用於參考查詢；只在行政區參考／基底回退來源補粗地理近鄰，其他列原樣。固定 2 公里半徑、兩端 extent≤2公里、同 city／建物型態／用途且允許跨 district。','',
      '每案 residual median 後取最近最多 40 案、至少 5 案；a=0.30×exp(−MAD/0.18)，G 截斷至 [0.65,1.5]×B_future，以 p3+(1−c)×a×(G−B_future) 修正，不重複 trend。','',
      '| 固定年度 | 保存狀態 |','|---|---|']
    for stage,item in report['stages'].items():
        text='已完成評估；門檻'+('通過' if item['decision']['passed'] else '未通過') if item['status']=='evaluated' else 'Development 失敗，依規則不執行' if item['status']=='not_required' else '尚無完成評估；分數為 null'
        lines.append('| '+stage+' | '+text+' |')
    lines+=['','## 結果與紀錄','']
    paths=[(REPORTS/'optimization_round8_report.md','完整方法、分組／季度與 gate'),(REPORTS/'round8_results.json','機器可讀結果與來源 SHA')]
    paths += [(ROOT/name,title) for name,title in [('reports/round8_validation_plan.md','固定驗證計畫'),('reports/round8_design_review.md','評分前研究審查'),('reports/round8_location_feasibility.md','純 X 地理可行性'),('reports/round8_work_log.md','工作紀錄')] if name in report['document_sha256']]
    for path,title in paths:lines.append('- '+link(path,title,ROOT))
    lines+=['','## 報告重生','','```powershell','.\\.venv\\Scripts\\python.exe src\\round8_report.py','```','',
      '只讀保存 JSON／收據與文件，不訓練、不讀 y／預測陣列、不重新評分或上傳。extent 是幾何範圍，不是住宅定位誤差。歷史三年度已跨輪使用；1% 不是統計顯著性，不能保證 Private。沒有外部成交答案或 Public 選型。','']
    return '\n'.join(lines)


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
    contents={REPORTS/'optimization_round8_report.md':render_report(report),REPORTS/'round8_results.json':json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',ROOT/'README_round8.md':render_readme(report)}
    verify_inputs(values[-1])
    for path,text in contents.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    print(json.dumps({'status':report['status'],'backtest_accepted':report['backtest_accepted'],'recommendation':'v3','outputs':[rel(p) for p in contents]},ensure_ascii=False))


if __name__=='__main__':main()
