"""Render round-four reports from receipt-verified JSON, without scoring/training.

Only main() writes reports/optimization_round4_report.md and reports/round4_results.json.
No project module, dataset, model, prediction array, or evaluator is imported/read.
Missing stages remain pending/null; an acceptance is not a submission-readiness check.
"""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'round4'
REPORTS = ROOT / 'reports'
PLAN_SHA256 = '062f2157633fe23bded175af0c8629d9dbe76feea3c1b721939102585535c9d0'
CONTROL = 'v2_control'
CANDIDATES = (CONTROL, 'anchor_blend', 'augmented_blend', 'spatial_raw_blend', 'combined')
STAGES = ('development', 'confirmation', 'auxiliary')
GROUPS = ('seen', 'new_nonmissing', 'missing')
QUARTERS = ('Q1', 'Q2', 'Q3', 'Q4')
PUBLIC_V2_MAE = 2883250.4138
STAGE_LABELS = {'development': 'Development 開發年度', 'confirmation': 'Confirmation 確認年度', 'auxiliary': 'Auxiliary 輔助年度'}
LABELS = {CONTROL: 'v2 基準', 'anchor_blend': '時間正規化融合', 'augmented_blend': '地址／案名前綴增強融合',
          'spatial_raw_blend': '空間總價 LightGBM 融合', 'combined': '固定綜合融合'}
CHECK_LABELS = {
    'overall_improves_at_least_1pct': '整體 MAE 至少改善 1%',
    'weighted_improves_at_least_1pct': '加權 MAE 至少改善 1%',
    'quarters_deteriorate_at_most_5pct': '每個預測季度惡化不超過 5%',
    'large_strata_deteriorate_at_most_5pct': '至少 500 筆的每組惡化不超過 5%',
    'overall_within_limit': '整體 MAE 在該階段上限內',
    'weighted_within_limit': '加權 MAE 在該階段上限內',
    'at_least_two_quarters_improve': '至少兩個預測季度改善',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_locked(path, snapshots, optional=False):
    """A missing JSON is pending; an incomplete or bad receipt is an error."""
    path = Path(path)
    receipt = path.with_suffix('.sha256')
    if not path.exists():
        require(not receipt.exists(), 'Orphan receipt: ' + rel(receipt))
        if optional:
            return None
        raise RuntimeError('Missing required JSON: ' + rel(path))
    require(path.is_file() and receipt.is_file(), 'Missing result/contract receipt: ' + rel(path))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(receipt.read_text(encoding='ascii').strip() == digest, 'SHA256 receipt mismatch: ' + rel(path))
    snapshots[rel(path)] = digest
    snapshots[rel(receipt)] = sha256(receipt)
    value = json.loads(raw.decode('utf-8-sig'))
    require(isinstance(value, dict), 'Expected JSON object: ' + rel(path))
    return value


def nonnegative(value, context):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, 'Invalid numeric value: ' + context)


def validate_metrics(metrics, expected_rows):
    for name, item in metrics.items():
        require(item['rows'] == expected_rows, 'Metric row count differs: ' + name)
        for key in ('overall_mae', 'weighted_mae'):
            nonnegative(item[key], name + '/' + key)
        for field, expected_keys in (('strata', GROUPS), ('forecast_quarters', QUARTERS)):
            require(set(item[field]) == set(expected_keys), 'Incomplete metric groups: ' + name + '/' + field)
            require(sum(part['rows'] for part in item[field].values()) == expected_rows,
                    'Metric groups do not cover the query rows: ' + name + '/' + field)
            for key, part in item[field].items():
                require(isinstance(part['rows'], int) and not isinstance(part['rows'], bool)
                        and part['rows'] >= 0, 'Invalid group row count: ' + key)
                if part['rows'] == 0:
                    require(part['mae'] is None, 'Empty group MAE must remain null: ' + key)
                else:
                    nonnegative(part['mae'], name + '/' + field + '/' + key)


def load_snapshot():
    snapshots = {}
    plan = read_locked(OUT / 'candidate_plan.json', snapshots)
    require(snapshots['outputs/round4/candidate_plan.json'] == PLAN_SHA256, 'This report expects the frozen round-four plan')
    require(set(plan['candidate_formulas']) == set(CANDIDATES), 'Unexpected fixed candidates')
    frozen = read_locked(OUT / 'frozen_candidate.json', snapshots, optional=True)
    if frozen is not None:
        require(frozen['plan_sha256'] == PLAN_SHA256, 'Frozen candidate uses another plan')
        require(frozen['selected_candidate'] in CANDIDATES, 'Unknown frozen candidate')
    results = {}
    for stage in STAGES:
        value = read_locked(OUT / 'evaluation' / stage / 'results.json', snapshots, optional=True)
        results[stage] = value
        if value is None:
            continue
        require(value['format_version'] == 1 and value['stage'] == stage, 'Unexpected result schema/stage')
        require(value['plan_sha256'] == PLAN_SHA256, 'Result uses another plan: ' + stage)
        require(value['candidate_formulas'] == plan['candidate_formulas'] and value['thresholds'] == plan['thresholds'],
                'Result formulas or gates differ from the frozen plan: ' + stage)
        require(value['prediction_units'] == 'TWD' and value['rounded'] is False, 'Unexpected metric units/rounding')
        require(value['anchor_component_scored_directly'] is False and value['reference_trend_applications_per_candidate'] == 1,
                'Unexpected anchor/trend scoring contract')
        require(value['historically_untouched'] is False and value['public_score_used_as_target'] is False,
                'Unexpected historical/public-score contract')
        expected = set(CANDIDATES) if stage == 'development' else None
        if stage != 'development':
            require(frozen is not None and frozen['selected_candidate'] != CONTROL, 'Later-stage result lacks a qualified frozen candidate')
            require(value['frozen_candidate_sha256'] == snapshots['outputs/round4/frozen_candidate.json'], 'Result uses another freeze')
            expected = {CONTROL, frozen['selected_candidate']}
            require(value['decision']['selected_candidate'] == frozen['selected_candidate'], 'Later-stage candidate was changed')
            require(isinstance(value['decision']['passed'], bool), 'Missing stage pass decision')
            require(value['decision']['candidate_reselection_performed'] is False, 'Unexpected later-stage reselection')
        require(set(value['evaluated_candidates']) == expected and set(value['metrics']) == expected,
                'Unexpected scored candidates: ' + stage)
        split = value['splits'][stage]
        require(split['dates'] == plan['splits'][stage], 'Result date window differs')
        require(split['fit_rows'] == plan['split_contract'][stage]['fit_rows']
                and split['query_rows'] == plan['split_contract'][stage]['query_rows'], 'Result split counts differ')
        validate_metrics(value['metrics'], split['query_rows'])
        require(set(value['target_group_weights']) == set(GROUPS), 'Unexpected target weight groups')
        for group, weight in value['target_group_weights'].items():
            nonnegative(weight, 'target weight/' + group)
        require(math.isclose(sum(value['target_group_weights'].values()), 1.0, abs_tol=1e-10), 'Target weights do not sum to one')
        if stage == 'development':
            decision = value['decision']
            require(decision['selected_candidate'] in CANDIDATES, 'Unknown development decision')
            require(set(decision['candidate_checks']) == set(CANDIDATES) - {CONTROL}, 'Incomplete development gate decisions')
            for item in decision['candidate_checks'].values():
                require(isinstance(item['eligible'], bool) and all(isinstance(ok, bool) for ok in item['checks'].values()),
                        'Invalid development gate flags')
    development = results['development']
    if frozen is not None:
        require(development is not None and frozen['selected_candidate'] == development['decision']['selected_candidate'],
                'Freeze differs from the saved development decision')
    acceptance = read_locked(OUT / 'accepted_candidate.json', snapshots, optional=True)
    if acceptance is not None:
        require(frozen is not None and acceptance['plan_sha256'] == PLAN_SHA256, 'Acceptance lacks the matching frozen plan')
        require(acceptance['frozen_candidate_sha256'] == snapshots['outputs/round4/frozen_candidate.json'], 'Acceptance uses another freeze')
        require(acceptance['selected_candidate'] == frozen['selected_candidate'], 'Acceptance selected another candidate')
        require(isinstance(acceptance['accepted'], bool), 'Acceptance is not finalized')
        require(acceptance['candidate_reselection_performed'] is False, 'Acceptance reselected a candidate')
        for stage in ('confirmation', 'auxiliary'):
            if frozen['selected_candidate'] != CONTROL:
                require(results[stage] is not None, 'Finalized candidate decision lacks a required stage')
                require(acceptance['stage_decisions'][stage] == results[stage]['decision'], 'Acceptance stage decision differs')
        if acceptance['accepted']:
            require(acceptance['deployment_decision'] == frozen['selected_candidate']
                    and all(results[stage]['decision']['passed'] is True for stage in ('confirmation', 'auxiliary')),
                    'Accepted status does not match the recorded gates')
        else:
            require(acceptance['deployment_decision'] == 'retain_v2', 'Rejected status must retain v2')
    return plan, frozen, results, acceptance, snapshots


def delta_percent(value, base):
    return None if value is None or base is None or base == 0 else 100.0 * (value / base - 1.0)


def amount(value):
    return '—（無樣本）' if value is None else f'{value:,.2f}'


def percent(value):
    return '—' if value is None else f'{value:+.3f}%'


def candidate_label(name):
    return LABELS[name] + ' (`' + name + '`)'


def gate_text(checks):
    failed = [CHECK_LABELS.get(key, key) for key, passed in checks.items() if passed is False]
    return '未通過：' + '；'.join(failed) if failed else '全部通過'


def link(path, title=None):
    path = Path(path)
    text = title or path.name
    if not path.exists():
        return f'{text}：待產生（`{rel(path)}`）'
    # Generated Markdown lives under reports; all links are workspace-relative.
    target = path.relative_to(REPORTS).as_posix() if path.is_relative_to(REPORTS) else '../' + rel(path)
    return f'[{text}]({target})'


def summarize(plan, frozen, results, acceptance, snapshots):
    status = 'still_pending' if acceptance is None else 'passed_backtests' if acceptance['accepted'] else 'retain_v2'
    selected = (frozen or {}).get('selected_candidate')
    if selected is None and results['development'] is not None:
        selected = results['development']['decision']['selected_candidate']
    stages = {}
    for stage, result in results.items():
        not_required = result is None and acceptance is not None and selected == CONTROL and not acceptance['accepted']
        stages[stage] = {
            'status': 'complete' if result is not None else 'not_required' if not_required else 'pending',
            'dates': plan['splits'][stage],
            'planned_fit_rows': plan['split_contract'][stage]['fit_rows'],
            'planned_query_rows': plan['split_contract'][stage]['query_rows'],
            'metrics': result['metrics'] if result is not None else None,
            'decision': result['decision'] if result is not None else None,
            'target_group_weights': result['target_group_weights'] if result is not None else None,
            'result_path': rel(OUT / 'evaluation' / stage / 'results.json'),
            'result_sha256': snapshots.get(rel(OUT / 'evaluation' / stage / 'results.json')),
            'comparison_path': rel(OUT / 'evaluation' / stage / 'comparison.csv'),
            'relative_changes_percent': None if result is None else {
                name: {key: delta_percent(item[key], result['metrics'][CONTROL][key]) for key in ('overall_mae', 'weighted_mae')}
                for name, item in result['metrics'].items()},
        }
    return {
        'round': 4, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'selected_candidate': selected, 'acceptance': acceptance,
        'plan_sha256': PLAN_SHA256, 'public_v2_mae': PUBLIC_V2_MAE,
        'public_score_source': 'Previously confirmed user-provided v2 Kaggle Public result; recorded only, never a tuning target',
        'private_mae': None, 'private_status': 'unknown',
        'stages': stages, 'candidate_formulas': plan['candidate_formulas'],
        'components': plan['components'], 'thresholds': plan['thresholds'],
        'prediction_units': 'TWD', 'lower_is_better': True,
        'relative_change_sign': 'negative means improvement versus the same-stage v2 control',
        'historically_untouched': False,
        'limitations': [
            'Development and historical confirmation have been used before; these are not untouched or Kaggle Private results.',
            'Auxiliary query rows are disjoint from the other query windows, but their labels occurred in earlier historical training.',
            'Geographic project keys and name prefixes are not verified physical-project/developer identities.',
            'Anchor-date component output cannot be directly compared with historical original-valid prices without aligning price dates.',
            'Passing fixed backtests does not guarantee Public or Private improvement.',
        ],
        'final_refit_and_new_csv': {
            'status': 'pending' if status == 'passed_backtests' else 'not_required' if status == 'retain_v2' else 'awaiting_finalized_backtest_decision',
            'performed_or_verified_by_this_report': False, 'submission_ready': None,
            'note': 'This report never inspects or certifies final training, a new CSV, or Kaggle upload readiness.',
        },
        'reporting_only': True, 'scoring_or_training_performed': False,
        'receipt_verified_inputs': snapshots, 'report_script_sha256': sha256(Path(__file__)),
    }


def stage_markdown(stage, item):
    lines = ['## ' + STAGE_LABELS[stage], '',
             f"固定檢查區間：{item['dates'][0]}～{item['dates'][1]}；計畫 fit {item['planned_fit_rows']:,} 筆、query {item['planned_query_rows']:,} 筆。", '']
    if item['status'] != 'complete':
        message = ('未執行：Development 無候選合格，最終決策已保留 v2；依凍結規則無需這個後續階段。'
                   if item['status'] == 'not_required' else '待檢查：尚無附有效 SHA256 收據的已完成結果，沒有分數或通過結論。')
        lines.extend([message, ''])
        if item['status'] != 'not_required':
            lines.extend([link(OUT / 'evaluation' / stage / 'results.json', '階段結果 JSON'), '',
                          link(OUT / 'evaluation' / stage / 'comparison.csv', '候選比較 CSV'), ''])
        return lines
    metrics, decision = item['metrics'], item['decision']
    names = [name for name in CANDIDATES if name in metrics]
    lines.extend(['單位為元，MAE 越低越好；相對變化的負號表示優於同年度 v2。', '',
                  '| 候選 | 整體 MAE | 加權 MAE | 整體相對 v2 | 加權相對 v2 | 已記錄門檻決策 |',
                  '|---|---:|---:|---:|---:|---|'])
    for name in names:
        metric, changes = metrics[name], item['relative_changes_percent'][name]
        if name == CONTROL:
            gate = '比較基準'
        elif stage == 'development':
            record = decision['candidate_checks'][name]
            gate = ('合格；' if record['eligible'] else '不合格；') + gate_text(record['checks'])
        else:
            gate = ('通過；' if decision['passed'] else '未通過；') + gate_text(decision['checks'])
        lines.append(f'| {candidate_label(name)} | {amount(metric["overall_mae"])} | {amount(metric["weighted_mae"])} | {percent(changes["overall_mae"])} | {percent(changes["weighted_mae"])} | {gate} |')
    selected = decision['selected_candidate']
    lines.extend(['', '評估器已記錄的選擇：' + candidate_label(selected) + '。報告只轉述決策，不重新選型。', ''])
    if stage != 'development':
        quarters = decision['improved_quarters']
        lines.extend(['改善的預測季度：' + ('、'.join(quarters) if quarters else '無') + '；未重新選候選。', ''])
    weights = item['target_group_weights']
    lines.extend([f'加權 MAE 使用 test 的 X 所決定之固定三組比例：已見 {weights["seen"]:.4%}、具名新建案 {weights["new_nonmissing"]:.4%}、缺名 {weights["missing"]:.4%}。', '',
                  '| 候選 | 已見 MAE | 具名新建案 MAE | 缺名 MAE |', '|---|---:|---:|---:|'])
    for name in names:
        lines.append('| ' + candidate_label(name) + ' | ' + ' | '.join(amount(metrics[name]['strata'][group]['mae']) for group in GROUPS) + ' |')
    base = metrics[CONTROL]
    lines.extend(['', '上述三組 query 筆數依序為：' + '／'.join(f'{base["strata"][group]["rows"]:,}' for group in GROUPS) + '。空組 MAE 保留 null，不當成 0 元。', '',
                  '| 候選 | Q1 相對 v2 | Q2 相對 v2 | Q3 相對 v2 | Q4 相對 v2 |', '|---|---:|---:|---:|---:|'])
    for name in names:
        changes = [delta_percent(metrics[name]['forecast_quarters'][q]['mae'], base['forecast_quarters'][q]['mae']) for q in QUARTERS]
        lines.append('| ' + candidate_label(name) + ' | ' + ' | '.join(percent(change) for change in changes) + ' |')
    lines.extend(['', 'Q1～Q4 為從檢查起日算起的第 0～91、92～183、184～275、276 天起四段，並非曆年季度。完整 MAE、月份與縣市分解保存在結果 JSON。', '',
                  link(OUT / 'evaluation' / stage / 'comparison.csv', '候選比較 CSV') + '；' + link(OUT / 'evaluation' / stage / 'results.json', '完整結果 JSON') + '。', ''])
    return lines


def render(report):
    if report['status'] == 'passed_backtests':
        outcome = '已通過固定回測門檻：' + candidate_label(report['selected_candidate']) + '。最終全量重訓與新 CSV 尚待完成；本報告不判定上傳檔已就緒。'
    elif report['status'] == 'retain_v2':
        outcome = '最終決策：保留 v2。第四輪沒有獲准取代它的新候選；不將本輪結果宣稱為新提交版本。'
    else:
        outcome = '目前仍待完成／定案（still pending）。以下僅列已封存的實際結果，未完成階段不填 0，也不先宣告新模型通過。'
    lines = ['# 第四輪固定候選回測報告', '', outcome, '',
             f'v2 已確認的 Kaggle Public MAE：**{PUBLIC_V2_MAE:,.4f} 元**；Private MAE：**未知**。Public 數字僅作既有紀錄，沒有用來調整本輪係數或門檻。', '',
             f'報告時間：{report["generated_at_utc"]}。計畫 SHA256：`{PLAN_SHA256}`。所有已讀結果皆先驗證 `.sha256` 收據；本程式只整理 JSON，不訓練或重新評分。', '']
    for stage in STAGES:
        lines.extend(stage_markdown(stage, report['stages'][stage]))
    lines.extend(['## 四個固定新候選與 v2 基準', '',
                  '下表為趨勢／局部參考前的基準預測，所有模型輸出先還原為總價元，再通過一次既有 v2 reference 的 trend 流程。融合權重與公式在 Development 評分前凍結。', '',
                  '| 候選 | 固定基準公式 |', '|---|---|'])
    for name in CANDIDATES:
        lines.append(f'| {candidate_label(name)} | `{report["candidate_formulas"][name]["base"]}` |')
    components = report['components']
    rules = components['augmented_catboost']['feature_rules']
    added_categories, added_numeric = rules['added_categorical_columns'], rules['added_numeric_columns']
    lines.extend(['', '`old_catboost` 為舊 raw 總價模型；`lightgbm_unit` 為舊面積加權單價模型已還原的總價；其餘三個名稱是本輪固定的新組件。', '',
        '- 時間正規化 CatBoost：fit 側最近四年，`G=exp(beta_time*(anchor_day-trade_day)/365.25)`，目標 `y*G/1e6`、權重 `1/G`；排除原始日期缺失列，移除三個絕對時間欄，固定 2,999 輪。組件輸出為 fit anchor 當日總價，本身沒有再加未來倍率。',
        '- 地址／案名前綴 CatBoost：原 87 欄加固定 10 欄（5 類別、5 數值），共 97 欄；詞彙、道路支持數、案名前綴支持數僅由當期 fit X 建立，沒有 query 標籤或目標編碼，固定 2,999 輪。',
        '- 空間 raw LightGBM：排除建案名、完整地址、棟戶與棟別四欄，保留 83 欄；總價目標 `y/1e6`、均等權重、固定 4,000 輪，fit 側類別映射，未知類別為 −1。', '',
        '新增類別欄：' + '、'.join('`'+name+'`' for name in added_categories) + '。新增數值欄：' + '、'.join('`'+name+'`' for name in added_numeric) + '。', '',
        '**建案前兩字是文字前綴，並非已驗證的建商／開發商身分。** 地理建案鍵也不保證對應唯一真實建案。本輪三組計分仍用原城市＋行政區＋完整案名規則，不因新增特徵改分組。', '',
        '**anchor 輸出不能直接與原 valid 的歷史成交價比 MAE。** 原 valid 含訓練末日以前交易，若未對齊價格時點便直接比較，會混淆模型誤差與時間差異；本輪只對完成既定融合與一次時間校正的年度候選評分。', '',
        '所有新組件均固定輪數，沒有 eval_set、early stopping 或看檢查結果後補調權重。參數與特徵完整內容見凍結計畫及本報告 JSON。', '',
        '## 固定門檻與歷史重用限制', '',
        '- Development：整體及固定三組加權 MAE 都至少優於 v2 1%；每個預測季度，以及筆數至少 500 的每個計分組，惡化均不得超過 5%。只在合格者中選加權 MAE 最低的一個，再凍結候選；沒有合格者即保留 v2。',
        '- Confirmation：只比較凍結候選與 v2，整體及加權 MAE 都不得惡化，至少兩個預測季度改善，至少 500 筆的每組惡化不超過 5%。不能換第二個候選重試。',
        '- Auxiliary：只作額外否決檢查，整體及加權 MAE 惡化上限 3%，至少 500 筆的每組上限 5%；季度僅作診斷。只有 Confirmation 與 Auxiliary 都通過才允許後續全量重訓。', '',
        'Development 與歷史 Confirmation 已在先前輪次使用，不能稱為新的獨立或從未接觸之驗證。Auxiliary 的檢查列雖與另兩個檢查年度不重疊，其標籤曾出現在過往歷史訓練；它增加不同年度的否決檢查，不能消除歷史選型偏差，也不能保證 Kaggle Public／Private 改善。', '',
        '全程只使用使用者提供的本機資料；未到內政部不動產成交案件實際資訊資料供應系統對答案。', '',
        '## 決策與可追溯紀錄', ''])
    if report['acceptance'] is None:
        lines.extend(['尚無 finalized 的 accepted_candidate.json；後續決策仍待檢查。', ''])
    else:
        acceptance = report['acceptance']
        lines.extend([f'既有決策紀錄：`accepted={str(acceptance["accepted"]).lower()}`、`deployment_decision={acceptance["deployment_decision"]}`。原因：{acceptance["reason"]}。', ''])
    for path, title in [
        (REPORTS/'round4_work_log.md', '第四輪工作紀錄'),
        (REPORTS/'round4_preprocessing_record.md', '第四輪資料處理紀錄'),
        (REPORTS/'round4_model_history.md', '既有模型歷史與候選理由'),
        (REPORTS/'round4_location_opportunity.md', '地點與案名特徵稽核'),
        (REPORTS/'round4_validation_plan.md', '輔助年度驗證計畫'),
        (OUT/'candidate_plan.json', '凍結候選計畫'),
        (OUT/'frozen_candidate.json', 'Development 凍結選擇'),
        (OUT/'accepted_candidate.json', '最終回測接受決策'),
    ]:
        lines.append('- ' + link(path, title))
    lines.extend(['- [完整報告 JSON](round4_results.json)', '',
                  '本報告不載入最終模型、不檢查或產生提交 CSV，也不宣告 Kaggle 上傳完成。通過回測時，最終重訓與新 CSV 仍須另行完成並查核。', ''])
    return '\n'.join(lines)


def main():
    plan, frozen, results, acceptance, snapshots = load_snapshot()
    report = summarize(plan, frozen, results, acceptance, snapshots)
    markdown = render(report)
    # Refuse a mixed snapshot if an input was replaced while this report was built.
    for path, digest in snapshots.items():
        require(sha256(ROOT / path) == digest, 'Input changed during report rendering: ' + path)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'optimization_round4_report.md').write_text(markdown, encoding='utf-8')
    (REPORTS / 'round4_results.json').write_text(encoded, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'selected_candidate': report['selected_candidate'],
                      'outputs': ['reports/optimization_round4_report.md', 'reports/round4_results.json']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
