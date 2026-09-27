"""Record the fixed R81 non-promotion decision using small saved evidence only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round81/record'
DECISION = ROOT / 'outputs/round81/decision.json'
REPORT = ROOT / 'reports/round81_delivery.md'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
SELECTOR_SHA = '357db63598a1b689a2128f0c53989dd97c2c79133218fb664f73019435f3eed7'
DOCS = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
STAGES = ('development', 'confirmation', 'auxiliary')
MARKER = '<!-- R81_RESEARCH_RETAINED_NO_NEW_CSV -->'
PINS = {
    'reports/round81_plan.md': '15ce1debd8bf9433c3038761b57bdffc5d6b9d3fb7520e5408c6205a560ecef5',
    'outputs/round81/plan.json': '7ce36eede04544afabb4512448981c6e27bfa3a68fdfa2f82339ca7e4de77005',
    'outputs/round81/results.json': '0b5dbf01b0d22290af534406a3cf8ce59f2f1200fc9270593838ef3f24f3012c',
    'reports/round81_diagnostics.json': '5f99a0e290cb3d29e95103cf934bc0608d39e4680cb3cbe852814e2feca7ade4',
    'outputs/round81/paired_vs_v22/results.json': '714a5adcc36625ce43e7788c34c4570e1460c4f5093f0f40884645bf6d9b0dba',
    'outputs/round81/results_review.json': 'd13284d627bc76426299a05cfccf473697fa8b60df0da15cfca024ac2dc30273',
    'reports/round81_report.md': '477f0acc64758c9496bfdc68a72802b9e26b24e7b802fb25edee4c3a354fed00',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def at(name):
    p = (ROOT / name).resolve()
    assert p.is_relative_to(ROOT) and p != ROOT, name
    return p


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def put(path, data):
    with path.open('xb') as f:
        f.write(data)


def write(path, value):
    data = encode(value)
    put(path, data)
    put(path.with_suffix('.sha256'), (digest(data) + '\n').encode('ascii'))


def render(main, review):
    lines = ['# R81 研究保留：不新增試交版本', '',
        '**本次不做 final fit、不產生 v24；v22 維持有限本機優先，v20 備選、v23 互補，v21 與其他歷史版本保留。** 已知 Public 最佳仍是 v18（2,388,386.4613）；Private 未知，最終版本尚未選定。', '',
        'R81 只在原 87 欄單價 LightGBM 的面積權重上乘保存的 365 日日期權重，固定 3999 樹、seed42；其餘樹、NN、reference／spatial 與 median 結構不變。差值為 R81−對照，負值代表歷史 MAE 較低。', '',
        '| 歷史窗口 | R81 MAE（元） | 整體差 vs v22 | 加權差 vs v22 | 整體差 vs v23 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for stage, title in zip(STAGES, ('開發／中間年度', '確認／最新年度', '輔助／較早年度')):
        a = main['stages'][stage]
        lines.append(f"| {title} | {a['metrics']['R81']['overall_mae']:,.2f} | {a['differences']['v22']['overall']['delta_TWD']:+,.2f} | {a['differences']['v22']['weighted']['delta_TWD']:+,.2f} | {a['differences']['v23']['overall']['delta_TWD']:+,.2f} |")
    lines += ['',
        '取捨在於：相對 v22，開發整體／加權與 testmix 退步；最新與較早年度整體雖改善，仍遜於既有互補 v23。三期億元以上價帶與臺北市對 v22 都退步。最新建案整群區間跨零，月份區塊区間為負；這些是反覆使用歷史窗口下的條件性診斷，不是 Private 勝率。', '',
        '**最強反論仍保留：**最新年度對 v22 的新案、已見建案與四季全部改善；相對 v23，R81 在最新已見建案、Q1、低價與億元以上群等仍有優勢，因此不是被 v23 全面支配。此次不採用是綜合跨期、群組代價與現有版本的取捨，不是「三期必須全贏」門檻，也不永久否定日期加權 LightGBM。', '',
        f"三次歷史 LightGBM fit 合計 {review['total_fit_seconds']:.3f} 秒，main {main['elapsed_seconds']:.3f} 秒。主流程保存模型、fit-only 詞表與權重，三期完整 query raw/TWD 重載逐位一致；此收尾只核小 JSON／收據，沒有再次載入模型、陣列或重算分數。",
        '本次登錄先核 24 份既有 CSV 與 registry 共 25 檔的 SHA，保留原 selector 與三份說明文件逐位備份；只更新研究進度 metadata，推薦、候選與未知欄位維持原值。程式、模型、來源和既有失敗／執行紀錄均保留，不刪除或無聲重試。',
        '沒有新 CSV、final fit、Kaggle 上傳／勾選或 GitHub 發布；GitHub 待最終版本確定後整理。沒有使用 Public 調參或外部成交答案，也沒有選定下一個實驗。', '',
        '[完整研究報告](round81_report.md) · [固定計畫](round81_plan.md) · [決策](../outputs/round81/decision.json) · [主結果](../outputs/round81/results.json) · [分組](round81_diagnostics.md) · [配對區間](round81_paired_vs_v22.md) · [登錄紀錄](../outputs/round81/record/record.json)', '']
    return '\n'.join(lines).encode('utf8')


def prefix(name):
    target = 'round81_delivery.md' if name.startswith('reports/') else 'reports/round81_delivery.md'
    return f'''{MARKER}
# 最新研究：R81 保留紀錄，不新增 CSV

R81 相對 v22 的開發／最新／較早年度整體 MAE 差為 +3,128.04／−5,451.75／−10,621.80 元；最新與較早年度仍遜於 v23，三期億元以上及臺北市有代價，故本次不做 final fit、不產生 v24。最新新舊案與四季皆改善，且對 v23 仍有分群優勢，不能解讀成全面無效或以三期全贏作硬門檻。

**v22 有限本機優先、v20 備選、v23 互補與 v21 歷史地位不變。** 已知 Public 最佳仍為 v18（2,388,386.4613），Private 未知、最終版本未定；沒有上傳或 GitHub 發布。[本次取捨與證據]({target})。以下原文逐位保留作歷史，以本節為最新研究進度。

---

'''.encode('utf8')


def prepare():
    assert not OUT.exists() and not REPORT.exists() and not DECISION.exists() and not DECISION.with_suffix('.sha256').exists()
    assert not SELECTOR.with_suffix('.sha256').exists()
    cache, hashes = {}, {}
    def capture(name, expected=None):
        p = at(name)
        assert p.suffix in ('.json', '.sha256', '.md', '.py') and p.stat().st_size < 2_000_000
        if name not in cache:
            cache[name] = p.read_bytes()
            hashes[name] = digest(cache[name])
        assert expected is None or hashes[name] == expected, name
        return cache[name]
    def small(name, expected=None):
        data = capture(name, expected or PINS.get(name))
        receipt = rel(at(name).with_suffix('.sha256'))
        assert capture(receipt).decode('ascii').strip() == hashes[name], name
        return json.loads(data)
    for name, expected in PINS.items():
        capture(name, expected)
    main, plan, generic, paired, review = (small(name) for name in ('outputs/round81/results.json', 'outputs/round81/plan.json',
        'reports/round81_diagnostics.json', 'outputs/round81/paired_vs_v22/results.json', 'outputs/round81/results_review.json'))
    assert main['round'] == plan['round'] == generic['round'] == review['round'] == 81
    assert main['candidate'] == plan['candidate'] == generic['candidate'] == paired['candidate'] == 'R81'
    assert main['status'] == 'three_periods_completed_no_adoption_decision' and generic['status'] == 'completed_saved_predictions_only'
    assert review['status'] == 'completed_JSON_only_review' and review['adoption_decision'] is None and not review['is_decision_authority']
    assert main['checked_stages'] == generic['checked_stages'] == review['checked_stages'] == plan['stage_order'] == list(STAGES)
    assert main['plan_sha256'] == hashes['outputs/round81/plan.json'] and main['text_plan_sha256'] == plan['text_plan_sha256'] == PINS['reports/round81_plan.md']
    assert main['source_sha256'] == plan['source_sha256'] and main['input_sha256'] == plan['input_sha256']
    assert review['report_sha256'] == PINS['reports/round81_report.md'] and not main['new_CSV'] and not main['new_submission_generated']
    assert main['new_fits'] == 3 and main['new_NN_fits'] == main['new_CatBoost_fits'] == 0
    assert paired['status'] == 'completed' and paired['control'] == 'v22' and paired['seed'] == 20260925 and paired['replicates'] == 2000
    assert paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    for label, name in (('main', 'outputs/round81/execution.json'), ('diagnostics', 'outputs/round81/diagnostics/execution.json'), ('paired', 'outputs/round81/paired_vs_v22_execution.json')):
        ex = small(name, review['input_sha256'][name])
        assert ex == review['executions'][label] and ex['exit_code'] == 0 and ex.get('failure') is None
    assert review['executions']['main']['source_sha256'] == main['source_sha256']['src/round81_experiment.py']
    assert review['executions']['diagnostics']['source_sha256'] == generic['source_sha256']['src/candidate_diagnostics_v3.py']
    assert review['executions']['paired']['source_sha256'] == paired['source_sha256']
    for name in ('outputs/round81/results.json', 'outputs/round81/plan.json', 'reports/round81_diagnostics.json', 'outputs/round81/paired_vs_v22/results.json'):
        assert review['input_sha256'][name] == hashes[name]
    assert generic['input_sha256']['outputs/round81/results.json'] == hashes['outputs/round81/results.json']
    for stage in STAGES:
        a, g, r, p = main['stages'][stage], generic['stages'][stage], review['stages'][stage], paired['stages'][stage]
        assert a['status'] == 'completed' and a['controls_full_metrics_exact'] and a['full_query_raw_TWD_reload_exact']
        assert g['all_full_saved_metrics_exact'] and p['point_metrics_reproduced'] and p['rows'] == a['query_rows']
        assert p['prediction_sha256']['R81'] == a['prediction_sha256']
        assert r['metrics'] == a['metrics'] and all(g['metrics'][k] == a['metrics'][k] for k in a['metrics'])
        for family in ('project', 'month'):
            for measure, key in (('overall', 'overall_mae'), ('weighted', 'weighted_mae')):
                v = p[family]['summary'][measure]
                assert math.isclose(v['candidate_mae'], a['metrics']['R81'][key], rel_tol=1e-12, abs_tol=1e-6)
                assert math.isclose(v['control_mae'], a['metrics']['v22'][key], rel_tol=1e-12, abs_tol=1e-6)
    assert math.isclose(sum(main['stages'][s]['fit_seconds'] for s in STAGES), review['total_fit_seconds'], abs_tol=1e-6)
    protected = plan['protected_before']
    assert main['protected_before'] == main['protected_after'] == protected and len(protected) == 25
    assert main['selector_sha256_before'] == main['selector_sha256_after'] == plan['selector_sha256_before'] == SELECTOR_SHA
    expected_csv = set(protected) - {'runs/versions.json'}
    assert len(expected_csv) == 24 and expected_csv == {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
    # Exactly one byte hash per protected CSV/registry; no CSV parsing or array/model loads.
    protected_observed = {name: sha(at(name)) for name in sorted(protected)}
    assert protected_observed == protected
    selector_original = capture(rel(SELECTOR), SELECTOR_SHA)
    previous = json.loads(selector_original)
    roles = {'private_priority': 'outputs/submission_v22_trial.csv', 'backup': 'outputs/submission_v20_trial.csv',
             'complementary_candidate': 'outputs/submission_v23_trial.csv', 'earlier_alternative': 'outputs/submission_v21_trial.csv'}
    assert all(previous[k] == v for k, v in roles.items())
    for path_key, hash_key in (('private_priority', 'sha256'), ('backup', 'backup_sha256'), ('complementary_candidate', 'complementary_candidate_sha256'), ('earlier_alternative', 'earlier_alternative_sha256')):
        assert protected_observed[previous[path_key]] == previous[hash_key]
    assert previous['known_public_best_filename'] == 'outputs/submission_v18_trial.csv' and previous['known_public_best_mae'] == 2388386.4613
    assert previous['private_mae'] is None and previous['final_version_chosen'] is False
    originals = {name: capture(name) for name in DOCS}
    assert all(MARKER.encode('utf8') not in b for b in originals.values())
    assert all(not at(n).with_suffix('.sha256').exists() and not Path(str(at(n)) + '.sha256').exists() for n in DOCS)
    capture(rel(Path(__file__)))
    now = datetime.now(timezone.utc).isoformat()
    decision = {'round': 81, 'candidate': 'R81', 'status': 'retain_research_no_new_CSV', 'completed_at_utc': now,
        'decision_scope': 'root_fixed_research_closeout; no new model selection rule', 'recommendation_changed': False,
        **roles, 'known_public_best_filename': previous['known_public_best_filename'], 'known_public_best_mae': previous['known_public_best_mae'],
        'private_mae': None, 'final_version_chosen': False, 'new_final_fits': 0, 'new_CSV': False, 'proposed_new_version': None,
        'uploaded': False, 'kaggle_selection_changed': False, 'GitHub_published': False, 'next_experiment_selected': False,
        'rationale': 'Development regresses versus v22; latest and auxiliary aggregate MAE remain worse than v23; all three >=100m and Taipei groups regress versus v22. Keep models and evidence without another final fit.',
        'strongest_counterargument': 'Latest seen/new and four quarters all improve versus v22; several groups beat v23. R81 is not uniformly dominated; this is not an all-three-windows-must-win threshold or a permanent rejection of date-weighted LightGBM.',
        'uncertainty': 'Latest project intervals cross zero and month intervals are negative; reused windows, no multiplicity correction, no Private probability.',
        'historical_fits': 3, 'historical_fit_seconds': review['total_fit_seconds'], 'main_seconds': main['elapsed_seconds'],
        'stage_metrics': {s: {'R81': main['stages'][s]['metrics']['R81'], 'differences': {k: main['stages'][s]['differences'][k] for k in ('v22', 'v23')}} for s in STAGES},
        'source_sha256': {rel(Path(__file__)): hashes[rel(Path(__file__))]}, 'input_sha256': dict(hashes),
        'report_path': 'reports/round81_report.md', 'report_sha256_observed_at_recording': PINS['reports/round81_report.md'],
        'protected_file_sha256_verified_once_before_writes': protected_observed, 'selector_sha256_before': SELECTOR_SHA,
        'models_arrays_loaded_by_recorder': 0, 'new_scores': 0, 'new_fits_by_recorder': 0}
    decision_bytes = encode(decision)
    updates = {'updated_at_utc': now,
        'latest_completed_research': {'round': 81, 'candidate': 'R81', 'status': decision['status'], 'decision': decision['status'],
            'decision_record': rel(DECISION), 'decision_sha256': digest(decision_bytes), 'report': rel(REPORT), 'new_final_fits': 0, 'new_CSV': False, 'completed_at_utc': now},
        'pending_research': {'round': 81, 'status': 'completed_retained_no_new_CSV', 'new_final_fits': 0, 'new_CSV': False, 'next_experiment_selected': False},
        'evidence': list(dict.fromkeys(previous.get('evidence', []) + list(PINS) + [rel(DECISION), rel(REPORT), rel(OUT / 'record.json')]))}
    updated = copy.deepcopy(previous)
    updated.update(updates)
    updated['selection_history'] = copy.deepcopy(previous.get('selection_history', [])) + [{'action': 'R81_research_metadata_only_no_recommendation_change',
        'selector_backup': rel(OUT / 'originals/selector.json'), 'selector_sha256': SELECTOR_SHA,
        'replaced_fields': {k: previous[k] for k in updates if k in previous}}]
    assert all(updated[k] == v for k, v in previous.items() if k not in {*updates, 'selection_history'})
    documents = {name: prefix(name) + content for name, content in originals.items()}
    assert all(documents[n].endswith(originals[n]) for n in DOCS)
    selector_bytes, report = encode(updated), render(main, review)
    assert json.loads(selector_bytes) == updated and json.loads(decision_bytes) == decision
    return dict(inputs=hashes, selector_original=selector_original, selector_bytes=selector_bytes, originals=originals,
        documents=documents, decision_bytes=decision_bytes, report=report, protected=protected_observed, completed_at_utc=now)


def commit(p):
    assert SELECTOR.read_bytes() == p['selector_original'] and all(at(n).read_bytes() == b for n, b in p['originals'].items())
    OUT.mkdir(exist_ok=False)
    for group in ('originals', 'staged'):
        (OUT / group).mkdir()
    put(OUT / 'originals/selector.json', p['selector_original'])
    put(OUT / 'originals/selector.sha256', (SELECTOR_SHA + '\n').encode('ascii'))
    for n in DOCS:
        for group, content in (('originals', p['originals'][n]), ('staged', p['documents'][n])):
            target = OUT / group / n
            target.parent.mkdir(parents=True, exist_ok=True)
            put(target, content)
    put(OUT / 'staged/selector.json', p['selector_bytes'])
    put(DECISION, p['decision_bytes'])
    put(DECISION.with_suffix('.sha256'), (digest(p['decision_bytes']) + '\n').encode('ascii'))
    put(REPORT, p['report'])
    for n in DOCS:
        assert at(n).read_bytes() == p['originals'][n]
        os.replace(OUT / 'staged' / n, at(n))
        assert at(n).read_bytes() == p['documents'][n]
    assert SELECTOR.read_bytes() == p['selector_original']
    os.replace(OUT / 'staged/selector.json', SELECTOR)
    assert SELECTOR.read_bytes() == p['selector_bytes']
    write(OUT / 'record.json', {'round': 81, 'status': 'research_decision_recorded_recommendation_unchanged',
        'decision_path': rel(DECISION), 'decision_sha256': digest(p['decision_bytes']), 'report_path': rel(REPORT),
        'report_sha256_observed_at_creation': digest(p['report']), 'source_sha256': sha(Path(__file__)), 'input_sha256': p['inputs'],
        'selector_sha256_before': SELECTOR_SHA, 'selector_sha256_after': digest(p['selector_bytes']), 'selector_metadata_updated': True,
        'recommendation_changed': False, 'candidates_and_unknown_fields_preserved': True, 'selector_original_backup': rel(OUT / 'originals/selector.json'),
        'original_document_sha256': {n: digest(b) for n, b in p['originals'].items()},
        'updated_document_sha256': {n: digest(b) for n, b in p['documents'].items()}, 'original_document_body_bytes_preserved': True,
        'protected_sha256_verified_once_before_writes': p['protected'], 'protected_files_written': False,
        'new_CSV': False, 'new_final_fits': 0, 'model_or_array_loads': 0, 'new_scores': 0,
        'uploaded': False, 'GitHub_published': False, 'completed_at_utc': p['completed_at_utc']})
    print(json.dumps({'status': 'retain_research_no_new_CSV', 'decision_sha256': digest(p['decision_bytes']),
        'selector_sha256': digest(p['selector_bytes']), 'report_sha256': digest(p['report'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    assert not OUT.exists(), 'Prior attempt exists; no automatic retry or overwrite'
    started, tick, code, failure = datetime.now(timezone.utc).isoformat(), time.monotonic(), 0, None
    try:
        prepared = prepare()  # All evidence checks and complete rendering precede output writes.
        commit(prepared)
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    OUT.mkdir(exist_ok=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure, 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - tick,
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
