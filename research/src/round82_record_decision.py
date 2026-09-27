"""Record the fixed R82 non-promotion decision using small saved evidence only."""
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
OUT = ROOT / 'outputs/round82/record'
DECISION = ROOT / 'outputs/round82/decision.json'
REPORT = ROOT / 'reports/round82_delivery.md'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
SELECTOR_SHA = '9e07fee7910dbe33d94938e2f846849cb43b98eecfa084f4a7169155f2c1488c'
DOCS = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
STAGES = ('development', 'confirmation', 'auxiliary')
MARKER = '<!-- R82_RESEARCH_RETAINED_NO_NEW_CSV -->'
PINS = {
    'reports/round82_plan.md': '0a442808919c9bd15de4b40b074d890bb4d6952a7ef597f780b9071b34a3b26a',
    'outputs/round82/plan.json': 'e25db9d4167d373afc282ac73ad5615d7f86bb706f32740b9642c8adfa30e298',
    'outputs/round82/results.json': '436576fe830bc02ddf5b5b1cd59d8d0c8b176507609c362dedd6d197a96f561f',
    'reports/round82_diagnostics.json': 'cc0dc9f39c36267aae1857c96b89068a214ef11811fe007878345791cf56a87b',
    'outputs/round82/paired_vs_R81/results.json': 'eb6f921ba5f3b24a30d975ae581b2f78976787fb05ec04d5ecfa1b6061d7bf76',
    'outputs/round82/results_review.json': '07e85d6cdf31faaea31bf5ceebe3c74effda586417e0d60929255e405c67d2c3',
    'reports/round82_report.md': 'bee5dff2ad756a377c9321f72b37d0d17fc2d05331cfb4079a36c9e84d103cf8',
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
    lines = ['# R82 研究保留：anchor 暫不新增試交', '',
        '**本次保留研究，不做 final fit、不產生 v24。v22 仍為有限本機優先、v20 備選、v23 互補，v21 與歷史版本保留。** 已知 Public 最佳仍為 v18（2,388,386.4613）；Private 未知、最終版本未定。', '',
        '唯一變更是把 R81 單價分支的 reference trend 改為 anchor；相同 base、local reference 日期調整、空間殘差的 beta／Bfuture、NN／log 與最外 median 都保留。這不等於移除整個系統的時間修正，也不證明原先有重複加價。', '',
        '以下差值為 R82−對照，負值表示歷史 MAE 較低。', '',
        '| 歷史窗口 | R82 MAE（元） | 整體差 vs R81 | 加權差 vs R81 | 整體差 vs v22 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for stage, title in zip(STAGES, ('開發／中間年度', '確認／最新年度', '輔助／較早年度')):
        a = main['stages'][stage]
        lines.append(f"| {title} | {a['metrics']['R82']['overall_mae']:,.2f} | {a['differences']['R81']['overall']['delta_TWD']:+,.2f} | {a['differences']['R81']['weighted']['delta_TWD']:+,.2f} | {a['differences']['v22']['overall']['delta_TWD']:+,.2f} |")
    lines += ['',
        '主要代價是三期新案都退步，最新新案相對 R81 約多 60,193 元 MAE；最新與輔助兩期的整體／加權建案與月份區間共八項全正，開發四項則跨零。三期 unit reference 與 spatial 層都退步；開發最終改善大多來自 ≥5,000 萬元價帶，其全期貢獻 −2,814.85 元，相對淨改善 −2,954.92 元。這是誤差分解，不能推論逐筆真價或制定價格規則。', '',
        '**反論必須保留：**開發整體、加權與 testmix 均改善，最新已見建案、Q1 與低於千萬元群也改善，所以 anchor 並非全面無效。但目前額外試交價值不足，保留程式與結果、不新增候選 CSV。這不是三期必須全勝的硬門檻，也不證明 trend 是最優方法或已知 Private 較好。', '',
        f"R82 0 新 fit、0 樹／NN模型載入或推論；main {main['elapsed_seconds']:.3f} 秒、分組診斷 {review['diagnostics_seconds']:.3f} 秒、paired {review['paired_seconds']:.3f} 秒。三期原 R81 base／trend reference／spatial／median 完整控制逐位重現；候選與控制共 reference、spatial 各六次，原生模型 reload 只是承接先前證據。",
        '歷史窗口反覆重用，配對區間未校正多輪選型，不是 Private 勝率。本登錄只讀小 JSON／收據、不再次載入向量或重算分數；24 份 CSV 與 registry 共 25 檔各核一次 SHA。原 selector 與三份說明逐位備份，僅更新研究 metadata，推薦、候選與其他欄位保持原值。',
        '原始完整報告保留，包含其中單價層表格的原排版；本摘要另提供連續三行主結果。所有程式、模型、來源及既有失敗／執行紀錄保留，不刪除或隱藏失敗。尚未選下一實驗，沒有 final fit、新 CSV、Kaggle 上傳／勾選或 GitHub 發布；GitHub 待定版整理。', '',
        '[完整報告](round82_report.md) · [固定計畫](round82_plan.md) · [決策](../outputs/round82/decision.json) · [主結果](../outputs/round82/results.json) · [分組](round82_diagnostics.md) · [配對區間](round82_paired_vs_R81.md) · [登錄紀錄](../outputs/round82/record/record.json)', '']
    return '\n'.join(lines).encode('utf8')


def prefix(name):
    target = 'round82_delivery.md' if name.startswith('reports/') else 'reports/round82_delivery.md'
    return f'''{MARKER}
# 最新研究：R82 保留紀錄，不新增 CSV

只把 R81 單價 reference 改用 anchor 後，開發／最新／較早年度整體 MAE 相对 R81 為 −2,954.92／+28,362.84／+22,126.85 元，新案三期皆退步；最新與較早年度八個配對區間全正。保留開發整體／加權／testmix及最新部分群組改善的反論，但目前不足以新增試交；不代表原 trend 已證明最優。

**v22 有限本機優先、v20 備選、v23 互補及 v21 歷史地位不變。** Public 已知最佳仍是 v18（2,388,386.4613），Private 未知、最終版本未定；0 新 fit、沒有新 CSV、上傳或 GitHub 發布，尚未選下一實驗。[清楚三期摘要與取捨]({target})。以下原文逐位保留作歷史。

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
    main, plan, generic, paired, review = (small(name) for name in ('outputs/round82/results.json', 'outputs/round82/plan.json',
        'reports/round82_diagnostics.json', 'outputs/round82/paired_vs_R81/results.json', 'outputs/round82/results_review.json'))
    assert main['round'] == plan['round'] == generic['round'] == review['round'] == 82
    assert main['candidate'] == plan['candidate'] == generic['candidate'] == paired['candidate'] == 'R82'
    assert main['status'] == 'three_periods_completed_no_adoption_decision' and generic['status'] == 'completed_saved_predictions_only'
    assert review['status'] == 'completed_JSON_only_review' and review['adoption_decision'] is None and not review['is_decision_authority']
    assert main['checked_stages'] == generic['checked_stages'] == review['checked_stages'] == plan['stage_order'] == list(STAGES)
    assert main['plan_sha256'] == hashes['outputs/round82/plan.json'] and main['text_plan_sha256'] == plan['text_plan_sha256'] == PINS['reports/round82_plan.md']
    assert main['source_sha256'] == plan['source_sha256'] and main['input_sha256'] == plan['input_sha256']
    assert review['report_sha256'] == PINS['reports/round82_report.md'] and not main['new_CSV'] and not main['new_submission_generated']
    assert main['new_fits'] == main['new_NN_fits'] == main['new_CatBoost_fits'] == main['model_predict_calls'] == main['tree_NN_model_loads'] == 0
    assert paired['status'] == 'completed' and paired['control'] == 'R81' and paired['seed'] == 20260925 and paired['replicates'] == 2000
    assert paired['fits'] == paired['model_predict_calls'] == 0 and not paired['new_CSV']
    for label, name in (('main', 'outputs/round82/execution.json'), ('diagnostics', 'outputs/round82/diagnostics/execution.json'), ('paired', 'outputs/round82/paired_vs_R81_execution.json')):
        ex = small(name, review['input_sha256'][name])
        assert ex == review['executions'][label] and ex['exit_code'] == 0 and ex.get('failure') is None
    assert review['executions']['main']['source_sha256'] == main['source_sha256']['src/round82_unit_anchor.py']
    assert review['executions']['diagnostics']['source_sha256'] == generic['source_sha256']['src/candidate_diagnostics_v3.py']
    assert review['executions']['paired']['source_sha256'] == paired['source_sha256']
    for name in ('outputs/round82/results.json', 'outputs/round82/plan.json', 'reports/round82_diagnostics.json', 'outputs/round82/paired_vs_R81/results.json'):
        assert review['input_sha256'][name] == hashes[name]
    assert generic['input_sha256']['outputs/round82/results.json'] == hashes['outputs/round82/results.json']
    for stage in STAGES:
        a, g, r, p = main['stages'][stage], generic['stages'][stage], review['stages'][stage], paired['stages'][stage]
        assert a['status'] == 'completed' and a['controls_full_metrics_exact'] and a['R81_base_reference_spatial_median_full_vectors_exact']
        assert g['all_full_saved_metrics_exact'] and p['point_metrics_reproduced'] and p['rows'] == a['query_rows']
        assert p['prediction_sha256']['R82'] == a['prediction_sha256']
        assert r['metrics'] == a['metrics'] and all(g['metrics'][k] == a['metrics'][k] for k in a['metrics'])
        for family in ('project', 'month'):
            for measure, key in (('overall', 'overall_mae'), ('weighted', 'weighted_mae')):
                v = p[family]['summary'][measure]
                assert math.isclose(v['candidate_mae'], a['metrics']['R82'][key], rel_tol=1e-12, abs_tol=1e-6)
                assert math.isclose(v['control_mae'], a['metrics']['R81'][key], rel_tol=1e-12, abs_tol=1e-6)
    assert main['reference_predict_calls'] == main['spatial_predict_calls'] == 6
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
    decision = {'round': 82, 'candidate': 'R82', 'status': 'retain_research_no_new_CSV', 'completed_at_utc': now,
        'decision_scope': 'root_fixed_research_closeout; no new model selection rule', 'recommendation_changed': False,
        **roles, 'known_public_best_filename': previous['known_public_best_filename'], 'known_public_best_mae': previous['known_public_best_mae'],
        'private_mae': None, 'final_version_chosen': False, 'new_final_fits': 0, 'new_CSV': False, 'proposed_new_version': None,
        'uploaded': False, 'kaggle_selection_changed': False, 'GitHub_published': False, 'next_experiment_selected': False,
        'rationale': 'Latest and auxiliary overall/weighted/testmix regress versus R81, new projects regress in all periods, and all three unit-reference/spatial layers worsen. Overall versus v22 also worsens in all periods. No sufficient additional trial value.',
        'strongest_counterargument': 'Development overall/weighted/testmix and latest seen/Q1/below-10m improve versus R81; anchor is not uniformly ineffective. No all-period-wins gate, no claim that trend is optimal or Private superiority is proved.',
        'uncertainty': 'Latest and auxiliary eight project/month overall/weighted intervals are positive; development four cross zero. Reused windows, no multiplicity correction, no Private probability.',
        'historical_fits': 0, 'historical_fit_seconds': 0, 'main_seconds': main['elapsed_seconds'], 'diagnostics_seconds': review['diagnostics_seconds'], 'paired_seconds': review['paired_seconds'],
        'stage_metrics': {s: {'R82': main['stages'][s]['metrics']['R82'], 'differences': {k: main['stages'][s]['differences'][k] for k in ('R81', 'v22', 'v23')}} for s in STAGES},
        'source_sha256': {rel(Path(__file__)): hashes[rel(Path(__file__))]}, 'input_sha256': dict(hashes),
        'report_path': 'reports/round82_report.md', 'report_sha256_observed_at_recording': PINS['reports/round82_report.md'],
        'protected_file_sha256_verified_once_before_writes': protected_observed, 'selector_sha256_before': SELECTOR_SHA,
        'models_arrays_loaded_by_recorder': 0, 'new_scores': 0, 'new_fits_by_recorder': 0}
    decision_bytes = encode(decision)
    updates = {'updated_at_utc': now,
        'latest_completed_research': {'round': 82, 'candidate': 'R82', 'status': decision['status'], 'decision': decision['status'],
            'decision_record': rel(DECISION), 'decision_sha256': digest(decision_bytes), 'report': rel(REPORT), 'new_final_fits': 0, 'new_CSV': False, 'completed_at_utc': now},
        'pending_research': {'round': 82, 'status': 'completed_retained_no_new_CSV', 'new_final_fits': 0, 'new_CSV': False, 'next_experiment_selected': False},
        'evidence': list(dict.fromkeys(previous.get('evidence', []) + list(PINS) + [rel(DECISION), rel(REPORT), rel(OUT / 'record.json')]))}
    updated = copy.deepcopy(previous)
    updated.update(updates)
    updated['selection_history'] = copy.deepcopy(previous.get('selection_history', [])) + [{'action': 'R82_research_metadata_only_no_recommendation_change',
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
    write(OUT / 'record.json', {'round': 82, 'status': 'research_decision_recorded_recommendation_unchanged',
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
