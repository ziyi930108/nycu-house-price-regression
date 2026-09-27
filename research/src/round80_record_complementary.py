"""After verified v23 release only: register a complement, retain v22/v20, update short delivery text."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round80/complementary_record'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
SELECTOR_SHA = '397705756fb368f576f93e5efd548e3e0328966bd890b0adcf39e6fc6e511c36'
REPORT = ROOT / 'reports/round80_delivery.md'
DOCS = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
MARKER = '<!-- R80_V23_COMPLEMENTARY_DELIVERY -->'
CSV = 'outputs/submission_v23_trial.csv'
RELEASE = 'outputs/round80/final_export/release.json'
PLAN = 'reports/round80_final_plan.md'
DECISION = 'outputs/round80/decision.json'
PINS = {
    PLAN: 'a2edc656f2c9ef64cb4febba4865033374a8358f8dbf7d7b8c65a07715fa9fb0',
    DECISION: '49bc4cc0c2197d0dc2e923bd42fffbc249250056ce1bc8160cb57557efe37d21',
    'src/round80_release.py': '442eb47dba96443ebb16ea1dbc968da7d56141859fe2c0eb34448378cf73855b',
    'src/round80_final.py': '77ffd5d77f08154d696a518974d4fb40ef510b324f65bd8e451a7b722d368923',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
    'outputs/round80/results_review.json': '4c0de183e8737ce441333e4801dfe2e96edc4bb687447e3a734462ce5e985eca',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def at(name):
    p = (ROOT / name).resolve()
    assert p.is_relative_to(ROOT) and p != ROOT, name
    return p


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def put(path, data):
    with path.open('xb') as f:
        f.write(data)


def write(path, value):
    put(path, encoded(value))
    put(path.with_suffix('.sha256'), (sha(path) + '\n').encode('ascii'))


def prepare():
    # No directory, backup, selector or document writes before all checks/rendering.
    assert not OUT.exists() and not REPORT.exists() and sha(SELECTOR) == SELECTOR_SHA
    assert not SELECTOR.with_suffix('.sha256').exists(), 'Unexpected mutable selector receipt'
    inputs = {}

    def capture(name, expected=None):
        p = at(name)
        assert p.suffix in ('.json', '.md', '.py', '.sha256') and p.stat().st_size < 10_000_000
        h = sha(p)
        assert expected is None or h == expected, name
        assert rel(p) not in inputs or inputs[rel(p)] == h
        inputs[rel(p)] = h
        return p

    def small(name, expected=None):
        p = capture(name, expected or PINS.get(name))
        receipt = capture(rel(p.with_suffix('.sha256')))
        assert receipt.read_text(encoding='ascii').strip() == sha(p), name
        return json.loads(p.read_text(encoding='utf8'))

    for name, digest in PINS.items(): capture(name, digest)
    release = small(RELEASE)
    ex = small('outputs/round80/final_export/execution.json')
    proof = small('outputs/round80/final/trial_release.json', release['candidate_proof_sha256'])
    cx = small('outputs/round80/final/execution.json')
    research, result, review = small(DECISION), small('outputs/round80/results.json'), small('outputs/round80/results_review.json')
    for name in ('reports/round80_report.md', 'reports/round79_input_tail_audit.md'):
        capture(name)
    assert ex['exit_code'] == cx['exit_code'] == 0 and ex['failure'] is cx['failure'] is None
    assert ex['source_sha256'] == PINS['src/round80_release.py'] == release['source_sha256']['src/round80_release.py']
    assert cx['source_sha256'] == PINS['src/round80_final.py'] == proof['source_sha256']['src/round80_final.py']
    assert release['round'] == 80 and release['version'] == 23 and release['rows'] == 47073
    assert release['status'] == 'experimental_v23_trial_generated' and release['path'] == CSV
    assert release['candidate_proof_path'] == 'outputs/round80/final/trial_release.json'
    assert release['plan_path'] == PLAN and release['plan_sha256'] == proof['plan_sha256'] == PINS[PLAN]
    assert release['decision_path'] == DECISION and release['decision_sha256'] == proof['decision_sha256'] == PINS[DECISION]
    assert research['status'] == 'advance_one_fixed_complementary_candidate_not_yet_released'
    assert research['proposed_filename'] == CSV and research['new_fits_allowed'] == 0
    assert proof['status'] == 'final_vector_and_export_payload_ready_no_CSV' and proof['round'] == 80
    assert proof['fit_rows'] == 467221 and proof['query_rows'] == proof['rows'] == 47073
    assert proof['prediction_sha256'] == release['prediction_sha256'] and proof['prediction_path'] == release['prediction_path']
    assert proof['priority_retained'] == research['priority_retained'] == 'outputs/submission_v22_trial.csv'
    assert proof['backup_retained'] == research['backup_retained'] == 'outputs/submission_v20_trial.csv'
    for k in ('positive_int64_full_roundtrip', 'original_unrounded_prediction_comparison_exact', 'UTF8_BOM',
              'all_sample_IDs_and_order_exact', 'exact_headers', 'old_CSV_unchanged', 'prediction_changes_are_not_errors'):
        assert release[k] is True, k
    for k in ('selector_updated', 'registry_updated', 'uploaded', 'kaggle_selection_changed', 'final_version_chosen'):
        assert release[k] is False, k
    assert release['public_mae'] is release['private_mae'] is None and release['new_fits'] == 0
    assert release['composer_declared_separately_completed_tree_fits'] == release['model_fit_load_predict_calls_in_release'] == 0
    assert proof['new_fits_in_this_composer'] == proof['separately_completed_new_tree_fits'] == proof['models_loaded_in_this_composer'] == proof['tree_NN_model_predict_calls_in_this_composer'] == 0
    assert proof['reference_predict_calls'] == proof['spatial_predict_calls'] == 2
    assert proof['saved_prediction_vector_reload_exact'] and proof['old_v22_CSV_exact']
    assert proof['independent_native_model_reload_in_this_composer'] is False
    protected = research['protected_before']
    assert release['protected_before'] == release['protected_after'] == proof['protected_before'] == proof['protected_after'] == protected
    assert release['selector_sha256_before'] == release['selector_sha256_after'] == proof['selector_sha256_before'] == proof['selector_sha256_after'] == SELECTOR_SHA
    expected_csv = {n for n in protected if n.startswith('outputs/submission_v') and n.endswith('.csv')}
    assert len(expected_csv) == 23 and len(protected) == 24 and CSV not in protected and 'runs/versions.json' in protected
    actual_csv = {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
    assert actual_csv == expected_csv | {CSV} and len(actual_csv) == 24
    # Hash each CSV once; no array/model loading, CSV parsing or re-authoring.
    file_hashes = {n: sha(at(n)) for n in (*sorted(actual_csv), 'runs/versions.json')}
    assert all(file_hashes[n] == h for n, h in protected.items()) and file_hashes[CSV] == release['sha256']
    original_selector = SELECTOR.read_bytes()
    previous = json.loads(original_selector.decode('utf8'))
    assert hashlib.sha256(original_selector).hexdigest() == SELECTOR_SHA
    assert previous['private_priority'] == proof['priority_retained'] and previous['backup'] == proof['backup_retained']
    assert previous['sha256'] == file_hashes[previous['private_priority']] and previous['backup_sha256'] == file_hashes[previous['backup']]
    assert previous['known_public_best_filename'] == 'outputs/submission_v18_trial.csv' and previous['known_public_best_mae'] == 2388386.4613
    assert previous['private_mae'] is None and previous['final_version_chosen'] is False
    assert previous['earlier_alternative'] == 'outputs/submission_v21_trial.csv'
    assert 'complementary_candidate' not in previous and all(c['path'] != CSV for c in previous['candidates'])
    assert result['checked_stages'] == review['checked_stages'] == ['development', 'confirmation', 'auxiliary']
    assert result['new_fits'] == 0 and result['status'] == 'three_periods_completed_no_adoption_decision'
    assert review['status'] == 'completed_JSON_only_review' and review['adoption_decision'] is None
    now = datetime.now(timezone.utc).isoformat()
    candidate = {'round': 80, 'path': CSV, 'sha256': release['sha256'], 'rows': 47073,
        'role': 'complementary_candidate_not_priority', 'final_proof': release['candidate_proof_path'],
        'final_proof_sha256': release['candidate_proof_sha256'], 'CSV_release': RELEASE,
        'CSV_release_sha256': inputs[RELEASE], 'public_mae': None, 'private_mae': None, 'uploaded': False}
    updates = {'updated_at_utc': now, 'complementary_candidate': CSV, 'complementary_candidate_sha256': release['sha256'],
        'complementary_candidate_release': candidate, 'candidates': copy.deepcopy(previous['candidates']) + [candidate],
        'latest_completed_research': {'round': 80, 'candidate': 'R80', 'status': 'verified_complementary_local_trial_released',
            'decision': 'retain_v22_priority_v20_backup_add_v23_complement', 'decision_record': DECISION,
            'decision_sha256': PINS[DECISION], 'report': 'reports/round80_report.md', 'new_final_fits': 0, 'final_version_chosen': False},
        'pending_research': {'round': 80, 'status': 'completed_verified_complementary_trial_release', 'new_final_fits': 0},
        'evidence': list(dict.fromkeys(previous.get('evidence', []) + [PLAN, DECISION, RELEASE, release['candidate_proof_path'],
            'outputs/round80/results_review.json', 'reports/round80_report.md', 'reports/round80_delivery.md']))}
    updated = copy.deepcopy(previous)
    updated.update(updates)
    updated['selection_history'] = copy.deepcopy(previous.get('selection_history', [])) + [{
        'action': 'add_complement_without_priority_change', 'selector_backup': rel(OUT / 'originals/selector.json'),
        'selector_sha256': SELECTOR_SHA, 'replaced_fields': {k: previous[k] for k in updates if k in previous}}]
    assert all(updated[k] == v for k, v in previous.items() if k not in {*updates, 'selection_history'})
    originals = {n: at(n).read_bytes() for n in DOCS}
    assert all(MARKER.encode('utf8') not in b for b in originals.values())
    assert all(not at(n).with_suffix('.sha256').exists() and not Path(str(at(n)) + '.sha256').exists() for n in DOCS)
    report = render(release, proof, cx, ex, result)
    documents = {n: prefix(n).encode('utf8') + b for n, b in originals.items()}
    selector_bytes = encoded(updated)
    assert json.loads(selector_bytes) == updated and all(documents[n].endswith(originals[n]) for n in DOCS)
    assert all('v22' in b.decode('utf8') and 'v23' in b.decode('utf8') for b in documents.values())
    return dict(release=release, proof=proof, inputs=inputs, csv_sha256_at_start=file_hashes, report=report.encode('utf8'),
        original_selector=original_selector, selector_bytes=selector_bytes, originals=originals, documents=documents,
        candidate=candidate, completed_at_utc=now)


def render(release, proof, cx, ex, result):
    lines = ['# R80 / v23：互補候選交付完成', '',
        f"已驗證 [v23](../{CSV}) 共47,073筆，列為互補候選。**v22仍為有限本機優先、v20仍為既有備選，v21與所有歷史版本保留**。沒有代上傳、Kaggle勾選或最終版本決定；Private未知。已知Public最佳仍為v18（2,388,386.4613），v23 Public／Private均未知。", '',
        '只把R79單價分支base從 .75×native＋.25×old_unit 改為相同native的100%；native已是總價元，不再乘面積。各自reference／spatial後與原NN、log取median，沒有新fit。', '',
        '| 歷史窗口 | v23配方MAE | 相對v22整體差 | 相對v22加權差 |', '| --- | ---: | ---: | ---: |']
    for s, title in (('development','開發'),('confirmation','最新確認'),('auxiliary','較早輔助')):
        a=result['stages'][s]; d=a['differences']['v22']
        lines.append(f"| {title} | {a['metrics']['R80']['overall_mae']:,.2f} | {d['overall']['delta_TWD']:+,.2f} | {d['weighted']['delta_TWD']:+,.2f} |")
    lines += ['', '保留v22優先是跨期取捨：v23配方在最新與最舊年度較好，但開發整體／加權／testmix、新舊案及五價帶全面退步；最新舊案、Q1、低價及億元以上群也有代價。最新與輔助兩種配對區間為負，開發月份區間為正、建案區間跨零。不是單一年度投票或Private勝率。完整數字與12區間見[研究報告](round80_report.md)。', '',
        f"全量組合承接467,221筆fit狀態，保存47,073筆未取整預測；composer {cx['elapsed_seconds']:.3f}秒、release {ex['elapsed_seconds']:.3f}秒，0新fit、0樹／NN模型載入／predict，reference與spatial各2次（包含v22控制）。新向量保存重讀exact；R79原生模型reload是承接既有證據，不是本輪重新驗模型。",
        f"CSV已通過ArtifactTool全量typed cells／CSV回讀，以及獨立release的sample ID順序、中文兩欄、正int64、BOM、對未取整向量的一次rint比較。SHA256：`{release['sha256']}`。原23份CSV與registry保留；本登錄程式不寫任何CSV。",
        f"v23最大未取整預測為 {release['prediction_max_TWD']:,.2f} 元；相對v22共 {proof['changed_unrounded_rows_vs_v22']:,} 列變動，平均絕對預測差 {proof['mean_absolute_prediction_change_vs_v22_TWD']:,.2f} 元。這些是預測差，不是真實誤差。", '',
        '先前[原始輸入尾端核對](round79_input_tail_audit.md)顯示巨額交易的面積／車位值確實來自提供檔案，但不能證明成交價或v23高尾外推正確。本次未另讀原始工作簿、未設價帽、未逐ID改價。',
        '歷史窗口已重複用於選型，沒有新盲測、未校正多輪挑選；Private仍未知。程式、模型、依賴、結果與失敗紀錄完整保留，GitHub待最終版本確定再整理，尚未發布。', '',
        '[固定final計畫](round80_final_plan.md) · [向量證據](../outputs/round80/final/trial_release.json) · [CSV發布驗證](../outputs/round80/final_export/release.json) · [本機補充紀錄](../outputs/round80/complementary_record/record.json)', '']
    return '\n'.join(lines)


def prefix(name):
    p = '../' if name.startswith('reports/') else ''
    delivery = 'round80_delivery.md' if p else 'reports/round80_delivery.md'
    return f'''{MARKER}
# 最新補充：v23已驗證，v22優先／v20備選維持

[v23]({p}{CSV})已完成47,073筆CSV驗證，只列為互補候選；[v22]({p}outputs/submission_v22_trial.csv)仍為有限本機優先，[v20]({p}outputs/submission_v20_trial.csv)仍為備選，v21及舊版本保留。本輪0新fit：最新／較早輔助MAE改善，但開發及部分分群退步，不能宣稱全面勝出。Private未知、最終版本未定；已回報Public最佳仍是v18（2,388,386.4613），v23尚無回報。

[交付與取捨]({delivery})。沒有代上傳或GitHub發布；程式與紀錄保留，待定版再整理。以下原文逐位保留作歷史，以本節為最新補充。

---

'''


def commit(p):
    assert sha(SELECTOR) == SELECTOR_SHA and all(at(n).read_bytes() == b for n, b in p['originals'].items())
    assert all(sha(at(n)) == h for n, h in p['inputs'].items())
    OUT.mkdir(parents=True, exist_ok=False)
    originals, staged = OUT / 'originals', OUT / 'staged'
    originals.mkdir(); staged.mkdir()
    put(originals / 'selector.json', p['original_selector'])
    put(originals / 'selector.sha256', (SELECTOR_SHA + '\n').encode('ascii'))
    for n, b in p['originals'].items():
        target=originals / n; target.parent.mkdir(parents=True,exist_ok=True); put(target,b)
    put(staged / 'selector.json', p['selector_bytes'])
    put(staged / 'delivery.md', p['report'])
    for n,b in p['documents'].items():
        target=staged / n; target.parent.mkdir(parents=True,exist_ok=True); put(target,b)
    put(REPORT,p['report'])
    for n in DOCS:
        assert at(n).read_bytes() == p['originals'][n]
        os.replace(staged / n,at(n))
        assert at(n).read_bytes() == p['documents'][n]
    assert sha(SELECTOR) == SELECTOR_SHA
    os.replace(staged / 'selector.json',SELECTOR)
    assert SELECTOR.read_bytes() == p['selector_bytes']
    write(OUT / 'record.json', {'round':80,'status':'verified_complement_registered_priority_unchanged',
        'candidate':p['candidate'],'private_priority':'outputs/submission_v22_trial.csv','backup':'outputs/submission_v20_trial.csv',
        'selector_sha256_before':SELECTOR_SHA,'selector_sha256_after':sha(SELECTOR),
        'selector_original_backup':rel(originals / 'selector.json'), 'unknown_selector_fields_preserved':True,
        'original_document_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in p['originals'].items()},
        'updated_document_sha256':{n:sha(at(n)) for n in DOCS},'original_document_body_bytes_preserved':True,
        'report_path':rel(REPORT),'report_sha256':sha(REPORT),'source_sha256':sha(Path(__file__)),
        'small_evidence_sha256':p['inputs'],'CSV_and_registry_sha256_verified_once_before_writes':p['csv_sha256_at_start'],
        'new_CSV':False,'CSV_files_written':False,'new_fits':0,'array_or_model_loads':0,'new_scores':0,
        'public_mae':None,'private_mae':None,'uploaded':False,'kaggle_selection_changed':False,
        'final_version_chosen':False,'GitHub_published':False,'completed_at_utc':p['completed_at_utc']})
    print(json.dumps({'status':'verified_complement_registered_priority_unchanged','selector_sha256':sha(SELECTOR),
        'report_sha256':sha(REPORT),'record_sha256':sha(OUT / 'record.json')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True); parser.parse_args()
    started,tick=datetime.now(timezone.utc).isoformat(),time.monotonic()
    code,failure=0,None
    try:
        prepared=prepare()
        commit(prepared)
    except Exception:
        code,failure=1,traceback.format_exc(); print(failure,flush=True)
    if OUT.exists():
        write(OUT / 'execution.json',{'command':sys.argv,'exit_code':code,'failure':failure,'started_at_utc':started,
            'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick,
            'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
