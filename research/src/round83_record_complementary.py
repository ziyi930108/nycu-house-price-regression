"""After verified v24 release only: register a complement, retain v22/v20, update short delivery text."""
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
OUT = ROOT / 'outputs/round83/complementary_record'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
SELECTOR_SHA = 'c993a861fa6766fb5c9da8c156521bb28886a459587653ee8734b2927b9e87a8'
REPORT = ROOT / 'reports/round83_delivery.md'
DOCS = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
MARKER = '<!-- R83_V24_COMPLEMENTARY_DELIVERY -->'
CSV = 'outputs/submission_v24_trial.csv'
RELEASE = 'outputs/round83/final_export/release.json'
PLAN = 'reports/round83_final_plan.md'
DECISION = 'outputs/round83/decision.json'
MODEL_RESULT = 'outputs/round83/final_model/results.json'
OLD_COMPLEMENT = 'outputs/submission_v23_trial.csv'
PINS = {
    PLAN: '792631bfde4e83e89134246e55a91fe05ce0196c2bc97c7b4ed360f48a10b4c6',
    DECISION: '759d2ec0cfafe3c05575ecc0536e4287b5e006286564b922cea60cb5f34c7b28',
    'src/round83_release.py': '7588608eb98d270959945e99d13de091985634889e70d07db38165c1abc568b2',
    'src/round83_final.py': '68a007f14de33c71f2a10cc1c256d9138a0f2d631a4b7120481d7aa47ea4391e',
    'src/round83_final_unit.py': '01eb89c0727f34e9c3e6a6e0e80e55ef7acd91ebf0902dd391c39f77ccc09494',
    'outputs/round83/results.json': '8db07138c15ef2725d848c8b0ca65a6095bcc183b8550611377b3e165c59583b',
    'outputs/round83/results_review.json': '651ab12658e46cbb483a0725e6dceed28ce9dc2b33dc8d131f8a89da02d5ce2a',
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
    ex = small('outputs/round83/final_export/execution.json')
    proof = small('outputs/round83/final/trial_release.json', release['candidate_proof_sha256'])
    cx = small('outputs/round83/final/execution.json')
    research, result, review = small(DECISION), small('outputs/round83/results.json'), small('outputs/round83/results_review.json')
    capture('reports/round83_report.md')
    model = small(MODEL_RESULT, proof['model_result_sha256'])
    mx = small('outputs/round83/final_model/execution.json')
    ax = small('outputs/round83/csv_authoring/execution.json')
    vp = capture(release['authoring_validation_path'], release['authoring_validation_sha256'])
    validation = json.loads(vp.read_text(encoding='utf8'))
    assert mx['exit_code'] == ax['exit_code'] == 0 and mx['failure'] is None
    assert mx['source_sha256'] == PINS['src/round83_final_unit.py'] == model['source_sha256']['src/round83_final_unit.py']
    assert model['round'] == 83 and model['status'] == 'final_unit_model_ready_no_CSV'
    assert model['fit_rows'] == 467221 and model['query_rows'] == model['reload_rows'] == 47073
    assert model['new_fits'] == 1 and model['new_NN_fits'] == 0 and not model['new_CSV']
    assert model['seed'] == model['actual_random_seed'] == model['parameters']['random_seed'] == 137
    assert model['changed_parameter_names'] == ['random_seed']
    assert model['full_query_raw_TWD_reload_exact'] and model['original_R79_legacy_feature_state_exact']
    assert model['query_labels_used'] is model['test_labels_loaded'] is False
    assert model['text_plan_sha256'] == mx['text_plan_sha256'] == PINS[PLAN]
    assert model['decision_sha256'] == mx['decision_sha256'] == PINS[DECISION]
    assert proof['model_result_path'] == MODEL_RESULT
    assert proof['native_prediction_path'] == model['native_prediction_path']
    assert proof['native_prediction_sha256'] == model['native_prediction_sha256']
    assert validation['status'] == 'authored_not_published' and validation['round'] == 83 and validation['version'] == 24
    assert validation['rows'] == 47073 and validation['rounding_performed_by_this_writer'] is False
    for k in ('all_typed_cells_exact','all_saved_CSV_cells_exact','utf8_BOM','ID_order_preserved','all_prices_safe_positive_integers'):
        assert validation[k] is True, k
    assert ex['exit_code'] == cx['exit_code'] == 0 and ex['failure'] is cx['failure'] is None
    assert ex['source_sha256'] == PINS['src/round83_release.py'] == release['source_sha256']['src/round83_release.py']
    assert cx['source_sha256'] == PINS['src/round83_final.py'] == proof['source_sha256']['src/round83_final.py']
    assert release['round'] == 83 and release['version'] == 24 and release['rows'] == 47073
    assert release['status'] == 'experimental_v24_trial_generated' and release['path'] == CSV
    assert release['candidate_proof_path'] == 'outputs/round83/final/trial_release.json'
    assert release['plan_path'] == PLAN and release['plan_sha256'] == proof['plan_sha256'] == PINS[PLAN]
    assert release['decision_path'] == DECISION and release['decision_sha256'] == proof['decision_sha256'] == PINS[DECISION]
    assert research['status'] == 'advance_one_fixed_complementary_candidate_not_yet_released'
    assert research['proposed_filename'] == CSV and research['new_fits_allowed'] == 1 and research['final_fit_seed'] == 137
    assert research['intended_complementary_replacement_of'] == OLD_COMPLEMENT
    assert proof['status'] == 'final_vector_and_export_payload_ready_no_CSV' and proof['round'] == 83
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
    assert release['composer_declared_separately_completed_tree_fits'] == 1 and release['model_fit_load_predict_calls_in_release'] == 0
    assert proof['new_fits_in_this_composer'] == proof['models_loaded_in_this_composer'] == proof['tree_NN_model_predict_calls_in_this_composer'] == 0
    assert proof['separately_completed_new_tree_fits'] == proof['inherited_R79_tree_fits'] == 1 and proof['new_NN_fits'] == 0
    assert proof['reference_predict_calls'] == proof['spatial_predict_calls'] == 2
    assert proof['saved_prediction_vector_reload_exact'] and proof['old_v23_CSV_exact']
    assert proof['native_model_full_query_raw_TWD_reload_exact'] and proof['seed42_full_query_raw_TWD_reload_exact']
    assert proof['independent_native_model_reload_in_this_composer'] is False
    protected = research['protected_before']
    assert release['protected_before'] == release['protected_after'] == proof['protected_before'] == proof['protected_after'] == model['protected_before'] == model['protected_after'] == protected
    assert release['selector_sha256_before'] == release['selector_sha256_after'] == proof['selector_sha256_before'] == proof['selector_sha256_after'] == model['selector_sha256_before'] == model['selector_sha256_after'] == SELECTOR_SHA
    expected_csv = {n for n in protected if n.startswith('outputs/submission_v') and n.endswith('.csv')}
    assert len(expected_csv) == 24 and len(protected) == 25 and CSV not in protected and 'runs/versions.json' in protected
    actual_csv = {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
    assert actual_csv == expected_csv | {CSV} and len(actual_csv) == 25
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
    assert previous['complementary_candidate'] == OLD_COMPLEMENT
    assert previous['complementary_candidate_sha256'] == file_hashes[OLD_COMPLEMENT]
    assert all(c['path'] != CSV for c in previous['candidates'])
    prior_complement = copy.deepcopy(previous['complementary_candidate_release'])
    assert prior_complement['path'] == OLD_COMPLEMENT and prior_complement['sha256'] == file_hashes[OLD_COMPLEMENT]
    prior_candidates = [c for c in previous['candidates'] if c['path'] == OLD_COMPLEMENT]
    assert len(prior_candidates) == 1 and prior_candidates[0] == prior_complement
    candidates = copy.deepcopy(previous['candidates'])
    for c in candidates:
        if c['path'] == OLD_COMPLEMENT:
            c['role'] = 'earlier_complementary_candidate_replaced_by_v24'
            assert {k:v for k,v in c.items() if k != 'role'} == {k:v for k,v in prior_complement.items() if k != 'role'}
    assert result['checked_stages'] == review['checked_stages'] == ['development', 'confirmation', 'auxiliary']
    assert result['new_fits'] == 3 and result['status'] == 'three_periods_completed_no_adoption_decision'
    assert review['status'] == 'completed_JSON_only_review' and review['adoption_decision'] is None
    assert review['round'] == 83 and review['new_fits_in_reporter'] == 0
    assert all(e['exit_code'] == 0 and e.get('failure') is None for e in review['executions'].values())
    for stage in ('development','confirmation','auxiliary'):
        assert result['stages'][stage]['metrics'] == review['stages'][stage]['metrics']
        for family in ('project','month'):
            for metric in ('overall','weighted'):
                ci = review['stages'][stage]['paired'][family][metric]
                assert ci['valid_replicates'] == 2000 and ci['invalid_replicates'] == 0
                if stage == 'confirmation': assert ci['percentile_2_5'] < 0 < ci['percentile_97_5']
                else: assert ci['percentile_97_5'] < 0
    now = datetime.now(timezone.utc).isoformat()
    candidate = {'round': 83, 'path': CSV, 'sha256': release['sha256'], 'rows': 47073,
        'role': 'complementary_candidate_not_priority', 'final_proof': release['candidate_proof_path'],
        'final_proof_sha256': release['candidate_proof_sha256'], 'CSV_release': RELEASE,
        'CSV_release_sha256': inputs[RELEASE], 'public_mae': None, 'private_mae': None, 'uploaded': False}
    updates = {'updated_at_utc': now, 'complementary_candidate': CSV, 'complementary_candidate_sha256': release['sha256'],
        'complementary_candidate_release': candidate, 'candidates': candidates + [candidate],
        'earlier_complementary_candidate': OLD_COMPLEMENT,
        'earlier_complementary_candidate_sha256': file_hashes[OLD_COMPLEMENT],
        'earlier_complementary_candidate_release': prior_complement,
        'earlier_complementary_candidate_role': 'v23 remains available with original evidence; v24 now fills the complementary role. Public and Private remain unknown.',
        'latest_completed_research': {'round': 83, 'candidate': 'R83', 'status': 'verified_complementary_local_trial_released',
            'decision': 'retain_v22_priority_v20_backup_replace_v23_complement_with_verified_v24', 'decision_record': DECISION,
            'decision_sha256': PINS[DECISION], 'report': 'reports/round83_report.md', 'new_final_fits': 1, 'final_version_chosen': False},
        'pending_research': {'round': 83, 'status': 'completed_verified_complementary_trial_release', 'new_final_fits': 1, 'next_experiment_selected': False},
        'evidence': list(dict.fromkeys(previous.get('evidence', []) + [PLAN, DECISION, RELEASE, release['candidate_proof_path'],
            'outputs/round83/results_review.json', 'reports/round83_report.md', 'reports/round83_delivery.md']))}
    updated = copy.deepcopy(previous)
    updated.update(updates)
    updated['selection_history'] = copy.deepcopy(previous.get('selection_history', [])) + [{
        'action': 'replace_complement_v23_with_v24_without_priority_change', 'selector_backup': rel(OUT / 'originals/selector.json'),
        'selector_sha256': SELECTOR_SHA, 'replaced_fields': {k: previous[k] for k in updates if k in previous}}]
    assert all(updated[k] == v for k, v in previous.items() if k not in {*updates, 'selection_history'})
    originals = {n: at(n).read_bytes() for n in DOCS}
    assert all(MARKER.encode('utf8') not in b for b in originals.values())
    assert all(not at(n).with_suffix('.sha256').exists() and not Path(str(at(n)) + '.sha256').exists() for n in DOCS)
    report = render(release, proof, cx, ex, result, review, model, mx)
    documents = {n: prefix(n).encode('utf8') + b for n, b in originals.items()}
    selector_bytes = encoded(updated)
    assert json.loads(selector_bytes) == updated and all(documents[n].endswith(originals[n]) for n in DOCS)
    assert all('v22' in b.decode('utf8') and 'v24' in b.decode('utf8') and 'v23' in b.decode('utf8') for b in documents.values())
    return dict(release=release, proof=proof, inputs=inputs, csv_sha256_at_start=file_hashes, report=report.encode('utf8'),
        original_selector=original_selector, selector_bytes=selector_bytes, originals=originals, documents=documents,
        candidate=candidate, completed_at_utc=now)


def render(release, proof, cx, ex, result, review, model, mx):
    lines = ['# R83 / v24：固定兩種子互補候選交付', '',
        f"[v24](../{CSV})已完成47,073筆發布驗證，接替v23的互補位置。**v22維持有限本機優先，v20維持備選**；v23檔案、原候選與發布證據完整保留，v21仍為較早替代方案。Public／Private均未知，沒有代上傳或最終版本決定。已知Public最佳仍為v18（2,388,386.4613）。", '',
        '固定方法：每支先依原單價逆轉換成非負TWD，.5×保存seed42＋.5×新seed137；均值經原trend reference與R39 spatial，再與原R71 NN、R46 log取median。不再乘面積，不挑seed或搜尋比例。', '',
        '以下為重用歷史窗口的候選減對照，負值較好；不是Private成績。', '',
        '| 期別 | v23 MAE | v22 MAE | R83 MAE | 差vs v23 | 差vs v22 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    stages = (('development','開發'),('confirmation','最新確認'),('auxiliary','最舊輔助'))
    for stage, title in stages:
        a=result['stages'][stage]; m=a['metrics']; d=a['differences']
        lines.append(f"| {title} | {m['v23']['overall_mae']:,.2f} | {m['v22']['overall_mae']:,.2f} | {m['R83']['overall_mae']:,.2f} | {d['v23']['overall']['delta_TWD']:+,.2f} | {d['v22']['overall']['delta_TWD']:+,.2f} |")
    lines += ['', '| 期別 | v23 加權MAE | v22 加權MAE | R83 加權MAE | 差vs v23 | 差vs v22 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for stage, title in stages:
        a=result['stages'][stage]; m=a['metrics']; d=a['differences']
        lines.append(f"| {title} | {m['v23']['weighted_mae']:,.2f} | {m['v22']['weighted_mae']:,.2f} | {m['R83']['weighted_mae']:,.2f} | {d['v23']['weighted']['delta_TWD']:+,.2f} | {d['v22']['weighted']['delta_TWD']:+,.2f} |")
    dev, latest = review['stages']['development'], review['stages']['confirmation']
    tail = next(x for x in latest['price_bands'] if x['band']=='>=100m')
    lines += ['',
        '相對v23，開發與最舊年度的建案／月份整體及加權共8個配對區間全負；最新年度4個區間跨零，整體近持平而加權、新案、Q2／Q4及testmix略退。支持固定平均作為互補候選，不能宣稱所有分群較好。',
        f"v22仍保留優先的取捨：開發testmix相對v22為 {dev['test_composition_differences']['v22']['delta_TWD']:+,.2f} 元；最新≥1億的{tail['rows']}列群MAE相對v22為 {tail['differences']['v22']['delta_TWD']:+,.2f} 元／列，對全期差貢獻 {tail['contribution_to_stage_overall_delta_TWD']['v22']:+,.2f} 元。這些是真價分組的歷史誤差診斷，不是推論路由、因果結論或逐ID規則。",
        '最強升優先反論也保留：R83對v22三期整體與固定加權點估計全較好，最新新案與testmix、最舊年度有實質優勢。暫不升優先不等於證明v22較佳；這是有限證據下的風險取捨，沒有新設三期必勝門檻。',
        'testmix只在有歷史支持的城市×新舊案範圍內重加權；12個區間皆以v23為主對照，不能移用成勝過v22的區間。所有窗口反覆參與研究，bootstrap未校正累积選型偏差，也不是Private勝率。', '',
        f"歷史新增3個seed137 fit合計 {review['total_fit_seconds']:.3f}秒；final另新增1個seed137 fit {model['fit_seconds']:.3f}秒，完整程序 {mx['elapsed_seconds']:.3f}秒。保存模型與wrapper後，全部47,073列raw unit及TWD經一次獨立重載逐位相同。seed42單價模型、NN及log重用，沒有重新訓練。",
        f"final composer {cx['elapsed_seconds']:.3f}秒，0新fit／0樹NN模型載入或predict，reference與spatial各2次（包含v23控制）；新向量保存重讀exact。seed42重載為承接R79證據，新137重載屬前一final-unit程序，不能算成本composer的模型重播。release {ex['elapsed_seconds']:.3f}秒，本登錄程式不讀模型或array、不重算分數。",
        f"CSV已通過ArtifactTool全量typed cells與保存CSV回讀，以及獨立release的sample IDs／順序、兩欄、正int64、UTF-8 BOM與原未取整向量的一次rint核對。SHA256：`{release['sha256']}`。原24份CSV與registry保持，本登錄只更新本機角色與說明，完整備份原selector及3份文件原bytes。",
        f"相對v23有 {proof['changed_unrounded_rows_vs_v23']:,} 列未取整預測改變，平均絕對預測差 {proof['mean_absolute_prediction_change_vs_v23_TWD']:,.2f} 元；v24最大預測 {proof['prediction_max_TWD']:,.2f} 元。這些是預測分歧，不是已知誤差，不能證明巨額物件成交價正確。", '',
        '兩個seed不足以代表完整訓練隨機性，GPU非完全位元決定；均值無法保證消除共同偏差或高尾外推。Private仍未知、最終版本未定；沒有Kaggle上傳／勾選或GitHub發布。程式、預處理、模型、報告、執行與失敗紀錄保留，GitHub待定版後整理。', '',
        '[研究報告](round83_report.md) · [固定final計畫](round83_final_plan.md) · [模型結果](../outputs/round83/final_model/results.json) · [向量證據](../outputs/round83/final/trial_release.json) · [CSV發布](../outputs/round83/final_export/release.json) · [登錄紀錄](../outputs/round83/complementary_record/record.json)', '']
    return '\n'.join(lines)


def prefix(name):
    p = '../' if name.startswith('reports/') else ''
    delivery = 'round83_delivery.md' if p else 'reports/round83_delivery.md'
    return f'''{MARKER}
# 最新補充：v24接替互補位置，v22優先／v20備選維持

[v24]({p}{CSV})已完成47,073筆CSV驗證，固定平均單價seed42與137，只接替v23的互補位置；[v22]({p}outputs/submission_v22_trial.csv)仍為有限本機優先，[v20]({p}outputs/submission_v20_trial.csv)仍為備選，v23與v21及全部歷史檔案保留。歷史新增3個seed137模型、final另1個；既有42、NN、log重用。

相對v23，開發／最舊年度8個配對區間全負，最新4個跨零且新案、加權、testmix略退；相對v22三期整體及加權點估計較好，但開發testmix與最新億元以上群有代價。[完整交付與最強反論]({delivery})。Private未知、最終版本未定；已回報Public最佳仍v18（2,388,386.4613），v24沒有Public／Private回報。沒有代上傳或GitHub發布；原本文逐位保留作歷史，以本節為最新補充。

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
    write(OUT / 'record.json', {'round':83,'status':'verified_complement_registered_priority_unchanged',
        'candidate':p['candidate'],'earlier_complementary_candidate':OLD_COMPLEMENT,'earlier_alternative':'outputs/submission_v21_trial.csv','private_priority':'outputs/submission_v22_trial.csv','backup':'outputs/submission_v20_trial.csv',
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
    assert not OUT.exists() and not REPORT.exists(), 'Prior record or delivery exists'
    started,tick=datetime.now(timezone.utc).isoformat(),time.monotonic()
    code,failure=0,None
    try:
        prepared=prepare()
        commit(prepared)
    except Exception:
        code,failure=1,traceback.format_exc(); print(failure,flush=True)
    if not OUT.exists(): OUT.mkdir(parents=True, exist_ok=False)
    if OUT.exists():
        write(OUT / 'execution.json',{'command':sys.argv,'exit_code':code,'failure':failure,'started_at_utc':started,
            'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick,
            'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    raise SystemExit(code)
