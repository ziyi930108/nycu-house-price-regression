"""Small JSON/text delivery writer; run only after v22 release and selector update."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round79/delivery'
REPORT = ROOT / 'reports/round79_delivery.md'
SELECTOR = ROOT / 'outputs/current_private_candidate.json'
RELEASE = ROOT / 'outputs/round79/final_export/release.json'
SELECTION = ROOT / 'outputs/round79/selection/decision.json'
DOCS = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
MARKER = '<!-- R79_V22_COMPLETED_DELIVERY -->'
READ = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def at(name):
    path = (ROOT / name.replace('\\', '/')).resolve()
    assert path.is_relative_to(ROOT) and path != ROOT, name
    return path


def read_small(path, receipt=True):
    path = Path(path)
    assert path.suffix in ('.json', '.md') and path.stat().st_size < 30_000_000
    digest = sha(path)
    if receipt:
        rp = path.with_suffix('.sha256')
        assert rp.read_text(encoding='ascii').strip() == digest, str(path)
        READ[rel(rp)] = sha(rp)
    READ[rel(path)] = digest
    return json.loads(path.read_text(encoding='utf8'))


def bound(name, digest):
    path = at(name)
    assert sha(path) == digest, name
    return read_small(path)


def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(path) + '\n')


def verified_inputs():
    release = read_small(RELEASE)
    selection = read_small(SELECTION)
    assert read_small(RELEASE.parent / 'execution.json')['exit_code'] == 0
    assert read_small(SELECTION.parent / 'execution.json')['exit_code'] == 0
    assert release['status'] == 'experimental_v22_trial_generated' and release['rows'] == 47073
    assert release['path'] == 'outputs/submission_v22_trial.csv' and at(release['path']).is_file()
    for key in ('positive_int64_full_roundtrip', 'original_unrounded_prediction_comparison_exact',
                'UTF8_BOM', 'all_sample_IDs_and_order_exact', 'exact_headers', 'old_CSV_unchanged'):
        assert release[key] is True, key
    assert len(release['protected_before']) == 23
    assert release['protected_before'] == release['protected_after']
    assert release['selector_sha256_before'] == release['selector_sha256_after']
    assert release['uploaded'] is False and release['private_mae'] is None
    assert selection['status'] == 'local_recommendation_updated'
    assert selection['release_path'] == rel(RELEASE) and selection['release_sha256'] == sha(RELEASE)
    assert selection['previous_selector_sha256'] == release['selector_sha256_after']
    assert selection['priority'] == release['path'] and selection['priority_sha256'] == release['sha256']
    assert selection['backup'] == 'outputs/submission_v20_trial.csv'
    assert selection['earlier_alternative'] == 'outputs/submission_v21_trial.csv'
    for key in ('new_CSV', 'final_version_chosen', 'uploaded'):
        assert selection[key] is False, key
    assert selection['private_mae'] is None
    selector = read_small(SELECTOR, receipt=False)
    assert sha(SELECTOR) == selection['new_selector_sha256']
    assert selector['private_priority'] == selection['priority'] and selector['sha256'] == release['sha256']
    assert selector['backup'] == selection['backup'] and selector['backup_sha256'] == selection['backup_sha256']
    assert selector['known_public_best_filename'] == 'outputs/submission_v18_trial.csv'
    assert selector['known_public_best_mae'] == 2388386.4613
    assert selector['private_mae'] is None and selector['final_version_chosen'] is False
    proof = bound(release['candidate_proof_path'], release['candidate_proof_sha256'])
    assert proof['status'] == 'final_vector_and_export_payload_ready_no_CSV'
    assert proof['rows'] == proof['query_rows'] == 47073 and proof['fit_rows'] == 467221
    assert proof['old_v20_CSV_exact'] and proof['saved_prediction_vector_reload_exact']
    assert proof['native_model_full_query_raw_TWD_reload_exact']
    assert proof['prediction_sha256'] == release['prediction_sha256']
    assert proof['protected_before'] == proof['protected_after'] == release['protected_before']
    model = bound(proof['model_result_path'], proof['model_result_sha256'])
    assert model['status'] == 'final_unit_model_ready_no_CSV'
    assert model['new_fits'] == 1 and model['new_NN_fits'] == 0
    assert model['fit_rows'] == 467221 and model['query_rows'] == 47073
    assert model['full_query_raw_TWD_reload_exact'] is True
    model_exec = read_small(at(proof['model_result_path']).parent / 'execution.json')
    compose_exec = read_small(at(release['candidate_proof_path']).parent / 'execution.json')
    assert model_exec['exit_code'] == compose_exec['exit_code'] == 0
    decision = bound(release['decision_path'], release['decision_sha256'])
    historical = bound(proof['historical_results'], proof['historical_results_sha256'])
    assert set(historical['stages']) == {'development', 'confirmation', 'auxiliary'}
    assert decision['input_sha256'][proof['historical_results']] == proof['historical_results_sha256']
    for name in ('reports/round79_diagnostics.json', 'outputs/round79/paired_vs_v20/results.json'):
        bound(name, decision['input_sha256'][name])
    return release, selection, selector, proof, model, model_exec, compose_exec, historical


def render_report(release, proof, model, model_exec, compose_exec, history):
    rows = []
    for stage, title in [('development', '開發'), ('confirmation', '最新確認'), ('auxiliary', '較早輔助')]:
        rec = history['stages'][stage]
        metric, difference = rec['metrics']['R79'], rec['differences']['v20']
        rows.append(f"| {title} | {metric['rows']:,} | {metric['overall_mae']:,.2f} | {difference['overall']['delta_TWD']:+,.2f} | {difference['overall']['relative_change_pct']:+.4f}% | {metric['weighted_mae']:,.2f} |")
    dev = history['stages']['development']['differences']['v20']
    return f'''# R79 / v22 交付完成

本機有限優先已更新為 [v22](../{release['path']})，備選為 [v20](../outputs/submission_v20_trial.csv)，[v21](../outputs/submission_v21_trial.csv) 保留為較早替代方案。這是本機建議；沒有代上傳或更動 Kaggle 勾選。Private 分數未知，最終版本尚未確定。已知 Public 最佳仍為 v18（2,388,386.4613）。

## 固定歷史比較

R79 僅在單價 CatBoost 原面積訓練權重上乘保存的 365 日日期權重，不重新正規化；原 NN、log、reference/spatial 與外層 median 保持。以下為保存結果，差值 R79−v20，負值表示 MAE 較低，單位新臺幣。

| 年度窗口 | 筆數 | R79 MAE | 與 v20 差值 | 相對變化 | R79 加權 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(rows) + f'''

最新確認窗口新舊建案及四季皆改善。開發窗口仍有代價：已見建案 MAE {dev['strata']['seen']['delta_TWD']:+,.2f} 元、Q4 {dev['forecast_quarters']['Q4']['delta_TWD']:+,.2f} 元；低於 1,000 萬及 5,000 萬至 1 億價帶在開發、最新確認都退步。較早輔助的 1 億以上群有 171 筆，MAE 約增加 363 萬元，不能隱去高價尾部風險。

成對診斷的開發窗口區間跨零；最新確認與較早輔助的兩種抽法、整體及加權區間均為負。這些是反覆使用歷史窗口後的條件診斷，沒有修正多輪選型，不能當成獨立新 holdout 或 Private 改善機率。詳見 [完整回測](round79_report.md)、[固定分組診斷](round79_diagnostics.md) 與 [成對結果](../outputs/round79/paired_vs_v20/results.json)。

## 最終產物與保留

最終單價模型使用 467,221 筆 fit、47,073 筆 query，僅新訓練 1 個 CatBoost、0 個 NN。實際 fit {model['fit_seconds']:.3f} 秒，模型流程 {model_exec['elapsed_seconds']:.3f} 秒，保存分支組合流程 {compose_exec['elapsed_seconds']:.3f} 秒。模型保存／重載的全部 query raw 與 TWD 逐列相同；composer 重現原 v20 控制，再輸出新的未取整向量。

已發布 CSV 共 47,073 筆，兩欄、UTF-8 BOM、sample ID 列序、正 int64 及對原浮點一次取整的全量對照均通過。CSV SHA256：`{release['sha256']}`。原 22 份 CSV 與版本 registry 維持不變；推薦 selector 由獨立步驟更新，本文只讀其成功收據。

全體最大未取整預測為 {proof['prediction_max_TWD']:,.2f} 元（約 {proof['prediction_max_TWD']/1e8:.3f} 億），v20 為 13,194,531,292.99 元（約 131.945 億）；test 真價未知。相對 v20 有 {proof['changed_unrounded_rows_vs_v20']:,} 筆未取整預測改變，平均絕對預測變動 {proof['mean_absolute_prediction_change_vs_v20_TWD']:,.2f} 元，這不是 MAE 或真價誤差。保留 [R78 純 X 尾部限制](round78_final_tail_inspection.md)，未做逐 ID 封頂或改價。

來源與模型仍保留：[R79 helper](../src/round79_recency_unit.py)、[三期 driver](../src/round79_experiment.py)、[final 訓練入口](../src/round79_final_unit.py)、[composer](../src/round79_final.py)、[CBM](../outputs/round79/final_model/model/model.cbm)、[feature state](../outputs/round79/final_model/model/feature_state.json)、[模型證據](../outputs/round79/final_model/results.json)、[向量證據](../outputs/round79/final/trial_release.json)、[發布驗證](../outputs/round79/final_export/release.json)、[選型收據](../outputs/round79/selection/decision.json)。本步沒有 fit、模型推論、CSV 改写或 GitHub 發布。
'''


def prefix(name, release):
    p = '../' if name.startswith('reports/') else ''
    report = 'round79_delivery.md' if p else 'reports/round79_delivery.md'
    return f'''{MARKER}
# 最新狀態：R79 已完成，v22 本機有限優先

目前優先比較 [{p}submission_v22_trial.csv]({p}{release['path']})，備選 [v20]({p}outputs/submission_v20_trial.csv)，v21 保留。最新確認 MAE 為 **1,383,228 元**；相對 v20，開發／最新確認／較早輔助整體 MAE 分別改善約 **0.62%／1.94%／2.65%**。完整 47,073 筆 CSV 格式、ID 列序、一次取整對照及模型保存重載已通過，原版本檔案仍保留。

這是有限本機建議：開發已見建案與 Q4、部分低價／高價帶仍退步，開發成對區間跨零，且歷史窗口已反覆用於選型。最大 test 預測約 **138.716 億元**（v20 約 131.945 億元），真價未知；預測差不能解讀為誤差改善。Private 未知、最終版本未定，沒有代上傳或變更 Kaggle 勾選；已知 Public 最佳仍為 v18（2,388,386.4613）。

[本次交付、模型保留與限制]({report})。GitHub 尚未發布。以下原文完整保留作沿革；其中「目前優先」及「R79 尚未執行」等語句代表當時狀態，以本節為最新。

---

'''.encode('utf8')


def run():
    values = verified_inputs()
    release, selection, selector, proof, model, model_exec, compose_exec, history = values
    originals = {name: at(name).read_bytes() for name in DOCS}
    assert all(MARKER.encode('utf8') not in body for body in originals.values())
    text = render_report(release, proof, model, model_exec, compose_exec, history)
    # All prerequisite reads precede writes. Keep every original document byte.
    assert all(sha(at(name)) == digest for name, digest in READ.items())
    before, after, backups = {}, {}, {}
    for name, body in originals.items():
        path = OUT / 'originals' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(body)
        before[name] = hashlib.sha256(body).hexdigest()
        assert sha(path) == before[name]
        backups[rel(path)] = before[name]
    with REPORT.open('x', encoding='utf8', newline='\n') as handle:
        handle.write(text)
    for name, body in originals.items():
        target = at(name)
        assert target.read_bytes() == body
        new = prefix(name, release) + body
        temp = target.with_name(target.name + '.round79.tmp')
        with temp.open('xb') as handle:
            handle.write(new)
        os.replace(temp, target)
        assert target.read_bytes().endswith(body)
        after[name] = sha(target)
    assert sha(SELECTOR) == selection['new_selector_sha256']
    write(OUT / 'results.json', {'round': 79, 'status': 'delivery_report_and_human_documents_updated',
        'report_path': rel(REPORT), 'report_sha256': sha(REPORT),
        'source_sha256': sha(Path(__file__)), 'read_small_evidence_sha256': READ,
        'original_document_sha256': before, 'updated_document_sha256': after,
        'original_backups': backups, 'original_document_body_bytes_preserved': True,
        'selector_sha256_before': selection['new_selector_sha256'], 'selector_sha256_after': sha(SELECTOR),
        'selector_updated': False, 'new_CSV': False, 'model_operations': 0,
        'CSV_or_array_or_model_files_read': False, 'uploaded': False, 'GitHub_published': False})
    print(json.dumps({'status': 'completed', 'report_sha256': sha(REPORT), 'updated_document_sha256': after}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    assert not OUT.exists() and not REPORT.exists(), 'No overwrite or automatic retry'
    OUT.mkdir(parents=True, exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
    code, failure = 0, None
    try:
        run()
    except Exception:
        code, failure = 1, traceback.format_exc()
        print(failure, flush=True)
    write(OUT / 'execution.json', {'command': sys.argv, 'exit_code': code, 'failure': failure,
        'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - tick, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
