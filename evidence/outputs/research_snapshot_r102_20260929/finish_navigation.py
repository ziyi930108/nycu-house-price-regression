"""Finish four R101/R102 navigation files after the completed evidence copy.

Read aggregate JSON only. Never train, score, package, commit or publish.
The root thread explicitly authorizes execution after copy_completed.py succeeds.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'deliverables/v22_github'
DEST = REPO / 'evidence/outputs/research_snapshot_r102_20260929'
BASE = '4217cc53e54c917a27a57a48f994eb16ea05b066'
PAGE = 'docs/RESEARCH_UPDATES_20260929_PRIOR_INPUT.md'
NAV = ['README.md', 'HANDOFF.md', 'research/README.md', PAGE]
STAGES = ['development', 'confirmation', 'auxiliary']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def save(p, value):
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def links(text):
    return [v.strip('<>') for v in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', text)]


def interval_counts(result, stage, control):
    counts = dict(negative=0, crossing=0, positive=0)
    for family in ['project', 'month']:
        for metric in ['overall', 'weighted']:
            v = result['stages'][stage]['paired'][control]['methods'][family]['summary'][metric]
            assert v['valid_replicates'] == 2000 and v['invalid_replicates'] == 0
            lo, hi = v['percentile_2_5'], v['percentile_97_5']
            assert lo <= hi
            counts['negative' if hi < 0 else 'positive' if lo > 0 else 'crossing'] += 1
    return counts


def yes(value):
    return '通過' if value else '未通過'


def build_documents(result, seal, costs, copied_paths):
    gate = result['research_gates']; primary = gate['primary']; mechanism = gate['mechanism']
    passed = gate['both_research_gates_passed']
    decision = '兩項研究門檻通過，但不自動採用' if passed else '研究門檻未全通過，不採用'
    deltas = [result['stages'][s]['metrics']['T']['overall_mae'] - result['stages'][s]['metrics']['v22']['overall_mae'] for s in STAGES]
    short_delta = '／'.join(f'{d:+,.2f}' for d in deltas)
    count_rows = [(s, control, interval_counts(result, s, control)) for s in STAGES for control in ['v22', 'C']]
    totals = {key: sum(row[2][key] for row in count_rows) for key in ['negative', 'crossing', 'positive']}
    assert sum(totals.values()) == result['conditional_intervals_reported'] == 24
    intro = (f'R101 診斷與 R102 六次 NN／三期隔離評分均已完成。T 相對 v22 的 development／confirmation／auxiliary MAE 差為 '
             f'**{short_delta} 元**。Primary gate {yes(primary["adoption_research_gate_passed"])}，'
             f'mechanism gate {yes(mechanism["mechanism_gate_passed"])}；**{decision}；唯一提交仍是 v22，Private 未知。**')
    page = ['# R101／R102：把時間價格先驗明確加入回歸輸入', '', '2026-09-29。' + intro, '',
        f'本頁於發布前核驗，基底 {BASE}；當前發布身分以 Git main 歷史及 SOURCE_MANIFEST 為準，根工作區另存发布收據。'
        'v22 Public MAE 1,998,568.5706 元是使用者回報；下表是本機回測。', '',
        '## 診斷與固定假設', '',
        '[R101 診斷](../evidence/reports/round101_report.md)只讀保存的 fit／無真價 query 分布，檢查 prior、ratio、日期權重及有效 ratio loss 權重。'
        '沒有新 prior、模型、MAE 或 bootstrap；不能把分布差異當成因果。R100 結束時提出的這項診斷現在已完成。', '',
        '[R102 固定計畫](../evidence/reports/round102_plan.md)檢查：當時間先驗已用於還原總價時，再讓 NN 看見先驗大小，是否改善回歸？'
        '兩組均重用同一期 R100T 的 eligible loss 列、factor、y、日期權重與原 144 欄；七個非 numeric array 位元保持，不再建立 prior。', '',
        '|組別|第 145 欄|作用|', '|---|---|---|',
        '|C：同寬度對照|float32 正零|控制新增一欄及相同架構，不是 R100 舊 C|',
        '|T：研究候選|log(單價 prior)，以 eligible fit 的 float64 平均／標準差標準化，再轉 float32|query 用原 full-fit prior 與同一組 fit 參數，無 clipping|', '',
        '兩組共用原 R63 trainer、seed 42、30 epochs、batch 2048 及相同初始化。只有新欄不同；完整模型仍保留原 area NN／LGB、R/S 與 H/U 分支，再取三支 median。'
        '六個 NN 與所有 query 預測重載封存後，隔離 scorer 才讀 query 真價。', '',
        '## 完整模型與原生 NN 結果', '',
        '單位為元，負差較好。C 僅作機制對照，不依結果另選提交。', '',
        '|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 加權差|native T−C|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        a = result['stages'][s]; m = a['metrics']; native = a['layer_metrics']
        values = [m['v22']['overall_mae'], m['C']['overall_mae'], m['T']['overall_mae'],
                  m['T']['overall_mae']-m['v22']['overall_mae'], m['T']['overall_mae']-m['C']['overall_mae'],
                  m['T']['weighted_mae']-m['v22']['weighted_mae'], native['T']['native']['overall_mae']-native['C']['native']['overall_mae']]
        page.append('|' + s + '|' + '|'.join(f'{v:,.2f}' if i < 3 else f'{v:+,.2f}' for i, v in enumerate(values)) + '|')
    labels = {'all_three_overall_improve':'三期完整 overall 均改善', 'all_three_weighted_improve':'三期完整 weighted 均改善',
        'confirmation_at_least_half_percent':'confirmation 至少改善 0.5%',
        'no_quarter_or_large_group_deterioration_above_5pct':'無季度或至少 500 列固定群退步超過 5%',
        'native_all_three_overall_improve':'三期原生 T 均優於 C', 'full_all_three_overall_improve':'三期完整 T overall 均優於 C',
        'full_all_three_weighted_improve':'三期完整 T weighted 均優於 C'}
    page += ['', '|門檻|固定條件|結果|', '|---|---|---|']
    for name, components in [('Primary', primary['components']), ('Mechanism', mechanism['components'])]:
        for key, value in components.items():
            page.append(f'|{name}|{labels[key]}|{yes(value)}|')
    adverse = sum(len(result['stages'][s]['comparison']['quarter_or_large_group_deterioration_above_5pct']) for s in STAGES)
    page += ['', f'觸及固定群／季度 >5% 限制的列共 {adverse} 條；全部群別、尾部、T−v24 及 C−v22 點估計保留於'
        '[完整評分](../evidence/outputs/round102/evaluation/report.md)與[聚合摘要](../evidence/reports/round102_report.md)。'
        '兩項 gate 分開判斷，不能用後處理改善代替原生 NN 改善，也不能把層間差加總成因果貢獻。', '',
        '## 配對區間與限制', '',
        f'固定 seed 20260925、每種 2,000 次 project／連續三個月 block 配對，共 24 CI：'
        f'{totals["negative"]} 全負、{totals["crossing"]} 跨零、{totals["positive"]} 全正。', '',
        '|窗口|比較|全負|跨零|全正|', '|---|---|---:|---:|---:|']
    page += [f'|{s}|T−{control}|{v["negative"]}|{v["crossing"]}|{v["positive"]}|' for s, control, v in count_rows]
    page += ['', '年度窗口反覆使用且只有一個 seed；這是條件於保存預測的區間，不是新的盲測或 Private 勝率。'
        'C−v22、T−v24 沒有另外新增 bootstrap。原 R100T 與 R100C 的 native 結果僅是歷史背景，不能混成本輪 Czero。', '',
        '## 實測耗時與來源', '', '|步驟|秒數|範圍|', '|---|---:|---|',
        f'|R101 診斷 main|{costs["diagnostic"]:.3f}|0 新模型／MAE|',
        f'|R102 六次 NN fit|{seal["fit_seconds"]:.3f}|已包含於 main，不相加|',
        f'|R102 main|{costs["main"]:.3f}|輸入、六 NN、重載與封存|',
        f'|R102 scorer|{costs["evaluate"]:.3f}|來源核驗、指標與 24 CI|',
        f'|R102 聚合摘要|{costs["summary"]:.3f}|只整理 JSON，沒有新評分|', '',
        '[R101 原碼](../research/src/round101_prior_diagnostics.py)／[統計 helper](../research/src/round101_statistics.py)；'
        '[R102 driver](../research/src/round102_prior_input_experiment.py)／[隔離 scorer](../research/src/round102_prior_input_evaluate.py)／'
        '[scorer 契約](../evidence/reports/round102_evaluation_contract.md)／'
        '[實際 metadata 前置核對](../evidence/reports/round102_evaluator_metadata_preflight.json)／'
        '[完整聚合結果](../evidence/outputs/round102/evaluation/results.json)。', '']
    audit = 'evidence/reports/round102_results_review.md'
    assert audit in copied_paths
    page += ['[保存結果有限核驗](../' + audit + ')另列實際範圍與作者角色，不將保存證據核對冒稱完全獨立重訓。', '']
    page += ['## 交接與下一步界線', '',
        '[本批精確複製清單](../evidence/outputs/research_snapshot_r102_20260929/copy_manifest.json)保留來源 bytes／SHA。'
        '本包不含原交易表、標籤、NPY／NPZ、模型、完整 prefix／price-prior state 或逐筆向量；原 source/report 的部分連結仍依賴本機工作區。'
        '這是文件與程式快照，並非從零可重訓備份或乾淨環境重現。', '',
        '[有限下一方向](../evidence/reports/round102_next_direction.md)只提出先另固定誤差互補與 median 路由診斷，尚未執行。'
        '原生 NN 三窗改善，但 confirmation 在固定 base 融合已翻負、development 到 median 才翻負；不能直接歸咎 R/S，'
        '也不能用這些 query 搜最佳權重。尚無新模型計畫、係數、篩選規則或 CSV。', '',
        '正式 v22 REPORT／PDF／PPTX、379 支歷史來源、既有 CSV 與未知 Private 維持。本頁不新增下一個候選、調參或訓練承諾；'
        '後續研究須另有固定假設与來源證據。[R100 歷史頁](RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)保留當時結果及反證。', '']
    latest = ('## 最新先驗輸入研究\n\n[R101／R102 完成頁](' + PAGE + ')：' + intro + '\n\n'
        f'24 個配對區間為 {totals["negative"]} 全負／{totals["crossing"]} 跨零／{totals["positive"]} 全正；窗口重用限制及所有群別反證保留。'
        'R101 僅做保存分布診斷；R102 比較同一 R100T 資料的 145 欄正零對照與 log-prior 候選，七個非 numeric array 不變。'
        '正式報告與簡報保持 v22；當前發布身分以 Git main／SOURCE_MANIFEST 為準。\n\n'
        '[上一輪 R100](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)及更早沿革由完成頁連結保留。\n\n')
    documents = {}
    s = (REPO / NAV[0]).read_text(encoding='utf8');start = s.index('## 最新時間前置先驗研究');end = s.index('## 簡報與專案更新', start)
    documents[NAV[0]] = s[:start] + latest + s[end:]
    s = (REPO / NAV[1]).read_text(encoding='utf8').replace('R85–R100', 'R85–R102')
    replacements = {'最新研究':f'R101 診斷＋R102 六 NN 完成；T−v22 {short_delta} 元；{decision}，無新 CSV',
        '本批發布':f'本頁於發布前核驗，base {BASE}；當前身分以 Git main／SOURCE_MANIFEST 為準',
        '後續界線':'R101／R102 已完成，v22 唯一提交；後續研究需另固定計畫，本頁不新增訓練承諾'}
    for key, value in replacements.items():
        s, count = re.subn(r'^\| ' + key + r' \|.*$', '| ' + key + ' | ' + value + ' |', s, flags=re.M);assert count == 1
    s = s.replace('- [最新 R100', '- [最新 R101／R102](' + PAGE + ')：診斷、145 欄對照、兩項門檻、24 CI 與依賴界線。\n- [前批 R100')
    start = s.index('## 最新研究與後續工作');end = s.index('使用者已選定v22，最新紀錄', start)
    s = s[:start] + '## 最新研究與後續工作\n\n[' + 'R101／R102 完成頁](' + PAGE + ')是最新入口。' + intro + '\n\n' + (
        'R101 的 fit／無真價 query 診斷已完成；R102 比較同一 R100T 七個共享 array 與同一 145 欄架構，只改末欄正零／標準化 log prior。'
        '全部六 NN 和 query 重載封存後才由隔離 scorer 讀真價。C 是同寬度機制對照，並非上一輪的 loss-mask C。\n\n'
        '完整四向量、五層指標、兩项 gate、24 CI、所有群別退步及實測耗時見完成頁與[摘要](evidence/reports/round102_report.md)。'
        '窗口重用偏差、單 seed、未知 Private 與從零重訓未驗證的限制保留。\n\n'
        '[R100](docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md)的當時下一方向已在 R101／R102 分別診斷與試驗，歷史頁不回寫。'
        '[下一方向](evidence/reports/round102_next_direction.md)只提先另固定誤差互補／median 路由診斷，尚未執行；'
        '最近窗在 base 已翻負，不能直接怪 R/S，也不能據 query 搜權重。本批不建立新的候選／調參／背景訓練承諾；部分價格狀態與模型依賴只在原工作區。\n\n') + s[end:]
    documents[NAV[1]] = s
    s = (REPO / NAV[2]).read_text(encoding='utf8').replace('R85–R100','R85–R102').replace('## 最新追加：R100','## 歷史追加：R100')
    note = ('## 最新追加：R101 診斷與 R102 先驗輸入\n\n[完成頁](../' + PAGE + ')：' + intro + '\n\n'
        '[R101 driver](src/round101_prior_diagnostics.py)／[statistics](src/round101_statistics.py)只做保存分布診斷；'
        '[R102 driver](src/round102_prior_input_experiment.py)串行六 NN；[scorer](src/round102_prior_input_evaluate.py)在全部封存後才評分。'
        '145 欄／七共享 array 與兩项 gate 詳見[計畫](../evidence/reports/round102_plan.md)及[摘要](../evidence/reports/round102_report.md)。\n\n'
        '[本批來源清單](../evidence/outputs/research_snapshot_r102_20260929/copy_manifest.json)保留 bytes；不含標籤／模型／NPZ／逐筆向量／價格 prior states。'
        '379 歷史來源與正式交付不改。下方歷史研究的「尚未執行」是各批當時狀態；R100 提議的診斷與本輪輸入假說現已完成。\n\n')
    split = s.find('\n') + 1;documents[NAV[2]] = s[:split] + '\n' + note + s[split:]
    documents[NAV[3]] = '\n'.join(page)
    return documents, totals


def main():
    assert (DEST/'copy_manifest.json').is_file() and (DEST/'baseline.json').is_file()
    assert not (DEST/'final_local_verification.json').exists() and not (REPO/PAGE).exists()
    baseline = read(DEST/'baseline.json');copied = read(DEST/'copy_manifest.json');input_sha = {}
    assert sha(REPO/'SOURCE_MANIFEST.json') == baseline['manifest_sha256']
    for row in baseline['files']:
        assert sha(REPO/row['path']) == row['sha256'], row['path']
    for row in copied['files']:
        assert sha(ROOT/row['source']) == row['sha256'] == sha(REPO/row['destination']), row['source']
    for name in NAV[:3]:
        assert (DEST/'navigation_before'/name).read_bytes() == (REPO/name).read_bytes()
    def bound(name):
        p = ROOT/name;value = read(p);input_sha[name] = sha(p);return value
    paths = ['outputs/round101/execution.json','outputs/round102/execution.json','outputs/round102/evaluation/execution.json',
             'outputs/round101_summary/execution.json','outputs/round102_summary/execution.json']
    executions = {p:bound(p) for p in paths}
    assert all(v['exit_code']==0 for v in executions.values())
    diagnostic = bound('outputs/round101_summary/summary.json');summary = bound('outputs/round102_summary/summary.json')
    bound('outputs/round102_summary/receipt.json')
    assert diagnostic['new_models']==0 and diagnostic['new_MAE']==0 and diagnostic['query_labels_read'] is False
    assert isinstance(summary,dict) and summary
    result = bound('outputs/round102/evaluation/results.json');seal = bound('outputs/round102/sealed_manifest.json')
    assert result['status']=='completed_three_window_six_fit_evaluation' and result['checked_stages']==STAGES
    assert seal['status']=='all_prior_input_predictions_sealed' and seal['new_NN_fits']==6
    assert result['selected_submission']=='v22' and result['Private_score'] is None and result['new_CSV'] is False
    copied_paths = {r['destination'] for r in copied['files']}
    for name in ['evidence/reports/round101_report.md','evidence/reports/round102_report.md',
                 'evidence/reports/round102_results_review.md','evidence/reports/round102_next_direction.md']:
        assert name in copied_paths
    costs = {'diagnostic':executions[paths[0]]['elapsed_seconds'],'main':executions[paths[1]]['elapsed_seconds'],
             'evaluate':executions[paths[2]]['elapsed_seconds'],'summary':executions[paths[4]]['elapsed_seconds']}
    docs, counts = build_documents(result,seal,costs,copied_paths);checks=[]
    for name,text in docs.items():
        for href in links(text):
            if re.match(r'^[a-z]+:|^#',href,re.I):continue
            q = ((REPO/name).parent/urllib.parse.unquote(href.split('#')[0])).resolve()
            exists = q.exists() or q == (REPO/PAGE).resolve()
            checks.append({'document':name,'href':href,'exists':exists});assert exists,(name,href)
    changes=[]
    for name,text in docs.items():
        p=REPO/name;before=sha(p) if p.exists() else None
        with p.open('w' if p.exists() else 'x',encoding='utf8',newline='\n') as f:f.write(text)
        changes.append({'path':name,'before_sha256':before,'after_sha256':sha(p),'bytes':p.stat().st_size})
    changed=[r['path'] for r in baseline['files'] if sha(REPO/r['path'])!=r['sha256']]
    assert set(changed)==set(NAV[:3]),changed
    for row in copied['files']:assert sha(REPO/row['destination'])==row['sha256'],row['destination']
    assert sha(REPO/'SOURCE_MANIFEST.json')==baseline['manifest_sha256']
    save(DEST/'document_changes.json',{'files':changes,'input_sha256':input_sha,'publication_claim':'verified before publication; current identity Git main/SOURCE_MANIFEST'})
    save(DEST/'navigation_links.json',{'status':'passed','navigation_count':4,'checks':checks})
    receipt={'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'published_base':BASE,
        'copied_count':copied['copied_count'],'copied_bytes':copied['copied_bytes'],'all_original_copies_exact':True,
        'baseline_files':len(baseline['files']),'baseline_changed_only':changed,'navigation_changes':changes,
        'navigation_local_links_checked':len(checks),'all_navigation_links_resolve':True,'interval_counts':counts,
        'input_sha256':input_sha,'source_sha256':sha(Path(__file__)),'SOURCE_MANIFEST_unchanged':True,
        'historical_379_sources_and_manifest_unchanged':True,'formal_REPORT_PDF_PPTX_unchanged':True,
        'original_workspace_dependencies':['raw data','labels','arrays','models','full price-prior/prefix states','cross-round saved private artifacts'],
        'new_fit_score_calls':0,'package_commit_push_performed':False}
    save(DEST/'final_local_verification.json',receipt)
    print(json.dumps({'status':'passed','copied_count':copied['copied_count'],'copied_bytes':copied['copied_bytes'],
        'navigation_links':len(checks),'receipt':str(DEST/'final_local_verification.json'),
        'receipt_sha256':sha(DEST/'final_local_verification.json')},ensure_ascii=False))


if __name__=='__main__':
    main()
