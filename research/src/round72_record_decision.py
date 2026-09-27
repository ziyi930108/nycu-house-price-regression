"""Record R72's non-adoption decision from completed, immutable local evidence."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import round48_nn_experiment as old

ROOT = old.ROOT
STAGES = ('development', 'confirmation', 'auxiliary')
NAMES = {'development': '中間年度', 'confirmation': '最新年度', 'auxiliary': '較早年度'}


def run():
    paths = {
        'results': 'outputs/round72/results.json',
        'execution': 'outputs/round72/execution.json',
        'diagnostics': 'reports/round72_diagnostics.json',
        'paired': 'outputs/round72/paired_vs_v19/results.json',
        'coverage': 'outputs/round72/road_coverage_review.json',
    }
    evidence = {k: old.locked(ROOT / p) for k, p in paths.items()}
    hashes = {k: old.sha(ROOT / p) for k, p in paths.items()}
    assert hashes['results'] == '93054804f13b0ed30d40e99c228b47bff744c5ced8b187efebbbcdf1d98beb08'
    assert hashes['diagnostics'] == '2ede671d347385593a12aeb07613ce1da42ea559f6bd543276aabe39dc6b5cda'
    r, d, b = (evidence[k] for k in ('results', 'diagnostics', 'paired'))
    assert r['checked_stages'] == list(STAGES) and r['new_NN_fits'] == 3
    assert evidence['execution']['exit_code'] == 0
    assert r['protected_before'] == r['protected_after']
    assert all(old.sha(ROOT / p) == digest for p, digest in r['protected_before'].items())
    assert all(old.sha(ROOT / p) == digest for p, digest in r['source_sha256'].items())
    assert b['replicates'] == 2000 and b['fits'] == 0 and b['model_predict_calls'] == 0
    assert not b['fresh_blind_validation']
    assert all(d['stages'][s]['all_full_saved_metrics_exact'] for s in STAGES)
    summary = {}
    for s in STAGES:
        v, z, pair = r['stages'][s], d['stages'][s], b['stages'][s]
        assert v['metrics']['R72'] == z['metrics']['R72']
        assert v['metrics']['v19'] == z['metrics']['v19']
        ci = {method: {metric: [pair[method]['summary'][metric]['percentile_2_5'],
                               pair[method]['summary'][metric]['percentile_97_5']]
                       for metric in ('overall', 'weighted')} for method in ('project', 'month')}
        assert all(low < 0 < high for item in ci.values() for low, high in item.values())
        bands = [{**{k: row[k] for k in ('band', 'rows', 'R72', 'v19')},
                  'delta_TWD': row['R72'] - row['v19'],
                  'overall_delta_contribution_TWD': (row['R72'] - row['v19']) * row['rows'] / v['query_rows']}
                 for row in z['price_bands']]
        summary[s] = {'dates': v['dates'], 'fit_rows': v['fit_rows'], 'query_rows': v['query_rows'],
                      'R72_MAE': v['metrics']['R72']['overall_mae'], 'v19_MAE': v['metrics']['v19']['overall_mae'],
                      'differences_vs_v19': v['differences']['v19'],
                      'test_composition_delta_TWD': z['test_composition']['models']['R72'] - z['test_composition']['models']['v19'],
                      'test_composition_coverage': z['test_composition']['test_mass_covered'],
                      'native_MAE_diagnostic': v['native_MAE_diagnostic'], 'price_bands': bands,
                      'road_feature_summary': v['preparation']['road_feature_summary'], 'bootstrap_95pct_intervals': ci,
                      'fit_seconds': v['fit_seconds']}
    selection = json.loads((ROOT / 'outputs/current_private_candidate.json').read_text(encoding='utf8'))
    assert selection['private_priority'] == 'outputs/submission_v19_trial.csv'
    assert old.sha(ROOT / selection['private_priority']) == selection['sha256']
    assert selection['known_public_best_filename'] == 'submission_v18_trial.csv'
    decision = {
        'round': 72, 'candidate': 'R72', 'decision': 'not_adopted_keep_v19_local_priority',
        'reason': 'Small mixed historical changes; newest-year overall, weighted, test-composition and both project strata regress; Q4 regresses in every period. All paired intervals cross zero. No compelling gain to justify the extra supervised input.',
        'primary_control': 'v19=R71 fixed seed42', 'private_priority': selection['private_priority'],
        'submission_sha256': selection['sha256'], 'backup': selection['backup'],
        'known_public_best_filename': selection['known_public_best_filename'],
        'known_public_best_mae': selection['known_public_best_mae'],
        'historical_fits': 3, 'new_final_fits': 0, 'fit_seconds': sum(v['fit_seconds'] for v in summary.values()),
        'historical_execution_seconds': evidence['execution']['elapsed_seconds'],
        'historical_main_seconds': r['elapsed_seconds'], 'new_CSV': False, 'proceed_final': False,
        'additional_seed_search': False, 'mixture_weight_search': False, 'individual_ID_rules': False,
        'Public_feedback_used_for_tuning': False, 'fresh_blind_validation': False,
        'private_mae': None, 'public_mae': None, 'kaggle_selection_changed': False,
        'final_version_chosen': False, 'GitHub_publication_deferred': True,
        'evidence': {k: {'path': p, 'sha256': hashes[k]} for k, p in paths.items()},
        'source_sha256': old.sha(Path(__file__)), 'report': 'reports/round72_report.md',
        'report_is_editable_explanation': True, 'summary': summary,
        'limitations': [
            'Repeated historical windows; bootstrap does not remove adaptive selection bias or estimate a Private probability.',
            'Only fixed seed42 fitted; no claim of equivalence or universal inferiority from intervals crossing zero.',
            'Raw 30-row support is not 30 independent projects or effective sample size.',
            'Group OOF is not forward-time OOF; project aliases and parsed address limitations remain.',
            'Full-fit query and fold-complement training have different support; populations also differ.',
            'Input width changes initialization; fallback-row prediction changes are possible.',
            'Existing reference correction already contains road-price information.',
            'Test-composition diagnostics cover only supported city/project cells and are not Private scores.',
        ],
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    old.write(ROOT / 'outputs/round72/decision.json', decision)
    lines = [
        '# R72：道路相對單價特徵實驗', '',
        '本輪不採用，維持 v19 為有限的本機 Private 優先候選，v18 為近備選及目前已知 Public 最佳（2,388,386.4613）。Private 未知，最終版本尚未選定。R72 沒有產生新提交 CSV。', '',
        '## 做了哪些動作', '',
        '在 R63／A144 神經網路輸入新增第 145 欄：道路單價先驗與既有 R60 基準單價先驗的 log 差。道路鍵為城市、行政區、道路、路段、交易類型與建物類型，至少 30 筆原始列才使用加權中位數；缺乏合格道路群組時 δ=0。沿用既有地址解析，沒有外部查價。', '',
        '訓練特徵沿用同一建案群組五折，完整排除 held-out fold；每折只用 complement 的價格與日期。查詢特徵只由完整 fit state 及已保存 R60 query 基準建立。365 日半衰期的日期 anchor／正規化使用全部 reference 列，含道路鍵缺失列；無效日期權重為中性 1。', '',
        '原 144 欄與另外 7 個 NPZ 陣列逐位不變，既有 encoder、價格 factor、權重皆重用；δ 不縮放、不裁切。固定 seed42、30 epochs、batch2048 與原模型設定，各年度一次 fit。R71 完整組合及外層 median 保持不變。查詢真值只用於評分，沒有送入道路 helper 或 trainer。', '',
        '## 三年度結果', '', '差值為 R72 − v19；負值代表 MAE 較低，單位新臺幣元。', '',
        '| 年度 | v19 MAE | R72 MAE | 整體差值 | 加權差值 | 測試組成差值 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for s in STAGES:
        v = summary[s]; diff = v['differences_vs_v19']
        lines.append(f"| {NAMES[s]} {v['dates'][0]}～{v['dates'][1]} | {v['v19_MAE']:,.2f} | {v['R72_MAE']:,.2f} | {diff['overall']['delta_TWD']:+,.2f}（{diff['overall']['relative_change_pct']:+.4f}%） | {diff['weighted']['delta_TWD']:+,.2f} | {v['test_composition_delta_TWD']:+,.2f} |")
    lines += ['', '城市 × 新舊建案組成權重沿用既有規則；中間／最新年度覆蓋測試列 99.6580%，較早年度 94.7847%。這是歷史 MAE 的組成調整，並非 Private 預測分數。', '',
              '## 為何不取代 v19', '',
              '最新年度整體、加權、新案、舊案與測試組成均退步；Q3、Q4 也退步。三個年度的 Q4 全都變差。較早年度雖整體略改善，測試組成卻略退步；一億元以上 171 筆的 MAE 增加約 887,125 元，抵銷其餘四個價位帶的改善。最新年度只有 1,000～2,000 萬價位帶改善，其餘四帶退步。', '',
              '未經後處理的原生 prior NN 在中間／最新年度也退步，較早年度改善。完整組合的微小變化不能解讀成新欄位帶來穩定收益；本輪沒有追加種子、權重、門檻或逐筆規則搜尋來尋找較好數字。', '',
              '以固定 2,000 次配對重抽樣比較建案群組及 3 個月區塊，三期的整體與加權 95% 區間全部跨過 0。這不是證明模型相等或 R72 一定較差，而是本輪沒有足夠一致的收益支持增加複雜度。這些窗口已反覆使用，區間未修正選型偏差。', '',
              '| 年度 | 建案群組：整體差值 95% 區間 | 月份區塊：整體差值 95% 區間 |', '|---|---:|---:|']
    for s in STAGES:
        ci = summary[s]['bootstrap_95pct_intervals']
        lines.append(f"| {NAMES[s]} | [{ci['project']['overall'][0]:+,.2f}, {ci['project']['overall'][1]:+,.2f}] | [{ci['month']['overall'][0]:+,.2f}, {ci['month']['overall'][1]:+,.2f}] |")
    lines += ['', '## 道路覆蓋與限制', '',
              '| 年度 | 訓練 OOF 支持率 | query full-fit 支持率 |', '|---|---:|---:|']
    for s in STAGES:
        f = summary[s]['road_feature_summary']
        lines.append(f"| {NAMES[s]} | {100*f['train']['selected_support_positive_rows']/f['train']['rows']:.4f}% | {100*f['query']['selected_support_positive_rows']/f['query']['rows']:.4f}% |")
    lines += ['', '依保存的 query support 分群，以下是各群 MAE 的 R72 − v19 差值：', '',
              '| 年度 | 有道路支持 | fallback |', '|---|---:|---:|']
    for stage in STAGES:
        groups = evidence['coverage']['stages'][stage]['groups']
        lines.append(f"| {NAMES[stage]} | {groups['supported']['MAE_delta_R72_minus_v19_TWD']:+,.2f} | {groups['fallback']['MAE_delta_R72_minus_v19_TWD']:+,.2f} |")
    lines += ['', '支持率以 support>0 定義，不能以 δ≠0 代替。OOF complement 與 full-fit 的樣本量不同，train/query 也不是同一人群，不能把覆蓋差全歸因於 OOF。30 筆不代表 30 個獨立建案；整案分折不是時間前向分折；建案別名、缺失與地址解析限制仍在。', '',
              '新增輸入寬度也改變第一層初始化與參數數量，因此 fallback 列仍可能改變預測。既有 reference 已使用道路價格資訊，不能將所有預測差異因果歸因於新增道路特徵。按支持／fallback 的事後診斷另存 reports/round72_road_coverage_review.md，不用它制定挑列規則。', '',
              '## 執行與保存', '',
              f"實際 3 次神經網路 fit，合計 {decision['fit_seconds']:.3f} 秒；主流程 {r['elapsed_seconds']:.3f} 秒，外層執行 {decision['historical_execution_seconds']:.3f} 秒，exit 0。所有年度皆執行，未因早期分數停止。保存模型完整 query 重載精確一致；11 個完整對照評分一致；20 個原提交 CSV 與版本登錄檔共 21 檔均保持不變。", '',
              '合成 helper 首次測試因人工日期移動到 1e8 天而觸發既有日期權重下溢保護，保留失敗證據，只把測試 fixture 改為 1,000 天後重試；沒有為此改真實資料或 helper 規則。真實 R72 三期沒有失敗或重試。', '',
              '程式保留於 src/round72_road_prior.py、src/round72_nn_experiment.py、src/round72_road_coverage_review.py 與本決策紀錄程式。沿用的訓練器／診斷器、全部模型、陣列、參數、來源 hash、執行紀錄與失敗測試均保留。正式單一重現入口與 GitHub 報告待使用者選定最終版本後整理；目前不發布。', '',
              '歷史實驗命令：', '', '```powershell', r'.\.venv\Scripts\python.exe -u -B -X utf8 src/round72_nn_experiment.py --run', '```', '',
              '命令依賴前序已保存 artifacts，且拒絕覆寫既有 round72 資料夾；目前保留的是完整研究鏈，尚未聲稱已整理成獨立交付套件。', '',
              '## 機器證據', '']
    for k, p in paths.items():
        lines.append(f"- {k}: `{p}`；SHA256 `{hashes[k]}`。")
    lines += ['', '本 Markdown 是可編輯說明；決策以列出的 JSON、來源與 artifact hashes 為準。', '']
    report = ROOT / 'reports/round72_report.md'
    assert not report.exists()
    report.write_text('\n'.join(lines), encoding='utf8')
    report.with_suffix('.sha256').write_text(old.sha(report) + '\n', encoding='utf8')
    print(json.dumps({'decision': decision['decision'], 'decision_sha256': old.sha(ROOT/'outputs/round72/decision.json'),
                      'report_sha256': old.sha(report)}, ensure_ascii=False))


if __name__ == '__main__':
    run()
