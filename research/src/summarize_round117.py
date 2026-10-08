"""Render only completed independently checked R117v2 aggregate JSON, no ML."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/round117_v2'
EVIDENCE = 'outputs/research_round117_v2_20261008'
ORDER = ('development', 'confirmation', 'auxiliary', 'offset_2023_02_11')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def record(name):
    return {'path': name, 'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}


def block(value):
    return ['```json', json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), '```', '']


def main():
    r = read(BASE + '/results.json')
    audit = read(EVIDENCE + '/postfit_review.json')
    train = read(EVIDENCE + '/train_execution.json')
    evaluate = read(EVIDENCE + '/evaluate_execution.json')
    original = read('outputs/research_round117_20261008/train_execution.json')
    if (r['status'] != 'four_reused_fixed_origins_evaluated' or set(r['stages']) != set(ORDER)
            or audit['status'] != 'passed' or audit['results'] != record(BASE + '/results.json')
            or train['exit_code'] or evaluate['exit_code'] or original['exit_code'] != 1
            or r['new_CSV'] or r['full_data_fit']
            or r['historical_aborted_batch_completed_NN_fits'] != 1
            or r['fresh_NN_fits_in_this_batch'] != 4
            or r['total_R117_actual_NN_fits_including_aborted'] != 5
            or r['Public_score'] is not None or r['Private_score'] is not None):
        raise ValueError('Require exact complete evaluation, independent postfit and historical cost')
    explore = r['gates']['combined_exploratory_submission']['passed']
    strong = r['gates']['combined_promotion']['passed']
    decision = ('combined探索門檻通過；仍需獨立全量規約與發行審查，本報告不產CSV。' if explore else
                'combined探索門檻失敗，不全量fit、不產v32 CSV。')
    lines = ['# R117：fit先驗固定交易五折比較', '', '2026-10-08。' + decision,
        f'combined強推薦門檻為{strong}。v22選定與上台材料保持，Public／Private沒有新回報。', '',
        '原本先驗為每筆房屋估計參考單價，訓練時用分組五折，避免同案價格互相暴露。'
        'R115純X盤點顯示交易五折可能增加細層級支持，因此本輪只把fit的OOF先驗改為'
        'canonical列順序上seed42固定交易五折；每一fold用自己的補集建立價格狀態、日期anchor及日期權重。'
        '保留四cell、至少30列支持、原weighted median與fallback；query仍使用完整fit狀態。'
        '同案交易可能高度相似，交易fold會暴露同案peers，而真query多為新案，這是可能退步的主要原因。', '',
        '九個NN陣列僅train_price_factor_million改變，其他八項逐bytes保持。'
        '原Softplus倍率NN、512/256/128架構、seed42、30epochs、2048batch、AdamW與cosine排程保持。'
        '價格因子改變會依原公式連動初始ratio/bias；不是固定相同數值的初始化。'
        'query prior及query factor逐bytes等於各起點原版；不混signed head、Depthwise或其他新設定。', '',
        '|起點|fit／query列|v22整體MAE|R117整體MAE|差vs v22|固定加權差|',
        '|---|---:|---:|---:|---:|---:|']
    for origin in ORDER:
        v = r['stages'][origin]
        stage = read(f'{BASE}/{origin}/stage.json')
        old, new = v['metrics']['v22'], v['metrics']['T']
        lines.append(f"|{origin}|{stage['fit_rows']:,}／{stage['query_rows']:,}|{old['overall_mae']:,.2f}|"
                     f"{new['overall_mae']:,.2f}|{new['overall_mae']-old['overall_mae']:+,.2f}|"
                     f"{new['weighted_mae']-old['weighted_mae']:+,.2f}|")
    lines += ['', '單位新臺幣元，負值較好。離線MAE差不是Kaggle分數或預測向量差。', '',
        '## 事前門檻與完整對照', ''] + block(r['gates'])
    lines += ['只替換原prior NN槽；與原area NN各半後乘.75，再加.25原LGB，'
        '沿原reference、spatial各一次，最後與原H/U逐筆median。'
        '舊分支與控制是各起點自己的保存狀態，不是同seed重新訓練的matched control。'
        '四個新prior、四個NN和全部五層封存並成功退出後，另命令讀query價格評分。', '']
    for origin in ORDER:
        v = r['stages'][origin]
        stage = read(f'{BASE}/{origin}/stage.json')
        prep = read(f'{BASE}/{origin}/prior/preparation.json')
        lines += [f'## {origin}完整診斷', '',
            f"prior／prepare實測{stage['prior_prepare_seconds']:.3f}秒，native NN fit {stage['fit_seconds']:.3f}秒。", '',
            '交易fold、初始化與fit價格因子變動：', ''] + block({key: prep[key] for key in (
                'fold_policy', 'fold_seed', 'folds', 'fold_rows', 'chronological_oof',
                'fit_factor_change_summary', 'original_initialization', 'transaction_initialization',
                'initial_bias_changes_linked_to_fit_factor', 'numeric_input_width', 'categorical_input_width',
                'encoder_refit', 'full_fit_state_and_query_prior_and_factor_exact_original',
                'query_prior_level_counts', 'legacy_state_group_rule_string_is_not_transaction_fold_evidence')})
        lines += ['|層|總價MAE|', '|---|---:|']
        for key, value in v['branch_metrics'].items():
            lines.append(f"|{key}|{value['overall_mae']:,.2f}|")
        lines += [f"|median|{v['metrics']['T']['overall_mae']:,.2f}|", '',
            '季度／至少500列群退步超5%（重疊群不能相加）：', '']
        lines += block(v['comparison']['quarter_or_large_group_deterioration_above_5pct'])
        lines += ['|條件抽樣|指標|點差vs v22|95%區間|', '|---|---|---:|---|']
        for family, item in v['paired']['methods'].items():
            for key, value in item['summary'].items():
                lines.append(f"|{family}|{key}|{value['point_delta']:+,.2f}|"
                             f"[{value['percentile_2_5']:+,.2f}, {value['percentile_97_5']:+,.2f}]|")
        lines += ['', '完整條件抽樣與支援數：', ''] + block(v['paired'])
        lines += ['v24次對照與反證（不代替v22門檻）：', ''] + block(v['comparison_vs_v24'])
        lines += ['完整城市／月份／新舊案／季度／價格帶／尾部：', ''] + block(v['metrics']['T'])
    lines += ['## 工程失敗、實際成本與核驗範圍', '',
        f"原R117 train命令{original['elapsed_seconds']:.3f}秒exit1；development先驗及NN已完成，"
        '但動態R52父來源未列入AST來源閉包，metadata核驗因缺鍵停止。'
        '沒有stage seal、其餘三fit或query分數。原目錄全部保留，不把已花成本當0或刪除。'
        '另固定工程恢復規約及新source/目錄，補明列動態來源與GPU前守衛；四prior／NN全部重新訓練，'
        '不沿用原partial模型。科學假說、切分及門檻不改。', '',
        f"新v2四fit命令{train['elapsed_seconds']:.3f}秒，另evaluate {evaluate['elapsed_seconds']:.3f}秒。"
        '本研究合計五次實際NN fit＝原失敗批一個＋新完整批四個，prior也五次。'
        '以上程序時間不等於專案工時或下一次ETA。', '',
        '[獨立postfit核驗](../outputs/research_round117_v2_20261008/postfit_review.json)'
        '說明真正重跑的檢查與限制；保存artifact核驗不是乾淨環境端到端重訓。', '']
    lines += block(audit.get('limits', audit.get('limitations', [])))
    lines += [r['caution'], '',
        '四個窗口已被反覆使用且重疊，有累積選型偏差；單一seed未估計重訓變異。'
        '條件bootstrap不含重訓或選型不確定性，不能稱Private勝率。'
        '沒有外部成交答案、Public搜參數、test ID人工調價或Kaggle操作。', '',
        '- [原科學規約](round117_plan.md)', '- [工程恢復規約](round117_engineering_recovery_v2.md)',
        '- [先驗方法](round117_method_review.md)', '- [四起點driver說明](round117_driver_review.md)',
        '- [交易OOF先驗](../src/round117_transaction_prior.py)',
        '- [原倍率NN trainer](../src/round117_transaction_prior_train.py)',
        '- [v2研究流程](../src/round117_research_v2.py)',
        '- [完整機器結果](../outputs/round117_v2/results.json)']
    name = 'reports/round117_report.md'
    with (ROOT / name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({**record(name), 'bytes': (ROOT / name).stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
