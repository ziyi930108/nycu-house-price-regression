"""Render completed, independently checked R116 aggregate JSON; no ML loading."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/round116'
EVIDENCE = 'outputs/research_round116_20261008'
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
    if (r['status'] != 'four_reused_fixed_origins_evaluated' or set(r['stages']) != set(ORDER)
            or audit['status'] != 'passed' or audit['results'] != record(BASE + '/results.json')
            or train['exit_code'] or evaluate['exit_code'] or r['new_CSV'] or r['full_data_fit']
            or r['Public_score'] is not None or r['Private_score'] is not None):
        raise ValueError('Require complete successful evaluation and exact independent postfit')
    gates = r['gates']
    explore = gates['combined_exploratory_submission']['passed']
    strong = gates['combined_promotion']['passed']
    decision = ('combined探索門檻通過；仍須另外全量規約與發行審查，本文不產CSV。' if explore else
                'combined探索門檻失敗，不全量fit、不產v31 CSV。')
    lines = ['# R116：單價CatBoost固定Depthwise比較', '', '2026-10-08。' + decision,
        f'combined強推薦門檻為{strong}。v22選定與上台材料保持，Public／Private沒有新增回報。', '',
        '假說是讓不同分支可以在同一深度使用不同切分，可能改善房價非線性。'
        '原SymmetricTree每層共用切分；本輪僅新增grow_policy=Depthwise。'
        '保持Plain、原日期乘面積的單價MAE、95欄／23類別、depth8、2999樹、learning_rate .065、'
        'seed42及其他原參數。不搜尋leaf最小列數或事後改深度，不混Ordered或新NN。'
        '樹形與容量表現會連動，不能將差異全歸因於某個單一節點。', '',
        '|起點|fit／query列|v22整體MAE|R116整體MAE|差vs v22|固定加權差|',
        '|---|---:|---:|---:|---:|---:|']
    for origin in ORDER:
        v = r['stages'][origin]
        stage = read(f'{BASE}/{origin}/stage.json')
        old, new = v['metrics']['v22'], v['metrics']['T']
        lines.append(f"|{origin}|{stage['fit_rows']:,}／{stage['query_rows']:,}|{old['overall_mae']:,.2f}|"
                     f"{new['overall_mae']:,.2f}|{new['overall_mae']-old['overall_mae']:+,.2f}|"
                     f"{new['weighted_mae']-old['weighted_mae']:+,.2f}|")
    lines += ['', '單位新臺幣元，負值較好。這是離線MAE差，不是預測向量差或Kaggle分數。', '',
        '## 原定判定與完整對照', ''] + block(gates)
    lines += ['只替換原單價CatBoost槽，再沿.75新CatBoost+.25原LGB、原reference、'
        '原spatial各一次，最後與原N/H分支逐筆median。原v22/v24及其餘native由各時間起點自己的保存狀態重播，'
        '不是同seed重新訓練的matched control。全部四fit及五層封存後才讀query價格評分。', '',
        '**診斷語意：** raw_unit_diagnostic_only.query_y_used_for_diagnostics=false只指封存時'
        '負單價／零投影及範圍統計；後加入的unprojected_TWD_overall_mae確實在封存後用query y計算。'
        '未投影單價先乘一次原價格因子才計總價MAE，不是直接拿單價減總價。'
        '原source/results不回寫；獨立語意補充另存於核驗目錄。', '']
    for origin in ORDER:
        v = r['stages'][origin]
        stage = read(f'{BASE}/{origin}/stage.json')
        meta = read(f'{BASE}/{origin}/model/metadata.json')
        lines += [f'## {origin}完整診斷', '', f"native fit實測{stage['fit_seconds']:.3f}秒。", '',
            '|層|總價MAE|', '|---|---:|']
        for key, value in v['branch_metrics'].items():
            lines.append(f"|{key}|{value['overall_mae']:,.2f}|")
        lines += [f"|median|{v['metrics']['T']['overall_mae']:,.2f}|", '',
            '所有季度／至少500列群退步超5%（群重疊不能相加）：', '']
        lines += block(v['comparison']['quarter_or_large_group_deterioration_above_5pct'])
        lines += ['原生CBM及葉數證據：', ''] + block({key: meta[key] for key in (
            'actual_grow_policy', 'actual_boosting_type', 'actual_model_depth', 'actual_trees',
            'actual_min_data_in_leaf', 'tree_leaf_counts_count', 'minimum_tree_leaf_count',
            'maximum_tree_leaf_count', 'tree_leaf_counts_distribution',
            'runtime_parameters_not_recovered_from_native_model')})
        lines += ['原單價逆轉換與未投影總價：', ''] + block(v['raw_unit_diagnostic_only'])
        lines += ['|條件抽樣|指標|點差vs v22|95%區間|', '|---|---|---:|---|']
        for family, item in v['paired']['methods'].items():
            for key, value in item['summary'].items():
                lines.append(f"|{family}|{key}|{value['point_delta']:+,.2f}|"
                    f"[{value['percentile_2_5']:+,.2f}, {value['percentile_97_5']:+,.2f}]|")
        lines += ['', '完整條件抽樣及支援數：', ''] + block(v['paired'])
        lines += ['v24次對照及反證（不以較弱對照替換v22）：', ''] + block(v['comparison_vs_v24'])
        lines += ['所有城市／月份／新舊案／價格帶／季度／誤差尾部：', ''] + block(v['metrics']['T'])
    lines += ['## 執行與核驗界線', '',
        f"四fit命令{train['elapsed_seconds']:.3f}秒，另evaluate命令{evaluate['elapsed_seconds']:.3f}秒。"
        '這是已測程序時間，不是整個專案工時或下一次ETA。'
        '18個可由native模型恢復的passed參數與實際Depthwise／Plain／GPU／2999樹逐項核對；'
        'allow_writing_files、thread_count、verbose是runtime宣告，不冒稱CBM能恢復。'
        '所有2999葉數保留SHA與分布，每樹1–256葉，未要求2的冪次。', '',
        '[獨立核驗](../outputs/research_round116_20261008/postfit_review.json)保存完整實作範圍。'
        'CPU真載入四個CBM、全query raw單價及TWD重播、五層與分群／gates核對是保存模型核驗，'
        '不是新環境從零重訓。', ''] + block(audit.get('limits', []))
    lines += [r['caution'], '',
        '四個已用且重疊的窗口有累積選型偏差，固定seed不代表重訓無變異；'
        '條件bootstrap不含重訓與選型不確定性，不是Private勝率。'
        '沒有外部成交答案、Public搜參數、test ID人工調價或Kaggle操作。', '',
        '- [事前規約](round116_plan.md)', '- [方法與工程參數說明](round116_method_review.md)',
        '- [四起點driver說明](round116_driver_review.md)',
        '- [單價trainer](../src/round116_depthwise_unit.py)', '- [研究流程](../src/round116_research.py)',
        '- [完整機器結果](../outputs/round116/results.json)']
    name = 'reports/round116_report.md'
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({**record(name), 'bytes': (ROOT/name).stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
