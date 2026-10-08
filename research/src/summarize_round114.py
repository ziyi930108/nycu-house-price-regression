"""Aggregate-JSON report only; never loads price tables, arrays or models."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
GROUP = 'outputs/research_round114_20261008'
ORDER = ('development', 'confirmation', 'auxiliary', 'offset_2023_02_11')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def record(name):
    return {'path': name, 'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}


def block(value):
    return ['```json', json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), '```', '']


def main():
    result = read('outputs/round114/results.json')
    audit = read(GROUP + '/postfit_review.json')
    train = read(GROUP + '/train_execution.json')
    evaluate = read(GROUP + '/evaluate_execution.json')
    if (result['status'] != 'four_reused_fixed_origins_evaluated' or set(result['stages']) != set(ORDER)
            or audit['status'] != 'passed' or audit['results'] != record('outputs/round114/results.json')
            or train['exit_code'] or evaluate['exit_code'] or result['new_CSV'] or result['full_data_fit']
            or result['Public_score'] is not None or result['Private_score'] is not None):
        raise ValueError('Complete four-origin evaluated and independently reviewed results required')
    clarification = {
        'status': 'interpretation_clarification_without_original_mutation',
        'results': record('outputs/round114/results.json'),
        'original_field': 'stages.*.raw_signed_total_diagnostic_only.query_y_used_for_diagnostics=false',
        'scope_of_original_false': 'Pre-scoring native negative-total/projection count and magnitude statistics copied from sealed model metadata.',
        'overall_mae_uses_query_labels': True,
        'overall_mae_computed_only_in_post_seal_evaluation': True,
        'original_results_source_and_gates_unchanged': True,
        'no_fit_or_query_vector_changes': True,
    }
    clarification_path = ROOT / GROUP / 'evaluation_semantics_clarification.json'
    with clarification_path.open('x', encoding='utf-8') as stream:
        json.dump(clarification, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    lines = ['# R114：參考總價加金額價差，未通過完整門檻', '',
        '2026-10-08。四個起點全部完成固定30 epochs訓練、封存及分開評分，獨立postfit已通過。'
        '原三窗探索／強門檻均失敗，額外2月起點在允許的微幅退步範圍內；combined門檻失敗。'
        '**不全量fit，不產v30 CSV，不改v22選定或上台材料。** Public／Private沒有新增分數。', '',
        '動機是檢查原「參考總價×NN正倍率」的參數化是否限制金額誤差的學習。'
        '改為「參考總價＋NN有正負值的價差」，保持原MLP、144數值／23類別、九array、'
        '日期加權總價MAE、optimizer、30epochs、seed42和固定組合。head activation、單位、bias、'
        '正值約束及梯度尺度一同改變，不能將差異只歸因於某一條梯度。', '',
        '|起點|fit／query列|v22整體MAE|R114整體MAE|差vs v22|固定加權差|',
        '|---|---:|---:|---:|---:|---:|']
    for stage in ORDER:
        v = result['stages'][stage]
        s = read(f'outputs/round114/{stage}/stage.json')
        m, b = v['metrics']['T'], v['metrics']['v22']
        lines.append(f"|{stage}|{s['fit_rows']:,}／{s['query_rows']:,}|{b['overall_mae']:,.2f}|"
                     f"{m['overall_mae']:,.2f}|{m['overall_mae']-b['overall_mae']:+,.2f}|"
                     f"{m['weighted_mae']-b['weighted_mae']:+,.2f}|")
    lines += ['', '單位為新臺幣元，負值較好。最新confirmation退步約0.3747%，兩指標都變差，'
        '且只有auxiliary整體改善，未達探索條件；auxiliary嘉義市731列退步5.6063%，'
        '也觸及大型分群限制。額外起點微幅退步但通過其上限，並不代表該起點改善。', '',
        '## 原定判定與負值政策', ''] + block(result['gates'])
    lines += ['訓練以未裁剪的F+h計算總價MAE，fit不clip；native推論才作一次max(total,0)，'
        '之後沿原reference／spatial及median組合。對非負真價，這個投影本身不會增加單筆絕對誤差；'
        '這個數學性質不代表R114勝過v22。保存的raw向量是未投影的總價，不是NN價差h。', '',
        '**評分欄位語意：** 原JSON的query_y_used_for_diagnostics=false只指先封存的負值／投影統計。'
        '同一區塊後加入的overall_mae是在封存後讀query真價計算，確實使用query y。'
        '原結果不回寫；[語意補充](../outputs/research_round114_20261008/evaluation_semantics_clarification.json)'
        '明列此界線，不能把MAE說成不需要答案的指標。', '']
    for stage in ORDER:
        v = result['stages'][stage]
        s = read(f'outputs/round114/{stage}/stage.json')
        lines += [f'## {stage}完整診斷', '', f"fit實測{s['fit_seconds']:.3f}秒。", '',
                  '|層|總價MAE|', '|---|---:|']
        for layer, values in v['branch_metrics'].items():
            lines.append(f"|{layer}|{values['overall_mae']:,.2f}|")
        lines.append(f"|median|{v['metrics']['T']['overall_mae']:,.2f}|")
        lines += ['', '未投影總價與固定非負投影：', ''] + block(v['raw_signed_total_diagnostic_only'])
        lines += ['所有季度／至少500列群退步超5%（群可重疊，不能相加）：', ''] + block(
            v['comparison']['quarter_or_large_group_deterioration_above_5pct'])
        lines += ['|條件抽樣|指標|點差vs v22|95%區間|', '|---|---|---:|---|']
        for family, item in v['paired']['methods'].items():
            for name, metric in item['summary'].items():
                lines.append(f"|{family}|{name}|{metric['point_delta']:+,.2f}|"
                    f"[{metric['percentile_2_5']:+,.2f}, {metric['percentile_97_5']:+,.2f}]|")
        lines += ['', '完整條件抽樣與支援數：', ''] + block(v['paired'])
        lines += ['v24完整次對照（不以較弱對照取代v22）：', ''] + block(v['comparison_vs_v24'])
        lines += ['完整城市／月份／新舊案／真價帶／季度／誤差尾部，僅事後診斷，不作未知test路由：', ''] + block(v['metrics']['T'])
    lines += ['## 執行、核驗與限制', '',
        f"四fit流程命令實測{train['elapsed_seconds']:.3f}秒；分開evaluate命令{evaluate['elapsed_seconds']:.3f}秒。"
        '這是程序實測，不是整個專案總工時。四起點各自encoder／group OOF fit prior／full-fit query prior、'
        '日期權重、reference／spatial均依相應外層切點，完整原v22/v24重播一致；不使用後期模型回測。', '',
        '[獨立核驗](../outputs/research_round114_20261008/postfit_review.json)核對全部來源／input／protected、'
        '模型metadata、signed native投影、五層、評分及人工gates。其完整實作範圍和限制如下：', ''] + block(audit.get('limits', []))
    lines += [result['caution'], '',
        '訓練前已固定R91加法CatBoost跨窗與高價群惡化、R104教師offset NN失敗、R102 native改善未保留到完整組合等反證。'
        '本結果繼續反對「改加法就會改善」；沒有事後再搜尋bias、learning rate、clip或融合比例。'
        '四窗口重疊且已被反覆使用，bootstrap不是Private勝率，保存重載也不是乾淨環境端到端重訓。'
        '禁止內政部／外部成交答案，沒有test ID人工改價、Public搜參數或Kaggle操作。', '',
        '- [事前固定規約](round114_plan.md)', '- [作者方法與梯度說明（訓練前狀態）](round114_method_review.md)',
        '- [訓練程式](../src/round114_additive_prior_train.py)', '- [四起點研究流程](../src/round114_research.py)',
        '- [完整機器結果](../outputs/round114/results.json)']
    name = 'reports/round114_report.md'
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({**record(name), 'bytes': (ROOT / name).stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
