"""Pure aggregate reporting; no price/array/model readers or fits."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('development', 'confirmation', 'auxiliary', 'offset_2023_02_11')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def median(item):
    return item['quantiles']['0.5']


def main():
    r = read('outputs/round115/results.json')
    audit = read('outputs/research_round115_20261008/postfit_review.json')
    if (r['status'] != 'all_four_origins_pure_X_audit_completed' or set(r['stages']) != set(ORDER)
            or r['new_fits'] or r['query_labels_read'] or r['NPZ_opened'] or r['MAE_calls']
            or audit['status'] != 'passed' or audit['results'] != {'path': 'outputs/round115/results.json', 'sha256': sha('outputs/round115/results.json')}):
        raise ValueError('Completed independently reviewed pure-X audit required')
    stages = {}
    for stage in ORDER:
        rec = r['stages'][stage]
        if sha(rec['path']) != rec['sha256']:
            raise ValueError('Original aggregate changed')
        stages[stage] = read(rec['path'])
    lines = ['# R115：分折政策的純特徵支持診斷', '',
        '2026-10-08。四個起點全部完成；本研究沒有讀逐筆價格、NPZ、模型或預測，沒有訓練回歸、MAE或bootstrap。'
        '既有權威JSON只作檔案來源與日期權重身分核對，可能含舊整體分數，不將其用作本輪統計。'
        '**此報告沒有模型改善分數。**', '',
        '問題是：原先驗對fit列排除整個建案group fold，query卻使用完整fit。若改成排除自身transaction fold，'
        '原四層行政區／交易類別／建物形態cell的支持、同案暴露與集中程度會有什麼差異。'
        '兩者都用相同outer-fit，query參考集合完全相同；沒有新增建案或道路價格cell。', '',
        '|起點|fit列|改用不同層級|比例|轉細|轉粗|所選支持比中位（交易／建案）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage, v in stages.items():
        finer = coarser = 0
        for key, cell in v['selected_level_transitions'].items():
            a, b = map(int, key.split('->'))
            finer += cell['rows'] if b < a else 0
            coarser += cell['rows'] if b > a else 0
        changed = finer + coarser
        lines.append(f"|{stage}|{v['fit_rows']:,}|{changed:,}|{changed/v['fit_rows']:.4%}|{finer:,}|{coarser:,}|"
                     f"{median(v['raw_support_transaction_over_group']):.5f}|")
    lines += ['', '這不是單純把每個cell筆數乘1.25。部分列改到更細層，其所選cell支持反而較小，'
        '少數也轉粗；不能說交易分折讓每列支持都增加。原始raw四層支持及所有空轉移格都保留。', '',
        '|起點|group-fit支持中位|交易fit支持中位|query支持中位|group-fit案質量ESS中位|交易fit ESS中位|query ESS中位|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage, v in stages.items():
        values = [v[k]['selected_support_and_exposure'] for k in ('group_fit', 'transaction_fit', 'full_fit_query')]
        cells = [f"{median(x['raw_rows']):,.0f}" for x in values] + [f"{median(x['date_weight_project_ESS']):.3f}" for x in values]
        lines.append('|' + stage + '|' + '|'.join(cells) + '|')
    lines += ['', 'ESS在此只量化保存outer-fit日期權重按project/fallback群加總後的集中程度，'
        '不是獨立樣本數、先驗置信度或逐fold重算的原R60中位數權重。raw支持30筆也可能只來自一案。', '',
        '|起點|交易fit同cluster參考peers均值|交易fit最大案日期質量占比中位|query已見／新案／缺名|query新案占比|',
        '|---|---:|---:|---|---:|']
    for stage, v in stages.items():
        t = v['transaction_fit']['selected_support_and_exposure']
        counts = v['query_by_project_status']
        display = '／'.join(f"{counts[x]['rows']:,}" for x in ('seen', 'new_nonmissing', 'missing'))
        lines.append(f"|{stage}|{t['same_cluster_peers_reference']['mean']:.3f}|{median(t['largest_project_date_mass_share']):.3%}|"
                     f"{display}|{counts['new_nonmissing']['rows']/v['query_rows']:.3%}|")
    lines += ['', '原group-fit同cluster peers全部為0。交易fold排除本列不代表排除同案或同批近複本，'
        '訓練參考看起來更像已見案，而query仍多數是新案，是強烈的過擬合反證。'
        'seen/new沿原city|district|project會員定義；缺名fallback群不是認證實體建案。', '',
        '## 完整層級轉移與query分群', '',
        '層0至3為原四層cell，4為global；global沒有30筆門檻。各起點25格均列出，包括0。']
    for stage, v in stages.items():
        lines += ['', f'### {stage}', '', '|group層→交易層|fit列|保存日期質量|', '|---|---:|---:|']
        for a in range(5):
            for b in range(5):
                cell = v['selected_level_transitions'][f'{a}->{b}']
                lines.append(f"|{a}→{b}|{cell['rows']:,}|{cell['saved_date_weight_mass']:.6f}|")
        lines += ['', '|query城市|列|支持中位|案質量ESS中位|同cluster peers均值|', '|---|---:|---:|---:|---:|']
        for city, group in v['query_by_city'].items():
            m = group['selected_support_and_exposure']
            lines.append(f"|{city}|{group['rows']:,}|{median(m['raw_rows']):,.0f}|{median(m['date_weight_project_ESS']):.3f}|"
                         f"{m['same_cluster_peers_reference']['mean']:.3f}|")
        lines += ['', '|query相對季度|列|支持中位|案質量ESS中位|', '|---|---:|---:|---:|']
        for quarter, group in v['query_by_forecast_quarter'].items():
            m = group['selected_support_and_exposure']
            lines.append(f"|{quarter}|{group['rows']:,}|{median(m['raw_rows']):,.0f}|{median(m['date_weight_project_ESS']):.3f}|")
        lines += ['', f"此origin實測{v['elapsed_seconds']:.3f}秒；[完整聚合JSON](../{r['stages'][stage]['path']})含"
                  '所有fold/reference cell、raw支持與集中度的min/mean/max、固定分位數、新舊案與缺名分組。']
    lines += ['', '## 接續判斷與限制', '',
        '約3–5%路由改變使這次不同於規約中「僅同層多案近比例」的不測情形，'
        '可支持另立一次固定交易fold先驗比較；這只允許提出假說，不能從純X推論價格中位數改變、'
        'NN效果或Private優勢。後續必須保留原R63 ratio/loss/架構、query fullfit prior與其他八array，'
        '只有fit factor變動並按原公式連動initial bias，不混其他成功／失敗變因。', '',
        '原144num已沿R47→R48→R52→R60→R63來源核對為72純X變換＋72非有限旗標，沒有prior派生numeric欄。'
        '價格先驗本身仍含fit標籤，不能因本次用純X診斷就稱後續先驗模型為純X。', '',
        'R67支持欄跨窗不穩定、R100時間先驗失敗、R65／R102 native改善未穩定傳到完整median等反證保持。'
        '窗口重疊且已反覆研究，沒有新的盲測、Private資料或獨立樣本認證。'
        '完整純X快取455,784列曾進記憶體，只有固定起點的canonical列參與統計；不能聲稱只解出fit列。', '',
        f"四origin流程實測{r['elapsed_seconds']:.3f}秒。獨立核驗核身份、聚合一致性及人工算術，"
        '不冒稱另以獨立演算法重算全部真資料逐筆統計。原模型與source不改，v22及v25角色保持，'
        '沒有Kaggle／外部成交答案／test ID調價。', '',
        '- [事前規約](round115_plan.md)', '- [診斷程式](../src/round115_prior_support_x_audit.py)',
        '- [獨立結果核驗](../outputs/research_round115_20261008/postfit_review.json)',
        '- [完整結果入口](../outputs/round115/results.json)']
    path = ROOT / 'reports/round115_report.md'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path.relative_to(ROOT).as_posix()), 'bytes': path.stat().st_size}))


if __name__ == '__main__':
    main()
