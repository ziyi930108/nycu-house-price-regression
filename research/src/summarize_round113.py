"""JSON-only R113 reporting; no price table, arrays, model inference or fitting."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def record(name):
    return {'path':name,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}


def main():
    result=read('outputs/round113/results.json')
    gate=read('outputs/research_round113_20261008/round113_release_gate.json')
    execution=read('outputs/research_round113_20261008/evaluation_execution.json')
    training=read('outputs/research_round113_20261008/training_execution.json')
    if (result['status']!='fixed_extra_origin_evaluated' or set(result['candidates'])!={'107','111','112'}
        or execution['exit_code'] or training['exit_code'] or result['Private_score'] is not None
        or gate['results']!=record('outputs/round113/results.json') or gate['plan']!=result['plan']):
        raise ValueError('Only the complete successful fixed three-method evaluation may be reported')
    lines=['# R113：2月起點的額外穩定性檢查','',
        '2026-10-08。fit截至2023-02-10，272,117列；query為2023-02-11至2024-02-10，107,169列。'
        'TWD MAE越低越好；本頁是本機離線分數，Public／Private沒有新增回報。','',
        '在R111／R112新分數出現前，已固定R107、R111、R112三者全部執行。'
        '原本三個8月切分的門檻保持，額外要求2月起點的整體與固定新舊案加權各不得退步超過0.25%，'
        '且任何季度或至少500列群不得退步超過5%。未達條件就不能發行新v28／v29，'
        '不會因其中一段較好就放寬條件。','',
        '|方法|整體MAE|差vs v22|固定加權MAE|差vs v22|額外穩定性|兩指標都改善|',
        '|---|---:|---:|---:|---:|---|---|']
    first=result['candidates']['107']['metrics']
    for key in ('v22','v24'):
        m=first[key];b=first['v22']
        lines.append(f"|R97/{key}|{m['overall_mae']:,.2f}|{m['overall_mae']-b['overall_mae']:+,.2f}|{m['weighted_mae']:,.2f}|{m['weighted_mae']-b['weighted_mae']:+,.2f}|對照|—|")
    for n in (107,111,112):
        v=result['candidates'][str(n)];m=v['metrics']['T'];b=v['metrics']['v22']
        if gate['candidate_gates'][str(n)]!=v['gate']:
            raise ValueError('Gate/result authority differs')
        lines.append(f"|R{n}|{m['overall_mae']:,.2f}|{m['overall_mae']-b['overall_mae']:+,.2f}|{m['weighted_mae']:,.2f}|{m['weighted_mae']-b['weighted_mae']:+,.2f}|{v['gate']['passed']}|{v['both_point_metrics_improve']}|")
    lines+=['','## 與原三窗門檻合看','',
        '|候選|原三窗探索門檻|本額外門檻|可進入獨立全量發行審查|','|---|---|---|---|']
    for n in (111,112):
        original=read(f'outputs/round{n}/results.json')['gates']['exploratory_submission']['passed']
        extra=result['candidates'][str(n)]['gate']['passed']
        lines.append(f'|R{n}|{original}|{extra}|{original and extra}|')
    lines+=['','表中的True只代表數值門檻；獨立postfit、獨立final preflight、另立全量計畫、'
        '單次全量fit與CSV完整核對仍是必要步驟，不能把準備說成已發行。'
        'v22仍是原選定版本；v25是前輪R107已保存的探索檔，本輪只新增後續反證，'
        '不改其既有release或價格。R113本身不產生CSV，也不操作Kaggle。']
    for n in (107,111,112):
        v=result['candidates'][str(n)]
        stage=read(f'outputs/round113/R{n}/stage.json')
        lines+=['',f'## R{n}完整診斷','',f"本次實測fit {stage['fit_seconds']:.3f}秒。",'',
            '```json',json.dumps(v['gate'],ensure_ascii=False,indent=2),'```','',
            '|層|MAE|','|---|---:|']
        for key,m in v['branch_metrics'].items():lines.append(f"|{key}|{m['overall_mae']:,.2f}|")
        lines.append(f"|median|{v['metrics']['T']['overall_mae']:,.2f}|")
        lines+=['','所有超5%季度／大型群（群可能重疊，不能相加）：','',
            '```json',json.dumps(v['comparison']['quarter_or_large_group_deterioration_above_5pct'],ensure_ascii=False,indent=2),'```','',
            '|條件抽樣|指標|點差vs v22|95%區間|','|---|---|---:|---|']
        for family,item in v['paired']['methods'].items():
            for name,s in item['summary'].items():
                interval=f"[{s['percentile_2_5']:+,.2f}, {s['percentile_97_5']:+,.2f}]"
                lines.append(f"|{family}|{name}|{s['point_delta']:+,.2f}|{interval}|")
        lines+=['','完整paired summary與支援數：','',
            '```json',json.dumps(v['paired'],ensure_ascii=False,indent=2),'```','',
            '與較強／較弱次對照v24的完整差異：','',
            '```json',json.dumps(v['comparison_vs_v24'],ensure_ascii=False,indent=2),'```','',
            '真價帶與誤差尾部（僅診斷，不能拿未知test真價路由）：','',
            '```json',json.dumps({k:v['metrics']['T'][k] for k in ('strata','forecast_quarters','price_bands','tails')},ensure_ascii=False,indent=2),'```']
    lines+=['','## 執行與可解釋範圍','',
        f"三模型串行流程實測 {training['elapsed_seconds']:.3f}秒；獨立評分命令實測 {execution['elapsed_seconds']:.3f}秒。這些是各程序的測量，不是整個專案總工時。",'',
        '原R97四分支、兩final及兩個槽的v22還原全部bytes一致。encoder、group-OOF fit先驗、'
        'full-fit query先驗、日期權重、reference與spatial使用此起點自己的保存狀態，'
        '不用後期或全量模型回測。三個新模型、metadata及五層向量全部直接封存，'
        '確認成功退出後才開raw價格表作評分；原fit-label隔離擷取器曾解開整張價格pickle的歷史限制仍保留。','',
        result['caution'],'',
        '這個窗口早已用過v22／v24，並與8月窗口重疊；不能說新盲測、四個獨立樣本或Private勝率。'
        '固定seed不包含重訓變異，保存模型可重播也不等於乾淨環境從原始資料端到端重訓。'
        '沒有查內政部答案、外部成交答案、test ID人工改價或Public搜參數。','',
        '- [事前固定計畫](round113_plan.md)',
        '- [原三窗結果](round111_112_report.md)',
        '- [機器結果](../outputs/round113/results.json)',
        '- [額外發行數值門檻](../outputs/research_round113_20261008/round113_release_gate.json)']
    path=ROOT/'reports/round113_report.md'
    with path.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'path':path.relative_to(ROOT).as_posix(),'sha256':record(path.relative_to(ROOT).as_posix())['sha256'],'bytes':path.stat().st_size}))


if __name__=='__main__':main()
