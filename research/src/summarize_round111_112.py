"""Render completed R111/R112 results only; never fit, load arrays or score y."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
STAGES=('development','confirmation','auxiliary')
ROUNDS=(111,112)


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def main():
    results={n:read(f'outputs/round{n}/results.json') for n in ROUNDS}
    execution=read('outputs/research_round111_112_20261008/training_execution.json')
    evaluation=read('outputs/research_round111_112_20261008/evaluation_execution.json')
    if execution['exit_code'] or evaluation['exit_code']:
        raise ValueError('Only completed successful training/evaluation may be summarized')
    if any(r['status']!='three_reused_windows_evaluated' or set(r['stages'])!=set(STAGES)
           or r['Private_score'] is not None for r in results.values()):
        raise ValueError('Unexpected result completeness or Private assertion')
    if results[111]['request']!=results[112]['request'] or results[111]['seal']!=results[112]['seal']:
        raise ValueError('Both hypotheses must share the same preregistered batch/seal')
    lines=['# R111／R112：固定架構與單價Ordered研究結果','',
           '2026-10-08。這是課程標籤的時間外離線MAE，單位新臺幣元，越低越好。不是新增Public或Private分數；v22的已回報Public為1,998,568.5706，Private未知。使用者最新要求持續優化至Codex剩餘20%，不用重置或額外點數。','',
           'R111把prior NN換成固定兩個殘差block，仍以日期加權總價MAE訓練正倍率；R112只把原R79 unit/date×面積CatBoost的Plain改Ordered。沒有合併兩候選、搜尋容量／比例／epoch，亦未帶入R107 SmoothL1或R110八邊界。六模型、metadata與五層向量直接封存並核對後，才讀query價格評分。','',
           '|窗口|v22 MAE|v24 MAE|R111 MAE|R111−v22|R112 MAE|R112−v22|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        a=results[111]['stages'][stage]['metrics']
        b=results[112]['stages'][stage]['metrics']
        lines.append(f"|{stage}|{a['v22']['overall_mae']:,.2f}|{a['v24']['overall_mae']:,.2f}|{a['T']['overall_mae']:,.2f}|{a['T']['overall_mae']-a['v22']['overall_mae']:+,.2f}|{b['T']['overall_mae']:,.2f}|{b['T']['overall_mae']-b['v22']['overall_mae']:+,.2f}|")
    lines+=['','|窗口|v22固定加權|v24固定加權|R111加權|差vs22|R112加權|差vs22|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        a=results[111]['stages'][stage]['metrics']
        b=results[112]['stages'][stage]['metrics']
        lines.append(f"|{stage}|{a['v22']['weighted_mae']:,.2f}|{a['v24']['weighted_mae']:,.2f}|{a['T']['weighted_mae']:,.2f}|{a['T']['weighted_mae']-a['v22']['weighted_mae']:+,.2f}|{b['T']['weighted_mae']:,.2f}|{b['T']['weighted_mae']-b['v22']['weighted_mae']:+,.2f}|")
    for n,result in results.items():
        gate=result['gates']
        lines+=['',f'## R{n}完整判定','',
                f"強推薦通過：**{gate['promotion']['adoption_research_gate_passed']}**；探索試交通過：**{gate['exploratory_submission']['passed']}**。兩套門檻事前固定，未因新分數放寬。",'',
                '```json',json.dumps(gate,ensure_ascii=False,indent=2),'```','',
                '|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|',
                '|---|---:|---:|---:|']
        for stage in STAGES:
            item=result['stages'][stage]
            strata=item['comparison']['strata']
            adverse=item['comparison']['quarter_or_large_group_deterioration_above_5pct']
            lines.append(f"|{stage}|{strata['seen']['delta']:+,.2f}|{strata['new_nonmissing']['delta']:+,.2f}|{len(adverse)}|")
        lines+=['','所有超門檻反證（群可能重疊，不可加總）：','']
        for stage in STAGES:
            bad=result['stages'][stage]['comparison']['quarter_or_large_group_deterioration_above_5pct']
            lines.append(f'- {stage}：'+(json.dumps(bad,ensure_ascii=False) if bad else '沒有超門檻季度或大型群。'))
        lines+=['','|窗口|層|MAE|','|---|---|---:|']
        for stage in STAGES:
            item=result['stages'][stage]
            for name,value in item['branch_metrics'].items():
                lines.append(f"|{stage}|{name}|{value['overall_mae']:,.2f}|")
            lines.append(f"|{stage}|median|{item['metrics']['T']['overall_mae']:,.2f}|")
        lines+=['','全部配對條件區間：','',
                '|窗口|抽法|指標|點差|條件摘要|','|---|---|---|---:|---|']
        for stage in STAGES:
            for family,detail in result['stages'][stage]['paired']['methods'].items():
                for name,item in detail['summary'].items():
                    extra={k:v for k,v in item.items() if k not in ('control_mae','candidate_mae','point_delta')}
                    lines.append(f"|{stage}|{family}|{name}|{item['point_delta']:+,.2f}|`{json.dumps(extra,ensure_ascii=False)}`|")
        fits=[read(f'outputs/round{n}/{stage}/stage.json')['fit_seconds'] for stage in STAGES]
        lines+=['',f'R{n}三次原生fit：'+', '.join(f'{value:.3f}' for value in fits)+f'秒，合計{sum(fits):.3f}秒。',
                f'[完整各城市／月份／季度／五價帶與v24反證](../outputs/round{n}/results.json)。']
    lines+=['','## 執行與限制','',
            f"序列主程序{execution['elapsed_seconds']:.3f}秒，已含六次fit／重載／組合；評分{evaluation['elapsed_seconds']:.3f}秒，不能把fit再加到主程序。整段聊天／開發工時未量測。",'',
            'R111為本題LayerNorm／正倍率／加權總價loss的ResNet適配，不是官方完整復刻。模型容量、初始化及亂數消耗同時改變，不能純歸因skip。R112保留R17 raw uniform Ordered的development整體+6.085%、新案+7.435%、Q2+9.205%反證；本輪只測unit/date×面積交互，不是Ordered首次試驗。原native CBM無法恢復的三項runtime控制在metadata另明示，不偽稱已由模型驗出。','',
            '原v22和v24重播是保存模型對照，不是同期matched重訓。固定seed不保證GPU重訓逐bytes相同；保存模型全query重載exact是不同檢查。群組OOF prior非時間OOF，三個年度已多輪重用，不能當新盲測。Bootstrap條件於保存預測，未校正累積選型、重訓或未來移轉，也不是Private勝率。','',
            '不取得成交來源答案、不按test ID或未知價格手改、不以Public搜尋。新CSV仍須門檻、另立全量規約及獨立發行核對；本報告本身不代表已完成fullfit、上傳或最終選定。保存可追溯不等於新環境從原始資料端到端重訓。','',
            '[事前規約](round111_112_plan.md) · [最新要求](../outputs/research_continue_20261008/user_request.json) · [凍結request](../outputs/research_round111_112_20261008/request.json) · [全模型封存](../outputs/research_round111_112_20261008/seal.json)','']
    target=ROOT/'reports/round111_112_report.md'
    with target.open('x',encoding='utf8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'report':target.relative_to(ROOT).as_posix(),
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}),flush=True)


if __name__=='__main__':main()
