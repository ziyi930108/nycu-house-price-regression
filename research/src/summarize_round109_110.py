"""Summarize already scored R109/R110 JSON, without models/arrays/labels."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))

def main():
    results={n:read(f'outputs/round{n}/results.json') for n in (109,110)}
    execution=read('outputs/research_round109_110_20261008/training_execution.json')
    evaluation=read('outputs/research_round109_110_20261008/evaluation_execution.json')
    lines=['# 2026-10-08：R109／R110房價模型研究結果','',
      '使用者重新授權模型研究與新試交。v22及上台材料保留，原Public為1,998,568.5706元；Private未知。以下都是重用歷史時間窗口的離線MAE，不能當作新Kaggle成績。','',
      'R109只對原日期加權MAE prior NN末10epoch作參數平均；R110只在原max_ctr_complexity=2的單價CatBoost新增全域ctr_target_border_count=8。兩候選沒有合併或掃參數，全部六模型／三窗向量封存後才評分。','',
      '|窗口|v22 MAE|v24 MAE|R109 MAE|R109−v22|R110 MAE|R110−v22|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for s in ('development','confirmation','auxiliary'):
        a,b=results[109]['stages'][s],results[110]['stages'][s]
        m=a['metrics'];n=b['metrics']
        lines.append(f"|{s}|{m['v22']['overall_mae']:,.2f}|{m['v24']['overall_mae']:,.2f}|{m['T']['overall_mae']:,.2f}|{m['T']['overall_mae']-m['v22']['overall_mae']:+,.2f}|{n['T']['overall_mae']:,.2f}|{n['T']['overall_mae']-n['v22']['overall_mae']:+,.2f}|")
    lines+=['','|窗口|v22加權|v24加權|R109加權|差vs22|R110加權|差vs22|','|---|---:|---:|---:|---:|---:|---:|']
    for s in ('development','confirmation','auxiliary'):
        m=results[109]['stages'][s]['metrics'];n=results[110]['stages'][s]['metrics']
        lines.append(f"|{s}|{m['v22']['weighted_mae']:,.2f}|{m['v24']['weighted_mae']:,.2f}|{m['T']['weighted_mae']:,.2f}|{m['T']['weighted_mae']-m['v22']['weighted_mae']:+,.2f}|{n['T']['weighted_mae']:,.2f}|{n['T']['weighted_mae']-n['v22']['weighted_mae']:+,.2f}|")
    for n,r in results.items():
        g=r['gates'];lines+=['',f'## R{n}判定','',
            f"強推薦條件通過：**{g['promotion']['adoption_research_gate_passed']}**；探索提交條件通過：**{g['exploratory_submission']['passed']}**。兩套條件已事前分開，不因結果放寬。",'',
            '```json',json.dumps(g,ensure_ascii=False,indent=2),'```','',
            '|窗口|既有建案MAE差|新建案MAE差|超5%退步群數|','|---|---:|---:|---:|']
        for s,v in r['stages'].items():
            st=v['comparison']['strata'];bad=v['comparison']['quarter_or_large_group_deterioration_above_5pct']
            lines.append(f"|{s}|{st['seen']['delta']:+,.2f}|{st['new_nonmissing']['delta']:+,.2f}|{len(bad)}|")
        lines+=['','### 全部退步門檻反證','']
        for s,v in r['stages'].items():
            bad=v['comparison']['quarter_or_large_group_deterioration_above_5pct']
            lines.append(f'- {s}：'+(json.dumps(bad,ensure_ascii=False) if bad else '沒有超5%的季度或大型群退步。'))
        if n==109:
            lines+=['','### 同軌跡epoch30對照','',
                    '|窗口|epoch30 MAE|平均參數MAE|平均−epoch30|epoch30與保存v22最大預測差|',
                    '|---|---:|---:|---:|---:|']
            for s,v in r['stages'].items():
                cm=v['same_trajectory_epoch30_metrics']['overall_mae'];am=v['metrics']['T']['overall_mae']
                lines.append(f"|{s}|{cm:,.2f}|{am:,.2f}|{am-cm:+,.2f}|{v['epoch30_vs_saved_v22_max_abs_prediction_difference_TWD']:,.2f}|")
            lines+=['','平均模型與epoch30來自同一次訓練；這個配對機制診斷不是另選一個epoch或融合比例。對保存v22的差仍可能含GPU重訓變化，不能保證固定seed逐bytes相同。']
        lines+=['','### 五層診斷','', '|窗口|層|MAE|','|---|---|---:|']
        for s,v in r['stages'].items():
            for k,m in v['branch_metrics'].items():lines.append(f"|{s}|{k}|{m['overall_mae']:,.2f}|")
            lines.append(f"|{s}|median|{v['metrics']['T']['overall_mae']:,.2f}|")
        lines+=['','### 配對區間（全部12個）','',
                '|窗口|抽法|指標|點差|95%條件區間|','|---|---|---|---:|---|']
        for s,v in r['stages'].items():
            for family,detail in v['paired']['methods'].items():
                for metric,item in detail['summary'].items():
                    # Keep exact native summary key names in the JSON evidence.
                    lines.append(f"|{s}|{family}|{metric}|{item['point_delta']:+,.2f}|`{json.dumps({k:x for k,x in item.items() if k not in ('control_mae','candidate_mae','point_delta')},ensure_ascii=False)}`|")
        fits=[read(f'outputs/round{n}/{s}/stage.json')['fit_seconds'] for s in ('development','confirmation','auxiliary')]
        lines+=['',f'R{n}三次原生fit秒數：'+', '.join(f'{x:.3f}' for x in fits)+f'；合計{sum(fits):.3f}秒。',
                f'[完整分群與五價帶結果](../outputs/round{n}/results.json)。分群互相重疊，不能加總；真價帶只診斷，不作test路由。']
    lines+=['','## 時間與限制','',
        f"序列訓練主程序{execution['elapsed_seconds']:.3f}秒（已含六次fit與重播／保存），獨立評分{evaluation['elapsed_seconds']:.3f}秒；不是整段聊天／開發工時，fit不重複加總。",'',
        '原v22完整鏈與原v24指標已精確重播；新模型全部query保存重載逐bytes相同，原CSV／selection／登錄及保護檔保持。R110沿用前輪審過的原生參數序列化比較：有限float用rtol1e-7，其餘型別／值精確，兩組診斷與微差完整保存；價格向量仍要求bytes精確。實際CTR邊界使用摘要是結構證據，不是泛化證據，未要求八個邊界全部入樹。','',
        '窗口多輪參與選型；單seed且保存v22不是同期matched重訓，小改善不能證明獨立因果效果。區塊bootstrap只條件於本次預測，沒有校正累積選型／未來分布／重訓變化，不能稱Private勝率。原fit與query狀態、標籤隔離不代表全新未見驗證。v24離線較強的反證完整保留。沒有對test ID／未知真價手改價格，也沒有查成交來源。','',
        '[事前計畫](round109_110_plan.md) · [恢復要求](../outputs/research_resume_20261008/user_request.json) · [凍結request](../outputs/research_round109_110_20261008/request.json) · [全模型預測封存](../outputs/research_round109_110_20261008/seal.json)','']
    p=ROOT/'reports/round109_110_report.md'
    with p.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps({'report':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
