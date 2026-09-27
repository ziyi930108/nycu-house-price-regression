"""Posthoc R72/v19 road-coverage diagnostic, saved vectors only; no fit/predict."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round72/road_coverage_review.json'
REPORT=ROOT/'reports/round72_road_coverage_review.md'
STAGES=('development','confirmation','auxiliary')
KT='outputs/round6/geography/development/train_source_cache_indices.npy'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def value_sha(a,dtype):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-7)


def write(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


def run():
    tick=time.perf_counter();started=stamp();inputs={}
    assert all(not p.exists() for p in (OUT,OUT.with_suffix('.sha256'),REPORT,REPORT.with_suffix('.sha256'))),'No overwrite'
    def bind(name,expected=None):
        path=ROOT/name;actual=sha(path)
        assert expected is None or actual==expected,name
        assert name not in inputs or inputs[name]==actual,name
        inputs[name]=actual;return path
    def locked(name):
        receipt=bind(str(Path(name).with_suffix('.sha256')).replace('\\','/'))
        return json.loads(bind(name,receipt.read_text(encoding='ascii').strip()).read_text(encoding='utf-8'))
    def array(name,expected):
        return np.load(bind(name,expected),allow_pickle=False)
    p48=locked('outputs/round48/plan.json')
    r72=locked('outputs/round72/results.json');r71=locked('outputs/round71/results.json')
    assert r72['status']==r71['status']=='three_periods_completed_no_adoption_decision'
    assert r72['checked_stages']==r71['checked_stages']==list(STAGES)
    source={str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/'):sha(Path(__file__))}
    for name in ('src/round39_city_road_experiment.py','src/round72_nn_experiment.py'):
        source[name]=sha(ROOT/name);assert source[name]==r72['source_sha256'][name]
    # Exact target-only projection of R39 load_training_rows: raw cache -> saved canonical K.
    raw=pd.read_pickle(bind('data_cache/train.pkl',p48['input_sha256']['data_cache/train.pkl']))
    k=array(KT,p48['input_sha256'][KT])
    assert len(raw)==455784 and raw.index.equals(pd.RangeIndex(455784))
    assert k.dtype==np.int64 and k.shape==(455255,) and (np.diff(k)>0).all() and k.min()>=0 and k.max()<len(raw)
    targets=raw['總價元'].to_numpy(dtype=np.float64)[k];del raw
    assert np.isfinite(targets).all() and (targets>0).all()
    result={'round':72,'status':'posthoc_saved_prediction_coverage_review','candidate':'R72','control':'v19=R71_seed42',
        'group_rule':{'supported':'saved query_road_support > 0','fallback':'saved query_road_support == 0'},
        'new_model_fits':0,'model_predictions':0,'reference_predictions':0,'spatial_predictions':0,
        'new_candidate_or_threshold_search':False,'post_selection_reused_windows':True,'stages':{},
        'read_scope':'Canonical query targets from local train cache; saved R72/v19 predictions, query support and small preparation summaries only. No road refit, model, feature cache, query delta or train support array read.',
        'canonical_target_alignment':'train.pkl[總價元].float64[saved train_source_cache_indices][saved stage query_indices]',
        'limitations':['Coverage groups are not randomized; no causal attribution to road information.',
            'Train OOF and query full-fit contain different populations; coverage gap is not solely an OOF/full-fit effect.',
            'Selected support is raw rows, not independent projects or ESS; delta zero is not equivalent to fallback.',
            'A retrained wider NN can change fallback-row predictions too; this diagnosis defines no new row selection rule.']}
    for stage in STAGES:
        a,b=r72['stages'][stage],r71['stages'][stage];contract=p48['split_contract'][stage];n=contract['query_rows']
        art={name.replace('\\','/'):h for name,h in a['artifact_sha256'].items()}
        art71={name.replace('\\','/'):h for name,h in b['artifact_sha256'].items()}
        qname=f'outputs/round72/{stage}/query_indices.npy';qi=array(qname,art[qname])
        bqname=f'outputs/round71/{stage}/query_indices.npy';bqi=array(bqname,art71[bqname])
        assert qi.dtype==bqi.dtype==np.int64 and qi.shape==(n,) and np.array_equal(qi,bqi)
        assert (np.diff(qi)>0).all() and qi.min()>=0 and qi.max()<len(targets)
        assert value_sha(qi,'<i8')==contract['query_indices_content_sha256']
        name=f'outputs/round72/{stage}/query_road_support.npy';support=array(name,art[name])
        name=f'outputs/round72/{stage}/predictions.npy';candidate=array(name,a['prediction_sha256'])
        assert art[name]==a['prediction_sha256']
        name=f'outputs/round71/{stage}/predictions.npy';control=array(name,b['prediction_sha256'])
        assert r72['input_sha256'][name]==b['prediction_sha256']
        assert support.dtype==np.int64 and support.shape==(n,) and ((support==0)|(support>=30)).all()
        assert all(v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all() for v in (candidate,control))
        prepname=f'outputs/round72/{stage}/preparation.json';prep=locked(prepname)
        assert inputs[prepname]==art[prepname] and prep==a['preparation']
        assert prep['selected_support_sha256']['query']==value_sha(support,'<i8')
        y=targets[qi];ec=np.abs(candidate-y);eb=np.abs(control-y)
        cm,bm=float(ec.mean()),float(eb.mean())
        assert close(cm,a['metrics']['R72']['overall_mae'])
        assert close(bm,a['metrics']['v19']['overall_mae']) and close(bm,b['metrics']['R71']['overall_mae'])
        groups={}
        for label,mask in (('supported',support>0),('fallback',support==0)):
            count=int(mask.sum());bc=None if not count else float(eb[mask].mean());cc=None if not count else float(ec[mask].mean())
            groups[label]={'rows':count,'coverage_fraction':count/n,'v19_MAE_TWD':bc,'R72_MAE_TWD':cc,
                'MAE_delta_R72_minus_v19_TWD':None if not count else cc-bc,
                'overall_delta_contribution_TWD':float((ec[mask]-eb[mask]).sum(dtype=np.float64)/n)}
        assert sum(g['rows'] for g in groups.values())==n
        contribution=sum(g['overall_delta_contribution_TWD'] for g in groups.values())
        assert close(contribution,cm-bm)
        assert close(sum(g['rows']*(g['R72_MAE_TWD'] or 0) for g in groups.values())/n,cm)
        assert close(sum(g['rows']*(g['v19_MAE_TWD'] or 0) for g in groups.values())/n,bm)
        coverage={}
        for split,rows in (('train',contract['fit_rows']),('query',n)):
            summary=prep['road_feature_summary'][split]
            assert summary['rows']==rows
            coverage[split]={'population':'fit group-fold OOF' if split=='train' else 'separate full-fit query population',
                'rows':rows,'selected_rows':summary['selected_support_positive_rows'],
                'coverage_fraction':summary['selected_support_positive_rows']/rows,
                'saved_summary':summary,'delta_quantile_probabilities':[0,.1,.5,.9,1]}
        assert coverage['query']['selected_rows']==groups['supported']['rows']
        result['stages'][stage]={'rows':n,'query_indices_sha256':value_sha(qi,'<i8'),'query_target_sha256':value_sha(y,'<f8'),
            'v19_MAE_TWD':bm,'R72_MAE_TWD':cm,'overall_delta_R72_minus_v19_TWD':cm-bm,'groups':groups,
            'coverage_and_delta_from_saved_preparation':coverage,
            'rows_reconcile':True,'weighted_group_MAEs_reconcile_to_overall':True,'delta_contributions_reconcile':True,
            'saved_overall_MAEs_matched':True,'contribution_roundoff_TWD':contribution-(cm-bm)}
    result.update(source_sha256=source,input_sha256=inputs,execution={'exit_code':0,'started_at_utc':started,
        'completed_at_utc':stamp(),'elapsed_seconds':time.perf_counter()-tick})
    lines=['# R72 道路 prior 覆蓋診斷','',
        '這是固定 support>0／support==0 的事後保存預測診斷，差值均為 R72−v19，負值較好。0 fit、0 模型/reference/spatial 推論；沒有重新建立道路特徵，也不產生挑列規則。','',
        '| 年度 | 群組 | 列數 | 占全 query | v19 MAE | R72 MAE | 群組 MAE 差（元） | 對整体 MAE 差貢獻（元） |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    labels={'development':'development','confirmation':'confirmation','auxiliary':'auxiliary'}
    for stage,a in result['stages'].items():
        for label,g in a['groups'].items():
            fmt=lambda v:'—' if v is None else f'{v:,.2f}'
            lines.append(f"| {labels[stage]} | {label} | {g['rows']:,} | {100*g['coverage_fraction']:.4f}% | {fmt(g['v19_MAE_TWD'])} | {fmt(g['R72_MAE_TWD'])} | {fmt(g['MAE_delta_R72_minus_v19_TWD'])} | {g['overall_delta_contribution_TWD']:+,.2f} |")
        lines.append(f"| {stage} | **全部** | {a['rows']:,} | 100% | {a['v19_MAE_TWD']:,.2f} | {a['R72_MAE_TWD']:,.2f} | {a['overall_delta_R72_minus_v19_TWD']:+,.2f} | {a['overall_delta_R72_minus_v19_TWD']:+,.2f} |")
    lines+=['','兩群列數、列數加權 MAE 與誤差貢獻均回到全體；保存的兩組 overall MAE 亦核對吻合。','',
            '| 年度 | train OOF 命中 | query full-fit 命中 | train delta P0/P10/P50/P90/P100 | query 同分位數 |',
            '|---|---:|---:|---|---|']
    for stage,a in result['stages'].items():
        tr,qu=(a['coverage_and_delta_from_saved_preparation'][s] for s in ('train','query'))
        quant=lambda s:', '.join(f'{x:.6g}' for x in s['saved_summary']['delta_quantiles'])
        lines.append(f"| {stage} | {tr['selected_rows']:,}/{tr['rows']:,} ({100*tr['coverage_fraction']:.4f}%) | {qu['selected_rows']:,}/{qu['rows']:,} ({100*qu['coverage_fraction']:.4f}%) | {quant(tr)} | {quant(qu)} |")
    lines+=['','覆蓋率與分位數直接引自已存 preparation；沒有重掃 train support 或 delta 陣列。Train 與 query 原本就是不同年份／人群，且 OOF 與 full-fit reference 規模不同，不能把覆蓋差全歸因於 OOF。',
        '命中依 support>0，不依 delta!=0；支持數不是獨立建案數或 ESS。更寬的 NN 重新訓練可能影響未命中列，分組誤差不是道路欄的因果效果。',
        '歷史窗口已反覆用於選型，本診斷不代表 Private 勝率，不增加新門檻、城市／價位／test-mix 分析或局部修正。','',
        f"執行：exit 0，{result['execution']['elapsed_seconds']:.3f} 秒；完整 source/input SHA、列序及點估計核對在 [JSON](../outputs/round72/road_coverage_review.json)。",
        '重跑程式：[round72_road_coverage_review.py](../src/round72_road_coverage_review.py)；既有輸出拒覆寫。']
    assert source[str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/')]==sha(Path(__file__))
    write(REPORT,'\n'.join(lines)+'\n');result['report_sha256']=sha(REPORT)
    write(OUT,json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['execution']['elapsed_seconds'],
        'results_sha256':sha(OUT),'report_sha256':sha(REPORT),
        'stages':{s:{'overall_delta_TWD':a['overall_delta_R72_minus_v19_TWD'],'groups':a['groups']} for s,a in result['stages'].items()}},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args();run()

