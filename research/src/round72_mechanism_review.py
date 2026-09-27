"""R72 fixed support/delta mechanism diagnosis: saved arrays, no fit or prediction."""
from pathlib import Path
import argparse, json, time, traceback
import numpy as np
import pandas as pd
import round3_evaluate as scorer
from round72_road_coverage_review import ROOT,KT,STAGES,sha,value_sha,close,stamp,write

OUT=ROOT/'outputs/round72/mechanism_review.json'
PLAN_PATH=ROOT/'outputs/round72/mechanism_review_plan.json'
EXECUTION=ROOT/'outputs/round72/mechanism_review_execution.json'
REPORT=ROOT/'reports/round72_mechanism_review.md'
GROUPS=('fallback','supported_zero_delta','supported_nonzero_delta')
CROSS_STRATA=('new_nonmissing','seen')
PLAN={'round':72,'diagnostic':'saved support/delta mechanism review','posthoc':True,
      'fixed_before_this_diagnostic_statistics':True,'groups':{
          'fallback':'saved query support == 0',
          'supported_zero_delta':'saved support > 0 and saved float32 delta == 0',
          'supported_nonzero_delta':'saved support > 0 and saved float32 delta != 0'},
      'strata':'original round3_evaluate.group_labels relative to each period fit X',
      'forecast_quarters':scorer.QUARTERS,'quarter_cross_strata':list(CROSS_STRATA),
      'empty_cells_retained':True,'missing_project_stratum':'retained separately for exhaustive reconciliation',
      'pairs':{'complete_pipeline':['R72','v19=R71_seed42'],'native_prior':['R72_prior','R63_prior']},
      'delta_orientation':'candidate minus control; negative is lower MAE',
      'native_saved_check_scope':'only overall native MAEs were previously saved; native subgroups are new diagnostics',
      'new_fits':0,'model_predict':0,'reference_predict':0,'spatial_predict':0,'feature_reconstruction':False,
      'no_seed_threshold_or_local_rule_search':True,'query_labels_used_only_for_scoring':True,
      'limitations':['Groups are descriptive, not randomized causal comparisons.',
          'The trained model received one numeric delta, not a separately tested hit indicator.',
          'The wider retrained NN may change fallback rows as well.',
          'Support is raw rows, not independent projects or ESS; exact float32 zero is the fixed rule.',
          'Windows were repeatedly reused; no Private win probability or new candidate is established.']}


def run(plan):
    inputs={}
    def bind(name,expected=None):
        path=ROOT/name;actual=sha(path)
        assert expected is None or actual==expected,name
        assert name not in inputs or inputs[name]==actual,name
        inputs[name]=actual;return path
    def locked(name):
        receipt=bind(str(Path(name).with_suffix('.sha256')).replace('\\','/'))
        return json.loads(bind(name,receipt.read_text(encoding='ascii').strip()).read_text(encoding='utf-8'))
    def array(record,name):
        mapping={k.replace('\\','/'):v for k,v in record['artifact_sha256'].items()}
        return np.load(bind(name,mapping[name]),allow_pickle=False)
    p48=locked('outputs/round48/plan.json')
    records={r:locked(f'outputs/round{r}/results.json') for r in (72,71,63)}
    assert all(r['checked_stages']==list(STAGES) and r['status']=='three_periods_completed_no_adoption_decision' for r in records.values())
    for name in ('src/round3_evaluate.py','src/round39_city_road_experiment.py','src/round72_nn_experiment.py'):
        assert sha(ROOT/name)==records[72]['source_sha256'][name]
    raw=pd.read_pickle(bind('data_cache/train.pkl',p48['input_sha256']['data_cache/train.pkl']))
    Xraw=pd.read_pickle(bind('data_cache/train_features.pkl',p48['input_sha256']['data_cache/train_features.pkl']))
    k=np.load(bind(KT,p48['input_sha256'][KT]),allow_pickle=False)
    assert len(raw)==len(Xraw)==455784 and raw.index.equals(pd.RangeIndex(455784)) and Xraw.index.equals(raw.index)
    assert k.dtype==np.int64 and k.shape==(455255,) and (np.diff(k)>0).all() and k.min()>=0 and k.max()<len(raw)
    Xall=Xraw.iloc[k].reset_index(drop=True);yall=raw['總價元'].to_numpy(dtype=np.float64)[k];del raw,Xraw
    assert np.isfinite(yall).all() and (yall>0).all()
    result={'round':72,'status':'posthoc_saved_vector_mechanism_review','plan':plan,'stages':{},
            'new_fits':0,'model_predict_calls':0,'reference_predict_calls':0,'spatial_predict_calls':0,
            'feature_reconstruction':False,'new_candidate_created':False,'source_sha256':plan['source_sha256'],
            'plan_sha256':sha(PLAN_PATH),'read_scope':'Canonical X/y, saved full/native predictions, query support/delta and small authority JSON only; no model/state/reference/geographic objects.'}
    for stage in STAGES:
        a,b,c=(records[r]['stages'][stage] for r in (72,71,63));contract=p48['split_contract'][stage];ix={}
        for kind in ('fit','query'):
            name=f'outputs/round72/{stage}/{kind}_indices.npy';values=array(a,name)
            assert values.dtype==np.int64 and values.shape==(contract[kind+'_rows'],) and (np.diff(values)>0).all()
            assert values.min()>=0 and values.max()<len(Xall) and value_sha(values,'<i8')==contract[kind+'_indices_content_sha256']
            for r,item in ((71,b),(63,c)):
                assert np.array_equal(values,array(item,f'outputs/round{r}/{stage}/{kind}_indices.npy'))
            ix[kind]=values
        fi,qi=ix['fit'],ix['query'];assert not np.intersect1d(fi,qi).size
        X,Xq=Xall.iloc[fi].reset_index(drop=True),Xall.iloc[qi].reset_index(drop=True);y=yall[qi];n=len(y)
        labels=scorer.group_labels(X,Xq)
        day0=(pd.Timestamp(p48['splits'][stage][0])-pd.Timestamp('2000-01-01')).days
        horizon=Xq['trade_day_index'].to_numpy(dtype=np.float64)-day0
        assert np.isfinite(horizon).all() and (horizon>=0).all()
        quarters={q:(horizon>=lo)&(True if hi is None else horizon<=hi) for q,(lo,hi) in scorer.QUARTERS.items()}
        assert np.all(sum(m.astype(np.int8) for m in quarters.values())==1)
        support=array(a,f'outputs/round72/{stage}/query_road_support.npy')
        delta=array(a,f'outputs/round72/{stage}/query_road_delta.npy')
        assert support.dtype==np.int64 and support.shape==(n,) and ((support==0)|(support>=30)).all()
        assert delta.dtype==np.float32 and delta.shape==(n,) and np.isfinite(delta).all() and (delta[support==0]==0).all()
        masks={'fallback':support==0,'supported_zero_delta':(support>0)&(delta==0),
               'supported_nonzero_delta':(support>0)&(delta!=0)}
        assert np.all(sum(m.astype(np.int8) for m in masks.values())==1)
        prep=a['preparation'];assert value_sha(support,'<i8')==prep['selected_support_sha256']['query']
        assert value_sha(delta,'<f4')==prep['added_feature_sha256']['query']
        vectors={'complete_pipeline':(array(a,f'outputs/round72/{stage}/predictions.npy'),
                                       array(b,f'outputs/round71/{stage}/predictions.npy')),
                 'native_prior':(array(a,f'outputs/round72/{stage}/model/predictions_twd.npy'),
                                 array(c,f'outputs/round63/{stage}/model/predictions_twd.npy'))}
        assert all(v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all() and (v>0).all()
                   for pair in vectors.values() for v in pair)
        errors={pair:tuple(np.abs(v-y) for v in values) for pair,values in vectors.items()}
        def part(pair,mask,scope_rows):
            ec,eb=errors[pair];count=int(mask.sum())
            sa,sb=float(ec[mask].sum(dtype=np.float64)),float(eb[mask].sum(dtype=np.float64))
            change=float((ec[mask]-eb[mask]).sum(dtype=np.float64))
            return {'rows':count,'candidate_MAE_TWD':None if not count else float(ec[mask].mean()),
                    'control_MAE_TWD':None if not count else float(eb[mask].mean()),
                    'MAE_delta_TWD':None if not count else float(ec[mask].mean()-eb[mask].mean()),
                    'candidate_abs_error_sum_TWD':sa,'control_abs_error_sum_TWD':sb,
                    'contribution_to_stage_delta_TWD':change/n,
                    'contribution_to_scope_delta_TWD':None if not scope_rows else change/scope_rows}
        def reconcile(parts,total):
            assert sum(p['rows'] for p in parts)==total['rows']
            for key in ('candidate_abs_error_sum_TWD','control_abs_error_sum_TWD','contribution_to_stage_delta_TWD'):
                assert close(sum(p[key] for p in parts),total[key]),key
        def section(mask):
            count=int(mask.sum());out={'rows':count,'pairs':{}}
            for pair in errors:
                whole=part(pair,mask,count);children={g:part(pair,mask&masks[g],count) for g in GROUPS}
                reconcile(list(children.values()),whole)
                if count:assert close(sum(x['contribution_to_scope_delta_TWD'] for x in children.values()),whole['MAE_delta_TWD'])
                out['pairs'][pair]={'overall':whole,'groups':children}
            return out
        full=section(np.ones(n,dtype=bool))
        strata={g:section(labels==g) for g in scorer.GROUPS}
        by_quarter={q:section(mask) for q,mask in quarters.items()}
        crosses={q:{g:section(mask&(labels==g)) for g in CROSS_STRATA} for q,mask in quarters.items()}
        missing={q:section(mask&(labels=='missing')) for q,mask in quarters.items()}
        for pair in errors:
            overall=full['pairs'][pair]['overall']
            reconcile([s['pairs'][pair]['overall'] for s in strata.values()],overall)
            reconcile([s['pairs'][pair]['overall'] for s in by_quarter.values()],overall)
            for q in quarters:
                reconcile([crosses[q][g]['pairs'][pair]['overall'] for g in CROSS_STRATA]+[missing[q]['pairs'][pair]['overall']],
                          by_quarter[q]['pairs'][pair]['overall'])
            for g in CROSS_STRATA:
                reconcile([crosses[q][g]['pairs'][pair]['overall'] for q in quarters],strata[g]['pairs'][pair]['overall'])
            for g in GROUPS:
                reconcile([s['pairs'][pair]['groups'][g] for s in strata.values()],full['pairs'][pair]['groups'][g])
                reconcile([s['pairs'][pair]['groups'][g] for s in by_quarter.values()],full['pairs'][pair]['groups'][g])
        for field,key in (('candidate_MAE_TWD','R72'),('control_MAE_TWD','v19')):
            saved=a['metrics'][key];p=full['pairs']['complete_pipeline']['overall']
            assert p['rows']==saved['rows'] and close(p[field],saved['overall_mae'])
            for g,s in strata.items():
                m=s['pairs']['complete_pipeline']['overall'];assert m['rows']==saved['strata'][g]['rows']
                assert (m[field] is None and saved['strata'][g]['mae'] is None) or close(m[field],saved['strata'][g]['mae'])
            for q,s in by_quarter.items():
                m=s['pairs']['complete_pipeline']['overall'];assert m['rows']==saved['forecast_quarters'][q]['rows']
                assert (m[field] is None and saved['forecast_quarters'][q]['mae'] is None) or close(m[field],saved['forecast_quarters'][q]['mae'])
            weighted=sum(w*strata[g]['pairs']['complete_pipeline']['overall'][field] for g,w in p48['target_group_weights'].items() if w>0)
            assert close(weighted,saved['weighted_mae'])
        assert a['metrics']['v19']==b['metrics']['R71']
        native=full['pairs']['native_prior']['overall']
        assert close(native['candidate_MAE_TWD'],a['native_MAE_diagnostic']['R72_prior'])
        assert close(native['control_MAE_TWD'],a['native_MAE_diagnostic']['R63_prior'])
        assert close(native['control_MAE_TWD'],c['native_MAE_diagnostic']['R63_prior'])
        result['stages'][stage]={'rows':n,'query_indices_sha256':value_sha(qi,'<i8'),'query_target_sha256':value_sha(y,'<f8'),
            'full':full,'strata':strata,'forecast_quarters':by_quarter,'quarter_by_new_seen':crosses,
            'missing_project_stratum_reconciliation':missing,
            'checks':{'complete_saved_overall_weighted_strata_quarters_matched':True,'native_saved_overall_matched':True,
                      'native_subgroups_newly_computed_not_previously_saved':True,'all_rows_and_error_contributions_reconciled':True}}
    result['input_sha256']=inputs
    return result


def render(result):
    lines=['# R72 三組 support／delta 機制診斷','',
        '固定分組：fallback=support 0；hit-zero=support>0 且保存 float32 delta=0；hit-nonzero=support>0 且 delta≠0。差值為 R72−對照，負值較好；只評保存向量，0 fit／模型、reference 或 spatial 推論。','',
        '完整對照為 v19/R71；原生對照為 R63 prior NN。全體、strata、季度及交叉組的列數與誤差貢獻已核對。完整融合核既存 overall/weighted/strata/quarters；原生僅 overall 有既存值可核，原生分群是本次新增診斷。','',
        '| 期間 | 組別 | n | 完整 MAE差 | 對完整overall貢獻 | 原生 MAE差 | 對原生overall貢獻 |',
        '|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:+,.2f}'
    for stage,a in result['stages'].items():
        for g in GROUPS:
            p=a['full']['pairs']['complete_pipeline']['groups'][g];q=a['full']['pairs']['native_prior']['groups'][g]
            lines.append(f"| {stage} | {g} | {p['rows']:,} | {fmt(p['MAE_delta_TWD'])} | {fmt(p['contribution_to_stage_delta_TWD'])} | {fmt(q['MAE_delta_TWD'])} | {fmt(q['contribution_to_stage_delta_TWD'])} |")
        p=a['full']['pairs']['complete_pipeline']['overall'];q=a['full']['pairs']['native_prior']['overall']
        lines.append(f"| {stage} | 全體 | {p['rows']:,} | {fmt(p['MAE_delta_TWD'])} | {fmt(p['contribution_to_stage_delta_TWD'])} | {fmt(q['MAE_delta_TWD'])} | {fmt(q['contribution_to_stage_delta_TWD'])} |")
    lines+=['','下表每格為「n；完整 MAE差／原生 MAE差（元）」。所有季度×new/seen及三個support組完整保留；空組為0及—，沒有剔除。完整兩側 MAE、誤差和與貢獻見 JSON。','',
        '| 期間 | 季度 | 原scorer群 | fallback | hit-zero | hit-nonzero |','|---|---|---|---|---|---|']
    for stage,a in result['stages'].items():
        for quarter,cross in a['quarter_by_new_seen'].items():
            for label,item in cross.items():
                cells=[]
                for g in GROUPS:
                    p=item['pairs']['complete_pipeline']['groups'][g];q=item['pairs']['native_prior']['groups'][g]
                    cells.append(f"{p['rows']:,}；{fmt(p['MAE_delta_TWD'])} / {fmt(q['MAE_delta_TWD'])}")
                lines.append('| '+ ' | '.join([stage,quarter,label]+cells)+' |')
    lines+=['','判讀界線：fallback 與 hit-zero 在新增欄都輸入0，但原來其他144欄與人群可不同。這個保存預測比較沒有訓練「額外命中旗標」的反事實模型，因此不能證明旗標有效，也不能將跨群差異當作道路資訊的因果效果。',
        '支持數不是獨立建案數或ESS。這些窗口多次重用，分組結果只供機制假說，不是 Private 勝率，不產生新挑列、門檻、種子或混合規則。',
        '',f"實際執行 exit {result['execution']['exit_code']}；{result['execution']['elapsed_seconds']:.3f} 秒。固定診斷 plan 在統計前保存，完整來源/input SHA、兩側MAE與空組皆在 [JSON](../outputs/round72/mechanism_review.json)。",
        '可重跑來源：[round72_mechanism_review.py](../src/round72_mechanism_review.py)，既有輸出拒覆寫。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
    paths=(OUT,PLAN_PATH,EXECUTION,REPORT)
    assert all(not p.exists() and not p.with_suffix('.sha256').exists() for p in paths),'Existing evidence: no retry/overwrite'
    started=stamp();tick=time.perf_counter()
    source={name:sha(ROOT/name) for name in ('src/round72_mechanism_review.py','src/round72_road_coverage_review.py',
        'src/round3_evaluate.py','src/round39_city_road_experiment.py','src/round72_nn_experiment.py')}
    plan={**PLAN,'source_sha256':source,'recorded_at_utc':started}
    write(PLAN_PATH,json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    code=0;failure=None;result=None
    try:result=run(plan);assert all(sha(ROOT/n)==h for n,h in source.items())
    except Exception:code=1;failure=traceback.format_exc();print(failure,flush=True)
    execution={'exit_code':code,'started_at_utc':started,'completed_at_utc':stamp(),'elapsed_seconds':time.perf_counter()-tick,
               'source_sha256':source['src/round72_mechanism_review.py'],'failure':failure,'automatic_retry':False}
    write(EXECUTION,json.dumps(execution,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if code:return code
    result['execution']=execution;result['execution_sha256']=sha(EXECUTION)
    write(REPORT,render(result));result['report_sha256_at_generation']=sha(REPORT)
    write(OUT,json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'elapsed_seconds':execution['elapsed_seconds'],'results_sha256':sha(OUT),
        'report_sha256':sha(REPORT),'stages':{s:a['full'] for s,a in result['stages'].items()}},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

