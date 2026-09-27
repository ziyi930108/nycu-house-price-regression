"""R73 JSON-only mechanism review; no arrays, model operations or new scoring."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,time
ROOT=Path(__file__).resolve().parents[1]
STAGES=('development','confirmation','auxiliary')
NAMES={'development':'開發（中間年度）','confirmation':'確認（最新年度）','auxiliary':'輔助（較早年度）'}
DRIVER_SHA='a8561516415fb292eedc6d26a84fac248224d01e90333d0d79ac1a3e03f378a9'
OUT=ROOT/'outputs/round73/results_review.json'
MD=ROOT/'reports/round73_report.md'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def write(p,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    with p.open('xb') as f:f.write(data)
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(hashlib.sha256(data).hexdigest()+'\n')

def measure(m,key):
    if key in ('overall','weighted'):return m[key+'_mae']
    if key in ('new_nonmissing','seen'):return m['strata'][key]['mae']
    return m['forecast_quarters'][key]['mae']

def review(paired_name=None):
    started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();inputs={}
    def locked(name):
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.suffix=='.json'
        receipt=p.with_suffix('.sha256');digest=sha(p)
        assert receipt.read_text(encoding='utf8').strip()==digest,name
        inputs[relative(p)]=digest;inputs[relative(receipt)]=sha(receipt)
        return json.loads(p.read_text(encoding='utf8'))
    # Incomplete execution stops before creating any review artifacts; never poll/retry here.
    assert (ROOT/'outputs/round73/results.json').is_file() and (ROOT/'outputs/round73/execution.json').is_file(),'R73 not complete; return to owner'
    run=locked('outputs/round73/execution.json');assert run['exit_code']==0 and run['failure'] is None
    records={n:locked(f'outputs/round{n}/results.json') for n in (73,72,71,63)}
    r=records[73]
    assert all(a['checked_stages']==list(STAGES) for a in records.values())
    assert r['status']=='three_periods_completed_diagnostic_only' and r['diagnostic_only'] and not r['eligible_for_export']
    assert r['new_NN_fits']==3 and not r['new_submission_generated'] and r['protected_before']==r['protected_after']
    assert len(r['protected_before'])==21 and run['source_sha256']==DRIVER_SHA
    assert r['source_sha256']['src/round73_null_road_experiment.py']==DRIVER_SHA
    assert not OUT.exists() and not OUT.with_suffix('.sha256').exists() and not MD.exists(),'No overwrite/retry'
    paired=locked(paired_name) if paired_name else None
    if paired is not None:assert paired['replicates']==2000 and paired['seed']==20260925 and set(paired['stages'])==set(STAGES)
    result={'round':73,'status':'completed_JSON_only_review','diagnostic_only':True,'eligible_for_export':False,
      'main_comparison':'R72 minus R73 null145; negative means road delta better',
      'secondary_comparison':'R73 null145 minus v19; background only, not candidate selection',
      'started_at_utc':started,'source_sha256':sha(Path(__file__)),'input_sha256':inputs,'checked_stages':[],'stages':{},
      'model_or_array_loads':0,'new_scores':0,'new_fits':0,'new_bootstrap':False,'new_CSV':False,
      'R73_fit_count':3,'R73_main_seconds':r['elapsed_seconds'],'R73_execution':run,
      'R73_fit_seconds':sum(r['stages'][s]['fit_seconds'] for s in STAGES),
      'full_saved_control_metrics_exact':True,'MAE_decomposition_is_algebra_not_causality':True,
      'initialization_and_RNG_shape_only_inferred_from_source':True,'bitwise_retraining_claimed':False,
      'paired_path':paired_name,'paired_included':paired is not None}
    for s in STAGES:
        a,b,c,d=(records[n]['stages'][s] for n in (73,72,71,63))
        assert all(a['fit_rows']==z['fit_rows'] and a['query_rows']==z['query_rows'] for z in (b,c,d))
        assert a['metrics']['R72']==b['metrics']['R72'] and a['metrics']['v19']==c['metrics']['R71'] and a['metrics']['v17']==d['metrics']['R63']
        assert a['diagnostic_only'] and not a['eligible_for_export'] and a['new_NN_fits']==1
        assert a['controls_full_metrics_exact'] and a['full_query_reload_exact'] and a['query_labels_used_only_by_scorer']
        assert a['runtime_Torch_numpy_CUDA_GPU_exact_to_R72'] and a['configuration_factor_weights_bias_exact_to_R72']
        assert a['numeric_count']==b['numeric_count']==145 and a['parameter_count']==b['parameter_count']
        prep=a['preparation'];assert prep['original_144_prefix_bitwise_same_as_R72_and_R63'] and prep['other_seven_arrays_bitwise_same_as_R72_and_R63']
        assert prep['only_changed_column_zero_based']==144 and not prep['encoder_prior_road_prior_weights_refit'] and not prep['query_labels_used']
        native={'R72':b['native_MAE_diagnostic']['R72_prior'],'R73':a['native_MAE_diagnostic']['R73_null145'],'v19':d['native_MAE_diagnostic']['R63_prior']}
        assert native['R72']==a['native_MAE_diagnostic']['R72_road145'] and native['v19']==b['native_MAE_diagnostic']['R63_prior']
        binding=c['native_seed_bindings']['42'];expected=f'outputs/round63/{s}/model/predictions_twd.npy'
        assert binding['path'].replace(chr(92),'/')==expected and binding['sha256']==d['artifact_sha256'][expected]
        metrics={'R72':b['metrics']['R72'],'R73':a['metrics']['R73'],'v19':c['metrics']['R71']}
        comparisons={}
        for key in ('overall','weighted','new_nonmissing','seen','Q1','Q2','Q3','Q4'):
            road,null,control=(measure(metrics[n],key) for n in ('R72','R73','v19'))
            assert all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in (road,null,control))
            main=road-null;background=null-control;total=road-control;residual=total-(main+background)
            assert abs(residual)<=1e-8
            comparisons[key]={'R72_minus_null_TWD':main,'R72_minus_null_pct':100*main/null,
              'null_minus_v19_TWD':background,'null_minus_v19_pct':100*background/control,
              'R72_minus_v19_TWD':total,'algebra_residual_TWD':residual}
        assert native['R72']-native['R73']==a['R72_minus_null145_native_MAE_TWD']
        for key in ('overall','weighted'):assert comparisons[key]['R72_minus_null_TWD']==a['R72_minus_null145'][key]['delta_TWD']
        item={'fit_rows':a['fit_rows'],'query_rows':a['query_rows'],'dates':a['dates'],'fit_seconds':a['fit_seconds'],
          'numeric_count':145,'parameter_count':a['parameter_count'],'native_MAE':native,'metrics':metrics,'comparisons':comparisons,
          'native_R72_minus_null_TWD':native['R72']-native['R73'],'native_null_minus_v19_TWD':native['R73']-native['v19'],
          'native_R72_minus_v19_TWD':native['R72']-native['v19'],'paired':None}
        if paired is not None:
            ps=paired['stages'][s];assert ps['point_metrics_reproduced']
            assert set(ps['prediction_sha256'].values())=={a['prediction_sha256'],b['prediction_sha256']}
            item['paired']={}
            for method in ('project','month'):
                item['paired'][method]={}
                for key in ('overall','weighted'):
                    x=ps[method]['summary'][key]
                    assert x['valid_replicates']==2000 and x['invalid_replicates']==0
                    # Bootstrap point uses grouped accumulation, so allow only tiny summation rounding.
                    assert math.isclose(x['candidate_mae'],measure(metrics['R72'],key),rel_tol=1e-12,abs_tol=1e-5)
                    assert math.isclose(x['control_mae'],measure(metrics['R73'],key),rel_tol=1e-12,abs_tol=1e-5)
                    assert math.isclose(x['point_delta'],comparisons[key]['R72_minus_null_TWD'],rel_tol=1e-9,abs_tol=1e-5)
                    item['paired'][method][key]={k:x[k] for k in ('point_delta','percentile_2_5','percentile_97_5')}
        result['stages'][s]=item;result['checked_stages'].append(s)
    lines=['# R73：同145欄、道路δ歸零的機制對照','',
      '**本輪僅機制診斷，不以零欄模型選優或發布CSV。** 主比較為R72（有道路δ）減R73（null145），負值表示δ較好；R73減v19僅背景。所有表均完整保留三期。','',
      '直接重用R72 prepared的九個陣列，只把train/query_num第145欄改為float32正零；原144欄與其他七陣列逐位同R72及R63。沒有重建encoder、道路／base prior、日期權重或因子。寬度145、類別維度、參數量、因子／權重／head bias及Torch、NumPy、CUDA、GPU與R72一致。',
      '沿原R63 trainer，不加入RNG wrapper；seed42、30epochs、batch2048與加權總價MAE固定，每期1fit、共3fit，完整query保存重載exact。query標籤僅評分，三期不按前一期成績停止。',
      '管線為`.75*(.5*R54area+.5*null145NN)+.25*oldunit`，reference與spatial各一次，再與R46log／R40unit逐列median；沒有oldraw子分支，也沒有新採用門檻。','',
      '| 期別 | 模型 | 原生prior NN MAE | 完整整體MAE | 完整加權MAE |','| --- | --- | ---: | ---: | ---: |']
    for s in STAGES:
        a=result['stages'][s]
        for name in ('R72','R73','v19'):
            m=a['metrics'][name];lines.append(f"| {NAMES[s]} | {name} | {a['native_MAE'][name]:,.2f} | {m['overall_mae']:,.2f} | {m['weighted_mae']:,.2f} |")
    lines+=['','v19的原生prior NN為保存R63/144欄seed42；完整v19取R71。原生與完整MAE量綱皆TWD，不能把原生改善直接當最終改善。','',
      '| R72−null145（元） | 原生NN | 完整整體 | 完整加權 | 新案 | 舊案 | Q1 | Q2 | Q3 | Q4 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in STAGES:
        a=result['stages'][s];vals=[a['native_R72_minus_null_TWD']]+[a['comparisons'][k]['R72_minus_null_TWD'] for k in ('overall','weighted','new_nonmissing','seen','Q1','Q2','Q3','Q4')]
        lines.append('| '+NAMES[s]+' | '+' | '.join(f'{v:+,.2f}' for v in vals)+' |')
    lines+=['','## MAE差值分解','',
      '`R72−v19 = (R72−null145) + (null145−v19)`只是同一批保存MAE的代數恆等式，並非初始化與道路資訊的無噪音因果分解。','',
      '| 期別 | 指標 | R72−v19 | R72−null145 | null145−v19 |','| --- | --- | ---: | ---: | ---: |']
    for s in STAGES:
        for key in ('overall','weighted'):
            c=result['stages'][s]['comparisons'][key]
            lines.append(f"| {NAMES[s]} | {key} | {c['R72_minus_v19_TWD']:+,.2f} | {c['R72_minus_null_TWD']:+,.2f} | {c['null_minus_v19_TWD']:+,.2f} |")
    lines+=['','## 直接成對區間','']
    if paired is None:lines+=['本報告尚未納入R72−null145直接成對結果；其他候選的區間不可套用於此比較。']
    else:
        lines+=['原固定seed20260925、每方法2000次；所有三期、建案群組與連續3月循環block、整體／加權均列出。','',
          '| 期別 | 抽法 | 指標 | R72−null點估计 | 2.5% | 97.5% |','| --- | --- | --- | ---: | ---: | ---: |']
        for s in STAGES:
            for method in ('project','month'):
                for key,x in result['stages'][s]['paired'][method].items():
                    lines.append(f"| {NAMES[s]} | {method} | {key} | {x['point_delta']:+,.2f} | {x['percentile_2_5']:+,.2f} | {x['percentile_97_5']:+,.2f} |")
        lines+=['',f"[直接成對結果](../{paired_name})。區間以保存模型與重用窗口為條件，未校正多輪選型，不是Private勝率或p值。"]
    lines+=['','## 主比較解讀','',
      '依開發／最新確認／較早輔助順序，R72−null完整整體MAE差為 '+ '／'.join(f"{result['stages'][s]['comparisons']['overall']['R72_minus_null_TWD']:+,.2f}元" for s in STAGES)+'；原生prior NN差為 '+ '／'.join(f"{result['stages'][s]['native_R72_minus_null_TWD']:+,.2f}元" for s in STAGES)+'。原生與完整管線的方向不能互相代替。',
      '所有季度、新舊案均列於主比較表，不能由平均值推定每個群組受益；零欄對照僅回答這個固定145欄訓練契約的機制問題，沒有取得輸出候選資格。']
    if paired is not None:
        for s in STAGES:
            descriptions=[]
            for method in ('project','month'):
                for key in ('overall','weighted'):
                    ci=result['stages'][s]['paired'][method][key]
                    direction='全低於零' if ci['percentile_97_5']<0 else '全高於零' if ci['percentile_2_5']>0 else '跨零'
                    descriptions.append(f'{method}/{key} {direction}')
            lines.append(f"{NAMES[s]}："+'；'.join(descriptions)+'。')
    lines+=['','## 執行與限制','',
      f"唯一主流程exit0，內部{result['R73_main_seconds']:.3f}秒、execution {run['elapsed_seconds']:.3f}秒；三次fit共{result['R73_fit_seconds']:.3f}秒。各期fit秒數："+'／'.join(f"{NAMES[s]} {result['stages'][s]['fit_seconds']:.3f}" for s in STAGES)+'。',
      '20份舊CSV加registry共21個保護檔前後相同；沒有CSV／final export、selector修改或零欄候選選優。本工具只讀JSON與收據，未載陣列或模型、重算逐列分數、重跑bootstrap。',
      '同145欄、同seed与未改trainer，僅能依source推定相同初始化及隨機呼叫形狀。R72未保存初始state_dict SHA／RNG trace，CUDA未強制完全決定性，不能宣稱逐位重訓或嚴格因果識別。零欄145也不等價於144模型。歷史年度已多輪使用；query y只供離線scorer，Private未知。',
      '', '[固定計畫](round73_plan.md) · [完整結果](../outputs/round73/results.json) · [JSON-only核對](../outputs/round73/results_review.json) · [主執行紀錄](../outputs/round73/execution.json)','']
    result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
    write(OUT,result)
    with MD.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'review_sha256':sha(OUT),'report_sha256':sha(MD),'paired_included':paired is not None},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.add_argument('--paired-json')
    a=p.parse_args();review(a.paired_json)
