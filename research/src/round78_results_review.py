"""Summarize R78 saved small JSON only; no arrays, models, scoring or selection."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/round78';REPORT=ROOT/'reports/round78_report.md'
STAGES=('development','confirmation','auxiliary');SEEDS=('42','137','2026')
PAIRS={'v20_minus_v19':('v19','v20'),'v21_minus_v20':('v20','v21'),'v21_minus_v19':('v19','v21')}
PN={k:k.replace('_minus_','−') for k in PAIRS}
NAMES={'development':'中間年度 development','confirmation':'最新年度 confirmation','auxiliary':'較早年度 auxiliary'}
PINS={'outputs/round78/results.json':'7eadafea0a3a6632053f7bf23583ce66953acb7416111b99699cb8c5832b2c77',
'outputs/round78/execution.json':'373890e47acd608373282f9871c8693b9a8d093cd978e41a718f83e7fa8f7378',
'src/round78_prior_seed_sensitivity.py':'40f09952da19cbd909448f1e28906949d55363aa366c7b78e227c7cacd817f57',
'reports/round78_plan.md':'e23b31479b857d590750bc3ab53cfc8b10a90bfd8420612594038e50819bb04a'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def fmt(x,signed=False):return '—' if x is None else format(x,'+,.2f' if signed else ',.2f')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def counts(v):return {'lower':sum(x is not None and x<0 for x in v),'equal':sum(x==0 for x in v if x is not None),'higher':sum(x is not None and x>0 for x in v),'no_samples':sum(x is None for x in v)}
def delta(a,b,d):
    if a is None or b is None:assert d is None
    else:assert math.isfinite(d) and math.isclose(b-a,d,rel_tol=1e-12,abs_tol=1e-8)
def run():
    tick=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat();inputs={}
    def read(n):
        p=ROOT/n;h=sha(p);assert n not in PINS or PINS[n]==h;inputs[n]=h
        q=p.with_suffix('.sha256');assert q.read_text(encoding='utf8').split()[0]==h;inputs[rel(q)]=sha(q)
        return json.loads(p.read_text(encoding='utf8'))
    assert all(not p.exists() for p in (REPORT,OUT/'results_review.json',OUT/'results_review.sha256')),'No overwrite'
    r=read('outputs/round78/results.json');e=read('outputs/round78/execution.json');p=read('outputs/round78/plan.json')
    for n,h in PINS.items():assert sha(ROOT/n)==h;inputs[n]=h
    inputs[rel(Path(__file__).resolve())]=sha(Path(__file__))
    assert r['round']==78 and r['status']=='completed_conditional_prior_seed_sensitivity'
    assert r['checked_stages']==list(STAGES) and r['seeds']==[42,137,2026] and r['fixed_area_seed']==42
    assert e['exit_code']==0 and e['failure'] is None and e['automatic_retry'] is False
    assert e['source_sha256']==r['source_sha256']['src/round78_prior_seed_sensitivity.py']==PINS['src/round78_prior_seed_sensitivity.py']
    assert r['text_plan_sha256']==p['text_plan_sha256']==PINS['reports/round78_plan.md'] and p['seeds']==r['seeds'] and p['fixed_area_seed']==42
    assert all(r[k]==0 for k in ('new_model_fits','model_loads','model_predict_calls'))
    assert r['reference_predict_calls']==r['spatial_predict_calls']==6
    assert all(r[k] is False for k in ('seed_selection','seed_average','new_CSV','new_bootstrap','fresh_blind_validation','full_model_multiseed_training'))
    assert len(r['protected_before'])==23 and r['protected_before']==r['protected_after']==p['protected_before']
    assert r['selector_sha256_before']==r['selector_sha256_after']==p['selector_sha256_before']
    stages={};scope={};bands={};stage_sign={};sign={c:{} for c in PAIRS}
    for s in STAGES:
        a=r['stages'][s];assert a['all_prior_seeds_evaluated'] and set(a['seeds'])==set(SEEDS)
        stages[s]={k:a[k] for k in ('fit_rows','query_rows','original_frame_sha256','query_target_sha256','test_mass_covered','unsupported_test_rows','elapsed_seconds')}
        stages[s]['seeds']={};scope[s]=[];bands[s]=[];stage_sign[s]={}
        for seed in SEEDS:
            x=a['seeds'][seed];m=x['metrics'];assert x['prior_seed']==int(seed) and x['area_seed']==42
            assert x['saved_v19_full_metrics_exact'] and x['seed42_R76_R77_controls_exact']==(seed=='42') and x['new_reference_spatial_calls']==int(seed!='42')
            assert set(m)=={'v19','v20','v21'} and all(z['rows']==a['query_rows'] for z in m.values())
            for c,(b,t) in PAIRS.items():
                for metric,key in (('overall','overall_mae'),('weighted','weighted_mae')):delta(m[b][key],m[t][key],x['differences'][c][metric]['delta_TWD'])
                delta(x['test_composition_MAE'][b],x['test_composition_MAE'][t],x['test_composition_differences_TWD'][c])
            for cat in ('strata','forecast_quarters'):
                for name,g in m['v19'][cat].items():
                    assert all(m[n][cat][name]['rows']==g['rows'] for n in m)
                    d={c:x['differences'][c][cat][name]['delta_TWD'] for c in PAIRS}
                    for c,(b,t) in PAIRS.items():delta(m[b][cat][name]['mae'],m[t][cat][name]['mae'],d[c])
                    scope[s].append({'seed':int(seed),'category':cat,'name':name,'rows':g['rows'],'MAE':{n:m[n][cat][name]['mae'] for n in m},'differences_TWD':d})
            assert len(x['price_bands'])==5 and sum(b['rows'] for b in x['price_bands'])==a['query_rows']
            for b in x['price_bands']:
                bands[s].append({'seed':int(seed),'band':b['band'],'rows':b['rows'],'MAE':{n:b[n] for n in m},'differences_TWD':{c:b[t]-b[z] if b['rows'] else None for c,(z,t) in PAIRS.items()}})
            stages[s]['seeds'][seed]={k:x[k] for k in ('prior_seed','area_seed','metrics','differences','test_composition_MAE','test_composition_differences_TWD','price_bands','saved_v19_full_metrics_exact','seed42_R76_R77_controls_exact','new_reference_spatial_calls')}
        for c in PAIRS:
            stage_sign[s][c]={}
            for metric in ('overall','weighted','testmix'):
                vals=[a['seeds'][k]['test_composition_differences_TWD'][c] if metric=='testmix' else a['seeds'][k]['differences'][c][metric]['delta_TWD'] for k in SEEDS]
                stage_sign[s][c][metric]={'values_in_seed_order':vals,**counts(vals)}
    for c in PAIRS:
        for metric in ('overall','weighted','testmix'):sign[c][metric]=counts([x for s in STAGES for x in stage_sign[s][c][metric]['values_in_seed_order']])
    sr=[]
    for c in PAIRS:sr.append([PN[c]]+[f"{sign[c][k]['lower']}／{sign[c][k]['equal']}／{sign[c][k]['higher']}" for k in ('overall','weighted','testmix')])
    lines=['# R78：固定 area seed42 的 prior 初始化敏感度',
'本輪只檢查已保存的 prior NN 三種初始化下，v20／v21 組合相對相同種子的 v19 是否維持改善。area 模型均固定 seed42；不是整套模型的三種子訓練。未挑最好種子、未做種子平均，沒有新 CSV 或推薦變更。',
'所有差值皆為候選減對照的 MAE（TWD）：負值改善、正值退步。v19_s／v20_s／v21_s 是相同 prior seed 的研究組合；只有 seed42 對應實際已發布版本。',
'## 結果要點',table(['比較','整體 改善／相同／退步','固定加權 改善／相同／退步','test 組成 改善／相同／退步'],sr)]
    for s in STAGES:
        chunks=[]
        for c in PAIRS:
            z=stage_sign[s][c];chunks.append(f"{PN[c]} 整體 {z['overall']['lower']}/3 改善、加權 {z['weighted']['lower']}/3、test 組成 {z['testmix']['lower']}/3")
        lines.append('- '+NAMES[s]+'：'+'；'.join(chunks)+'。')
    lines += ['v20、v21 對各自 v19 的整體／加權改善不只出現在 seed42；但 v21 相對 v20 在最新期的整體、加權與 test 組成三種子皆退步，較早兩期則皆改善。這支持既有年際取捨具有條件式一致性，不能把九組結果當九次獨立試驗或 Private 勝率。',
'## 設計、執行與核對範圍',
'A0 是 R54 area42 與各 seed A144 prior 的完整 NN 分支，直接重用 R71；A1 以 R74 日期加權 area42 配對相同 prior。L=R46 log、U0=原 R40 unit、U1=R76 新 unit。v19_s=median(A0_s,L,U0)，v20_s=median(A0_s,L,U1)，v21_s=median(A1_s,L,U1)。',
'A1 seed42 重用 R74 分支；137、2026 的 base 為 .75×(.5×R74 area42+.5×prior_s)+.25×old_unit，再各做一次既有 reference trend 與 spatial。三期共六次 reference、六次 spatial；沒有樹／NN模型載入、模型推論或新 fit。',
f"主程式 {r['elapsed_seconds']:.3f} 秒，外層 {e['elapsed_seconds']:.3f} 秒，exit 0。保存證據記錄九個 v19_s 向量／完整 metrics 一致；seed42 v20/v21 與 R76/R77 的向量、完整 metrics、test 組成及五價帶一致。22 份既有 CSV 加 registry 共 23 檔與 selector 前後未變。",
'本報告器只核小 JSON 收據、來源身分、保存差值算術與完成欄位，沒有獨立重讀預測陣列或重新計算逐筆 MAE。主流程確有 reference／spatial state pickle 讀取，不能解讀為完全沒有任何 state 載入。',
'## 九組整體、固定加權與 test 組成',
'固定加權沿原 scorer 新／舊案權重。test 組成沿既有城市×新／舊案規則（每格至少 200 筆），在受支持的 test 質量上正規化；不是未知 test 真實誤差。']
    for metric,key,title in (('overall','overall_mae','整體 MAE'),('weighted','weighted_mae','固定加權 MAE'),('testmix',None,'test 組成 MAE')):
        rows=[]
        for s in STAGES:
            for seed in SEEDS:
                x=stages[s]['seeds'][seed];v=x['test_composition_MAE'] if key is None else {n:x['metrics'][n][key] for n in ('v19','v20','v21')}
                d=x['test_composition_differences_TWD'] if key is None else {c:x['differences'][c][metric]['delta_TWD'] for c in PAIRS}
                rows.append([s,seed]+[fmt(v[n]) for n in ('v19','v20','v21')]+[fmt(d[c],True) for c in PAIRS])
        lines += ['### '+title,table(['期別','prior seed','v19_s','v20_s','v21_s']+[PN[c] for c in PAIRS],rows)]
    lines.append(table(['期別','query 筆數','支持的 test 比例','不支持 test 筆數'],[[s,str(stages[s]['query_rows']),f"{stages[s]['test_mass_covered']:.4%}",str(stages[s]['unsupported_test_rows'])] for s in STAGES]))
    lines += ['## 局部符號一致性与例外','三種子同向只描述此條件式診斷。下方保留全部新／舊案、四季、五真價帶，不省略不利分群。']
    for s in STAGES:
        for c in PAIRS:
            good=[];bad=[];mixed=[]
            groups=[('strata','seen'),('strata','new_nonmissing')]+[('forecast_quarters',q) for q in ('Q1','Q2','Q3','Q4')]
            for cat,name in groups:
                v=[a['differences_TWD'][c] for a in scope[s] if a['category']==cat and a['name']==name];label={'seen':'舊案','new_nonmissing':'新案'}.get(name,name)
                (good if all(z<0 for z in v) else bad if all(z>0 for z in v) else mixed).append(label)
            for b in stages[s]['seeds']['42']['price_bands']:
                v=[a['differences_TWD'][c] for a in bands[s] if a['band']==b['band']]
                (good if all(z is not None and z<0 for z in v) else bad if all(z is not None and z>0 for z in v) else mixed).append(b['band'])
            lines.append(f"- {s}，{PN[c]}：三種子皆改善＝{'、'.join(good) or '無'}；皆退步＝{'、'.join(bad) or '無'}；方向不一致或相同＝{'、'.join(mixed) or '無'}。")
    for s in STAGES:
        rows=[[str(a['seed']),{'seen':'舊案','new_nonmissing':'新案','missing':'建案缺名'}.get(a['name'],a['name']),str(a['rows'])]+[fmt(a['differences_TWD'][c],True) for c in PAIRS] for a in scope[s]]
        lines += ['### '+NAMES[s]+'：新／舊案與四季差值',table(['prior seed','分組','筆數']+[PN[c] for c in PAIRS],rows)]
        rows=[[str(a['seed']),a['band'],str(a['rows'])]+[fmt(a['MAE'][n]) for n in ('v19','v20','v21')]+[fmt(a['differences_TWD'][c],True) for c in PAIRS] for a in bands[s]]
        lines += ['### '+NAMES[s]+'：五真價帶完整表',table(['prior seed','真價帶','筆數','v19_s MAE','v20_s MAE','v21_s MAE']+[PN[c] for c in PAIRS],rows)]
    lines += ['## 限制與保留',
'真價帶只用於事後誤差診斷，不能在預測時知道，也不能據此逐 ID 修價。完整城市／月份 metrics 保留在機器結果與本報告 JSON；不把三期九組投票當採用規則。',
'只改變保存 prior NN 的初始化，area、log、unit 與後處理固定，未涵蓋所有模型的初始化變異。沒有新 bootstrap；窗口已反覆使用，沒有校正選型偏差。跨種子同號不是泛化或 Private 保證。',
'本輪不選 seed、不平均、不改既有 CSV／selector。v20 優先、v21 替代的既有本機候選狀態由主控維護；Private 仍未知、最終版本未定。程式、模型與來源證據保留，GitHub 待最終版本選定後整理，尚未發布。',
'## 可追溯來源',
'- [事前計畫](round78_plan.md)；[執行程式](../src/round78_prior_seed_sensitivity.py)；[本報告程式](../src/round78_results_review.py)。',
'- [完整保存結果](../outputs/round78/results.json)；[執行收據](../outputs/round78/execution.json)；[JSON 彙整](../outputs/round78/results_review.json)。',
'- [R76 報告](round76_report.md)；[R77 報告](round77_report.md)。']
    report='\n\n'.join(lines)+'\n';rh=hashlib.sha256(report.encode('utf8')).hexdigest()
    summary={'round':78,'status':'saved_JSON_review_completed','stages':stages,'scope_tables':scope,'price_band_tables':bands,
'sign_counts':sign,'stage_sign_consistency':stage_sign,'fixed_area_seed':42,'prior_seeds':[42,137,2026],
'new_fits':0,'tree_NN_model_loads':0,'model_predict_calls':0,'reference_predict_calls':6,'spatial_predict_calls':6,
'seed_selection':False,'seed_average':False,'new_CSV':False,'selector_updated':False,'protected_files_recorded_unchanged':23,
'main_elapsed_seconds':r['elapsed_seconds'],'outer_elapsed_seconds':e['elapsed_seconds'],'input_sha256':inputs,
'report_path':rel(REPORT),'report_sha256':rh,'new_adoption_decision':False,
'limitations':['fixed area seed42; prior initialization only','three seeds are not independent validation samples','reused windows without selection-bias correction','no new bootstrap or Private probability'],
'review_execution':{'started_at_utc':started,'elapsed_seconds':time.monotonic()-tick,'exit_code':0,'arrays_read':False,'models_loaded':False,'scoring_called':False}}
    data=(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8');h=hashlib.sha256(data).hexdigest()
    with REPORT.open('xb') as f:f.write(report.encode('utf8'))
    with (OUT/'results_review.json').open('xb') as f:f.write(data)
    with (OUT/'results_review.sha256').open('x',encoding='utf8') as f:f.write(h+'\n')
    print(json.dumps({'status':summary['status'],'review_sha256':h,'report_sha256':rh,'source_sha256':sha(Path(__file__)),'elapsed_seconds':summary['review_execution']['elapsed_seconds'],'sign_counts':sign},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.parse_args();run()
