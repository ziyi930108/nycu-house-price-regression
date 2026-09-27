"""Record the R78 diagnostic decision and insert concise human summaries; JSON/MD only."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/round78'
DOCS={'CURRENT_STATUS.md':'1fbc9fad8e01a29064629380fb3d4743adfd1f3a457a42675bac057a5f1a9ad0',
'PRIVATE_SELECTION.md':'68e101b6d4fc7eaf8011ad0db3a2a62b43661a501ec11f07ddb4a3b186433c87',
'reports/model_selection_explanation.md':'79bc345309e296ad6774e9a1a2252358f335d9b1fb790c0ee97ca6998aabca53'}
PINS={'outputs/round78/results.json':'7eadafea0a3a6632053f7bf23583ce66953acb7416111b99699cb8c5832b2c77',
'outputs/round78/execution.json':'373890e47acd608373282f9871c8693b9a8d093cd978e41a718f83e7fa8f7378',
'outputs/round78/results_review.json':'ea6a1898931d1af3f67024a84e6085e72ad20dd11c1f689ce91d4ca8165834c4',
'reports/round78_report.md':'a87af9598fe495d49b4c8d8356548802570e755fa4c36a60241c5815906ce794',
'reports/round78_final_tail_inspection.json':'54ba1c9949c5d670ed9c8dbd78a96c3da91d611350fefed2ce0d577852c2cd0c',
'reports/round78_final_tail_inspection.md':'f9e431974a0aa5243f2a9c4402a8f40ef18fd26c63dc52c7e81b3daa65a8f829',
'outputs/current_private_candidate.json':'a488fa83d3a3581bc939d414897314d7caa443a803ca12096644b78d5245af56',
'src/round78_prior_seed_sensitivity.py':'40f09952da19cbd909448f1e28906949d55363aa366c7b78e227c7cacd817f57',
'src/round78_results_review.py':'c8b7338d11dcfd43df06d613eb3cfcf2fa6ccfbe92f64ab13453c91d4c7d254c'}
MARK='<!-- R78_COMPLETED_DIAGNOSTIC -->'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    tick=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();inputs={};notes={}
    assert not (OUT/'decision.json').exists() and not (OUT/'decision.sha256').exists(),'Decision already exists'
    for n,h in PINS.items():
        p=ROOT/n;assert sha(p)==h,n
        (notes if p.suffix=='.md' else inputs)[n]=h
    def read(n,require_receipt=True):
        p=ROOT/n;h=PINS[n];receipts=[q for q in (p.with_suffix('.sha256'),Path(str(p)+'.sha256')) if q.is_file()]
        assert receipts or not require_receipt,n
        for q in receipts:assert q.read_text(encoding='utf8').split()[0]==h;inputs[q.relative_to(ROOT).as_posix()]=sha(q)
        return json.loads(p.read_text(encoding='utf8'))
    result=read('outputs/round78/results.json');execution=read('outputs/round78/execution.json');review=read('outputs/round78/results_review.json')
    tail=read('reports/round78_final_tail_inspection.json',False);selector=read('outputs/current_private_candidate.json',False)
    assert result['status']=='completed_conditional_prior_seed_sensitivity' and execution['exit_code']==0
    assert review['status']=='saved_JSON_review_completed' and review['report_sha256']==notes['reports/round78_report.md']
    assert result['new_model_fits']==0 and not result['new_CSV'] and result['fixed_area_seed']==42
    assert result['seeds']==[42,137,2026] and result['protected_before']==result['protected_after']
    assert result['selector_sha256_before']==result['selector_sha256_after']==PINS['outputs/current_private_candidate.json']
    for c in ('v20_minus_v19','v21_minus_v19'):
        assert all(review['sign_counts'][c][m]['lower']==9 for m in ('overall','weighted'))
    for k in ('42','137','2026'):
        assert result['stages']['confirmation']['seeds'][k]['differences']['v21_minus_v20']['overall']['delta_TWD']>0
        assert result['stages']['development']['seeds'][k]['test_composition_differences_TWD']['v20_minus_v19']>0
    assert selector['private_priority']=='outputs/submission_v20_trial.csv' and selector['backup']=='outputs/submission_v21_trial.csv'
    assert selector['private_mae'] is None and not selector['final_version_chosen'] and selector['known_public_best_mae']==2388386.4613
    assert tail['status']=='completed_target_free_tail_inspection' and tail['rows']==47073 and tail['execution']['exit_code']==0
    assert not tail['target_labels_read'] and not tail['MAE_or_error_estimated'] and tail['prediction_changes_are_not_errors']
    assert all(tail['checks'].values()) and tail['checks']['new_unit_base_exact_TWD_only']
    assert tail['report_sha256_at_generation']==notes['reports/round78_final_tail_inspection.md']
    max_twd=tail['distributions']['all_test']['predictions_TWD']['v20']['max'];assert max_twd==13194531292.989578
    assert max_twd==tail['distributions']['all_test']['predictions_TWD']['v21']['max']
    highest=tail['selected_records']['highest_v20'][0];assert highest['predictions_TWD']['v20']==max_twd and highest['features']['main_use']=='商業用'
    assert tail['protected_before']==tail['protected_after']
    assert (ROOT/'reports/round79_plan.md').is_file() and not (ROOT/'outputs/round79').exists(),'R79 status advanced; do not write stale status'
    inputs['reports/round79_plan.md']=sha(ROOT/'reports/round79_plan.md')
    inputs[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    registry=ROOT/'runs/versions.json';registry_sha=sha(registry)
    old={};new={};blocks={}
    for n,h in DOCS.items():
        p=ROOT/n;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==h and MARK.encode() not in raw,n
        assert not p.with_suffix('.sha256').exists() and not Path(str(p)+'.sha256').exists(),'Unexpected document receipt'
        assert raw.startswith(b'# ') and b'\n' in raw
        isreport=n.startswith('reports/');rp='' if isreport else 'reports/';op='../outputs/' if isreport else 'outputs/'
        block=(f"\n{MARK}\n**R78 已完成條件式 prior-seed 診斷，維持 v20 有限本機優先、v21 替代。** 兩版相對同 seed 的 v19，九組整體／固定加權皆改善；最新確認期 v20 勝 v21 的方向在三個 prior seed 一致。代價亦一致：開發期 v20 的 test 組成 MAE 三種子皆退步。只有 seed42 對應實際 CSV，137／2026 只是研究組合，沒有另產九份 CSV。\n\n"
               f"area 固定 seed42，並非整套模型多 seed；本輪 0 fit，沒有挑 seed 或做平均，也不是新盲測，重用窗口未校正選型偏差，不能推算 Private 勝率。[完整 R78 報告]({rp}round78_report.md) · [結案紀錄]({op}round78/decision.json)。\n\n"
               f"[尾端純 X 檢查]({rp}round78_final_tail_inspection.md)未見再次乘面積，但最高商業總價約 **131.945 億元**仍可能是分支共同外推；沒有真價可驗，不加價帽、不逐 ID 改價。v18 仍是已回報 Public 最佳 **2,388,386.4613**；Private 未知、最終未選，GitHub 待定版後整理。本段更新時 [R79 固定計畫]({rp}round79_plan.md)／driver 尚在準備，**尚未 fit**。以下原文保留當時紀錄。\n")
        first,rest=raw.split(b'\n',1);b=block.encode('utf8');updated=first+b'\n'+b+rest
        assert updated.replace(b,b'',1)==raw
        old[n]=raw;new[n]=updated;blocks[n]=b
        assert not p.with_name(p.name+'.round78.tmp').exists()
    docsha={n:{'before':DOCS[n],'after':hashlib.sha256(b).hexdigest(),'original_title_and_body_byte_preserved':True,'hash_role':'editable_human_document_change_record_not_decision_authority'} for n,b in new.items()}
    decision={'round':78,'status':'completed_diagnostic_keep_existing_local_priority',
'decision':'keep_v20_priority_v21_alternative_after_conditional_prior_seed_diagnostic','recorded_at_utc':stamp,
'private_priority':selector['private_priority'],'private_priority_sha256':selector['sha256'],'alternative':selector['backup'],'alternative_sha256':selector['backup_sha256'],
'known_public_best':{'path':selector['known_public_best_filename'],'MAE_TWD':selector['known_public_best_mae']},
'private_mae':None,'final_version_chosen':False,'selector_updated':False,'registry_updated':False,'new_CSV':False,'new_fits':0,'GitHub_published':False,
'reasons':{'v20_vs_v19_overall_weighted_improved_pairs':9,'v21_vs_v19_overall_weighted_improved_pairs':9,'pairs':9,
'confirmation_v20_better_than_v21_prior_seeds':3,'development_v20_testmix_worse_than_v19_prior_seeds':3},
'limitations':['Only prior seed varies: area seed42 and other branches fixed.','Only seed42 corresponds to delivered CSVs; other seeds are research compositions.','No fresh holdout or correction for repeated-window selection.','No seed picking or averaging; sign consistency is not Private probability.'],
'final_tail':{'target_free':True,'no_extra_area_multiplication_in_checked_composition':True,'maximum_v20_v21_TWD':max_twd,'maximum_is_commercial_use':True,'shared_extrapolation_risk_remains':True,'true_test_price_known':False,'price_cap_added':False,'ID_specific_changes':False,'interpretation':'Arithmetic consistency does not validate upstream measurements or prices.'},
'next_research_at_record':{'round':79,'status':'fixed_plan_driver_preparation_not_fit','fit_started':False,'plan':'reports/round79_plan.md'},
'input_sha256':inputs,'explanation_sha256_at_record':notes,'human_document_updates':docsha,
'main_elapsed_seconds':result['elapsed_seconds'],'outer_elapsed_seconds':execution['elapsed_seconds'],
'protected_selector_sha256_before':PINS['outputs/current_private_candidate.json'],'protected_registry_sha256_before':registry_sha,
'program_scope':'Saved JSON and human Markdown only; no arrays, models, scoring, prediction, CSV or selector writes.'}
    for n,b in new.items():
        p=ROOT/n;assert p.read_bytes()==old[n],n
        temp=p.with_name(p.name+'.round78.tmp')
        with temp.open('xb') as f:f.write(b)
        os.replace(temp,p)
    assert sha(ROOT/'outputs/current_private_candidate.json')==PINS['outputs/current_private_candidate.json'] and sha(registry)==registry_sha
    for n,b in blocks.items():assert (ROOT/n).read_bytes().replace(b,b'',1)==old[n]
    decision['protected_selector_sha256_after']=sha(ROOT/'outputs/current_private_candidate.json');decision['protected_registry_sha256_after']=sha(registry)
    decision['execution']={'exit_code':0,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False}
    data=(json.dumps(decision,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8');h=hashlib.sha256(data).hexdigest()
    with (OUT/'decision.json').open('xb') as f:f.write(data)
    with (OUT/'decision.sha256').open('x',encoding='utf8') as f:f.write(h+'\n')
    print(json.dumps({'status':decision['status'],'decision_sha256':h,'source_sha256':sha(Path(__file__)),'human_document_sha256':{n:v['after'] for n,v in docsha.items()},'selector_unchanged':True,'registry_unchanged':True,'elapsed_seconds':decision['execution']['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.parse_args();run()
