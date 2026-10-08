"""CPU final-release audit for R109/v26 or R110/v27, after release only.

--self-check reads no data. --release-complete requires actual successful
final request/seal/release receipts before loading arrays or final fit prices.
This tool's author also authored the final driver; root and the separate source
preflight provide independent author review. This audit is an additional
artifact/runtime check, not an independent-author source certification.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
import round107_final_candidate as final
import round109_110_full_release as driver
import round109_110_research as history


def same(left,right,name):c.need(left==right,name)


def locked(name):
    rec=c.record(name)
    same(c.safe(name).with_suffix('.sha256').read_text(encoding='ascii').strip(),rec['sha256'],
        'Receipt sidecar differs: '+str(name))
    return c.read(name),rec


def record_matches(rec):
    same(c.record(rec['path']),{key:rec[key] for key in ('path','sha256')},'Artifact record changed')


def timestamp(value):
    result=datetime.fromisoformat(value)
    c.need(result.tzinfo is not None,'Evidence timestamps need an explicit timezone')
    return result


def inventory_policy(before,after,own,peer,peer_proof=None):
    """Only a later, properly frozen peer release may extend an earlier audit."""
    c.need(own not in before and own in after,'Own CSV must be the one new release')
    c.need(set(before)<=set(after) and all(after[key]==value for key,value in before.items()),
        'Every originally protected file must retain its frozen hash')
    extras=set(after)-set(before)-{own}
    if not extras:return {'later_peer_allowed':False,'additional_CSVs':[]}
    c.need(extras=={peer} and peer_proof is not None,'No unrelated new protected CSVs are allowed')
    same(peer_proof['protected_before'],{**before,own:after[own]},
        'Later peer freeze must protect exactly the earlier frozen set plus own CSV')
    c.need(timestamp(peer_proof['own_release_at'])<=timestamp(peer_proof['peer_freeze_at'])
        <=timestamp(peer_proof['peer_release_at']),'Peer release chronology must follow this completed release')
    return {'later_peer_allowed':True,'additional_CSVs':[peer]}


def csv_bytes(ids,values):
    c.need(len(ids)==len(values) and len(set(ids))==len(ids)
        and all(isinstance(key,str) and key.isascii() and key.isdigit() for key in ids),
        'Unique digit-only sample IDs required for the exact CSV encoding')
    return ('\ufeff編號,總價元\r\n'+''.join(key+','+str(int(value))+'\r\n'
        for key,value in zip(ids,values,strict=True))).encode('utf8')


def guard(p):
    """Verify completed receipts and identities before deserializing prices."""
    release,rr=locked(p['base']+'/release.json')
    request,qr=locked(p['request']);seal,sr=locked(p['seal'])
    execution,er=locked(p['base']+'/fit_execution.json')
    same(release['status'],'exploratory_submission_ready_not_selected','No actual completed release')
    same((release['round'],release['version']),(p['round'],p['version']),'Release identity')
    same(release['request'],qr,'Release request anchor');same(release['seal'],sr,'Release seal anchor')
    same(request['status'],'frozen_before_one_exploratory_final_fit','Frozen request required')
    same(request['paths'],p,'Round-specific frozen paths');same(request['new_fits'],1,'One final fit')
    same(seal['status'],'full_model_prediction_sealed','Completed prediction seal')
    same(seal['request'],qr,'Seal request anchor')
    same((seal['round'],seal['version']),(p['round'],p['version']),'Seal identity')
    c.need(execution['exit_code']==0 and execution['failure'] is None
        and execution['automatic_retry'] is False,'Successful final fit required')
    for document in (request,release):
        c.need(document['Public_score'] is None and document['Private_score'] is None
            and document['selected_submission_preserved']=='v22','Unknown scores and v22 selection preserved')
    c.need(release['automatic_Kaggle_upload_or_selection'] is False and not release['query_labels_used']
        and not seal['query_labels_used'] and not request['query_labels_available'],'No query labels or Kaggle action')
    c.verify_hashes(request['sources']);c.verify_hashes(request['inputs']);c.verify_hashes(request['protected_before'])
    record_matches(request['postfit_review'])
    review=c.read(request['postfit_review']['path'])
    results=c.read(f"outputs/round{p['round']}/results.json")
    driver.gate(review,results,p['round'])
    same(review['results'][str(p['round'])],c.record(f"outputs/round{p['round']}/results.json"),'Postfit result anchor')
    source_preflight=c.read(p['preflight'])
    c.need(source_preflight['status']=='passed' and source_preflight['round']==p['round']
        and source_preflight['version']==p['version'],'Separate source preflight required')
    c.verify_hashes(source_preflight['source_sha256'])
    same(request['gates'],release['gate'],'Same frozen eligibility gate')
    start,startrec=locked(p['base']+'/fit_started.json')
    same(start['request'],qr,'Fit invocation identity')
    c.need(timestamp(request['created_at_utc'])<=timestamp(start['created_at_utc'])
        <=timestamp(release['created_at_utc']),'Freeze-fit-release timestamp order')
    record_matches(seal['training']);model_records={};driver.bind_records(seal['model_artifacts'],model_records)
    same(release['CSV'],c.record(p['CSV']),'Actual CSV file identity')
    author=c.read(p['base']+'/csv_authoring_receipt.json')
    same(release['authoring'],c.record(p['base']+'/csv_authoring_receipt.json'),'Actual authoring receipt identity')
    return request,seal,release,author,{'request':qr,'seal':sr,'release':rr,'fit_execution':er,'fit_started':startrec}


def checkpoint_cpu(path):
    """Inspect weights-only on CPU without creating a model or doing inference."""
    code="""import hashlib,json,sys,torch
p=torch.load(sys.argv[1],weights_only=True,map_location='cpu')
assert set(p)=={'numeric_count','cardinalities','configuration','parameter_selection','parameter_state_epochs','state_dict'}
s=p.pop('state_dict');assert isinstance(s,dict) and s
params={}
for k,v in s.items():
 assert isinstance(k,str) and isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.dtype==torch.float32 and bool(torch.isfinite(v).all())
 params[k]={'shape':list(v.shape),'dtype':str(v.dtype),'values_sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
print(json.dumps({'payload':p,'named_parameter_order':list(s),'parameters':params,'parameter_count':sum(v.numel() for v in s.values()),'CPU_only':True,'native_inference':False},allow_nan=False))
"""
    process=subprocess.run([str(ROOT/'.venv_torch/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(c.safe(path))],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',timeout=120)
    c.need(process.returncode==0,'CPU checkpoint inspection failed: '+process.stderr)
    return json.loads(process.stdout)


def check_nn(meta,r,ctx,p):
    original=c.read(r['context']['original_training']['path']);model=c.safe(p['base'])/'model'
    records=driver.check_nn(meta,r,original,model);diagnostics={}
    for key in ('averaged_candidate','epoch30_control'):
        item=meta[key];saved=checkpoint_cpu(model/item['model_file']);payload=saved['payload']
        same(payload['configuration'],meta['configuration'],'Actual saved NN configuration')
        same(payload['parameter_selection'],item['parameter_selection'],'Actual saved parameter selection')
        same(payload['parameter_state_epochs'],item['parameter_state_epochs'],'Actual selected state epochs')
        same((payload['numeric_count'],payload['cardinalities']),(meta['numeric_count'],meta['cardinalities']),
            'Actual NN checkpoint feature dimensions')
        average=meta['tail_parameter_average']
        same(saved['named_parameter_order'],average['named_parameter_order'],'Actual checkpoint parameter order')
        same({name:value['shape'] for name,value in saved['parameters'].items()},average['parameter_shapes'],
            'Actual checkpoint parameter shapes')
        same(saved['parameter_count'],meta['parameter_count'],'Actual parameter count')
        diagnostics[key]={'records':records[key],'CPU_checkpoint':saved}
    return diagnostics


def check_ctr(meta,r,ctx,weights,p):
    from catboost import CatBoostRegressor
    import round110_target_borders_unit as helper
    model_dir=c.safe(p['base'])/'model';driver.check_ctr(meta,r,ctx,weights,model_dir)
    state=c.read(model_dir/'feature_state.json');model=CatBoostRegressor()
    model.load_model(str(model_dir/'model.cbm'))
    actual=helper.check_model(model,state)
    comparison=helper.compare_reload_diagnostics(meta['reloaded_native_diagnostics'],actual)
    frames={'fit_X':ctx['X'],'query_X':ctx['Xq'],'fit_geo':ctx['G'],'query_geo':ctx['Gq'],
        'fit_augmented':helper.original.augment_features(ctx['X'],ctx['G']),
        'query_augmented':helper.original.augment_features(ctx['Xq'],ctx['Gq'])}
    same(meta['frame_sha256'],{name:helper.base.frame_sha(frame) for name,frame in frames.items()},
        'Actual R110 canonical full-fit/query frame identities')
    with tempfile.TemporaryDirectory(prefix='r110_final_review_') as folder:
        path=Path(folder)/'model.json';model.save_model(str(path),format='json')
        document=json.loads(path.read_text(encoding='utf8'))
        reproduced=helper.summarize_target_borders(document,2999)
    for key,value in reproduced.items():same(meta['target_border_summary'][key],value,'Actual CTR usage '+key)
    return {'loaded_native_diagnostics':actual,'reload_serialization_comparison':comparison,
        'observed_target_border_summary':reproduced,'CPU_native_model_loaded':True,'native_inference':False}


def protected_inventory(p,r,release):
    before=r['protected_before'];after=history.protected()
    peer_n=110 if p['round']==109 else 109;peer=driver.paths(peer_n,driver.VERSIONS[peer_n])
    extras=set(after)-set(before)-{p['CSV']};proof=None;records=None
    if extras:
        c.need(extras=={peer['CSV']},'Unrelated protected inventory additions')
        pr,ps,pl,_,peer_records=guard(peer)
        same(pl['CSV'],c.record(peer['CSV']),'Later peer CSV identity')
        proof={'protected_before':pr['protected_before'],'own_release_at':release['created_at_utc'],
            'peer_freeze_at':pr['created_at_utc'],'peer_release_at':pl['created_at_utc']}
        records=peer_records
    policy=inventory_policy(before,after,p['CSV'],peer['CSV'],proof)
    return {'frozen_protected_before':before,'protected_at_review':after,**policy,
        'later_peer_evidence':records,'later_peer_timestamp_proof':proof}


def run(round_number):
    tick=time.monotonic();started=c.now();p=driver.paths(round_number,driver.VERSIONS[round_number])
    output=driver.FOLDER+f'round{round_number}_final_release_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),
        'Exclusive final review output; preserve previous evidence')
    r,seal,release,author,records=guard(p)
    c.need(timestamp(release['created_at_utc'])<=timestamp(started),'Review must follow completed release')
    inventory=protected_inventory(p,r,release)
    # All completion and file identities above must pass before numerical loads.
    ctx=final.load_context(r['context']);original,array_ids,weights=driver.prepared(ctx)
    same(array_ids,seal['prepared_array_identities'],'Every one of nine frozen NPZ arrays')
    same(seal['prepared'],ctx['prepared'],'Original nine-array archive identity')
    same(ctx['replay_checks'],seal['v22_replay_checks'],'Original v22 CPU replay evidence')
    c.need(set(ctx['replay_checks'])==driver.shared.REPLAY_KEYS and all(ctx['replay_checks'].values()),
        'All eleven original v22 replay checks')
    meta=c.read(seal['training']['path'])
    same(set(seal['layers']),driver.shared.LAYERS,'Five candidate layers')
    layers={key:c.load_array(rec) for key,rec in seal['layers'].items()}
    model=c.safe(p['base'])/'model';controls=None
    if round_number==109:
        model_diagnostics=check_nn(meta,r,ctx,p)
        native=np.load(model/'predictions_twd.npy',allow_pickle=False)
        chain=final.apply_prior(ctx,native)[0]
        same(set(seal['same_trajectory_epoch30_layers']),driver.shared.LAYERS,'Five epoch30 control layers')
        controls={key:c.load_array(rec) for key,rec in seal['same_trajectory_epoch30_layers'].items()}
        control_native=np.load(model/'predictions_epoch30_twd.npy',allow_pickle=False)
        control_chain=final.apply_prior(ctx,control_native)[0]
        for key in driver.shared.LAYERS:c.exact(control_chain[key],controls[key],'Actual epoch30 '+key+' replay')
    else:
        same(seal['same_trajectory_epoch30_layers'],None,'R110 has no NN epoch30 control')
        model_diagnostics=check_ctr(meta,r,ctx,weights,p)
        native=np.load(model/'native.npy',allow_pickle=False);chain=driver.unit_chain(ctx,native)
        original_unit=driver.unit_chain(ctx,ctx['vectors']['unit_native'])
        for key,saved in [('base','unit_base'),('reference','unit_reference'),('spatial','unit'),('median','v22')]:
            c.exact(original_unit[key],ctx['vectors'][saved],'Original unit/v22 replay '+key)
    for key in driver.shared.LAYERS:c.exact(chain[key],layers[key],'Actual final '+key+' CPU replay')
    final.verify_unchanged(ctx)
    prediction=layers['median'];rounded=driver.shared.rounded_prices(prediction)
    rows=c.read(p['rows']);same(rows['status'],'full_model_prediction_sealed','Authoring input status')
    same(rows['seal'],records['seal'],'Authoring input seal')
    with c.safe(r['context']['sample_path']).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);same(next(reader),['編號','總價元'],'Sample headers');sample=list(reader)
    c.need(len(sample)==47073 and all(len(row)==2 for row in sample),'Sample rows/columns')
    ids=[row[0] for row in sample];same(ids,ctx['ids'],'Original sample/canonical query order')
    expected_rows=[[key,int(value)] for key,value in zip(ids,rounded,strict=True)]
    same(rows['rows'],expected_rows,'All authored sample IDs/once-rounded prices')
    same(c.safe(p['CSV']).read_bytes(),csv_bytes(ids,rounded),'Every CSV byte, BOM, header and CRLF')
    with c.safe(p['CSV']).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);same(next(reader),['編號','總價元'],'CSV headers');actual=list(reader)
    same(actual,[[key,str(value)] for key,value in zip(ids,rounded,strict=True)],'CSV reader all 47073 IDs/prices')
    c.need(author['status']=='artifact_typed_range_exported_to_csv' and author['rows']==47073
        and author['columns']==['編號','總價元'] and author['input_json_sha256']==c.sha(p['rows'])
        and author['csv_sha256']==c.sha(p['CSV']) and author['full_workbook_values_exact']
        and author['formula_count']==0 and author['utf8_bom'],'Artifact authoring contract')
    preview=c.safe(p['base']+'/'+author['preview'])
    c.need(preview.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'),'Preview PNG file')
    diff=prediction-ctx['vectors']['v22']
    changes={'mean_signed_TWD':float(diff.mean()),'mean_absolute_TWD':float(np.mean(np.abs(diff))),
        'max_absolute_TWD':float(np.max(np.abs(diff))),
        'changed_rounded_rows':int(np.count_nonzero(rounded!=driver.shared.rounded_prices(ctx['vectors']['v22'])))}
    same(changes,release['test_prediction_difference_not_MAE'],'Reported differences are exact, not MAE')
    c.need(meta['saved_model_reload_exact'] and not meta['query_labels_used'] and not meta['query_labels_available'],
        'Native trainer/helper reload and query scope')
    c.need(release['clean_environment_end_to_end_retraining_verified'] is False,'Reproducibility scope')
    c.verify_hashes(r['sources']);c.verify_hashes(r['inputs']);c.verify_hashes(r['protected_before'])
    same(protected_inventory(p,r,release),inventory,'Inventory changed during audit')
    payload={'status':'passed','created_at_utc':c.now(),'review_started_at_utc':started,
        'reviewer':'final_driver_author_CPU_artifact_audit_after_root_and_source_preflight',
        'scope':'Post-release read-only CPU artifact/model-payload/saved-native composition and complete CSV review; no fits, GPU or native model inference.',
        'round':round_number,'version':p['version'],**records,
        'authoring':c.record(p['base']+'/csv_authoring_receipt.json'),'training':seal['training'],
        'CSV':c.record(p['CSV']),'source_sha256':r['sources'],'review_tool':c.record(__file__),
        'review_source_sha256':c.source_hashes(__file__,driver.TRAINER,driver.HELPER),
        'source_preflight':c.record(p['preflight']),'postfit_review':r['postfit_review'],
        'prepared':ctx['prepared'],'all_nine_array_identities':array_ids,
        'v22_replay_checks':ctx['replay_checks'],'five_layers_file_and_values_hashes_verified':seal['layers'],
        'five_layers_independent_CPU_saved_native_composition_bytes_exact':True,
        'epoch30_five_layers_replayed':controls is not None,'model_identity_diagnostics':model_diagnostics,
        'trainer_saved_model_reload_exact':True,'independent_native_inference_performed':False,
        'CSV_rows':47073,'all_sample_ids_and_rounded_values_exact':True,
        'all_sample_IDs_prices_order_headers_and_encoding_exact':True,
        'CSV_once_rounded_from_unrounded_median':True,'positive_safe_integer_prices':True,
        'preview':c.record(preview),'protected_inventory':inventory,'old_protected_files_unchanged':True,
        'only_new_protected_inventory_entry':p['CSV'] if not inventory['later_peer_allowed'] else None,
        'prediction_difference_not_MAE':changes,'query_or_test_labels_loaded':False,'query_labels_used':False,
        'review_new_fits':0,'review_GPU_calls':0,'review_native_models_loaded':2 if round_number==109 else 1,
        'review_native_model_inference':False,'review_new_CSV':False,
        'selected_submission_preserved':'v22','automatic_Kaggle_upload_or_selection':False,
        'Public_score':None,'Private_score':None,'elapsed_seconds':time.monotonic()-tick,
        'limits':['Reviewer also authored the final driver; this is not independent-author source review. Root and the separately hash-bound final preflight supplied source review before fit.',
            'Native checkpoints/CBM are inspected on CPU without inference; exact full-query native reload is evidence from the frozen trainer/helper.',
            'R109 individual ten epoch states were not retained, so the audit cannot independently re-sum actual trained snapshots.',
            'Saved-native CPU composition replay does not establish clean-environment six-model retraining or Private improvement.',
            'Protected metadata uses UTC freeze/fit-invocation/release records; native metadata without timestamps is hash-bound, not assigned an invented creation time.']}
    c.write(output,payload)
    print(json.dumps({'status':'passed','round':round_number,'review':c.record(output),'CSV':payload['CSV'],
        'elapsed_seconds':payload['elapsed_seconds']},ensure_ascii=False),flush=True)


def self_check():
    checks=[]
    def ok(name,value):c.need(value,name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Invalid audit contract accepted: '+name)
    before={'old.csv':'a','v25.csv':'b'};own='v26.csv';peer='v27.csv'
    after={**before,own:'c'}
    ok('own_only_inventory',not inventory_policy(before,after,own,peer)['later_peer_allowed'])
    proof={'protected_before':after,'own_release_at':'2026-10-08T06:00:00+00:00',
        'peer_freeze_at':'2026-10-08T06:01:00+00:00','peer_release_at':'2026-10-08T06:02:00+00:00'}
    ok('valid_later_peer',inventory_policy(before,{**after,peer:'d'},own,peer,proof)['later_peer_allowed'])
    reject('changed_old_file',lambda:inventory_policy(before,{**after,'old.csv':'x'},own,peer))
    reject('unknown_added_CSV',lambda:inventory_policy(before,{**after,'v99.csv':'e'},own,peer))
    reject('missing_peer_proof',lambda:inventory_policy(before,{**after,peer:'d'},own,peer))
    bad={**proof,'protected_before':before}
    reject('peer_did_not_freeze_own_CSV',lambda:inventory_policy(before,{**after,peer:'d'},own,peer,bad))
    bad={**proof,'peer_freeze_at':'2026-10-08T05:59:00+00:00'}
    reject('peer_freeze_before_own_release',lambda:inventory_policy(before,{**after,peer:'d'},own,peer,bad))
    reject('own_CSV_missing',lambda:inventory_policy(before,before,own,peer))
    reject('naive_timestamp_rejected',lambda:timestamp('2026-10-08T06:00:00'))
    ok('exact_digit_ID_UTF8_BOM_CRLF',csv_bytes(['0001','2'],np.array([1,4],dtype=np.int64))
        =='\ufeff編號,總價元\r\n0001,1\r\n2,4\r\n'.encode('utf8'))
    reject('duplicate_IDs',lambda:csv_bytes(['1','1'],[1,2]))
    reject('unsafe_CSV_field',lambda:csv_bytes(['1,2'],[3]))
    reject('non_ASCII_ID',lambda:csv_bytes(['１２'],[3]))
    for name in c.source_hashes(__file__,driver.TRAINER,driver.HELPER):
        compile(c.safe(name).read_text(encoding='utf-8-sig'),name,'exec')
    ok('Python_source_closure_memory_compile',True)
    return {'status':'passed','check_count':len(checks),'checks':checks,'real_data_read':False,
        'query_prices_or_scores_read':False,'new_fits':0,'GPU_calls':0,'new_CSV':False,
        'true_final_review_executed':False,'review_tool':c.record(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',dest='round_number',type=int,choices=(109,110))
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-check',action='store_true');group.add_argument('--release-complete',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    else:
        if args.round_number is None:parser.error('--round is explicitly required')
        run(args.round_number)
