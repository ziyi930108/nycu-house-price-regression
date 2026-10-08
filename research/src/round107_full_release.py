"""One gated R107 full-data fit and separately verified exploratory v25 release."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import time
import traceback
import numpy as np
import round100_contract as c
import round107_final_candidate as final
import round107_108_research as history

BASE='outputs/round107/final'
REQUEST=BASE+'/request.json'
SEAL=BASE+'/seal.json'
CSV='outputs/submission_v25_trial.csv'
ROWS='outputs/research_csv_intermediate_20261008/v25_rows.json'
PLAN='reports/round107_final_plan.md'
TRAINER='src/round107_weighted_smooth_train.py'
BUILDER='outputs/research_resume_20261008/build_submission.mjs'
NODE=Path('C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
REPLAY_KEYS={'NN_native_mean','v22_NN_base','v22_NN_reference','v22_NN_spatial','v22_NN_median',
    'unit_spatial_fallback','v22_unit_base','v22_unit_reference','v22_unit_spatial',
    'v22_full_unrounded_median','v22_CSV_all_IDs_prices_order_and_whole_SHA'}
LAYERS={'native','base','reference','spatial','median'}


def rounded_prices(values):
    values=np.asarray(values)
    c.need(values.shape==(47073,) and np.isfinite(values).all() and (values>=0).all(),
        'Valid full prediction before integer conversion')
    rounded=np.maximum(np.rint(values),1)
    c.need((rounded<=2**53-1).all(),'CSV prices must be Node safe integers')
    return rounded.astype(np.int64)


def freeze():
    c.need(not c.safe(REQUEST).exists() and not c.safe(CSV).exists(),'Preserve final request/CSV')
    results=c.read('outputs/round107/results.json')
    c.need(results['gates']['exploratory_submission']['passed'],'Exploratory gate required')
    review=c.read('outputs/research_resume_20261008/postfit_review.json')
    c.need(review['status']=='passed','Independent postfit review required')
    c.need(review['results']['107']==c.record('outputs/round107/results.json')
        and review['manual_gate_verdicts']['107']['exploratory_submission_eligible'],
        'Reviewed R107 results and manual exploratory gate required')
    c.verify_hashes(review['source_sha256'])
    for key in ('request','seal','evaluation_execution'):
        c.need(review[key]==c.record(review[key]['path']),'Reviewed research evidence changed')
    configurations=[]
    for stage in c.STAGES:
        rec=review['artifacts']['107/'+stage]['training_metadata']
        c.need(rec==c.record(rec['path']),'Reviewed R107 training metadata changed')
        metadata=c.read(rec['path'])
        c.need(metadata['source_sha256']==c.sha(TRAINER),'Reviewed R107 trainer identity')
        configurations.append(metadata['configuration'])
    c.need(all(config==configurations[0] for config in configurations),'Three-window recipe must agree')
    preflight=c.read('outputs/research_resume_20261008/final_release_preflight.json')
    c.need(preflight['status']=='passed','Independent final release source review required')
    c.verify_hashes(preflight['source_sha256'])
    spec=final.specification()
    sources=c.source_hashes(__file__,TRAINER)
    sources[BUILDER]=c.sha(BUILDER)
    inputs=dict(spec['input_sha256'])
    inputs.update(review['source_sha256'])
    for stage in c.STAGES:
        rec=review['artifacts']['107/'+stage]['training_metadata'];inputs[rec['path']]=rec['sha256']
    for key in ('request','seal','evaluation_execution'):
        inputs[review[key]['path']]=review[key]['sha256']
    for name in (PLAN,'outputs/round107/results.json','outputs/research_resume_20261008/postfit_review.json',
                 'outputs/research_resume_20261008/final_context_preflight.json',
                 'outputs/research_resume_20261008/final_release_preflight.json'):
        inputs[name]=c.sha(name)
    c.verify_hashes(sources);c.verify_hashes(inputs)
    c.safe(BASE).mkdir(parents=True,exist_ok=False)
    c.write(REQUEST,{'status':'frozen_before_one_exploratory_final_fit','created_at_utc':c.now(),
        'sources':sources,'inputs':inputs,'context':spec,'protected_before':history.protected(),
        'expected_training_configuration':configurations[0],
        'gates':results['gates'],'new_fits':1,'CSV':CSV,'query_labels_available':False,
        'selected_submission_preserved':'v22','Public_score':None,'Private_score':None})


def request():
    r=c.read(REQUEST)
    c.need(r['status']=='frozen_before_one_exploratory_final_fit','Frozen request status')
    c.verify_hashes(r['sources']);c.verify_hashes(r['inputs']);c.verify_hashes(r['protected_before'])
    return r


def fit():
    r=request();tick=time.monotonic();failure=None
    try:
        ctx=final.load_context(r['context'])
        c.need(set(ctx['replay_checks'])==REPLAY_KEYS and all(v is True for v in ctx['replay_checks'].values()),
            'All eleven original v22 replay checks required')
        prepared=ctx['prepared']
        with np.load(c.safe(prepared['path']),allow_pickle=False) as z:
            c.need(set(z.files)==set(c.ARRAY_NAMES),'Original final nine arrays only')
            c.exact(z['train_y_million'],np.asarray(ctx['y']/1e6,dtype=np.float32),'Final canonical y alignment')
            c.need(z['train_num'].shape==(467221,144) and z['train_cat'].shape==(467221,23)
                and z['query_num'].shape==(47073,144) and z['query_cat'].shape==(47073,23),'Final NN dimensions')
            prep=c.read(r['context']['preparation']['path'])
            c.need(c.values_sha(z['train_sample_weight'])==prep['train_sample_weight_sha256'],'Final saved date weights')
            original=c.read(r['context']['original_training']['path'])
            for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
                c.need(c.values_sha(z[key])==original[key+'_sha256'],'Final saved '+key+' identity')
            c.need(z['cardinalities'].tolist()==original['cardinalities'],'Original final categorical cardinalities')
            array_identities={key:c.identity(z[key]) for key in c.ARRAY_NAMES}
        cmd=[str(c.ROOT/'.venv_torch/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.safe(TRAINER)),
             '--data',str(c.safe(prepared['path'])),'--out',str(c.safe(BASE)/'model')]
        with (c.safe(BASE)/'train.log').open('x',encoding='utf8') as log:
            process=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        c.need(process.returncode==0,'Final fit failed, preserve without automatic retry')
        meta=c.read(BASE+'/model/training.json')
        c.need(meta['status']=='completed' and meta['saved_model_reload_exact'] and meta['reload_rows']==47073
            and meta['fit_rows']==467221 and not meta['query_labels_used']
            and meta['input_sha256']==prepared['sha256'],'Final native model contract')
        c.need(meta['source_sha256']==r['sources'][TRAINER]
            and meta['configuration']==r['expected_training_configuration']
            and [e['epoch'] for e in meta['epochs']]==list(range(1,31))
            and meta['smooth_l1_beta_million']==.1 and meta['smooth_l1_beta_TWD']==100000
            and not meta['epoch_selection_uses_training_metrics'] and not meta['query_labels_available']
            and not meta['sample_weights_batch_renormalized']
            and meta['numeric_count']==144 and meta['cardinalities']==original['cardinalities'],
            'Fixed reviewed SmoothL1 final training recipe')
        for key in ('train_price_factor_million_sha256','query_price_factor_million_sha256',
                    'train_sample_weight_sha256','initial_training_median_ratio','initial_head_bias'):
            c.need(meta[key]==original[key],'Final original prepared contract: '+key)
        native=np.load(c.safe(BASE+'/model/predictions_twd.npy'),allow_pickle=False)
        chain,_,_=final.apply_prior(ctx,native)
        c.need(set(chain)==LAYERS,'Five final prediction layers required')
        c.exact(chain['native'],native,'Composed native prediction unchanged')
        rounded=rounded_prices(chain['median'])
        records={k:c.save_array(c.safe(BASE)/(k+'.npy'),v) for k,v in chain.items()}
        final.verify_unchanged(ctx);request()
        c.need(history.protected()==r['protected_before'],'Only new artifacts allowed before CSV')
        c.write(SEAL,{'status':'full_model_prediction_sealed','request':c.record(REQUEST),
            'model':c.record(BASE+'/model/model.pt'),'training':c.record(BASE+'/model/training.json'),
            'layers':records,'v22_replay_checks':ctx['replay_checks'],'new_fits':1,
            'prepared':prepared,'prepared_array_identities':array_identities,
            'fit_seconds':meta['fit_seconds'],'elapsed_seconds':time.monotonic()-tick,
            'query_labels_used':False,'new_CSV':False})
        c.safe(ROWS).parent.mkdir(parents=True,exist_ok=True)
        c.write(ROWS,{'status':'full_model_prediction_sealed','seal':c.record(SEAL),
            'rounding':'maximum(rint(final_TWD),1).astype(int64)',
            'rows':[[key,int(value)] for key,value in zip(ctx['ids'],rounded,strict=True)]})
    except Exception:failure=traceback.format_exc();raise
    finally:
        c.write(BASE+'/fit_execution.json',{'exit_code':1 if failure else 0,'failure':failure,
            'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False})


def release():
    r=request();tick=time.monotonic()
    seal=c.read(SEAL);c.need(seal['status']=='full_model_prediction_sealed'
        and seal['request']==c.record(REQUEST),'Final seal identity')
    for rec in [seal['model'],seal['training']]:
        c.need(c.record(rec['path'])==rec,'Sealed model/training identity')
    c.need(set(seal['layers'])==LAYERS,'Five sealed layers required')
    layers={key:c.load_array(rec) for key,rec in seal['layers'].items()}
    c.need(set(seal['v22_replay_checks'])==REPLAY_KEYS and all(v is True for v in seal['v22_replay_checks'].values()),
        'Original v22 replay evidence')
    rows=c.read(ROWS);c.need(rows['status']=='full_model_prediction_sealed' and rows['seal']==c.record(SEAL),'CSV input identity')
    p=layers['median'];rounded=rounded_prices(p)
    # Re-read independently from the bound sample, not only the authored input.
    with c.safe(r['context']['sample_path']).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);c.need(next(reader)==['編號','總價元'],'Sample headers')
        sample_rows=list(reader)
    ids=[v[0] for v in sample_rows]
    c.need(len(ids)==47073 and len(set(ids))==47073
           and rows['rows']==[[key,int(value)] for key,value in zip(ids,rounded,strict=True)],'CSV input values/order')
    cmd=[str(NODE),str(c.safe(BUILDER)),str(c.safe(ROWS)),str(c.safe(CSV)),str(c.safe(BASE))]
    with (c.safe(BASE)/'csv_authoring.log').open('x',encoding='utf8') as log:
        process=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    c.need(process.returncode==0,'CSV authoring failed; preserve evidence')
    with c.safe(CSV).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);c.need(next(reader)==['編號','總價元'],'Final CSV exact headers')
        values=list(reader)
    c.need(len(values)==47073 and all(len(v)==2 for v in values),'Final CSV 47073 two-field rows')
    c.need([v[0] for v in values]==ids and [v[1] for v in values]==[str(v) for v in rounded],
        'Independent whole-file sample ID / integer prediction equality')
    c.need(c.safe(CSV).read_bytes().startswith(b'\xef\xbb\xbf'),'UTF8 BOM')
    author=c.read(BASE+'/csv_authoring_receipt.json')
    c.need(author['status']=='artifact_typed_range_exported_to_csv'
        and author['input_json_sha256']==c.sha(ROWS) and author['csv_sha256']==c.sha(CSV)
        and author['rows']==47073 and author['columns']==['編號','總價元']
        and author['full_workbook_values_exact'] and author['formula_count']==0 and author['utf8_bom'],
        'Artifact export identity')
    request()
    before=r['protected_before'];after=history.protected()
    c.need(set(after)==set(before)|{CSV} and all(after[k]==v for k,v in before.items()),'Original protected files preserved')
    original=np.load(c.safe(final.VECTORS['v22']),allow_pickle=False);diff=p-original
    receipt={'status':'exploratory_submission_ready_not_selected','created_at_utc':c.now(),
        'version':25,'round':107,'CSV':c.record(CSV),'rows':47073,'columns':['編號','總價元'],
        'request':c.record(REQUEST),'seal':c.record(SEAL),'authoring':c.record(BASE+'/csv_authoring_receipt.json'),
        'all_sample_ids_and_rounded_values_exact':True,'positive_integer_prices':True,
        'same_order_as_sample':True,'old_protected_unchanged':True,'query_labels_used':False,
        'Public_score':None,'Private_score':None,'automatic_Kaggle_upload_or_selection':False,
        'selected_submission_preserved':'v22','gate':r['gates'],
        'test_prediction_difference_not_MAE':{'mean_signed_TWD':float(diff.mean()),
            'mean_absolute_TWD':float(np.mean(np.abs(diff))),'max_absolute_TWD':float(np.max(np.abs(diff))),
            'changed_rounded_rows':int(np.count_nonzero(rounded!=np.maximum(np.rint(original),1).astype(np.int64)))},
        'prediction_min_TWD':float(p.min()),'prediction_max_TWD':float(p.max()),
        'elapsed_seconds':time.monotonic()-tick,'clean_environment_end_to_end_retraining_verified':False}
    c.write(BASE+'/release.json',receipt)
    print(json.dumps({'CSV':CSV,'sha256':receipt['CSV']['sha256'],'rows':47073,'status':receipt['status']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--fit',action='store_true');group.add_argument('--release',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.fit:fit()
    else:release()
