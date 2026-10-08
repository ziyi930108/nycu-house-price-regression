"""R117 original R63 ratio/weighted-MAE trainer with transaction-OOF factors.

Only fit price-factor construction changes in the caller. The actual R47 model,
initial-bias formula, loss, RNG, optimizer, batches and schedule remain R63.
This process validates exactly nine numeric arrays, never fold/date provenance
or query labels. Root must freeze the four-origin plan and preflight before fit.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import round47_nn_train as prior
import round63_recency_weighted_prior_train as baseline

ROOT = Path(__file__).resolve().parents[1]
PARENT_TRAINER = Path(baseline.__file__)
PARENT_TRAINER_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
ARCHITECTURE_SHA = '68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
MODEL_CLASS = 'round47_nn_train.EmbeddingMLP'
FOLD_POLICY = 'transaction-permutation-5-fold'
EmbeddingMLP = prior.EmbeddingMLP
ARRAY_NAMES = ('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
               'train_price_factor_million','query_price_factor_million','train_sample_weight')
OUTPUT_UNITS = baseline.OUTPUT_UNITS
PRICE_FACTOR_FORMULA = baseline.PRICE_FACTOR_FORMULA
TRAIN_FACTOR_CONSTRUCTION = 'balanced transaction-fold OOF fit-label prior supplied by driver; seed42 permutation then arange%5'
QUERY_FACTOR_CONSTRUCTION = baseline.QUERY_FACTOR_CONSTRUCTION
BASELINE_CONFIG = copy.deepcopy(baseline.CONFIG)
PROVENANCE_CONFIG_OVERRIDES = {'train_price_factor_construction': TRAIN_FACTOR_CONSTRUCTION}
CONFIG = {**copy.deepcopy(BASELINE_CONFIG), **copy.deepcopy(PROVENANCE_CONFIG_OVERRIDES)}
weighted_loss = baseline.weighted_loss
sample_weight_values = baseline.sample_weight_values
sample_weight_summary = baseline.sample_weight_summary
price_factor_values = baseline.price_factor_values
predict_ratio = baseline.predict_ratio
to_twd = baseline.to_twd
predict = baseline.predict
write = baseline.write


def need(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def source_hashes():
    return {p.resolve().relative_to(ROOT).as_posix():sha(p) for p in
            (Path(__file__),PARENT_TRAINER,Path(prior.__file__),baseline.PARENT_TRAINER)}


def identity(a):
    return {'dtype':str(a.dtype),'shape':list(a.shape),
            'values_sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def check_configuration():
    need(sha(PARENT_TRAINER)==PARENT_TRAINER_SHA and sha(prior.__file__)==ARCHITECTURE_SHA,
         'Original R63/R47 sources changed')
    baseline.check_configuration()
    need(BASELINE_CONFIG==baseline.CONFIG and CONFIG=={**BASELINE_CONFIG,**PROVENANCE_CONFIG_OVERRIDES}
         and PROVENANCE_CONFIG_OVERRIDES=={'train_price_factor_construction':TRAIN_FACTOR_CONSTRUCTION},
         'Only transaction prior descriptive provenance may change from R63')
    need(set(CONFIG)==set(BASELINE_CONFIG) and [k for k in CONFIG if CONFIG[k]!=BASELINE_CONFIG[k]]
         ==['train_price_factor_construction'], 'No actual ML configuration or extra key may change')
    need(all(CONFIG[k]==v for k,v in prior.CONFIG.items() if k!='loss'), 'Original R47 training/architecture recipe changed')
    need(CONFIG['loss']==baseline.CONFIG['loss'] and CONFIG['loss_formula']==baseline.CONFIG['loss_formula']
         and CONFIG['initial_bias']==baseline.CONFIG['initial_bias'] and EmbeddingMLP is prior.EmbeddingMLP
         and weighted_loss is baseline.weighted_loss, 'Original ratio MLP/weighted MAE/initialization differs')


def materialize_archive(archive):
    need(len(archive.files)==len(ARRAY_NAMES) and set(archive.files)==set(ARRAY_NAMES),
         'Exactly nine original NPZ keys before any array read; no query y')
    return {key:archive[key] for key in ARRAY_NAMES}


def validate_inputs(a):
    need(set(a)==set(ARRAY_NAMES) and all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in a.values()),
         'Exactly nine nonobject numeric arrays')
    xn,xc,y,qn,qc,card,af,qaf,sw=[a[key] for key in ARRAY_NAMES]
    need(xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and len(xn)>0 and len(qn)>0, 'Nonempty input matrices')
    need(xn.shape[1]==qn.shape[1]==144 and xc.shape[1]==qc.shape[1]==23, 'Original 144 numeric/23 categorical inputs')
    need(len(xn)==len(xc)==len(y) and len(qn)==len(qc) and y.ndim==1, 'Aligned input rows')
    need(xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64,
         'Original float32 numeric/y and int64 category dtypes')
    need(card.dtype==np.int64 and card.shape==(23,), 'Original 23 int64 cardinalities')
    need(np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all(),
         'Finite numeric inputs and positive fit target')
    af=price_factor_values(af,len(xn));qaf=price_factor_values(qaf,len(qn));sw=sample_weight_values(sw,len(xn))
    cards=[int(k) for k in card]
    for i,k in enumerate(cards):
        need(k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all(),
             'Category codes outside original fit vocabulary')
    return xn,xc,y,qn,qc,cards,af,qaf,sw


def initial_bias(y, factor):
    need(isinstance(y,np.ndarray) and y.dtype==np.float32 and y.ndim==1 and len(y)>0
         and np.isfinite(y).all() and (y>0).all(), 'Positive original float32 fit target')
    price_factor_values(factor,len(y))
    median=float(np.median(y.astype(np.float64)/factor.astype(np.float64)))
    need(np.isfinite(median) and median>0, 'Finite positive fit ratio median')
    bias=median+float(np.log(-np.expm1(-median)))
    need(np.isfinite(bias) and np.isfinite(np.float32(bias)), 'Finite original inverseSoftplus head bias')
    return median,bias


def validate_checkpoint(checkpoint):
    check_configuration()
    need(isinstance(checkpoint,dict) and set(checkpoint)=={'numeric_count','cardinalities','configuration',
         'actual_model_class','architecture_source_sha256','state_dict'}, 'Exact saved checkpoint keyset')
    need(type(checkpoint['numeric_count']) is int and checkpoint['numeric_count']==144
         and type(checkpoint['cardinalities']) is list and len(checkpoint['cardinalities'])==23
         and all(type(k) is int and k>=1 for k in checkpoint['cardinalities'])
         and checkpoint['configuration']==CONFIG and checkpoint['actual_model_class']==MODEL_CLASS
         and checkpoint['architecture_source_sha256']==ARCHITECTURE_SHA, 'Original model/configuration checkpoint contract')
    model=EmbeddingMLP(checkpoint['numeric_count'],checkpoint['cardinalities'])
    expected=model.state_dict();saved=checkpoint['state_dict']
    need(isinstance(saved,dict) and set(saved)==set(expected), 'Exact original state-dict tensor names')
    for key,value in saved.items():
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==expected[key].dtype
             and value.shape==expected[key].shape and torch.isfinite(value).all().item(),
             'Finite original CPU checkpoint tensor dtype/shape: '+key)
    model.load_state_dict(saved,strict=True)
    need(type(model).__module__+'.'+type(model).__name__==MODEL_CLASS, 'Actual original model class')
    return model


def exact(a,b,name):
    need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name)


def save_prediction(path,values):
    with Path(path).open('xb') as f:np.save(f,values,allow_pickle=False)
    exact(values,np.load(path,allow_pickle=False),'Saved prediction bytes differ')
    return {'path':Path(path).relative_to(ROOT).as_posix(),'sha256':sha(path),**identity(values)}


def _run_once(data_path,out):
    start=time.monotonic();data_path=Path(data_path).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing R117 model directory; no retry/overwrite')
    check_configuration();sources=source_hashes();input_sha=sha(data_path)
    with np.load(data_path,allow_pickle=False) as archive:a=materialize_archive(archive)
    xn,xc,y,qn,qc,card,af,qaf,sw=validate_inputs(a)
    inputs={key:identity(value) for key,value in a.items()}
    median,inverse_median=initial_bias(y,af)
    # No added RNG-consuming operations before this unchanged R47 setup/model construction.
    device=prior.setup()
    out.mkdir(parents=True,exist_ok=False)
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':copy.deepcopy(CONFIG),
        'source_sha256':sha(__file__),'complete_source_sha256':sources,
        'parent_trainer_source':'src/round63_recency_weighted_prior_train.py','parent_trainer_source_sha256':PARENT_TRAINER_SHA,
        'architecture_source_sha256':ARCHITECTURE_SHA,'actual_model_class':MODEL_CLASS,
        'changed_configuration_fields':['train_price_factor_construction'],'added_configuration_fields':[],
        'provenance_configuration_overrides':copy.deepcopy(PROVENANCE_CONFIG_OVERRIDES),
        'ML_configuration_changes':[],'loss_configuration_changes':[],
        'training_objective_name':'weighted MAE','epoch_selection_uses_training_metrics':False,
        'input_sha256':input_sha,'input_arrays':inputs,
        'fit_rows':len(xn),'query_rows':len(qn),'numeric_count':xn.shape[1],'cardinalities':card,
        'query_labels_available':False,'query_labels_used':False,'prediction_units':'TWD','output_units':OUTPUT_UNITS,
        'price_factor_formula':PRICE_FACTOR_FORMULA,'price_factor_uses_fit_labels':True,
        'legacy_price_factor_group_word_means_cell_prior_not_fold_proof':True,
        'train_price_factor_construction':TRAIN_FACTOR_CONSTRUCTION,'query_price_factor_construction':QUERY_FACTOR_CONSTRUCTION,
        'fit_prior_exclusion_policy':FOLD_POLICY,'transaction_fold_provenance_verified_in_trainer':False,
        'price_factor_provenance_verified_in_trainer':False,'query_factor_original_exact_requires_driver_verification':True,
        'train_price_factor_million_sha256':inputs['train_price_factor_million']['values_sha256'],
        'query_price_factor_million_sha256':inputs['query_price_factor_million']['values_sha256'],
        'price_factor_million_summary':{name:{'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean(dtype=np.float64))}
            for name,v in [('train',af),('query',qaf)]},
        'train_sample_weight_sha256':inputs['train_sample_weight']['values_sha256'],
        'train_sample_weight_shape':list(sw.shape),'train_sample_weight_dtype':'float32',
        'train_sample_weight_summary':sample_weight_summary(sw),'sample_weight_provenance_verified_in_trainer':False,
        'sample_weights_recomputed_in_trainer':False,'sample_weights_batch_renormalized':False,
        'initial_training_median_ratio':median,'initial_head_bias':inverse_median,
        'initial_head_bias_float32':float(np.float32(inverse_median)),
        'initial_bias_formula_unchanged_from_R63':True,'initial_bias_value_may_change_with_fit_factor':True,
        'old_new_factor_ratio_and_bias_comparison_requires_driver':True,
        'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':True,'price_or_ratio_clipping':False,
        'new_CSV':False,'automatic_retry_used':False,'automatic_fallback_used':False}
    write(out/'run.json',meta)
    model=EmbeddingMLP(xn.shape[1],card).to(device)
    with torch.no_grad():model.head.bias.fill_(inverse_median)
    need(float(model.head.bias.detach().cpu().item())==meta['initial_head_bias_float32'],'Actual original head initialization')
    tn=torch.as_tensor(xn,device=device);tc=torch.as_tensor(xc,device=device);ty=torch.as_tensor(y,device=device)
    ta=torch.as_tensor(af,device=device);tw=torch.as_tensor(sw,device=device)
    qtn=torch.as_tensor(qn,device=device);qtc=torch.as_tensor(qc,device=device)
    del xn,xc,y,qn,qc,af,sw
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    epochs=[];fit_started=time.monotonic()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device)
        total=torch.zeros((),device=device);total_mae=torch.zeros((),device=device)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            index=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            ratio=model(tn[index],tc[index]);prediction=ratio*ta[index]
            loss=weighted_loss(prediction,ty[index],tw[index]);mae=(prediction.detach()-ty[index]).abs().mean()
            loss.backward();optimizer.step();total+=loss.detach()*len(index);total_mae+=mae*len(index)
        value=float((total/len(tn)).item());mae_value=float((total_mae/len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):raise FloatingPointError('Nonfinite original weighted MAE; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':mae_value,
                       'training_weighted_mae_million_with_dropout':value})
        schedule.step()
        if epoch==1 or epoch%5==0:print(json.dumps({'event':'epoch_complete',**epochs[-1],'elapsed_seconds':time.monotonic()-fit_started}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
    checkpoint={'numeric_count':meta['numeric_count'],'cardinalities':card,'configuration':copy.deepcopy(CONFIG),
        'actual_model_class':MODEL_CLASS,'architecture_source_sha256':ARCHITECTURE_SHA,
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    with (out/'model.pt').open('xb') as f:torch.save(checkpoint,f)
    ratios=predict_ratio(model,qtn,qtc);predictions=to_twd(ratios,qaf)
    loaded=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    restored=validate_checkpoint(loaded).to(device)
    reloaded_ratio=predict_ratio(restored,qtn,qtc);reloaded=to_twd(reloaded_ratio,qaf)
    exact(ratios,reloaded_ratio,'Full-query saved original ratio reload differs')
    exact(predictions,reloaded,'Full-query saved total TWD reload differs')
    artifacts={'ratio':save_prediction(out/'ratio_predictions.npy',ratios),
               'TWD':save_prediction(out/'predictions_twd.npy',predictions)}
    need(sha(data_path)==input_sha and {key:identity(v) for key,v in a.items()}==inputs
         and source_hashes()==sources,'Original source/input arrays changed')
    check_configuration()
    meta.update(status='completed',epochs=epochs,parameter_count=sum(p.numel() for p in model.parameters()),
        fit_seconds=fit_seconds,saved_model_reload_exact=True,saved_ratio_reload_exact=True,reload_rows=len(predictions),
        prediction_artifacts=artifacts,model_sha256=sha(out/'model.pt'),
        prediction_min_twd=float(predictions.min()),prediction_max_twd=float(predictions.max()),
        elapsed_seconds=time.monotonic()-start,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ('status','fit_rows','query_rows','fit_seconds','elapsed_seconds','saved_model_reload_exact')}),flush=True)


def run(data_path,out):
    out=Path(out);existed=out.exists()
    try:return _run_once(data_path,out)
    except Exception as exc:
        if not existed and out.is_dir() and not (out/'failed.json').exists():
            write(out/'failed.json',{'status':'failed_preserved_no_retry','exception_type':type(exc).__name__,
                'automatic_retry_used':False,'automatic_fallback_used':False,'query_labels_used':False,
                'new_CSV':False,'configuration':copy.deepcopy(CONFIG),'source_sha256':source_hashes()})
        raise


def cpu_unit_check():
    """Artificial CPU arrays/loss/model serialization; zero optimizer steps/CUDA."""
    import tempfile
    checks=[]
    def mark(name,condition):need(bool(condition),name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,TypeError,FileExistsError):checks.append(name)
        else:raise AssertionError('Invalid artificial case accepted: '+name)
    before=(copy.deepcopy(prior.CONFIG),copy.deepcopy(baseline.CONFIG),source_hashes())
    check_configuration()
    mark('only_descriptive_provenance_configuration_change',set(CONFIG)==set(baseline.CONFIG)
         and [k for k in CONFIG if CONFIG[k]!=baseline.CONFIG[k]]==['train_price_factor_construction'])
    mark('exact_original_model_and_weighted_MAE_function',EmbeddingMLP is prior.EmbeddingMLP and weighted_loss is baseline.weighted_loss)
    prediction=torch.tensor([-2.,-.5,0.,.5,2.],dtype=torch.float64,requires_grad=True)
    target=torch.zeros_like(prediction);w=torch.tensor([.5,.5,.5,.5,3.],dtype=torch.float64)
    loss=weighted_loss(prediction,target,w);loss.backward()
    mark('weighted_MAE_value_and_gradient',loss.item()==1.5 and torch.allclose(prediction.grad,torch.tensor([-.1,-.1,0.,.1,.6],dtype=torch.float64),rtol=0,atol=1e-15))
    mark('no_batch_weight_renormalization',weighted_loss(prediction.detach(),target,w*2).item()
         !=(w*2*prediction.detach().abs()).sum().item()/(w*2).sum().item())
    chunks=(weighted_loss(prediction.detach()[:2],target[:2],w[:2])*2+weighted_loss(prediction.detach()[2:],target[2:],w[2:])*3)/5
    mark('uneven_batch_row_count_aggregation',torch.allclose(chunks,loss.detach(),rtol=0,atol=1e-15))
    ratio=torch.tensor([.5,2.],dtype=torch.float64,requires_grad=True);factor=torch.tensor([2.,3.],dtype=torch.float64)
    objective=weighted_loss(ratio*factor,torch.tensor([2.,2.],dtype=torch.float64),torch.tensor([1.,4.],dtype=torch.float64));objective.backward()
    mark('original_factor_scaled_ratio_gradient',torch.equal(ratio.grad,torch.tensor([-1.,6.],dtype=torch.float64)))
    y=np.array([1.,2.,3.],np.float32);f=np.array([2.,2.,2.],np.float32)
    median,bias=initial_bias(y,f)
    mark('original_inverseSoftplus_median_formula',median==1. and bias==1.+float(np.log(-np.expm1(-1.))))
    mark('changed_factor_changes_median_bias_by_original_formula',initial_bias(y,f*2.)!=initial_bias(y,f))
    mark('ratio_float64_TWD_original_inverse_no_clip',to_twd(np.array([.25,2.,3.]),f).tobytes()==np.array([.5e6,4e6,6e6],np.float64).tobytes())
    arrays={'train_num':np.zeros((3,144),np.float32),'train_cat':np.zeros((3,23),np.int64),'train_y_million':y,
        'query_num':np.zeros((2,144),np.float32),'query_cat':np.zeros((2,23),np.int64),'cardinalities':np.full(23,5,np.int64),
        'train_price_factor_million':f,'query_price_factor_million':np.ones(2,np.float32),'train_sample_weight':np.ones(3,np.float32)}
    validate_inputs(arrays);mark('original_nine_array_contract_144_23',True)
    class Archive:
        def __init__(self,files):self.files,self.reads=files,[]
        def __getitem__(self,key):self.reads.append(key);return arrays[key]
    for name,files in [('extra_query_y',list(ARRAY_NAMES)+['query_y']),('duplicate_key',list(ARRAY_NAMES)+['train_num']),('missing_key',list(ARRAY_NAMES[:-1]))]:
        archive=Archive(files);reject('archive_'+name,lambda:materialize_archive(archive));mark('archive_'+name+'_before_reads',not archive.reads)
    for name,key,value in [('factor_f64','train_price_factor_million',f.astype(np.float64)),('factor_zero','train_price_factor_million',np.zeros(3,np.float32)),
        ('weight_f64','train_sample_weight',np.ones(3,np.float64)),('weight_nan','train_sample_weight',np.full(3,np.nan,np.float32)),
        ('numeric_width','train_num',np.zeros((3,143),np.float32)),('category_width','query_cat',np.zeros((2,22),np.int64)),
        ('out_of_vocabulary','query_cat',np.full((2,23),5,np.int64)),('query_numeric_nan','query_num',np.full((2,144),np.nan,np.float32)),
        ('target_negative','train_y_million',-y)]:reject(name,lambda key=key,value=value:validate_inputs({**arrays,key:value}))
    torch.manual_seed(42);model=EmbeddingMLP(144,[5]*23)
    torch.manual_seed(42);control=baseline.EmbeddingMLP(144,[5]*23)
    mark('original_initial_tensor_keys_values_RNG_exact',all(torch.equal(v,control.state_dict()[k]) for k,v in model.state_dict().items()))
    mark('original_Softplus_head_and_hidden_architecture',type(model.positive) is torch.nn.Softplus and model.head.in_features==128)
    with torch.no_grad():model.head.bias.fill_(bias)
    mark('actual_bias_is_original_float32_parameter',float(model.head.bias.item())==float(np.float32(bias)))
    numeric=torch.arange(7*144,dtype=torch.float32).reshape(7,144)/1000;categorical=torch.zeros((7,23),dtype=torch.int64)
    values=predict_ratio(model,numeric,categorical)
    mark('positive_finite_dimensionless_ratio_eval',values.dtype==np.float64 and values.shape==(7,) and np.isfinite(values).all() and (values>0).all())
    # Row-wise models can show BLAS batch-size rounding; use equal batch size permutation.
    perm=torch.arange(6,-1,-1);reversed_values=predict_ratio(model,numeric[perm],categorical[perm])
    mark('eval_row_independence_no_cross_row_state',np.allclose(values[::-1],reversed_values,rtol=1e-6,atol=1e-7))
    checkpoint={'numeric_count':144,'cardinalities':[5]*23,'configuration':copy.deepcopy(CONFIG),'actual_model_class':MODEL_CLASS,
        'architecture_source_sha256':ARCHITECTURE_SHA,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    buffer=io.BytesIO();torch.save(checkpoint,buffer);buffer.seek(0)
    restored=validate_checkpoint(torch.load(buffer,map_location='cpu',weights_only=True))
    exact(values,predict_ratio(restored,numeric,categorical),'Synthetic full-query saved ratio reload')
    mark('CPU_checkpoint_full_ratio_reload_exact',True)
    factor7=np.linspace(1.,2.,7,dtype=np.float32)
    exact(to_twd(values,factor7),predict(restored,numeric,categorical,factor7),'Synthetic full-query TWD reload')
    mark('CPU_checkpoint_full_TWD_reload_exact',True)
    for name,field,value in [('wrong_class','actual_model_class','wrong'),('wrong_config','configuration',baseline.CONFIG),
                              ('wrong_width','numeric_count',143),('wrong_architecture','architecture_source_sha256','wrong')]:
        reject('checkpoint_'+name,lambda field=field,value=value:validate_checkpoint({**checkpoint,field:value}))
    key=next(iter(checkpoint['state_dict']))
    for name,value in [('wrong_tensor_dtype',checkpoint['state_dict'][key].double()),('nonfinite_tensor',torch.full_like(checkpoint['state_dict'][key],float('nan')))]:
        reject('checkpoint_'+name,lambda value=value:validate_checkpoint({**checkpoint,'state_dict':{**checkpoint['state_dict'],key:value}}))
    with tempfile.TemporaryDirectory(prefix='r117_cpu_') as directory:
        out=Path(directory);reject('existing_output_refused_before_input_or_setup',lambda:run(Path('not-read.npz'),out))
        mark('existing_output_unchanged',not list(out.iterdir()))
    mark('original_parent_CONFIG_sources_unchanged',before==(prior.CONFIG,baseline.CONFIG,source_hashes()))
    return {'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':source_hashes(),
        'configuration':copy.deepcopy(CONFIG),'actual_model_class':MODEL_CLASS,'real_data_read':False,
        'true_prior_builds':0,'model_fits':0,'optimizer_steps':0,'GPU_calls':0,'query_labels_read':False,'new_CSV':False,
        'author_selfcheck_not_independent_preflight':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--cpu-unit-check','--self-check',dest='cpu_unit_check',action='store_true');args=parser.parse_args()
    if args.cpu_unit_check:print(json.dumps(cpu_unit_check(),ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
    else:
        if args.data is None or args.out is None:parser.error('--data and --out are required')
        run(args.data,args.out)
