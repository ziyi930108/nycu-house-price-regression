"""R106 original R63 ratio objective with independent/shared identity dropout.

Both arms draw the original full categorical mask; T only copies project_name's
mask into the other three named columns. No original module is monkeypatched.
The driver owns encoder/prior/date provenance and canonical row alignment.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import round63_recency_weighted_prior_train as baseline

ROOT=Path(__file__).resolve().parents[1]
PLAN='reports/round106_plan.md'
PLAN_SHA='2748f8f5cd76df02e99717a10a13f88186b1547d1196affb54bd603cb0351f51'
R63='src/round63_recency_weighted_prior_train.py'
R63_SHA='26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
IDENTITY_COLUMNS=('project_name','address_raw','building_and_unit','building_block')
ARRAY_NAMES=('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
             'train_price_factor_million','query_price_factor_million','train_sample_weight')
MASK_MODES={'C':'independent_original_23_columns','T':'four_columns_share_original_project_name_mask'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def identity(a):
    return {'dtype':str(a.dtype),'shape':list(a.shape),'values_sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def source_hashes():
    names=[Path(__file__).resolve().relative_to(ROOT).as_posix(),R63,'src/round52_nn_train.py','src/round47_nn_train.py']
    return {name:sha(ROOT/name) for name in names}


def parameters():
    return {'round':106,'mask_modes':MASK_MODES.copy(),'identity_columns':list(IDENTITY_COLUMNS),
        'input_keys':list(ARRAY_NAMES),'numeric_count':144,'categorical_count':23,'category_dropout':.1,
        'complete_original_mask_sampled':True,'shared_anchor':'project_name','other_categorical_columns_unchanged':19,
        'eval_category_mask_applied':False,'seed':42,'epochs':30,'batch_size':2048,'query_batch_size':4096,
        'loss_equals_R63':True,'sample_weights_renormalized':False,'query_labels_available':False}


def configuration(mask_mode):
    need(mask_mode in MASK_MODES,'Mask mode must be C or T')
    cfg=copy.deepcopy(baseline.CONFIG)
    cfg.update(category_dropout_mode=MASK_MODES[mask_mode],identity_dropout_columns=list(IDENTITY_COLUMNS),
               category_dropout_full_mask_draw=True,eval_category_dropout=False)
    return cfg


def check_configuration():
    need(sha(ROOT/PLAN)==PLAN_SHA and sha(ROOT/R63)==R63_SHA,'Frozen plan/R63 source changed')
    baseline.check_configuration()
    need(sha(Path(baseline.prior.__file__))==baseline.ARCHITECTURE_SHA,'Original architecture source changed')
    for mode in MASK_MODES:
        cfg=configuration(mode)
        need(all(cfg[k]==v for k,v in baseline.CONFIG.items()),'Original training configuration changed')
        need((cfg['seed'],cfg['epochs'],cfg['batch_size'],cfg['category_dropout'])==(42,30,2048,.1),'Fixed setup differs')
    need({k for k in configuration('C') if configuration('C')[k]!=configuration('T')[k]}=={'category_dropout_mode'},'More than mask mode differs')


def validate_indices(indices):
    need(isinstance(indices,(list,tuple)) and len(indices)==4 and len(set(indices))==4
         and all(type(i) is int and 0<=i<23 for i in indices),'Four distinct ordered categorical indices required')
    return tuple(indices)


def check_identity_contract(doc):
    need(set(doc)=={'schema_version','categorical_columns','identity_columns','identity_indices','encoder_source','prepared_source'},'Identity contract keys differ')
    need(doc['schema_version']==1 and doc['identity_columns']==list(IDENTITY_COLUMNS),'Identity names/order differ')
    cols=doc['categorical_columns'];idx=validate_indices(doc['identity_indices'])
    need(isinstance(cols,list) and len(cols)==23 and all(isinstance(v,str) and v for v in cols) and len(set(cols))==23,'Expected 23 unique encoder columns')
    need([cols[i] for i in idx]==list(IDENTITY_COLUMNS),'Indices do not map to frozen encoder names')
    for key in ('encoder_source','prepared_source'):
        r=doc[key];need(isinstance(r,dict) and set(r)=={'path','sha256'} and isinstance(r['path'],str)
                       and len(r['sha256'])==64 and all(c in '0123456789abcdef' for c in r['sha256']),'Invalid declared source record')
    return idx


def transform_mask(full_mask,mask_mode,indices):
    """Pure mask transformation; RNG is consumed only by the preceding full draw."""
    if mask_mode=='C':return full_mask
    mask=full_mask.clone();anchor=full_mask[:,indices[0]]
    for column in indices[1:]:mask[:,column]=anchor
    return mask


class EmbeddingMLP(baseline.EmbeddingMLP):
    def __init__(self,numeric_count,cardinalities,mask_mode,identity_indices):
        need(numeric_count==144 and len(cardinalities)==23,'Fixed 144/23 architecture required')
        need(mask_mode in MASK_MODES,'Unknown mask mode')
        indices=validate_indices(identity_indices)
        super().__init__(numeric_count,cardinalities)
        self.mask_mode=mask_mode;self.identity_indices=indices

    def forward(self,numeric,categorical):
        if self.training:
            full_mask=torch.rand(categorical.shape,device=categorical.device)<baseline.prior.CONFIG['category_dropout']
            categorical=categorical.masked_fill(transform_mask(full_mask,self.mask_mode,self.identity_indices),0)
        embedded=[emb(categorical[:,i]) for i,emb in enumerate(self.embeddings)]
        joined=torch.cat([numeric,*embedded],dim=1)
        return self.positive(self.head(self.body(joined))).squeeze(1)


def materialize_archive(archive):
    need(len(archive.files)==len(ARRAY_NAMES) and set(archive.files)==set(ARRAY_NAMES),'Expected exactly nine original arrays; no query labels')
    return {k:archive[k] for k in ARRAY_NAMES}


def validate_inputs(a):
    need(set(a)==set(ARRAY_NAMES) and all(isinstance(v,np.ndarray) for v in a.values()),'Exactly nine NumPy arrays required')
    xn,xc,y,qn,qc,card=(a[k] for k in ARRAY_NAMES[:6])
    need(xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and len(xn)>0 and len(qn)>0,'Empty/dimension mismatch')
    n,q=len(xn),len(qn)
    need(xn.dtype==qn.dtype==np.float32 and xn.shape==(n,144) and qn.shape==(q,144)
         and np.isfinite(xn).all() and np.isfinite(qn).all(),'Expected 144 finite float32 numeric columns')
    need(card.dtype==np.int64 and card.shape==(23,) and (card>=1).all(),'Expected 23 positive cardinalities')
    need(xc.dtype==qc.dtype==np.int64 and xc.shape==(n,23) and qc.shape==(q,23),'Category dimensions/dtype differ')
    for i,size in enumerate(card):need(((xc[:,i]>=0)&(xc[:,i]<size)).all() and ((qc[:,i]>=0)&(qc[:,i]<size)).all(),'Category code outside cardinality')
    for key,rows in [('train_y_million',n),('train_price_factor_million',n),('query_price_factor_million',q)]:baseline.price_factor_values(a[key],rows)
    baseline.sample_weight_values(a['train_sample_weight'],n)
    return n,q


def initial_bias(y,factor):
    median=float(np.median(y.astype(np.float64)/factor.astype(np.float64)))
    need(np.isfinite(median) and median>0,'Invalid fit ratio median')
    bias=median+float(np.log(-np.expm1(-median)))
    need(np.isfinite(np.float32(bias)),'Invalid initial bias')
    return median,bias


def state_identity(model):
    h=hashlib.sha256()
    for key,v in model.state_dict().items():
        a=v.detach().cpu().numpy();h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def checkpoint(model,card,cfg,indices):
    return {'numeric_count':144,'cardinalities':card,'configuration':cfg,'mask_mode':model.mask_mode,
            'identity_columns':list(IDENTITY_COLUMNS),'identity_indices':list(indices),
            'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}


def restore(payload,card,cfg,mode,indices,device):
    need(payload['numeric_count']==144 and payload['cardinalities']==card and payload['configuration']==cfg
         and payload['mask_mode']==mode and payload['identity_columns']==list(IDENTITY_COLUMNS)
         and payload['identity_indices']==list(indices),'Checkpoint configuration differs')
    model=EmbeddingMLP(144,card,mode,indices).to(device);model.load_state_dict(payload['state_dict']);return model


def run(data_path,out,mask_mode,contract_path):
    tick=time.monotonic();check_configuration();cfg=configuration(mask_mode);sources=source_hashes();input_sha=sha(data_path)
    contract_sha=sha(contract_path);contract=json.loads(contract_path.read_text(encoding='utf8'));indices=check_identity_contract(contract)
    need((ROOT/contract['prepared_source']['path']).resolve()==data_path.resolve() and contract['prepared_source']['sha256']==input_sha,'Contract is for a different prepared file')
    need(not out.exists() or not any(out.iterdir()),'Preserve existing output/failure; no retry');out.mkdir(parents=True,exist_ok=True)
    with np.load(data_path,allow_pickle=False) as archive:arrays=materialize_archive(archive)
    n,q=validate_inputs(arrays);identities={k:identity(v) for k,v in arrays.items()}
    xn,xc,y,qn,qc,card,af,qaf,sw=(arrays[k] for k in ARRAY_NAMES);card=[int(k) for k in card]
    median,bias=initial_bias(y,af);device=baseline.prior.setup()
    model=EmbeddingMLP(144,card,mask_mode,indices).to(device)
    with torch.no_grad():model.head.bias.fill_(bias)
    meta={'round':106,'mask_mode':mask_mode,'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':cfg,
        'parameters':parameters(),'source_sha256':sha(__file__),'source_hashes':sources,'plan_sha256':PLAN_SHA,
        'parent_trainer_source':R63,'parent_trainer_source_sha256':R63_SHA,'architecture_parent_source_sha256':baseline.ARCHITECTURE_SHA,
        'identity_contract':{'path':contract_path.resolve().relative_to(ROOT).as_posix(),'sha256':contract_sha},
        'categorical_columns':contract['categorical_columns'],'identity_columns':list(IDENTITY_COLUMNS),'identity_indices':list(indices),
        'encoder_source':contract['encoder_source'],'encoder_provenance_verified_in_trainer':False,
        'input_sha256':input_sha,'input_array_identities':identities,'fit_rows':n,'query_rows':q,'numeric_count':144,'categorical_count':23,'cardinalities':card,
        'initial_training_median_ratio':median,'initial_head_bias':bias,'initial_state_sha256':state_identity(model),
        'initial_torch_CPU_rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'initial_torch_CUDA_rng_sha256':hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
        'loss_equals_R63':True,'complete_original_category_mask_draw':True,'other_19_masks_unchanged':True,
        'eval_category_mask_applied':False,'query_labels_available':False,'query_labels_used':False,
        'price_factor_provenance_verified_in_trainer':False,'sample_weight_provenance_verified_in_trainer':False,
        'canonical_row_alignment_verified_in_trainer':False,'train_price_factor_construction':baseline.TRAIN_FACTOR_CONSTRUCTION,
        'query_price_factor_construction':baseline.QUERY_FACTOR_CONSTRUCTION,
        'train_price_factor_million_sha256':identities['train_price_factor_million']['values_sha256'],
        'query_price_factor_million_sha256':identities['query_price_factor_million']['values_sha256'],
        'train_sample_weight_sha256':identities['train_sample_weight']['values_sha256'],'train_sample_weight_summary':baseline.sample_weight_summary(sw),
        'sample_weights_recomputed_in_trainer':False,'sample_weights_batch_renormalized':False,'price_or_ratio_clipping':False,
        'prediction_units':'TWD','output_units':baseline.OUTPUT_UNITS,'epoch_selection_uses_training_metrics':False,
        'automatic_retry':False,'bitwise_retraining_determinism_claimed':False,'torch':torch.__version__,'numpy':np.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    baseline.write(out/'run.json',meta)
    tn=torch.as_tensor(xn,device=device);tc=torch.as_tensor(xc,device=device);ty=torch.as_tensor(y,device=device)
    ta=torch.as_tensor(af,device=device);tw=torch.as_tensor(sw,device=device);qtn=torch.as_tensor(qn,device=device);qtc=torch.as_tensor(qc,device=device)
    del xn,xc,y,qn,qc,af,sw,arrays
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['minimum_learning_rate'])
    epochs=[];fit_started=time.monotonic()
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device);total=torch.zeros((),device=device);total_mae=torch.zeros((),device=device)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),cfg['batch_size']):
            index=order[first:first+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            ratio=model(tn[index],tc[index]);prediction=ratio*ta[index];loss=baseline.weighted_loss(prediction,ty[index],tw[index])
            mae=(prediction.detach()-ty[index]).abs().mean()
            loss.backward();optimizer.step();total+=loss.detach()*len(index);total_mae+=mae*len(index)
        value=float((total/len(tn)).item());mae_value=float((total_mae/len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):raise FloatingPointError('Nonfinite objective; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':mae_value,'training_weighted_mae_million_with_dropout':value})
        schedule.step()
        if epoch==1 or epoch%5==0:print(json.dumps({'event':'epoch_complete',**epochs[-1],'elapsed_seconds':time.monotonic()-fit_started}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started;model_path=out/'model.pt'
    torch.save(checkpoint(model,card,cfg,indices),model_path)
    predictions=baseline.predict(model,qtn,qtc,qaf)
    restored=restore(torch.load(model_path,map_location='cpu',weights_only=True),card,cfg,mask_mode,indices,device)
    reloaded=baseline.predict(restored,qtn,qtc,qaf)
    need(predictions.tobytes()==reloaded.tobytes(),'Full-query saved-model TWD reload differs')
    need(sha(data_path)==input_sha and sha(contract_path)==contract_sha and source_hashes()==sources and sha(ROOT/PLAN)==PLAN_SHA,'Input/source/plan changed')
    pred_path=out/'predictions_twd.npy'
    with pred_path.open('xb') as f:np.save(f,predictions,allow_pickle=False)
    need(np.load(pred_path,allow_pickle=False).tobytes()==predictions.tobytes(),'Saved NPY differs')
    meta.update(status='completed',epochs=epochs,fit_seconds=fit_seconds,parameter_count=sum(p.numel() for p in model.parameters()),
        saved_model_reload_exact=True,reload_rows=len(predictions),reload_batch_size=4096,NPY_roundtrip_exact=True,
        prediction_min_twd=float(predictions.min()),prediction_max_twd=float(predictions.max()),model_sha256=sha(model_path),
        predictions_sha256=sha(pred_path),elapsed_seconds=time.monotonic()-tick,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    baseline.write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ('status','mask_mode','fit_rows','query_rows','fit_seconds','saved_model_reload_exact')}),flush=True)


def cpu_self_check():
    tick=time.monotonic();check_configuration();checks=[];original_configs=copy.deepcopy((baseline.CONFIG,baseline.prior.CONFIG))
    def mark(name,ok):need(bool(ok),name);checks.append(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    indices=(9,2,17,5);other=[i for i in range(23) if i not in indices]
    cols=['col_'+str(i) for i in range(23)]
    for i,name in zip(indices,IDENTITY_COLUMNS):cols[i]=name
    doc={'schema_version':1,'categorical_columns':cols,'identity_columns':list(IDENTITY_COLUMNS),'identity_indices':list(indices),
         'encoder_source':{'path':'synthetic_encoder.json','sha256':'a'*64},'prepared_source':{'path':'synthetic.npz','sha256':'b'*64}}
    mark('encoder_order_nonfixed_indices',check_identity_contract(doc)==indices)
    bad=copy.deepcopy(doc);bad['identity_indices'][1]=indices[0];rejects('duplicate_index_rejected',lambda:check_identity_contract(bad))
    bad=copy.deepcopy(doc);bad['identity_indices']=list(reversed(indices));rejects('wrong_name_mapping_rejected',lambda:check_identity_contract(bad))
    raw=torch.zeros((5,23),dtype=torch.bool);raw[0,indices[0]]=True;raw[1,list(indices[1:])]=True;raw[2,:]=True;raw[3,other]=True
    before=raw.clone();joint=transform_mask(raw,'T',indices)
    mark('joint_uses_project_anchor_only',all(torch.equal(joint[:,i],raw[:,indices[0]]) for i in indices))
    mark('other_19_masks_bitwise_preserved',torch.equal(joint[:,other],raw[:,other]))
    mark('C_full_mask_unchanged',torch.equal(transform_mask(raw,'C',indices),raw))
    mark('source_mask_not_mutated',torch.equal(raw,before))
    numeric=torch.linspace(-1,1,7*144).reshape(7,144);categorical=torch.ones((7,23),dtype=torch.long)
    card=[3]*23;models={}
    for mode in ('old','C','T'):
        torch.manual_seed(42)
        models[mode]=baseline.EmbeddingMLP(144,card) if mode=='old' else EmbeddingMLP(144,card,mode,indices)
    mark('same_initial_state_and_parameter_count',len({state_identity(m) for m in models.values()})==1 and len({sum(p.numel() for p in m.parameters()) for m in models.values()})==1)
    outputs={};rng={}
    for mode,model in models.items():
        torch.manual_seed(817);model.train();outputs[mode]=model(numeric,categorical);rng[mode]=torch.get_rng_state()
    mark('C_training_forward_exact_original',torch.equal(outputs['old'],outputs['C']))
    mark('whole_forward_CPU_RNG_consumption_equal',all(torch.equal(rng['old'],v) for v in rng.values()))
    # Known mask semantics already checked above; this draw checks full-mask RNG use.
    draws={};states={}
    for mode in ('C','T'):
        torch.manual_seed(100);full=torch.rand((31,23))<.1;draws[mode]=transform_mask(full,mode,indices);states[mode]=torch.get_rng_state()
    mark('drawn_other_19_exact',torch.equal(draws['C'][:,other],draws['T'][:,other]))
    mark('full_23_draw_same_RNG_state',torch.equal(states['C'],states['T']))
    eval_outputs=[]
    for model in models.values():
        model.eval();before_rng=torch.get_rng_state()
        with torch.inference_mode():eval_outputs.append(model(numeric,categorical))
        mark('eval_no_RNG_'+str(len(eval_outputs)),torch.equal(before_rng,torch.get_rng_state()))
    mark('eval_forward_exact_original_C_T',all(torch.equal(eval_outputs[0],v) for v in eval_outputs))
    p=torch.tensor([1.,4.,8.],requires_grad=True);y=torch.tensor([2.,3.,6.]);w=torch.tensor([.5,2.,1.])
    loss=baseline.weighted_loss(p,y,w);loss.backward()
    mark('original_weighted_L1_value_gradient',torch.equal(loss,torch.tensor(1.5)) and torch.equal(p.grad,torch.tensor([-1/6,2/3,1/3])))
    arrays={'train_num':np.zeros((3,144),np.float32),'train_cat':np.zeros((3,23),np.int64),'train_y_million':np.array([2.,3.,6.],np.float32),
        'query_num':np.zeros((2,144),np.float32),'query_cat':np.zeros((2,23),np.int64),'cardinalities':np.full(23,3,np.int64),
        'train_price_factor_million':np.ones(3,np.float32),'query_price_factor_million':np.ones(2,np.float32),'train_sample_weight':np.ones(3,np.float32)}
    before_arrays={k:identity(v) for k,v in arrays.items()};mark('nine_array_contract',validate_inputs(arrays)==(3,2))
    class Archive:
        files=[*ARRAY_NAMES,'query_y']
        def __getitem__(self,key):raise AssertionError('Forbidden payload accessed')
    rejects('extra_label_rejected_before_payload',lambda:materialize_archive(Archive()))
    rejects('145_numeric_rejected',lambda:validate_inputs({**arrays,'train_num':np.zeros((3,145),np.float32)}))
    median,bias=initial_bias(arrays['train_y_million'],arrays['train_price_factor_million'])
    mark('original_median_inverseSoftplus_bias',median==3. and np.isclose(np.logaddexp(0.,bias),3.,rtol=0,atol=1e-15))
    for mode in ('C','T'):
        model=models[mode]
        with torch.no_grad():model.head.bias.fill_(bias)
        buffer=io.BytesIO();torch.save(checkpoint(model,card,configuration(mode),indices),buffer);buffer.seek(0)
        loaded=restore(torch.load(buffer,map_location='cpu',weights_only=True),card,configuration(mode),mode,indices,torch.device('cpu'))
        factor=np.linspace(1,2,7,dtype=np.float32)
        pred=baseline.predict(model,numeric,categorical,factor);again=baseline.predict(loaded,numeric,categorical,factor)
        mark(mode+'_CPU_save_reload_TWD_exact',pred.dtype==np.float64 and pred.tobytes()==again.tobytes())
    mark('inputs_unchanged',before_arrays=={k:identity(v) for k,v in arrays.items()})
    mark('original_configs_unchanged',(baseline.CONFIG,baseline.prior.CONFIG)==original_configs)
    mark('only_mask_mode_configuration_differs',{k for k in configuration('C') if configuration('C')[k]!=configuration('T')[k]}=={'category_dropout_mode'})
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,'source_sha256':sha(__file__),
        'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,'parameters':parameters(),'configurations':{m:configuration(m) for m in MASK_MODES},
        'synthetic_only':True,'real_data_read':False,'true_fit_calls':0,'optimizer_steps':0,'GPU_calls':0,
        'CPU_RNG_equivalence_checked':True,'CUDA_cross_run_determinism_claimed':False,
        'synthetic_CPU_model_forward_and_save_reload':True,'elapsed_seconds':time.monotonic()-tick}


def write_receipt(path,value):
    side=path.with_suffix('.sha256');need(not path.exists() and not side.exists(),'Preserve existing self-check receipt')
    baseline.write(path,value)
    with side.open('x',encoding='ascii') as f:f.write(sha(path)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--mask-mode',choices=['C','T']);parser.add_argument('--identity-contract',type=Path)
    parser.add_argument('--cpu-self-check',action='store_true');parser.add_argument('--receipt',type=Path,default=ROOT/'reports/round106_trainer_selfcheck.json')
    args=parser.parse_args()
    if args.cpu_self_check:
        try:r=cpu_self_check()
        except Exception:
            write_receipt(args.receipt,{'status':'failed','exit_code':1,'failure':traceback.format_exc(),'source_sha256':sha(__file__),
                'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,'synthetic_only':True,'real_data_read':False});raise
        write_receipt(args.receipt,r);print(json.dumps({'status':r['status'],'checks':r['check_count'],'elapsed_seconds':r['elapsed_seconds']}))
    else:
        if args.data is None or args.out is None or args.mask_mode is None or args.identity_contract is None:parser.error('--data/--out/--mask-mode/--identity-contract required')
        run(args.data,args.out,args.mask_mode,args.identity_contract)
