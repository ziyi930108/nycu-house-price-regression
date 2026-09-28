"""R94 fixed FT-style prior-ratio regression; no query-label trainer interface.

Structure adapted from Yandex rtdl-revisiting-models, commit
 e3ed46cac38568785289d8fa16b8cfa585bde27e (Apache-2.0, Copyright 2023 Yandex LLC).
Preserved source/license: outputs/round94_upstream/upstream/package/.
Modified for this task: small fixed model, CLS-only final residual/FFN, category
 dropout, positive prior-ratio head, weighted TWD L1, accumulated batches,
 fixed last epoch and all-parameter AdamW. Not an official full reproduction.
V2 engineering change: verify NPZ member names before any member read.
No upstream module/package is imported; no real data or CUDA on import.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import io
import json
import math
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = 'e3ed46cac38568785289d8fa16b8cfa585bde27e'
PLAN_SHA = 'c44e8d40e8eabe5c00f4e69905aa6987e9b68adf5bdccd920f84deb5cbf38c0a'
SOURCE_DEPENDENCIES = {
    'src/round63_recency_weighted_prior_train.py':'26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a',
    'src/round47_nn_train.py':'68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7',
    'outputs/round94_upstream/upstream/package/rtdl_revisiting_models.py':'ca7cef3b44cd54b1aeb51f2cc733bef01fbfe3eff00b5c2bd28ca4b4db6b01b2',
    'outputs/round94_upstream/upstream/package/LICENSE':'35381b68041ebc3408e1b76ab24ce5eed1f7ea93afd62d109ffdabf1b0daa324',
    'outputs/round94_upstream/manifest.json':'699800b3ceb2c12c6dccee1931cd27a1c0eb53ba94e7977f5cdfa019963cbc8f',
    'reports/round94_plan.md':PLAN_SHA,
}
CONFIG = {
    'model_class':'AttentionPriorRegressor','architecture':'local small FT-style prior-ratio',
    'numeric_count':144,'categorical_count':23,'token_count':168,
    'd_token':64,'n_heads':4,'n_blocks':2,'ffn_hidden':128,'ffn_activation':'ReGLU',
    'attention_dropout':0.1,'ffn_dropout':0.1,'residual_dropout':0.0,'category_dropout':0.1,
    'first_attention_layer_norm':False,'later_attention_layer_norm':True,
    'last_query_tokens':'CLS only','last_residual_and_FFN_tokens':'CLS only',
    'attention_implementation':'explicit QK softmax probability-dropout AV; no compression/SDPA',
    'tokenizer_and_CLS_initialization':'uniform +/- 1/sqrt(d_token)',
    'attention_projection_initialization':'torch Linear defaults; Q/K/V/Out bias zero',
    'FFN_and_head_initialization':'torch Linear defaults; override head bias from fit ratio median',
    'layer_norm_eps':0.00001,'output_activation':'Softplus',
    'seed':42,'epochs':30,'batch_size':2048,'microbatch_size':256,'predict_batch_size':256,
    'gradient_accumulation':'microbatch weighted L1 sum / actual optimizer-group row count',
    'optimizer':'AdamW','learning_rate':0.001,'weight_decay':0.0001,
    'optimizer_parameter_groups':'all parameters, one group, no decay exclusions',
    'scheduler':'CosineAnnealingLR','minimum_learning_rate':0.00001,
    'float_precision':'float32; matmul precision high','thread_count':4,
    'loss':'fit-recency weighted MAE in million TWD',
    'loss_formula':'mean(train_sample_weight * abs(ratio * train_price_factor_million - train_y_million))',
    'price_factor_formula':'effective_area_A * group_prior_TWD_per_m2 / 1e6',
    'output_units':'dimensionless positive price ratio','prediction_units':'TWD',
    'initial_bias':'inverseSoftplus(median_fit(train_y_million / train_price_factor_million))',
    'epoch_selection':'fixed last epoch; no query labels or early stopping',
    'extra_sample_weights':True,'sample_weight_normalization':'saved fit-global normalization; no batch renormalization',
    'price_or_ratio_clipping':False,'smoke_warmup_groups':2,'smoke_timed_groups':3,
    'maximum_estimated_three_stage_seconds':5400,
}
FROZEN_CONFIG_JSON = json.dumps(CONFIG, sort_keys=True, separators=(',', ':'), allow_nan=False)
ARRAY_NAMES = ('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities',
               'train_price_factor_million','query_price_factor_million','train_sample_weight')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {**SOURCE_DEPENDENCIES,Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__)}


def check_configuration():
    need(json.dumps(CONFIG,sort_keys=True,separators=(',',':'),allow_nan=False)==FROZEN_CONFIG_JSON,'R94 configuration mutated')
    for name,digest in SOURCE_DEPENDENCIES.items():
        need(sha(ROOT/name)==digest,'Frozen dependency changed: '+name)
    return True


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def value_sha(values):
    a=np.asarray(values)
    need(a.dtype in (np.dtype('float32'),np.dtype('int64')),'Expected original float32 or int64 array')
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<f4' if a.dtype==np.float32 else '<i8').tobytes()).hexdigest()


def array_record(a):
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':value_sha(a)}


def positive32(values,rows,name):
    a=np.asarray(values)
    need(a.dtype==np.float32 and a.shape==(rows,) and np.isfinite(a).all() and (a>0).all(),name+' must be positive finite row-aligned float32')
    return a


def validate_arrays(arrays):
    need(set(arrays)==set(ARRAY_NAMES),'Expected exactly nine arrays, no query labels')
    a={k:np.asarray(arrays[k]) for k in ARRAY_NAMES}
    xn,xc,qn,qc=a['train_num'],a['train_cat'],a['query_num'],a['query_cat']
    card=a['cardinalities']
    need(card.dtype==np.int64 and card.shape==(23,) and (card>0).all(),'Expected 23 positive int64 cardinalities')
    for n,c,label in [(xn,xc,'fit'),(qn,qc,'query')]:
        need(n.ndim==c.ndim==2 and len(n)>0 and n.shape==(len(n),144)
             and c.shape==(len(n),23) and n.dtype==np.float32 and c.dtype==np.int64,
             label+' numeric/category shape or dtype differs')
        need(np.isfinite(n).all(),label+' numeric values nonfinite')
        need((c>=0).all() and (c<card[None,:]).all(),label+' category out of range')
    positive32(a['train_y_million'],len(xn),'Fit prices')
    positive32(a['train_price_factor_million'],len(xn),'Fit factors')
    positive32(a['query_price_factor_million'],len(qn),'Query factors')
    positive32(a['train_sample_weight'],len(xn),'Fit dates')
    return a


def load_prepared(path):
    """Check NPZ member names before deserializing any arrays, including extras."""
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(ARRAY_NAMES),'Expected exactly nine NPZ members before any array read')
        return validate_arrays({k:z[k] for k in ARRAY_NAMES})


def fit_head_bias(y,factor):
    positive32(y,len(y),'Fit prices');positive32(factor,len(y),'Fit factors')
    median=float(np.median(y.astype(np.float64)/factor.astype(np.float64)))
    need(np.isfinite(median) and median>0,'Fit median ratio invalid')
    bias=median+float(np.log(-np.expm1(-median)))
    need(np.isfinite(np.float32(bias)),'Initial head bias invalid')
    return median,bias


def to_twd(ratio,price_factor_million):
    r=np.asarray(ratio,dtype=np.float64)
    need(r.ndim==1 and len(r)>0 and np.isfinite(r).all() and (r>0).all(),'Ratio outputs invalid')
    factor=positive32(price_factor_million,len(r),'Query factors')
    with np.errstate(over='raise',invalid='raise'):
        result=r*factor.astype(np.float64)*1e6
    need(np.isfinite(result).all() and (result>0).all(),'TWD outputs invalid; no clipping')
    return result


class FeatureTokenizer(nn.Module):
    def __init__(self,cardinalities):
        super().__init__();d=CONFIG['d_token']
        self.cls=nn.Parameter(torch.empty(d))
        self.numeric_weight=nn.Parameter(torch.empty(144,d));self.numeric_bias=nn.Parameter(torch.empty(144,d))
        self.embeddings=nn.ModuleList([nn.Embedding(k,d) for k in cardinalities])
        self.categorical_bias=nn.Parameter(torch.empty(23,d))
        for v in [self.cls,self.numeric_weight,self.numeric_bias,*[e.weight for e in self.embeddings],self.categorical_bias]:
            nn.init.uniform_(v,-d**-.5,d**-.5)

    def forward(self,numeric,categorical):
        if self.training:
            drop=torch.rand(categorical.shape,device=categorical.device)<CONFIG['category_dropout']
            categorical=categorical.masked_fill(drop,0)
        values=numeric[:,:,None]*self.numeric_weight+self.numeric_bias[None,:,:]
        cats=torch.stack([e(categorical[:,i]) for i,e in enumerate(self.embeddings)],dim=1)+self.categorical_bias
        return torch.cat([self.cls.expand(len(numeric),1,-1),values,cats],dim=1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__();d=CONFIG['d_token']
        self.W_q=nn.Linear(d,d);self.W_k=nn.Linear(d,d);self.W_v=nn.Linear(d,d);self.W_out=nn.Linear(d,d)
        self.dropout=nn.Dropout(CONFIG['attention_dropout'])
        for layer in [self.W_q,self.W_k,self.W_v,self.W_out]: nn.init.zeros_(layer.bias)

    def forward(self,xq,xkv):
        b=len(xq);h=CONFIG['n_heads'];d=CONFIG['d_token']//h
        def split(x): return x.reshape(b,-1,h,d).transpose(1,2)
        q,k,v=split(self.W_q(xq)),split(self.W_k(xkv)),split(self.W_v(xkv))
        probs=self.dropout(torch.softmax(q@k.transpose(-1,-2)/math.sqrt(d),dim=-1))
        joined=(probs@v).transpose(1,2).reshape(b,xq.shape[1],CONFIG['d_token'])
        return self.W_out(joined)


class ReGLU(nn.Module):
    def forward(self,x):
        a,b=x.chunk(2,dim=-1)
        return a*F.relu(b)


class TransformerBlock(nn.Module):
    def __init__(self,index):
        super().__init__();d=CONFIG['d_token'];f=CONFIG['ffn_hidden']
        self.attention=Attention()
        self.attention_norm=nn.LayerNorm(d,eps=CONFIG['layer_norm_eps']) if index else None
        self.attention_residual_dropout=nn.Dropout(CONFIG['residual_dropout'])
        self.ffn_norm=nn.LayerNorm(d,eps=CONFIG['layer_norm_eps'])
        self.ffn=nn.Sequential(nn.Linear(d,2*f),ReGLU(),nn.Dropout(CONFIG['ffn_dropout']),nn.Linear(f,d))
        self.ffn_residual_dropout=nn.Dropout(CONFIG['residual_dropout'])

    def forward(self,x,last=False):
        identity=x[:,:1] if last else x
        normalized=self.attention_norm(x) if self.attention_norm is not None else x
        x=identity+self.attention_residual_dropout(self.attention(normalized[:,:1] if last else normalized,normalized))
        return x+self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))


class AttentionPriorRegressor(nn.Module):
    def __init__(self,cardinalities):
        super().__init__()
        need(len(cardinalities)==23 and all(type(k) is int and k>0 for k in cardinalities),'Invalid fit cardinalities')
        self.cardinalities=list(cardinalities);self.tokenizer=FeatureTokenizer(cardinalities)
        self.blocks=nn.ModuleList([TransformerBlock(i) for i in range(CONFIG['n_blocks'])])
        self.head_norm=nn.LayerNorm(CONFIG['d_token'],eps=CONFIG['layer_norm_eps'])
        self.head=nn.Linear(CONFIG['d_token'],1)

    def head_output(self,x):
        return F.softplus(self.head(F.relu(self.head_norm(x)))).squeeze(-1)

    def forward(self,numeric,categorical):
        x=self.tokenizer(numeric,categorical)
        for i,block in enumerate(self.blocks): x=block(x,last=i==len(self.blocks)-1)
        return self.head_output(x[:,0])


def upstream_broadcast_reference(model,numeric,categorical):
    """CPU/eval reference of pinned upstream L653-668, sharing supplied weights.

    Last attention is CLS-query, but residual broadcasts to all tokens and
    the final FFN keeps all tokens. Never called by training or inference.
    """
    need(not model.training,'Reference equivalence requires eval/dropout disabled')
    x=model.tokenizer(numeric,categorical)
    for i,block in enumerate(model.blocks):
        identity=x
        normalized=block.attention_norm(x) if block.attention_norm is not None else x
        x=identity+block.attention_residual_dropout(block.attention(normalized[:,:1] if i==len(model.blocks)-1 else normalized,normalized))
        identity=x
        x=identity+block.ffn_residual_dropout(block.ffn(block.ffn_norm(x)))
    return model.head_output(x[:,0])


def weighted_loss(total,target,weights):
    need(total.ndim==1 and target.shape==total.shape==weights.shape,'Loss vectors not aligned')
    return (weights*(total-target).abs()).mean()


def optimizer_for(model):
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    expected=[id(p) for p in model.parameters()]
    actual=[id(p) for g in optimizer.param_groups for p in g['params']]
    need(len(optimizer.param_groups)==1 and len(set(actual))==len(actual) and set(actual)==set(expected),'Optimizer parameters incomplete/duplicated')
    need(all(g['weight_decay']==CONFIG['weight_decay'] for g in optimizer.param_groups),'All-parameter decay differs')
    return optimizer


def setup(device_name):
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['thread_count']);torch.set_float32_matmul_precision('high')
    if device_name=='cuda':
        need(torch.cuda.is_available(),'R94 requires scheduled CUDA; no automatic CPU fallback')
        torch.cuda.manual_seed_all(CONFIG['seed'])
    return torch.device('cuda:0' if device_name=='cuda' else 'cpu')


def train_group(model,optimizer,numeric,categorical,target,factor,weights,indices):
    """One optimizer update; sum each micro-loss over actual whole-group rows."""
    count=len(indices);need(0<count<=CONFIG['batch_size'],'Invalid optimizer group size')
    optimizer.zero_grad(set_to_none=True)
    loss_sum=torch.zeros((),device=numeric.device,dtype=numeric.dtype);mae_sum=torch.zeros_like(loss_sum)
    for start in range(0,count,CONFIG['microbatch_size']):
        ix=indices[start:start+CONFIG['microbatch_size']]
        ratio=model(numeric[ix],categorical[ix]);pred=ratio*factor[ix]
        errors=(pred-target[ix]).abs();numerator=(weights[ix]*errors).sum()
        need(bool(torch.isfinite(numerator)) and bool(torch.isfinite(ratio).all()) and bool((ratio>0).all()),'Nonfinite/nonpositive training output')
        (numerator/count).backward()
        loss_sum+=numerator.detach();mae_sum+=errors.detach().sum()
    norms=torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None])
    need(bool(torch.isfinite(norms).all()),'Nonfinite training gradient; no clipping/retry')
    optimizer.step()
    return loss_sum,mae_sum


@torch.inference_mode()
def predict_ratio(model,numeric,categorical,batch_size=None):
    model.eval();size=CONFIG['predict_batch_size'] if batch_size is None else batch_size
    need(size>0 and len(numeric)>0 and len(numeric)==len(categorical),'Invalid query shape/batch')
    parts=[model(numeric[i:i+size],categorical[i:i+size]).cpu().numpy() for i in range(0,len(numeric),size)]
    values=np.concatenate(parts).astype(np.float64)
    need(values.shape==(len(numeric),) and np.isfinite(values).all() and (values>0).all(),'Invalid ratio prediction')
    return values


def weight_summary(values):
    v=np.asarray(values,dtype=np.float64);total=float(v.sum());squares=float(np.dot(v,v))
    return {'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
            'sum':total,'sum_squares':squares,'effective_sample_size':total*total/squares}


def runtime():
    return {'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'device_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory}


def metadata_for(a,data_path):
    median,bias=fit_head_bias(a['train_y_million'],a['train_price_factor_million'])
    return {
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':copy.deepcopy(CONFIG),
        'configuration_sha256':content_sha(CONFIG),'source_sha256':sha(__file__),'architecture_source_sha256':sha(__file__),
        'source_dependencies':dict(SOURCE_DEPENDENCIES),'source_hashes':source_hashes(),
        'plan_sha256':PLAN_SHA,'upstream_commit':UPSTREAM_COMMIT,
        'upstream_source_sha256':SOURCE_DEPENDENCIES['outputs/round94_upstream/upstream/package/rtdl_revisiting_models.py'],
        'changed_from_R63':['architecture/token embeddings/dropout structure','microbatch accumulation','predict batch 256','final CLS computation pruning'],
        'input_sha256':sha(data_path),'all_input_arrays':{k:array_record(v) for k,v in a.items()},
        'fit_rows':len(a['train_num']),'query_rows':len(a['query_num']),'numeric_count':144,'categorical_count':23,
        'cardinalities':a['cardinalities'].tolist(),'train_y_million_sha256':value_sha(a['train_y_million']),
        'train_sample_weight_sha256':value_sha(a['train_sample_weight']),
        'train_sample_weight_shape':list(a['train_sample_weight'].shape),'train_sample_weight_dtype':'float32',
        'train_sample_weight_summary':weight_summary(a['train_sample_weight']),
        'train_price_factor_million_sha256':value_sha(a['train_price_factor_million']),
        'query_price_factor_million_sha256':value_sha(a['query_price_factor_million']),
        'price_factor_million_summary':{k:weight_summary(a[k]) for k in ('train_price_factor_million','query_price_factor_million')},
        'initial_training_median_ratio':median,'initial_head_bias':bias,
        'query_labels_available':False,'query_labels_used':False,'prediction_units':'TWD','output_units':CONFIG['output_units'],
        'price_factor_formula':CONFIG['price_factor_formula'],'price_factor_uses_fit_labels':True,
        'train_price_factor_construction':'same-window group-fold OOF fit-label prior supplied by driver',
        'query_price_factor_construction':'same-window full-fit label prior supplied by driver; no query labels',
        'price_factor_provenance_verified_in_trainer':False,'sample_weight_provenance_verified_in_trainer':False,
        'sample_weights_recomputed_in_trainer':False,'sample_weights_batch_renormalized':False,
        'training_objective_name':'weighted MAE','epoch_selection_uses_training_metrics':False,
        'extra_sample_weights':True,'price_or_ratio_clipping':False,'bitwise_retraining_determinism_claimed':False,
        'gradient_clipping':False,'optimizer_all_parameters_same_decay':True,'fit_uses_eval_set':False,'early_stopping_used':False,
    }


def save_array(path,a):
    with Path(path).open('xb') as stream: np.save(stream,a,allow_pickle=False)


def run(data_path,out):
    started=time.monotonic();check_configuration()
    need(not out.exists(),'Output must be a new directory; preserve prior evidence')
    out.mkdir(parents=True)
    a=load_prepared(data_path)
    meta=metadata_for(a,data_path);device=setup('cuda');meta.update(runtime())
    model=AttentionPriorRegressor(meta['cardinalities']).to(device)
    with torch.no_grad(): model.head.bias.fill_(meta['initial_head_bias'])
    qfactor=a['query_price_factor_million'].copy()
    tensors={k:torch.as_tensor(v,device=device) for k,v in a.items() if k!='cardinalities'}
    del a
    optimizer=optimizer_for(model)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    write(out/'run.json',meta);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    epochs=[];updates=0;n=meta['fit_rows']
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(n,device=device);total=torch.zeros((),device=device);mae=torch.zeros_like(total)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,n,CONFIG['batch_size']):
            indices=order[first:first+CONFIG['batch_size']]
            s,e=train_group(model,optimizer,tensors['train_num'],tensors['train_cat'],tensors['train_y_million'],
                            tensors['train_price_factor_million'],tensors['train_sample_weight'],indices)
            total+=s;mae+=e;updates+=1
        value=float((total/n).item());mae_value=float((mae/n).item())
        need(np.isfinite(value) and np.isfinite(mae_value),'Nonfinite epoch metrics; no adaptive retry')
        entry={'epoch':epoch,'learning_rate':lr,'training_weighted_mae_million_with_dropout':value,
               'training_mae_million_with_dropout':mae_value}
        epochs.append(entry);schedule.step()
        if epoch==1 or epoch%5==0: print(json.dumps({'event':'epoch_complete',**entry,'elapsed_seconds':time.monotonic()-began}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-began
    need(updates==CONFIG['epochs']*math.ceil(n/CONFIG['batch_size']),'Optimizer update count differs')
    torch.save({'numeric_count':144,'cardinalities':meta['cardinalities'],'configuration':copy.deepcopy(CONFIG),
                'configuration_sha256':content_sha(CONFIG),'source_sha256':sha(__file__),'source_dependencies':dict(SOURCE_DEPENDENCIES),
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},out/'model.pt')
    ratios=predict_ratio(model,tensors['query_num'],tensors['query_cat']);prices=to_twd(ratios,qfactor)
    loaded=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    need(loaded['configuration']==CONFIG and loaded['configuration_sha256']==content_sha(CONFIG)
         and loaded['source_sha256']==sha(__file__) and loaded['source_dependencies']==SOURCE_DEPENDENCIES
         and loaded['numeric_count']==144 and loaded['cardinalities']==meta['cardinalities'],'Saved model identity changed')
    restored=AttentionPriorRegressor(meta['cardinalities']).to(device);restored.load_state_dict(loaded['state_dict'],strict=True)
    ratios2=predict_ratio(restored,tensors['query_num'],tensors['query_cat']);prices2=to_twd(ratios2,qfactor)
    need(ratios.tobytes()==ratios2.tobytes() and prices.tobytes()==prices2.tobytes(),'Full query ratio/TWD reload differs')
    save_array(out/'predictions_ratio.npy',ratios);save_array(out/'predictions_twd.npy',prices)
    check_configuration();need(sha(data_path)==meta['input_sha256'] and sha(__file__)==meta['source_sha256'],'Fit source or input changed')
    meta.update(status='completed',epochs=epochs,fit_seconds=fit_seconds,optimizer_steps=updates,
                parameter_count=sum(p.numel() for p in model.parameters()),reload_ratio_exact=True,reload_TWD_exact=True,
                saved_model_reload_exact=True,reload_rows=len(ratios),predict_batch_size=256,
                prediction_min_twd=float(prices.min()),prediction_max_twd=float(prices.max()),
                ratio_min=float(ratios.min()),ratio_max=float(ratios.max()),
                elapsed_seconds=time.monotonic()-started,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                maximum_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ['status','fit_rows','query_rows','fit_seconds','saved_model_reload_exact']}),flush=True)
    return meta


def receipt_identity():
    return {'plan_sha256':PLAN_SHA,'configuration':copy.deepcopy(CONFIG),'configuration_sha256':content_sha(CONFIG),
            'trainer_source_sha256':sha(__file__),'architecture_source_sha256':sha(__file__),
            'source_sha256':source_hashes(),'source_dependencies':dict(SOURCE_DEPENDENCIES),
            'upstream_commit':UPSTREAM_COMMIT,'synthetic_only':True,'real_data_read':False,'new_real_native_fits':0}


def synthetic_tensors(rows,cardinalities,device,dtype=torch.float32):
    generator=torch.Generator(device='cpu').manual_seed(CONFIG['seed'])
    numeric=torch.randn(rows,144,generator=generator,dtype=dtype).to(device)
    categorical=torch.stack([torch.randint(k,(rows,),generator=generator) for k in cardinalities],dim=1).to(device)
    target=(1+torch.rand(rows,generator=generator,dtype=dtype)*30).to(device)
    factor=(5+torch.rand(rows,generator=generator,dtype=dtype)*20).to(device)
    weights=(.2+torch.rand(rows,generator=generator,dtype=dtype)*2).to(device)
    return numeric,categorical,target,factor,weights


def smoke(cardinalities_path,out):
    started=time.monotonic();check_configuration()
    need(not out.exists(),'Smoke output must be a new directory')
    out.mkdir(parents=True)
    description=json.loads(cardinalities_path.read_text(encoding='utf-8-sig'))
    cards=description['cardinalities'];counts=description['fit_rows_by_stage'];steps=description['optimizer_steps_per_stage']
    need(len(cards)==23 and all(type(x) is int and x>0 for x in cards),'Smoke cards invalid')
    need(description['synthetic_only'] is True and description['prices_read'] is False,'Smoke input provenance not synthetic-only')
    need(set(counts)==set(steps)=={'development','confirmation','auxiliary'},'Smoke stages differ')
    need(all(steps[k]==CONFIG['epochs']*math.ceil(counts[k]/CONFIG['batch_size']) for k in counts),'Smoke optimizer step estimate differs')
    for name,digest in description['source_sha256'].items(): need(sha(ROOT/name)==digest,'Smoke cardinality metadata changed')
    path_sha=sha(cardinalities_path);device=setup('cuda')
    model=AttentionPriorRegressor(cards).to(device);model.train()
    optimizer=optimizer_for(model)
    n,c,y,f,w=synthetic_tensors(CONFIG['batch_size'],cards,device)
    indices=torch.arange(CONFIG['batch_size'],device=device)
    meta={**receipt_identity(),**runtime(),'status':'running','cardinalities':cards,
          'cardinalities_json_sha256':path_sha,'cardinalities_metadata_sources':description['source_sha256'],
          'optimizer_steps_per_stage':steps,'total_optimizer_steps':sum(steps.values()),
          'warmup_groups':CONFIG['smoke_warmup_groups'],'timed_groups':CONFIG['smoke_timed_groups'],
          'optimizer_batch_rows':CONFIG['batch_size'],'microbatch_rows':CONFIG['microbatch_size'],
          'synthetic_data_shape':[CONFIG['batch_size'],144,23],'parameter_count':sum(p.numel() for p in model.parameters()),
          'cost_gate_seconds':CONFIG['maximum_estimated_three_stage_seconds'],
          'same_train_group_as_true_fit':True,'includes_finite_and_gradient_checks':True,
          'full_real_prepared_arrays_resident':False,'estimation_excludes_prepare_reload_and_diagnostics':True}
    write(out/'run.json',meta)
    for _ in range(CONFIG['smoke_warmup_groups']): train_group(model,optimizer,n,c,y,f,w,indices)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(CONFIG['smoke_timed_groups']):
        torch.cuda.synchronize();tick=time.monotonic()
        train_group(model,optimizer,n,c,y,f,w,indices)
        torch.cuda.synchronize();times.append(time.monotonic()-tick)
    avg=float(np.mean(times));estimate=avg*sum(steps.values())
    need(np.isfinite(avg) and avg>0 and np.isfinite(estimate),'Invalid smoke timing')
    check_configuration();need(sha(cardinalities_path)==path_sha,'Smoke input changed')
    passed=estimate<=CONFIG['maximum_estimated_three_stage_seconds']
    meta.update(status='passed' if passed else 'cost_gate_failed',exit_code=0 if passed else 1,
                optimizer_group_seconds=times,mean_optimizer_group_seconds=avg,
                estimated_three_stage_fit_seconds=estimate,cost_gate_passed=passed,
                synthetic_optimizer_steps=CONFIG['smoke_warmup_groups']+CONFIG['smoke_timed_groups'],
                maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.monotonic()-started)
    write(out/'smoke.json',meta)
    print(json.dumps({k:meta[k] for k in ['status','mean_optimizer_group_seconds','estimated_three_stage_fit_seconds','cost_gate_passed','maximum_gpu_allocated_bytes','maximum_gpu_reserved_bytes']}),flush=True)
    need(passed,'Estimated three-stage fit exceeds 90 minutes; stop without altering candidate')
    return meta


def cpu_self_check():
    """Small synthetic CPU tests; no real arrays or GPU setup/access."""
    check_configuration();torch.set_num_threads(4);torch.manual_seed(42)
    checks=[];tolerances={}
    def ok(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def reject(name,operation):
        try:operation()
        except (ValueError,FloatingPointError,RuntimeError):checks.append(name)
        else:raise AssertionError(name+' did not reject')
    cards=[3]*23
    n,c,y,f,w=synthetic_tensors(9,cards,torch.device('cpu'))
    a={'train_num':n.numpy().copy(),'train_cat':c.numpy().copy(),'train_y_million':y.numpy().copy(),
       'query_num':n.numpy().copy(),'query_cat':c.numpy().copy(),'cardinalities':np.array(cards,dtype=np.int64),
       'train_price_factor_million':f.numpy().copy(),'query_price_factor_million':f.numpy().copy(),
       'train_sample_weight':w.numpy().copy()}
    validated=validate_arrays(a)
    ok('nine_arrays_exact_shapes_dtypes_positive_finite',set(validated)==set(ARRAY_NAMES))
    buffer=io.BytesIO()
    np.savez(buffer,**a,query_y_million=np.array([None],dtype=object))
    buffer.seek(0)
    try:
        load_prepared(buffer)
    except ValueError as exc:
        ok('extra_NPZ_member_rejected_before_deserialization',
           str(exc)=='Expected exactly nine NPZ members before any array read')
    else:
        raise AssertionError('extra member accepted')
    buffer=io.BytesIO();np.savez(buffer,**a);buffer.seek(0)
    roundtrip=load_prepared(buffer)
    ok('valid_nine_member_NPZ_exact_roundtrip',
       all(v.dtype==roundtrip[k].dtype and v.shape==roundtrip[k].shape and v.tobytes()==roundtrip[k].tobytes() for k,v in a.items()))

    reject('query_labels_member_rejected',lambda:validate_arrays({**a,'query_y_million':np.ones(9,dtype=np.float32)}))
    reject('float64_numeric_rejected',lambda:validate_arrays({**a,'train_num':a['train_num'].astype(np.float64)}))
    reject('wrong_numeric_width_rejected',lambda:validate_arrays({**a,'query_num':a['query_num'][:,:143]}))
    reject('wrong_cardinalities_count_rejected',lambda:validate_arrays({**a,'cardinalities':np.array([3]*22,dtype=np.int64)}))
    for name in ['train_y_million','train_price_factor_million','query_price_factor_million','train_sample_weight']:
        for bad in [0.,np.nan,np.inf]:
            v=a[name].copy();v[0]=bad
            reject(name+'_invalid_'+str(bad)+'_rejected',lambda name=name,v=v:validate_arrays({**a,name:v}))
    bad=a['query_cat'].copy();bad[0,0]=3
    reject('unknown_out_of_cardinality_rejected',lambda:validate_arrays({**a,'query_cat':bad}))
    bad=a['train_cat'].copy();bad[0,0]=-1
    reject('negative_category_rejected',lambda:validate_arrays({**a,'train_cat':bad}))
    initial=fit_head_bias(a['train_y_million'],a['train_price_factor_million'])
    changed=copy.deepcopy(a);changed['query_num']*=100;changed['query_price_factor_million']*=200
    ok('query_values_do_not_change_fit_head_initialization',initial==fit_head_bias(changed['train_y_million'],changed['train_price_factor_million']))
    ok('inverseSoftplus_fit_median_recovers_ratio',np.isclose(np.logaddexp(0,initial[1]),initial[0],rtol=2e-15))
    model=AttentionPriorRegressor(cards).eval()
    tok=model.tokenizer(n,c)
    ok('tokenizer_shape168_by64',tok.shape==(9,168,64))
    ok('numeric_token_formula_and_order',torch.equal(tok[:,1:145],n[:,:,None]*model.tokenizer.numeric_weight+model.tokenizer.numeric_bias))
    ok('CLS_is_first_token',torch.equal(tok[:,0],model.tokenizer.cls.expand(9,-1)))
    ok('categorical_token_formula_and_order',torch.equal(tok[:,145:],torch.stack([e(c[:,i]) for i,e in enumerate(model.tokenizer.embeddings)],dim=1)+model.tokenizer.categorical_bias))
    ok('tokenizer_and_CLS_init_bound_0125',all(bool((v.detach().abs()<=.125).all()) for v in [model.tokenizer.cls,model.tokenizer.numeric_weight,model.tokenizer.numeric_bias,*[e.weight for e in model.tokenizer.embeddings],model.tokenizer.categorical_bias]))
    ok('all_four_attention_biases_zero',all(torch.count_nonzero(layer.bias).item()==0 for block in model.blocks for layer in [block.attention.W_q,block.attention.W_k,block.attention.W_v,block.attention.W_out]))
    ok('only_first_attention_norm_absent',model.blocks[0].attention_norm is None and isinstance(model.blocks[1].attention_norm,nn.LayerNorm) and all(isinstance(b.ffn_norm,nn.LayerNorm) for b in model.blocks))
    ok('LayerNorm_initial_gamma_beta_epsilon',all(torch.equal(m.weight,torch.ones_like(m.weight)) and torch.count_nonzero(m.bias).item()==0 and m.eps==1e-5 for m in model.modules() if isinstance(m,nn.LayerNorm)))
    ok('ReGLU_left_times_relu_right',torch.equal(ReGLU()(torch.tensor([[2.,-3.,-4.,5.]])),torch.tensor([[0.,-15.]])))
    ok('FFN_fixed_shape_and_activation',all(b.ffn[0].weight.shape==(256,64) and isinstance(b.ffn[1],ReGLU) and b.ffn[3].weight.shape==(64,128) for b in model.blocks))
    seen=[]
    handle=model.blocks[1].attention.register_forward_pre_hook(lambda module,args:seen.append((args[0].shape,args[1].shape)))
    ratio=model(n,c);handle.remove()
    ok('final_attention_Q1_KV168',seen==[(torch.Size([9,1,64]),torch.Size([9,168,64]))])
    ok('final_block_residual_and_FFN_CLS_only',model.blocks[1](model.blocks[0](tok),last=True).shape==(9,1,64))
    ok('positive_finite_ratio_shape',ratio.shape==(9,) and bool(torch.isfinite(ratio).all()) and bool((ratio>0).all()))
    # Independent broadcast-last-residual reference mirrors pinned upstream source.
    mdl=copy.deepcopy(model).double().eval();ref=copy.deepcopy(mdl)
    nn64=n[:3].double();cc=c[:3]
    got=mdl(nn64,cc);expected=upstream_broadcast_reference(ref,nn64,cc)
    difference=float((got-expected).abs().max().item())
    torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-11)
    got.sum().backward();expected.sum().backward()
    maximum=0.
    for (name,p1),(name2,p2) in zip(mdl.named_parameters(),ref.named_parameters()):
        assert name==name2
        torch.testing.assert_close(p1.grad,p2.grad,rtol=1e-9,atol=1e-10)
        maximum=max(maximum,float((p1.grad-p2.grad).abs().max().item()))
    tolerances['upstream_broadcast_equivalence']={'dtype':'float64','output_rtol':1e-10,'output_atol':1e-11,'max_output_abs_difference':difference,'gradient_rtol':1e-9,'gradient_atol':1e-10,'max_gradient_abs_difference':maximum}
    ok('same_weight_upstream_broadcast_CLS_output_and_parameter_gradients',True)
    zero_c=torch.zeros_like(c[:1]);model.zero_grad(set_to_none=True);model(n[:1],zero_c).sum().backward()
    ok('category_zero_is_trainable_not_padding',all(float(e.weight.grad[0].abs().sum())>0 for e in model.tokenizer.embeddings))
    baseline=predict_ratio(model,n,c)
    perturbed=n.clone();perturbed[1:]*=100
    altered=predict_ratio(model,perturbed,c)
    ok('no_attention_across_transaction_rows',np.array_equal(baseline[:1],altered[:1]))
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    permutation=torch.tensor([7,2,0,8,3,1,6,4,5]);one=predict_ratio(model,n,c,batch_size=1)
    perm=predict_ratio(model,n[permutation],c[permutation],batch_size=3)
    np.testing.assert_allclose(one,baseline,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(perm,baseline[permutation.numpy()],rtol=2e-6,atol=2e-7)
    tolerances['eval_batch_permutation']={'rtol':2e-6,'atol':2e-7,'max_batch_abs_difference':float(np.max(np.abs(one-baseline))),'max_permutation_abs_difference':float(np.max(np.abs(perm-baseline[permutation.numpy()])))}
    ok('eval_batch_partition_and_permutation_semantically_equal',True)
    ok('query_prediction_leaves_model_state_unchanged',all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    stream=io.BytesIO();torch.save({'configuration':CONFIG,'cards':cards,'state_dict':model.state_dict()},stream);stream.seek(0)
    saved=torch.load(stream,map_location='cpu',weights_only=True)
    loaded=AttentionPriorRegressor(saved['cards']);loaded.load_state_dict(saved['state_dict'])
    repeat=predict_ratio(loaded,n,c);prices=to_twd(baseline,a['query_price_factor_million']);prices2=to_twd(repeat,a['query_price_factor_million'])
    ok('CPU_serialization_ratio_and_TWD_reload_bitwise_same_batch',baseline.tobytes()==repeat.tobytes() and prices.tobytes()==prices2.tobytes())
    ok('float64_factor_once_TWD_inverse',np.array_equal(prices,baseline*a['query_price_factor_million'].astype(np.float64)*1e6))
    pred=torch.tensor([-2.,-.5,0.,.5,2.],dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([.5,.5,.5,.5,3.],dtype=torch.float64)
    loss=weighted_loss(pred,torch.zeros_like(pred),weights);loss.backward()
    torch.testing.assert_close(loss,torch.tensor(1.5,dtype=torch.float64),rtol=0,atol=1e-15)
    torch.testing.assert_close(pred.grad,torch.tensor([-.1,-.1,0.,.1,.6],dtype=torch.float64),rtol=0,atol=1e-15)
    ok('weighted_total_L1_hand_loss_gradient_no_weight_sum_division',True)
    # Exercise the ACTUAL train_group with unequal 256+256+1 slices cheaply.
    class TinyPositive(nn.Module):
        def __init__(self):super().__init__();self.coefficient=nn.Parameter(torch.tensor(.37,dtype=torch.float64))
        def forward(self,numeric,categorical):return F.softplus(numeric[:,0]*self.coefficient+1)
    tm=TinyPositive();fm=copy.deepcopy(tm)
    tn=torch.linspace(-1,1,513,dtype=torch.float64)[:,None];tc=torch.zeros(513,1,dtype=torch.int64)
    ty=torch.linspace(.5,4,513,dtype=torch.float64);tf=torch.linspace(.4,2,513,dtype=torch.float64);tw=torch.linspace(.2,3,513,dtype=torch.float64)
    opt=optimizer_for(tm);fullopt=optimizer_for(fm)
    actual_s,actual_e=train_group(tm,opt,tn,tc,ty,tf,tw,torch.arange(513))
    fullopt.zero_grad();full_loss=weighted_loss(fm(tn,tc)*tf,ty,tw);full_loss.backward()
    grad_difference=float((tm.coefficient.grad-fm.coefficient.grad).abs().item())
    torch.testing.assert_close(tm.coefficient.grad,fm.coefficient.grad,rtol=1e-12,atol=1e-12)
    fullopt.step();torch.testing.assert_close(tm.coefficient,fm.coefficient,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual_s/513,full_loss.detach(),rtol=1e-12,atol=1e-12)
    tolerances['actual_unequal_microbatch_accumulation']={'rows':513,'microbatch_rows':[256,256,1],'rtol':1e-12,'atol':1e-12,'max_gradient_abs_difference':grad_difference}
    ok('actual_train_group_unequal_microbatches_matches_full_gradient_and_step',True)
    opt=optimizer_for(model)
    ok('optimizer_all_parameters_one_weight_decay_no_exclusions',len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(list(model.parameters())) and opt.param_groups[0]['weight_decay']==.0001)
    original_lr=CONFIG['learning_rate'];CONFIG['learning_rate']=.1
    try:reject('configuration_mutation_rejected',check_configuration)
    finally:CONFIG['learning_rate']=original_lr
    check_configuration()
    return {**receipt_identity(),'status':'passed','checks':checks,'check_count':len(checks),'tolerances':tolerances,
            'GPU_calls':0,'synthetic_optimizer_updates':2,'real_model_fits':0,'torch':str(torch.__version__),
            'CPU_model_state_reload_verified':True,'upstream_same_weight_eval_and_gradient_reference_verified':True,
            'true_fit_provenance_verified':False,'real_query_reload_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--cpu-self-check',action='store_true');group.add_argument('--smoke',action='store_true')
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cardinalities-json',type=Path)
    args=parser.parse_args();started=time.monotonic();failure=None
    mode='cpu-self-check' if args.cpu_self_check else 'smoke' if args.smoke else 'fit'
    if mode=='fit' and (args.data is None or args.cardinalities_json is not None):parser.error('Fit needs --data, without --cardinalities-json')
    if mode=='smoke' and (args.cardinalities_json is None or args.data is not None):parser.error('Smoke needs --cardinalities-json, without --data')
    if mode=='cpu-self-check' and (args.data is not None or args.cardinalities_json is not None):parser.error('CPU self-check accepts no real data or cardinalities input')
    existed=args.out.exists()
    if existed:raise FileExistsError('Output already exists; preserve prior evidence')
    try:
        if mode=='cpu-self-check':
            result=cpu_self_check();result.update(elapsed_seconds=time.monotonic()-started,exit_code=0)
            write(args.out,result);print(json.dumps(result,ensure_ascii=False),flush=True)
        elif mode=='smoke':smoke(args.cardinalities_json,args.out)
        else:run(args.data,args.out)
    except Exception:
        failure=traceback.format_exc();print(failure,flush=True)
        if mode=='cpu-self-check' and not args.out.exists():
            write(args.out,{**receipt_identity(),'status':'failed','exit_code':1,'failure':failure,'elapsed_seconds':time.monotonic()-started})
    if mode!='cpu-self-check':
        if not args.out.exists():args.out.mkdir(parents=True)
        write(args.out/'execution.json',{'mode':mode,'exit_code':1 if failure else 0,'failure':failure,
                                       'elapsed_seconds':time.monotonic()-started,'source_sha256':sha(__file__),
                                       'plan_sha256':PLAN_SHA,'configuration_sha256':content_sha(CONFIG)})
    raise SystemExit(1 if failure else 0)


if __name__=='__main__':main()
