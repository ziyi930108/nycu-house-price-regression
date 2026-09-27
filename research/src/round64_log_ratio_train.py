"""R64 fixed recency-weighted log-ratio MLP, derived from immutable R63.

Targets use fit labels and driver-supplied group-OOF price factors only.
The linear head changes target/output representation as well as objective:
weighted log-ratio MAE is not total-price MAE. No clipping, epsilon, bias
correction, batch normalization of weights, query labels, or epoch selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import round47_nn_train as prior

ARCHITECTURE_SHA='68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
PRICE_FACTOR_FORMULA='effective_area_A * group_prior_TWD_per_m2 / 1e6'
OUTPUT_UNITS='dimensionless log price ratio'
TRAIN_FACTOR_CONSTRUCTION='group-fold OOF fit-label prior supplied by driver'
QUERY_FACTOR_CONSTRUCTION='full-fit label prior supplied by driver; no query labels'
CONFIG=copy.deepcopy(prior.CONFIG)
CONFIG.update(prediction_head='Softplus dimensionless ratio; ratio * price_factor_million yields million TWD',
              output_units='dimensionless positive price ratio',price_factor_formula=PRICE_FACTOR_FORMULA,
              price_factor_uses_fit_labels=True,
              train_price_factor_construction=TRAIN_FACTOR_CONSTRUCTION,
              query_price_factor_construction=QUERY_FACTOR_CONSTRUCTION,
              loss_formula='abs(ratio_output * train_price_factor_million - train_y_million).mean()',
              initial_bias='inverseSoftplus(median_fit(train_y_million / train_price_factor_million))',
              extra_sample_weights=False,price_or_ratio_clipping=False)

PARENT_TRAINER=Path(__file__).resolve().with_name('round63_recency_weighted_prior_train.py')
PARENT_TRAINER_SHA='26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
CONFIG.update(loss='fit-recency weighted MAE in million TWD',
    loss_formula='mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))',
    extra_sample_weights=True,
    sample_weight_normalization='driver-supplied fit-global normalization; no batch renormalization')
BASELINE_CONFIG=copy.deepcopy(CONFIG)
CONFIG_OVERRIDES={
    'prediction_head':'Linear log price ratio; exp(log_ratio) * price_factor_million yields million TWD',
    'positive_output':'None; unconstrained linear log-ratio head',
    'output_units':OUTPUT_UNITS,
    'loss':'fit-recency weighted MAE in log price ratio',
    'loss_formula':'mean(train_sample_weight * abs(log_ratio_output - train_log_ratio_target))',
    'initial_bias':'median_fit(log(float64(train_y_million)) - log(float64(train_price_factor_million))) (unweighted)',
    'target_formula':'float32(log(float64(train_y_million)) - log(float64(train_price_factor_million)))',
    'inverse_formula':'exp(float64(log_ratio_output)) * float64(price_factor_million) * 1e6',
}
CONFIG.update(CONFIG_OVERRIDES)


class LogRatioMLP(prior.EmbeddingMLP):
    """Identical inherited construction/RNG; forward omits only Softplus."""
    def forward(self,numeric,categorical):
        if self.training:
            drop=torch.rand(categorical.shape,device=categorical.device)<prior.CONFIG['category_dropout']
            categorical=categorical.masked_fill(drop,0)
        embedded=[emb(categorical[:,i]) for i,emb in enumerate(self.embeddings)]
        joined=torch.cat([numeric,*embedded],dim=1)
        return self.head(self.body(joined)).squeeze(1)


def check_configuration():
    if CONFIG!={**BASELINE_CONFIG,**CONFIG_OVERRIDES}:
        raise ValueError('Only fixed log-ratio representation/objective changes are allowed')
    if any(CONFIG[k]!=v for k,v in prior.CONFIG.items() if k not in ('loss','positive_output')):
        raise ValueError('R47 setup/architecture configuration changed')
    if CONFIG['seed']!=42 or CONFIG['epochs']!=30 or CONFIG['batch_size']!=2048:
        raise ValueError('Fixed seed/epochs/batch differ')
    if hashlib.sha256(PARENT_TRAINER.read_bytes()).hexdigest()!=PARENT_TRAINER_SHA:
        raise ValueError('Parent R63 trainer source changed')
    if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()!=ARCHITECTURE_SHA:
        raise ValueError('R47 architecture source changed')


def sample_weight_values(values,rows):
    values=np.asarray(values)
    if rows<=0 or values.dtype!=np.float32 or values.shape!=(rows,) or not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Train sample weights must be ordered positive finite float32 vectors')
    return values


def sample_weight_summary(values):
    v=np.asarray(values,dtype=np.float64)
    total=float(v.sum(dtype=np.float64));squares=float(np.dot(v,v))
    return {'rows':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
            'sum':total,'sum_squares':squares,'effective_sample_size':total*total/squares}


def weighted_loss(log_ratio,target_log_ratio,sample_weight):
    if (log_ratio.ndim!=1 or target_log_ratio.shape!=log_ratio.shape or sample_weight.shape!=log_ratio.shape):
        raise ValueError('Weighted log loss requires aligned one-dimensional tensors')
    return (sample_weight*(log_ratio-target_log_ratio).abs()).mean()


def log_target_values(y_million,price_factor_million):
    y=np.asarray(y_million)
    if y.dtype!=np.float32 or y.ndim!=1 or not len(y) or not np.isfinite(y).all() or not (y>0).all():
        raise ValueError('Log targets require positive finite float32 fit prices')
    factor=price_factor_values(price_factor_million,len(y))
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        values=np.log(y.astype(np.float64))-np.log(factor.astype(np.float64))
        stored=values.astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(stored).all():
        raise FloatingPointError('Nonfinite log targets')
    return values,stored



def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def price_factor_values(values,rows):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(rows,) or not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Price factors must be ordered positive finite float32 vectors')
    return values


def to_twd(log_ratio,price_factor_million):
    z=np.asarray(log_ratio,dtype=np.float64)
    if z.ndim!=1 or not np.isfinite(z).all():raise ValueError('Log-ratio outputs must be finite vectors; signed values are allowed')
    factor=price_factor_values(price_factor_million,len(z))
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        ratio=np.exp(z)
        values=ratio*factor.astype(np.float64)*1e6
    if not np.isfinite(values).all() or not (values>0).all() or not (ratio>0).all():
        raise FloatingPointError('Nonfinite/nonpositive exp/total-price predictions; no clipping')
    return values


def total_million_for_diagnostics(log_ratio,price_factor_million):
    """Same forward, detached float64 diagnostics; never backpropagated."""
    ratio=torch.exp(log_ratio.detach().to(torch.float64))
    values=ratio*price_factor_million.detach().to(torch.float64)
    tiny=torch.finfo(torch.float64).tiny
    if (not torch.isfinite(ratio).all() or not torch.isfinite(values).all()
        or not (ratio>=tiny).all() or not (values>=tiny).all()):
        raise FloatingPointError('Overflow/underflow in detached total-price diagnostics')
    return values


@torch.inference_mode()
def predict_log_ratio(model,numeric,categorical):
    model.eval()
    chunks=[model(numeric[start:start+4096],categorical[start:start+4096]).cpu().numpy()
            for start in range(0,len(numeric),4096)]
    values=np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(values).all():raise ValueError('Nonfinite log-ratio outputs')
    return values


def predict(model,numeric,categorical,price_factor_million):
    return to_twd(predict_log_ratio(model,numeric,categorical),price_factor_million)



def run(data_path,out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('run.json','model.pt','training.json','predictions_twd.npy','log_ratio.npy')):
        raise FileExistsError('Run/output already exists; preserve evidence instead of retrying')
    expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million','train_sample_weight'}
    with np.load(data_path,allow_pickle=False) as arrays:
        if set(arrays.files)!=expected:raise ValueError('Expected exactly nine arrays including fit weights, no query labels')
        xn,xc,y,qn,qc,card,af,qaf,sw=[arrays[k] for k in ('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million','train_sample_weight')]
    assert xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and len(xn)>0 and len(qn)>0
    assert card.dtype==np.int64 and card.ndim==1
    card=[int(k) for k in card]
    assert xn.shape[1]==qn.shape[1] and xc.shape[1]==qc.shape[1]==len(card)
    assert len(xn)==len(xc)==len(y) and len(qn)==len(qc) and y.ndim==1
    assert xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all()
    af=price_factor_values(af,len(xn));qaf=price_factor_values(qaf,len(qn))
    sw=sample_weight_values(sw,len(xn))
    log64,log_target=log_target_values(y,af)
    for i,k in enumerate(card):
        assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    check_configuration()
    assert hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()==ARCHITECTURE_SHA
    device=prior.setup()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    if not np.isfinite(median) or median<=0:raise ValueError('Invalid fit ratio-target median')
    initial=float(np.median(log64))
    if not np.isfinite(np.float32(initial)):raise ValueError('Nonfinite initial log-ratio head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'parent_trainer_source':'src/round63_recency_weighted_prior_train.py','parent_trainer_source_sha256':PARENT_TRAINER_SHA,
       'changed_configuration_fields':['prediction_head','positive_output','output_units','loss','loss_formula','initial_bias'],
       'added_configuration_fields':['target_formula','inverse_formula'],
       'configuration_overrides':copy.deepcopy(CONFIG_OVERRIDES),
       'model_class':'LogRatioMLP','model_output_activation':'linear',
       'target_formula':CONFIG['target_formula'],'inverse_formula':CONFIG['inverse_formula'],
       'training_objective_name':'weighted log-ratio MAE','epoch_selection_uses_training_metrics':False,
       'train_sample_weight_sha256':hashlib.sha256(sw.astype('<f4').tobytes()).hexdigest(),
       'train_sample_weight_shape':list(sw.shape),'train_sample_weight_dtype':'float32',
       'train_sample_weight_summary':sample_weight_summary(sw),
       'sample_weight_provenance_verified_in_trainer':False,'sample_weights_recomputed_in_trainer':False,
       'sample_weights_batch_renormalized':False,
       'architecture_source_sha256':hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
       'input_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
       'fit_rows':len(xn),'query_rows':len(qn),'numeric_count':xn.shape[1],'cardinalities':card,
       'query_labels_available':False,'query_labels_used':False,'prediction_units':'TWD','output_units':OUTPUT_UNITS,
       'price_factor_formula':PRICE_FACTOR_FORMULA,'price_factor_uses_fit_labels':True,
       'train_price_factor_construction':TRAIN_FACTOR_CONSTRUCTION,'query_price_factor_construction':QUERY_FACTOR_CONSTRUCTION,
       'price_factor_provenance_verified_in_trainer':False,
       'train_price_factor_million_sha256':hashlib.sha256(af.astype('<f4').tobytes()).hexdigest(),
       'query_price_factor_million_sha256':hashlib.sha256(qaf.astype('<f4').tobytes()).hexdigest(),
       'price_factor_million_summary':{name:{'rows':len(a),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean(dtype=np.float64))}
                              for name,a in [('train',af),('query',qaf)]},
       'initial_training_median_ratio':median,'initial_training_median_log_ratio':initial,
       'initial_head_bias':initial,'initial_head_bias_float32':float(np.float32(initial)),
       'train_log_ratio_target_sha256':hashlib.sha256(log_target.astype('<f4').tobytes()).hexdigest(),
       'train_log_ratio_target_float64_sha256':hashlib.sha256(log64.astype('<f8').tobytes()).hexdigest(),
       'train_log_ratio_target_summary':{'rows':len(log_target),'min':float(log_target.min()),'max':float(log_target.max()),'mean':float(log_target.mean(dtype=np.float64))},
       'log_ratio_clipping':False,'prediction_bias_correction':False,
       'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
       'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':True,'price_or_ratio_clipping':False}
    write(out/'run.json',meta)
    model=LogRatioMLP(xn.shape[1],card).to(device)
    with torch.no_grad():model.head.bias.fill_(initial)
    tn=torch.as_tensor(xn,device=device);tc=torch.as_tensor(xc,device=device);ty=torch.as_tensor(y,device=device)
    tl=torch.as_tensor(log_target,device=device);ta=torch.as_tensor(af,device=device);tw=torch.as_tensor(sw,device=device);qtn=torch.as_tensor(qn,device=device);qtc=torch.as_tensor(qc,device=device)
    del xn,xc,y,qn,qc,af,sw,log64,log_target
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    epochs=[];fit_started=time.monotonic()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device)
        total=torch.zeros((),device=device,dtype=torch.float64)
        total_mae=torch.zeros_like(total);total_weighted_mae=torch.zeros_like(total)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            index=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            z=model(tn[index],tc[index]);loss=weighted_loss(z,tl[index],tw[index])
            prediction=total_million_for_diagnostics(z,ta[index])
            error=(prediction-ty[index].to(torch.float64)).abs()
            mae=error.mean();weighted_mae=(tw[index].to(torch.float64)*error).mean()
            loss.backward();optimizer.step()
            total+=loss.detach().to(torch.float64)*len(index)
            total_mae+=mae*len(index);total_weighted_mae+=weighted_mae*len(index)
        value=float((total/len(tn)).item());mae_value=float((total_mae/len(tn)).item())
        weighted_mae_value=float((total_weighted_mae/len(tn)).item())
        if not np.isfinite([value,mae_value,weighted_mae_value]).all():
            raise FloatingPointError('Nonfinite log objective/total MAE; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':mae_value,
                       'training_weighted_mae_million_with_dropout':weighted_mae_value,
                       'training_weighted_log_ratio_mae_with_dropout':value})
        schedule.step()
        if epoch==1 or epoch%5==0:
            print(json.dumps({'event':'epoch_complete',**epochs[-1],'elapsed_seconds':time.monotonic()-fit_started}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
    model_path=out/'model.pt'
    torch.save({'numeric_count':meta['numeric_count'],'cardinalities':card,'configuration':CONFIG,
                'model_class':'LogRatioMLP','model_output_activation':'linear',
                'source_sha256':meta['source_sha256'],'parent_trainer_source_sha256':PARENT_TRAINER_SHA,
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},model_path)
    log_values=predict_log_ratio(model,qtn,qtc);predictions=to_twd(log_values,qaf)
    loaded=torch.load(model_path,map_location='cpu',weights_only=True)
    assert loaded['configuration']==CONFIG and loaded['cardinalities']==card and loaded['numeric_count']==meta['numeric_count']
    assert loaded['model_class']=='LogRatioMLP' and loaded['model_output_activation']=='linear'
    assert loaded['source_sha256']==meta['source_sha256'] and loaded['parent_trainer_source_sha256']==PARENT_TRAINER_SHA
    restored=LogRatioMLP(loaded['numeric_count'],loaded['cardinalities']).to(device)
    restored.load_state_dict(loaded['state_dict'])
    reloaded_log=predict_log_ratio(restored,qtn,qtc);reloaded=to_twd(reloaded_log,qaf)
    if log_values.tobytes()!=reloaded_log.tobytes():raise ValueError('Saved model full-query log-ratio reload differs')
    if predictions.tobytes()!=reloaded.tobytes():raise ValueError('Saved model full-query TWD reload differs')
    with (out/'log_ratio.npy').open('xb') as f:np.save(f,log_values,allow_pickle=False)
    with (out/'predictions_twd.npy').open('xb') as f:np.save(f,predictions,allow_pickle=False)
    meta.update(status='completed',epochs=epochs,parameter_count=sum(p.numel() for p in model.parameters()),
       fit_seconds=fit_seconds,saved_model_reload_exact=True,saved_model_log_ratio_reload_exact=True,reload_rows=len(predictions),
       log_ratio_file='log_ratio.npy',log_ratio_values_sha256=hashlib.sha256(log_values.astype('<f8').tobytes()).hexdigest(),
       prediction_min_twd=float(predictions.min()),prediction_max_twd=float(predictions.max()),
       elapsed_seconds=time.monotonic()-start,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ('status','fit_rows','query_rows','fit_seconds','elapsed_seconds','saved_model_reload_exact')}),flush=True)


def cpu_unit_check():
    """One tiny CPU formula/gradient/rejection check; no setup or model."""
    check_configuration();before=copy.deepcopy(prior.CONFIG)
    y=np.array([1.,2.,8.,32.],dtype=np.float32);f=np.array([2.,2.,2.,2.],dtype=np.float32)
    target64,target=log_target_values(y,f)
    assert target.tobytes()==(np.log(y.astype(np.float64))-np.log(f.astype(np.float64))).astype(np.float32).tobytes()
    np.testing.assert_allclose(to_twd(target64,f),y.astype(np.float64)*1e6,rtol=1e-15,atol=0)
    signed=np.array([-1.,0.,1.],dtype=np.float64);factor=np.array([.25,2.,4.],dtype=np.float32)
    np.testing.assert_array_equal(to_twd(signed,factor),np.exp(signed)*factor.astype(np.float64)*1e6)
    z=torch.tensor([-2.,0.,2.],dtype=torch.float64,requires_grad=True)
    t=torch.zeros_like(z);w=torch.tensor([.5,1.,2.],dtype=torch.float64)
    loss=weighted_loss(z,t,w);loss.backward()
    torch.testing.assert_close(loss,torch.tensor(5./3,dtype=torch.float64))
    torch.testing.assert_close(z.grad,torch.tensor([-1./6,0.,2./3],dtype=torch.float64))
    torch.testing.assert_close(weighted_loss(z.detach(),t,torch.ones_like(w)),z.detach().abs().mean(),rtol=0,atol=0)
    np.testing.assert_allclose(total_million_for_diagnostics(z,torch.ones_like(z)).numpy(),np.exp(z.detach().numpy()),rtol=1e-15)
    bad=0
    for value in (0.,-1.,np.nan,np.inf):
        yy=y.copy();yy[0]=value
        try:log_target_values(yy,f)
        except (ValueError,FloatingPointError):bad+=1
        else:raise AssertionError('Invalid log target accepted')
    for value in (np.nan,np.inf,1000.,-1000.):
        try:to_twd(np.array([value]),np.ones(1,dtype=np.float32))
        except (ValueError,FloatingPointError):bad+=1
        else:raise AssertionError('Invalid/overflow/underflow output accepted')
    for value in (0.,np.nan):
        ff=f.copy();ff[0]=value
        try:log_target_values(y,ff)
        except ValueError:bad+=1
        else:raise AssertionError('Invalid factor accepted')
    assert prior.CONFIG==before and CONFIG=={**BASELINE_CONFIG,**CONFIG_OVERRIDES}
    return {'status':'CPU_weighted_log_ratio_passed','log_target_float64_then_float32':True,
            'signed_zero_log_outputs_and_TWD_scale':True,'weighted_loss_gradient_and_uniform_reduction':True,
            'invalid_overflow_underflow_rejections':bad,'R47_configuration_unchanged':True,
            'model_fits':0,'GPU_calls':0,'real_data_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--cpu-unit-check',action='store_true')
    args=parser.parse_args()
    if args.cpu_unit_check:print(json.dumps(cpu_unit_check()),flush=True)
    else:
        if args.data is None or args.out is None:parser.error('--data and --out are required')
        run(args.data,args.out)
