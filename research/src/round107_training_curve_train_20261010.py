"""R107 prior-ratio MLP: R63 recency weights with fixed R62 SmoothL1.

The only training-objective change from immutable R63 is
mean(w * smooth_l1(total_million, y_million, beta=.1, reduction='none')).
Beta=.1 million TWD (100,000 TWD), with unit linear-tail slope.
Epoch logs distinguish weighted SmoothL1, weighted MAE and unweighted MAE;
none selects an epoch. Seed42 and all 30 epochs remain fixed.

The nine arrays, fit-only recency weights, group-fold OOF fit price factors,
full-fit query factors, architecture, initialization and inference follow R63.
This trainer validates numeric contracts, not date/fold provenance. There is
no batch renormalization, clipping, query weight, query label or early stopping.
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
import round63_recency_weighted_prior_train as baseline

EmbeddingMLP=prior.EmbeddingMLP
ARCHITECTURE_SHA='68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
PRICE_FACTOR_FORMULA='effective_area_A * group_prior_TWD_per_m2 / 1e6'
OUTPUT_UNITS='dimensionless positive price ratio'
TRAIN_FACTOR_CONSTRUCTION='group-fold OOF fit-label prior supplied by driver'
QUERY_FACTOR_CONSTRUCTION='full-fit label prior supplied by driver; no query labels'
PARENT_TRAINER=Path(__file__).resolve().with_name('round63_recency_weighted_prior_train.py')
PARENT_TRAINER_SHA='26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
BASELINE_CONFIG=copy.deepcopy(baseline.CONFIG)
SMOOTH_L1_BETA_MILLION=0.1
LOSS_CONFIG_OVERRIDES={
    'loss':'fit-recency weighted SmoothL1 in million TWD',
    'loss_formula':"mean(train_sample_weight * smooth_l1_loss(ratio_output * train_price_factor_million, train_y_million, beta=0.1, reduction='none'))",
    'smooth_l1_beta_million':SMOOTH_L1_BETA_MILLION,
    'loss_units':'million TWD',
}
CONFIG={**copy.deepcopy(BASELINE_CONFIG),**LOSS_CONFIG_OVERRIDES}


def check_configuration():
    if hashlib.sha256(PARENT_TRAINER.read_bytes()).hexdigest()!=PARENT_TRAINER_SHA:
        raise ValueError('Parent R63 trainer source changed')
    baseline.check_configuration()
    if BASELINE_CONFIG!=baseline.CONFIG or CONFIG!={**BASELINE_CONFIG,**LOSS_CONFIG_OVERRIDES}:
        raise ValueError('Only the fixed weighted-SmoothL1 configuration may differ from R63')
    if CONFIG['seed']!=42 or CONFIG['epochs']!=30 or CONFIG['batch_size']!=2048:
        raise ValueError('Fixed seed/epochs/batch differ')
    if SMOOTH_L1_BETA_MILLION!=0.1 or CONFIG['smooth_l1_beta_million']!=0.1:
        raise ValueError('Fixed SmoothL1 beta differs')
    if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()!=ARCHITECTURE_SHA:
        raise ValueError('R47 setup/architecture source changed')


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


def weighted_loss(total_prediction_million,target_million,sample_weight):
    """Weighted SmoothL1 of total-price residuals; never divide by batch sum(w)."""
    if (total_prediction_million.ndim!=1 or target_million.shape!=total_prediction_million.shape
        or sample_weight.shape!=total_prediction_million.shape):
        raise ValueError('Weighted loss requires aligned one-dimensional tensors')
    per_row=torch.nn.functional.smooth_l1_loss(
        total_prediction_million,target_million,
        beta=SMOOTH_L1_BETA_MILLION,reduction='none')
    return (sample_weight*per_row).mean()



def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def price_factor_values(values,rows):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(rows,) or not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Price factors must be ordered positive finite float32 vectors')
    return values


def to_twd(ratio,price_factor_million):
    ratio=np.asarray(ratio,dtype=np.float64)
    factor=price_factor_values(price_factor_million,len(ratio))
    if ratio.ndim!=1 or not np.isfinite(ratio).all() or not (ratio>0).all():
        raise ValueError('Network ratio predictions must be positive finite vectors')
    with np.errstate(over='raise',invalid='raise'):
        values=ratio*factor.astype(np.float64)*1e6
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Nonfinite/nonpositive total-price predictions; no clipping')
    return values


@torch.inference_mode()
def predict_ratio(model,numeric,categorical):
    model.eval()
    chunks=[model(numeric[start:start+4096],categorical[start:start+4096]).cpu().numpy()
            for start in range(0,len(numeric),4096)]
    values=np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Nonfinite/nonpositive ratio outputs')
    return values


def predict(model,numeric,categorical,price_factor_million):
    return to_twd(predict_ratio(model,numeric,categorical),price_factor_million)


def run(data_path,out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('run.json','model.pt','training.json','predictions_twd.npy')):
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
    for i,k in enumerate(card):
        assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    check_configuration()
    assert hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()==ARCHITECTURE_SHA
    device=prior.setup()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    if not np.isfinite(median) or median<=0:raise ValueError('Invalid fit ratio-target median')
    inverse_median=median+float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(inverse_median)):raise ValueError('Nonfinite initial ratio head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'parent_trainer_source':'src/round63_recency_weighted_prior_train.py','parent_trainer_source_sha256':PARENT_TRAINER_SHA,
       'changed_configuration_fields':['loss','loss_formula'],
       'added_configuration_fields':['smooth_l1_beta_million','loss_units'],
       'loss_configuration_overrides':copy.deepcopy(LOSS_CONFIG_OVERRIDES),
       'training_objective_name':'weighted SmoothL1',
       'smooth_l1_beta_million':SMOOTH_L1_BETA_MILLION,'smooth_l1_beta_TWD':100000,
       'epoch_selection_uses_training_metrics':False,
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
       'initial_training_median_ratio':median,'initial_head_bias':inverse_median,
       'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
       'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':True,'price_or_ratio_clipping':False}
    write(out/'run.json',meta)
    model=EmbeddingMLP(xn.shape[1],card).to(device)
    with torch.no_grad():model.head.bias.fill_(inverse_median)
    tn=torch.as_tensor(xn,device=device);tc=torch.as_tensor(xc,device=device);ty=torch.as_tensor(y,device=device)
    ta=torch.as_tensor(af,device=device);tw=torch.as_tensor(sw,device=device);qtn=torch.as_tensor(qn,device=device);qtc=torch.as_tensor(qc,device=device)
    del xn,xc,y,qn,qc,af,sw
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    epochs=[];fit_started=time.monotonic()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device);total=torch.zeros((),device=device);total_mae=torch.zeros((),device=device)
        total_weighted_mae=torch.zeros((),device=device)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            index=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            ratio=model(tn[index],tc[index]);prediction=ratio*ta[index]
            loss=weighted_loss(prediction,ty[index],tw[index])
            absolute=(prediction.detach()-ty[index]).abs()
            mae=absolute.mean();weighted_mae=(tw[index]*absolute).mean()
            loss.backward();optimizer.step();total+=loss.detach()*len(index);total_mae+=mae*len(index)
            total_weighted_mae+=weighted_mae*len(index)
        value=float((total/len(tn)).item());mae_value=float((total_mae/len(tn)).item())
        weighted_mae_value=float((total_weighted_mae/len(tn)).item())
        if not all(np.isfinite(v) for v in (value,mae_value,weighted_mae_value)):
            raise FloatingPointError('Nonfinite weighted SmoothL1/weighted MAE/unweighted MAE; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':mae_value,
                       'training_weighted_mae_million_with_dropout':weighted_mae_value,
                       'training_weighted_smooth_l1_million_with_dropout':value})
        schedule.step()
        checkpoint_dir=out/'epoch_checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path=checkpoint_dir/f'epoch_{epoch:02d}.pt'
        with checkpoint_path.open('xb') as checkpoint_file:
            torch.save({'numeric_count':meta['numeric_count'],'cardinalities':card,'configuration':CONFIG,
                        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},checkpoint_file)
        if epoch==1 or epoch%5==0:
            print(json.dumps({'event':'epoch_complete',**epochs[-1],'elapsed_seconds':time.monotonic()-fit_started}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
    model_path=out/'model.pt'
    torch.save({'numeric_count':meta['numeric_count'],'cardinalities':card,'configuration':CONFIG,
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},model_path)
    predictions=predict(model,qtn,qtc,qaf)
    loaded=torch.load(model_path,map_location='cpu',weights_only=True)
    assert loaded['configuration']==CONFIG and loaded['cardinalities']==card and loaded['numeric_count']==meta['numeric_count']
    restored=EmbeddingMLP(loaded['numeric_count'],loaded['cardinalities']).to(device)
    restored.load_state_dict(loaded['state_dict'])
    reloaded=predict(restored,qtn,qtc,qaf)
    if predictions.tobytes()!=reloaded.tobytes():raise ValueError('Saved model full-query TWD reload differs')
    with (out/'predictions_twd.npy').open('xb') as f:np.save(f,predictions,allow_pickle=False)
    meta.update(status='completed',epochs=epochs,parameter_count=sum(p.numel() for p in model.parameters()),
       fit_seconds=fit_seconds,saved_model_reload_exact=True,reload_rows=len(predictions),
       prediction_min_twd=float(predictions.min()),prediction_max_twd=float(predictions.max()),
       elapsed_seconds=time.monotonic()-start,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ('status','fit_rows','query_rows','fit_seconds','elapsed_seconds','saved_model_reload_exact')}),flush=True)


def cpu_unit_check():
    """Tiny CPU-only formula/gradient/weight checks; no setup, model or real data."""
    check_configuration();before=copy.deepcopy(prior.CONFIG)
    baseline_before=copy.deepcopy(baseline.CONFIG)
    w_np=np.array([.5,1.,2.,3.,4.,1.5,2.],dtype=np.float32)
    valid=sample_weight_values(w_np,7);summary=sample_weight_summary(valid)
    assert summary['sum']==14. and summary['sum_squares']==36.5
    assert summary['effective_sample_size']==196./36.5
    residual=torch.tensor([-1.,-.1,-.05,0.,.05,.1,1.],
                          dtype=torch.float64,device='cpu',requires_grad=True)
    target=torch.zeros_like(residual);w=torch.tensor(w_np,dtype=torch.float64,device='cpu')
    expected_rows=torch.tensor([.95,.05,.0125,0.,.0125,.05,.95],
                               dtype=torch.float64,device='cpu')
    loss=weighted_loss(residual,target,w);loss.backward()
    expected=(w*expected_rows).mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=2e-15)
    expected_grad=w*torch.tensor([-1.,-1.,-.5,0.,.5,1.,1.],dtype=torch.float64)/7.
    torch.testing.assert_close(residual.grad,expected_grad,rtol=0,atol=2e-15)
    uniform=weighted_loss(residual.detach(),target,torch.ones_like(w))
    torch.testing.assert_close(uniform,expected_rows.mean(),rtol=0,atol=2e-15)
    weighted_mae=(w*residual.detach().abs()).mean()
    gap=float((weighted_mae-loss.detach()).item())
    assert 0.<=gap<=SMOOTH_L1_BETA_MILLION/2.*float(w.mean())+2e-15
    assert loss.detach()!=(w*expected_rows).sum()/w.sum()
    chunks=(weighted_loss(residual.detach()[:2],target[:2],w[:2])*2+
            weighted_loss(residual.detach()[2:],target[2:],w[2:])*5)/7.
    torch.testing.assert_close(chunks,loss.detach(),rtol=0,atol=2e-15)
    scaled=weighted_loss(residual.detach(),target,w*7.)
    torch.testing.assert_close(scaled,loss.detach()*7.,rtol=0,atol=2e-15)
    f32=residual.detach().float().requires_grad_(True)
    f32loss=weighted_loss(f32,torch.zeros_like(f32),torch.from_numpy(w_np));f32loss.backward()
    assert torch.isfinite(f32loss) and torch.isfinite(f32.grad).all()
    torch.testing.assert_close(f32loss.double(),expected,rtol=1e-6,atol=1e-7)
    bad=[np.ones((7,1),dtype=np.float32),np.ones(7,dtype=np.float64),
         np.ones(6,dtype=np.float32)]
    for value in (0.,-1.,np.nan,np.inf):
        a=w_np.copy();a[0]=value;bad.append(a)
    for values in bad:
        try:sample_weight_values(values,7)
        except ValueError:pass
        else:raise AssertionError('Invalid sample weight accepted')
    for args in ((residual[:,None],target,w),(residual,target[:-1],w),
                 (residual,target,w[:,None])):
        try:weighted_loss(*args)
        except ValueError:pass
        else:raise AssertionError('Misaligned loss tensors accepted')
    assert prior.CONFIG==before and baseline.CONFIG==baseline_before
    assert CONFIG=={**BASELINE_CONFIG,**LOSS_CONFIG_OVERRIDES}
    return {'status':'CPU_weighted_SmoothL1_loss_gradient_passed',
            'beta_million_TWD':0.1,'beta_TWD':100000,
            'zero_symmetry_boundary_tail_slope_checked':True,
            'nonuniform_loss_and_gradient_checked':True,
            'uniform_equals_unweighted_SmoothL1':True,
            'weighted_MAE_minus_smooth_bound_million':[0.,.1],
            'uneven_chunks_row_count_aggregation_checked':True,
            'weight_scaling_without_batch_renormalization_checked':True,
            'weight_contract_rejections':len(bad),'loss_shape_rejections':3,
            'ESS_summary_checked':True,'float32_training_dtype_checked':True,
            'original_setup_configuration_unchanged':True,
            'parent_R63_configuration_unchanged':True,
            'model_fits':0,'GPU_calls':0,'real_data_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--cpu-unit-check',action='store_true')
    args=parser.parse_args()
    if args.cpu_unit_check:print(json.dumps(cpu_unit_check()),flush=True)
    else:
        if args.data is None or args.out is None:parser.error('--data and --out are required')
        run(args.data,args.out)
