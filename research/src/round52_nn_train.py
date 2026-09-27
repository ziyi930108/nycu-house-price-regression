"""R52 fixed prior-relative positive-ratio MLP; total-price MAE, no query labels.

Price factors are supplied by the driver: fit group-fold OOF label priors for
training rows and full-fit label priors for query rows. They are not pure X.
This trainer validates their numeric contract, not the driver's fold provenance.
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

EmbeddingMLP=prior.EmbeddingMLP
ARCHITECTURE_SHA='68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
PRICE_FACTOR_FORMULA='effective_area_A * group_prior_TWD_per_m2 / 1e6'
OUTPUT_UNITS='dimensionless positive price ratio'
TRAIN_FACTOR_CONSTRUCTION='group-fold OOF fit-label prior supplied by driver'
QUERY_FACTOR_CONSTRUCTION='full-fit label prior supplied by driver; no query labels'
CONFIG=copy.deepcopy(prior.CONFIG)
CONFIG.update(prediction_head='Softplus dimensionless ratio; ratio * price_factor_million yields million TWD',
              output_units=OUTPUT_UNITS,price_factor_formula=PRICE_FACTOR_FORMULA,
              price_factor_uses_fit_labels=True,
              train_price_factor_construction=TRAIN_FACTOR_CONSTRUCTION,
              query_price_factor_construction=QUERY_FACTOR_CONSTRUCTION,
              loss_formula='abs(ratio_output * train_price_factor_million - train_y_million).mean()',
              initial_bias='inverseSoftplus(median_fit(train_y_million / train_price_factor_million))',
              extra_sample_weights=False,price_or_ratio_clipping=False)


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
    expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million'}
    with np.load(data_path,allow_pickle=False) as arrays:
        if set(arrays.files)!=expected:raise ValueError('Expected exactly eight arrays, no query labels')
        xn,xc,y,qn,qc,card,af,qaf=[arrays[k] for k in ('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_price_factor_million','query_price_factor_million')]
    assert xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and len(xn)>0 and len(qn)>0
    assert card.dtype==np.int64 and card.ndim==1
    card=[int(k) for k in card]
    assert xn.shape[1]==qn.shape[1] and xc.shape[1]==qc.shape[1]==len(card)
    assert len(xn)==len(xc)==len(y) and len(qn)==len(qc) and y.ndim==1
    assert xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all()
    af=price_factor_values(af,len(xn));qaf=price_factor_values(qaf,len(qn))
    for i,k in enumerate(card):
        assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    # R47's class/setup read its CONFIG. Model and optimizer settings are identical;
    # this module only appends descriptions of the new output/price-factor interpretation.
    assert all(CONFIG[k]==v for k,v in prior.CONFIG.items())
    assert hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()==ARCHITECTURE_SHA
    device=prior.setup()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    if not np.isfinite(median) or median<=0:raise ValueError('Invalid fit ratio-target median')
    inverse_median=median+float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(inverse_median)):raise ValueError('Nonfinite initial ratio head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
       'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':False,'price_or_ratio_clipping':False}
    write(out/'run.json',meta)
    model=EmbeddingMLP(xn.shape[1],card).to(device)
    with torch.no_grad():model.head.bias.fill_(inverse_median)
    tn=torch.as_tensor(xn,device=device);tc=torch.as_tensor(xc,device=device);ty=torch.as_tensor(y,device=device)
    ta=torch.as_tensor(af,device=device);qtn=torch.as_tensor(qn,device=device);qtc=torch.as_tensor(qc,device=device)
    del xn,xc,y,qn,qc,af
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    epochs=[];fit_started=time.monotonic()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device);total=torch.zeros((),device=device)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            index=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            ratio=model(tn[index],tc[index]);loss=(ratio*ta[index]-ty[index]).abs().mean()
            loss.backward();optimizer.step();total+=loss.detach()*len(index)
        value=float((total/len(tn)).item())
        if not np.isfinite(value):raise FloatingPointError('Nonfinite total-price training loss; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':value})
        schedule.step()
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
    """Single tiny arithmetic check; no tensors, setup, model, fit or GPU call."""
    ratio=np.array([.5,1.25,2.],dtype=np.float32)
    factor=np.array([2.,8.,.25],dtype=np.float32)
    values=to_twd(ratio,factor)
    assert values.dtype==np.float64 and np.array_equal(values,np.array([1e6,1e7,5e5]))
    y=np.array([1.,4.,9.],dtype=np.float32);f=np.array([2.,2.,3.],dtype=np.float32)
    median=float(np.median(y.astype(np.float64)/f.astype(np.float64)))
    bias=median+float(np.log(-np.expm1(-median)))
    assert median==2.0 and np.isclose(np.logaddexp(0.,bias),median,rtol=0,atol=1e-15)
    assert all(CONFIG[k]==v for k,v in prior.CONFIG.items()) and CONFIG['seed']==42 and CONFIG['epochs']==30
    try:to_twd(np.array([1.],dtype=np.float32),np.array([0.],dtype=np.float32))
    except ValueError:pass
    else:raise AssertionError('Nonpositive factor accepted')
    return {'status':'CPU_price_factor_conversion_passed','cases':4,'model_fits':0,'GPU_calls':0,
            'output_units':OUTPUT_UNITS,'prediction_units':'TWD','price_factor_uses_fit_labels':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--cpu-unit-check',action='store_true')
    args=parser.parse_args()
    if args.cpu_unit_check:print(json.dumps(cpu_unit_check()),flush=True)
    else:
        if args.data is None or args.out is None:parser.error('--data and --out are required')
        run(args.data,args.out)
