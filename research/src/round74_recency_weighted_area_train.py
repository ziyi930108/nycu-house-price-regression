"""R74 fixed area-linked MLP with saved fit-date weights on total-price MAE.

Relative to R48, only the supplied train_sample_weight affects the training
objective. Unit output is still 1e4 TWD per effective square metre; A/100
converts it once to million TWD before loss. Prediction converts to TWD once.
Weights are reused from R63 by the driver, not recomputed or batch-normalized.
No query labels, extra weights from area, clipping, or adaptive epoch selection.
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
import round48_nn_train as area_parent

EmbeddingMLP=prior.EmbeddingMLP
AREA_FORMULA='finite positive building_area else finite positive parking_area else 1'
AREA_FACTOR_FORMULA='A / 100'
OUTPUT_UNITS='1e4 TWD per square metre of effective area A'
PARENT_TRAINER=Path(__file__).resolve().with_name('round48_nn_train.py')
PARENT_TRAINER_SHA='38fb91e01eabb2ed3163f1260e16c826ca55592951c22ce1107ad68c24b37082'
ARCHITECTURE_SHA='68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
WEIGHT_VALIDATION_SOURCE_SHA='26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'
BASELINE_CONFIG=copy.deepcopy(area_parent.CONFIG)
LOSS_CONFIG_OVERRIDES={
    'loss':'fit-recency weighted MAE in million TWD',
    'loss_formula':'mean(train_sample_weight * abs(unit_output * train_area_factor - train_y_million))',
    'extra_sample_weights':True,
    'sample_weight_normalization':'driver-supplied fit-global normalization; no batch renormalization',
}
CONFIG={**BASELINE_CONFIG,**LOSS_CONFIG_OVERRIDES}


def check_configuration():
    if BASELINE_CONFIG!=area_parent.CONFIG or CONFIG!={**BASELINE_CONFIG,**LOSS_CONFIG_OVERRIDES}:
        raise ValueError('Only the fixed weighted-MAE configuration may differ from R48')
    if not all(BASELINE_CONFIG[k]==v for k,v in prior.CONFIG.items()):
        raise ValueError('R47 setup/architecture configuration changed')
    if CONFIG['seed']!=42 or CONFIG['epochs']!=30 or CONFIG['batch_size']!=2048:
        raise ValueError('Fixed seed/epochs/batch differ')
    if (AREA_FORMULA!=area_parent.AREA_FORMULA or AREA_FACTOR_FORMULA!=area_parent.AREA_FACTOR_FORMULA
        or OUTPUT_UNITS!=area_parent.OUTPUT_UNITS):
        raise ValueError('Area formulas or output units differ from R48')
    if hashlib.sha256(PARENT_TRAINER.read_bytes()).hexdigest()!=PARENT_TRAINER_SHA:
        raise ValueError('Parent R48 trainer source changed')
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

def weighted_loss(total_prediction_million,target_million,sample_weight):
    if (total_prediction_million.ndim!=1 or target_million.shape!=total_prediction_million.shape
        or sample_weight.shape!=total_prediction_million.shape):
        raise ValueError('Weighted loss requires aligned one-dimensional tensors')
    return (sample_weight*(total_prediction_million-target_million).abs()).mean()

def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def area_values(values,rows):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(rows,) or not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Area factors must be ordered positive finite float32 vectors')
    return values


def to_twd(unit,area_factor):
    unit=np.asarray(unit,dtype=np.float64)
    factor=area_values(area_factor,len(unit))
    if unit.ndim!=1 or not np.isfinite(unit).all() or not (unit>0).all():
        raise ValueError('Network unit predictions must be positive finite vectors')
    with np.errstate(over='raise',invalid='raise'):
        values=unit*factor.astype(np.float64)*1e6
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Nonfinite/nonpositive total-price predictions; no clipping')
    return values


@torch.inference_mode()
def predict_unit(model,numeric,categorical):
    model.eval()
    chunks=[model(numeric[start:start+4096],categorical[start:start+4096]).cpu().numpy()
            for start in range(0,len(numeric),4096)]
    values=np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Nonfinite/nonpositive unit outputs')
    return values


def predict(model,numeric,categorical,area_factor):
    return to_twd(predict_unit(model,numeric,categorical),area_factor)



def run(data_path,out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('run.json','model.pt','training.json','predictions_twd.npy')):
        raise FileExistsError('Run/output already exists; preserve evidence instead of retrying')
    expected={'train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor','train_sample_weight'}
    with np.load(data_path,allow_pickle=False) as arrays:
        if set(arrays.files)!=expected:raise ValueError('Expected exactly nine arrays including fit weights, no query labels')
        xn,xc,y,qn,qc,card,af,qaf,sw=[arrays[k] for k in ('train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor','train_sample_weight')]
    assert xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and len(xn)>0 and len(qn)>0
    assert card.dtype==np.int64 and card.ndim==1
    card=[int(k) for k in card]
    assert xn.shape[1]==qn.shape[1] and xc.shape[1]==qc.shape[1]==len(card)
    assert len(xn)==len(xc)==len(y) and len(qn)==len(qc) and y.ndim==1
    assert xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all()
    af=area_values(af,len(xn));qaf=area_values(qaf,len(qn))
    sw=sample_weight_values(sw,len(xn))
    for i,k in enumerate(card):
        assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    # Keep R47 architecture/setup unchanged; only the explicit R48 loss keys differ.
    check_configuration()
    device=prior.setup()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    if not np.isfinite(median) or median<=0:raise ValueError('Invalid fit unit-target median')
    inverse_median=median+float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(inverse_median)):raise ValueError('Nonfinite initial unit head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'parent_trainer_source':'src/round48_nn_train.py','parent_trainer_source_sha256':PARENT_TRAINER_SHA,
       'weight_validation_source_sha256':WEIGHT_VALIDATION_SOURCE_SHA,
       'changed_configuration_fields':['loss','loss_formula','extra_sample_weights'],
       'added_configuration_fields':['sample_weight_normalization'],
       'loss_configuration_overrides':copy.deepcopy(LOSS_CONFIG_OVERRIDES),
       'training_objective_name':'weighted MAE','epoch_selection_uses_training_metrics':False,
       'train_sample_weight_sha256':hashlib.sha256(sw.astype('<f4').tobytes()).hexdigest(),
       'train_sample_weight_shape':list(sw.shape),'train_sample_weight_dtype':'float32',
       'train_sample_weight_summary':sample_weight_summary(sw),
       'sample_weight_provenance_verified_in_trainer':False,'sample_weights_recomputed_in_trainer':False,
       'sample_weights_batch_renormalized':False,
       'architecture_source_sha256':hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
       'input_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
       'fit_rows':len(xn),'query_rows':len(qn),'numeric_count':xn.shape[1],'cardinalities':card,
       'query_labels_available':False,'query_labels_used':False,'prediction_units':'TWD','output_units':OUTPUT_UNITS,
       'area_formula':AREA_FORMULA,'area_factor_formula':AREA_FACTOR_FORMULA,'area_from':'driver pure-X inputs; never inferred from target',
       'train_area_factor_sha256':hashlib.sha256(af.astype('<f4').tobytes()).hexdigest(),
       'query_area_factor_sha256':hashlib.sha256(qaf.astype('<f4').tobytes()).hexdigest(),
       'area_factor_summary':{name:{'rows':len(a),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean(dtype=np.float64))}
                              for name,a in [('train',af),('query',qaf)]},
       'initial_training_median_unit_1e4_twd_per_m2':median,'initial_head_bias':inverse_median,
       'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
       'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':True,'price_or_unit_clipping':False}
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
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            index=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            unit=model(tn[index],tc[index]);prediction=unit*ta[index]
            loss=weighted_loss(prediction,ty[index],tw[index])
            mae=(prediction.detach()-ty[index]).abs().mean()
            loss.backward();optimizer.step();total+=loss.detach()*len(index);total_mae+=mae*len(index)
        value=float((total/len(tn)).item());mae_value=float((total_mae/len(tn)).item())
        if not np.isfinite(value) or not np.isfinite(mae_value):
            raise FloatingPointError('Nonfinite weighted objective/unweighted MAE; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':mae_value,
                       'training_weighted_mae_million_with_dropout':value})
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
    """Tiny CPU arithmetic check only: no setup, model, CUDA call, or housing data."""
    before=copy.deepcopy(prior.CONFIG);area_before=copy.deepcopy(area_parent.CONFIG)
    check_configuration()
    u=torch.tensor([2.,4.],dtype=torch.float64,device='cpu',requires_grad=True)
    a=torch.tensor([.5,2.],dtype=torch.float64,device='cpu')
    y=torch.tensor([2.,6.],dtype=torch.float64,device='cpu')
    w_np=np.array([.5,1.5],dtype=np.float32)
    w=torch.as_tensor(sample_weight_values(w_np,2),dtype=torch.float64,device='cpu')
    total=u*a;loss=weighted_loss(total,y,w);loss.backward()
    torch.testing.assert_close(loss,torch.tensor(1.75,dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(u.grad,torch.tensor([-.125,1.5],dtype=torch.float64),rtol=0,atol=0)
    mae=(total.detach()-y).abs().mean()
    assert mae.item()==1.5
    torch.testing.assert_close(weighted_loss(total.detach(),y,torch.ones_like(w)),mae,rtol=0,atol=0)
    first=weighted_loss(total.detach()[:1],y[:1],w[:1])
    last=weighted_loss(total.detach()[1:],y[1:],w[1:])
    assert first.item()==.5
    torch.testing.assert_close((first+last)/2,loss.detach(),rtol=0,atol=0)
    prices=to_twd(np.array([2.,4.]),np.array([.5,2.],dtype=np.float32))
    assert prices.dtype==np.float64 and np.array_equal(prices,np.array([1e6,8e6]))
    assert sample_weight_summary(w_np)=={'rows':2,'min':.5,'max':1.5,'mean':1.,'sum':2.,'sum_squares':2.5,'effective_sample_size':1.6}
    u32=torch.tensor([2.,4.],dtype=torch.float32,device='cpu',requires_grad=True)
    loss32=weighted_loss(u32*a.float(),y.float(),w.float());loss32.backward()
    assert loss32.item()==1.75 and torch.isfinite(u32.grad).all().item()
    for bad in [np.array([.5,0.],dtype=np.float32),np.array([.5,np.nan],dtype=np.float32),
                w_np.astype(np.float64),w_np.reshape(2,1),w_np[:1]]:
        try:sample_weight_values(bad,2)
        except ValueError:pass
        else:raise AssertionError('Invalid weight accepted')
    changed=sorted(k for k in BASELINE_CONFIG if CONFIG[k]!=BASELINE_CONFIG[k])
    added=sorted(set(CONFIG)-set(BASELINE_CONFIG))
    assert changed==['extra_sample_weights','loss','loss_formula']
    assert added==['sample_weight_normalization']
    assert prior.CONFIG==before and area_parent.CONFIG==area_before
    return {'status':'CPU_area_weighted_MAE_math_passed','weighted_loss_million':loss.item(),
       'unweighted_MAE_million':mae.item(),'gradient_wrt_unit':u.grad.tolist(),'inference_TWD':prices.tolist(),
       'uniform_weights_equal_original_MAE':True,'batch_recombination_without_renormalization':True,
       'float32_loss_gradient_finite':True,'invalid_weight_cases_rejected':5,
       'changed_configuration_fields':changed,'added_configuration_fields':added,
       'original_parent_configurations_unchanged':True,'model_fits':0,'GPU_calls':0,'real_data_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--cpu-unit-check',action='store_true')
    args=parser.parse_args()
    if args.cpu_unit_check:print(json.dumps(cpu_unit_check()),flush=True)
    else:
        if args.data is None or args.out is None:parser.error('--data and --out are required for a housing fit')
        run(args.data,args.out)
