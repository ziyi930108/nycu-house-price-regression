"""R58 local TabM-style model with frozen R54 fit-only numerical ramps.
Same R57 training/loss/head; first-r groups original values, flags and their ramps.
Formula adaptation, not a full official reproduction: https://github.com/yandex-research/tabm
Changing the first-layer width also changes initialization random-number consumption.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
import round57_tabm_train as original
import round54_numeric_encode as numeric_encoder
from round57_tabm_train import BatchEnsembleLinear,loss_value,setup,write,sha,OFFICIAL_URLS

area=original.area
CONFIG=copy.deepcopy(original.CONFIG)
CONFIG.update(numeric_encoding='frozen R54 quantile ramps appended to original 144',
    first_scaling='Normal Z[32,95]; copy by original numeric/flag/ramp and categorical group; independent trainable elements',
    initialization_stream_identical_to_R57_claimed=False)


def group_index_for(state,cardinalities):
    """Validate frozen R54 layout; return int64 original-feature IDs for all inputs."""
    if (state['version']!=1 or state['input_width']!=144 or state['numeric_columns']!=72
        or state['quantiles']!=np.linspace(0.,1.,17).tolist() or state['quantile_method']!='linear'
        or state['fit_rows']<=0 or state['fit_uses_target'] is not False or state['fit_uses_query'] is not False
        or state['original_144_preserved'] is not True or state['ramp_standardization'] is not False
        or state['output_dtype']!='float32' or len(state['columns'])!=72
        or len(cardinalities)!=23 or any(not isinstance(k,int) or k<1 for k in cardinalities)):
        raise ValueError('R58 requires the original fit-only R54 state and 23 category cardinalities')
    groups=list(range(72))*2;offset=144
    for j,entry in enumerate(state['columns']):
        knots=np.asarray(entry['knots'],dtype=np.float64)
        if (knots.ndim!=1 or not 1<=len(knots)<=17 or not np.isfinite(knots).all()
            or not (np.diff(knots)>0).all() or entry['column']!=j or entry['offset']!=offset
            or entry['width']!=len(knots)-1):raise ValueError('Invalid ramp offsets/knots')
        groups.extend([j]*entry['width']);offset+=entry['width']
    if state['output_width']!=offset or state['ramp_width']!=offset-144:raise ValueError('Numeric width differs')
    for c,k in enumerate(cardinalities):groups.extend([72+c]*min(32,max(4,math.ceil(math.sqrt(k)))))
    return np.asarray(groups,dtype=np.int64)


class TabMModel(original.TabMModel):
    """Reuse R57 forward/input dropout, replacing only width and initial r mapping."""
    def __init__(self,numeric_state,cardinalities,initial_bias=0.):
        nn.Module.__init__(self)
        groups=group_index_for(numeric_state,cardinalities)
        self.numeric_state=copy.deepcopy(numeric_state);self.numeric_count=numeric_state['output_width']
        self.k=CONFIG['ensemble_size'];self.dimensions=[min(32,max(4,math.ceil(math.sqrt(k)))) for k in cardinalities]
        self.register_buffer('group_index',torch.from_numpy(groups.copy()))
        self.embeddings=nn.ModuleList([nn.Embedding(k,d) for k,d in zip(cardinalities,self.dimensions)])
        sizes=[len(groups),*CONFIG['hidden']]
        self.layers=nn.ModuleList([BatchEnsembleLinear(a,b,self.k) for a,b in zip(sizes[:-1],sizes[1:])])
        self.dropout=nn.Dropout(CONFIG['dropout'])
        with torch.no_grad():
            z=torch.randn(self.k,95)
            self.layers[0].r.copy_(z[:,self.group_index])
        bound=sizes[-1]**-.5
        self.head_weight=nn.Parameter(torch.empty(self.k,sizes[-1]).uniform_(-bound,bound))
        self.head_bias=nn.Parameter(torch.full((self.k,),float(initial_bias)))


@torch.inference_mode()
def predict_unit(model,numeric,categorical):
    model.eval();chunks=[]
    for start in range(0,len(numeric),CONFIG['predict_batch_size']):
        u=model(numeric[start:start+512],categorical[start:start+512]).to(torch.float64)
        if not torch.isfinite(u).all() or not (u>0).all():raise ValueError('Invalid positive member unit prediction')
        chunks.append(u.mean(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def predict(model,numeric,categorical,factor):
    return area.to_twd(predict_unit(model,numeric,categorical),factor)


def save_model(path,model,cards):
    if path.exists():raise FileExistsError(path)
    torch.save({'numeric_count':model.numeric_count,'numeric_state':model.numeric_state,
        'group_index':model.group_index.cpu().tolist(),'cardinalities':cards,'configuration':CONFIG,
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)


def reload_model(path,device,cards,numeric_state):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    expected=group_index_for(numeric_state,cards)
    if (saved['configuration']!=CONFIG or saved['numeric_state']!=numeric_state
        or saved['numeric_count']!=numeric_state['output_width'] or saved['cardinalities']!=cards
        or saved['group_index']!=expected.tolist()):raise ValueError('Saved model recipe/state differs')
    model=TabMModel(numeric_state,cards).to(device);model.load_state_dict(saved['state_dict'])
    if not np.array_equal(model.group_index.cpu().numpy(),expected):raise ValueError('Reloaded group mapping differs')
    return model


def run(data_path,numeric_state_path,out):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ('run.json','model.pt','training.json','predictions_twd.npy')):raise FileExistsError('Existing attempt; no overwrite or automatic retry')
    numeric_state=json.loads(numeric_state_path.read_text(encoding='utf-8'))
    numeric_state_sha=sha(numeric_state_path)
    keys=['train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor']
    with np.load(data_path,allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('Expected eight R48 arrays; query targets forbidden')
        xn,xc,y,qn,qc,card,af,qaf=[z[k] for k in keys]
    assert xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and xn.shape[1]==qn.shape[1]==numeric_state['output_width']
    assert card.dtype==np.int64 and card.shape==(23,);cards=card.tolist()
    assert xc.shape==(len(xn),23) and qc.shape==(len(qn),23) and y.shape==(len(xn),) and len(xn)>0 and len(qn)>0
    assert xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all()
    af=area.area_values(af,len(xn));qaf=area.area_values(qaf,len(qn))
    for i,k in enumerate(cards):assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    groups=group_index_for(numeric_state,cards)
    assert numeric_state['fit_rows']==len(xn)
    write(out/'numeric_state.json',numeric_state)
    device=setup();torch.cuda.reset_peak_memory_stats()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    initial=median+float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(initial)):raise ValueError('Invalid fit-only head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
        'source_sha256':sha(__file__),'architecture_source_sha256':sha(__file__),
        'source_dependencies':{'src/'+Path(p).name:sha(p) for p in (original.__file__,numeric_encoder.__file__,area.__file__,area.prior.__file__)},
        'official_urls':OFFICIAL_URLS,'local_adaptation_not_official_reproduction':True,
        'input_sha256':sha(data_path),'fit_rows':len(xn),'query_rows':len(qn),'numeric_count':numeric_state['output_width'],'cardinalities':cards,
        'numeric_state_path':numeric_state_path.resolve().as_posix(),'numeric_state_sha256':numeric_state_sha,
        'numeric_state_copy_sha256':sha(out/'numeric_state.json'),
        'group_index':groups.tolist(),'group_index_sha256':hashlib.sha256(groups.astype('<i8').tobytes()).hexdigest(),
        'initialization_group_count':95,'numeric_group_count':72,'feature_group_count':95,'groups_tied_only_at_initialization':True,
        'initialization_stream_identical_to_R57_claimed':False,
        'ensemble_size':32,'query_labels_available':False,'query_labels_used':False,
        'prediction_units':'TWD','output_units':area.OUTPUT_UNITS,'area_formula':area.AREA_FORMULA,'area_factor_formula':area.AREA_FACTOR_FORMULA,
        'train_area_factor_sha256':hashlib.sha256(af.astype('<f4').tobytes()).hexdigest(),
        'query_area_factor_sha256':hashlib.sha256(qaf.astype('<f4').tobytes()).hexdigest(),
        'initial_training_median_unit_1e4_twd_per_m2':median,'initial_head_bias':initial,
        'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':False,'price_or_unit_clipping':False}
    write(out/'run.json',meta)
    model=TabMModel(numeric_state,cards,initial).to(device)
    tn,tc,ty,ta,qtn,qtc=[torch.as_tensor(a,device=device) for a in (xn,xc,y,af,qn,qc)]
    del xn,xc,y,af,qn,qc,card
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'],eta_min=CONFIG['minimum_learning_rate'])
    epochs=[];started=time.monotonic()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(tn),device=device);total=torch.zeros((),device=device)
        lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(tn),CONFIG['batch_size']):
            idx=order[first:first+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=loss_value(model(tn[idx],tc[idx]),ta[idx],ty[idx]);loss.backward();optimizer.step();total+=loss.detach()*len(idx)
        value=float((total/len(tn)).item())
        if not np.isfinite(value):raise FloatingPointError('Nonfinite training loss; no adaptive retry')
        epochs.append({'epoch':epoch,'learning_rate':lr,'training_mae_million_with_dropout':value});scheduler.step()
        if epoch==1 or epoch%5==0:print(json.dumps({'event':'epoch_complete',**epochs[-1],'elapsed_seconds':time.monotonic()-started}),flush=True)
    torch.cuda.synchronize();fit_seconds=time.monotonic()-started
    save_model(out/'model.pt',model,cards)
    values=predict(model,qtn,qtc,qaf);restored=reload_model(out/'model.pt',device,cards,numeric_state)
    replay=predict(restored,qtn,qtc,qaf)
    if values.tobytes()!=replay.tobytes():raise ValueError('Full-query saved model reload differs')
    with (out/'predictions_twd.npy').open('xb') as f:np.save(f,values,allow_pickle=False)
    meta.update(status='completed',epochs=epochs,fit_seconds=fit_seconds,parameter_count=sum(p.numel() for p in model.parameters()),
        saved_model_reload_exact=True,reload_rows=len(values),prediction_min_twd=float(values.min()),prediction_max_twd=float(values.max()),
        elapsed_seconds=time.monotonic()-tick,maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write(out/'training.json',meta)
    print(json.dumps({k:meta[k] for k in ('status','fit_rows','query_rows','fit_seconds','elapsed_seconds','saved_model_reload_exact')}),flush=True)

def smoke_test():
    tick=time.monotonic();device=setup();torch.cuda.reset_peak_memory_stats();cards=[5+i%5 for i in range(23)]
    rng=np.random.default_rng(42);base=np.zeros((512,144),dtype=np.float32)
    base[:,:28]=rng.standard_normal((512,28)).astype(np.float32);base[:,72:]=rng.integers(0,2,(512,72))
    state=numeric_encoder.fit_state(base);state=json.loads(json.dumps(state,allow_nan=False))
    encoded=numeric_encoder.transform(state,base)
    assert state['output_width']==592 and state['columns'][40]['width']==0
    assert encoded[:,:144].tobytes()==base.tobytes()
    model=TabMModel(state,cards,1.).to(device);groups=group_index_for(state,cards);r=model.layers[0].r
    assert np.array_equal(groups[:72],np.arange(72)) and np.array_equal(groups[72:144],np.arange(72))
    for group in range(95):
        pos=np.flatnonzero(groups==group);assert len(pos)>=1
        assert torch.equal(r[:,pos],r[:,pos[0]:pos[0]+1].expand(-1,len(pos)))
    assert set(np.flatnonzero(groups==40))=={40,112}
    assert groups[592]==72 and groups[-1]==94 and groups[144]==0
    assert not torch.equal(r[0],r[1]) and all(torch.equal(l.s,torch.ones_like(l.s)) for l in model.layers)
    assert all(torch.equal(l.r,torch.ones_like(l.r)) for l in model.layers[1:])
    assert all(torch.equal(l.bias,l.bias[:1].expand_as(l.bias)) for l in model.layers)
    numeric=torch.as_tensor(encoded,device=device)
    cats=torch.stack([torch.randint(k,(512,),device=device) for k in cards],dim=1)
    factors=np.resize(np.array([.25,1,4,10],dtype=np.float32),512);factor=torch.as_tensor(factors,device=device)
    target=torch.linspace(1,20,512,device=device)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    u=model(numeric,cats);assert u.shape==(512,32)
    loss=loss_value(u,factor,target);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grad=model.layers[0].r.grad
    assert grad.abs().sum()>0 and not torch.equal(grad[:,0],grad[:,72]) and not torch.equal(grad[:,0],grad[:,144])
    optimizer.step();model.eval()
    with torch.inference_mode():
        h=model.inputs(numeric[:7],cats[:7]);loop=[]
        for member in range(32):
            x=h
            for layer in model.layers:x=torch.relu((x*layer.r[member])@layer.weight.T*layer.s[member]+layer.bias[member])
            loop.append(torch.nn.functional.softplus((x*model.head_weight[member]).sum(-1)+model.head_bias[member]))
        expected=torch.stack(loop,dim=1);actual=model(numeric[:7],cats[:7])
        error=float((expected-actual).abs().max());torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
        assert not torch.equal(actual[:,0],actual[:,1])
    assert loss_value(torch.tensor([[0.,2.]],device=device),torch.ones(1,device=device),torch.ones(1,device=device)).item()==1.
    units=predict_unit(model,numeric,cats);values=predict(model,numeric,cats,factors)
    assert values.dtype==np.float64 and np.array_equal(values,units*factors.astype(np.float64)*1e6)
    assert np.array_equal(area.to_twd(np.array([2.,2.5]),np.array([1.,3.],dtype=np.float32)),[2e6,7.5e6])
    with tempfile.TemporaryDirectory(prefix='round58_smoke_') as folder:
        path=Path(folder)/'model.pt';save_model(path,model,cards);restored=reload_model(path,device,cards,state)
        assert values.tobytes()==predict(restored,numeric,cats,factors).tobytes()
        assert torch.equal(model.group_index,restored.group_index)
    torch.cuda.synchronize()
    result={'status':'synthetic_GPU_smoke_passed','source_sha256':sha(__file__),'rows':512,'ensemble_size':32,
        'numeric_count':state['output_width'],'categorical_count':23,'initialization_group_count':95,
        'noncontiguous_numeric_flag_ramp_groups_verified':True,'constant_zero_width_verified':True,'shifted_cat_groups_verified':True,
        'initial_equal_values_independent_gradients_verified':True,'member_diversity_verified':True,'original_144_bitwise_preserved':True,
        'loss_of_members_counterexample_passed':True,'vectorized_vs_loop_max_abs_error':error,
        'vectorized_vs_loop_tolerance':{'rtol':2e-4,'atol':2e-5},'state_JSON_roundtrip':True,
        'full_512_TWD_reload_bitwise_equal':True,'checkpoint_numeric_state_and_group_index_verified':True,
        'area_factor_applied_once':True,'real_housing_fit':False,
        'maximum_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'maximum_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.monotonic()-tick,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    print(json.dumps(result,ensure_ascii=False),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--numeric-state',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.smoke_test:smoke_test()
    else:
        if args.data is None or args.numeric_state is None or args.out is None:parser.error('--data, --numeric-state and --out required')
        run(args.data,args.numeric_state,args.out)
