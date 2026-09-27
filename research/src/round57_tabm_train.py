"""R57 local TabM-style adaptation; not a complete official reproduction.
Sources: https://arxiv.org/html/2410.24210v3
https://github.com/yandex-research/tabm (architecture/initialization ideas).
Independently implemented formulas; shared embeddings, R48 area head and MAE.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
import round48_nn_train as area

OFFICIAL_URLS=['https://arxiv.org/html/2410.24210v3','https://github.com/yandex-research/tabm']
CONFIG=copy.deepcopy(area.CONFIG)
CONFIG.update(architecture='local TabM-style BatchEnsemble; no LayerNorm',ensemble_size=32,
    hidden=[512,256,128],batch_size=512,predict_batch_size=512,
    loss_formula='mean_{row,member}(abs(unit[row,member]*area_factor[row]-y_million[row]))',
    inference_formula='mean(float64 positive member units) * float64 area_factor * 1e6',
    first_scaling='Normal(0,1); numeric j and missing j+72 paired; category channels grouped at initialization only',
    layer_scaling='all r/s ones except first r',head='independent member linear then Softplus')
write=area.write

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class BatchEnsembleLinear(nn.Module):
    def __init__(self,d_in,d_out,k):
        super().__init__();bound=d_in**-.5
        self.weight=nn.Parameter(torch.empty(d_out,d_in).uniform_(-bound,bound))
        self.r=nn.Parameter(torch.ones(k,d_in));self.s=nn.Parameter(torch.ones(k,d_out))
        initial=torch.empty(d_out).uniform_(-bound,bound)
        self.bias=nn.Parameter(initial.unsqueeze(0).repeat(k,1))
    def forward(self,h):
        return torch.matmul(h*self.r,self.weight.T)*self.s+self.bias

class TabMModel(nn.Module):
    def __init__(self,numeric_count,cardinalities,initial_bias=0.):
        super().__init__()
        if numeric_count!=144 or len(cardinalities)!=23:raise ValueError('R57 expects original 144 numeric and 23 categorical inputs')
        self.k=CONFIG['ensemble_size']
        self.dimensions=[min(32,max(4,math.ceil(math.sqrt(k)))) for k in cardinalities]
        self.embeddings=nn.ModuleList([nn.Embedding(k,d) for k,d in zip(cardinalities,self.dimensions)])
        sizes=[numeric_count+sum(self.dimensions),*CONFIG['hidden']]
        self.layers=nn.ModuleList([BatchEnsembleLinear(a,b,self.k) for a,b in zip(sizes[:-1],sizes[1:])])
        self.dropout=nn.Dropout(CONFIG['dropout'])
        with torch.no_grad():
            first=self.layers[0].r;v=torch.randn(self.k,72)
            first[:,:72].copy_(v);first[:,72:144].copy_(v);offset=144
            for width in self.dimensions:
                first[:,offset:offset+width].copy_(torch.randn(self.k,1).expand(-1,width));offset+=width
        bound=sizes[-1]**-.5
        self.head_weight=nn.Parameter(torch.empty(self.k,sizes[-1]).uniform_(-bound,bound))
        self.head_bias=nn.Parameter(torch.full((self.k,),float(initial_bias)))
    def inputs(self,numeric,categorical):
        if self.training:
            mask=torch.rand(categorical.shape,device=categorical.device)<CONFIG['category_dropout']
            categorical=categorical.masked_fill(mask,0)
        return torch.cat([numeric,*[emb(categorical[:,i]) for i,emb in enumerate(self.embeddings)]],dim=1)
    def forward(self,numeric,categorical):
        h=self.inputs(numeric,categorical).unsqueeze(1).expand(-1,self.k,-1)
        for layer in self.layers:h=self.dropout(torch.relu(layer(h)))
        return torch.nn.functional.softplus((h*self.head_weight).sum(-1)+self.head_bias)

def loss_value(units,factors,target):
    return (units*factors[:,None]-target[:,None]).abs().mean()

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

def setup():
    for key in ('seed','thread_count','float_precision'):
        if CONFIG[key]!=area.prior.CONFIG[key]:raise ValueError('Shared setup contract differs')
    return area.prior.setup()

def save_model(path,model,cards):
    if path.exists():raise FileExistsError(path)
    torch.save({'numeric_count':144,'cardinalities':cards,'configuration':CONFIG,
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)

def reload_model(path,device,cards):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['configuration']!=CONFIG or saved['numeric_count']!=144 or saved['cardinalities']!=cards:raise ValueError('Saved model recipe differs')
    model=TabMModel(144,cards).to(device);model.load_state_dict(saved['state_dict']);return model

def run(data_path,out):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ('run.json','model.pt','training.json','predictions_twd.npy')):raise FileExistsError('Existing attempt; no overwrite or automatic retry')
    keys=['train_num','train_cat','train_y_million','query_num','query_cat','cardinalities','train_area_factor','query_area_factor']
    with np.load(data_path,allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('Expected eight R48 arrays; query targets forbidden')
        xn,xc,y,qn,qc,card,af,qaf=[z[k] for k in keys]
    assert xn.ndim==qn.ndim==xc.ndim==qc.ndim==2 and xn.shape[1]==qn.shape[1]==144
    assert card.dtype==np.int64 and card.shape==(23,);cards=card.tolist()
    assert xc.shape==(len(xn),23) and qc.shape==(len(qn),23) and y.shape==(len(xn),) and len(xn)>0 and len(qn)>0
    assert xn.dtype==qn.dtype==y.dtype==np.float32 and xc.dtype==qc.dtype==np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y>0).all()
    af=area.area_values(af,len(xn));qaf=area.area_values(qaf,len(qn))
    for i,k in enumerate(cards):assert k>=1 and ((xc[:,i]>=0)&(xc[:,i]<k)).all() and ((qc[:,i]>=0)&(qc[:,i]<k)).all()
    device=setup();torch.cuda.reset_peak_memory_stats()
    median=float(np.median(y.astype(np.float64)/af.astype(np.float64)))
    initial=median+float(np.log(-np.expm1(-median)))
    if not np.isfinite(np.float32(initial)):raise ValueError('Invalid fit-only head bias')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':CONFIG,
        'source_sha256':sha(__file__),'architecture_source_sha256':sha(__file__),
        'source_dependencies':{Path(p).name:sha(p) for p in (area.__file__,area.prior.__file__)},
        'official_urls':OFFICIAL_URLS,'local_adaptation_not_official_reproduction':True,
        'input_sha256':sha(data_path),'fit_rows':len(xn),'query_rows':len(qn),'numeric_count':144,'cardinalities':cards,
        'ensemble_size':32,'query_labels_available':False,'query_labels_used':False,
        'prediction_units':'TWD','output_units':area.OUTPUT_UNITS,'area_formula':area.AREA_FORMULA,'area_factor_formula':area.AREA_FACTOR_FORMULA,
        'train_area_factor_sha256':hashlib.sha256(af.astype('<f4').tobytes()).hexdigest(),
        'query_area_factor_sha256':hashlib.sha256(qaf.astype('<f4').tobytes()).hexdigest(),
        'initial_training_median_unit_1e4_twd_per_m2':median,'initial_head_bias':initial,
        'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'bitwise_retraining_determinism_claimed':False,'extra_sample_weights':False,'price_or_unit_clipping':False}
    write(out/'run.json',meta)
    model=TabMModel(144,cards,initial).to(device)
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
    values=predict(model,qtn,qtc,qaf);restored=reload_model(out/'model.pt',device,cards)
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
    model=TabMModel(144,cards,1.).to(device);r=model.layers[0].r
    assert torch.equal(r[:,:72],r[:,72:144]);offset=144
    for width in model.dimensions:
        assert torch.equal(r[:,offset:offset+width],r[:,offset:offset+1].expand(-1,width));offset+=width
    assert not torch.equal(r[0],r[1]) and all(torch.equal(l.s,torch.ones_like(l.s)) for l in model.layers)
    assert all(torch.equal(l.r,torch.ones_like(l.r)) for l in model.layers[1:])
    assert all(torch.equal(l.bias,l.bias[:1].expand_as(l.bias)) for l in model.layers)
    assert torch.equal(model.head_bias,torch.ones_like(model.head_bias)) and not torch.equal(model.head_weight[0],model.head_weight[1])
    numeric=torch.randn(512,144,device=device)
    cats=torch.stack([torch.randint(k,(512,),device=device) for k in cards],dim=1)
    factors=np.resize(np.array([.25,1,4,10],dtype=np.float32),512);factor=torch.as_tensor(factors,device=device)
    target=torch.linspace(1,20,512,device=device)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    u=model(numeric,cats);assert u.shape==(512,32)
    loss=loss_value(u,factor,target);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.layers[0].r.grad.abs().sum()>0 and model.head_weight.grad.abs().sum()>0
    assert not torch.equal(model.layers[0].r.grad[:,:72],model.layers[0].r.grad[:,72:144])
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
    counter=loss_value(torch.tensor([[0.,2.]],device=device),torch.ones(1,device=device),torch.ones(1,device=device))
    assert counter.item()==1. and (torch.tensor([0.,2.]).mean()-1.).abs().item()==0.
    units=predict_unit(model,numeric,cats);values=predict(model,numeric,cats,factors)
    assert values.dtype==np.float64 and np.array_equal(values,units*factors.astype(np.float64)*1e6)
    assert np.array_equal(area.to_twd(np.array([2.,2.5]),np.array([1.,3.],dtype=np.float32)),[2e6,7.5e6])
    with tempfile.TemporaryDirectory(prefix='round57_smoke_') as folder:
        path=Path(folder)/'model.pt';save_model(path,model,cards);restored=reload_model(path,device,cards)
        assert values.tobytes()==predict(restored,numeric,cats,factors).tobytes()
    torch.cuda.synchronize()
    result={'status':'synthetic_GPU_smoke_passed','source_sha256':sha(__file__),'rows':512,'ensemble_size':32,'numeric_count':144,'categorical_count':23,
        'loss_of_members_counterexample_passed':True,'initial_feature_groups_and_gradients_verified':True,'member_diversity_verified':True,
        'vectorized_vs_loop_max_abs_error':error,'vectorized_vs_loop_tolerance':{'rtol':2e-4,'atol':2e-5},
        'full_512_TWD_reload_bitwise_equal':True,'area_factor_applied_once':True,'real_housing_fit':False,
        'maximum_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'maximum_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.monotonic()-tick,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.smoke_test:smoke_test()
    else:
        if args.data is None or args.out is None:parser.error('--data and --out required')
        run(args.data,args.out)
