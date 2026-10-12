"""Inference-only v22 neural heads; no saved prediction vectors are read."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime_guard import check_checkpoint_metadata
from query_schema import validate_query_arrays
import numpy as np
import torch
from torch import nn

class EmbeddingMLP(nn.Module):
    """Names and architecture match the archived R47 state_dict."""
    def __init__(self,numeric_count,cardinalities,configuration):
        super().__init__()
        if configuration['hidden']!=[512,256,128]:
            raise ValueError('Unexpected hidden dimensions')
        self.dimensions=[min(32,max(4,math.ceil(math.sqrt(k)))) for k in cardinalities]
        self.embeddings=nn.ModuleList([nn.Embedding(k,d) for k,d in zip(cardinalities,self.dimensions)])
        layers=[];width=numeric_count+sum(self.dimensions)
        for hidden in configuration['hidden']:
            layers.extend([nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(configuration['dropout'])])
            width=hidden
        self.body=nn.Sequential(*layers);self.head=nn.Linear(width,1);self.positive=nn.Softplus()
    def forward(self,numeric,categorical):
        if self.training:raise RuntimeError('Inference only; call eval()')
        embedded=[emb(categorical[:,i]) for i,emb in enumerate(self.embeddings)]
        return self.positive(self.head(self.body(torch.cat([numeric,*embedded],dim=1)))).squeeze(1)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

@torch.inference_mode()
def predict(model_path,inputs_path,spec_path,role,device='cuda:0'):
    """Return float64 TWD; inputs are query-only, without targets or fit arrays."""
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8-sig'))['neural_models'][role]
    if sha256(model_path)!=spec['expected_model_sha256']:
        raise ValueError('Role-specific checkpoint SHA differs before loading')
    ck=torch.load(model_path,map_location='cpu',weights_only=True)
    check_checkpoint_metadata(ck,spec)
    with np.load(inputs_path,allow_pickle=False) as z:
        if set(z.files)!={'query_num','query_cat','cardinalities','query_factor_million'}:
            raise ValueError('Expected only query features/cardinalities/conversion factor')
        numeric,categorical,cardinalities,factor=validate_query_arrays({k:z[k] for k in z.files},spec)
    rows=len(numeric)
    if numeric.dtype!=np.float32 or categorical.dtype!=np.int64 or factor.dtype!=np.float32:
        raise ValueError('Frozen dtypes differ')
    if numeric.shape!=(rows,ck['numeric_count']) or categorical.shape!=(rows,len(ck['cardinalities'])):
        raise ValueError('Checkpoint/input dimensions differ')
    if factor.shape!=(rows,) or not np.isfinite(numeric).all() or not np.isfinite(factor).all() or not (factor>0).all():
        raise ValueError('Invalid query values')
    if cardinalities.dtype!=np.int64 or cardinalities.tolist()!=ck['cardinalities']:
        raise ValueError('Cardinalities differ')
    for i,k in enumerate(cardinalities):
        if not ((categorical[:,i]>=0)&(categorical[:,i]<k)).all():raise ValueError('Invalid category code')
    model=EmbeddingMLP(ck['numeric_count'],ck['cardinalities'],ck['configuration']).to(device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    qn,qc=torch.as_tensor(numeric,device=device),torch.as_tensor(categorical,device=device)
    parts=[model(qn[s:s+4096],qc[s:s+4096]).cpu().numpy() for s in range(0,rows,4096)]
    head=np.concatenate(parts).astype(np.float64)
    result=head*factor.astype(np.float64)*1e6
    if result.shape!=(rows,) or not np.isfinite(result).all() or not (result>0).all():
        raise ValueError('Invalid total-price outputs')
    return result

def check_environment(device):
    """Check imports, device initialization and NumPy bridge without model files."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    # The transfer checks actual device initialization, not only driver discovery.
    probe=torch.ones(1,device=device).cpu().numpy()
    if probe.shape!=(1,) or probe[0]!=1:raise RuntimeError('Torch/NumPy device probe failed')
    return {'status':'ready','python_executable':sys.executable,'torch':str(torch.__version__),
            'numpy':np.__version__,'device':device,'cuda':torch.version.cuda,
            'models_loaded':0,'new_fits':0}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--spec',type=Path);p.add_argument('--role',choices=['N_area','N_prior'])
    p.add_argument('--check-environment',action='store_true',help='Check imports and device without loading models or writing files')
    a=p.parse_args()
    if a.check_environment:
        if any(v is not None for v in (a.model,a.inputs,a.out,a.spec,a.role)):
            p.error('--check-environment cannot be combined with inference inputs')
        print(json.dumps(check_environment(a.device)));return
    if any(v is None for v in (a.model,a.inputs,a.out,a.spec,a.role)):
        p.error('--model, --inputs, --out, --spec and --role are required for prediction')
    if a.out.exists() or a.out.with_suffix('.json').exists():raise FileExistsError('Use a new output path')
    start=time.perf_counter();result=predict(a.model,a.inputs,a.spec,a.role,a.device)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('xb') as f:np.save(f,result,allow_pickle=False)
    receipt={'rows':len(result),'model_sha256':sha256(a.model),'inputs_sha256':sha256(a.inputs),
       'output_sha256':sha256(a.out),'torch':str(torch.__version__),'numpy':np.__version__,
       'device':a.device,'cuda':torch.version.cuda,'prediction_units':'TWD',
       'saved_prediction_vectors_read':False,'models_loaded':1,'new_fits':0,
       'elapsed_seconds':time.perf_counter()-start}
    with a.out.with_suffix('.json').open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()

