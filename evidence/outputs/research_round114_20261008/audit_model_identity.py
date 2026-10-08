"""CPU-only saved R114 checkpoint structure audit; no inference or fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c


def inspect_payload(payload):
    import torch
    import round114_additive_prior_train as trainer
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(114)
    trainer.check_configuration()
    model=trainer.restore_checkpoint(payload,torch.device('cpu'))
    c.need(not list(model.named_buffers()),'No running/BatchNorm buffers')
    c.need(isinstance(model.positive,torch.nn.Identity),'Actual additive signed linear output')
    values={}
    for name,value in payload['state_dict'].items():
        c.need(value.device.type=='cpu' and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Finite CPU F32 checkpoint')
        values[name]={'shape':list(value.shape),'dtype':str(value.dtype),'values_sha256':c.values_sha(value.numpy())}
    return {'payload':{k:v for k,v in payload.items() if k!='state_dict'},'parameters':values,
            'state_order':list(payload['state_dict']),'parameter_count':sum(v.numel() for v in model.parameters()),
            'embedding_dimensions':model.dimensions,'buffer_count':0,'signed_head_Identity':True,
            'CPU_only':True,'model_inference_performed':False,'new_fits':0,'GPU_work':False}


def checkpoint_cpu(path):
    command=[str(ROOT/'.venv_torch/Scripts/python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--checkpoint',str(c.safe(path))]
    p=subprocess.run(command,cwd=ROOT,text=True,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,
                     env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    c.need(p.returncode==0,'CPU checkpoint inspection: '+p.stderr)
    return json.loads(p.stdout)


def self_check():
    import torch
    import round114_additive_prior_train as trainer
    torch.set_num_threads(1);torch.random.default_generator.manual_seed(114)
    model=trainer.AdditivePriorMLP(144,[5]*23)
    payload=trainer.checkpoint_payload(model,[5]*23)
    result=inspect_payload(payload)
    c.need(result['parameter_count']==sum(v.numel() for v in model.parameters()) and result['signed_head_Identity'],
           'Independent synthetic checkpoint constructor/count')
    result.update(status='passed',real_model_or_data_read=False,review_tool=c.record(__file__))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint');g.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    else:
        import torch
        print(json.dumps(inspect_payload(torch.load(c.safe(a.checkpoint),weights_only=True,map_location='cpu')),ensure_ascii=False,allow_nan=False))
