"""CPU-only engineering-v2 R117 ratio checkpoint inspection; no inference/fit/GPU."""
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
    import round117_transaction_prior_train as trainer
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(117)
    model=trainer.validate_checkpoint(payload)
    c.need(not list(model.named_buffers()),'Original MLP has no running/BatchNorm buffers')
    c.need(isinstance(model.positive,torch.nn.Softplus),'Actual original positive ratio head')
    values={}
    for name,value in payload['state_dict'].items():
        c.need(value.device.type=='cpu' and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Finite CPU F32 checkpoint')
        values[name]={'shape':list(value.shape),'dtype':str(value.dtype),'values_sha256':c.values_sha(value.numpy())}
    return {'payload':{k:v for k,v in payload.items() if k!='state_dict'},'parameters':values,
        'state_order':list(payload['state_dict']),'parameter_count':sum(v.numel() for v in model.parameters()),
        'embedding_dimensions':model.dimensions,'buffer_count':0,'positive_ratio_Softplus_head':True,
        'CPU_only':True,'model_inference_performed':False,'new_fits':0,'GPU_work':False}


def checkpoint_cpu(path):
    command=[str(ROOT/'.venv_torch/Scripts/python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--checkpoint',str(c.safe(path))]
    result=subprocess.run(command,cwd=ROOT,text=True,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    c.need(result.returncode==0,'CPU checkpoint inspection: '+result.stderr)
    return json.loads(result.stdout)


def self_check():
    import copy
    import torch
    import round117_transaction_prior_train as trainer
    torch.set_num_threads(1);torch.random.default_generator.manual_seed(117)
    model=trainer.EmbeddingMLP(144,[5]*23)
    payload={'numeric_count':144,'cardinalities':[5]*23,'configuration':copy.deepcopy(trainer.CONFIG),
        'actual_model_class':trainer.MODEL_CLASS,'architecture_source_sha256':trainer.ARCHITECTURE_SHA,
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    result=inspect_payload(payload)
    c.need(result['parameter_count']==sum(v.numel() for v in model.parameters()) and result['positive_ratio_Softplus_head'],
        'Independent original synthetic checkpoint constructor/count')
    rejected=[]
    for name,field,value in [('configuration','configuration',trainer.BASELINE_CONFIG),('width','numeric_count',143),
        ('class','actual_model_class','wrong'),('architecture','architecture_source_sha256','wrong')]:
        try:inspect_payload({**payload,field:value})
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid checkpoint accepted '+name)
    result.update(status='passed',rejected_checkpoint_corruption_cases=rejected,real_model_or_data_read=False,review_tool=c.record(__file__))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint');group.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check()
    else:
        import torch
        print(json.dumps(inspect_payload(torch.load(c.safe(args.checkpoint),weights_only=True,map_location='cpu')),ensure_ascii=False,allow_nan=False))
