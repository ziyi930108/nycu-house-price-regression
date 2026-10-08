"""Execute only AST-extracted artifact validators with in-memory synthetic files.

No driver import, real artifact load, filesystem mutation, fit, or GPU access.
The caller can retain stdout as a receipt. This checks contracts, not ML quality.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT/'src/round111_112_research.py'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def array_identity(a):
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'values_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


class FakePath:
    def __init__(self, name, state):
        self.name, self.state = str(name), state
    def __truediv__(self, name):
        return FakePath(str(PurePosixPath(self.name)/name), self.state)
    def __lt__(self, other):
        return self.name < other.name
    @property
    def suffix(self):
        return PurePosixPath(self.name).suffix
    def is_file(self):
        return self.name in self.state['files']
    def iterdir(self):
        prefix = self.name+'/'
        return [FakePath(name, self.state) for name in sorted(set(self.state['files'])|set(self.state['dirs']))
                if name.startswith(prefix) and '/' not in name[len(prefix):]]


def fixture(round_number):
    folder=f'outputs/round{round_number}/development/model'
    native='predictions_twd.npy' if round_number==111 else 'native.npy'
    checkpoint='model.pt' if round_number==111 else 'model.cbm'
    metadata='training.json' if round_number==111 else 'metadata.json'
    stage='outputs/stage.json'
    state={'files': {p: hashlib.sha256(p.encode()).hexdigest() for p in
                    [stage,folder+'/'+native,folder+'/'+checkpoint,folder+'/'+metadata]},
           'dirs': set(), 'arrays': {folder+'/'+native: np.array([1.,2.],dtype=np.float64)}}
    def record(name):
        name=name.name if isinstance(name,FakePath) else str(name)
        return {'path':name,'sha256':state['files'][name]}
    layers={}
    for key in ('native','base','reference','spatial','median'):
        path='outputs/'+key+'.npy';value=np.array([1.,2.],dtype=np.float64)
        state['files'][path]=hashlib.sha256(path.encode()).hexdigest();state['arrays'][path]=value
        layers[key]={**record(path),**array_identity(value)}
    state['item']={'stage':'development','candidate_round':round_number,'fit_rows':3,'query_rows':2,
                   'query_labels_available':False,'model_metadata':record(folder+'/'+metadata),
                   'layers':layers,'sealed_models':[record(p) for p in sorted(state['files']) if p.startswith(folder+'/')]}
    def load_array(rec):
        need(record(rec['path'])=={k:rec[k] for k in ('path','sha256')},'Array file SHA differs')
        value=state['arrays'][rec['path']]
        need(array_identity(value)=={k:rec[k] for k in ('dtype','shape','values_sha256')},'Array identity differs')
        return value.copy()
    def exact(a,b,message):
        need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)
    def prices(a,shape,name):
        need(a.dtype==np.float64 and a.shape==shape and np.isfinite(a).all() and (a>=0).all(),name)
    namespace={'c':SimpleNamespace(ROOT=FakePath('',state),need=need,record=record,
                read=lambda p:copy.deepcopy(state['item']),load_array=load_array,exact=exact,
                SPLITS={'development':('unused','unused',3,2)}),
               'np':SimpleNamespace(load=lambda p,**kw:state['arrays'][p.name].copy()),
               'metric':SimpleNamespace(prices=prices)}
    tree=ast.parse(DRIVER.read_text(encoding='utf-8-sig'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('seal_model_folder','validate_stage_artifacts')]
    need(len(functions)==2,'Two validator functions required')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(DRIVER),'exec'),namespace)
    return state,namespace['validate_stage_artifacts'],record(stage),folder


def run():
    checks=[]
    for n in (111,112):
        state,validate,record,folder=fixture(n)
        validate(n,'development',record);checks.append(f'R{n}_valid_complete_stage')
        mutations={
            'metadata_changed':lambda s:s['files'].__setitem__(s['item']['model_metadata']['path'],'f'*64),
            'checkpoint_changed':lambda s:s['files'].__setitem__(folder+'/model.'+('pt' if n==111 else 'cbm'),'f'*64),
            'unsealed_native_member':lambda s:s['files'].__setitem__(folder+'/extra.npy','f'*64),
            'unexpected_suffix':lambda s:s['files'].__setitem__(folder+'/extra.log','f'*64),
            'nested_model_folder':lambda s:s['dirs'].add(folder+'/nested'),
            'wrong_round':lambda s:s['item'].__setitem__('candidate_round',999),
            'wrong_rows':lambda s:s['item'].__setitem__('query_rows',3),
            'query_flag':lambda s:s['item'].__setitem__('query_labels_available',True),
            'missing_layer':lambda s:s['item']['layers'].pop('reference'),
            'array_values_change':lambda s:s['arrays'].__setitem__('outputs/base.npy',np.array([3.,4.])),
            'native_layer_mismatch':lambda s:s['arrays'].__setitem__(folder+'/'+('predictions_twd.npy' if n==111 else 'native.npy'),np.array([3.,4.])),
        }
        for label,mutation in mutations.items():
            state,validate,record,folder=fixture(n);mutation(state)
            try:
                validate(n,'development',record)
            except ValueError:
                checks.append(f'R{n}_reject_{label}')
            else:
                raise ValueError('Validator accepted mutation: '+label)
    return {'status':'passed','checks':checks,'check_count':len(checks),
            'driver_sha256':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
            'test_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'AST-extracted artifact validators only; in-memory fake files/arrays',
            'driver_imported':False,'real_arrays_or_prices_loaded':False,'new_fits':0,'GPU_calls':0}


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
