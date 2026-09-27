"""Immutable plans, chronological partitions and provenance for round five."""
from __future__ import annotations
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from model_common import ROOT,load_data,temporal_indices,full_training_data
from prepare_data import sha256
OUT=ROOT/'outputs/round5'
SPLITS={'development':('2022-08-11','2023-08-10'),'confirmation':('2023-08-11','2024-08-10'),'auxiliary':('2021-08-11','2022-08-10')}
COMPONENTS=('project_mask_catboost',)
FIT_SOURCES=('round5_common.py','round5_runner.py','round5_project_mask.py',
 'round3_features.py','house_features.py','model_common.py','prepare_data.py','train_lightgbm.py','round2_neighborhood.py')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def verify_hashes(mapping):
    for name,digest in mapping.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file(),'Missing/unsafe dependency '+name)
        require(sha256(path)==digest,'Changed dependency '+name)
def fit_source_hashes():return {'src/'+name:sha256(ROOT/'src'/name) for name in FIT_SOURCES}
def read_locked(path):
    path=Path(path);receipt=path.with_suffix('.sha256')
    require(path.is_file() and receipt.is_file(),'Missing immutable contract: '+str(path))
    require(receipt.read_text(encoding='ascii').strip()==sha256(path),'Contract receipt differs: '+str(path))
    return read(path)
def locked_write(path,value):
    path=Path(path);receipt=path.with_suffix('.sha256')
    require(not path.exists() and not receipt.exists(),'Refusing to replace immutable contract')
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8') as handle:handle.write(text)
    with receipt.open('x',encoding='ascii') as handle:handle.write(sha256(path)+'\n')
def plan():
    value=read_locked(OUT/'candidate_plan.json')
    require(value['round']==5 and set(value['candidates'])=={'project_mask_blend'},'Unexpected round/candidates')
    verify_hashes(value['fit_source_sha256']);verify_hashes(value['evaluation_source_sha256']);verify_hashes(value['data_sha256'])
    require(value['splits']=={k:list(v) for k,v in SPLITS.items()},'Split plan changed')
    return value

def authorize(stage,component):
    require(stage in (*SPLITS,'final'),'Unexpected training stage')
    require(component in (*COMPONENTS,'control'),'Unexpected training component')
    require(component!='control' or stage=='auxiliary','Only auxiliary baseline fitting is allowed')
    config=plan()
    if stage!='development':
        frozen=read_locked(OUT/'frozen_candidate.json')
        require(frozen['selected_candidate']=='project_mask_blend','No new candidate qualified; later-stage work prohibited')
        require(frozen['plan_sha256']==sha256(OUT/'candidate_plan.json'),'Selection was made under another plan')
        verify_hashes(frozen['source_sha256']);verify_hashes(frozen['development_artifact_sha256'])
        if component!='control':require(component in frozen['required_components'],'Component not required by frozen candidate')
        if stage=='final':
            acceptance=read_locked(OUT/'accepted_candidate.json')
            require(acceptance['accepted'] is True and acceptance['selected_candidate']==frozen['selected_candidate'],'Final training needs accepted confirmation and auxiliary results')
            require(acceptance['frozen_candidate_sha256']==sha256(OUT/'frozen_candidate.json'),'Acceptance under another candidate')
            verify_hashes(acceptance['evidence_sha256'])
    return config

def partition(stage):
    data=load_data()
    if stage=='final':
        X,y=full_training_data(data);Xq=data['X_test'].reset_index(drop=True)
        fi=np.arange(len(X),dtype='int64');qi=np.arange(len(Xq),dtype='int64');yq=None
    else:
        fi,qi=temporal_indices(data,*SPLITS[stage])
        require(not np.intersect1d(fi,qi).size,'Fit/query overlap')
        X=data['X_train'].iloc[fi].reset_index(drop=True);y=data['y_train'][fi]
        Xq=data['X_train'].iloc[qi].reset_index(drop=True);yq=data['y_train'][qi]
    contract=read_locked(OUT/'candidate_plan.json')
    if stage!='final':
        expected=contract['split_contract'][stage]
        require(expected['fit_rows']==len(fi) and expected['query_rows']==len(qi),'Planned fit/query count differs')
        for label,index in [('fit',fi),('query',qi)]:
            digest=hashlib.sha256(np.asarray(index,dtype='<i8').tobytes()).hexdigest()
            require(expected[label+'_indices_content_sha256']==digest,'Planned index contents differ')
    else:
        require(len(X)==contract['final_contract']['fit_rows'] and len(Xq)==contract['final_contract']['query_rows'],'Full-data partition differs')
    return data,X,y,Xq,yq,fi,qi

def baseline_folder(stage):
    return ROOT/('outputs/round5/baseline/auxiliary' if stage=='auxiliary' else 'outputs/round3/baseline/'+stage)

def load_reference(stage,X):
    if stage=='final':
        path=ROOT/'outputs/round2/neighborhood/final_reference.pkl'
        require(sha256(path)=='7c0bf589b6a8a998d96e4a545a45ce8f4dca71ea68ac42ba8550d3781d2d7bca','Final v2 reference changed')
    else:
        path=baseline_folder(stage)/'reference_v2.pkl'
        if stage in ('development','confirmation'):
            manifest=read(ROOT/'runs/round3_experiments/snapshot_manifest.json')
            require(sha256(path)==manifest['copied_files'][rel(path)]['original_sha256'],'Archived baseline reference differs')
        else:
            metadata=read(baseline_folder(stage)/'v2_control.json')
            require(sha256(path)==metadata['model_sha256'],'Auxiliary reference changed')
    with path.open('rb') as handle:reference=pickle.load(handle)
    require(reference['fit_report']['fitting_rows']==len(X),'Reference fitting-row count differs')
    require(reference['fit_report']['hedonic']['maximum_fit_day_index']==float(np.nanmax(X['trade_day_index'])),'Reference date anchor differs')
    return reference,path
