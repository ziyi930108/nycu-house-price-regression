"""Assemble versioned release configurations from audited, saved model artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import importlib.metadata as md
import numpy as np
from model_common import ROOT, load_data, temporal_indices, mae
from prepare_data import sha256


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def build(base_only=False):
    folder=ROOT/'models'/'round2'; folder.mkdir(parents=True,exist_ok=True)
    data=load_data()
    feature_hash=sha256(ROOT/'src'/'house_features.py')
    base=read('runs/v1/models/final_config.json')
    trained=read('runs/v1/reports/final_training.json')
    if trained['config_sha256']!=sha256(ROOT/'runs/v1/models/final_config.json'):
        raise ValueError('Archived baseline configuration mismatch')
    if feature_hash!=base['feature_source_sha256']:
        raise ValueError('Baseline feature code changed')
    records={r['name']:r for r in trained['components']}
    components={}
    for c in base['components']:
        record={'family':c['family'],'target_mode':c['target_mode'],'iterations':c['iterations'],
                'feature_columns':base['feature_columns'],'categorical_columns':base['categorical_columns'],
                'training_rows':trained['training_rows'],'model_path':'runs/v1/'+c['model_path'],
                'model_sha256':records[c['name']]['model_sha256']}
        if sha256(ROOT/record['model_path'])!=record['model_sha256']: raise ValueError('Baseline model mismatch')
        if 'category_maps_path' in c:
            record['category_maps_path']='runs/v1/'+c['category_maps_path']
            record['category_maps_sha256']=records[c['name']]['category_maps_sha256']
            if sha256(ROOT/record['category_maps_path'])!=record['category_maps_sha256']: raise ValueError('Baseline mapping mismatch')
        components[c['name']]=record
    reference=read('outputs/round2/neighborhood/final_reference.json')
    assert reference['training_rows']==467221
    assert reference['training_sources']==['train','valid']
    assert reference['source_sha256']==data['metadata']['source_sha256']
    assert reference['feature_source_sha256']==feature_hash
    assert reference['script_sha256']==sha256(ROOT/'src/round2_neighborhood.py')
    assert reference['pickle_sha256']==sha256(ROOT/reference['pickle_file'])
    for variant in ['trend','anchor']:
        name='neighborhood_hedonic_'+variant
        components[name]={'family':'neighborhood','variant':variant,'model_path':reference['pickle_file'],
                          'model_sha256':reference['pickle_sha256'],'training_rows':reference['training_rows'],
                          'dependencies':['catboost_raw','lightgbm_l1_unit63'],
                          'base_components':[['catboost_raw',.75],['lightgbm_l1_unit63',.25]]}
    if not base_only:
        path='outputs/round2/lightgbm/final_spatial63/metadata.json'
        m=read(path)
        assert m['training_rows']==467221
        assert m['source_sha256']==data['metadata']['source_sha256']
        assert m['feature_source_sha256']==feature_hash
        assert m['encoding_source_sha256']==sha256(ROOT/'src/train_lightgbm.py')
        assert m['common_source_sha256']==sha256(ROOT/'src/model_common.py')
        assert m['script_sha256']==sha256(ROOT/'src/round2_lightgbm.py')
        model_path='outputs/round2/lightgbm/final_spatial63/model.txt'
        maps_path='outputs/round2/lightgbm/final_spatial63/category_maps.json'
        assert sha256(ROOT/model_path)==m['model_sha256']
        assert sha256(ROOT/maps_path)==m['category_maps_sha256']
        components['lightgbm_unit_spatial63']={'family':'lightgbm','target_mode':'unit',
            'iterations':m.get('iterations',m.get('best_iteration')),'feature_columns':m['feature_columns'],
            'categorical_columns':m['categorical_columns'],'training_rows':m['training_rows'],
            'model_path':model_path,'model_sha256':m['model_sha256'],
            'category_maps_path':maps_path,'category_maps_sha256':m['category_maps_sha256'],
            'metadata_path':path}
    known=sorted((set(data['X_train'].project_name)|set(data['X_valid'].project_name))-{'__MISSING__'})
    known_path=folder/'known_projects.json'
    known_path.write_text(json.dumps(known,ensure_ascii=False),encoding='utf-8')
    catalog={}
    for branch in ['neighborhood','lightgbm','catboost']:
        catalog.update({c['name']:c for c in read(f'outputs/round2/{branch}/candidates.json')})
    fit,ti=temporal_indices(data,'2024-04-01',end='2024-08-10')
    seen=data['X_train'].iloc[ti].project_name.isin(set(data['X_train'].iloc[fit].project_name)-{'__MISSING__'}).to_numpy()
    test_unseen=float((~data['X_test'].project_name.isin(known)).mean())
    variants=[]
    specs=[('v2','相似交易＋訓練側時間趨勢',[['neighborhood_hedonic_trend',1.]]),
           ('v2_anchor','相似交易參考；整體基準不另加未來趨勢',[['neighborhood_hedonic_anchor',1.]])]
    if not base_only:
        specs.append(('v2_mix','75%趨勢參考＋25%空間特徵LightGBM',[['neighborhood_hedonic_trend',.75],['lightgbm_unit_spatial63',.25]]))
    for name,label,parts in specs:
        pv=np.zeros(len(data['y_valid'])); pt=np.zeros(len(ti))
        for candidate,weight in parts:
            c=catalog[candidate]
            if not c.get('valid_predictions'): raise ValueError(f'Original validation missing: {candidate}')
            assert np.array_equal(np.load(ROOT/c['temporal_indices']),ti)
            pv+=weight*np.load(ROOT/c['valid_predictions'])
            pt+=weight*np.load(ROOT/c['temporal_predictions'])
        metrics={'validation_mae':mae(data['y_valid'],pv),
                 'clean_validation_mae':mae(data['y_valid'][data['valid_clean_mask']],pv[data['valid_clean_mask']]),
                 'temporal_mae':mae(data['y_train'][ti],pt),
                 'seen_project_mae':mae(data['y_train'][ti][seen],pt[seen]),
                 'unseen_project_mae':mae(data['y_train'][ti][~seen],pt[~seen])}
        metrics['reweighted_temporal_mae']=(1-test_unseen)*metrics['seen_project_mae']+test_unseen*metrics['unseen_project_mae']
        variants.append({'name':name,'label':label,'gates':{'seen':parts,'unseen':parts},'metrics':metrics})
    sources=['src/predict_round2.py','src/house_features.py','src/prepare_data.py','src/train_lightgbm.py','src/round2_neighborhood.py']
    config={'format_version':1,'recommended_variant':'v2','feature_columns':base['feature_columns'],
            'inference_source_sha256':{p:sha256(ROOT/p) for p in sources},'components':components,'variants':variants,
            'known_projects_path':'models/round2/known_projects.json','known_projects_sha256':sha256(known_path),
            'source_sha256':data['metadata']['source_sha256'],'test_unseen_project_rate':test_unseen,
            'training_sources':['train','valid'],'training_rows':467221,
            'recommendation_basis':'Recommend the simpler local-reference trend model. Keep anchor and spatial blend as distinct public-score experiments; no Kaggle score is available.',
            'runtime_versions':{p:md.version(p) for p in ['numpy','pandas','catboost','lightgbm','scipy','threadpoolctl']}}
    destination=folder/('release_neighborhood.json' if base_only else 'release_config.json')
    destination.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'config':str(destination),'variants':[{'name':v['name'],**v['metrics']} for v in variants]},ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-only',action='store_true')
    args=parser.parse_args()
    build(args.base_only)