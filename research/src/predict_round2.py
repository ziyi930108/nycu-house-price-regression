"""Reproduce versioned submissions with saved models and local price references."""
from __future__ import annotations
import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from prepare_data import ROOT, DATA_DIR, ID, TARGET, SAMPLE_FILENAME, sha256
from house_features import build_features
from train_lightgbm import encode_features, price_factor


def predict_file(config_path, input_path, output_dir):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    for relative,digest in config['inference_source_sha256'].items():
        if sha256(ROOT/relative)!=digest:
            raise RuntimeError(f'Inference source changed: {relative}')
    known_path=ROOT/config['known_projects_path']
    if sha256(known_path)!=config['known_projects_sha256']:
        raise RuntimeError('Training project reference changed')
    records=config['components']
    for name,record in records.items():
        for path_key,hash_key in [('model_path','model_sha256'),('category_maps_path','category_maps_sha256')]:
            if path_key in record and sha256(ROOT/record[path_key])!=record[hash_key]:
                raise RuntimeError(f'Saved artifact changed: {name}/{path_key}')
    raw=pd.read_excel(input_path,engine='openpyxl',dtype=object)
    if ID not in raw or TARGET in raw or raw[ID].isna().any() or not raw[ID].is_unique:
        raise ValueError('Expected test features with unique, nonmissing IDs and no target')
    x=build_features(raw)
    if list(x.columns)!=config['feature_columns']:
        raise ValueError('Feature schema mismatch')
    known=set(json.loads(known_path.read_text(encoding='utf-8')))
    seen=x['project_name'].isin(known).to_numpy()
    predictions={}
    pending=dict(records)
    reference_cache={}
    while pending:
        progressed=False
        for name,record in list(pending.items()):
            if any(dep not in predictions for dep in record.get('dependencies',[])):
                continue
            family=record['family']
            if family=='catboost':
                features=x[record['feature_columns']]
                model=CatBoostRegressor()
                model.load_model(str(ROOT/record['model_path']))
                p=model.predict(Pool(features,cat_features=record['categorical_columns']),thread_count=5)
                p=p*price_factor(x,record['target_mode'])
            elif family=='lightgbm':
                features=x[record['feature_columns']]
                maps=json.loads((ROOT/record['category_maps_path']).read_text(encoding='utf-8'))
                encoded,_,_,_=encode_features({'train':features},record['categorical_columns'],category_maps=maps)
                model=lgb.Booster(model_file=str(ROOT/record['model_path']))
                p=model.predict(encoded['train'],num_iteration=record['iterations'],num_threads=4)
                p=p*price_factor(x,record['target_mode'])
            elif family=='neighborhood':
                from round2_neighborhood import predict_reference
                model_path=record['model_path']
                if model_path not in reference_cache:
                    with (ROOT/model_path).open('rb') as handle:
                        reference_cache[model_path]=pickle.load(handle)
                baseline=sum(weight*predictions[part] for part,weight in record['base_components'])
                p=predict_reference(reference_cache[model_path],x,baseline,variant=record['variant'])
            else:
                raise ValueError(f'Unsupported model family: {family}')
            p=np.maximum(np.asarray(p,dtype='float64'),0.)
            if p.shape!=(len(x),) or not np.isfinite(p).all():
                raise ValueError(f'Invalid predictions: {name}')
            predictions[name]=p
            del pending[name]
            progressed=True
            print(f'Predicted {name}: {len(p):,} rows',flush=True)
        if not progressed:
            raise ValueError('Unresolved or cyclic component dependencies')
    output_dir.mkdir(parents=True,exist_ok=True)
    sample_path=DATA_DIR/SAMPLE_FILENAME
    if input_path.resolve()==(DATA_DIR/'test-reindex-test-v2.2.xlsx').resolve():
        sample=pd.read_csv(sample_path,encoding='utf-8-sig')
        if raw[ID].astype(str).tolist()!=sample[ID].astype(str).tolist():
            raise ValueError('Input IDs differ from submission template')
    records_out=[]
    for variant in config['variants']:
        name=variant['name']
        if not re.fullmatch(r'[A-Za-z0-9_]+',name): raise ValueError('Invalid version name')
        by_gate={}
        for gate,parts in variant['gates'].items():
            if gate not in {'seen','unseen'} or any(weight<0 for _,weight in parts):
                raise ValueError('Invalid gate or weight')
            if not np.isclose(sum(weight for _,weight in parts),1.):
                raise ValueError('Blend weights must sum to one')
            by_gate[gate]=sum(weight*predictions[part] for part,weight in parts)
        if set(by_gate)!={'seen','unseen'}: raise ValueError('Both project gates are required')
        p=np.where(seen,by_gate['seen'],by_gate['unseen'])
        if not np.isfinite(p).all() or (p>=np.iinfo('int64').max).any():
            raise ValueError('Invalid final prices')
        price=np.maximum(np.rint(p),1).astype('int64')
        submission=pd.DataFrame({ID:raw[ID].to_numpy(),TARGET:price})
        path=output_dir/f'submission_{name}.csv'
        submission.to_csv(path,index=False,encoding='utf-8-sig')
        reread=pd.read_csv(path,encoding='utf-8-sig')
        if list(reread.columns)!=[ID,TARGET] or not np.array_equal(reread[TARGET],price):
            raise ValueError('CSV round-trip check failed')
        if reread[ID].astype(str).tolist()!=raw[ID].astype(str).tolist():
            raise ValueError('CSV ID order changed')
        audit={'name':name,'rows':len(price),'path':str(path),'columns':[ID,TARGET],
               'id_order_verified':True,'positive_finite_integer_prices':True,
               'minimum':int(price.min()),'maximum':int(price.max()),'median':float(np.median(price)),
               'sha256':sha256(path),'test_source_sha256':sha256(input_path),'config_sha256':sha256(config_path),
               'seen_project_rows':int(seen.sum()),'unseen_project_rows':int((~seen).sum())}
        (output_dir/f'submission_{name}_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        records_out.append(audit)
    cache=ROOT/'outputs'/'round2'/'final_predictions'
    cache.mkdir(parents=True,exist_ok=True)
    for name,p in predictions.items(): np.save(cache/f'{name}_test.npy',p)
    (cache/'manifest.json').write_text(json.dumps({'config_sha256':sha256(config_path),'test_source_sha256':sha256(input_path),
        'rows':len(x),'components':list(predictions)},indent=2),encoding='utf-8')
    print(json.dumps(records_out,ensure_ascii=True),flush=True)
    return records_out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'models'/'round2'/'release_config.json')
    parser.add_argument('--input',type=Path,default=DATA_DIR/'test-reindex-test-v2.2.xlsx')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    predict_file(args.config,args.input,args.output_dir)