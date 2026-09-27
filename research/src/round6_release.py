"""Build one v3 submission only after the fixed R6 acceptance and full-data fit.

Uses the frozen blend/trend rule; never reads test labels, selects weights or scores.
Existing v1/v2 files are read-only. A new submission is published only after
reproducing the old v2 CSV exactly from its preserved component models.
"""
from __future__ import annotations
import json,pickle,uuid,hashlib,argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor,Pool
import round6_common as c
from round6_geo import predict_model,frame_sha,parameters
from round6_prepare_geo import load_geographic_partition
from round6_evaluate import build_predictions,FORMULAS
from train_lightgbm import encode_features,price_factor
from prepare_data import ID,TARGET,DATA_DIR,SAMPLE_FILENAME
ROOT,OUT=c.ROOT,c.OUT
V2_SHA='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
BASE_FILES={'runs/v1/models/final/catboost_raw.cbm':'a7716df5ae940a26d56c1ef13c79b8e5858aa9521baeb3ba15970b76d1a59356','runs/v1/models/final/lightgbm_l1_unit63/model.txt':'9ea53f7398e4717e4ac544d72122ee601208c7d4d98d675a9a22731ddd4eda63','runs/v1/models/final/lightgbm_l1_unit63/category_maps.json':'72829a220c399cb62dcd9dbb2657ccf12a4160447ee36bcf3738105cafca8bee'}

def round_price(values):
    values=np.asarray(values,dtype='float64')
    c.require(np.isfinite(values).all() and (values>=0).all() and (values<np.iinfo('int64').max).all(),'Invalid prices')
    return np.maximum(np.rint(values),1).astype('int64')

def main(verify_only=False):
    plan=c.authorize('final','geography_catboost')
    c.require(plan['candidate_formulas']==FORMULAS,'Formula differs from fixed plan')
    c.require(c.sha256(ROOT/'outputs/submission_v2.csv')==V2_SHA,'Retained v2 changed')
    c.verify_hashes(BASE_FILES)
    versions_before=c.sha256(ROOT/'runs/versions.json')
    release=OUT/'release';submission=ROOT/'outputs/submission_v3.csv'
    if verify_only:c.require(release.is_dir() and submission.is_file(),'No completed release to verify')
    else:c.require(not release.exists() and not submission.exists(),'Release/submission already exists; never overwrite')
    data,X,y,Xq,unused,fi,qi=c.partition('final')
    cats=data['categorical_columns'];geo,geoq,geomanifest=load_geographic_partition('final',fi,qi)
    folder=OUT/'models/final/geography_catboost';meta=c.read(folder/'metadata.json');state=c.read(folder/'feature_state.json')
    for filename,index in [('fit_indices.npy',fi),('query_indices.npy',qi)]:
        c.require(np.array_equal(np.load(folder/filename,allow_pickle=False),index),'Final fit/query indices differ')
    c.require(meta['plan_sha256']==c.sha256(OUT/'candidate_plan.json') and meta['frozen_candidate_sha256']==c.sha256(OUT/'frozen_candidate.json'),'Full model plan/selection differs')
    c.require(meta['parameters']==parameters() and meta['iterations']==2999 and meta['feature_count']==95 and meta['categorical_feature_count']==23,'Full model configuration differs')
    c.require(meta['outer_fit_rows']==467221 and meta['query_rows']==47073 and meta['reload_64_rows_verified'] is True,'Full model partition/reload differs')
    c.require(meta['fit_target_values_sha256']==hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest(),'Full fit target identity differs')
    c.require(meta['fit_geo_feature_values_sha256']==frame_sha(geo) and meta['query_geo_feature_values_sha256']==frame_sha(geoq),'Final geographic values differ')
    c.require(meta['geography_manifest_sha256']==c.sha256(OUT/'geography/final/manifest.json') and geomanifest['snapshot_year']==2024,'Final snapshot identity differs')
    for filename,key in [('model.cbm','model_sha256'),('feature_state.json','feature_state_sha256'),('predictions.npy','prediction_sha256')]:
        c.require(c.sha256(folder/filename)==meta[key],'Saved full component changed: '+filename)
    c.verify_hashes(meta['fit_source_sha256'])
    new=CatBoostRegressor();new.load_model(str(folder/'model.cbm'))
    c.require(new.tree_count_==2999,'Full component tree count differs')
    geographic=predict_model(new,state,Xq,geoq,cats)
    c.require(np.allclose(geographic,np.load(folder/'predictions.npy',allow_pickle=False),rtol=1e-12,atol=1e-5),'Full replay differs')
    old=CatBoostRegressor();old.load_model(str(ROOT/'runs/v1/models/final/catboost_raw.cbm'))
    old_prediction=np.maximum(np.asarray(old.predict(Pool(Xq,cat_features=cats,thread_count=4),thread_count=4))*1e6,0.)
    maps=c.read(ROOT/'runs/v1/models/final/lightgbm_l1_unit63/category_maps.json')
    encoded,_,_,_=encode_features({'train':Xq},cats,category_maps=maps)
    unit_model=lgb.Booster(model_file=str(ROOT/'runs/v1/models/final/lightgbm_l1_unit63/model.txt'))
    unit_prediction=np.maximum(unit_model.predict(encoded['train'],num_iteration=3999,num_threads=4)*price_factor(Xq,'unit'),0.)
    reference,reference_path=c.load_reference('final',X)
    predictions=build_predictions(['v2_control','geography_blend'],old_prediction,unit_prediction,{'geography_catboost':geographic},reference,Xq)
    sample=pd.read_csv(DATA_DIR/SAMPLE_FILENAME,encoding='utf-8-sig')
    old_csv=pd.read_csv(ROOT/'outputs/submission_v2.csv',encoding='utf-8-sig')
    ids=pd.Series(data['ids_test']).astype(str).tolist()
    c.require(len(Xq)==len(ids)==len(sample)==47073 and pd.Series(ids).is_unique,'Test row/ID contract differs')
    c.require(sample[ID].astype(str).tolist()==old_csv[ID].astype(str).tolist()==ids,'Sample/v2/test ID sequence differs')
    c.require(np.array_equal(round_price(predictions['v2_control']),old_csv[TARGET].to_numpy()),'Preserved models do not reproduce v2 CSV exactly')
    prices=round_price(predictions['geography_blend'])
    if verify_only:
        proof=c.read_locked(release/'release.json')
        c.require(proof['release_source_sha256']==c.sha256(Path(__file__)) and proof['sha256']==c.sha256(submission),'Release source/CSV changed')
        c.require(proof['accepted_candidate_sha256']==c.sha256(OUT/'accepted_candidate.json') and proof['plan_sha256']==c.sha256(OUT/'candidate_plan.json'),'Release plan/acceptance changed')
        c.require(proof['unrounded_prediction_sha256']==c.sha256(release/'predictions.npy'),'Saved unrounded prediction changed')
        c.require(np.allclose(predictions['geography_blend'],np.load(release/'predictions.npy',allow_pickle=False),rtol=1e-12,atol=1e-5),'Unrounded replay differs')
        back=pd.read_csv(submission,encoding='utf-8-sig')
        c.require(list(back.columns)==[ID,TARGET] and back[ID].astype(str).tolist()==ids and np.array_equal(back[TARGET],prices),'Published CSV replay differs')
        c.require(c.sha256(ROOT/'outputs/submission_v2.csv')==V2_SHA and c.sha256(ROOT/'runs/versions.json')==versions_before,'Verification modified prior release/registry')
        print(json.dumps({'status':'verified','submission':c.rel(submission),'rows':47073,'all_predictions_recomputed':True,'v2_reproduced_exactly':True}),flush=True)
        return
    release.parent.mkdir(parents=True,exist_ok=True)
    staging=release.parent/('.release-staging-'+uuid.uuid4().hex);staging.mkdir()
    np.save(staging/'predictions.npy',predictions['geography_blend'],allow_pickle=False)
    csv_path=staging/'submission_v3.csv'
    pd.DataFrame({ID:data['ids_test'],TARGET:prices}).to_csv(csv_path,index=False,encoding='utf-8-sig')
    back=pd.read_csv(csv_path,encoding='utf-8-sig')
    c.require(list(back.columns)==[ID,TARGET] and back[ID].astype(str).tolist()==ids and np.array_equal(back[TARGET],prices),'CSV round-trip differs')
    model_inputs={c.rel(p):c.sha256(p) for p in folder.iterdir() if p.is_file()}
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'release_version':'v3','experiment_round':6,'candidate':'geography_blend','rows':47073,'training_rows':467221,'columns':[ID,TARGET],
      'path':c.rel(submission),'sha256':c.sha256(csv_path),'plan_sha256':c.sha256(OUT/'candidate_plan.json'),'frozen_candidate_sha256':c.sha256(OUT/'frozen_candidate.json'),'accepted_candidate_sha256':c.sha256(OUT/'accepted_candidate.json'),
      'geography_manifest_path':c.rel(OUT/'geography/final/manifest.json'),'geography_manifest_sha256':c.sha256(OUT/'geography/final/manifest.json'),
      'full_component_sha256':model_inputs,'baseline_component_sha256':BASE_FILES,'reference_sha256':c.sha256(reference_path),'release_source_sha256':c.sha256(Path(__file__)),
      'formula':FORMULAS['geography_blend'],'full_component_replayed_all_rows':True,'v2_csv_reproduced_exactly':True,'v2_unchanged_sha256':V2_SHA,
      'id_order_verified':True,'positive_finite_integer_prices':True,'rounding':'max(rint(TWD),1)','minimum':int(prices.min()),'maximum':int(prices.max()),'median':float(np.median(prices)),
      'unrounded_prediction_sha256':c.sha256(staging/'predictions.npy'),'private_score_known':False,'new_public_score_known':False,'external_property_answers_used':False,'uploaded_to_kaggle':False,'versions_file_modified':False,'versions_context_sha256':versions_before}
    c.locked_write(staging/'release.json',proof)
    c.authorize('final','geography_catboost');c.verify_hashes(BASE_FILES)
    c.require(c.sha256(ROOT/'outputs/submission_v2.csv')==V2_SHA and c.sha256(ROOT/'runs/versions.json')==versions_before,'Preserved release/version registry changed')
    c.require(not release.exists() and not submission.exists(),'Destination appeared during release')
    c.require(staging.resolve().parent==release.resolve().parent and release.resolve().is_relative_to(ROOT.resolve()) and submission.resolve().is_relative_to(ROOT.resolve()),'Unsafe publication path')
    staging.rename(release)
    (release/'submission_v3.csv').rename(submission)
    c.require(c.sha256(submission)==proof['sha256'],'Published submission hash differs')
    print(json.dumps({'status':'created','submission':c.rel(submission),'rows':47073,'sha256':proof['sha256'],'v2_reproduced_exactly':True,'uploaded':False},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    main(parser.parse_args().verify)
