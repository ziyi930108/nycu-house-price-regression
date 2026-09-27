"""R22 fixed synthetic MAE/Huber GPU compatibility evidence; no property rows."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor,Pool
import round6_geo as base
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round22_research'
SEED=20260925
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data)
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(hashlib.sha256(data).hexdigest()+'\n')
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'huber_feasibility.json').exists()
    baseline=base.parameters()
    assert baseline['loss_function']=='MAE' and baseline['eval_metric']=='MAE'
    assert baseline['iterations']==2999 and baseline['learning_rate']==.065
    assert baseline['task_type']=='GPU' and baseline['depth']==8
    candidate=dict(baseline,loss_function='Huber:delta=1')
    assert [k for k in baseline if baseline[k]!=candidate[k]]==['loss_function']
    rng=np.random.default_rng(SEED);n=544
    cats=['category_'+str(i) for i in range(21)]
    X=pd.DataFrame({c:pd.Series(['g'+str((row+i)%(11+i%3)) for row in range(n)],dtype=object) for i,c in enumerate(cats)})
    for i in range(66):X['numeric_'+str(i)]=rng.normal(size=n).astype('float32')
    geo=pd.DataFrame({c:np.zeros(n,dtype='float32') for c in base.NUMERIC})
    geo['geo_latitude']=np.full(n,25.,dtype='float32')
    geo['geo_longitude']=np.full(n,121.,dtype='float32')
    geo['geo_has_position']=np.ones(n,dtype='float32')
    geo['geo_match_status']=pd.Series(['road_level']*n,dtype=object)
    geo['geo_grid_002']=pd.Series(['1250:6050']*n,dtype=object)
    geo=geo[base.FEATURES]
    view=base.augment_features(X,geo);categories=base.categories(X,cats)
    assert view.shape==(544,95) and len(categories)==23
    y=2.+.25*np.tanh(X.numeric_0.to_numpy(dtype='float64'))+.5*np.maximum(X.numeric_1.to_numpy(dtype='float64'),0)+(np.arange(n)%19==0)*3.
    assert np.isfinite(y).all() and (y>=0).all()
    result={}
    feature_sha=base.frame_sha(view)
    for name,params in [('mae_control',baseline),('huber_candidate',candidate)]:
        passed=dict(params,iterations=2,verbose=False)
        model=CatBoostRegressor(**passed)
        t=time.monotonic()
        model.fit(Pool(view.iloc[:512],y[:512],cat_features=categories,thread_count=4))
        fit_parameters=model.get_all_params()
        path=OUT/(name+'.cbm');assert not path.exists();model.save_model(str(path))
        loaded=CatBoostRegressor();loaded.load_model(str(path))
        query=Pool(view.iloc[512:],cat_features=categories,thread_count=4)
        before=np.asarray(model.predict(query,thread_count=4),dtype='float64')
        after=np.asarray(loaded.predict(query,thread_count=4),dtype='float64')
        native=loaded.get_all_params()
        assert np.isfinite(before).all() and np.array_equal(before,after)
        assert loaded.tree_count_==2 and len(loaded.feature_names_)==95 and len(loaded.get_cat_feature_indices())==23
        assert native['loss_function']==params['loss_function'] and native['eval_metric']=='MAE'
        assert native['boosting_type']=='Plain' and native['depth']==8 and native['grow_policy']=='SymmetricTree'
        assert np.isclose(native['learning_rate'],.065,rtol=1e-7,atol=0)
        result[name]={'passed_parameters':passed,'fitted_native_parameters':fit_parameters,'saved_native_parameters':native,
            'tree_count':loaded.tree_count_,'feature_count':len(loaded.feature_names_),
            'categorical_positions':loaded.get_cat_feature_indices(),'fit_rows':512,'query_rows':32,
            'saved_reload_predictions_exact':True,'synthetic_query_prediction_sha256':hashlib.sha256(before.astype('<f8').tobytes()).hexdigest(),
            'native_values_changed_during_save':{k:{'fitted':fit_parameters.get(k),'saved':native.get(k)} for k in set(fit_parameters)|set(native) if fit_parameters.get(k)!=native.get(k)},
            'model_path':path.relative_to(ROOT).as_posix(),'model_sha256':SHA(path),'elapsed_seconds':time.monotonic()-t}
    assert base.frame_sha(view)==feature_sha
    a=result['mae_control']['saved_native_parameters'];b=result['huber_candidate']['saved_native_parameters']
    differences={k:{'control':a.get(k),'candidate':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
    evidence={'round':22,'status':'synthetic_GPU_compatibility_passed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
      'synthetic_only':True,'property_X_y_models_or_predictions_loaded':False,'property_MAE_computed':False,
      'baseline_configuration_read_for_parameters_only':True,'synthetic_seed':SEED,'synthetic_model_count':2,
      'feasibility_overrides':{'iterations':2,'verbose':False},'declared_candidate_loss':'Huber:delta=1','declared_delta_TWD':1000000,
      'candidate_passed_parameter_changes':['loss_function'],'candidate_parameters':candidate,'baseline_parameters':baseline,
      'synthetic_feature_sha256':feature_sha,'synthetic_target_sha256':hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),
      'native_control_candidate_differences':differences,'models':result,
      'source_sha256':{name:SHA(ROOT/name) for name in ['src/round22_huber_feasibility.py','src/round6_geo.py','src/prepare_data.py']},
      'input_sha256':{'runs/v1/models/final_config.json':SHA(ROOT/'runs/v1/models/final_config.json')},
      'runtime':{'catboost':catboost.__version__,'numpy':np.__version__,'pandas':pd.__version__},
      'official_documentation':[
        {'url':'https://catboost.ai/docs/en/concepts/loss-functions-regression','accessed':'2026-09-25','observation':'Huber has mandatory delta and is listed as optimizable with GPU support.'},
        {'url':'https://catboost.ai/docs/en/references/training-parameters/common','accessed':'2026-09-25','observation':'Automatic leaf estimation defaults depend on the objective; local native behavior is checked separately.'}],
      'limitations':['Synthetic two-tree compatibility does not establish price accuracy, formal 2999-tree success, or generalization.',
        'Only passed loss_function changes; native automatic defaults may change together with it.',
        'delta=1 is fixed in million-TWD units, not estimated from targets or validation residuals.',
        'Huber is a different risk objective; its minimizer need not equal the conditional median or improve MAE.']}
    save(OUT/'huber_feasibility.json',evidence)
    print(json.dumps({'status':evidence['status'],'evidence_sha256':SHA(OUT/'huber_feasibility.json'),'native_differences':differences},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
