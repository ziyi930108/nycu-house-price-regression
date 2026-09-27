"""Release one uniform half-v10/half-v11 candidate from saved unrounded forecasts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def run():
    tick=time.monotonic()
    evaluation=read(ROOT/'outputs/round50/results.json')
    assert evaluation['checked_stages']==['development','confirmation','auxiliary']
    a_path=ROOT/'outputs/round46/trial/predictions.npy'
    b_path=ROOT/'outputs/round48/final/median_recency_log.npy'
    a=np.load(a_path,allow_pickle=False);b=np.load(b_path,allow_pickle=False)
    assert a.shape==b.shape==(47073,) and a.dtype==b.dtype==np.float64
    assert np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all()
    v10=ROOT/'outputs/submission_v10_trial.csv';v11=ROOT/'outputs/submission_v11_trial.csv'
    old={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs').glob('submission_v*.csv')}
    assert sha(v10)=='0c838090470cd09aae3766d20e5e4210c708d0109a68e91d91054c8e06264eac'
    assert sha(v11)=='0ac402455f656f16f28f69c61f41e38870d24399f8aa34955b7953e142b50106'
    sample=ROOT/'台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/test-reindex-test-sample-submission-v2.csv'
    ids=pd.read_csv(sample,encoding='utf-8-sig')['編號'].to_numpy()
    for path,values in [(v10,a),(v11,b)]:
        df=pd.read_csv(path,encoding='utf-8-sig')
        assert list(df)==['編號','總價元'] and len(df)==47073
        np.testing.assert_array_equal(df['編號'].to_numpy(),ids)
        np.testing.assert_array_equal(df['總價元'].to_numpy(),np.maximum(np.rint(values),1).astype(np.int64))
    values=.5*a+.5*b
    assert np.isfinite(values).all() and (values>0).all() and values.max()<np.iinfo(np.int64).max
    rounded=np.maximum(np.rint(values),1).astype(np.int64)
    folder=ROOT/'outputs/round50/trial';folder.mkdir(parents=True,exist_ok=False)
    np.save(folder/'predictions.npy',values,allow_pickle=False)
    path=ROOT/'outputs/submission_v12_trial.csv'
    assert not path.exists()
    pd.DataFrame({'編號':ids,'總價元':rounded}).to_csv(path,index=False,encoding='utf-8-sig',lineterminator='\n')
    check=pd.read_csv(path,encoding='utf-8-sig')
    assert list(check)==['編號','總價元'] and len(check)==47073 and path.read_bytes().startswith(b'\xef\xbb\xbf')
    np.testing.assert_array_equal(check['編號'].to_numpy(),ids)
    np.testing.assert_array_equal(check['總價元'].to_numpy(),rounded)
    assert (check['總價元']>0).all() and pd.api.types.is_integer_dtype(check['總價元'])
    np.testing.assert_array_equal(np.load(folder/'predictions.npy',allow_pickle=False),values)
    assert all(sha(ROOT/n)==h for n,h in old.items())
    changes={}
    for name,base in [('v10',a),('v11',b)]:
        delta=values-base
        changes[name]={'mean_absolute_prediction_change_TWD':float(np.abs(delta).mean()),'maximum_absolute_prediction_change_TWD':float(np.abs(delta).max()),'changed_rounded_rows':int(np.count_nonzero(rounded!=np.maximum(np.rint(base),1).astype(np.int64)))}
    earlier=read(ROOT/'outputs/round48/final/prediction_changes.json')
    old_rows=earlier['top12_absolute_change']['recency_log']['v10']
    top=[]
    for item in old_rows:
        i=item['position']
        top.append({'position':i,'ID':str(ids[i]),'pure_X':item['pure_X'],'v10_TWD':float(a[i]),'v11_TWD':float(b[i]),'v12_TWD':float(values[i]),'signed_change_vs_v10_TWD':float(values[i]-a[i]),'true_price_unknown':True})
    proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'trial_csv_generated','path':str(path.relative_to(ROOT)),'sha256':sha(path),
        'formula':'0.5 * saved_v10_unrounded_TWD + 0.5 * saved_v11_unrounded_TWD, uniformly for every row','rows':47073,
        'rounding':'max(rint(unrounded mixture),1), once','sample_ID_order_verified':True,'positive_integer_roundtrip':True,'UTF8_BOM':True,
        'saved_prediction_reload_exact':True,'new_model_fits':0,'model_predict_calls':0,'reference_spatial_calls':0,
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (a_path,b_path,sample,ROOT/'outputs/round50/results.json')},
        'source_sha256':sha(Path(__file__)),'old_submission_hashes_unchanged':True,'protected_before':old,'prediction_changes':changes,
        'top12_absolute_change_vs_v10':top,'prediction_changes_are_not_errors':True,'individual_ID_rules':False,'target_labels_used':False,
        'method_selected_after_test_X_and_prediction_disagreement_review':True,'ratio_search_performed':False,
        'new_public_mae':None,'private_mae':None,'uploaded':False,'registry_changed':False,'elapsed_seconds':time.monotonic()-tick}
    (folder/'release.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':proof['status'],'path':proof['path'],'sha256':proof['sha256'],'rows':proof['rows'],'prediction_changes':changes,'elapsed_seconds':proof['elapsed_seconds']},ensure_ascii=False))

if __name__=='__main__':
    run()
