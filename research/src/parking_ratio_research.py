"""One target-free feasibility profile; no price/model reads or candidate evaluation."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parking_ratio_research'
EXPECTED={
'data_cache/train_features.pkl':'7ad92824e7410ae86a79e11c17199f3136651f50908b3ac06b1949734ac4a7a7',
'data_cache/test_features.pkl':'5c6c1b220923bbc405e531c31b25595717b63b49afcffb2c3c438facaf9691d4',
'outputs/round6/geography/development/train_source_cache_indices.npy':'7694e4fcc14989221c3f82abac1210b5258339c8ebd2b84261df4b2f74941ee4',
'outputs/round6/models/development/geography_catboost/fit_indices.npy':'8ce930622540ef9321c94b347c58adad4005c7e4dd209484874594b2bb5fe0ac',
'outputs/round6/models/development/geography_catboost/query_indices.npy':'2e1d097650f727b3e03a19d1d6aca811b8aab546573a7900a6edafeb61ed080d',
'src/house_features.py':'108e42d4429961a10a704f85bd66791599d904e6e96381c4c2f6dd4275316388'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def profile(X):
    area=X['parking_area'].to_numpy(dtype='float64')
    count=X['parking_count'].to_numpy(dtype='float64')
    valid_count=np.isfinite(count)&(count>0)&(count==np.floor(count))
    valid=valid_count&np.isfinite(area)&(area>0)
    ratio=np.full(len(X),np.nan);ratio[valid]=area[valid]/count[valid]
    assert np.isfinite(ratio[valid]).all()
    r=ratio[valid];qs=[0,.01,.25,.5,.75,.99,1]
    return {'rows':len(X),'available_rows':int(valid.sum()),'available_fraction':float(valid.mean()),
      'positive_area_without_valid_positive_integer_count':int((np.isfinite(area)&(area>0)&~valid_count).sum()),
      'positive_count_zero_area_rows':int((valid_count&(area==0)).sum()),
      'negative_area_rows':int((area<0).sum()),
      'noninteger_finite_count_rows':int((np.isfinite(count)&(count!=np.floor(count))).sum()),
      'available_one_slot_rows':int((valid&(count==1)).sum()),
      'available_multiple_slot_rows':int((valid&(count>=2)).sum()),
      'available_ratio_different_from_raw_area_rows':int((valid&(ratio!=area)).sum()),
      'available_area_over_60_one_slot_rows':int((valid&(area>60)&(count==1)).sum()),
      'available_area_over_60_multiple_slots_rows':int((valid&(area>60)&(count>=2)).sum()),
      'ratio_quantiles':{str(q):float(v) for q,v in zip(qs,np.quantile(r,qs))} if len(r) else {},
      'maximum_positive_count':float(count[valid_count].max()) if valid_count.any() else None}
def main():
    assert not (OUT/'profile.json').exists()
    before={k:sha(ROOT/k) for k in EXPECTED};assert before==EXPECTED
    train=pd.read_pickle(ROOT/'data_cache/train_features.pkl');test=pd.read_pickle(ROOT/'data_cache/test_features.pkl')
    for X in (train,test):
        assert X.shape[1]==87 and not {'總價元','編號','target','y'}.intersection(X.columns)
    assert len(train)==455784 and len(test)==47073
    K=np.load(ROOT/'outputs/round6/geography/development/train_source_cache_indices.npy',allow_pickle=False)
    fi=np.load(ROOT/'outputs/round6/models/development/geography_catboost/fit_indices.npy',allow_pickle=False)
    qi=np.load(ROOT/'outputs/round6/models/development/geography_catboost/query_indices.npy',allow_pickle=False)
    for a,n,limit in ((K,455255,len(train)),(fi,240316,len(K)),(qi,81234,len(K))):
        assert a.dtype==np.dtype('int64') and a.shape==(n,) and len(np.unique(a))==n and a.min()>=0 and a.max()<limit
    assert not np.intersect1d(fi,qi).size
    original=train[['parking_area','parking_count']];canonical=original.iloc[K]
    frames={'canonical_train':canonical,'development_fit':canonical.iloc[fi],
            'development_query':canonical.iloc[qi],'test':test[['parking_area','parking_count']]}
    result={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'scope':'target-free feature caches only; canonical and development indices reused, not recomputed; no raw tables/valid/target/model/prediction reads',
      'rule':'reported parking_area / parking_count only when area finite positive and count finite positive integer; otherwise missing; no caps/fit/query learning',
      'semantics':'reported transferred area per parsed slot count, not verified net slot size or market value; existing count parser is not changed',
      'prior_error_summary_consulted':'reports/round18_error_diagnostic.md; previously reused development, not blind; no new price calculation',
      'existing_features':'parking_area, parking_count, parking_fraction, has_parking already exist; only multi-slot rows make this ratio differ from parking_area',
      'formal_model_candidate_created':False,'source_sha256':{str(Path(__file__).relative_to(ROOT)).replace(chr(92),'/'):sha(Path(__file__))},
      'input_sha256':before,'populations':{k:profile(v) for k,v in frames.items()}}
    assert before=={k:sha(ROOT/k) for k in EXPECTED}
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'profile.json'
    with p.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
    print(json.dumps({'profile_sha256':sha(p),'populations':result['populations']},ensure_ascii=False))
if __name__=='__main__':main()
