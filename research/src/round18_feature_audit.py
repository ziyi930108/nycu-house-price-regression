import sys, json, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path('src').resolve()))
import house_features as h
paths=[Path('data_cache/train_features.pkl'),Path('data_cache/test_features.pkl')]
def sha(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):d.update(b)
    return d.hexdigest()
before={str(p):sha(p) for p in paths}
C=r'[0-9零〇○一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰]+'
patterns={
'basement_shared_list':rf'^(?:地下|B)\s*{C}(?:層|樓)?(?:[、,，]\s*{C}(?:層|樓)?)+(?:層|樓)?$',
'basement_shared_range':rf'^(?:地下|B)\s*{C}(?:層|樓)?\s*(?:至|到|~|～|-)\s*{C}(?:層|樓)?$',
'unit_before_hyphen':rf'{C}(?:層|樓)\s*[-－]\s*{C}',
'roof_context':r'屋頂|突出|頂樓',
'roof_only_explicit_number':rf'^(?:屋頂突出物|屋頂突出|屋頂|頂樓)\s*{C}(?:層|樓)?$',
'rooftop_plus_ordinary':r'(?=.*(?:屋頂|突出|頂樓))(?=.*[、,，]).+',
'odd_glyph':r'ㄧ|[０-９]',
'floor_nonmissing_numeric_missing':None,
}
result={'scope':'target-free feature caches only; before 529 dedup; no valid/raw/targets/models/predictions','inputs':before,'synthetic':{s:h._floor_tuple(s) for s in ['地下一、二層','地下1至3層','地下一層至地下三層','二十層、屋頂突出物一層','1層-2層','A棟12樓','一層、地下一層']},'splits':{}}
for p in paths:
    X=pd.read_pickle(p)
    assert X.shape[1]==87 and len(h.get_categorical_columns(X))==21
    assert not set(['總價元','編號','y','target']).intersection(X.columns)
    s=X['transfer_floor_raw'].astype(str)
    stats={'rows':len(X),'pattern_counts':{},'floor_examples':{},'quality':{}}
    for name,pat in patterns.items():
        m=((s!=h.MISSING)&X.floor_mean.isna()) if pat is None else s.str.contains(pat,regex=True,na=False)
        stats['pattern_counts'][name]=int(m.sum())
        stats['floor_examples'][name]=dict(s[m].value_counts().head(20))
    q=stats['quality']
    masks={
     'floor_count_gt1':X.floor_count.gt(1),
     'unit_floor_missing':X.unit_floor.isna(),
     'relative_floor_over1':X.relative_floor.gt(1),
     'missing_parking_type_but_has_parking':X.parking_type.eq(h.MISSING)&X.has_parking.eq(1),
     'parking_present_zero_area':X.has_parking.eq(1)&X.parking_area.eq(0),
     'parking_present_zero_count':X.has_parking.eq(1)&X.parking_count.eq(0),
     'building_area_positive_count_zero':X.building_area.gt(0)&X.building_count.eq(0),
     'building_area_positive_indoor_nonpositive':X.building_area.gt(0)&X.indoor_area.le(0),
     'rooms_zero':X.rooms.eq(0),
     'rooms_zero_living_positive':X.rooms.eq(0)&X.living_rooms.gt(0),
     'date_invalid':X.trade_date_invalid.eq(1),
     'address_district_ne_supplied':X.address_district.ne(X.district),
    }
    for name,m in masks.items():
        q[name]={'rows':int(m.sum()),'transaction_types':{str(k):int(v) for k,v in X.loc[m,'transaction_type'].value_counts().items()}}
    result['splits'][p.stem]=stats
assert before=={str(p):sha(p) for p in paths}
print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
