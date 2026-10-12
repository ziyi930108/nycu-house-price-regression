"""Match parsed input rows to exact geographic road keys, without target labels."""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--parsed',nargs='+',required=True,help='split=path pairs');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();index=a.index.resolve();assert not out.exists(),'Refuse overwrite'
    rows=[]
    with (index/'road_index.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);rows.append({k:v for k,v in r.items() if k not in ('geometry','source_way_ids','bbox','coordinate_interpretation','highway_types')})
    roads=pd.DataFrame(rows);assert roads['key'].is_unique
    cols=['latitude','longitude','quality','road_length_m','extent_diagonal_m','connected_components_with_50m_gap_tolerance','components_with_1mm_gap_tolerance','nearby_fragment_grouping','source_way_count']
    lookup=roads.set_index('key')[cols]
    assert np.isfinite(roads.loc[roads.quality.eq('road_level'),['latitude','longitude']].to_numpy()).all()
    files={};seen=set()
    for spec in a.parsed:
        split,name=spec.split('=',1);assert split in ('train','valid','test') and split not in seen;seen.add(split)
        p=Path(name).resolve();assert p.is_file();files[split]=p
    out.mkdir(parents=True)
    audit={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),'geographic_snapshot_year':2022,'purpose':'Pure X coverage diagnostic; these 2022 matches are not final features for all forecast windows','target_values_read':False,'model_fitting_or_scoring_performed':False,'transductive_fit_rules_learned':False,'exact_city_district_road_key_only':True,'multiple_road_and_land_parcel_coordinates_withheld':True,'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [index/'road_index.jsonl',index/'audit.json',*files.values()]},'splits':{}}
    required=['source_id','source_row_index','source_excel_row','is_embedded_header','city','district_full','road1_name','road_count','road_query_eligible','land_parcel_context']
    for split,path in files.items():
        parsed=pd.read_pickle(path);assert set(required)<=set(parsed.columns)
        assert parsed['source_row_index'].is_unique
        frame=parsed[required].copy()
        key=frame[['city','district_full','road1_name']].fillna('').astype(str).agg('|'.join,axis=1)
        eligible=(frame['road_count'].eq(1)&frame['road_query_eligible'].fillna(False).astype(bool)&~frame['land_parcel_context'].fillna(False).astype(bool)&~frame['is_embedded_header'].fillna(False).astype(bool))
        frame['geo_key']=key
        for col in cols:frame['geo_'+col]=key.map(lookup[col]).where(eligible)
        frame['geo_match_status']='ineligible_address'
        frame.loc[frame['is_embedded_header'].astype(bool),'geo_match_status']='embedded_header'
        frame.loc[~frame['is_embedded_header'].astype(bool)&frame['land_parcel_context'].astype(bool),'geo_match_status']='land_parcel'
        frame.loc[~frame['is_embedded_header'].astype(bool)&~frame['land_parcel_context'].astype(bool)&frame['road_count'].gt(1),'geo_match_status']='multiple_roads'
        frame.loc[eligible,'geo_match_status']='road_key_not_found'
        found=eligible & frame['geo_quality'].notna()
        frame.loc[found,'geo_match_status']=frame.loc[found,'geo_quality'].astype(str)
        usable=frame.geo_match_status.eq('road_level')
        assert frame.loc[~usable,['geo_latitude','geo_longitude']].isna().all().all()
        assert np.isfinite(frame.loc[usable,['geo_latitude','geo_longitude']].to_numpy(dtype=float)).all()
        assert frame.loc[usable,'geo_latitude'].between(20,28).all() and frame.loc[usable,'geo_longitude'].between(116,125).all()
        filename=out/f'{split}_matched.pkl';frame.to_pickle(filename)
        normal=~frame.is_embedded_header.astype(bool)
        bycity=[]
        for city,grp in frame.loc[normal].groupby('city',dropna=False):
            n=len(grp);m=int(grp.geo_match_status.eq('road_level').sum())
            bycity.append({'city':None if pd.isna(city) else str(city),'rows':n,'usable_road_coordinates':m,'fraction':m/n,'status_counts':{str(k):int(v) for k,v in grp.geo_match_status.value_counts().items()}})
        audit['splits'][split]={'rows_including_headers':len(frame),'non_header_rows':int(normal.sum()),'single_road_eligible_rows':int(eligible.sum()),'road_key_found_rows':int(found.sum()),'usable_road_coordinates':int(usable.sum()),'usable_fraction_of_non_header_rows':float(usable.sum()/normal.sum()),'status_counts':{str(k):int(v) for k,v in frame.geo_match_status.value_counts().items()},'by_city':bycity,'output_path':filename.relative_to(ROOT).as_posix(),'output_sha256':sha(filename)}
        print(json.dumps({'split':split,**{k:v for k,v in audit['splits'][split].items() if k not in ('by_city','output_sha256')}},ensure_ascii=False),flush=True)
    for path,expected in audit['input_sha256'].items():assert sha(ROOT/path)==expected
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
