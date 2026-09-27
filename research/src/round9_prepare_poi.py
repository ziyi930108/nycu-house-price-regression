"""Pure geographic R9 POI features; reads only frozen eight-column R6 geo frames.
Run with .venv-geo Python. No model, target, prediction or raw ML workbook imports.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, platform, time, uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round9'
STAGE_YEAR={'development':2022,'confirmation':2023,'auxiliary':2021,'final':2024}
POI_KEYS=('railway_station','subway_entrance','bus_station','school','kindergarten','higher_education','hospital','clinic','pharmacy','supermarket','convenience','park','motorway_junction')
POI_FEATURES=tuple('poi_log_distance_'+key for key in POI_KEYS)+('poi_position_usable',)
GEO_FEATURES=('geo_latitude','geo_longitude','geo_log_road_length','geo_log_extent','geo_has_position','geo_nearby_fragments','geo_match_status','geo_grid_002')
PROJECTED_CRS='+proj=aeqd +lat_0=23.5 +lon_0=121 +datum=WGS84 +units=m +no_defs'
MAX_EXTENT_METRES=2000.0
RULES={'poi_keys':list(POI_KEYS),'columns':list(POI_FEATURES),'dtypes':'float32','input_crs':'EPSG:4326','projected_crs':PROJECTED_CRS,'distance':'minimum projected point-to-original-valid-geometry distance; polygon interior zero; holes retained','transform':'log1p(distance_metres); no cap, radii, density or learned scaling','query_position':'existing R6 road representative/intersection point; not exact house location','usable':'geo_has_position==1, finite coordinates and log_extent, supported status, stored float32 log_extent <= float32(log1p(2000m))','missing':'all 13 distances NaN if position unusable; individual class NaN if no valid geometry exists in that annual extract','extent_metres_limit':MAX_EXTENT_METRES,'names_or_ids_as_model_features':False,'geodesic_or_travel_time_claim':False,'target_or_prediction_values_deserialized':False}

def require(ok,message):
    if not ok: raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def locked(path):
    path=Path(path)
    require(path.is_file() and path.with_suffix('.sha256').is_file(),'Missing locked manifest: '+str(path))
    require(sha(path)==path.with_suffix('.sha256').read_text(encoding='ascii').strip(),'Manifest hash differs: '+str(path))
    return read(path)
def write_locked(path,value):
    require(not Path(path).exists(),'Refusing manifest overwrite')
    write(path,value);Path(path).with_suffix('.sha256').write_text(sha(path)+'\n',encoding='ascii')
def frame_sha(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
def source_hashes():
    pending=[Path(__file__).resolve()];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/'src'/(name.split('.')[0]+'.py')
                if local.is_file() and local not in found:pending.append(local)
    return {rel(path):sha(path) for path in sorted(found)}
def checked_path(name):
    path=Path(name)
    if not path.is_absolute():path=ROOT/path
    path=path.resolve();require(path.is_relative_to(ROOT),'Dependency outside workspace');return path

def check_geo(geo):
    require(tuple(geo.columns)==GEO_FEATURES,'Only original eight geographic columns may be loaded')
    for col in GEO_FEATURES[:6]:require(str(geo[col].dtype)=='float32','Geographic numeric dtype changed')
    require(geo.index.is_unique,'Geographic row index must be unique')
    has=geo.geo_has_position.to_numpy()
    require(np.isin(has,[0,1]).all(),'Invalid geo position flag')
    for col in ('geo_latitude','geo_longitude','geo_log_extent'):
        a=geo[col].to_numpy(dtype='float64')
        require(not np.isinf(a).any(),'Infinite geographic value')
        require(np.isfinite(a[has==1]).all(),'Missing positioned geography')
    require(geo.loc[has==1,'geo_match_status'].isin(['road_level','planar_intersection']).all(),'Unsupported positioned geography')

def distance_features(geo,trees,transformer):
    check_geo(geo)
    require(set(trees)==set(POI_KEYS),'Unexpected category set')
    lat=geo.geo_latitude.to_numpy(dtype='float64');lon=geo.geo_longitude.to_numpy(dtype='float64')
    extent=geo.geo_log_extent.to_numpy(dtype='float64')
    usable=(geo.geo_has_position.to_numpy()==1)&np.isfinite(lat)&np.isfinite(lon)&np.isfinite(extent)&(extent>=0)&(extent<=float(np.float32(np.log1p(MAX_EXTENT_METRES))))
    require(((lat[usable]>=-90)&(lat[usable]<=90)&(lon[usable]>=-180)&(lon[usable]<=180)).all(),'WGS84 range invalid')
    output=pd.DataFrame(np.full((len(geo),len(POI_FEATURES)),np.nan,dtype='float32'),index=geo.index,columns=POI_FEATURES)
    output['poi_position_usable']=usable.astype('float32')
    rows=np.flatnonzero(usable)
    coords,inverse=np.unique(np.column_stack([lon[usable],lat[usable]]),axis=0,return_inverse=True)
    if len(coords):
        x,y=transformer.transform(coords[:,0],coords[:,1])
        require(np.isfinite(x).all() and np.isfinite(y).all(),'Projection returned nonfinite query')
        points=shapely.points(x,y)
        for key in POI_KEYS:
            tree=trees[key]
            if tree is None:continue
            nearest=tree.nearest(points)
            d=shapely.distance(points,tree.geometries[nearest])
            require(np.isfinite(d).all() and (d>=0).all(),'Invalid nearest distance')
            output.iloc[rows,output.columns.get_loc('poi_log_distance_'+key)]=np.log1p(d[inverse]).astype('float32')
    require(output.loc[~usable,list(POI_FEATURES[:-1])].isna().all().all(),'Unusable position received a distance')
    require((output.dtypes.astype(str)=='float32').all(),'POI dtype differs')
    audit={'rows':len(geo),'geo_position_rows':int((geo.geo_has_position==1).sum()),'poi_position_usable_rows':int(usable.sum()),'positioned_but_extent_rejected_rows':int(((geo.geo_has_position==1)&~usable).sum()),'distinct_usable_coordinate_pairs':len(coords),'classes':{}}
    for key in POI_KEYS:
        arr=output['poi_log_distance_'+key].to_numpy(dtype='float64');finite=np.isfinite(arr)
        metres=np.expm1(arr[finite])
        audit['classes'][key]={'finite_rows':int(finite.sum()),'missing_rows':int((~finite).sum()),'zero_distance_rows':int((arr[finite]==0).sum()),'metres_quantiles_from_stored_float32':None if not len(metres) else {str(q):float(np.quantile(metres,q)) for q in [0,.25,.5,.75,.9,.99,1]}}
    return output,audit

def load_index(stage):
    year=STAGE_YEAR[stage];folder=OUT/'poi_index'/str(year);audit=locked(folder/'audit.json')
    require(audit['status']=='complete','Incomplete POI extract')
    pbf=checked_path(audit['source']['path'])
    require(pbf.is_file() and sha(pbf)==audit['source']['sha256'] and pbf.stat().st_size==audit['source']['bytes'],'PBF changed')
    dependencies={rel(pbf):sha(pbf),rel(folder/'audit.json'):sha(folder/'audit.json'),rel(folder/'audit.sha256'):sha(folder/'audit.sha256')}
    for name,digest in audit['source_sha256'].items():
        path=checked_path(name);require(sha(path)==digest,'Extractor source changed');dependencies[rel(path)]=digest
    path=folder/'pois.jsonl'
    require(sha(path)==audit['files']['pois']['sha256'],'POI index hash changed')
    require(path.stat().st_size==audit['files']['pois']['bytes'],'POI bytes changed')
    dependencies[rel(path)]=sha(path)
    transformer=Transformer.from_crs('EPSG:4326',CRS.from_user_input(PROJECTED_CRS),always_xy=True)
    geoms={key:[] for key in POI_KEYS};identities=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line);key=item['category']
            require(key in geoms and item['osm_type'] in ('node','way','relation'),'Unexpected POI identity')
            identity=(key,item['osm_type'],item['osm_id']);require(identity not in identities,'Duplicate category/OSM identity');identities.add(identity)
            source=shapely.from_geojson(json.dumps(item['geometry']))
            require(source.geom_type in ('Point','Polygon','MultiPolygon') and source.is_valid and not source.is_empty,'Invalid original POI geometry')
            require(np.isfinite(source.bounds).all(),'Nonfinite source bounds')
            require(-180<=source.bounds[0]<=source.bounds[2]<=180 and -90<=source.bounds[1]<=source.bounds[3]<=90,'Invalid WGS84 geometry')
            projected=transform_geometry(transformer.transform,source)
            require(projected.is_valid and not projected.is_empty and np.isfinite(projected.bounds).all(),'Invalid projected geometry; no repair allowed')
            geoms[key].append(projected)
    trees={key:STRtree(values) if values else None for key,values in geoms.items()}
    return trees,transformer,audit,dependencies,{key:len(values) for key,values in geoms.items()}

def prepare(stage):
    initial_sources=source_hashes()
    destination=OUT/'poi_features'/stage
    require(not destination.exists(),'Prepared POI features already exist; refusing overwrite')
    start=time.monotonic();trees,transformer,index_audit,dependencies,counts=load_index(stage)
    old_path=ROOT/'outputs/round6/geography'/stage/'manifest.json';old=locked(old_path)
    require(old['snapshot_year']==STAGE_YEAR[stage],'R6 year differs')
    require(old['source_pbf_sha256']==index_audit['source']['sha256'],'POI and query location maps differ')
    stamp=index_audit['source']['header']['replication_timestamp']
    require(old['replication_timestamp']==stamp,'Actual map header differs')
    require(datetime.fromisoformat(stamp.replace('Z','+00:00'))<datetime.fromisoformat(old['forecast_origin_utc'].replace('Z','+00:00')),'Map is not before forecast origin')
    dependencies.update({rel(old_path):sha(old_path),rel(old_path.with_suffix('.sha256')):sha(old_path.with_suffix('.sha256'))})
    for name,digest in old['source_sha256'].items():
        path=checked_path(name);require(sha(path)==digest,'Frozen R6 source changed');dependencies[rel(path)]=digest
    for name,digest in old['index_files_sha256'].items():
        path=checked_path(name);require(sha(path)==digest,'Frozen R6 alignment index changed');dependencies[rel(path)]=digest
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    features={};coverage={}
    for name in ('train','valid','test','full'):
        source=checked_path(old['features'][name]['path']);require(sha(source)==old['features'][name]['sha256'],'R6 geographic frame changed')
        dependencies[rel(source)]=sha(source)
        geo=pd.read_pickle(source);check_geo(geo)
        frame,audit=distance_features(geo,trees,transformer)
        require(frame.index.equals(geo.index),'POI index differs from R6')
        target=staging/(name+'.pkl');frame.to_pickle(target,protocol=5)
        features[name]={'path':rel(destination/target.name),'sha256':sha(target),'rows':len(frame),'frame_sha256':frame_sha(frame),'geographic_frame_sha256':frame_sha(geo),'index_values_sha256':hashlib.sha256(np.asarray(frame.index,dtype='<i8').tobytes()).hexdigest()}
        coverage[name]=audit
        print(json.dumps({'event':'poi_partition_prepared','stage':stage,'partition':name,'rows':len(frame),'usable_rows':audit['poi_position_usable_rows']}),flush=True)
    write(staging/'coverage.json',coverage)
    dependencies.update(initial_sources)
    manifest={'round':9,'stage':stage,'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),'rules':RULES,'source_sha256':initial_sources,'dependency_sha256':dependencies,'geography_manifest_sha256':sha(old_path),'poi_index_audit_sha256':sha(OUT/'poi_index'/str(STAGE_YEAR[stage])/'audit.json'),'snapshot_year':STAGE_YEAR[stage],'source_pbf_sha256':index_audit['source']['sha256'],'replication_timestamp':stamp,'forecast_origin_utc':old['forecast_origin_utc'],'features':features,'coverage':{'path':rel(destination/'coverage.json'),'sha256':sha(staging/'coverage.json')},'geometry_records_by_class':counts,'geographic_values_learned_from_targets':False,'raw_workbooks_or_target_cache_deserialized':False,'prediction_files_read':False,'historical_qualification':'Map available before forecast origin; not proof of facility opening date or exact geometry on every older fit transaction date. Missing map entry is not absence. All distances use coarse query positions.','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'shapely':shapely.__version__,'pyproj':version('pyproj'),'proj':__import__('pyproj').proj_version_str},'elapsed_seconds':time.monotonic()-start}
    for name,digest in dependencies.items():require(sha(ROOT/name)==digest,'Dependency changed during preparation')
    require(source_hashes()==initial_sources,'Preparation source changed during execution')
    write_locked(staging/'manifest.json',manifest)
    require(not destination.exists(),'Destination appeared during preparation');staging.rename(destination)
    print(json.dumps({'event':'poi_features_complete','stage':stage,'manifest_sha256':sha(destination/'manifest.json'),'elapsed_seconds':time.monotonic()-start}),flush=True)
    return manifest

def self_check():
    transformer=Transformer.from_crs('EPSG:4326',CRS.from_user_input(PROJECTED_CRS),always_xy=True)
    def fixture(coords,extent=None):
        n=len(coords);data={k:np.zeros(n,dtype='float32') for k in GEO_FEATURES[:6]}
        data['geo_latitude']=np.array([p[1] for p in coords],dtype='float32');data['geo_longitude']=np.array([p[0] for p in coords],dtype='float32');data['geo_has_position']=np.ones(n,dtype='float32')
        data['geo_log_extent']=np.array(extent if extent is not None else np.zeros(n),dtype='float32');data['geo_match_status']=['road_level']*n;data['geo_grid_002']=['x']*n
        return pd.DataFrame(data,index=np.arange(10,10+n))
    base=fixture([(121,23.5),(121,23.5),(121.001,23.5)])
    x,y=transformer.transform(121,23.5)
    square=Polygon([(x-100,y-100),(x+100,y-100),(x+100,y+100),(x-100,y+100)],holes=[[(x-10,y-10),(x+10,y-10),(x+10,y+10),(x-10,y+10)]])
    trees={k:STRtree([Point(x,y)]) for k in POI_KEYS};trees['park']=STRtree([square]);trees['pharmacy']=None
    checks={}
    out,audit=distance_features(base,trees,transformer)
    checks['preserve_index_and_fourteen_float32']=out.index.equals(base.index) and len(out.columns)==14 and (out.dtypes.astype(str)=='float32').all()
    checks['duplicate_query_coordinate_same']=np.allclose(out.iloc[0],out.iloc[1],equal_nan=True)
    checks['point_at_origin_zero']=out.iloc[0]['poi_log_distance_school']==0
    checks['polygon_hole_not_filled']=np.isclose(out.iloc[0]['poi_log_distance_park'],np.log1p(10),rtol=1e-6)
    checks['missing_class_remains_nan']=out.poi_log_distance_pharmacy.isna().all()
    checks['all_distances_nonnegative']=all((out[col].dropna()>=0).all() for col in POI_FEATURES[:-1])
    off=base.copy();off.loc[10,'geo_has_position']=0
    off.loc[10,['geo_latitude','geo_longitude','geo_log_extent']]=np.nan;off.loc[10,'geo_match_status']='missing'
    result,_=distance_features(off,trees,transformer)
    checks['missing_query_all_nan']=result.loc[10,'poi_position_usable']==0 and result.loc[10,list(POI_FEATURES[:-1])].isna().all()
    boundary=fixture([(121,23.5)]*3,[np.log1p(2000),np.log1p(2001),0])
    result,_=distance_features(boundary,trees,transformer)
    checks['fixed_extent_boundary']=result.poi_position_usable.tolist()==[1,0,1] and result.loc[11,list(POI_FEATURES[:-1])].isna().all()
    empty=base.iloc[:0];result,_=distance_features(empty,trees,transformer);checks['empty_query_supported']=result.shape==(0,14)
    duplicate={k:(STRtree([Point(x,y),Point(x,y)]) if k!='pharmacy' else None) for k in POI_KEYS}
    result,_=distance_features(base,duplicate,transformer);checks['duplicate_geometry_does_not_inflate_distance']=np.array_equal(result.poi_log_distance_school,out.poi_log_distance_school)
    for label,bad in [('extra_price_column',base.assign(fake_price=10)),('bad_flag',base.assign(geo_has_position=np.full(3,2,dtype='float32'))),('bad_dtype',base.assign(geo_latitude=base.geo_latitude.astype('float64')))]:
        try:distance_features(bad,trees,transformer);checks[label+'_rejected']=False
        except RuntimeError:checks[label+'_rejected']=True
    inside=Polygon([(x-100,y-100),(x+100,y-100),(x+100,y+100),(x-100,y+100)])
    t=dict(trees);t['park']=STRtree([inside]);result,_=distance_features(base,t,transformer);checks['polygon_interior_zero']=result.iloc[0].poi_log_distance_park==0
    require(all(checks.values()),'POI synthetic failure: '+str(checks))
    return {'all_checks_passed':True,'synthetic_only':True,'checks':{k:bool(v) for k,v in checks.items()},'source_sha256':source_hashes(),'rules':RULES}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['self-check','prepare']);parser.add_argument('--stage',choices=list(STAGE_YEAR),default='development');args=parser.parse_args()
    if args.command=='self-check':
        result=self_check();path=OUT/'audit/poi_preparation_self_check.json';require(not path.exists(),'Self-check receipt exists');write(path,result);print(json.dumps(result),flush=True)
    else:prepare(args.stage)
if __name__=='__main__':main()
