"""Build a historical road-by-administration index; geographic inputs only."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re,unicodedata
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import shapely
from shapely.geometry import shape,mapping
from shapely.ops import transform,unary_union,nearest_points
from shapely.strtree import STRtree
from pyproj import Transformer
from geo_address_audit import normalize_road_name

ROOT=Path(__file__).resolve().parents[1]
CITIES=set('臺北市 新北市 基隆市 桃園市 新竹市 新竹縣 苗栗縣 臺中市 彰化縣 南投縣 雲林縣 嘉義市 嘉義縣 臺南市 高雄市 屏東縣 宜蘭縣 花蓮縣 臺東縣 澎湖縣 金門縣 連江縣'.split())
NAMES=('name','name:zh','name:zh-Hant','alt_name','short_name')
PROJECT=Transformer.from_crs(4326,3826,always_xy=True).transform
UNPROJECT=Transformer.from_crs(3826,4326,always_xy=True).transform

def norm(s):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s or ''))).replace('台','臺')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def names(properties,road=False):
    values=set()
    for key in NAMES:
        for value in re.split(r'[;；]',properties.get(key) or ''):
            value=normalize_road_name(value) if road else norm(value)
            if value:values.add(value)
    return sorted(values)

def read_features(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def linear(g):
    if g.geom_type in ('LineString','MultiLineString'):return g
    parts=[]
    if hasattr(g,'geoms'):
        for child in g.geoms:
            if child.geom_type in ('LineString','MultiLineString'):parts.append(child)
    return unary_union(parts)

def valid_polygon(g):
    if not g.is_valid:g=shapely.make_valid(g)
    if g.geom_type in ('Polygon','MultiPolygon'):return g
    return unary_union([c for c in getattr(g,'geoms',[]) if c.geom_type in ('Polygon','MultiPolygon')])

def boundaries(path):
    records=[];city_parts=collections.defaultdict(list);city_sources=collections.defaultdict(list)
    for f in read_features(path):
        g=valid_polygon(shape(f['geometry']))
        if g.is_empty:continue
        aliases=names(f['properties'])
        rec={'aliases':aliases,'geometry':g,'source_id':f['id'],'level':f['properties'].get('admin_level')}
        records.append(rec)
        for name in set(aliases)&CITIES:
            city_parts[name].append(g);city_sources[name].append(f['id'])
    cities={k:unary_union(v) for k,v in city_parts.items()}
    parts=collections.defaultdict(list);sources=collections.defaultdict(list);levels=collections.defaultdict(set)
    rejected=[]
    for r in records:
        ds=[n for n in r['aliases'] if n not in CITIES and re.fullmatch(r'[\u3400-\u9fff]{1,10}[區鄉鎮市]',n)]
        if not ds:continue
        g=r['geometry'];candidates=[]
        for city,cg in cities.items():
            if not cg.intersects(g):continue
            fraction=cg.intersection(g).area/g.area if g.area else 0
            if fraction>=.98:candidates.append(city)
        if len(candidates)!=1:
            rejected.append({'source_id':r['source_id'],'names':ds,'candidate_cities':candidates});continue
        city=candidates[0]
        clipped=g.intersection(cities[city])
        for district in ds:
            key=(city,district);parts[key].append(clipped);sources[key].append(r['source_id']);levels[key].add(str(r['level']))
    districts=[]
    for key,gs in sorted(parts.items()):
        geometry=unary_union(gs)
        districts.append({'city':key[0],'district':key[1],'geometry':geometry,'sources':sources[key],'admin_levels':sorted(levels[key])})
    audit={'cities_found':sorted(cities),'cities_missing':sorted(CITIES-set(cities)),'city_source_ids':dict(city_sources),'districts':len(districts),'districts_by_city':dict(collections.Counter(d['city'] for d in districts)),'rejected_district_assignments':rejected,'method':'Exact named city boundary, district-name suffix, at least 98% polygon overlap in exactly one city; no presumed admin_level'}
    return districts,audit

def road_summary(key,items):
    g=unary_union([item[0] for item in items]);m=transform(PROJECT,g)
    buffered=m.buffer(25)
    ncomp=len(buffered.geoms) if buffered.geom_type=='MultiPolygon' else 1
    fine=m.buffer(0.0005)
    fine_components=len(fine.geoms) if fine.geom_type=='MultiPolygon' else 1
    minx,miny,maxx,maxy=m.bounds
    diagonal=math.hypot(maxx-minx,maxy-miny)
    point=nearest_points(m,m.centroid)[0]
    ll=transform(UNPROJECT,point)
    # These are geometry quality policies fixed before any price modelling.
    quality='disconnected' if ncomp>1 else 'broad_road' if diagonal>10000 else 'road_level'
    usable=quality=='road_level'
    return {'city':key[0],'district':key[1],'road_name':key[2],'key':'|'.join(key),'latitude':ll.y if usable else None,'longitude':ll.x if usable else None,'quality':quality,'road_length_m':m.length,'extent_diagonal_m':diagonal,'connected_components_with_50m_gap_tolerance':ncomp,'components_with_1mm_gap_tolerance':fine_components,'nearby_fragment_grouping':fine_components>ncomp,'source_way_count':len({i[1] for i in items}),'source_way_ids':sorted({i[1] for i in items}),'highway_types':sorted({i[2] for i in items}),'bbox':list(g.bounds),'geometry':mapping(g),'coordinate_interpretation':'Point on the matched road closest to its length-weighted centroid; not a house location'}

def build(source,out):
    assert not out.exists(),'Output exists; use a new directory for revised geographic policy'
    roads=source/'roads.jsonl';areas=source/'administrative_areas.jsonl';audit_path=source/'audit.json'
    assert all(p.is_file() for p in (roads,areas,audit_path))
    original_hashes={str(p.relative_to(ROOT)):digest(p) for p in (roads,areas,audit_path)}
    districts,admin_audit=boundaries(areas)
    assert districts,'No reliable districts found'
    tree=STRtree([d['geometry'] for d in districts])
    groups=collections.defaultdict(list);counts=collections.Counter()
    for road in read_features(roads):
        counts['source_road_ways']+=1
        aliases=names(road['properties'],True)
        g=shape(road['geometry'])
        if not aliases or g.is_empty:continue
        matched=False
        for i in tree.query(g,predicate='intersects'):
            d=districts[int(i)];clipped=linear(g.intersection(d['geometry']))
            if clipped.is_empty or clipped.length==0:continue
            matched=True
            for name in aliases:
                groups[(d['city'],d['district'],name)].append((clipped,road['osm_id'],road['properties']['highway']))
            counts['road_district_fragments']+=1
        if matched:counts['ways_assigned_to_district']+=1
        if counts['source_road_ways']%50000==0:print(dict(counts),flush=True)
    out.mkdir(parents=True)
    quality=collections.Counter();bycity=collections.Counter()
    with (out/'road_index.jsonl').open('w',encoding='utf-8') as f:
        for key,items in sorted(groups.items()):
            row=road_summary(key,items);quality[row['quality']]+=1;bycity[row['city']]+=1
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    admin_rows=[{k:v for k,v in d.items() if k!='geometry'} for d in districts]
    (out/'district_inventory.json').write_text(json.dumps(admin_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    payload={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':original_hashes,'source_sha256':{str(Path(__file__).relative_to(ROOT)):digest(__file__),'src/geo_address_audit.py':digest(ROOT/'src/geo_address_audit.py')},'versions':{p:version(p) for p in ('shapely','pyproj','numpy','pandas')},'administration':admin_audit,'counts':dict(counts),'road_keys':len(groups),'quality_counts':dict(quality),'road_keys_by_city':dict(bycity),'projection':'EPSG:3826 approximate metric geometry; stored output coordinates EPSG:4326','fixed_geographic_rules':{'district_city_overlap_minimum':.98,'component_buffer_metres':25,'maximum_usable_diagonal_metres':10000,'exact_alias_only':True,'all_disconnected_after_50m_gap_tolerance_withhold_coordinates':True,'nearby_fragments_can_share_a_road_representative':True,'physical_road_connectivity_is_not_proven':True},'targets_read':False,'house_coordinates_claimed':False,'price_models_trained':False,'license':'ODbL-1.0; OpenStreetMap contributors; Geofabrik extract','output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    for p,sha in original_hashes.items():assert digest(ROOT/p)==sha
    (out/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','road_keys':len(groups),'quality':dict(quality),'districts':len(districts)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    build(a.source.resolve(),a.output.resolve())
