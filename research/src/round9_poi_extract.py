"""Extract the fixed R9 POI catalog from local 2022 OSM, with original geometry.

Only node Points and normal pyosmium closed-way/multipolygon areas are retained.
No prices, housing tables, model predictions, centroid, name merge or repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from importlib.metadata import version
import json,math,platform,sys,tempfile,time,uuid
from pathlib import Path
import round9_poi_inventory as inventory
from geo_osm_extract import sha256,source_header,geometry_failure

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'outputs/round9/poi_index/2022'
CATEGORY_TAGS={
 'railway_station':[('railway','station'),('railway','halt')],
 'subway_entrance':[('railway','subway_entrance')],
 'bus_station':[('amenity','bus_station')],
 'school':[('amenity','school')],'kindergarten':[('amenity','kindergarten')],
 'higher_education':[('amenity','college'),('amenity','university')],
 'hospital':[('amenity','hospital')],'clinic':[('amenity','clinic')],
 'pharmacy':[('amenity','pharmacy')],'supermarket':[('shop','supermarket')],
 'convenience':[('shop','convenience')],'park':[('leisure','park')],
 'motorway_junction':[('highway','motorway_junction')]}
TARGETS=tuple(pair for pairs in CATEGORY_TAGS.values() for pair in pairs)
TYPES=('node','way','relation')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def relative(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def rules():
    return {'categories':list(CATEGORY_TAGS),'category_tag_map':{c:[list(p) for p in pairs] for c,pairs in CATEGORY_TAGS.items()},
      'exact_tag_values_only':True,'inactive_rule_source':'src/round9_poi_inventory.py:inactive_evidence',
      'lifecycle_statuses':list(inventory.LIFECYCLE),'lifecycle_false_values':list(inventory.FALSE_VALUES),'state_tag_keys':list(inventory.STATE_KEYS),
      'inactive_evidence_rule':'nonfalse lifecycle flag, listed state-key lifecycle value, or same-target lifecycle prefix; exclude per category',
      'without_inactive_evidence_is_not_confirmed_operation':True,
      'node_geometry':'valid original WGS84 Point',
      'way_geometry':'closed original tagged way, normal pyosmium assembled valid Polygon/MultiPolygon',
      'relation_geometry':'original type=multipolygon and own POI tags; normal valid assembled Polygon/MultiPolygon',
      'reject_open_ways_and_nonpolygon_relations':True,'linear_platforms_not_selected':True,
      'preserve_holes':True,'centroid_used':False,'geometry_repair_used':False,'convex_hull_used':False,
      'names_used_for_deduplication':False,'categories_can_overlap':True,
      'original_element_qualification_required':True,'untagged_members_not_independent_POIs':True}


def category_evidence(tags):
    found={}
    for category,pairs in CATEGORY_TAGS.items():
        matched=[(k,v) for k,v in pairs if tags.get(k)==v]
        if matched:
            evidence=sorted({reason for k,v in matched for reason in inventory.inactive_evidence(tags,k,v)})
            found[category]=evidence
    return found


def geometry_problem(geometry,allowed):
    reason=geometry_failure(geometry,allowed)
    if reason:return reason
    if geometry.geom_type in ('Polygon','MultiPolygon') and (not math.isfinite(geometry.area) or geometry.area<=0):return 'nonpositive_polygon_area'
    return None


def write_line(handle,value):handle.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')


def extract_core(pbf,pois_path,errors_path,osmium,shapely,pool,progress=True):
    counts=Counter();errors=Counter();relation_types=Counter();inactive=Counter();samples=[]
    per_category={c:{t:Counter() for t in TYPES} for c in CATEGORY_TAGS}
    originals={};expected=set();assembled=set();emitted=set();pending={};seen_area=set()
    factory=osmium.geom.WKBFactory();started=time.perf_counter();last=started
    with Path(pois_path).open('x',encoding='utf-8',newline='\n') as output,Path(errors_path).open('x',encoding='utf-8',newline='\n') as error_output:
        def failure(kind,oid,reason,categories=(),detail=None):
            errors[reason]+=1
            item={'osm_type':kind,'osm_id':int(oid),'reason':reason,'categories':list(categories)}
            if detail:item['detail']=str(detail)[:300]
            write_line(error_output,item)
            if len(samples)<100:samples.append(item)
        def emit(kind,oid,categories,geometry):
            problem=geometry_problem(geometry,{'Point'} if kind=='node' else {'Polygon','MultiPolygon'})
            if problem:
                for c in categories:per_category[c][kind]['geometry_rejected']+=1
                failure(kind,oid,problem,categories);return
            payload=json.loads(shapely.to_geojson(geometry))
            for category in categories:
                identity=(category,kind,int(oid))
                need(identity not in emitted,'Duplicate original element/category output')
                emitted.add(identity)
                write_line(output,{'category':category,'osm_type':kind,'osm_id':int(oid),'geometry':payload})
                per_category[category][kind]['written']+=1;counts['rows_written']+=1
                counts['geometry_'+geometry.geom_type]+=1
            if geometry.geom_type in ('Polygon','MultiPolygon'):
                pieces=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
                holes=sum(len(part.interiors) for part in pieces)
                counts['original_area_geometries_written']+=1
                counts['area_geometries_with_holes']+=holes>0;counts['interior_rings_preserved']+=holes
        def area_result(key,wkb,conversion_error):
            record=originals.get(key)
            if record is None:
                pending[key]=(wkb,conversion_error);return
            kind,oid=key;categories=record['allowed']
            if not categories or not record['geometry_eligible']:
                counts['assembled_area_without_eligible_original']+=1;return
            assembled.add(key)
            for c in categories:per_category[c][kind]['area_observed']+=1
            if conversion_error is not None:
                failure(kind,oid,'area_geometry_exception',categories,conversion_error);return
            try:geometry=shapely.from_wkb(wkb)
            except (ValueError,RuntimeError,shapely.errors.GEOSException) as exc:
                failure(kind,oid,'area_geometry_exception',categories,exc);return
            emit(kind,oid,categories,geometry)
        processor=(osmium.FileProcessor(str(pbf),thread_pool=pool).with_locations('flex_mem')
          .with_areas(osmium.filter.TagFilter(*TARGETS)).with_filter(osmium.filter.TagFilter(*TARGETS)))
        for obj in processor:
            counts['tag_filtered_objects']+=1
            if obj.is_area():
                key=('way' if obj.from_way() else 'relation',int(obj.orig_id()))
                if key in seen_area:
                    failure(*key,'duplicate_assembled_area');continue
                seen_area.add(key);counts['areas_received']+=1
                wkb=None;error=None
                try:wkb=factory.create_multipolygon(obj)
                except (RuntimeError,ValueError,shapely.errors.GEOSException) as exc:error=str(exc)
                area_result(key,wkb,error)
            elif obj.is_node() or obj.is_way() or obj.is_relation():
                kind='node' if obj.is_node() else 'way' if obj.is_way() else 'relation';oid=int(obj.id)
                tags={str(t.k):str(t.v) for t in obj.tags};categories=category_evidence(tags)
                if not categories:continue
                counts['original_'+kind+'_candidates']+=1
                allowed=[]
                for category,reasons in categories.items():
                    record=per_category[category][kind];record['direct_candidates']+=1
                    if reasons:
                        record['inactive_evidence_excluded']+=1
                        inactive.update(category+'|'+kind+'|'+reason for reason in reasons)
                    else:allowed.append(category);record['without_inactive_evidence']+=1
                if kind=='node':
                    if not allowed:continue
                    if not obj.location.valid():
                        failure(kind,oid,'missing_node_location',allowed);continue
                    try:geometry=shapely.Point(float(obj.location.lon),float(obj.location.lat))
                    except (ValueError,RuntimeError,shapely.errors.GEOSException) as exc:
                        failure(kind,oid,'point_geometry_exception',allowed,exc);continue
                    emit(kind,oid,allowed,geometry)
                else:
                    key=(kind,oid);need(key not in originals,'Duplicate original way/relation')
                    geometry_eligible=False
                    if kind=='way':
                        closed=bool(obj.is_closed())
                        if allowed and closed:expected.add(key);geometry_eligible=True
                        if allowed and not closed:
                            for c in allowed:per_category[c][kind]['open_way_skipped']+=1
                            failure(kind,oid,'open_way_not_selected_geometry',allowed)
                        if allowed and closed:
                            missing=[int(n.ref) for n in obj.nodes if not n.location.valid()]
                            if missing:
                                geometry_eligible=False
                                failure(kind,oid,'closed_way_missing_node_locations',allowed,{'count':len(missing),'first_20':missing[:20]})
                    else:
                        relation_type=str(tags.get('type','__MISSING__'));relation_types[relation_type]+=1
                        if allowed and relation_type=='multipolygon':expected.add(key);geometry_eligible=True
                        if allowed and relation_type!='multipolygon':
                            for c in allowed:per_category[c][kind]['nonpolygon_relation_skipped']+=1
                            failure(kind,oid,'nonpolygon_relation_type',allowed,relation_type)
                    originals[key]={'allowed':allowed,'geometry_eligible':geometry_eligible,'original_tags':tags}
                    if key in pending:area_result(key,*pending.pop(key))
            if progress and time.perf_counter()-last>=20:
                print(json.dumps({'event':'poi_extraction_progress','seconds':round(time.perf_counter()-started,1),'counts':dict(counts)}),flush=True);last=time.perf_counter()
        for (kind,oid),(wkb,error) in sorted(pending.items()):
            failure(kind,oid,'area_without_original_tagged_element',detail='Area tags alone do not qualify an untagged member as a POI')
        for kind,oid in sorted(expected-assembled):
            categories=originals[(kind,oid)]['allowed']
            for c in categories:per_category[c][kind]['area_not_assembled']+=1
            failure(kind,oid,'closed_way_or_multipolygon_not_assembled',categories)
    return {'counts':dict(sorted(counts.items())),'category_counts':{c:{t:dict(sorted(v.items())) for t,v in types.items()} for c,types in per_category.items()},
        'error_counts':dict(sorted(errors.items())),'error_samples_first_100':samples,'relation_type_counts':dict(sorted(relation_types.items())),
        'inactive_evidence_counts':dict(sorted(inactive.items())),'expected_area_originals':len(expected),'observed_eligible_area_originals':len(assembled),
        'unassembled_area_originals':len(expected-assembled),'area_without_qualifying_original':len(pending),
        'unique_original_elements_written':len({(kind,oid) for _,kind,oid in emitted}),
        'elapsed_seconds':time.perf_counter()-started}


def self_check():
    import osmium,shapely
    from shapely.geometry import Polygon,Point,LineString
    checks=[]
    def check(label,truth):need(truth,'Synthetic check failed: '+label);checks.append(label)
    check('exactly_13_fixed_categories_no_bus_stop_or_generic_station',len(CATEGORY_TAGS)==13 and not category_evidence({'highway':'bus_stop'}) and not category_evidence({'public_transport':'station'}))
    check('lifecycle_evidence_reused',category_evidence({'amenity':'school','disused':'yes'})=={'school':['flag:disused=yes']} and category_evidence({'railway':'station','abandoned:railway':'station'})['railway_station'])
    check('multiple_categories_on_same_original',set(category_evidence({'amenity':'clinic','shop':'convenience'}))=={'clinic','convenience'})
    check('invalid_polygon_rejected',geometry_problem(Polygon([(120,23),(121,24),(121,23),(120,24),(120,23)]),{'Polygon','MultiPolygon'}) is not None)
    check('nan_and_infinite_coordinates_rejected',all(geometry_problem(Point(x,23),{'Point'}) is not None for x in [float('nan'),float('inf')]))
    check('linear_geometry_rejected_for_selected_catalog',geometry_problem(LineString([(120,23),(121,24)]),{'Point','Polygon','MultiPolygon'}) is not None)
    xml='''<osm version="0.6" generator="r9-synthetic">
<node id="1" lat="23" lon="120"/><node id="2" lat="23" lon="121"/><node id="3" lat="24" lon="121"/><node id="4" lat="24" lon="120"/>
<node id="5" lat="23.2" lon="120.2"/><node id="6" lat="23.2" lon="120.8"/><node id="7" lat="23.8" lon="120.8"/><node id="8" lat="23.8" lon="120.2"/>
<node id="20" lat="23.1" lon="120.1"><tag k="amenity" v="clinic"/><tag k="shop" v="convenience"/></node>
<node id="21" lat="23.1" lon="120.1"><tag k="amenity" v="school"/><tag k="disused" v="yes"/></node>
<node id="22" lat="23.1" lon="120.1"><tag k="public_transport" v="station"/></node>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
<way id="11"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/></way>
<way id="12"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="amenity" v="school"/></way>
<way id="13"><nd ref="1"/><nd ref="2"/><tag k="amenity" v="hospital"/></way>
<relation id="100"><member type="way" ref="10" role="outer"/><member type="way" ref="11" role="inner"/><tag k="type" v="multipolygon"/><tag k="leisure" v="park"/></relation>
<relation id="101"><member type="way" ref="12" role="site"/><tag k="type" v="site"/><tag k="amenity" v="hospital"/></relation>
<relation id="102"><member type="way" ref="99999" role="outer"/><tag k="type" v="multipolygon"/><tag k="amenity" v="school"/></relation>
</osm>'''
    temp_parent=ROOT/'outputs/round9';temp_parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.poi-synthetic-',dir=temp_parent) as temporary:
        folder=Path(temporary).resolve();need(folder.is_relative_to(temp_parent.resolve()),'Unsafe synthetic temporary directory')
        source=folder/'fixture.osm';source.write_text(xml,encoding='utf-8')
        result=extract_core(source,folder/'pois.jsonl',folder/'errors.jsonl',osmium,shapely,osmium.io.ThreadPool(2,4),progress=False)
        rows=[json.loads(line) for line in (folder/'pois.jsonl').read_text(encoding='utf-8').splitlines()]
        identities={(r['category'],r['osm_type'],r['osm_id']) for r in rows}
        check('original_node_multitag_emits_twice_without_merge',('clinic','node',20) in identities and ('convenience','node',20) in identities)
        check('inactive_and_nonselected_nodes_not_emitted',not any(r['osm_type']=='node' and r['osm_id'] in (21,22) for r in rows))
        check('closed_tagged_way_keeps_way_source_identity',('school','way',12) in identities)
        check('untagged_relation_members_not_independent_POIs',not any(r['osm_type']=='way' and r['osm_id'] in (10,11) for r in rows))
        park=next(r for r in rows if r['category']=='park');geometry=shapely.from_geojson(json.dumps(park['geometry']))
        polygons=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
        check('multipolygon_hole_preserved_without_centroid',park['osm_type']=='relation' and park['osm_id']==100 and sum(len(p.interiors) for p in polygons)==1 and not geometry.covers(Point(120.5,23.5)) and geometry.covers(Point(120.1,23.1)))
        check('openway_and_site_relation_skipped_and_audited',not any(r['category']=='hospital' for r in rows) and result['error_counts'].get('open_way_not_selected_geometry')==1 and result['error_counts'].get('nonpolygon_relation_type')==1)
        check('unassembled_multipolygon_explicitly_audited',result['unassembled_area_originals']==1 and result['error_counts'].get('closed_way_or_multipolygon_not_assembled')==1)
        check('no_pending_area_has_unqualified_source',result['area_without_qualifying_original']==0)
    return checks


def extract(output=DEFAULT_OUTPUT):
    import osmium,shapely
    output=Path(output).resolve();pbf=inventory.PBF.resolve()
    need(output.is_relative_to((ROOT/'outputs/round9/poi_index').resolve()) and output.name=='2022' and not output.exists(),'Use a new 2022 directory under outputs/round9/poi_index')
    need(version('osmium')=='4.3.1','Use .venv-geo osmium 4.3.1')
    source_paths=[Path(__file__),ROOT/'src/round9_poi_inventory.py',ROOT/'src/geo_osm_extract.py']
    sources={relative(p):sha256(p) for p in source_paths}
    need(sources['src/round9_poi_inventory.py']=='bc0f40d71d3deaae9daa05ae346b82000a0d4eeb8b95243dffbd7ccb71a550f0','Frozen inventory rules changed')
    evidence_paths=[inventory.EXTRACTION,inventory.DOWNLOAD,ROOT/'outputs/round9/audit/poi_inventory_2022.json']
    evidence={relative(p):sha256(p) for p in evidence_paths}
    old=json.loads(inventory.EXTRACTION.read_text(encoding='utf-8-sig'))
    pool=osmium.io.ThreadPool(4,8);header=source_header(pbf,osmium,pool);initial_sha=sha256(pbf);initial_bytes=pbf.stat().st_size
    need(initial_sha==inventory.EXPECTED_SHA==old['source']['sha256'] and initial_bytes==old['source']['bytes'],'PBF source differs')
    need(header['replication_timestamp']==inventory.EXPECTED_TIME==old['source']['header']['replication_timestamp'],'Actual PBF header differs')
    checks=self_check()
    output.parent.mkdir(parents=True,exist_ok=True);staging=output.parent/('.2022-staging-'+uuid.uuid4().hex)
    need(staging.parent==output.parent and staging.is_relative_to(ROOT.resolve()),'Unsafe staging');staging.mkdir()
    try:
        print(json.dumps({'event':'poi_extract_started','source_sha256':initial_sha,'replication_timestamp':header['replication_timestamp'],'categories':list(CATEGORY_TAGS),'synthetic_checks':len(checks)}),flush=True)
        result=extract_core(pbf,staging/'pois.jsonl',staging/'errors.jsonl',osmium,shapely,pool)
        ending_header=source_header(pbf,osmium,pool)
        need(sha256(pbf)==initial_sha and pbf.stat().st_size==initial_bytes and ending_header==header,'Source PBF/header changed during extraction')
        need(sources=={relative(p):sha256(p) for p in source_paths} and evidence=={relative(p):sha256(p) for p in evidence_paths},'Source code/evidence changed during extraction')
        original_inventory=json.loads(evidence_paths[-1].read_text(encoding='utf-8'))
        for category,pairs in CATEGORY_TAGS.items():
            for kind in TYPES:
                expected=sum(original_inventory['by_tag'][k+'='+v][kind]['direct_elements'] for k,v in pairs)
                need(result['category_counts'][category][kind].get('direct_candidates',0)==expected,'Original element count differs from frozen inventory: '+category+'/'+kind)
        files={key:{'path':relative(output/(key+'.jsonl')),'sha256':sha256(staging/(key+'.jsonl')),'bytes':(staging/(key+'.jsonl')).stat().st_size} for key in ('pois','errors')}
        audit={'format_version':1,'status':'complete','year':2022,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source':{'path':relative(pbf),'bytes':initial_bytes,'sha256':initial_sha,'header':header},'source_header_after':ending_header,'source_before_after_verified':True,
            'script':{'path':relative(__file__),'sha256':sha256(__file__)},'source_sha256':sources,'input_evidence_sha256':evidence,
            'versions':{'python':platform.python_version(),'osmium':version('osmium'),'shapely':version('shapely'),'geos':str(shapely.geos_version_string)},
            'rules':rules(),'crs':'EPSG:4326','coordinate_order':['longitude','latitude'],'geometry_format':'GeoJSON','io_threads':4,'location_cache':'flex_mem',
            'assembly':'normal pyosmium area assembly including relation members before final tag filtering; original tagged source ID required',
            'files':files,'synthetic_checks':checks,'original_candidate_counts_match_frozen_inventory':True,
            'license':{'name':'Open Database License 1.0','spdx':'ODbL-1.0','attribution':'OpenStreetMap contributors; Geofabrik extract','url':'https://www.openstreetmap.org/copyright'},
            'ml_data_or_predictions_or_scores_read':False,'external_property_or_transaction_answers_used':False,'downloads_performed':False,
            'limitations':['No inactive tag is not evidence of operation.','Catalog elements are not deduplicated facilities; different OSM IDs and overlapping categories remain separate.','Mapping completeness and snapshot time are not facility opening dates.','Line geometry and non-multipolygon relation types are excluded only for these 13 predeclared categories.','Unassembled or invalid geometry is omitted, never repaired, replaced by centroid, or extrapolated.'],**result}
        (staging/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (staging/'audit.sha256').write_text(sha256(staging/'audit.json')+'\n',encoding='ascii')
        need(not output.exists(),'Output appeared; never overwrite');staging.rename(output)
    except BaseException:
        print('Incomplete extraction; staging retained: '+str(staging),file=sys.stderr,flush=True);raise
    print(json.dumps({'event':'poi_extract_complete','output':relative(output),'audit_sha256':sha256(output/'audit.json'),'pois_sha256':files['pois']['sha256'],'category_counts':audit['category_counts'],'errors':audit['error_counts'],'elapsed_seconds':audit['elapsed_seconds']},ensure_ascii=True),flush=True)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true');parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    if args.self_check:print(json.dumps({'status':'passed','checks':self_check(),'synthetic_only':True},ensure_ascii=True))
    else:extract(args.output)