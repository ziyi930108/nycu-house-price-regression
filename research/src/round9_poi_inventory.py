"""One-pass, local 2022 OSM tag inventory. No ML data, prices or polygon assembly.

Run in .venv-geo. Counts describe tag evidence, not confirmed operating facilities.
Import and self_check perform no file reads. Actual scan refuses output overwrite.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,platform,time,unicodedata
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PBF=ROOT/'data_external/geography/taiwan-220101.osm.pbf'
EXTRACTION=ROOT/'outputs/geography/osm_2022/audit.json'
DOWNLOAD=ROOT/'data_external/geography/taiwan-220101.download.json'
EXPECTED_SHA='b7650b8c77deecb27592782cada8241f88b7cdbc2838bf8e14b20c7957e63ff9'
EXPECTED_TIME='2022-01-01T21:21:26Z'
OUTPUT=ROOT/'outputs/round9/audit/poi_inventory_2022.json'
TARGETS=(('railway','station'),('railway','halt'),('railway','subway_entrance'),
    ('public_transport','station'),('amenity','school'),('amenity','kindergarten'),
    ('amenity','college'),('amenity','university'),('amenity','hospital'),
    ('amenity','clinic'),('amenity','pharmacy'),('amenity','bus_station'),
    ('shop','supermarket'),('shop','convenience'),('leisure','park'),
    ('highway','motorway_junction'))
TYPES=('node','way','relation')
LIFECYCLE=('construction','disused','abandoned','proposed','planned','demolished','razed','removed','was')
FALSE_VALUES=('', 'no','false','0')
STATE_KEYS=('status','state','lifecycle','building','natural','man_made','railway','public_transport','amenity','shop','leisure','highway')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def classify(tags):
    """Exact tags only; lifecycle-only target evidence remains separate."""
    direct=[(key,value) for key,value in TARGETS if tags.get(key)==value]
    prefixed={}
    for key,value in TARGETS:
        status=[prefix for prefix in LIFECYCLE if tags.get(prefix+':'+key)==value]
        if status:prefixed[(key,value)]=status
    return direct,prefixed


def inactive_evidence(tags,key,value):
    evidence=[]
    for status in LIFECYCLE:
        if status in tags and str(tags[status]).strip().casefold() not in FALSE_VALUES:
            evidence.append('flag:'+status+'='+str(tags[status]))
        if tags.get(status+':'+key)==value:evidence.append('prefix:'+status+':'+key+'='+value)
    for state_key in STATE_KEYS:
        state=str(tags.get(state_key,'')).strip().casefold()
        if state in LIFECYCLE:evidence.append('state:'+state_key+'='+state)
    return sorted(set(evidence))


def normal_name(tags):
    value=next((tags[k] for k in ('name','name:zh','name:zh-Hant') if tags.get(k)),None)
    if value is None:return None
    return ''.join(unicodedata.normalize('NFKC',str(value)).split()).replace('台','臺').casefold() or None


def new_record(element_type):
    return {'direct_elements':0,'with_inactive_evidence':0,'without_inactive_evidence':0,
        'named_without_inactive_evidence':0,'inactive_evidence_counts':Counter(),
        'lifecycle_prefixed_elements':0,'lifecycle_only_elements':0,'lifecycle_prefix_counts':Counter(),
        'coordinates':{'valid_wgs84':0,'invalid_or_unavailable':0,'valid_without_inactive_evidence':0,
            'valid_in_study_bounds':0,'valid_in_study_bounds_without_inactive_evidence':0}
            if element_type=='node' else None}


class Inventory:
    def __init__(self):
        self.total=Counter();self.with_tags=Counter();self.direct_unique=Counter()
        self.direct_unique_inactive=Counter();self.direct_multi=Counter();self.lifecycle_unique=Counter()
        self.by_tag={key+'='+value:{t:new_record(t) for t in TYPES} for key,value in TARGETS}
        self.name_buckets=defaultdict(Counter)
    def consume(self,element_type,tags,coordinates=None):
        self.total[element_type]+=1
        if not tags:return
        self.with_tags[element_type]+=1
        direct,prefixed=classify(tags)
        if prefixed:self.lifecycle_unique[element_type]+=1
        if not direct and not prefixed:return
        name=normal_name(tags)
        if direct:
            self.direct_unique[element_type]+=1
            self.direct_multi[element_type]+=len(direct)>1
        any_evidence=False
        for key,value in direct:
            label=key+'='+value;record=self.by_tag[label][element_type]
            record['direct_elements']+=1
            evidence=inactive_evidence(tags,key,value);any_evidence|=bool(evidence)
            record['with_inactive_evidence' if evidence else 'without_inactive_evidence']+=1
            record['inactive_evidence_counts'].update(evidence)
            if not evidence and name:
                record['named_without_inactive_evidence']+=1
                self.name_buckets[(label,name)][element_type]+=1
            if element_type=='node':
                valid=coordinates is not None and len(coordinates)==2 and all(math.isfinite(v) for v in coordinates) and -180<=coordinates[0]<=180 and -90<=coordinates[1]<=90
                stats=record['coordinates'];stats['valid_wgs84' if valid else 'invalid_or_unavailable']+=1
                if valid:
                    inside=116<=coordinates[0]<=125 and 20<=coordinates[1]<=28
                    stats['valid_without_inactive_evidence']+=not evidence
                    stats['valid_in_study_bounds']+=inside
                    stats['valid_in_study_bounds_without_inactive_evidence']+=inside and not evidence
        if direct and any_evidence:self.direct_unique_inactive[element_type]+=1
        for (key,value),statuses in prefixed.items():
            record=self.by_tag[key+'='+value][element_type]
            record['lifecycle_prefixed_elements']+=1
            record['lifecycle_only_elements']+=(key,value) not in direct
            record['lifecycle_prefix_counts'].update(statuses)
    def result(self):
        overlap={label:{'same_normalized_name_cross_type_buckets':0,'node_way_shared_name_buckets':0,
            'nodes_in_node_way_shared_name_buckets':0,'ways_in_node_way_shared_name_buckets':0,
            'maximum_elements_with_one_normalized_name':0} for label in self.by_tag}
        for (label,_),counts in self.name_buckets.items():
            value=overlap[label];value['same_normalized_name_cross_type_buckets']+=len(counts)>1
            value['maximum_elements_with_one_normalized_name']=max(value['maximum_elements_with_one_normalized_name'],sum(counts.values()))
            if counts['node'] and counts['way']:
                value['node_way_shared_name_buckets']+=1
                value['nodes_in_node_way_shared_name_buckets']+=counts['node']
                value['ways_in_node_way_shared_name_buckets']+=counts['way']
        records={}
        for label,by_type in self.by_tag.items():
            records[label]={}
            for kind,record in by_type.items():
                value=dict(record)
                value['inactive_evidence_counts']=dict(sorted(value['inactive_evidence_counts'].items()))
                value['lifecycle_prefix_counts']=dict(sorted(value['lifecycle_prefix_counts'].items()))
                if value['coordinates'] is not None:
                    value['coordinates']=dict(value['coordinates'])
                    c=value['coordinates'];n=value['direct_elements'];retained=value['without_inactive_evidence']
                    c['valid_wgs84_fraction_of_direct_nodes']=c['valid_wgs84']/n if n else None
                    c['valid_fraction_without_inactive_evidence']=c['valid_without_inactive_evidence']/retained if retained else None
                records[label][kind]=value
        return {'elements_scanned':{t:self.total[t] for t in TYPES},
            'tagged_elements_scanned':{t:self.with_tags[t] for t in TYPES},
            'unique_direct_target_elements':{t:self.direct_unique[t] for t in TYPES},
            'unique_direct_target_elements_with_any_inactive_evidence':{t:self.direct_unique_inactive[t] for t in TYPES},
            'unique_elements_matching_multiple_direct_targets':{t:self.direct_multi[t] for t in TYPES},
            'unique_elements_with_target_lifecycle_prefix':{t:self.lifecycle_unique[t] for t in TYPES},
            'by_tag':records,'cross_element_name_overlap':overlap}


def self_check():
    checks=[]
    def check(name,value):need(value,'Synthetic check failed: '+name);checks.append(name)
    check('fixed_16_exact_target_pairs',len(TARGETS)==16 and len(set(TARGETS))==16)
    check('direct_target_exact_no_semicolon_inference',classify({'amenity':'school;college'})[0]==[])
    check('absence_of_flags_is_only_absence_of_evidence',inactive_evidence({'amenity':'school'},'amenity','school')==[])
    check('construction_disused_abandoned_are_separate',all(inactive_evidence({'amenity':'school',s:'yes'},'amenity','school')==['flag:'+s+'=yes'] for s in ('construction','disused','abandoned')))
    check('false_lifecycle_flags_not_excluded',not inactive_evidence({'amenity':'school','disused':'no','abandoned':'0','construction':'false'},'amenity','school'))
    check('building_construction_is_explicit_evidence',inactive_evidence({'amenity':'school','building':'construction'},'amenity','school')==['state:building=construction'])
    check('prefixed_target_kept_separate',classify({'disused:railway':'station'})==([],{('railway','station'):['disused']}))
    check('unrelated_prefix_does_not_disable_other_target',not inactive_evidence({'amenity':'school','disused:shop':'convenience'},'amenity','school'))
    inv=Inventory();inv.consume('node',{'railway':'station','public_transport':'station','name':'Synthetic'},(121.,25.))
    inv.consume('way',{'railway':'station','name':'Synthetic'})
    inv.consume('node',{'railway':'station','disused':'yes'},None)
    inv.consume('relation',{'abandoned:amenity':'school'})
    out=inv.result();node=out['by_tag']['railway=station']['node']
    check('multi_target_elements_count_once_in_union',out['unique_direct_target_elements']['node']==2 and out['unique_elements_matching_multiple_direct_targets']['node']==1)
    check('valid_node_coordinate_and_inactive_counts_separate',node['direct_elements']==2 and node['coordinates']['valid_wgs84']==1 and node['with_inactive_evidence']==1)
    check('way_relation_geometry_is_unknown_not_invented',out['by_tag']['railway=station']['way']['coordinates'] is None and out['by_tag']['amenity=school']['relation']['coordinates'] is None)
    check('name_overlap_is_only_risk_counter',out['cross_element_name_overlap']['railway=station']['node_way_shared_name_buckets']==1)
    check('lifecycle_only_not_added_to_direct_count',out['by_tag']['amenity=school']['relation']['lifecycle_only_elements']==1 and out['by_tag']['amenity=school']['relation']['direct_elements']==0)
    check('json_serializable_counts',bool(json.dumps(out,allow_nan=False)))
    return checks


def scan(pbf=PBF,output=OUTPUT):
    import osmium
    pbf=Path(pbf).resolve();output=Path(output).resolve()
    need(pbf==PBF.resolve() and pbf.is_file(),'Only the preserved local 2022 PBF is authorized')
    need(output.is_relative_to((ROOT/'outputs/round9').resolve()) and not output.exists(),'Use a new output file inside outputs/round9; never overwrite')
    need(version('osmium')=='4.3.1','Use the frozen .venv-geo osmium 4.3.1')
    source_before={rel(p):sha(p) for p in (EXTRACTION,DOWNLOAD,Path(__file__))}
    extraction=json.loads(EXTRACTION.read_text(encoding='utf-8-sig'));download=json.loads(DOWNLOAD.read_text(encoding='utf-8-sig'))
    initial_stat=pbf.stat();digest=sha(pbf)
    need(digest==EXPECTED_SHA==extraction['source']['sha256']==download['sha256'],'Original PBF SHA differs')
    need(initial_stat.st_size==extraction['source']['bytes']==download['bytes'],'Original PBF size differs')
    with osmium.io.Reader(str(pbf),osmium.osm.NOTHING) as reader:
        header=reader.header();stamp=str(header.get('osmosis_replication_timestamp',''))
        need(not header.has_multiple_object_versions,'History or diff input is unsupported')
    need(stamp==EXPECTED_TIME==extraction['source']['header']['replication_timestamp'],'Actual replication timestamp differs')
    checks=self_check();inventory=Inventory()
    class Handler(osmium.SimpleHandler):
        def node(self,obj):
            tags={str(t.k):str(t.v) for t in obj.tags}
            coordinates=None
            if tags and obj.location.valid():coordinates=(float(obj.location.lon),float(obj.location.lat))
            inventory.consume('node',tags,coordinates)
        def way(self,obj):inventory.consume('way',{str(t.k):str(t.v) for t in obj.tags})
        def relation(self,obj):inventory.consume('relation',{str(t.k):str(t.v) for t in obj.tags})
    print(json.dumps({'event':'source_verified_scan_started','pbf_sha256':digest,'replication_timestamp':stamp,'one_osmium_object_scan':True,'price_data_used':False}),flush=True)
    started=time.perf_counter();handler=Handler();handler.apply_file(str(pbf),locations=False)
    elapsed=time.perf_counter()-started;final_stat=pbf.stat()
    need((initial_stat.st_size,initial_stat.st_mtime_ns)==(final_stat.st_size,final_stat.st_mtime_ns),'PBF stat changed during scan')
    need(source_before=={rel(p):sha(p) for p in (EXTRACTION,DOWNLOAD,Path(__file__))},'Source manifests or script changed during scan')
    result={'format_version':1,'round':9,'status':'pure_map_inventory_completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':{'pbf_path':rel(pbf),'sha256':digest,'bytes':initial_stat.st_size,'replication_timestamp':stamp,'header_field':'osmosis_replication_timestamp','not_filename_date':True,'source_stat_unchanged':True},
        'source_files_sha256':source_before,'versions':{'python':platform.python_version(),'osmium':version('osmium')},
        'fixed_rules':{'target_tags':[list(p) for p in TARGETS],'exact_value_matching':True,'lifecycle_statuses':list(LIFECYCLE),'lifecycle_false_values':list(FALSE_VALUES),'state_tag_keys':list(STATE_KEYS),
            'inactive_flag_rule':'a lifecycle key with a nonempty value other than no/false/0, or a lifecycle value on a listed state key, or a same-target lifecycle prefix',
            'without_inactive_evidence_is_not_proof_of_operation':True,'geometry_for_nodes_only':'native valid WGS84 node location','study_bounds_lon_lat':[116,20,125,28],
            'way_relation_coordinate_policy':'not assembled, not evaluated, coordinates=null','lifecycle_only_not_in_direct_counts':True,'overlapping_reason_counts_not_additive':True,
            'cross_element_duplicates':'not resolved; same normalized name within same target tag is a risk indicator, never a duplicate entity count',
            'name_normalization':'name or name:zh or name:zh-Hant; NFKC/remove whitespace/Tai-to-Taiwan character normalization/casefold'},
        'scan':{'osmium_object_passes':1,'sha_verification_byte_passes':1,'header_only_reads':1,'locations_index_built':False,'areas_or_polygons_assembled':False,'elapsed_seconds':elapsed},
        'synthetic_checks':checks,'ml_data_or_predictions_or_scores_read':False,'external_property_or_transaction_answers_used':False,'downloads_performed':False,
        'limitations':['Missing lifecycle tags do not establish an operating or open facility.','Node location validity is not complete geographic coverage or verified facility precision.','Ways and relations may contain facilities absent from node counts; their geometry was not assembled.','A facility may have node, way and relation representations or multiple entrances/platforms; counts are not unique facility counts.','Shared names may refer to distinct branches or unrelated facilities; names were not used for deduplication.','Lifecycle flags can describe a structure or part of a site; exclusions are conservative evidence summaries, not operating-status verification.','Exact tag inventory omits synonymous or nonstandard tag conventions; no formal prediction features were selected.','Snapshot timestamp dates the OSM extract, not the opening date or completeness of each facility.'],
        **inventory.result()}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps({'status':result['status'],'output':rel(output),'output_sha256':sha(output),'source_sha256':sha(__file__),'elapsed_seconds':elapsed,'counts':{tag:{t:values[t]['direct_elements'] for t in TYPES} for tag,values in result['by_tag'].items()}},ensure_ascii=True),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--pbf',type=Path,default=PBF)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.self_check:print(json.dumps({'checks':self_check(),'real_data_read':False},ensure_ascii=True))
    else:scan(args.pbf,args.output)