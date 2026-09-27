"""Annual provenance wrapper around the unchanged R9 2022 extraction core.

Only the predeclared local 2021/2023/2024 snapshots are accepted. Category,
lifecycle and geometry rules are imported unchanged; no globals are patched.
"""
from __future__ import annotations
import argparse,json,platform,sys,time,uuid
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import round9_poi_extract as core

ROOT=Path(__file__).resolve().parents[1]
YEARS={
 2021:{'stage':'auxiliary','sha256':'7f9845363ec6d03e2b2b752b6e1a6a3efa5b8757090b8033974cc1d8e4525323','timestamp':'2021-01-01T21:42:03Z','bytes':84556154},
 2023:{'stage':'confirmation','sha256':'891dd721eca709a4c09d8dd6e2367975c5bf2a52b8f5903fa4bcea752fc5a4ba','timestamp':'2023-01-01T21:21:53Z','bytes':114165920},
 2024:{'stage':'final','sha256':'0e087ae554ecc657b57ff3e26a8d4f79ed8df712ed43056ed2f8e1ff0d453ba1','timestamp':'2024-01-01T21:21:15Z','bytes':242830461}}
FROZEN_SOURCES={'src/round9_poi_extract.py':'3a747c2afbc756b64d69de07a76dd4d53556f5bc0e90fe4a383eb049e54f95a1',
 'src/round9_poi_inventory.py':'bc0f40d71d3deaae9daa05ae346b82000a0d4eeb8b95243dffbd7ccb71a550f0',
 'src/geo_osm_extract.py':'dfa1b06dfd113ad1ca2b58a48209ec1b62838603192c327f7b0e0e9cd1668ba1'}
need,sha,rel=core.need,core.sha256,core.relative


def locked(path):
    path=Path(path);need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(path),'Receipt mismatch: '+rel(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def extract(year):
    import osmium,shapely
    need(year in YEARS,'Only the fixed 2021/2023/2024 snapshots are authorized')
    spec=YEARS[year];output=ROOT/f'outputs/round9/poi_index/{year}'
    need(not output.exists(),'Annual output exists; never overwrite')
    need(version('osmium')=='4.3.1','Use the preserved .venv-geo osmium 4.3.1')
    sources={name:sha(ROOT/name) for name in FROZEN_SOURCES}
    need(sources==FROZEN_SOURCES,'Frozen category/geometry/lifecycle source changed')
    sources[rel(__file__)]=sha(__file__)
    pbf=ROOT/f'data_external/geography/taiwan-{str(year)[2:]}0101.osm.pbf'
    old_path=ROOT/f'outputs/geography/osm_{year}/audit.json'
    download_path=pbf.with_name(pbf.name.replace('.osm.pbf','.download.json'))
    r6_path=ROOT/f"outputs/round6/geography/{spec['stage']}/manifest.json"
    catalog_2022=ROOT/'outputs/round9/poi_index/2022/audit.json'
    old=locked(old_path);r6=locked(r6_path);original_catalog=locked(catalog_2022)
    download=json.loads(download_path.read_text(encoding='utf-8-sig'))
    evidence_paths=[old_path,old_path.with_suffix('.sha256'),download_path,r6_path,r6_path.with_suffix('.sha256'),catalog_2022,catalog_2022.with_suffix('.sha256')]
    evidence={rel(p):sha(p) for p in evidence_paths}
    need(r6['stage']==spec['stage'] and r6['geography_provenance_chain']['verified'] is True,'Wrong R6 annual geography contract')
    need(original_catalog['rules']==core.rules(),'Rules differ from completed 2022 catalog')
    initial_sha=sha(pbf);initial_bytes=pbf.stat().st_size
    need(initial_sha==spec['sha256']==old['source']['sha256']==download['sha256']==r6['source_sha256'][rel(pbf)],'Annual PBF SHA differs')
    need(initial_bytes==spec['bytes']==old['source']['bytes']==download['bytes'],'Annual PBF size differs')
    for path in (old_path,old_path.with_suffix('.sha256'),ROOT/'src/geo_osm_extract.py'):
        need(r6['source_sha256'][rel(path)]==sha(path),'Old annual extraction evidence differs from R6')
    pool=osmium.io.ThreadPool(4,8);header=core.source_header(pbf,osmium,pool)
    need(header['replication_timestamp']==spec['timestamp']==old['source']['header']['replication_timestamp'],'Annual PBF header differs')
    checks=core.self_check()
    output.parent.mkdir(parents=True,exist_ok=True);staging=output.parent/(f'.{year}-staging-'+uuid.uuid4().hex)
    need(staging.parent==output.parent and staging.is_relative_to(ROOT.resolve()),'Unsafe staging directory');staging.mkdir()
    try:
        print(json.dumps({'event':'annual_poi_extract_started','year':year,'source_sha256':initial_sha,'timestamp':header['replication_timestamp'],'synthetic_checks':len(checks)}),flush=True)
        result=core.extract_core(pbf,staging/'pois.jsonl',staging/'errors.jsonl',osmium,shapely,pool)
        ending=core.source_header(pbf,osmium,pool)
        need(sha(pbf)==initial_sha and pbf.stat().st_size==initial_bytes and ending==header,'Annual PBF/header changed during extraction')
        need(sources=={name:sha(ROOT/name) for name in sources} and evidence=={rel(p):sha(p) for p in evidence_paths},'Source code or annual provenance changed during extraction')
        need(original_catalog['rules']==core.rules(),'Extraction rules mutated')
        files={key:{'path':rel(output/(key+'.jsonl')),'sha256':sha(staging/(key+'.jsonl')),'bytes':(staging/(key+'.jsonl')).stat().st_size} for key in ('pois','errors')}
        audit={'format_version':1,'status':'complete','year':year,'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'source':{'path':rel(pbf),'bytes':initial_bytes,'sha256':initial_sha,'header':header},'source_header_after':ending,'source_before_after_verified':True,
          'script':{'path':rel(__file__),'sha256':sha(__file__)},'source_sha256':sources,'input_evidence_sha256':evidence,
          'frozen_core':{'path':'src/round9_poi_extract.py','sha256':FROZEN_SOURCES['src/round9_poi_extract.py']},
          'same_rules_as_2022_catalog':True,'reference_2022_audit_sha256':sha(catalog_2022),
          'r6_geography_manifest':{'path':rel(r6_path),'sha256':sha(r6_path),'stage':spec['stage']},
          'versions':{'python':platform.python_version(),'osmium':version('osmium'),'shapely':version('shapely'),'geos':str(shapely.geos_version_string)},
          'rules':core.rules(),'crs':'EPSG:4326','coordinate_order':['longitude','latitude'],'geometry_format':'GeoJSON','io_threads':4,'location_cache':'flex_mem',
          'assembly':original_catalog['assembly'],'files':files,'synthetic_checks':checks,
          'original_candidate_counts_match_frozen_inventory':None,'inventory_comparison_note':'The 2022 tag counts are not an expected inventory for another year; only rules and synthetic tests are shared.',
          'license':original_catalog['license'],'ml_data_or_predictions_or_scores_read':False,'external_property_or_transaction_answers_used':False,'downloads_performed':False,
          'limitations':original_catalog['limitations'],**result}
        (staging/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (staging/'audit.sha256').write_text(sha(staging/'audit.json')+'\n',encoding='ascii')
        need(not output.exists(),'Annual output appeared; never overwrite');staging.rename(output)
    except BaseException:
        print('Incomplete annual extraction; staging retained: '+str(staging),file=sys.stderr,flush=True);raise
    print(json.dumps({'event':'annual_poi_extract_complete','year':year,'output':rel(output),'audit_sha256':sha(output/'audit.json'),'pois_sha256':files['pois']['sha256'],
      'rows':result['counts'].get('rows_written',0),'geometry_counts':{k:v for k,v in result['counts'].items() if k.startswith('geometry_')},'errors':result['error_counts'],'elapsed_seconds':result['elapsed_seconds'],'wrapper_sha256':sha(__file__)},ensure_ascii=True),flush=True)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--year',type=int,required=True,choices=sorted(YEARS));args=parser.parse_args();extract(args.year)