"""Align frozen geographic matches to the existing model rows; do not train/score.

Existing price-containing duplicate rules are reproduced solely for row alignment.
No price is used to compute coordinates, quality, grids, or distance features.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from model_common import ROOT,load_data,full_training_data
from prepare_data import ID
from round6_common import OUT,require,read,read_locked,locked_write,write,rel,sha256,verify_hashes
from round6_geo import encode_geo_features,FEATURE_RULES,FEATURES
YEARS={'development':2022,'confirmation':2023,'auxiliary':2021,'final':2024}

def source_files(folders):
    result={}
    for folder in folders:
        folder=Path(folder).resolve()
        require(folder.is_dir() and folder.is_relative_to(ROOT.resolve()),'Missing or unsafe geographic directory')
        for path in folder.rglob('*'):
            if path.is_file() and not any(p.startswith('.') for p in path.relative_to(folder).parts) and path.suffix in {'.json','.jsonl','.sha256','.pkl','.csv'}:
                result[rel(path)]=sha256(path)
    return result

def _chain_path(root,value,base=None):
    path=Path(value)
    if not path.is_absolute():path=(root if base is None else base)/path
    path=path.resolve()
    require(path.is_relative_to(root) and path.is_file(),'Missing/unsafe recorded dependency: '+str(value))
    return path


def _chain_recorded_paths(value,audit_path,root):
    """Read existing path/hash declarations; never invent a hash for a source."""
    found={}
    def add(name,digest,base=None):
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdefABCDEF' for c in digest),'Malformed recorded SHA')
        path=_chain_path(root,name,base);digest=digest.lower()
        require(path not in found or found[path]==digest,'Conflicting recorded dependency SHA: '+str(path))
        found[path]=digest
    def walk(node):
        if isinstance(node,list):
            for item in node:walk(item)
        elif isinstance(node,dict):
            if 'path' in node and 'sha256' in node:add(node['path'],node['sha256'])
            if isinstance(node.get('source'),str) and isinstance(node.get('source_sha256'),str):add(node['source'],node['source_sha256'])
            for key,item in node.items():
                if key.endswith('_path') and key[:-5]+'_sha256' in node:
                    add(item,node[key[:-5]+'_sha256'])
                if key.endswith('_sha256') and isinstance(item,dict):
                    for name,digest in item.items():
                        if isinstance(digest,str):
                            base=audit_path.parent if key=='output_sha256' and Path(name).name==name else None
                            add(name,digest,base)
                walk(item)
    walk(value)
    return found


def verify_geography_chain(stage,matches,index,extraction,*,root=None,parser_folder=None):
    """Read-only audit verification for v2/v3 indices and annual snapshots.

    The matcher/parser historically have no receipt: verify their recorded
    source/input/output hashes, then bind the audit bytes in the returned map.
    Existing extraction/index/parent receipts are mandatory and checked. No
    retroactive receipts, geographic features, price labels or models are made.
    """
    root=Path(ROOT if root is None else root).resolve()
    require(stage in YEARS,'Unknown geographic stage')
    matches,index,extraction=[(root/Path(p)).resolve() for p in (matches,index,extraction)]
    parser_folder=(root/'outputs/geography/address_audit' if parser_folder is None else root/Path(parser_folder)).resolve()
    for folder in [matches,index,extraction,parser_folder]:
        require(folder.is_relative_to(root) and folder.is_dir(),'Missing/unsafe geographic directory')
    verified={};audits={};receipts={}
    def bind(path,expected=None):
        path=_chain_path(root,path)
        actual=verified.get(path)
        if actual is None:actual=sha256(path);verified[path]=actual
        require(expected is None or actual==expected.lower(),'Recorded geographic SHA differs: '+str(path))
        return actual
    def load_audit(path,receipt_required):
        path=_chain_path(root,path)
        if path in audits:return audits[path]
        digest=bind(path);receipt=path.with_suffix('.sha256')
        if receipt.exists():
            parts=receipt.read_text(encoding='utf-8-sig').strip().split()
            require(bool(parts) and parts[0].lower()==digest,'Geographic audit receipt differs: '+str(path))
            require(len(parts)==1 or (len(parts)==2 and parts[1].lstrip('*')==path.name),'Unexpected audit receipt filename')
            bind(receipt);receipts[path.relative_to(root).as_posix()]='verified_existing_receipt'
        else:
            require(not receipt_required,'Missing existing-required audit receipt: '+str(path))
            receipts[path.relative_to(root).as_posix()]='no_original_receipt; audit_SHA_bound_by_R6_manifest_and_plan'
        value=read(path);declared=_chain_recorded_paths(value,path,root)
        for dependency,expected in declared.items():bind(dependency,expected)
        audits[path]=(value,declared)
        return value,declared
    def edge(declared,path,label):
        path=_chain_path(root,path)
        require(path in declared,'Missing explicit provenance link: '+label)
        bind(path,declared[path])
    ext,ext_paths=load_audit(extraction/'audit.json',True)
    header=ext['source']['header'];timestamp=header['replication_timestamp']
    require(header['replication_timestamp_header_key']=='osmosis_replication_timestamp','Timestamp is not from the PBF replication header')
    year=YEARS[stage];origin=pd.Timestamp(f'{year}-08-11',tz='Asia/Taipei').tz_convert('UTC')
    stamp=pd.Timestamp(timestamp)
    require(stamp.tzinfo is not None and stamp<origin and stamp.year==year,'Wrong snapshot year or unavailable at forecast origin')
    pbf=_chain_path(root,ext['source']['path'])
    require(pbf.name==f'taiwan-{str(year)[2:]}0101.osm.pbf','Wrong prespecified annual PBF filename')
    pbf_sha=bind(pbf,ext['source']['sha256']);areas=extraction/'administrative_areas.jsonl';roads=extraction/'roads.jsonl'
    for key,path in [('administrative_areas',areas),('roads',roads)]:
        require(_chain_path(root,ext['files'][key]['path'])==path,'Extraction output points outside supplied source folder')
        edge(ext_paths,path,'extraction '+key)
    idx,idx_paths=load_audit(index/'audit.json',True)
    require(pd.Timestamp(idx['source_replication_timestamp'])==stamp,'Index and extraction timestamps differ')
    for path in [roads,areas,extraction/'audit.json',extraction/'audit.sha256',pbf]:edge(idx_paths,path,'index to extraction/PBF')
    for path in [index/'road_index.jsonl',index/'district_inventory.json']:edge(idx_paths,path,'index output')
    parent_audits=[]
    for path in idx_paths:
        if path.name!='audit.json' or path==extraction/'audit.json':continue
        candidate=read(path)
        if isinstance(candidate.get('inputs'),dict) and 'source_extraction_audit' in candidate['inputs'] and 'exported_subareas' in candidate:
            parent,dependencies=load_audit(path,True)
            require(pd.Timestamp(parent['source_replication_timestamp'])==stamp,'Parent proof uses another snapshot timestamp')
            for key,expected in [('pbf',pbf),('administrative_areas',areas),('source_extraction_audit',extraction/'audit.json'),('source_extraction_receipt',extraction/'audit.sha256')]:
                require(_chain_path(root,parent['inputs'][key]['path'])==expected,'Parent proof points to another extraction: '+key)
                edge(dependencies,expected,'parent to extraction');edge(idx_paths,expected,'index to parent dependency')
            for record in parent['files'].values():edge(idx_paths,_chain_path(root,record['path']),'index to parent output')
            edge(idx_paths,path.with_suffix('.sha256'),'index to parent receipt')
            for item in parent['exported_subareas']+parent.get('missing_subareas',[]):
                require(item['source_pbf_sha256']==pbf_sha and item['source_areas_sha256']==bind(areas)
                        and item['source_extraction_audit_sha256']==bind(extraction/'audit.json')
                        and pd.Timestamp(item['source_replication_timestamp'])==stamp,'Parent member has mixed source provenance')
            parent_audits.append(path)
    require(len(parent_audits)==1,'Expected one explicitly recorded parent-proof audit')
    proof=idx['administration']['parent_evidence']
    require(proof['parent_audit_sha256']==bind(parent_audits[0]) and proof['source_pbf_sha256']==pbf_sha
            and proof['source_areas_sha256']==bind(areas) and pd.Timestamp(proof['source_replication_timestamp'])==stamp,
            'Index parent evidence does not belong to supplied extraction')
    parsed_audit,parsed_paths=load_audit(parser_folder/'audit.json',False)
    parser=root/'src/geo_address_audit.py';bind(parser,parsed_audit['source_sha256'])
    if 'road_query_counts_sha256' in parsed_audit:bind(parser_folder/'road_query_counts.csv',parsed_audit['road_query_counts_sha256'])
    match,match_paths=load_audit(matches/'audit.json',False)
    require((root/Path(match['index_directory'])).resolve()==index,'Matching audit points to a different supplied index')
    for path in [index/'road_index.jsonl',index/'audit.json',parser]:edge(match_paths,path,'matches to index/parser')
    require(match['target_values_read'] is False and match['model_fitting_or_scoring_performed'] is False,'Matcher used price labels/models')
    records={item['split']:item for item in parsed_audit['splits']}
    for split in ['train','valid','test']:
        parsed=parser_folder/f'{split}_parsed.pkl';output=matches/f'{split}_matched.pkl'
        require(_chain_path(root,records[split]['parsed_path'])==parsed,'Parser output path differs from canonical input')
        edge(parsed_paths,parsed,'parser output');edge(match_paths,parsed,'matches to parsed input')
        record=match['splits'][split]
        require(_chain_path(root,record['output_path'])==output,'Matched output outside supplied folder')
        edge(match_paths,output,'matched output')
    # Recheck after traversing the chain, including all existing declaration maps.
    for path,digest in verified.items():require(sha256(path)==digest,'Source changed during geographic chain verification: '+str(path))
    report={'verified':True,'schema_version':1,'stage':stage,'snapshot_year':year,
            'replication_timestamp':timestamp,'forecast_origin_utc':origin.isoformat(),
            'audit_receipts':receipts,'parent_audit_path':parent_audits[0].relative_to(root).as_posix(),
            'matches_to_index_to_extraction_PBF_verified':True,'parser_and_parsed_outputs_verified':True,
            'recorded_source_input_output_hashes_verified':True,'price_data_deserialized':False,
            'retroactive_receipts_created':False,'verified_files':len(verified)}
    return report,{path.relative_to(root).as_posix():digest for path,digest in verified.items()}

def align_one(split,data,matches):
    raw_cache=pd.read_pickle(ROOT/'data_cache'/f'{split}.pkl')
    parsed=pd.read_pickle(ROOT/'outputs/geography/address_audit'/f'{split}_parsed.pkl')
    matched=pd.read_pickle(matches/f'{split}_matched.pkl')
    require(len(parsed)==len(matched) and parsed.index.equals(matched.index),'Parsed/matched row structure differs')
    for col in ['source_id','source_row_index','source_excel_row','existing_feature_cache_row_index']:
        pd.testing.assert_series_equal(parsed[col],matched[col],check_names=True)
    usable=matched.existing_feature_cache_row_index.ge(0)
    aligned=matched.loc[usable].sort_values('existing_feature_cache_row_index')
    p=parsed.loc[usable].sort_values('existing_feature_cache_row_index')
    require(np.array_equal(aligned.existing_feature_cache_row_index.to_numpy(),np.arange(len(raw_cache))),'Cache source indices not a complete one-to-one sequence')
    require(not aligned.is_embedded_header.astype(bool).any(),'Header in cache-aligned geography')
    for source,target in [('source_id',ID),('source_address','土地位置建物門牌'),('source_district','鄉鎮市區'),('source_transaction_type','交易標的')]:
        left=p[source].reset_index(drop=True).fillna('').astype(str)
        right=raw_cache[target].reset_index(drop=True).fillna('').astype(str)
        require(left.equals(right),'Raw/cache provenance differs: '+split+'/'+target)
    encoded=encode_geo_features(aligned.reset_index(drop=True))
    keep=np.ones(len(raw_cache),dtype=bool)
    if split=='train':
        keep=~raw_cache.duplicated(subset=[c for c in raw_cache if c!=ID]).to_numpy()
        require(int((~keep).sum())==529,'Original duplicate count changed')
    raw_kept=raw_cache.loc[keep].reset_index(drop=True)
    require(raw_kept.equals(data['raw_'+split]),'Canonical model raw rows differ')
    geo=encoded.loc[keep].reset_index(drop=True)
    require(len(geo)==len(data['X_'+split]),'Canonical feature count differs')
    return geo,np.flatnonzero(keep),{'original_workbook_rows':len(parsed),'header_removed_cache_rows':len(raw_cache),'canonical_rows':len(geo),'removed_duplicate_rows':int((~keep).sum()),'original_id_address_district_transaction_type_verified':True,'source_cache_index_is_complete_sequence':True,'geo_feature_count':len(FEATURES),'position_rows':int(geo.geo_has_position.sum())}

def prepare(stage,matches,index,extraction):
    year=YEARS[stage];destination=OUT/'geography'/stage
    require(not destination.exists(),'Refusing to replace prepared geographic features')
    matches,index,extraction=map(lambda p:Path(p).resolve(),(matches,index,extraction))
    chain,sources=verify_geography_chain(stage,matches,index,extraction)
    timestamp=chain['replication_timestamp']
    expected_start=pd.Timestamp(chain['forecast_origin_utc'])
    pbf=Path(read(extraction/'audit.json')['source']['path']).resolve()
    sources.update(source_files([matches,index,extraction]))
    for p in [ROOT/'src/round6_geo.py',ROOT/'src/round6_prepare_geo.py',ROOT/'src/geo_address_audit.py',ROOT/'outputs/geography/address_audit/audit.json']:
        sources[rel(p)]=sha256(p)
    for split in ('train','valid','test'):
        for p in [ROOT/'data_cache'/f'{split}.pkl',ROOT/'outputs/geography/address_audit'/f'{split}_parsed.pkl']:
            sources[rel(p)]=sha256(p)
    data=load_data();features={};indices={};reports={}
    for split in ('train','valid','test'):
        features[split],indices[split],reports[split]=align_one(split,data,matches)
    full_raw=pd.concat([data['raw_train'],data['raw_valid']],ignore_index=True)
    full_keep=~full_raw.duplicated(subset=[c for c in full_raw if c!=ID]).to_numpy()
    features['full']=pd.concat([features['train'],features['valid']],ignore_index=True).loc[full_keep].reset_index(drop=True)
    full_X,_=full_training_data(data)
    require(len(features['full'])==len(full_X)==467221 and int((~full_keep).sum())==28,'Full-data geography dedup alignment differs')
    verify_hashes(sources)
    destination.mkdir(parents=True)
    output={}
    for split,frame in features.items():
        path=destination/f'{split}.pkl';frame.to_pickle(path)
        output[split]={'path':rel(path),'sha256':sha256(path),'rows':len(frame)}
    for split,values in {**indices,'full':np.flatnonzero(full_keep)}.items():np.save(destination/f'{split}_source_cache_indices.npy',values,allow_pickle=False)
    manifest={'stage':stage,'snapshot_year':year,'replication_timestamp':timestamp,'forecast_origin_utc':expected_start.isoformat(),'created_at_utc':datetime.now(timezone.utc).isoformat(),'feature_rules':FEATURE_RULES,'source_sha256':sources,'source_pbf_sha256':sha256(pbf),'geography_provenance_chain':chain,'matching_audit_path':rel(matches/'audit.json'),'matching_audit_sha256':sha256(matches/'audit.json'),'geo_index_audit_path':rel(index/'audit.json'),'geo_index_audit_sha256':sha256(index/'audit.json'),'geographic_values_learned_from_targets':False,'existing_target_containing_dedup_rule_reproduced_for_alignment':True,'alignment':reports,'full_train_valid_overlap_removed':28,'features':output,'index_files_sha256':{rel(p):sha256(p) for p in destination.glob('*.npy')},'price_model_trained':False,'scores_computed':False}
    locked_write(destination/'manifest.json',manifest)
    print(json.dumps({'stage':stage,'status':'prepared','snapshot_year':year,'features':output,'alignment':reports},ensure_ascii=False),flush=True)

def load_geographic_partition(stage,fi,qi):
    manifest=read_locked(OUT/'geography'/stage/'manifest.json')
    require(manifest['stage']==stage and manifest['geography_provenance_chain']['verified'] is True,'Geographic provenance chain/stage not verified')
    require(manifest['feature_rules']==FEATURE_RULES and manifest['snapshot_year']==YEARS[stage],'Geographic schema/year differs')
    verify_hashes(manifest['source_sha256']);verify_hashes(manifest['index_files_sha256'])
    for item in manifest['features'].values():require(sha256(ROOT/item['path'])==item['sha256'],'Geographic feature data changed')
    if stage=='final':
        fit=pd.read_pickle(ROOT/manifest['features']['full']['path']);query=pd.read_pickle(ROOT/manifest['features']['test']['path'])
        require(len(fit)==len(fi) and len(query)==len(qi),'Final geo partition size differs')
    else:
        all_train=pd.read_pickle(ROOT/manifest['features']['train']['path'])
        fit=all_train.iloc[fi].reset_index(drop=True);query=all_train.iloc[qi].reset_index(drop=True)
    return fit,query,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=list(YEARS),required=True);p.add_argument('--matches',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--extraction',type=Path,required=True)
    a=p.parse_args();prepare(a.stage,a.matches,a.index,a.extraction)
