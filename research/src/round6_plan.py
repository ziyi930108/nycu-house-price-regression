"""Freeze the one geographic candidate before any round-six model fitting."""
from datetime import datetime,timezone
from round6_common import *
from prepare_data import DATA_DIR

def main():
    from round6_geo import FEATURE_RULES,parameters
    from round6_evaluate import FORMULAS,REQUIRED_COMPONENTS,THRESHOLDS,source_hashes
    from round3_evaluate import target_weights,runtime_versions
    from round6_prepare_geo import load_geographic_partition
    for name in ['models','evaluation','baseline','candidate_plan.json','candidate_plan.sha256']:
        require(not (OUT/name).exists(),'Cannot predeclare after experiment artifacts or a plan exist')
    require(sha256(ROOT/'outputs/submission_v2.csv')=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b','Retained v2 changed')
    require(sha256(ROOT/'runs/round5_experiments/snapshot_manifest.json')=='10444c2e1be03f2e3a6354784c4b6cb9bf5baf8dcf2b5b4e99026c5c9f75d4bf','R5 snapshot changed')
    qa=read(OUT/'audit/preflight_final.json');require(qa['status']=='passed','Final synthetic preflight missing')
    require(qa['helper_source_sha256']==sha256(ROOT/'src/round6_geo.py') and qa['evaluator_source_sha256']==sha256(ROOT/'src/round6_evaluate.py'),'Preflight source differs')
    require(qa['preflight_source_sha256']==sha256(ROOT/'src/round6_preflight.py'),'Preflight verifier differs')
    require(sha256(ROOT/'src/round6_prepare_geo.py')=='b8be469fc1217af9487534e1fb2ed048704ba9008e966a382d2f7ea8108344f0','Reviewed provenance/alignment source differs')
    audit_path=ROOT/'outputs/geography/road_index_2022_annual_v3/audit.json'
    audit=read(audit_path)
    require(audit_path.with_suffix('.sha256').read_text().split()[0]==sha256(audit_path),'Annual index audit receipt differs')
    eq=audit['equivalence_check']
    require(eq['road_index_bytes_identical'] and eq['district_identity_inventory_identical'],'Annual index algorithm not equivalent on development inputs')
    require(eq['road_index_sha256']==sha256(ROOT/'outputs/geography/road_index_2022_parent_verified/road_index.jsonl')==sha256(audit_path.parent/'road_index.jsonl'),'Annual development road parity differs')
    verify_hashes(audit['source_sha256'])
    data=load_data();previous=read_locked(ROOT/'outputs/round5/candidate_plan.json')
    contract={}
    for stage,(start,end) in SPLITS.items():
        fi,qi=temporal_indices(data,start,end)
        contract[stage]={'fit_rows':len(fi),'query_rows':len(qi),
            'fit_indices_content_sha256':hashlib.sha256(np.asarray(fi,dtype='<i8').tobytes()).hexdigest(),
            'query_indices_content_sha256':hashlib.sha256(np.asarray(qi,dtype='<i8').tobytes()).hexdigest()}
    require(contract==previous['split_contract'] and THRESHOLDS==previous['thresholds'],'Annual split/gates changed')
    fi,qi=temporal_indices(data,*SPLITS['development']);geo,geoq,manifest=load_geographic_partition('development',fi,qi)
    require(len(geo)==240316 and len(geoq)==81234,'Geographic partition differs')
    inputs=dict(previous['data_sha256']);verify_hashes(inputs)
    for p in [ROOT/'runs/round5_experiments/snapshot_manifest.json',ROOT/'outputs/round5/candidate_plan.json',ROOT/'data_external/geography/requirements-geo.txt',OUT/'geography/development/manifest.json',OUT/'geography/development/manifest.sha256']:
        inputs[rel(p)]=sha256(p)
    inputs.update(manifest['source_sha256'])
    for item in manifest['features'].values():inputs[item['path']]=item['sha256']
    inputs.update(manifest['index_files_sha256'])
    geographic_sources=['geo_address_audit.py','geo_osm_extract.py','geo_parent_coverage.py','geo_parent_coverage_v2.py','geo_road_index.py','geo_road_index_v2.py','geo_road_index_v3.py','geo_match_queries.py','geo_match_queries_v2.py']
    geo_hashes={'src/'+name:sha256(ROOT/'src'/name) for name in geographic_sources}
    full_X,full_y=full_training_data(data)
    evidence=[OUT/'research_decision.json',OUT/'audit/preflight_final.json',OUT/'audit/provenance_chain_synthetic_checks.json',OUT/'audit/provenance_chain_real_draft.json',ROOT/'reports/round6_validation_plan.md',ROOT/'reports/round6_preprocessing_record.md',audit_path,audit_path.with_suffix('.sha256'),ROOT/'outputs/geography/kaohsiung_parent_verified_2022_annual_v2/audit.json']
    value={'round':6,'created_at_utc':datetime.now(timezone.utc).isoformat(),'control':'v2_control',
       'candidate_formulas':FORMULAS,'required_components':REQUIRED_COMPONENTS,'thresholds':THRESHOLDS,
       'splits':{k:list(v) for k,v in SPLITS.items()},'split_contract':contract,
       'final_contract':{'fit_rows':len(full_X),'query_rows':len(data['X_test'])},
       'fit_source_sha256':fit_source_hashes(),'evaluation_source_sha256':source_hashes(),'geography_transform_source_sha256':geo_hashes,
       'data_sha256':inputs,'original_input_sha256':data['metadata']['source_sha256'],
       'research_evidence_sha256':{rel(p):sha256(p) for p in evidence},'plan_creator_sha256':sha256(Path(__file__)),
       'runtime_versions':runtime_versions(),'candidates':{'geography_blend':{'formula':'.75*(.5*old_catboost+.5*geography_catboost)+.25*unit_lightgbm','components':['geography_catboost']}},
       'components':{'geography_catboost':{'parameters':parameters(),'feature_rules':FEATURE_RULES,'target':'y/1e6','weight':'uniform','features':'original 87 unchanged plus 8 fixed geographic fields; 95 columns, 23 categorical'}},
       'geography':{'years':{'development':2022,'confirmation':2023,'auxiliary':2021,'final':2024},'filename':'taiwan-YY0101.osm.pbf',
          'timestamp_evidence':'actual PBF replication timestamp, strictly before stage August 10 16:00 UTC',
          'same_snapshot_for_fit_and_query':True,'source':'Geofabrik historical OpenStreetMap; no property prices',
          'index_policy':'v3 annual wrapper preserves base districts and adds only valid explicit parent subareas absent from base; 2022 existing v2 values must match v3 exactly',
          'development_manifest_sha256':sha256(OUT/'geography/development/manifest.json'),'later_snapshots':'Acquire fixed annual snapshot and lock its complete provenance before each stage fit; never choose a version using error scores'},
       'training':{'eval_set':False,'early_stopping':False,'new_development_components':1,'new_development_candidates':1,
           'query_labels_for_features_or_training':False,'hyperparameter_sweep':False,'seed_sweep':False,'blend_weight_sweep':False,'geographic_grid_sweep':False},
       'prediction_rule':'Use original 87-column X for old models, scoring groups and the fit-side v2 reference. Apply trend exactly once to each fixed composite.',
       'development_rules':previous['development_rules'],'confirmation_rules':previous['confirmation_rules'],'auxiliary_rules':previous['auxiliary_rules'],
       'selection':'Single prespecified candidate; if every development gate passes freeze it; otherwise retain v2 and stop this candidate. Later stages veto only; never reselection.',
       'target_group_weights':target_weights(data),'quarter_definition':previous['quarter_definition'],
       'historically_untouched':False,'public_score_policy':'Record uploaded v2 score only; never tune to Public','external_property_answers':False,
       'prior_round5_manifest_sha256':sha256(ROOT/'runs/round5_experiments/snapshot_manifest.json'),
       'limitations':['Development and confirmation labels were used in earlier rounds; selection bias remains.',
           'Geographic inputs use forecast-origin maps, not the map available at each older training transaction.',
           'Road and planar-intersection coordinates are approximate and do not identify exact homes or road grade connectivity.',
           'One percent is an engineering acceptance gate, not statistical significance. Private improvement is unknown.',
           'Existing target-containing deduplication is reproduced for alignment only; no price-derived geography or external property answers.']}
    locked_write(OUT/'candidate_plan.json',value)
    print(json.dumps({'plan':rel(OUT/'candidate_plan.json'),'sha256':sha256(OUT/'candidate_plan.json'),'candidates':1,'split_contract':contract,'final_contract':value['final_contract']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
