"""Freeze sources/requests before all six R102 fits; metadata only."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
import round102_prior_input_experiment as d
import round102_prior_input_evaluate as e
HERE=Path(__file__).resolve().parent

def main():
 for name,source in [('reports/round102_driver_selfcheck.json','src/round102_prior_input_experiment.py'),('reports/round102_evaluator_selfcheck.json','src/round102_prior_input_evaluate.py')]:
  result=c.read(name);c.need(result['status']=='passed' and result['source_sha256']==c.sha(source),'Selfcheck is not final source: '+source)
 protected=c.protected_hashes()
 frozen={**d.source_hashes(),**e.source_hashes(),d.PLAN:d.PLAN_SHA,c.rel(__file__):c.sha(__file__),
         c.rel(HERE/'launch.py'):c.sha(HERE/'launch.py')}
 for name in ['reports/round102_driver_selfcheck.json','reports/round102_evaluator_selfcheck.json','reports/round102_driver_metadata_preflight.json']:
  frozen[name]=c.sha(name)
 r={'status':'frozen_before_real_prior_or_fit','approval_for_six_fits':True,'created_at_utc':c.now(),
    'plan_sha256':d.PLAN_SHA,'parameters':d.parameters(),'source_sha256':d.source_hashes(),
    'frozen_source_and_plan':frozen,'input_authority_sha256':{p:c.sha(p) for p in d.AUTHORITY_PATHS},
    'protected_before':protected,'automatic_retries':False,'query_scores_available_to_model':False}
 c.write(HERE/'research_request.json',r)
 q={'schema_version':1,'plan':{'path':e.PLAN,'sha256':e.PLAN_SHA},'parameters':e.parameters(),
    'source_sha256':e.source_hashes(),'protected_sha256':protected,'launch_request':c.record(HERE/'research_request.json'),
    'R100_seal':e.R100_SEAL,'created_at_utc':c.now(),'frozen_before_new_models':True}
 c.write(HERE/'evaluation_request.json',q)
 d.check_launch(HERE/'research_request.json');c.verify_hashes(frozen)
 c.write(HERE/'freeze_verification.json',{'status':'passed','research_request':c.record(HERE/'research_request.json'),
     'evaluation_request':c.record(HERE/'evaluation_request.json'),'frozen_files':len(frozen),
     'driver_sources':len(r['source_sha256']),'evaluator_sources':len(q['source_sha256']),
     'input_authorities':len(r['input_authority_sha256']),'protected_files':len(protected),'real_arrays_or_models_read':False})
 print(json.dumps(c.read(HERE/'freeze_verification.json')))
if __name__=='__main__':main()
