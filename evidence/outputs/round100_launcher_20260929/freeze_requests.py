"""Freeze fixed R100 fit/evaluation requests after synthetic/source checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
import round100_temporal_prior_experiment as driver
import round100_temporal_prior_evaluate as scorer

HERE=Path(__file__).resolve().parent

def main():
    assert not (ROOT/'outputs/round100').exists(), 'A real run already exists'
    checks=['reports/round100_helper_selfcheck.json','reports/round100_driver_selfcheck.json',
            'reports/round100_evaluator_selfcheck.json']
    for p in checks:
        r=c.read(p);assert r['status']=='passed' and r['real_data_read'] is False
    review='reports/round100_helper_source_review.json'
    r=c.read(review);assert r['status']=='passed', 'Unresolved helper source review'
    sources={**driver.source_hashes(),**scorer.source_hashes()}
    artifacts=[c.PLAN,'reports/round100_x_audit_plan.md','reports/round100_evaluation_contract.md',
        'reports/round100_driver_preflight.md',
        'reports/round100_helper_source_review.md',review,*checks,
        c.rel(__file__),c.rel(HERE/'launch.py')]
    frozen={**sources,**{p:c.sha(p) for p in artifacts}}
    authority={p:c.sha(p) for p in driver.AUTHORITY_PATHS}
    c.need(c.sha(c.PLAN)==c.PLAN_SHA and c.sha(c.TRAINER)==c.TRAINER_SHA,'Plan/trainer changed')
    protected=c.protected_hashes()
    c.need(protected==c.read('outputs/github_publish_r100_20260929/protected_before.json'),'Protected bytes changed')
    research={'status':'frozen_before_real_prior_or_fit','approval_for_six_fits':True,
        'created_at_utc':c.now(),'plan_sha256':c.PLAN_SHA,'source_sha256':driver.source_hashes(),
        'frozen_source_and_plan':frozen,'input_authority_sha256':authority,'protected_before':protected,
        'stage_order':list(c.STAGES),'arms_in_fixed_order':['C','T'],'maximum_NN_fits':6,
        'selfchecks':{p:c.record(p) for p in checks},'helper_source_review':c.record(review),
        'query_evaluation_not_started':True,'new_CSV':False,'selected_submission':'v22','Private_score':None}
    c.write(HERE/'research_request.json',research)
    evaluation={'status':'evaluation_frozen_before_query_score','created_at_utc':c.now(),
        'plan_sha256':c.PLAN_SHA,'parameters':scorer.parameters(),'source_sha256':scorer.source_hashes(),
        'protected_sha256':protected,'research_request':c.record(HERE/'research_request.json')}
    c.write(HERE/'evaluation_request.json',evaluation)
    driver.check_launch(HERE/'research_request.json')
    c.write(HERE/'freeze_verification.json',{'status':'passed','checked_at_utc':c.now(),
        'source_and_plan_files':len(frozen),'input_authorities':len(authority),'protected_files':len(protected),
        'research_request':c.record(HERE/'research_request.json'),
        'evaluation_request':c.record(HERE/'evaluation_request.json'),
        'real_data_deserialized':False,'new_prior_builds':0,'new_NN_fits':0,'query_score_calls':0})
    print(c.read(HERE/'freeze_verification.json'))

if __name__=='__main__':main()
