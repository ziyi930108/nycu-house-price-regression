"""Issue source-specific independent preflight using already measured CPU evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
import round114_research as driver
FOLDER='outputs/research_round114_20261008/'
tick=time.monotonic()
source={**driver.sources(),driver.PLAN:c.sha(driver.PLAN)}
c.need(source[driver.PLAN]==driver.PLAN_SHA=='45b4095430f6adffab7977b37e60eb0e30f958e1a4634e6dd251739acd5b126a','Scientific plan changed')
c.need(source[c.rel(driver.__file__)]=='444113bfcb9ad321015c8af76f6218dfe3c7cd1fd1e659dde762ba765a39c95d','Reviewed driver changed')
c.need(source[driver.TRAINER]=='9b9e44554fe998312a3cd41e1b94cba1c810860edc3b200d95d2a0baeb154e66','Reviewed trainer changed')
c.verify_hashes(source)
def locked(path):
    c.need(c.safe(path).with_suffix('.sha256').read_text(encoding='ascii').strip()==c.sha(path),'Receipt sidecar '+path)
    return c.read(path)
checks=locked(FOLDER+'independent_preflight_checks.json')
c.need(checks['status']=='passed' and checks['check_count']==49 and checks['source_sha256']==source,'Independent CPU guard cases')
formal={}
for name,(entry,environment,flag) in driver.formal_checks().items():
    path=FOLDER+name;receipt=locked(path)
    c.need(receipt['exit_code']==0 and receipt['source_sha256']==c.source_hashes(entry)
           and receipt['command']==[str(ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],
           'Formal CPU command/closure mismatch')
    log=FOLDER+name.replace('_execution.json','.log')
    result=json.loads(c.safe(log).read_text(encoding='utf8').strip())
    c.need(result['status']=='passed' and result['new_fits']==0 and result['GPU_calls']==0,'Formal CPU only scope')
    formal[name]={'execution':c.record(path),'log':c.record(log),'check_count':result['check_count']}
root_replay_path=FOLDER+'root_context_preflight.json';replay=locked(root_replay_path)
c.need(replay['status']=='passed' and replay['source_sha256']==source and replay['origin_order']==list(driver.ORIGINS)
       and set(replay['origins'])==set(driver.ORIGINS) and replay['protected_unchanged']
       and replay['query_prices_read'] is False and replay['raw_price_pickle_deserialized'] is False
       and replay['new_fits']==0 and replay['GPU_calls']==0 and replay['native_model_inference_calls']==0,
       'Root actual fit-only replay scope')
execution=locked(FOLDER+'root_context_preflight_command_execution.json')
c.need(execution['exit_code']==0,'Root actual replay command failure')
c.need(replay['review_tool']==c.record(FOLDER+'replay_preflight.py'),'Actual root replay source unchanged')
for origin,item in replay['origins'].items():
    nf,nq=driver.COUNTS[origin]
    c.need((item['fit_rows'],item['query_rows'])==(nf,nq) and item['original_v22_v24_replayed_exact'],'Replay row/baseline scope')
    arrays=item['prepared_array_identities']
    c.need(set(arrays)==set(c.ARRAY_NAMES) and arrays['train_num']['shape']==[nf,144]
           and arrays['query_num']['shape']==[nq,144] and arrays['train_cat']['shape']==[nf,23]
           and arrays['query_cat']['shape']==[nq,23] and arrays['cardinalities']['shape']==[23],
           'All nine real fit-only input identities')
inputs={};specs=driver.august_specifications(inputs)
extra=driver.offset_comp.specification();driver.merge_hashes(inputs,extra['input_sha256'])
c.need(inputs==replay['input_sha256'],'Independently recomputed metadata-only input closure equals actual replay')
c.verify_hashes(inputs)
c.need(driver.protected()==c.read('outputs/round113/request.json')['protected_before'],'Original protected membership/content')
recipe=driver.trainer_recipe()
trainer_result=json.loads(c.safe(formal['R114_trainer_selfcheck_execution.json']['log']['path']).read_text(encoding='utf8'))
c.need(recipe['configuration']==trainer_result['configuration']
       and recipe['baseline_configuration']==trainer_result['baseline_configuration'],'AST recipe vs actual imported trainer CPU config')
# Cross-check the actual combined-gate implementation with independent arithmetic,
# using complete artificial group tables, not preselected boolean outcomes.
primitive='outputs/research_continue_20261008/pipeline_review/audit_primitives.py'
spec=importlib.util.spec_from_file_location('independent_r114_math',c.safe(primitive));manual=importlib.util.module_from_spec(spec);spec.loader.exec_module(manual)
def metric(v=100.):
    item={'overall_mae':v,'weighted_mae':v,'price_bands':[{'band':'x','rows':500,'mae':100.}]}
    for section in ('strata','forecast_quarters','months','cities','relative_forecast_months'):
        item[section]={'x':{'rows':500,'mae':100.}}
    return item
cases=0
for values in ((99.,99.,99.),(99.,100.,99.),(99.,100.1,99.),(100.3,99.,99.),(101.,101.,101.)):
    stages={s:{'metrics':{'v22':metric(),'T':metric(v)},'comparison':manual.manual_comparison(metric(),metric(v),'v22')}
            for s,v in zip(c.STAGES,values)}
    august=manual.manual_gates(stages)
    for extra_value in (99.,100.25,100.25001):
        extra_metrics={'v22':metric(),'T':metric(extra_value)}
        extra_comparison=manual.manual_comparison(extra_metrics['v22'],extra_metrics['T'],'v22')
        data={**stages,driver.OFFSET:{'metrics':extra_metrics,'comparison':extra_comparison}}
        actual=driver.combined_gates(data);offset_pass=extra_value<=100.*1.0025
        c.need(actual['combined_exploratory_submission']['passed']==(august['exploratory_submission_eligible'] and offset_pass)
               and actual['combined_promotion']['passed']==(august['strong_research_gate_passed'] and offset_pass),
               'Independent combined gates differ')
        cases+=1
c.verify_hashes(source);c.need(source=={**driver.sources(),driver.PLAN:c.sha(driver.PLAN)},'Source changed before issue')
payload={'status':'passed','created_at_utc':c.now(),'reviewer':'research_preflight_20261008',
         'source_sha256':source,'plan':c.record(driver.PLAN),'review_tool':c.record(__file__),
         'checks':c.record(FOLDER+'independent_preflight_checks.json'),'independent_guard_case_count':49,
         'independent_complete_arithmetic_combined_gate_cases':cases,'formal_CPU_selfchecks':formal,
         'root_actual_fit_only_replay':c.record(root_replay_path),
         'root_actual_fit_only_replay_execution':c.record(FOLDER+'root_context_preflight_command_execution.json'),
         'input_sha256_independently_reverified':inputs,'protected_unchanged':True,
         'new_fits':0,'GPU_work':False,'query_y_read':False,'blocking_findings_remaining':[],
         'reviewed_contracts':['One fixed additive linear-head parameterization, original random head weights and 30-epoch date-weighted total MAE.',
                               'Training raw total unprojected; float32 residual represented in float64, add saved F32 factor once, TWD once, then nonnegative projection once.',
                               'Every origin fit-only encoder/prior/dateweight/state is original and immutable; all four unconditional before any scoring.',
                               'Exact complete native checkpoint/signed/projected predictions/metadata folder and all five layers sealed before y.',
                               'Separate evaluation requires all four seals and successful training; fixed original v22/v24 controls and group gates.',
                               'Source/plan exact independent review plus formal CPU command/closure receipts required at freeze; no overwrites/automatic retry/fullfit/CSV/selection.'],
         'limits':['Actual four-origin fit-only context replay was executed by root (125.969 seconds internal), not independently repeated by this reviewer.',
                   'This reviewer independently read full trainer/driver/plan, executed49 artificial guard plus15 complete-arithmetic gate cases and rehashed all metadata-only dependencies.',
                   'The34 trainer and31 driver CPU checks were root formal executions; their exact commands/source closures/logs are verified.',
                   'No fitted R114 model or new accuracy result exists at review time. Postfit audit remains required.',
                   'Four reused overlapping origins and conditional bootstrap do not establish blind/Private performance or clean-environment end-to-end retraining.'],
         'review_elapsed_seconds':time.monotonic()-tick}
c.write(FOLDER+'preflight_review.json',payload)
print(json.dumps({'status':'passed','review':c.record(FOLDER+'preflight_review.json'),'source_count':len(source),
                  'inputs':len(inputs),'combined_gate_cases':cases},ensure_ascii=False))
