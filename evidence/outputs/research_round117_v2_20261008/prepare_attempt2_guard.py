"""CPU-only attempt2 path/import/schema guard; no model/data array loading."""
import ast,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import round100_contract as c
FOLDER='outputs/research_round117_v2_20261008/'
TOOL=FOLDER+'verify_completed_v2_roots_attempt2.py'
NATIVE=FOLDER+'audit_model_identity_v2.py'
old=c.safe(FOLDER+'verify_completed_v2_roots.py').read_text(encoding='utf8');new=c.safe(TOOL).read_text(encoding='utf8')
c.need(old.count("FOLDER+'audit_model_identity.py'")==1 and new==old.replace("FOLDER+'audit_model_identity.py'","FOLDER+'audit_model_identity_v2.py'"),'Exactly one dependency literal changed')
module_spec=importlib.util.spec_from_file_location('audit_attempt2_guard',c.safe(TOOL));module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
assign=next(x for x in ast.walk(ast.parse(new)) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='deps' for t in x.targets))
keys=eval(compile(ast.Expression(assign.value.generators[0].iter),'<dependency-path-expression>','eval'),{'c':c,'__file__':str(c.safe(TOOL)),'FOLDER':FOLDER,'PRIMITIVE':module.PRIMITIVE})
c.need(tuple(keys)==(TOOL,NATIVE,module.PRIMITIVE),'Actual dependency expression resolves all intended modules')
dependency_sha256={name:c.sha(name) for name in keys}
c.need(all(callable(getattr(module,k,None)) for k in ('archive_arrays','check_stage','check_native','run','self_check')),'Audit imports/callables')
c.need(all(callable(getattr(module.native_audit,k,None)) for k in ('checkpoint_cpu','inspect_payload','self_check')),'Native imported callables')
c.need(c.rel(module.native_audit.__file__)==NATIVE,'Actual imported native module path')
import round117_research_v2 as driver
names=[driver.REQUEST,driver.SEAL,FOLDER+'training_execution.json',FOLDER+'evaluation_execution.json',FOLDER+'preflight_review.json',driver.RESULTS,module.ORACLE_REVIEW]
names += [FOLDER+n for n in driver.formal_checks()]
locked={name:module.locked(name)[1] for name in names}
request,rr=module.locked(driver.REQUEST);seal,sr=module.locked(driver.SEAL);training,tr=module.locked(FOLDER+'training_execution.json');evaluation,er=module.locked(FOLDER+'evaluation_execution.json')
module.same((training['exit_code'],training['failure'],training['request']),(0,None,rr),'Actual completed train schema')
module.same((evaluation['exit_code'],evaluation['failure'],evaluation['request'],evaluation['seal'],evaluation['training_execution']),(0,None,rr,sr,tr),'Actual completed eval schema')
source_before={**dependency_sha256,**driver.sources(),driver.PLAN:c.sha(driver.PLAN),driver.RECOVERY_PLAN:c.sha(driver.RECOVERY_PLAN)}
checks={};commands={'audit_30_CPU_guards':[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(c.safe(TOOL)),'--self-check'],'native_artificial_checkpoint_guards':[str(ROOT/'.venv_torch/Scripts/python.exe'),'-B','-X','utf8',str(c.safe(NATIVE)),'--self-check']}
for name,command in commands.items():
    log=FOLDER+'attempt2_'+name+'.log';tick=time.monotonic()
    with c.safe(log).open('x',encoding='utf8',newline='\n') as stream:
        result=subprocess.run(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT)
    elapsed=time.monotonic()-tick
    c.need(result.returncode==0,name+' failed: preserve log')
    payload=json.loads(c.safe(log).read_text(encoding='utf8'))
    c.need(payload['status']=='passed' and payload['GPU_work'] is False and payload['new_fits']==0,'Artificial CPU scope')
    if name=='audit_30_CPU_guards':c.need(payload['check_count']==30 and payload['real_arrays_or_query_y_read'] is False,'Thirty actual guard cases')
    else:c.need(payload['real_model_or_data_read'] is False and len(payload['rejected_checkpoint_corruption_cases'])==4,'Synthetic native corruption guards')
    checks[name]={'command':command,'exit_code':result.returncode,'elapsed_seconds':elapsed,'log':c.record(log),'check_count':payload.get('check_count'),'rejected_checkpoint_corruption_cases':payload.get('rejected_checkpoint_corruption_cases')}
c.verify_hashes(source_before)
out=FOLDER+'postfit_attempt2_guard_review.json'
c.write(out,{'status':'passed','tool':c.record(TOOL),'native_audit':c.record(NATIVE),'launcher':c.record(FOLDER+'launch_independent_postfit_attempt2.py'),'guard_tool':c.record(__file__),'source_sha256':source_before,'dependency_paths_and_sha256':dependency_sha256,'actual_receipt_sidecars_verified':locked,'exact_one_literal_fix':True,'actual_imported_callables_checked':True,'CPU_checks':checks,'new_model_fits':0,'GPU_work':False,'query_labels_or_arrays_read':False,'real_checkpoints_loaded':False,'formal_attempt2_executed':False,'failed_attempt1':c.record(FOLDER+'independent_postfit_execution.json')})
print(json.dumps({'status':'passed','review':c.record(out),'CPU_checks':checks},ensure_ascii=False))
