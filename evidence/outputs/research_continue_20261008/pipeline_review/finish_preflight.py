"""CPU/source-only one-time independent preflight receipt; never freeze or fit."""
from __future__ import annotations
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c

FOLDER=ROOT/'outputs/research_continue_20261008/pipeline_review'
DRIVER='src/round111_112_research.py'
TRAINER='src/round111_residual_prior_train.py'
HELPER='src/round112_ordered_unit.py'
PLAN='reports/round111_112_plan.md'
EXPECTED={DRIVER:'5eca9b030e0809490c7e218c3dc0c5b51c918f88d99b39c728de7d5d936eb85c',
          TRAINER:'318aae73904ceb8c02fdf42107bd80db4388d35717cdd123e5085f8ec298f0db',
          HELPER:'c8b6b64577b57c3b5c4e5ff8bde0accfe72866d1e5b6d4338746dd6ebaff3b73',
          PLAN:'3234fa23aeab952f335f4dbf66b14e012df38d5b601f7cee8f80dfc4cae76649'}


def load_tool(name):
    spec=importlib.util.spec_from_file_location(name,FOLDER/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    c.need(not c.safe('outputs/research_round111_112_20261008/request.json').exists(),'Review must precede request/fits')
    c.verify_hashes(EXPECTED)
    before={**c.source_hashes(DRIVER,TRAINER,HELPER),PLAN:c.sha(PLAN)}
    for name in before:
        if name.endswith('.py'):
            compile(c.safe(name).read_text(encoding='utf-8-sig'),name,'exec')
    primitives=load_tool('audit_primitives')
    primitive_checks=primitives.self_check()
    validators=load_tool('preflight_driver_selfcheck').run()
    c.need(validators['driver_sha256']==before[DRIVER],'Validator checked different driver')
    author_receipts={}
    for number,name in ((111,'independent_R111_cpu_checks_v2.json'),(112,'independent_R112_cpu_checks.json')):
        value=c.read(FOLDER/name);result=value['result']
        c.need(value['exit_code']==0 and result['status']=='passed','CPU check failed')
        if number==111:c.need(result['source_sha256']==before[TRAINER],'CPU checked obsolete NN source')
        else:c.verify_hashes(result['source_sha256'])
        author_receipts[str(number)]={'receipt':c.record(FOLDER/name),'check_count':result['check_count']}

    # Execute the actual NPZ-with-body using a fake object whose reads are counted.
    trainer_tree=ast.parse(c.safe(TRAINER).read_text(encoding='utf-8-sig'))
    run=next(n for n in trainer_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    block=next(n for n in run.body if isinstance(n,ast.With)
               and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
                       and i.context_expr.func.attr=='load' for i in n.items))
    function=ast.FunctionDef(name='check_npz_header',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='z')],
                vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=block.body,decorator_list=[])
    namespace={'need':c.need,'ARRAY_NAMES':c.ARRAY_NAMES}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),TRAINER,'exec'),namespace)
    class FakeNPZ:
        def __init__(self,names):self.files,self.reads=list(names),[]
        def __getitem__(self,key):self.reads.append(key);return 'synthetic'
    header_checks=[]
    good=FakeNPZ(c.ARRAY_NAMES);namespace['check_npz_header'](good)
    c.need(good.reads==list(c.ARRAY_NAMES),'Valid header reads wrong order')
    header_checks.append('valid_original_nine_keys')
    cases={'extra_query_y':list(c.ARRAY_NAMES)+['query_y'],
           'same_count_query_y_replacement':list(c.ARRAY_NAMES[:-1])+['query_y'],
           'duplicate_original_key':list(c.ARRAY_NAMES[:-1])+[c.ARRAY_NAMES[0]],
           'missing_original_key':list(c.ARRAY_NAMES[:-1])}
    for label,names in cases.items():
        bad=FakeNPZ(names)
        try:namespace['check_npz_header'](bad)
        except ValueError:c.need(not bad.reads,'Invalid NPZ values read before header rejection')
        else:raise ValueError('Invalid NPZ header accepted: '+label)
        header_checks.append(label+'_rejected_before_value_access')

    # Run the actual driver gates and actual immutable strong-gate body over
    # synthetic values; compare against independently implemented arithmetic.
    metric_tree=ast.parse(c.safe('src/round100_temporal_prior_evaluate_v2.py').read_text(encoding='utf-8-sig'))
    metric_fn=next(n for n in metric_tree.body if isinstance(n,ast.FunctionDef) and n.name=='research_gate')
    metric_ns={'STAGES':c.STAGES};exec(compile(ast.Module(body=[metric_fn],type_ignores=[]),'<actual strong gate>','exec'),metric_ns)
    driver_tree=ast.parse(c.safe(DRIVER).read_text(encoding='utf-8-sig'))
    gate_fn=next(n for n in driver_tree.body if isinstance(n,ast.FunctionDef) and n.name=='gates')
    gate_ns={'metric':SimpleNamespace(research_gate=metric_ns['research_gate'])}
    exec(compile(ast.Module(body=[gate_fn],type_ignores=[]),DRIVER,'exec'),gate_ns)
    gate_checks=[]
    values=[(99.,99.,99.),(99.,99.5,99.),(99.,99.50001,99.),(99.,100.,99.),
            (100.,100.,99.),(99.,100.00001,99.),(100.25,99.,99.),(100.25001,99.,99.)]
    for value in values:
        for adverse in (False,True):
            stages={s:{'metrics':{'v22':{'overall_mae':100.,'weighted_mae':100.},
                                    'T':{'overall_mae':v,'weighted_mae':v}},
                       'comparison':{'quarter_or_large_group_deterioration_above_5pct':[{'synthetic':True}] if adverse else []}}
                    for s,v in zip(c.STAGES,value)}
            actual=gate_ns['gates'](stages);manual=primitives.manual_gates(stages)
            c.need(actual['promotion']['components']==manual['strong_components'],'Strong gate arithmetic differs')
            c.need(actual['promotion']['adoption_research_gate_passed']==manual['strong_research_gate_passed'],'Strong verdict differs')
            c.need(actual['exploratory_submission']['components']==manual['components'],'Exploratory gate arithmetic differs')
            c.need(actual['exploratory_submission']['passed']==manual['exploratory_submission_eligible'],'Exploratory verdict differs')
            c.need(actual['selected_submission']=='v22' and not actual['automatic_selection'] and actual['private_score'] is None,
                   'Gate changed selection/private semantics')
            gate_checks.append({'mae':list(value),'adverse':adverse})

    # Metadata only: no NPZ payloads or query labels deserialized here.
    original_metadata={}
    authority=c.read('outputs/round63/results.json')
    labels=c.read('outputs/round100_inputs/manifest.json')
    for s in c.STAGES:
        path=f'outputs/round63/{s}/model/training.json'
        c.need(c.sha(path)==authority['stages'][s]['artifact_sha256'][path],'Original metadata authority differs')
        meta=c.read(path)
        c.need((meta['numeric_count'],len(meta['cardinalities']),meta['fit_rows'],meta['query_rows'])==(144,23,*c.SPLITS[s][2:]),
               'Original R63 dimensions/rows differ')
        c.need(set(labels['stages'][s]['arrays'])=={'fit_indices','fit_source_cache_indices','fit_y'},'Fit-only labels manifest changed')
        original_metadata[s]=c.record(path)

    facts={'status':'passed','source_sha256':before,'created_at_utc':c.now(),
           'compiled_python_sources':len(before)-1,'arithmetic_checks':primitive_checks,
           'artifact_validator_checks':validators,'author_CPU_checks_independently_executed':author_receipts,
           'actual_NPZ_header_checks':header_checks,'actual_driver_vs_manual_gate_checks':gate_checks,
           'original_R63_metadata':original_metadata,'model_fits':0,'optimizer_steps':0,'GPU_calls':0,
           'query_labels_loaded':False,'real_NPZ_payloads_loaded':False}
    c.need(before=={**c.source_hashes(DRIVER,TRAINER,HELPER),PLAN:c.sha(PLAN)},'Source mutated during review')
    c.write(FOLDER/'final_preflight_checks.json',facts)
    review={'status':'passed','created_at_utc':c.now(),'source_sha256':before,
            'scope':'Independent source/plan review plus CPU synthetic contracts before freeze; not actual housing fit/evaluation approval.',
            'checks':c.record(FOLDER/'final_preflight_checks.json'),'review_tool':c.record(__file__),
            'reviewed_launch_contract':c.record('outputs/research_round111_112_20261008/launch.py'),
            'workflow_review':c.record(FOLDER/'pipeline_review.json'),
            'blocking_findings':[],
            'formal_launcher_selfchecks_already_performed':False,
            'formal_launcher_selfchecks_required_before_freeze':True,
            'query_labels_loaded':False,'new_fits':0,'GPU_calls':0,'original_files_modified':False,
            'limits':['No actual new native model/checkpoint, stage seal or query score exists in this preflight.',
                      'CPU author self-check execution supplements independent source review; it is not independent ML accuracy evidence.',
                      'Full input-file SHA verification is required by freeze/stage/seal; this review did not reload original NPZ payloads.',
                      'Reused annual windows and saved-model controls do not remove selection or seed variation.',
                      'Postfit independent audit and separate fullfit/release preflight remain required.']}
    c.need(before=={**c.source_hashes(DRIVER,TRAINER,HELPER),PLAN:c.sha(PLAN)},'Source changed before issuing receipt')
    output='outputs/research_round111_112_20261008/preflight_review.json'
    c.write(output,review)
    print({'status':'passed','review':c.record(output),'source_records':len(before),
           'CPU_contract_checks':40+38+24+14+len(header_checks)+len(gate_checks),'new_fits':0})


if __name__=='__main__':main()
