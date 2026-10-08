"""Independent R117 recovery v2 source/metadata/CPU-contract preflight; no true prior/fit.

The root's actual four-context replay is attributed to root, not repeated or
misrepresented as independent execution here. The independent arithmetic
oracle uses artificial labels and dates only. Exclusive review write.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import round100_contract as c
import round117_research_v2 as driver

FOLDER='outputs/research_round117_v2_20261008/'
ORIGINAL_ORACLE='outputs/research_round117_20261008/'
SOURCE_AUDIT=FOLDER+'independent_source_review.json'
IMPLEMENTATION_PARENT={'path':'outputs/research_round117_20261008/independent_preflight.py','sha256':'5e53733dee15c75a01cda8b8bd624c942ad761561ac8e8da33344f51ac71aca0'}


def same(a,b,message):c.need(a==b,message)


def locked(name):
    path=c.safe(name)
    same(path.with_suffix('.sha256').read_text(encoding='ascii').strip(),c.sha(path),'Sidecar exact '+name)
    return c.read(path),c.record(path)


def deny(*args,**kwargs):raise AssertionError('Real array/pickle/prior/fit access forbidden in independent preflight')


def parsed(path):return ast.parse(c.safe(path).read_text(encoding='utf-8-sig'))


def training_ast_contract():
    old=next(n for n in parsed('src/round63_recency_weighted_prior_train.py').body if isinstance(n,ast.FunctionDef) and n.name=='run')
    new=next(n for n in parsed(driver.TRAINER).body if isinstance(n,ast.FunctionDef) and n.name=='_run_once')
    def epoch(f):
        candidates=[n for n in f.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
        same(len(candidates),1,'One fixed epoch loop')
        node=copy.deepcopy(candidates[0])
        # Exception wording is descriptive only; actual tests and raise type stay.
        for x in ast.walk(node):
            if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call):
                x.exc.args=[ast.Constant(value='error-message') if isinstance(v,ast.Constant) and isinstance(v.value,str) else v for v in x.exc.args]
        return ast.dump(node,include_attributes=False)
    same(epoch(old),epoch(new),'Actual entire original R63 epoch arithmetic/RNG/batches/gradient/schedule AST')
    assignments={}
    for key in ('optimizer','schedule','model'):
        def find(f):
            nodes=[n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)]
            same(len(nodes),1,'Single original training assignment '+key)
            return ast.dump(nodes[0],include_attributes=False)
        same(find(old),find(new),'Exact original training assignment AST '+key);assignments[key]=True
    return {'entire_epoch_loop_AST_equal_except_error_message':True,'optimizer_schedule_model_constructor_AST_equal':assignments,
            'initial_bias_formula_source_reviewed_unchanged_but_value_changes_with_factor':True,
            'new_checkpoint_guards_and_descriptive_provenance_are_engineering_changes':True}


def scoring_guards():
    checks=[];events=[]
    seal={'status':'four_priors_models_and_layers_sealed_unscored','request':{},'origin_order':list(driver.ORIGINS),
          'records':{s:{} for s in driver.ORIGINS},'new_fits':4,'new_prior_builds':4,'query_scores_computed':False}
    execution={'exit_code':0,'failure':None,'request':{}}
    def validate(origin,*args):events.append(origin);return {},{}
    with patch.object(driver,'request',return_value={'input_sha256':{c.RAW:driver.RAW_SHA}}), \
         patch.object(c,'read',side_effect=lambda name:seal if name==driver.SEAL else execution), \
         patch.object(c,'record',return_value={}),patch.object(c,'sha',return_value=driver.RAW_SHA), \
         patch.object(driver,'validate',side_effect=validate):
        driver.scoring_preflight();same(events,list(driver.ORIGINS),'All four fixed origins checked before labels');checks.append('all_four_before_raw')
        for name,mutate,restore in (
            ('failed_training',lambda:execution.update(exit_code=1),lambda:execution.update(exit_code=0)),
            ('scored_seal',lambda:seal.update(query_scores_computed=True),lambda:seal.update(query_scores_computed=False)),
            ('missing_offset',lambda:seal['records'].pop(driver.OFFSET),lambda:seal['records'].update({driver.OFFSET:{}})),
            ('wrong_prior_build_count',lambda:seal.update(new_prior_builds=3),lambda:seal.update(new_prior_builds=4))):
            events.clear();mutate()
            try:driver.scoring_preflight()
            except (ValueError,KeyError):same(events,[],'Rejected before stage/label work '+name);checks.append(name)
            else:raise AssertionError('Scoring preflight accepted '+name)
            finally:restore()
    evaluate=next(n for n in parsed('src/round117_research_v2.py').body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    def calls(name):return [n.lineno for n in ast.walk(evaluate) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
    preflight,reader=calls('scoring_preflight'),calls('pd.read_pickle')
    same(len(reader),1,'Exactly one raw-price reader in scorer');c.need(min(preflight)<reader[0],'Guard before raw reader AST')
    return {'independent_injected_guard_cases':checks,'guard_calls_before_only_raw_reader_AST':True}


def main():
    tick=time.monotonic();output=FOLDER+'preflight_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(),'Do not overwrite preflight')
    closure={**driver.sources(),driver.PLAN:c.sha(driver.PLAN),driver.RECOVERY_PLAN:c.sha(driver.RECOVERY_PLAN)};protected=driver.protected()
    same(len(closure),53,'51 Python plus unchanged science and fixed engineering plans')
    same(closure[driver.RECOVERY_PLAN],driver.RECOVERY_PLAN_SHA,'Exact recovery amendment')
    source_audit,source_audit_record=locked(SOURCE_AUDIT)
    same(source_audit['status'],'passed_bounded_independent_source_audit','Independent minimal source review passed')
    same(source_audit['source_sha256'],closure,'Independent actual source/lineage binding')
    same(source_audit['independent_dynamic_guard_count'],8,'All actual missing/wrong-SHA lineage guards')
    same(source_audit['independent_AST_mutation_guard_count'],5,'Actual AST mutation guards')
    same(c.record(IMPLEMENTATION_PARENT['path']),IMPLEMENTATION_PARENT,'Immutable original preflight implementation')
    same(closure[driver.PLAN],driver.PLAN_SHA,'Fixed before R116 score plan SHA')
    deps={p:c.sha(p) for p in (c.rel(__file__),ORIGINAL_ORACLE+'independent_prior_oracle.py',FOLDER+'independent_source_check.py')}
    with patch.object(pd,'read_pickle',side_effect=deny),patch.object(np,'load',side_effect=deny), \
         patch.object(driver.prior,'fit_oof_prior',side_effect=deny),patch.object(driver.prior.original,'_build',side_effect=deny):
        spec=driver.specifications()
        actual_author_guards=driver.self_check()
        same(actual_author_guards['status'],'passed','Source-level CPU contract replay')
        ast_contract=training_ast_contract();guard_contract=scoring_guards()
    oracle,oracle_record=locked(ORIGINAL_ORACLE+'independent_prior_oracle_review.json')
    oracle_execution,oracle_execution_record=locked(ORIGINAL_ORACLE+'independent_prior_oracle_execution.json')
    same(oracle['status'],'passed','Independent manual oracle passed');same(oracle['check_count'],27,'All independent oracle cases')
    c.need(oracle['oracle_uses_original_key_weight_or_median_helpers'] is False
           and oracle['real_arrays_or_query_y_read'] is False and oracle['real_prior_builds']==0
           and oracle['new_model_fits']==0 and oracle['GPU_work'] is False,'Independent oracle scope')
    c.verify_hashes(oracle['source_sha256']);same(oracle_execution['exit_code'],0,'Oracle actual saved execution')
    same(oracle_execution['review'],oracle_record,'Oracle receipt/review identity')
    root_replay,root_replay_record=locked(FOLDER+'root_context_preflight.json')
    same(root_replay['status'],'passed','Root actual context replay passed')
    same(root_replay['source_sha256'],closure,'Root replay exact complete source+plan')
    same(root_replay['input_sha256'],spec['input_sha256'],'Root replay exact frozen input set')
    same(root_replay['review_tool'],c.record(FOLDER+'replay_preflight.py'),'Root replay implementation identity')
    same(root_replay['R115_postfit'],spec['R115_postfit'],'R115 independent completed prerequisite')
    same(root_replay['engineering_recovery'],spec['engineering_recovery'],'Root replay exact original-abort and fresh-recovery identities')
    same(root_replay['origin_order'],list(driver.ORIGINS),'Root replay all fixed origins')
    same(set(root_replay['origins']),set(driver.ORIGINS),'Every root context replay present')
    c.need(root_replay['protected_unchanged'] and root_replay['saved_fit_labels_read']
           and root_replay['raw_price_pickle_deserialized'] is False and root_replay['query_prices_read'] is False
           and root_replay['new_CSV'] is False,'Root replay label/scope guard')
    for key in ('new_prior_builds','new_fits','GPU_calls','new_prepared_arrays','native_model_inference_calls'):
        same(root_replay[key],0,'Root replay no new work '+key)
    for origin,row in root_replay['origins'].items():
        same((row['fit_rows'],row['query_rows']),tuple(driver.COUNTS[origin]),'Actual origin row counts')
        c.need(row['original_full_fit_query_prior_and_both_factors_exact'] and row['original_v22_v24_and_prior_slot_replayed_exact'],
               'Original state/prior/factor/whole chain replay')
        same(set(row['original_prepared_array_identities']),set(c.ARRAY_NAMES),'Original exact nine array identities')
    checks={}
    for name,(entry,environment,flag) in driver.formal_checks().items():
        value,rec=locked(FOLDER+name);same(value['exit_code'],0,'Formal selfcheck successful')
        same(value['command'],[str(ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],'Formal exact command')
        same(value['source_sha256'],c.source_hashes(entry),'Formal exact source closure');checks[name]=rec
    same(driver.sources(),{k:v for k,v in closure.items() if k not in (driver.PLAN,driver.RECOVERY_PLAN)},'Source stable during independent review')
    c.verify_hashes(closure);c.verify_hashes(spec['input_sha256']);c.verify_hashes(deps)
    same(driver.protected(),protected,'Protected original source/models/predictions/CSV unchanged')
    c.need(not c.safe(driver.BASE).exists(),'No actual R117 preparation/fits before complete preflight')
    payload={'status':'passed','reviewer':'research_resume_audit_20261008','round':117,'created_at_utc':c.now(),
        'source_sha256':closure,'input_sha256':spec['input_sha256'],'plan':c.record(driver.PLAN),
        'review_tool':c.record(__file__),'review_dependency_source_sha256':deps,
        'independent_source_review':source_audit_record,'implementation_parent':IMPLEMENTATION_PARENT,
        'recovery_plan':c.record(driver.RECOVERY_PLAN),'engineering_recovery':spec['engineering_recovery'],
        'original_scientific_prior_oracle_reused_from_original_attempt':True,
        'independent_prior_oracle':oracle_record,'independent_prior_oracle_execution':oracle_execution_record,
        'independent_artificial_prior_oracle_checks':27,'training_AST_contract':ast_contract,'independent_scoring_guard_contract':guard_contract,
        'source_CPU_contract_replay_check_count':actual_author_guards['check_count'],
        'formal_selfcheck_execution':checks,'root_actual_context_replay':root_replay_record,
        'root_context_replay_independently_repeated':False,'R115_postfit':spec['R115_postfit'],
        'protected_before':protected,'protected_unchanged':True,
        'fit_query_isolation_reviewed':True,'all_four_priors_models_layers_sealed_before_query_y_guard_reviewed':True,
        'original_eight_arrays_and_fullfit_query_state_prior_factor_exact_guard_reviewed':True,
        'fit_factor_and_linked_bias_change_together_explicit':True,'single_fixed_transaction_five_fold_policy':True,
        'unmodified_original_R63_ML_math_and_RNG_loop_AST_verified':True,
        'prior_weights_complement_specific_original_R60_not_R115_full_outer':True,
        'whole_chain_primary_v22_and_secondary_v24_gate_arithmetic_and_limits_reviewed':True,
        'blocking_findings_remaining':[],'new_model_fits':0,'new_prior_builds':0,'GPU_work':False,
        'real_arrays_or_query_y_read_by_independent_preflight':False,'new_prepared_arrays':0,
        'Public_score':None,'Private_score':None,'selected_submission':'v22','new_CSV':False,
        'limits':['The prior oracle is the original 27-case independent receipt, not newly rerun; prior/trainer/scientific methods are unchanged. Root four-context replay is new but attributed to root, not this reviewer.',
            'Original failed one-native attempt is preserved and hash-bound; this preflight allows four fresh prior/NN fits in new paths only after coordinator freeze.',
            'Source/metadata/independent artificial arithmetic and root-attributed real context replay are preflight evidence; actual new prior build/NN fit/accuracy is not yet performed.',
            'Held whole transaction-fold labels/dates are excluded, but other same-project peers may enter; reused overlapping windows and selection bias remain.',
            'Only fit factor changes among nine numeric arrays, with linked ratio median/bias and truthful descriptive provenance changes; not a pure causal loss-only experiment.',
            'Real root context replay was checked through exact source/input/receipt identities; independent agent did not repeat all real contexts.',
            'No full-data fit/CSV/release/Kaggle selection follows automatically from preflight pass.'],
        'review_elapsed_seconds':time.monotonic()-tick}
    c.write(output,payload);print(json.dumps({'status':'passed','review':c.record(output),'source_count':len(closure),'input_count':len(spec['input_sha256'])}))


if __name__=='__main__':main()
