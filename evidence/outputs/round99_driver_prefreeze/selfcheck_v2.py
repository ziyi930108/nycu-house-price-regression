"""Synthetic X plus JSON-metadata R99 driver checks; no fit, labels or scoring."""
from pathlib import Path
import sys, json, inspect, ast, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import round99_contract as c
import round99_temporal_stack as driver
import round99_l1_stacker as helper

def main():
    checks=[];start=time.perf_counter()
    def check(name,ok):
        assert ok,name
        checks.append(name)
    def rejects(name,a):
        try:c.stacker_matrix(a)
        except ValueError:checks.append(name);return
        raise AssertionError(name)
    P=np.array([[1,9,2],[5,5,8],[0,0,0],[9,6,4]],dtype=np.float64)
    before=P.copy();Q=c.stacker_matrix(P)
    check('derived_dtype_shape',Q.dtype==np.float64 and Q.shape==(4,4))
    check('original_three_columns_bitwise',Q[:,:3].tobytes()==P.tobytes())
    check('median_ties_zero_order',Q[:,3].tobytes()==np.array([2,5,0,6],dtype=np.float64).tobytes())
    check('caller_input_unchanged',P.tobytes()==before.tobytes())
    check('derived_identity_four_columns',c.matrix_identity(Q)['shape']==[4,4])
    reversed_P=P[::-1];reversed_Q=c.stacker_matrix(reversed_P)
    check('noncontiguous_source_order_retained',reversed_Q.tobytes()==Q[::-1].tobytes())
    rejects('float32_rejected',P.astype(np.float32))
    rejects('integer_rejected',P.astype(np.int64))
    rejects('one_dimensional_rejected',P[:,0])
    rejects('two_columns_rejected',P[:,:2])
    rejects('already_four_columns_rejected',Q)
    bad=P.copy();bad[0,0]=np.nan;rejects('nonfinite_rejected',bad)
    bad=P.copy();bad[0,0]=-1;rejects('negative_rejected',bad)
    check('plan_bytes_match',c.sha(c.PLAN)==c.PLAN_SHA)
    check('original_preparation_bytes_match',c.sha(c.PREPARATION)==c.PREPARATION_SHA)
    prep=c.locked(c.PREPARATION)
    check('original_plan_not_rebound_to_R99',prep['plan_sha256']==c.PREPARATION_PLAN_SHA==c.sha(c.PREPARATION_PLAN) and c.PLAN_SHA!=c.PREPARATION_PLAN_SHA)
    check('original_branch_order',prep['branch_order']==list(c.BRANCHES))
    check('helper_four_branch_API',helper.parameters()['branch_order']==list(c.STACKER_BRANCHES))
    check('helper_fit_signature',tuple(inspect.signature(helper.fit_stacker).parameters)==('P','y','prediction_row_ids','target_row_ids'))
    check('helper_save_signature',tuple(inspect.signature(helper.save_artifacts).parameters)==('out','state','solver_arrays','metadata'))
    seal=c.locked(c.R98_SEALED_MANIFEST)
    check('R98_root_seal_SHA',c.sha(c.R98_SEALED_MANIFEST)==c.R98_SEALED_MANIFEST_SHA)
    needed={'meta_predictions','meta_y','meta_indices','meta_source_cache_indices','meta_trade_days','query_predictions','query_indices','query_source_cache_indices','query_trade_days','fit_indices','control_v22'}
    for stage in c.STAGES:
        ref=seal['stages'][stage]['stage_manifest']
        assert c.sha(ref['path'])==ref['sha256']
        old=c.locked(ref['path']);spec=prep['stages'][stage]
        check(stage+'_actual_readset_metadata',old['input_arrays']==spec['arrays'] and set(spec['arrays'])==needed and all(v['path']==f'outputs/round98_inputs/{stage}/{k}.npy' for k,v in spec['arrays'].items()))
        for name in ('state.json','fit_metadata.json','receipt.json'):
            path=f'outputs/round98/{stage}/model/{name}'
            assert c.sha(path)==old['model_artifact_sha256'][path]
        check(stage+'_plain_JSON_artifact_contract',True)
    for name in ('src/round99_contract.py','src/round99_temporal_stack.py'):
        ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
    check('contract_driver_AST_passed',True)
    tree=ast.parse((ROOT/'src/round99_temporal_stack.py').read_text(encoding='utf-8-sig'))
    saves=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='helper' and node.func.attr=='save_artifacts']
    check('exclusive_model_directory_saved_exactly_once',len(saves)==1)

    result={'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':c.sha('src/round99_temporal_stack.py'),'source_hashes':c.source_hashes('src/round99_temporal_stack.py'),'selfcheck_source_sha256':c.sha(__file__),'plan_sha256':c.PLAN_SHA,'parameters':helper.parameters(),'source_dataset_plan_sha256':c.PREPARATION_PLAN_SHA,'data_array_or_label_loads':0,'existing_model_JSON_metadata_only':True,'real_meta_fits':0,'query_score_calls':0,'GPU_calls':0,'elapsed_seconds':time.perf_counter()-start,'scope':'13 synthetic X-only matrix checks plus actual small JSON/API/source contracts; no solver invocation','author_role':'driver author; not a fully independent review','completed_at_utc':c.now()}
    c.write('reports/round99_driver_selfcheck_v2.json',result)
    print(json.dumps({'status':result['status'],'checks':len(checks),'elapsed_seconds':result['elapsed_seconds']}))
if __name__=='__main__':main()
