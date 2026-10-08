"""R117 original R60 estimator with one fixed balanced transaction-fold policy.

No loader, raw-price table, query labels, saved prediction, NN or scorer. The
returned full-fit query state is the untouched R60 state. Its legacy group-fold
wording describes estimator provenance, never this module's fold assignment.
The independent evidence below is the transaction-fold authority.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import round60_recency_price_prior as original

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '4a549939ee688060252c30eb171528fa04496456a0a34d527f6430c247f8004f'
FOLD_POLICY = 'transaction-permutation-5-fold'
FOLD_RULE = 'per origin: p=np.random.default_rng(42).permutation(n); folds[p]=np.arange(n)%5'
LEVELS = copy.deepcopy(original.LEVELS)
ESTIMATOR_RULES = copy.deepcopy(original.RULES)
WEIGHT_RULES = copy.deepcopy(original.WEIGHT_RULES)
RULES = {**copy.deepcopy(ESTIMATOR_RULES), 'fold_hash': None,
         'fold_policy': FOLD_POLICY, 'fold_assignment': FOLD_RULE,
         'training': 'exclude complete held transaction fold; other folds may contain same project/batch',
         'same_project_exclusion': False, 'independent_transactions_claimed': False,
         'legacy_full_fit_state_rules_are_not_fold_evidence': True}
need = original.need
values_sha = original.values_sha
effective_area = original.original.effective_area


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), Path(original.__file__), Path(original.original.__file__),
             Path(original.original.grouping.__file__), ROOT / 'src/round19_unit_geo.py']
    return {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check_configuration():
    need(sha(original.__file__) == PARENT_SHA and original.LEVELS == LEVELS
         and original.RULES == ESTIMATOR_RULES and original.WEIGHT_RULES == WEIGHT_RULES,
         'Original R60 estimator/source/globals changed')
    need(RULES == {**copy.deepcopy(ESTIMATOR_RULES), 'fold_hash': None,
         'fold_policy': FOLD_POLICY, 'fold_assignment': FOLD_RULE,
         'training': 'exclude complete held transaction fold; other folds may contain same project/batch',
         'same_project_exclusion': False, 'independent_transactions_claimed': False,
         'legacy_full_fit_state_rules_are_not_fold_evidence': True}, 'Fixed R117 fold rules changed')


def transaction_folds(n):
    need(type(n) is int and n >= 5, 'Five balanced folds require at least five canonical fit rows')
    folds = np.empty(n, dtype=np.int64)
    folds[np.random.default_rng(42).permutation(n)] = np.arange(n, dtype=np.int64) % 5
    return folds


def state_content_sha(state):
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def predict_prior(state, Xquery, return_diagnostics=False):
    """Exact original full-fit query prediction; no query dates or labels."""
    check_configuration()
    return original.predict_prior(state, Xquery, return_diagnostics=return_diagnostics)


def fit_oof_prior(Xfit, yfit, source_indices):
    """Return OOF float64 TWD/m2, original full-fit state, int64 folds, evidence.

    Fixed canonical row order is part of the input identity. Labels and dates of
    a held fold cannot affect its own prior. Same-project references in other
    transaction folds are deliberately allowed; no chronological claim is made.
    """
    check_configuration()
    sources = source_hashes()
    keys = original.original._keys(Xfit)
    n = len(Xfit); indices = np.asarray(source_indices); dates = original._dates(Xfit)
    need(indices.dtype == np.int64 and indices.shape == (n,) and n >= 5
         and (indices >= 0).all() and (np.diff(indices) > 0).all(),
         'Sorted unique nonnegative canonical int64 source indices required')
    if isinstance(yfit, pd.Series): need(yfit.index.equals(Xfit.index), 'Fit target Series index differs')
    y = np.asarray(yfit, dtype=np.float64)
    need(y.shape == (n,) and np.isfinite(y).all() and (y > 0).all(), 'Positive finite aligned fit labels only')
    area = effective_area(Xfit)
    with np.errstate(over='raise', invalid='raise', divide='raise'): units = y / area
    need(np.isfinite(units).all() and (units > 0).all(), 'Positive finite fit unit prices')
    groups, fallback = original.original.grouping.cluster_keys(Xfit, indices)
    folds = transaction_folds(n)
    oof = np.empty(n, dtype=np.float64); records = []
    def support(st):
        return [{k: level[k] for k in ('columns','cell_count','minimum_observed_support')} for level in st['levels']]
    for fold in range(5):
        held = folds == fold; reference = ~held
        need(held.any() and reference.any(), 'Balanced transaction folds must have nonempty held/reference rows')
        state = original._build(keys, units, dates, reference, 'complement_fit_for_held_fold')
        oof[held], counts = original._predict(state, [key[held] for key in keys], True)
        records.append({'fold': fold, 'held_rows': int(held.sum()), 'query_rows': int(held.sum()),
            'fit_rows': int(reference.sum()), 'reference_rows': int(reference.sum()),
            'held_groups': int(len(np.unique(groups[held]))), 'reference_groups': int(len(np.unique(groups[reference]))),
            'same_groups_may_cross_transaction_folds': True,
            'reference_indices_sha256': values_sha(indices[reference], '<i8'),
            'reference_target_sha256': values_sha(y[reference]), 'reference_unit_sha256': values_sha(units[reference]),
            'reference_global_median': state['global_median'], 'date_support': state['date_support'],
            'reference_level_support': support(state), 'held_prior_level_counts': counts})
    need(np.isfinite(oof).all() and (oof > 0).all(), 'Finite positive OOF prior')
    # Exact original full-fit state and its original internal RULES; no rewriting.
    full = original._build(keys, units, dates, np.ones(n, dtype=bool), 'full_fit_for_query')
    evidence = {'rules': copy.deepcopy(RULES), 'legacy_estimator_rules': copy.deepcopy(ESTIMATOR_RULES),
        'legacy_internal_group_wording_is_not_this_fold_evidence': True,
        'legacy_full_fit_state_unmodified': True, 'fold_policy': FOLD_POLICY, 'fold_assignment': FOLD_RULE,
        'fold_assignment_uses_labels_dates_groups_or_query': False, 'fold_assignment_identical_to_R52_rule': False,
        'fit_rows': n, 'groups': int(len(np.unique(groups))), 'fallback_counts': fallback,
        'folds': records, 'full_state_level_support': support(full), 'full_fit_date_support': full['date_support'],
        'full_fit_global_median': full['global_median'],
        'training_prior_role': 'transaction-fold OOF; complement fit labels/dates only; same-project references allowed',
        'query_prior_role': 'unchanged original R60 full fit state', 'uses_fit_labels': True,
        'uses_query_labels': False, 'chronological_oof': False, 'independent_OOF_claimed': False,
        'fit_target_sha256': values_sha(y), 'effective_area_sha256': values_sha(area), 'fit_unit_sha256': values_sha(units),
        'source_indices_sha256': values_sha(indices, '<i8'), 'fold_ids_sha256': values_sha(folds, '<i8'),
        'oof_prior_sha256': values_sha(oof),
        'group_keys_sha256': hashlib.sha256(json.dumps(groups.tolist(), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'state_content_sha256': state_content_sha(full), 'source_sha256': sources,
        'full_fit_and_query_original_exact_requires_driver_saved_state_comparison': True,
        'date_weights_rebuilt_in_each_complement_by_original_R60': True,
        'R115_full_outer_dateweights_used_for_prior_median': False}
    need(source_hashes() == sources, 'Prior sources changed during construction')
    check_configuration()
    return oof, full, folds, evidence


def self_check():
    """Artificial prices/dates only; no real prior build, NPZ or training."""
    checks = []
    def mark(name, condition): need(bool(condition), name); checks.append(name)
    def reject(name, call):
        try: call()
        except (ValueError, KeyError, TypeError): checks.append(name)
        else: raise AssertionError('Accepted invalid fixture: ' + name)
    parent_before = (copy.deepcopy(original.RULES), copy.deepcopy(original.WEIGHT_RULES),
                     copy.deepcopy(original.LEVELS), copy.deepcopy(original.original.RULES), source_hashes())
    n = 360
    X = pd.DataFrame({'project_name':[f'artificial_project_{i//9}' for i in range(n)],
        'address_city':['A']*180+['B']*180, 'address_district':['D']*n, 'transaction_type':['house']*n,
        'building_type':['apartment']*n, 'address_raw':['__MISSING__']*n, 'address_road':['__MISSING__']*n,
        'address_section':['__MISSING__']*n, 'building_area':np.full(n,100,np.float32), 'parking_area':np.zeros(n,np.float32),
        'trade_day_index':np.arange(n,dtype=np.float64)+1000, 'trade_date_invalid':np.zeros(n), 'trade_date_out_of_range':np.zeros(n)})
    y = pd.Series((np.arange(n,dtype=np.float64)+100)*100, index=X.index)
    indices = np.arange(1000,1000+n,dtype=np.int64)
    before, ybefore, ibefore = X.copy(deep=True), y.copy(), indices.copy()
    oof,full,folds,evidence = fit_oof_prior(X,y,indices)
    expected = np.empty(n,np.int64); expected[np.random.default_rng(42).permutation(n)] = np.arange(n)%5
    mark('exact_single_seed42_permutation_rule', folds.tobytes()==expected.tobytes())
    mark('balanced_fold_sizes', max(np.bincount(folds))-min(np.bincount(folds))<=1)
    mark('same_projects_may_span_folds', any(len(set(folds[i:i+9]))>1 for i in range(0,n,9)))
    mark('evidence_exact_transaction_authority_not_legacy_group_words', evidence['fold_policy']==FOLD_POLICY
         and evidence['legacy_internal_group_wording_is_not_this_fold_evidence']
         and evidence['fold_assignment_identical_to_R52_rule'] is False)
    held = folds==0; revisedX=X.copy(deep=True); revisedY=y.copy()
    revisedX.loc[held,'trade_day_index']+=3650; revisedY.loc[held]*=17
    revised,_,revisedfolds,revised_evidence=fit_oof_prior(revisedX,revisedY,indices)
    mark('held_fold_label_and_date_invariance', oof[held].tobytes()==revised[held].tobytes()
         and folds.tobytes()==revisedfolds.tobytes() and evidence['folds'][0]==revised_evidence['folds'][0])
    _,oldfull,oldfolds,_=original.fit_oof_prior(X,y,indices)
    mark('original_full_fit_state_content_exact', full==oldfull and state_content_sha(full)==state_content_sha(oldfull))
    mark('transaction_policy_differs_from_project_hash', folds.tobytes()!=oldfolds.tobytes())
    query=X.iloc[[0,359,23]].drop(columns=list(original.DATE_COLUMNS)).copy()
    qp=predict_prior(full,query)
    mark('query_fullfit_original_exact_and_no_date_required', qp.tobytes()==original.predict_prior(oldfull,query).tobytes())
    restored=json.loads(json.dumps(full,allow_nan=False))
    mark('JSON_fullstate_query_reload_and_order_exact', predict_prior(restored,query).tobytes()==qp.tobytes()
         and predict_prior(full,query.iloc[::-1]).tobytes()==qp[::-1].tobytes())
    query.loc[:,['address_city','address_district','transaction_type','building_type']]='unknown'
    mark('original_global_fallback', np.array_equal(predict_prior(full,query),np.full(3,full['global_median'])))
    for fold in range(5):
        reference=folds!=fold
        _,_,s=original._weights(tuple(v[reference] for v in original._dates(X)))
        mark('complement_specific_original_dateweight_'+str(fold),evidence['folds'][fold]['date_support']==s)
    mark('exact_half_weighted_median_rule_preserved',original.weighted_median([1.,9.],[1.,1.])==5.)
    w,_,s=original._weights((np.array([1000.,1365.,np.nan]),np.array([0.,0.,1.]),np.zeros(3)))
    mark('invalid_date_neutral1_original_normalization',np.array_equal(w,np.array([2/3,4/3,1.])) and s['neutral_date_rows']==1)
    small=X.iloc[:31].copy();small['trade_day_index']=0.;small.loc[30,'address_district']='rare'
    st=original._build(original.original._keys(small),np.arange(1.,32.),original._dates(small),np.ones(31,bool),'full_fit_for_query')
    sk=original.original._keys(small)
    mark('raw30_cell_support_and_raw29_global_rule_unchanged',st['levels'][0]['cells'][sk[0][0]]['rows']==30
         and sk[0][30] not in st['levels'][0]['cells'])
    tiny=small.iloc[:29].copy();tinykeys=original.original._keys(tiny)
    tiny_state=original._build(tinykeys,np.arange(1.,30.),original._dates(tiny),np.ones(29,bool),'full_fit_for_query')
    mark('global_reference29_not_subject_to_cell30_threshold',tiny_state['global_raw_rows']==29
         and all(level['cell_count']==0 for level in tiny_state['levels'])
         and np.array_equal(predict_prior(tiny_state,tiny),np.full(29,tiny_state['global_median'])))
    for name,call in [('less_than_five_rows',lambda:transaction_folds(4)),
                      ('duplicate_canonical_indices',lambda:fit_oof_prior(X,y,np.zeros(n,np.int64))),
                      ('unordered_canonical_indices',lambda:fit_oof_prior(X,y,indices[::-1])),
                      ('wrong_index_dtype',lambda:fit_oof_prior(X,y,indices.astype(np.int32))),
                      ('zero_label',lambda:fit_oof_prior(X,np.zeros(n),indices)),
                      ('misaligned_label_series',lambda:fit_oof_prior(X,y.iloc[::-1],indices))]: reject(name,call)
    mark('inputs_labels_and_indices_unchanged',X.equals(before) and y.equals(ybefore) and np.array_equal(indices,ibefore))
    mark('original_parent_globals_and_sources_unchanged',parent_before==(original.RULES,original.WEIGHT_RULES,
         original.LEVELS,original.original.RULES,source_hashes()))
    return {'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':source_hashes(),
            'fold_policy':FOLD_POLICY,'real_data_read':False,'true_prior_builds':0,
            'synthetic_OOF_prior_build_calls':3,'synthetic_direct_reference_state_build_calls':2,
            'query_labels_read':False,'NN_fits':0,'GPU_calls':0,'new_CSV':False,
            'author_selfcheck_not_independent_preflight':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true',required=True)
    parser.parse_args();print(json.dumps(self_check(),ensure_ascii=False,sort_keys=True,allow_nan=False))
