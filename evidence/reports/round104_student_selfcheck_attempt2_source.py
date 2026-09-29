"""R104 matched 10-key student preparation from frozen forward teachers only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = '106fda0c56cc7c5660c6255bb572fd78c59ccac9c13d25604bbe592204e43778'
R102_PLAN_SHA = 'ee59dba0da8abb79c88cb6e1b73a6417a91944de784f3c7f77e61755a68bc355'
R102_ROOT = {'path':'outputs/round102/sealed_manifest.json','sha256':'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
WINDOWS = {'pre_auxiliary': ('2020-08-11', '2021-08-10'), 'auxiliary': ('2021-08-11', '2022-08-10'),
           'development': ('2022-08-11', '2023-08-10'), 'confirmation': ('2023-08-11', '2024-08-10')}
TEACHERS = {'auxiliary': ('pre_auxiliary',), 'development': ('pre_auxiliary', 'auxiliary'),
            'confirmation': ('pre_auxiliary', 'auxiliary', 'development')}
TRAIN_KEYS = ('train_num', 'train_cat', 'train_y_million', 'train_price_factor_million', 'train_sample_weight')
QUERY_KEYS = ('query_num', 'query_cat', 'query_price_factor_million', 'cardinalities')
NINE_KEYS = TRAIN_KEYS + QUERY_KEYS
OFFSET = 'train_base_offset_million'
SOURCE_KEYS = {'stage_seal', 'prepared', 'fit_indices', 'query_indices', 'eligible_positions', 'features', 'canonical_mapping'}


def need(ok, message):
    if not ok: raise ValueError(message)
def day(s): return (pd.Timestamp(s)-pd.Timestamp('2000-01-01')).days
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(a):
    a = np.asarray(a)
    return {'dtype': str(a.dtype), 'shape': list(a.shape), 'values_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
def safe(path):
    p = Path(path); p = (ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT), 'Path outside workspace'); return p
def record(path):
    p = safe(path); return {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)}
def write(path, obj):
    with path.open('x', encoding='utf8', newline='\n') as f: json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    with path.with_suffix('.sha256').open('x', encoding='ascii') as f: f.write(sha(path)+'\n')
def save_array(path, a):
    with path.open('xb') as f: np.save(f, a, allow_pickle=False)
    with path.open('rb') as f: need(identity(np.load(f, allow_pickle=False)) == identity(a), 'Saved mapping roundtrip differs')
    return {**record(path), **identity(a)}
def source_hashes(): return {'src/round104_student_preparation.py': sha(__file__)}
def parameters():
    return {'round':104, 'plan_sha256':PLAN_SHA, 'source':'R102 T only, own stage',
        'teacher_origins': {k:list(v) for k,v in TEACHERS.items()}, 'numeric_count':145,
        'exact_keys':list(NINE_KEYS)+[OFFSET], 'offset_formula':'float32((.375*teacher_area_TWD+.25*teacher_lgb_TWD)/1e6)',
        'sample_weights_renormalized':False, 'selection_uses_labels_or_errors':False,
        'query_four_arrays_unchanged':True, 'geography_is_historically_asof':False}


def indices(a, upper, name, nonempty=True):
    a = np.asarray(a)
    need(a.dtype == np.int64 and a.ndim == 1 and (len(a)>0 or not nonempty)
         and np.all((a >= 0) & (a < upper)) and np.all(np.diff(a)>0), 'Invalid sorted canonical/index positions: '+name)
    return a


def collect_input_records(stage, source_records, teacher_records):
    """Metadata-only exact readset; hash documents, declare arrays without opening.

    Caller freezes the returned mapping, including the completed pre-aux teacher
    seal, before a stage preparation. No raw labels, arrays or NPZ are loaded here.
    """
    need(stage in TEACHERS and set(source_records)==SOURCE_KEYS and set(teacher_records)==set(TEACHERS[stage]), 'Unexpected input record roles')
    inputs={}
    def add(rec, expected=None):
        need(isinstance(rec,dict) and {'path','sha256'}<=set(rec),'Invalid input record')
        name=safe(rec['path']).relative_to(ROOT).as_posix();h=rec['sha256']
        need(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid SHA')
        need(expected is None or name==expected,'Unexpected input role path')
        need(name not in inputs or inputs[name]==h,'Conflicting input identity');inputs[name]=h;return ROOT/name
    def doc(rec, expected=None):
        p=add(rec,expected);need(sha(p)==rec['sha256'],'Bound document changed');return json.loads(p.read_text(encoding='utf-8-sig'))
    p=add({'path':'reports/round104_plan.md','sha256':PLAN_SHA});need(sha(p)==PLAN_SHA,'R104 plan changed')
    root=doc(R102_ROOT)
    need(root['status']=='all_prior_input_predictions_sealed' and root['plan_sha256']==R102_PLAN_SHA and root['all_query_reload_exact'],'R102 authority changed')
    need(source_records['stage_seal']==root['stages'][stage]['stage_manifest'],'Source stage not bound by R102 root')
    source=doc(source_records['stage_seal'],f'outputs/round102/{stage}/stage_seal.json')
    need(source['stage']==stage and (source['cutoff'],source['query_end'])==WINDOWS[stage]
         and source['status']=='prior_input_stage_predictions_sealed' and source['plan_sha256']==R102_PLAN_SHA
         and source['all_query_reload_exact'] and source['numeric_count']==145,'Source stage contract differs')
    for k in SOURCE_KEYS-{'stage_seal','prepared'}:
        need(source_records[k]==source[k],'Source role differs: '+k);add(source_records[k])
    need(source_records['features']==root['features'] and source_records['canonical_mapping']==root['canonical_mapping'],'Root feature identity differs')
    need(source_records['prepared']==source['arms']['T']['prepared'],'Source is not own-stage R102 T')
    pp=add(source_records['prepared']);need(pp.relative_to(ROOT).as_posix().startswith(f'outputs/round102/{stage}/T/') and pp.suffix=='.npz','Foreign prepared NPZ')
    for origin in TEACHERS[stage]:
        need(set(teacher_records[origin])=={'stage_seal'},'Unexpected teacher record roles');rec=teacher_records[origin]['stage_seal']
        if origin=='pre_auxiliary':expected='outputs/round104_teacher/models/sealed_manifest.json'
        else:
            need(rec==root['stages'][origin]['stage_manifest'],'Historical teacher not bound by R102 root');expected=f'outputs/round102/{origin}/stage_seal.json'
        t=doc(rec,expected)
        need(t['stage']==origin and (t['cutoff'],t['query_end'])==WINDOWS[origin] and t['all_query_reload_exact'] and t['query_labels_available'] is False,'Teacher window contract differs')
        need(t['features']==source['features'] and t['canonical_mapping']==source['canonical_mapping'],'Teacher feature identity differs')
        if origin=='pre_auxiliary':
            need(t['status']=='pre_auxiliary_teachers_sealed' and t['new_native_fits']==2,'Pre teacher incomplete');pred=t['native_predictions']
        else:
            need(t['status']=='prior_input_stage_predictions_sealed' and t['plan_sha256']==R102_PLAN_SHA,'Historical teacher contract differs');pred=t['fixed_branches']
        for rec in (t['query_indices'],pred['area'],pred['lgb']):need(add(rec).suffix=='.npy','Teacher arrays must be NPY')
    return inputs


def build_arrays(stage, arrays, fit_indices, query_indices, eligible_positions, trade_days, teachers):
    """Pure array operation; full date axis is canonical, never raw-cache positions.

    Teachers map fixed origins to {indices, area, lgb}; all full forward rows are
    checked before intersecting the original eligible source rows. No y-based mask.
    """
    need(stage in TEACHERS and set(arrays) == set(NINE_KEYS), 'Unexpected stage/source NPZ keys')
    need(set(teachers) == set(TEACHERS[stage]), 'Wrong teacher origins for student')
    days = np.asarray(trade_days); need(days.dtype == np.float64 and days.ndim == 1, 'Canonical dates must be float64')
    fi = indices(fit_indices, len(days), 'source fit'); qi = indices(query_indices, len(days), 'source query')
    kept = indices(eligible_positions, len(fi), 'source eligible positions')
    first, last = map(day, WINDOWS[stage]); finite = np.isfinite(days)
    need(np.array_equal(fi, np.flatnonzero(finite & (days < first))), 'Source full fit date mask differs')
    need(np.array_equal(qi, np.flatnonzero(finite & (days >= first) & (days <= last))), 'Source full query date mask differs')
    ne, nq = len(kept), len(qi)
    for k, a in arrays.items():
        need(isinstance(a, np.ndarray) and np.isfinite(a).all(), 'Nonfinite source array: '+k)
        rows = ne if k in TRAIN_KEYS else nq
        if k != 'cardinalities': need(a.shape[0] == rows, 'Source row count: '+k)
        need(a.dtype == (np.int64 if k in ('train_cat','query_cat','cardinalities') else np.float32), 'Source dtype: '+k)
    need(arrays['train_num'].shape == (ne,145) and arrays['query_num'].shape == (nq,145), 'Source must retain 145 features')
    need(arrays['cardinalities'].ndim == 1 and (arrays['cardinalities']>0).all(), 'Invalid cardinalities')
    nc = len(arrays['cardinalities'])
    for k, rows in (('train_cat',ne),('query_cat',nq)):
        a=arrays[k];need(a.shape==(rows,nc) and (a>=0).all() and (a<arrays['cardinalities']).all(), 'Invalid categories')
    for k in ('train_y_million','train_price_factor_million','train_sample_weight','query_price_factor_million'):
        need(arrays[k].ndim==1 and (arrays[k]>0).all(), 'Invalid positive source factor/target/weight')
    pieces=[]; counts={}
    for code, origin in enumerate(TEACHERS[stage]):
        t=teachers[origin];need(set(t)=={'indices','area','lgb'}, 'Unexpected teacher arrays')
        ix=indices(t['indices'],len(days),origin); start,end=map(day,WINDOWS[origin])
        need(start<first and end<first and np.array_equal(ix,np.flatnonzero(finite & (days>=start) & (days<=end))), 'Incomplete or illegal forward teacher window')
        for k in ('area','lgb'):
            a=t[k];need(isinstance(a,np.ndarray) and a.dtype==np.float64 and a.shape==ix.shape and np.isfinite(a).all() and (a>=0).all(), 'Invalid native TWD teacher '+k)
        pieces.append((ix,t['area'],t['lgb'],np.full(len(ix),code,dtype=np.int64),np.arange(len(ix),dtype=np.int64)))
        counts[origin]={'full_forward_rows':len(ix),'cutoff':WINDOWS[origin][0],'query_end':WINDOWS[origin][1]}
    merged=[np.concatenate([p[k] for p in pieces]) for k in range(5)]; order=np.argsort(merged[0],kind='stable')
    merged=[a[order] for a in merged]
    need(len(np.unique(merged[0]))==len(merged[0]), 'Overlapping canonical teacher predictions; no arbitrary duplicate choice')
    selected, positions, teacher_positions=np.intersect1d(fi[kept],merged[0],assume_unique=True,return_indices=True)
    need(len(selected)>0 and (np.diff(selected)>0).all(), 'No eligible forward student rows')
    positions=positions.astype(np.int64); teacher_positions=teacher_positions.astype(np.int64)
    area=merged[1][teacher_positions]; light=merged[2][teacher_positions]
    offset=np.asarray((.375*area+.25*light)/1e6,dtype=np.float32)
    need(np.isfinite(offset).all() and (offset>=0).all(), 'Offset cast overflow or invalid unit')
    result={k:(a[positions].copy() if k in TRAIN_KEYS else a.copy()) for k,a in arrays.items()};result[OFFSET]=offset
    for k in TRAIN_KEYS: need(identity(result[k])==identity(arrays[k][positions]), 'Source subset bytes changed')
    for k in QUERY_KEYS: need(identity(result[k])==identity(arrays[k]), 'Query/cardinality bytes changed')
    codes=merged[3][teacher_positions]
    for code,origin in enumerate(TEACHERS[stage]): counts[origin]['selected_student_rows']=int((codes==code).sum())
    ratio=result['train_y_million'].astype(np.float64)/result['train_price_factor_million'].astype(np.float64)
    median=float(np.median(ratio));bias=median+float(np.log(-np.expm1(-median)))
    need(np.isfinite(np.float32(bias)) and median>0, 'Invalid shared original ratio initialization')
    mappings={'student_fit_indices':selected.astype(np.int64),'source_npz_row_positions':positions,
        'student_outer_fit_positions':kept[positions],'teacher_origin_codes':codes,
        'teacher_query_positions':merged[4][teacher_positions]}
    info={'stage':stage,'cutoff':WINDOWS[stage][0],'query_end':WINDOWS[stage][1],
        'source_outer_fit_rows':len(fi),'source_eligible_rows':ne,'source_ineligible_rows':len(fi)-ne,
        'union_teacher_rows':len(merged[0]),'union_unique_teacher_rows':len(merged[0]),
        'eligible_without_forward_teacher_rows':ne-len(selected),'teacher_rows_not_in_source_eligible':len(merged[0])-len(selected),
        'fit_rows':len(selected),'query_rows':nq,'teacher_counts':counts,'teacher_origin_order':list(TEACHERS[stage]),
        'source_sample_weight_sum_float64':float(arrays['train_sample_weight'].sum(dtype=np.float64)),
        'selected_sample_weight_sum_float64':float(result['train_sample_weight'].sum(dtype=np.float64)),
        'sample_weights_renormalized':False,'query_four_arrays_bitwise_unchanged':True,
        'all_five_train_arrays_exact_source_subset':True,'initial_training_median_ratio':median,'initial_head_bias':bias,
        'selection_uses_labels_or_errors':False,'offset_from_float64_then_float32':True,
        'nonpositive_equivalent_residual_rows_retained':int((result['train_y_million'].astype(np.float64)-offset.astype(np.float64)<=0).sum()),
        'all_input_arrays':{k:identity(a) for k,a in result.items()}}
    return result,mappings,info


def prepare(stage, source_records, teacher_records, out):
    """Explicit real-I/O entry; caller must freeze records/source before invoking.

    Only the own-stage R102 T NPZ is opened. Other origins contribute sealed
    native predictions and indices, not NPZs, models, raw prices or fit labels.
    """
    need(stage in TEACHERS and set(source_records)==SOURCE_KEYS and set(teacher_records)==set(TEACHERS[stage]), 'Unexpected input record roles')
    out=safe(out);need(not out.exists(),'Preserve existing preparation');out.mkdir(parents=True)
    tick=time.monotonic();inputs={};failure=None;code=1;sources=source_hashes()
    def checked(rec, expected=None):
        need(isinstance(rec,dict) and {'path','sha256'}<=set(rec), 'Invalid bound record')
        p=safe(rec['path']);name=p.relative_to(ROOT).as_posix()
        need(expected is None or name==expected,'Unexpected bound path: '+name)
        need(name not in inputs or inputs[name]==rec['sha256'],'Conflicting input identity')
        need(sha(p)==rec['sha256'],'Bound input changed: '+name);inputs[name]=rec['sha256'];return p
    def doc(rec, expected=None):return json.loads(checked(rec,expected).read_text(encoding='utf-8-sig'))
    def array(rec):
        p=checked(rec);need(p.suffix=='.npy','Only NPY array records');a=np.load(p,allow_pickle=False)
        for k,v in identity(a).items():
            if k in rec:need(v==rec[k],'Bound array schema/value identity')
        return a
    try:
        expected_inputs=collect_input_records(stage,source_records,teacher_records)
        inputs.update({p:expected_inputs[p] for p in ('reports/round104_plan.md',R102_ROOT['path'])})
        source=doc(source_records['stage_seal'],f'outputs/round102/{stage}/stage_seal.json')
        need(source['status']=='prior_input_stage_predictions_sealed' and source['stage']==stage and source['plan_sha256']==R102_PLAN_SHA
             and source['all_query_reload_exact'] and source['numeric_count']==145,'Original R102 source seal differs')
        for k in SOURCE_KEYS-{'stage_seal','prepared'}: need(source_records[k]==source[k],'Source record role mismatch: '+k)
        need(source_records['prepared']==source['arms']['T']['prepared'],'Source is not original T prepared')
        for k in ('fit_indices','query_indices','eligible_positions'):
            need(source_records[k]['path'].startswith(f'outputs/round102/{stage}/'),'Foreign student indices')
        fi,qi,kept=(array(source_records[k]) for k in ('fit_indices','query_indices','eligible_positions'))
        mapping=array(source_records['canonical_mapping']);indices(mapping,455784,'canonical to raw')
        need(len(mapping)==455255,'Canonical mapping dimensions')
        fp=checked(source_records['features'],'data_cache/train_features.pkl');X=pd.read_pickle(fp)
        need(len(X)==455784 and X.index.equals(pd.RangeIndex(len(X))) and not {'總價元','y','target'}.intersection(X.columns),'Pure-X schema differs')
        days=X.iloc[mapping]['trade_day_index'].to_numpy(dtype=np.float64);del X
        teachers={};authorities={}
        for origin in TEACHERS[stage]:
            need(set(teacher_records[origin])=={'stage_seal'},'Unexpected teacher input role')
            rec=teacher_records[origin]['stage_seal']
            expected='outputs/round104_teacher/models/sealed_manifest.json' if origin=='pre_auxiliary' else f'outputs/round102/{origin}/stage_seal.json'
            t=doc(rec,expected);start,end=WINDOWS[origin]
            need(t['stage']==origin and t['cutoff']==start and t['query_end']==end and t['all_query_reload_exact'] and t['query_labels_available'] is False,'Teacher time/status scope differs')
            need(t['features']==source['features'] and t['canonical_mapping']==source['canonical_mapping'],'Teacher/student feature mapping differs')
            if origin=='pre_auxiliary':
                need(t['status']=='pre_auxiliary_teachers_sealed' and t['new_native_fits']==2,'Early teacher not completed')
                pred=t['native_predictions']
            else:
                need(t['status']=='prior_input_stage_predictions_sealed' and t['plan_sha256']==R102_PLAN_SHA,'Historical teacher seal differs')
                pred=t['fixed_branches']
            teachers[origin]={'indices':array(t['query_indices']),'area':array(pred['area']),'lgb':array(pred['lgb'])}
            authorities[origin]={'stage_seal':rec,'query_indices':t['query_indices'],'area':pred['area'],'lgb':pred['lgb'],'cutoff':start,'query_end':end}
        prepared=checked(source_records['prepared']);name=prepared.relative_to(ROOT).as_posix()
        need(name.startswith(f'outputs/round102/{stage}/T/') and prepared.suffix=='.npz','Only own-stage R102 T NPZ permitted')
        with np.load(prepared,allow_pickle=False) as z:
            need(set(z.files)==set(NINE_KEYS),'Exactly nine keys required before materialization');arrays={k:z[k].copy() for k in NINE_KEYS}
        need({k:identity(a) for k,a in arrays.items()}==source['arms']['T']['all_input_arrays'],'Original source array identity differs')
        new,mappings,info=build_arrays(stage,arrays,fi,qi,kept,days,teachers)
        path=out/'prepared.npz'
        with path.open('xb') as f:np.savez_compressed(f,**new)
        with np.load(path,allow_pickle=False) as z:
            need(set(z.files)==set(NINE_KEYS)|{OFFSET},'Saved 10-key schema changed')
            need(all(identity(z[k])==identity(a) for k,a in new.items()),'Prepared full-array roundtrip differs')
        saved={k:save_array(out/(k+'.npy'),a) for k,a in mappings.items()}
        need(inputs==expected_inputs,'Actual input readset differs from metadata-only collection')
        need(sources==source_hashes() and all(sha(ROOT/p)==h for p,h in inputs.items()),'Preparation source/input changed')
        result={**info,**saved,'status':'matched_base_aware_student_prepared','schema_version':1,'round':104,
            'parameters':parameters(),'prepared':record(path),'preparation':{'path':(out/'preparation.json').relative_to(ROOT).as_posix()},
            'source_records':source_records,'teacher_records':authorities,'source_input_arrays':source['arms']['T']['all_input_arrays'],
            'source_sha256':sources,'input_sha256':inputs,'plan_sha256':PLAN_SHA,'same_prepared_for_C_and_T':True,
            'query_labels_available':False,'raw_price_pickle_opened':False,'other_stage_NPZ_opened':False,
            'new_prior_encoder_model_calls':0,'NPZ_full_roundtrip_exact':True,'elapsed_seconds':time.monotonic()-tick}
        result.pop('preparation');write(out/'preparation.json',result);code=0
        return {**result,'preparation':record(out/'preparation.json')}
    except Exception:failure=traceback.format_exc();raise
    finally:write(out/'execution.json',{'exit_code':code,'failure':failure,'source_sha256':sources,'input_sha256':inputs,'elapsed_seconds':time.monotonic()-tick,'automatic_retry':False})


def self_check():
    tick=time.monotonic();checks={}
    def mark(name,ok):need(bool(ok),name);checks[name]=True
    def reject(name,fn):
        try:fn()
        except ValueError:checks[name]=True
        else:raise AssertionError(name)
    dates=np.array([day(s) for s in ('2022-09-01','2019-01-01','2020-09-01','2023-09-01','2021-09-01','2020-08-11','2022-08-10','2023-08-10','2021-08-10')],dtype=np.float64)
    def fixture(stage):
        start,end=map(day,WINDOWS[stage]);fi=np.flatnonzero(dates<start).astype(np.int64);qi=np.flatnonzero((dates>=start)&(dates<=end)).astype(np.int64)
        kept=np.arange(len(fi),dtype=np.int64);n=len(kept);q=len(qi)
        a={'train_num':np.arange(n*145,dtype=np.float32).reshape(n,145),'query_num':np.ones((q,145),dtype=np.float32),
           'train_cat':np.zeros((n,2),dtype=np.int64),'query_cat':np.zeros((q,2),dtype=np.int64),'cardinalities':np.array([2,3],dtype=np.int64),
           'train_y_million':np.arange(1,n+1,dtype=np.float32),'train_price_factor_million':np.full(n,2,dtype=np.float32),
           'query_price_factor_million':np.full(q,3,dtype=np.float32),'train_sample_weight':np.arange(1,n+1,dtype=np.float32)*.17}
        teachers={}
        for origin in TEACHERS[stage]:
            first,last=map(day,WINDOWS[origin]);ix=np.flatnonzero((dates>=first)&(dates<=last)).astype(np.int64)
            teachers[origin]={'indices':ix,'area':ix.astype(np.float64)*1e6+1e7,'lgb':ix.astype(np.float64)*2e6+2e7}
        return a,fi,qi,kept,teachers
    for stage in TEACHERS:
        a,fi,qi,kept,t=fixture(stage);before={k:identity(v) for k,v in a.items()};new,m,info=build_arrays(stage,a,fi,qi,kept,dates,t)
        mark(stage+'_date_only_sorted_subset',np.all(np.diff(m['student_fit_indices'])>0) and np.all(dates[m['student_fit_indices']]<day(WINDOWS[stage][0])))
        mark(stage+'_shared_bytes_and_no_mutation',before=={k:identity(v) for k,v in a.items()} and all(identity(new[k])==identity(a[k]) for k in QUERY_KEYS))
        mark(stage+'_weights_not_renormalized',identity(new['train_sample_weight'])==identity(a['train_sample_weight'][m['source_npz_row_positions']]))
        mark(stage+'_negative_residual_retained',info['nonpositive_equivalent_residual_rows_retained']==info['fit_rows'])
    a,fi,qi,kept,t=fixture('confirmation');new,m,info=build_arrays('confirmation',a,fi,qi,kept,dates,t)
    expected=np.asarray((.375*(m['student_fit_indices']*1e6+1e7)+.25*(m['student_fit_indices']*2e6+2e7))/1e6,dtype=np.float32)
    mark('TWD_f64_then_million_f32_offset_exact',identity(expected)==identity(new[OFFSET]))
    y=a['train_y_million'].copy();a['train_y_million']*=7
    _,m2,_=build_arrays('confirmation',a,fi,qi,kept,dates,t)
    mark('label_changes_cannot_change_selection',all(identity(m[k])==identity(m2[k]) for k in m));a['train_y_million']=y
    a['query_num']*=11;_,m2,i2=build_arrays('confirmation',a,fi,qi,kept,dates,t)
    mark('query_values_cannot_change_subset_or_bias',identity(m2['student_fit_indices'])==identity(m['student_fit_indices']) and i2['initial_head_bias']==info['initial_head_bias'])
    subset=np.arange(1,len(kept),dtype=np.int64);aa={k:(v[subset].copy() if k in TRAIN_KEYS else v.copy()) for k,v in a.items()}
    _,mm,ii=build_arrays('confirmation',aa,fi,qi,kept[subset],dates,t)
    mark('original_eligibility_intersection_and_NPZ_axis',np.array_equal(mm['student_fit_indices'],np.intersect1d(fi[kept[subset]],m['student_fit_indices'])))
    reject('extra_source_key_rejected',lambda:build_arrays('confirmation',{**a,'query_y':np.zeros(len(qi))},fi,qi,kept,dates,t))
    reject('wrong_origins_rejected',lambda:build_arrays('confirmation',a,fi,qi,kept,dates,{'pre_auxiliary':t['pre_auxiliary']}))
    bad={k:dict(v) for k,v in t.items()};bad['pre_auxiliary']['indices']=bad['pre_auxiliary']['indices'][::-1]
    reject('reordered_teacher_indices_rejected',lambda:build_arrays('confirmation',a,fi,qi,kept,dates,bad))
    bad={k:dict(v) for k,v in t.items()};bad['pre_auxiliary']['area']=bad['pre_auxiliary']['area'].astype(np.float32)
    reject('non_float64_teacher_rejected',lambda:build_arrays('confirmation',a,fi,qi,kept,dates,bad))
    mark('strict_serializable_receipt',isinstance(json.loads(json.dumps(info,allow_nan=False)),dict))
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,'parameters':parameters(),'plan_sha256':PLAN_SHA,
        'source_sha256':source_hashes(),'synthetic_only':True,'real_data_read':False,'real_prepare_fit_predict_calls':0,'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true');p.add_argument('--receipt',default=str(ROOT/'reports/round104_student_selfcheck.json'));a=p.parse_args()
    if not a.self_check:p.error('Only --self-check is available; real preparation is caller-controlled')
    path=safe(a.receipt);need(not path.exists() and not path.with_suffix('.sha256').exists(),'Preserve prior receipt')
    try:r=self_check()
    except Exception:write(path,{'status':'failed','exit_code':1,'failure':traceback.format_exc(),'source_sha256':source_hashes(),'synthetic_only':True,'real_data_read':False});raise
    write(path,r);print(json.dumps({'status':r['status'],'checks':r['check_count'],'elapsed_seconds':r['elapsed_seconds']}))
