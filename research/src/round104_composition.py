"""Apply immutable original-cutoff R/S and median to R104 student predictions.

No fit, raw price load, teacher load, model load, or scoring. Preparation and
training remain separate. Explicit records are returned for the caller's freeze.
"""
from pathlib import Path
import hashlib
import pickle
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as metric
import round48_nn_experiment as old

ROOT_SEAL={'path':'outputs/round102/sealed_manifest.json','sha256':'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
P48='outputs/round48/plan.json'


def specification(stage):
    """Metadata only, never numerical arrays or pickle deserialization."""
    c.need(stage in c.STAGES,'Unknown stage')
    c.need(c.sha(ROOT_SEAL['path'])==ROOT_SEAL['sha256'],'Original R102 root seal changed')
    main=c.read(ROOT_SEAL['path']);sr=main['stages'][stage]['stage_manifest']
    c.need(c.sha(sr['path'])==sr['sha256'],'Original R102 stage seal changed')
    sm=c.read(sr['path']);digest=sm['original_input_sha256'][P48]
    c.need(c.sha(P48)==digest,'Original R/S authority changed')
    p48=c.read(P48);b=p48['bindings'][stage]
    records={ROOT_SEAL['path']:ROOT_SEAL['sha256'],sr['path']:sr['sha256'],P48:digest}
    for key in ('features','canonical_mapping','fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices'):
        r=sm[key];records[r['path']]=r['sha256']
    for r in list(sm['fixed_branches'].values())+list(sm['layers']['v22'].values())+[sm['predictions']['v22']]:
        records[r['path']]=r['sha256']
    for key in ('reference','state','state_summary','query_geo'):
        name=b[key];h=p48['input_sha256'][name]
        c.need(sm['original_input_sha256'][name]==h,'R/S identity disagrees')
        records[name]=h
    c.need(c.RAW not in records and not any(x.endswith('.npz') for x in records),'Forbidden composition input')
    return {'stage':stage,'source_stage_seal':sr,'input_sha256':records,
            'R_S_binding':{k:b[k] for k in ('reference','state','state_summary','query_geo')},
            'geography_frame_sha256':b['manifest']['features']['query']['frame_sha256']}


def load_context(spec):
    stage=spec['stage'];c.need(spec==specification(stage),'Frozen context specification differs')
    c.verify_hashes(spec['input_sha256'])
    def path(rec):
        c.need(spec['input_sha256'].get(rec['path'])==rec['sha256'],'Input not frozen')
        return c.safe(rec['path'])
    sm=c.read(path(spec['source_stage_seal']))
    def arr(rec):path(rec);return c.load_array(rec)
    ids={k:arr(sm[k]) for k in ('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices')}
    mapping=np.load(path(sm['canonical_mapping']),allow_pickle=False)
    metric.index(mapping,455784,'canonical mapping');c.need(len(mapping)==455255,'Canonical row count')
    raw=pd.read_pickle(path(sm['features']))
    c.need(len(raw)==455784 and '總價元' not in raw and raw.index.equals(pd.RangeIndex(len(raw))),'Pure X schema')
    fi,qi=ids['fit_indices'],ids['query_indices'];days=raw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64)
    metric.check_temporal(stage,fi,qi,ids['query_source_cache_indices'],days,mapping)
    c.exact(mapping[fi],ids['fit_source_cache_indices'],'Fit source cache indices')
    X=raw.iloc[mapping[fi]].reset_index(drop=True);Xq=raw.iloc[mapping[qi]].reset_index(drop=True);del raw
    b=spec['R_S_binding']
    with c.safe(b['reference']).open('rb') as f:reference=pickle.load(f)
    with c.safe(b['state']).open('rb') as f:state=pickle.load(f)
    summary=c.read(b['state_summary']);geo=pd.read_pickle(c.safe(b['query_geo']))
    c.need(state['state_summary']==summary and summary['fit_X_sha256']==old.spatial.frame_sha(X)
           and not summary['query_used_in_build'] and reference['fit_report']['fitting_rows']==len(fi),'R/S fit-only state mismatch')
    c.need(state['parameters']==old.spatial.parameters() and geo.index.equals(Xq.index)
           and old.spatial.frame_sha(geo)==spec['geography_frame_sha256'],'R/S geography mismatch')
    fixed={k:arr(r) for k,r in sm['fixed_branches'].items()}
    original={k:arr(r) for k,r in sm['layers']['v22'].items()};original['median']=arr(sm['predictions']['v22'])
    for a in [*fixed.values(),*original.values()]:metric.prices(a,(len(qi),),'Saved composition vector')
    context={'stage':stage,'specification':spec,'source':sm,'indices':ids,'fit_X':X,'query_X':Xq,
             'geo':geo,'reference':reference,'spatial_state':state,'fixed':fixed,'original':original,
             'state_identity':hashlib.sha256(pickle.dumps((reference,state),protocol=5)).hexdigest(),
             'frame_identity':[old.spatial.frame_sha(f) for f in (X,Xq)]}
    replay,_,_=apply(context,original['native'])
    for k,v in replay.items():c.exact(v,original[k],'Original v22 '+k+' replay')
    verify_unchanged(context)
    return context


def apply(context,native):
    metric.prices(native,(len(context['query_X']),),'Native NN total')
    Xq=context['query_X'];fixed=context['fixed']
    base=.75*(.5*fixed['area']+.5*native)+.25*fixed['lgb']
    ref,diag,_=old.original.predict_reference(context['reference'],Xq,base,variant='trend',return_diagnostics=True)
    spatial,detail,_=old.spatial.predict(context['spatial_state'],Xq,context['geo'],base,ref,diag)
    fallback=~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback],ref[fallback],'Spatial fallback changed')
    median=np.median(np.column_stack([spatial,fixed['H'],fixed['U']]),axis=1)
    return {'native':native,'base':base,'reference':ref,'spatial':spatial,'median':median},diag,detail


def verify_unchanged(context):
    c.need(context['state_identity']==hashlib.sha256(pickle.dumps((context['reference'],context['spatial_state']),protocol=5)).hexdigest(),'R/S state mutated')
    c.need(context['frame_identity']==[old.spatial.frame_sha(f) for f in (context['fit_X'],context['query_X'])],'X frames mutated')
    c.verify_hashes(context['specification']['input_sha256'])


def save_chain(context,native,out):
    out=c.safe(out);out.mkdir(parents=True,exist_ok=False)
    values,diag,detail=apply(context,native)
    records={k:c.save_array(out/(k+'.npy'),v) for k,v in values.items()}
    diagnostics={}
    for name,frame in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
        p=out/(name+'.pkl')
        with p.open('xb') as f:pickle.dump(frame,f,protocol=5)
        with p.open('rb') as f:pd.testing.assert_frame_equal(frame,pickle.load(f),check_exact=True)
        diagnostics[name]=c.record(p)
    verify_unchanged(context)
    return {'layers':{k:v for k,v in records.items() if k!='median'},'prediction':records['median'],
            'diagnostics':diagnostics,'R_S_state_and_frames_unchanged':True}
