"""Small algebra/acceptance checks; never train or score competition examples."""
import sys,json,copy
from pathlib import Path
import numpy as np
import pandas as pd
import round4_evaluate as e
from round4_common import ROOT

def metric(value):
    return {'overall_mae':value,'weighted_mae':value,
      'forecast_quarters':{f'Q{i}':{'rows':300,'mae':value} for i in range(1,5)},
      'strata':{'seen':{'rows':600,'mae':value},'new_nonmissing':{'rows':600,'mae':value},'missing':{'rows':0,'mae':None}}}
base=metric(100.);tiny=metric(99.5);good=metric(98.)
assert e.development_selection({e.CONTROL:base,'anchor_blend':tiny})['selected_candidate']==e.CONTROL
assert e.development_selection({e.CONTROL:base,'anchor_blend':good})['selected_candidate']=='anchor_blend'
bad=copy.deepcopy(good);bad['weighted_mae']=100.
assert not e.development_selection({e.CONTROL:base,'anchor_blend':bad})['qualified_candidates']
bad=copy.deepcopy(good);bad['forecast_quarters']['Q4']['mae']=106.
assert not e.development_selection({e.CONTROL:base,'anchor_blend':bad})['qualified_candidates']
bad=copy.deepcopy(good);bad['strata']['new_nonmissing']['mae']=106.
assert not e.development_selection({e.CONTROL:base,'anchor_blend':bad})['qualified_candidates']
assert e.stage_decision('confirmation',{e.CONTROL:base,'anchor_blend':good},'anchor_blend')['passed']
one=copy.deepcopy(good)
for q in ['Q2','Q3','Q4']:one['forecast_quarters'][q]['mae']=100.
assert not e.stage_decision('confirmation',{e.CONTROL:base,'anchor_blend':one},'anchor_blend')['passed']
assert e.stage_decision('auxiliary',{e.CONTROL:base,'anchor_blend':metric(102.)},'anchor_blend')['passed']
assert not e.stage_decision('auxiliary',{e.CONTROL:base,'anchor_blend':metric(103.1)},'anchor_blend')['passed']
tied=e.development_selection({e.CONTROL:base,'combined':good,'anchor_blend':good})
assert tied['selected_candidate']=='anchor_blend'
original=e.predict_reference;calls=[]
def probe(reference,query,base,variant):
    calls.append(variant);return base
try:
    e.predict_reference=probe
    p=e.build_predictions(list(e.FORMULAS),np.array([10.]),np.array([20.]),
      {'anchor_catboost':np.array([30.]),'augmented_catboost':np.array([40.]),'spatial_raw_lightgbm':np.array([50.])},{},pd.DataFrame({'x':[1]}))
finally:e.predict_reference=original
expected={e.CONTROL:12.5,'anchor_blend':20.,'augmented_blend':23.75,'spatial_raw_blend':16.25,'combined':25.625}
assert all(np.allclose(p[k],v) for k,v in expected.items())
assert calls==['trend']*5
report={'status':'passed','subthreshold_improvement_rejected':True,'both_main_metrics_required':True,
 'quarter_and_large_stratum_vetoes_verified':True,'confirmation_requires_two_improved_quarters':True,
 'auxiliary_3pct_limit_verified':True,'deterministic_tie_break_verified':True,'candidate_blend_units_verified':True,
 'exactly_one_reference_trend_call_per_candidate':True,'uses_synthetic_inputs_only':True,'new_models_trained':False,'holdout_scores_observed':False}
(ROOT/'outputs/round4/audit/evaluation_preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
