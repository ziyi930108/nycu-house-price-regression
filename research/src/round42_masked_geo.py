"""R42 candidate: R5 fit-only project masking plus original R6 geography.
Only the tree sees masked X. Reference, spatial grouping, and scoring must
continue to receive original X. No data load or training on import.
"""
from pathlib import Path
import copy, hashlib, time
import numpy as np
import round5_project_mask as masking
import round6_geo as geography

RULES={'method':'R5 deterministic 50% whole-project mask + R6 geographic raw CatBoost',
 'mask_columns':list(masking.IDENTIFIER_COLUMNS),'mask_probability_per_project':0.5,
 'mask_prefix':masking.MASK_RULES['hash_prefix'],'unseen_query_projects_masked':True,
 'geography_for_tree':'original R6 annual geo, not R39 fallback geo',
 'reference_and_scoring_use_original_X':True,'feature_count':95,'categorical_count':23,
 'target':'y_TWD/1e6','prediction':'max(z*1e6,0)','sample_weight':'uniform'}

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parameters(): return geography.parameters()

def fit_model(X,y,geo,cats):
    started=time.monotonic()
    mask_state=masking.fit_feature_state(X)
    masked=masking.transform_features(X,mask_state)
    model,geo_state,metadata=geography.fit_model(masked,y,geo,cats)
    state={'rules':copy.deepcopy(RULES),'source_sha256':_sha(__file__),
       'mask_state':mask_state,'geo_state':geo_state,
       'all_feature_names':list(geo_state['all_feature_names']),
       'original_fit_X_sha256':geography.frame_sha(X)}
    state['state_content_sha256']=geography.content_sha(state)
    metadata=copy.deepcopy(metadata)
    metadata.update(component='masked_geography_raw',helper_source_sha256=_sha(__file__),
        mask_rules=copy.deepcopy(masking.MASK_RULES),mask_summary=mask_state['fit_summary'],
        original_fit_X_sha256=state['original_fit_X_sha256'],
        wrapped_feature_state_sha256=state['state_content_sha256'],
        feature_state_content_sha256=geo_state['state_content_sha256'],
        mask_uses_query_or_target=False,elapsed_seconds=time.monotonic()-started)
    return model,state,metadata

def transformed_query(state,X):
    if state['rules']!=RULES or state['source_sha256']!=_sha(__file__):
        raise ValueError('Mask-geography helper changed')
    expected=geography.content_sha({k:v for k,v in state.items() if k!='state_content_sha256'})
    if state['state_content_sha256']!=expected:
        raise ValueError('Wrapper state changed')
    return masking.transform_features(X,state['mask_state'])

def predict_model(model,state,X,geo,cats):
    masked=transformed_query(state,X)
    return geography.predict_model(model,state['geo_state'],masked,geo,cats)
