"""Two fixed target-free unit-prefix rules; standard library and strings only.

A returned code is a string feature, never a physical building/stack identity.
No dataset, model, date, project statistics or labels are accepted or loaded.
"""
from __future__ import annotations
import re
import unicodedata

MISSING='__MISSING__'
RULES={
    'version':1,
    'normalization':['Unicode NFKC','strip and remove all whitespace','uppercase'],
    'missing_literals':['','NAN','NONE','NULL','<NA>',MISSING],
    'code_floor_regex':r'^(?P<code>[A-Z]{1,3}[0-9]{0,3})-(?P<floor>B?[0-9]{1,3})(?:F|樓|層)號?$',
    'explicit_block_regex':r'^(?P<code>[A-Z]{1,3}[0-9]{0,3})[棟幢](?P<suffix>.*)$',
    'multi_object_separator_regex':r'[,、;/&及與暨]',
    'range_marker_regex':r'至|到|~',
    'maximum_block_markers':1,
    'precedence':['missing','multiple_object_separator','range_marker','multiple_block_markers',
                  'code_floor','explicit_block','unmatched'],
    'unknown_or_ambiguous':MISSING,
    'no_floor_inference_from_unmarked_numbers':True,
    'no_chinese_number_or_reversed_floor_rule':True,
    'no_project_or_query_vocabulary_learning':True,
    'output_semantics':'string prefix code only; cross-format equality is not a physical-object identity',
}
_CODE_FLOOR=re.compile(RULES['code_floor_regex'])
_EXPLICIT_BLOCK=re.compile(RULES['explicit_block_regex'])
_MULTI=re.compile(RULES['multi_object_separator_regex'])
_RANGE=re.compile(RULES['range_marker_regex'])


def normalize(value):
    if value is None:return MISSING
    text=unicodedata.normalize('NFKC',str(value)).strip().upper()
    text=re.sub(r'\s+','',text)
    return MISSING if text in RULES['missing_literals'] else text


def prefix_details(value):
    """Diagnostic origin/reason for the same rules used by extract_prefix."""
    text=normalize(value)
    reason=None
    if text==MISSING:reason='missing'
    elif _MULTI.search(text):reason='multiple_object_separator'
    elif _RANGE.search(text):reason='range_marker'
    elif text.count('棟')+text.count('幢')>RULES['maximum_block_markers']:reason='multiple_block_markers'
    if reason is not None:return {'prefix_code':MISSING,'source':'none','reason':reason}
    for source,pattern in [('code_floor',_CODE_FLOOR),('explicit_block',_EXPLICIT_BLOCK)]:
        match=pattern.fullmatch(text)
        if match:return {'prefix_code':match.group('code'),'source':source,'reason':'accepted'}
    return {'prefix_code':MISSING,'source':'none','reason':'unmatched'}


def extract_prefix(value):
    """Return a category string; every excluded/unmatched value is MISSING."""
    return prefix_details(value)['prefix_code']


def self_check():
    checks=[]
    expected={'A1-5F':'A1','A1-5F號':'A1','Ａ１－５ｆ號':'A1',' A1 - 5F 號 ':'A1',
              'A1-B1F號':'A1','A-4樓號':'A','A1棟5F號':'A1','Ａ１棟五樓號':'A1',
              'A棟A1-11F號':'A','B2幢3號':'B2','A棟3-1號':'A'}
    for value,wanted in expected.items():
        if extract_prefix(value)!=wanted:raise AssertionError(value)
        checks.append('accept_'+value)
    rejected=[None,'',MISSING,'A1-5','A1-5號','A1戶','A1號','A1-五樓號','甲棟5F號',
              '5F-A號','A1—5F號','A1-5F號、A2-6F號','A1棟1,2號','A1棟1；2號',
              'A1棟1/2號','A1棟1&2號','A1棟1及2號','A1棟1與2號','A1棟1暨2號',
              'A1棟1至2號','A1棟1到2號','A1棟1～2號','A棟B幢1號']
    for value in rejected:
        if extract_prefix(value)!=MISSING:raise AssertionError('Expected reject: '+str(value))
        checks.append('reject_'+str(value))
    if prefix_details('A1-5F號')['source']!='code_floor':raise AssertionError('Code-floor origin')
    if prefix_details('A1棟5F號')['source']!='explicit_block':raise AssertionError('Block origin')
    checks.append('same_lexical_code_distinct_origins_preserved_in_diagnostics')
    return {'passed':True,'check_count':len(checks),'checks':checks,'data_or_model_IO':False}
