"""R25 pure-X address detail categories; syntax eligibility, not verified locations.
No cache, target, prediction or model is read. Only CLI self-check may read an
explicit artificial-case JSON. Existing address parsing is never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geo_address_audit as legacy

MISSING = '__MISSING__'
FEATURE_COLUMNS = ['addr_lane_path', 'addr_house_block100', 'addr_detail_eligible']
CATEGORICAL_COLUMNS = FEATURE_COLUMNS[:2]
NO_LANE, NO_ALLEY = '__NO_LANE__', '__NO_ALLEY__'
_FINANCIAL = str.maketrans({'〇':'零','兩':'二','壹':'一','貳':'二','參':'三','肆':'四',
                          '伍':'五','陸':'六','柒':'七','捌':'八','玖':'九','拾':'十','佰':'百','仟':'千'})
_DIGITS = '零一二三四五六七八九'
_NUM = r'[0-9零〇一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰仟]+'
_DETAIL = re.compile(r'(?:(?P<section>'+_NUM+r')段)?'
    r'(?:(?P<lane>'+_NUM+r'(?:之'+_NUM+r')?)巷)?'
    r'(?:(?P<alley>'+_NUM+r'(?:之'+_NUM+r')?)弄)?'
    r'(?P<main>'+_NUM+r')(?:之(?P<before>'+_NUM+r'))?號'
    r'(?:之(?P<after>'+_NUM+r'))?(?P<tail>.*)')
_FLOOR = re.compile(r'(?:地下|地上|B)?(?P<floor>'+_NUM+r')(?:樓|層|F)'
                    r'(?:之(?P<suffix>'+_NUM+r')|(?P<room>'+_NUM+r')室)?')
_REQUIRED = ['house_query_eligible','is_embedded_header','land_parcel_context',
    'intersection_explicit','intersection_inferred','near_or_opposite_or_unassigned',
    'house_number_range','district_prefix_conflict','road_count','road_mentions_count',
    'road_parse_uncertain','road_candidates_json','address_remainder','locality_prefix',
    'city','district_full','road1_base','road1_name','road1_section','road1_lane','road1_alley',
    'house_number','house_number_count','house_numbers_json']
DETAIL_COLUMNS = ['eligible','reason','main_number','house_has_suffix','house_suffix_number',
                  'section','lane','alley','house_number']
RULES = {
    'feature_columns': FEATURE_COLUMNS,
    'categorical_columns': CATEGORICAL_COLUMNS,
    'feature_dtypes': {'addr_lane_path':'object','addr_house_block100':'object','addr_detail_eligible':'float32'},
    'source_X_columns': ['address_raw','district'],
    'legacy_parser': 'geo_address_audit.parse_geo_queries',
    'legacy_normalization': 'NFKC; remove whitespace; 台 to 臺; existing missing literals unchanged',
    'existing_columns_modified': False,
    'learned_or_fit_state': False,
    'target_or_query_labels_used': False,
    'positive_main_house_number_required': True,
    'other_number_tokens_may_be_zero': True,
    'number_tokens': 'ASCII digits OR Chinese digit-only string OR canonical Chinese units through 千; financial glyphs mapped; parse/canonical roundtrip; 十 and 一十 accepted; no mixed systems or omitted zero placeholders',
    'suffix': 'at most one explicit 之NUM per lane/alley/house; house suffix before 號 or after 號; raw hyphen never accepted',
    'complete_grammar': 'exact locality_prefix then road1_base; optional SECTION/LANE/ALLEY and one HOUSE; all remainder must be an allowed floor tail',
    'tail': 'empty OR optional one leading comma then one floor, optionally enclosed in one pair of parentheses; floor=(地下|地上|B)?NUM(樓|層|F), optionally followed by either 之NUM or NUM室',
    'tail_ambiguous_suffix_plus_room': 'reject an undelimited simultaneous floor subnumber and room number',
    'legacy_gate': 'house_query_eligible; one road; one road mention; one HOUSE token; no header/parcel/intersection/near/range/admin conflict; first context_uncertain false is an existing-guard consistency check',
    'canonical_consistency': 'section/lane/alley/house/road name match all corresponding legacy parsed canonical fields',
    'lane_path': 'JSON [city,district_full,road1_name,lane_or___NO_LANE__,alley_or___NO_ALLEY__]',
    'house_block100': 'same JSON path plus integer (main_house_number-1)//100',
    'bucket_is_distance': False,
    'ineligible': 'both categories __MISSING__; float32 flag 0',
    'eligible': 'two deterministic string categories; float32 flag 1',
    'address_existence_or_physical_identity_verified': False,
    'new_OSM_or_external_data_access': False,
}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical_cn(n):
    if n == 0:
        return '零'
    if not 0 < n < 10000:
        return None
    result, gap = '', False
    for divisor, unit in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
        digit, n = divmod(n, divisor)
        if digit:
            if gap:
                result += '零'
            result += ('' if divisor == 10 and digit == 1 and not result else _DIGITS[digit]) + unit
            gap = False
        elif result and n:
            gap = True
    return result


def strict_number(token):
    """Return a nonnegative Python int; reject mixed or noncanonical numerals."""
    _need(isinstance(token,str) and bool(token), 'number_token_empty_or_not_string')
    if re.fullmatch(r'[0-9]+',token):
        return int(token)
    token = token.translate(_FINANCIAL)
    _need(re.fullmatch('['+_DIGITS+'十百千]+',token) is not None, 'mixed_or_unknown_number_token')
    if not any(c in token for c in '十百千'):
        return int(''.join(str(_DIGITS.index(c)) for c in token))
    total, digit = 0, 0
    units = {'十':10,'百':100,'千':1000}
    for char in token:
        if char in units:
            total += (digit or 1) * units[char]
            digit = 0
        else:
            digit = _DIGITS.index(char)
    total += digit
    canonical = _canonical_cn(total)
    _need(token == canonical or (10 <= total <= 19 and token == '一'+canonical),
          'noncanonical_chinese_number_token')
    return total


def _suffix(token):
    if token is None:
        return ''
    return '-'.join(str(strict_number(x)) for x in token.split('之'))


def _tail_valid(tail):
    if not tail:
        return True
    if tail.startswith(','):
        tail = tail[1:]
    if tail.startswith('(') and tail.endswith(')'):
        tail = tail[1:-1]
    match = _FLOOR.fullmatch(tail)
    if match is None:
        return False
    try:
        for token in match.groupdict().values():
            if token is not None:
                strict_number(token)
    except ValueError:
        return False
    return True


def _record(reason, **kwargs):
    return dict(eligible=False, reason=reason, main_number=None, house_has_suffix=False,
                house_suffix_number=None, section='', lane='', alley='', house_number='', **kwargs)


def _eligibility(row):
    gates = [('is_embedded_header','embedded_header'),('land_parcel_context','land_context'),
        ('intersection_explicit','intersection'),('intersection_inferred','intersection'),
        ('near_or_opposite_or_unassigned','near_or_unassigned'),('house_number_range','number_range'),
        ('district_prefix_conflict','admin_conflict')]
    for column, reason in gates:
        if row[column]:
            return _record(reason)
    if row['road_count'] != 1:
        return _record('not_one_road')
    if row['road_mentions_count'] != 1:
        return _record('not_one_road_mention')
    rest = legacy.normalize_text(row['address_remainder'])
    if len(list(legacy.HOUSE.finditer(rest))) != 1:
        return _record('not_one_house_token')
    if not row['house_query_eligible']:
        return _record('legacy_house_ineligible')
    if row['road_parse_uncertain']:
        return _record('legacy_road_uncertain')
    try:
        roads = json.loads(row['road_candidates_json'])
        if len(roads) != 1 or roads[0].get('context_uncertain') is not False:
            return _record('first_road_context_uncertain_or_invalid')
        road = roads[0]
        if not all(isinstance(row[k],str) and row[k] for k in ('city','district_full','road1_base','road1_name')):
            return _record('missing_admin_or_road')
        locality = row['locality_prefix']
        if not isinstance(locality,str) or not rest.startswith(locality):
            return _record('locality_prefix_mismatch')
        rest = rest[len(locality):]
        if not rest.startswith(row['road1_base']):
            return _record('road_prefix_mismatch')
        body = rest[len(row['road1_base']):]
        match = _DETAIL.fullmatch(body)
        if match is None:
            return _record('incomplete_detail_grammar')
        parts = match.groupdict()
        if parts['before'] is not None and parts['after'] is not None:
            return _record('multiple_house_suffixes')
        section = '' if parts['section'] is None else str(strict_number(parts['section']))
        lane, alley = _suffix(parts['lane']), _suffix(parts['alley'])
        main = strict_number(parts['main'])
        suffix_raw = parts['before'] if parts['before'] is not None else parts['after']
        suffix = None if suffix_raw is None else strict_number(suffix_raw)
        if main <= 0:
            return _record('nonpositive_main_house_number')
        house = str(main) + ('' if suffix is None else '-'+str(suffix))
        if not _tail_valid(parts['tail']):
            return _record('unsupported_tail')
        road_name = row['road1_base'] + (section+'段' if section else '')
        wanted = {'road_base':row['road1_base'],'section_number':section,'lane_number':lane,
                  'alley_number':alley,'house_number':house,'road_name':road_name}
        if any(road.get(k) != v for k,v in wanted.items()):
            return _record('legacy_road_canonical_mismatch')
        canonical = {'road1_section':section,'road1_lane':lane,'road1_alley':alley,
                     'house_number':house,'road1_name':road_name}
        if any(row[k] != v for k,v in canonical.items()) or row['house_number_count'] != 1 or json.loads(row['house_numbers_json']) != [house]:
            return _record('legacy_field_canonical_mismatch')
        return dict(eligible=True,reason='eligible',main_number=main,house_has_suffix=suffix is not None,
                    house_suffix_number=suffix,section=section,lane=lane,alley=alley,house_number=house)
    except ValueError as exc:
        return _record('invalid_number_token' if 'number_token' in str(exc) else 'invalid_parsed_record')
    except (KeyError,TypeError,AttributeError):
        return _record('invalid_parsed_record')


def eligibility_details(P):
    """Same-row-index audit table; no observed data or vocabulary is learned."""
    _need(isinstance(P,pd.DataFrame) and P.columns.is_unique, 'Parsed frame columns must be unique')
    _need(set(_REQUIRED).issubset(P.columns), 'Missing parsed address columns')
    rows = [_eligibility(dict(zip(_REQUIRED,values)))
            for values in P[_REQUIRED].itertuples(index=False,name=None)]
    result = pd.DataFrame(rows,index=P.index,columns=DETAIL_COLUMNS)
    for column in ('eligible','house_has_suffix'):
        result[column] = result[column].astype(bool)
    for column in ('main_number','house_suffix_number'):
        result[column] = pd.Series([r[column] for r in rows],index=P.index,dtype=object)
    for column in ('reason','section','lane','alley','house_number'):
        result[column] = result[column].astype(object)
    return result


def features_from_parsed(P):
    details = eligibility_details(P)
    paths, blocks = [], []
    for values, eligible, main, lane, alley in zip(P[['city','district_full','road1_name']].itertuples(index=False,name=None),
            details.eligible,details.main_number,details.lane,details.alley):
        if not eligible:
            paths.append(MISSING);blocks.append(MISSING);continue
        key = [*values,lane or NO_LANE,alley or NO_ALLEY]
        paths.append(json.dumps(key,ensure_ascii=False,separators=(',',':')))
        blocks.append(json.dumps([*key,(main-1)//100],ensure_ascii=False,separators=(',',':')))
    return pd.DataFrame({'addr_lane_path':pd.Series(paths,index=P.index,dtype=object),
        'addr_house_block100':pd.Series(blocks,index=P.index,dtype=object),
        'addr_detail_eligible':details.eligible.to_numpy(dtype='float32')},index=P.index)


def features_from_X(X):
    _need(isinstance(X,pd.DataFrame) and X.columns.is_unique, 'X columns must be unique')
    _need({'address_raw','district'}.issubset(X.columns), 'Required X fields: address_raw,district')
    if not len(X):
        return pd.DataFrame({c:pd.Series(index=X.index,dtype='float32' if c==FEATURE_COLUMNS[2] else object)
                             for c in FEATURE_COLUMNS},index=X.index)
    raw = X[['address_raw','district']].copy(deep=True)
    raw.columns = ['土地位置建物門牌','鄉鎮市區']
    return features_from_parsed(legacy.parse_geo_queries(raw))


def self_check(edge_cases=None):
    checks=[]
    def ok(name,condition):
        _need(bool(condition),'SELF_CHECK: '+name);checks.append(name)
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    for token,expected in [('0',0),('0001',1),('一〇一',101),('壹佰零貳',102),('十',10),('一十',10),
                            ('一百二十',120),('一千零一十',1010),('兩百零一',201)]:
        ok('strict_number_'+token,strict_number(token)==expected)
    for token in ['10一','一二十','一百二','一百一','一千一十','零十','百','12-18','-1','１２','',None]:
        reject('strict_number_reject_'+str(token),lambda token=token:strict_number(token))
    good_addresses = ['臺北市大安區中山路一段１２巷三弄１０號之１',
        '臺北市大安區中山路壹段壹佰零貳巷參弄壹佰零壹號',
        '臺北市大安區中山一路10號','臺北市大安區中山路十二之二巷三之四弄十號之六',
        '臺北市大安區中山路一十號','臺北市大安區中山路一〇一號',
        '臺北市大安區中山路12之1號','臺北市大安區中山路10號地下1樓',
        '臺北市大安區中山路10號，(地上二層之壹)','臺北市大安區中山路10號B2F',
        '臺北市大安區中山路10號3樓301室','臺北市大安區仁愛里中山路10號',
        '臺北市大安區中山路0段0巷0弄1號','臺北市大安區中山路10號0樓']
    bad_addresses = ['臺北市大安區中山路10一段12號','臺北市大安區中山路12之1號之2',
        '臺北市大安區中山路12-1巷10號','臺北市大安區中山路1弄10-18號',
        '臺北市大安區中山路10號3樓之2301室','臺北市大安區中山路10號((3樓))',
        '臺北市大安區中山路10號3樓,','臺北市大安區中山路10號,,3樓',
        '臺北市大安區中山路10號3樓及4樓','臺北市大安區中山路10號不明',
        '臺北市大安區中山路10號3樓一二十室','臺北市大安區中山路10號(臨時門牌)']
    addresses=good_addresses+bad_addresses
    idx=pd.Index(np.arange(len(addresses))*3+7,name='arbitrary_index')
    X=pd.DataFrame({'address_raw':addresses,'district':'大安區','untouched':np.arange(len(addresses),dtype='int32')},index=idx)
    original=X.copy(deep=True)
    raw=X[['address_raw','district']].rename(columns={'address_raw':'土地位置建物門牌','district':'鄉鎮市區'})
    P=legacy.parse_geo_queries(raw);P_before=P.copy(deep=True)
    d=eligibility_details(P);f=features_from_parsed(P)
    ok('normal_supported_grammar',d.eligible.iloc[:len(good_addresses)].all())
    ok('unsupported_grammar_rejected',not d.eligible.iloc[len(good_addresses):].any())
    ok('schema_dtypes',list(f)==FEATURE_COLUMNS and f.dtypes.astype(str).tolist()==['object','object','float32'])
    ok('parsed_and_X_identical',f.equals(features_from_X(X)))
    ok('input_values_dtypes_index_preserved',X.equals(original) and X.dtypes.equals(original.dtypes) and P.equals(P_before) and P.dtypes.equals(P_before.dtypes) and f.index.equals(idx))
    ok('ineligible_both_missing_flag0',f.iloc[len(good_addresses):][CATEGORICAL_COLUMNS].eq(MISSING).all().all()
       and f.addr_detail_eligible.iloc[len(good_addresses):].eq(0).all())
    ok('eligible_flag1',f.addr_detail_eligible.iloc[:len(good_addresses)].eq(1).all())
    ok('JSON_key_canonical',json.loads(f.addr_lane_path.iloc[0])==['臺北市','大安區','中山路1段','12','3'])
    ok('no_lane_and_no_alley_distinct_sentinels',json.loads(f.addr_lane_path.iloc[2])[-2:]==[NO_LANE,NO_ALLEY])
    ok('explicit_suffix_recorded',d.main_number.iloc[0]==10 and d.house_has_suffix.iloc[0] and d.house_suffix_number.iloc[0]==1)
    shuffled=X.iloc[::-1]
    ok('reordered_rows_preserved',features_from_X(shuffled).equals(f.iloc[::-1]))
    ok('batch_equals_sequence',pd.concat([features_from_X(X.iloc[[i]]) for i in range(len(X))]).equals(f))
    duplicate=X.iloc[:3].copy();duplicate.index=[4,4,2]
    ok('duplicate_index_preserved',features_from_X(duplicate).index.equals(duplicate.index))
    ok('empty_input',features_from_X(X.iloc[:0]).shape==(0,3))
    missing=X.iloc[:4].copy();missing.loc[:,'address_raw']=[None,'__MISSING__','土地位置建物門牌','']
    ok('missing_and_header_ineligible',features_from_X(missing).addr_detail_eligible.eq(0).all())
    bX=pd.DataFrame({'address_raw':['臺北市大安區中山路'+str(n)+'號' for n in (1,100,101,200,201)],'district':'大安區'})
    ok('bucket_boundaries_1_based', [json.loads(v)[-1] for v in features_from_X(bX).addr_house_block100]==[0,0,1,1,2])
    altered=P.iloc[:1].copy();altered.loc[:,'road1_lane']='999'
    ok('canonical_mismatch_rejected',not eligibility_details(altered).eligible.iloc[0])
    altered=P.iloc[:1].copy();roads=json.loads(altered.road_candidates_json.iloc[0]);roads[0]['context_uncertain']=True
    altered.loc[:,'road_candidates_json']=json.dumps(roads,ensure_ascii=False)
    ok('existing_context_guard_consistency',not eligibility_details(altered).eligible.iloc[0])
    reject('missing_X_column',lambda:features_from_X(X.drop(columns='district')))
    reject('missing_parsed_column',lambda:features_from_parsed(P.drop(columns='house_number')))
    edge_results=[]
    if edge_cases is not None:
        expected_true={'normal','number_in_road','floor_tail'}
        case_names=[r['observed_output']['case'] for r in edge_cases]
        ok('all_existing_22_artificial_cases',len(edge_cases)==22 and len(set(case_names))==22)
        frame=pd.DataFrame([r['input'] for r in edge_cases])
        ed=eligibility_details(legacy.parse_geo_queries(frame))
        for name,row in zip(case_names,ed.to_dict(orient='records')):
            ok('historical_artificial_'+name,row['eligible']==(name in expected_true))
            edge_results.append({'case':name,'eligible':row['eligible'],'reason':row['reason']})
    return {'status':'passed','synthetic_only':True,'real_inputs_read':False,'model_loaded':False,
            'training_or_scoring_performed':False,'check_count':len(checks),'checks':checks,
            'historical_artificial_cases':edge_results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--self-check-only',action='store_true',required=True)
    cli.add_argument('--artificial-cases',type=Path)
    args=cli.parse_args()
    casefile=None if args.artificial_cases is None else json.loads(args.artificial_cases.read_text(encoding='utf-8-sig'))
    print(json.dumps(self_check(None if casefile is None else casefile['cases']),ensure_ascii=True,allow_nan=False))
