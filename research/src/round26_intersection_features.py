"""R26 pure-X explicit two-road text key, not a verified physical intersection.

No data cache, target, model, geometry or vocabulary is read or fitted. The
original parser and the original model features are not modified.
"""
from __future__ import annotations
import argparse
import json
import re
import numpy as np
import pandas as pd
import geo_address_audit as legacy
from round25_address_features import strict_number

MISSING = '__MISSING__'
FEATURE_COLUMNS = ['addr_explicit_road_pair']
CATEGORICAL_COLUMNS = FEATURE_COLUMNS.copy()
DETAIL_COLUMNS = ['eligible', 'reason', 'roads', 'key']
CONNECTORS = ('與', '和', '及', '、', '/')
SUFFIXES = ('交叉路口', '交叉口', '交口', '路口', '口', '交會處', '交叉處')
_NUM = r'[0-9零〇一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰仟]+'
_SECTION = re.compile('(' + _NUM + ')段')
_NEIGHBOUR_PREFIX = re.compile('^' + _NUM + '鄰')
_REQUIRED = [
    'address_normalized', 'address_remainder', 'city', 'district_full',
    'district_evidence', 'locality_prefix', 'repeated_city_prefix',
    'repeated_district_prefix', 'postal_prefix_present', 'is_embedded_header',
    'land_parcel_context', 'near_or_opposite_or_unassigned', 'house_number_range',
    'district_prefix_conflict', 'admin_query_eligible', 'road_query_eligible',
    'road_parse_uncertain', 'road_count', 'road_mentions_count',
    'road_candidates_json', 'road1_name', 'road2_name',
]
RULES = {
    'feature_columns': FEATURE_COLUMNS,
    'categorical_columns': CATEGORICAL_COLUMNS,
    'feature_dtypes': {'addr_explicit_road_pair': 'object'},
    'source_X_columns': ['address_raw', 'district'],
    'legacy_parser': 'geo_address_audit.parse_geo_queries',
    'number_validator': 'round25_address_features.strict_number',
    'normalization': 'legacy NFKC; remove whitespace; 台 to 臺; no historical administrative renaming',
    'key': 'JSON [city,district_full,*sorted(two distinct complete road_name strings)]',
    'complete_grammar': 'exact administrative/locality prefixes, ROAD1+CONNECTOR+ROAD2+SUFFIX; exactly one complete reconstruction',
    'connectors': list(CONNECTORS),
    'suffixes': list(SUFFIXES),
    'section': 'absent OR original strict positive numeral token followed by 段; must match stored canonical section and road name',
    'road_name_numbers': 'preserve numbers inside the road base; 中山一路 is not 中山路1段',
    'eligibility': 'exactly two distinct road names and two mentions; zero raw HOUSE occurrences and no lane/alley; valid admin/road and no header/land/near/range/conflict/uncertainty',
    'neighbour_prefix': 'reject an explicit leading NUM鄰 in the post-locality text or either road base; do not strip it or globally alter 里/鄰',
    'legacy_connector_limitation': '和 is in the grammar, but when the old parser absorbs it into road_base and exact reconstruction fails, return missing; never strip 和平路 or claim all 和 sentences are supported',
    'ineligible': MISSING,
    'existing_columns_modified': False,
    'learned_or_fit_state': False,
    'fit_support_filter': False,
    'query_vocabulary_used': False,
    'target_or_query_labels_used': False,
    'physical_intersection_or_address_existence_verified': False,
    'new_OSM_or_external_data_access': False,
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _record(reason, roads=(), key=MISSING):
    return {'eligible': reason == 'eligible', 'reason': reason, 'roads': roads, 'key': key}


def _body(row):
    """Verify the parser's prefix removal against normalized original text."""
    text = legacy.normalize_text(row['address_normalized'])
    postal = re.match(r'^[0-9]{3,6}(?=[\u4e00-\u9fff])', text)
    _need(bool(postal) == row['postal_prefix_present'], 'postal_prefix_mismatch')
    if postal:
        text = text[postal.end():]
    city, district = row['city'], row['district_full']
    _need(isinstance(city, str) and city and isinstance(district, str) and district,
          'missing_admin')
    _need(text.startswith(city), 'city_prefix_mismatch')
    text = text[len(city):]
    if row['repeated_city_prefix']:
        _need(text.startswith(city), 'repeated_city_prefix_mismatch')
        text = text[len(city):]
    if text.startswith(district):
        text = text[len(district):]
    else:
        _need(row['district_evidence'] == 'supplied_column_only', 'district_prefix_mismatch')
    if row['repeated_district_prefix']:
        _need(text.startswith(district), 'repeated_district_prefix_mismatch')
        text = text[len(district):]
    _need(text == row['address_remainder'], 'administrative_remainder_mismatch')
    locality = row['locality_prefix']
    _need(isinstance(locality, str) and text.startswith(locality), 'locality_prefix_mismatch')
    text = text[len(locality):]
    _need(_NEIGHBOUR_PREFIX.match(text) is None, 'neighbour_prefix_contamination')
    return text


def _road(road):
    _need(isinstance(road, dict), 'invalid_road_record')
    _need(road.get('context_uncertain') is False and road.get('section_unsupported') is False,
          'road_context_uncertain')
    _need(all(road.get(k) == '' for k in ('lane_raw', 'lane_number', 'alley_raw', 'alley_number', 'house_number')),
          'lane_alley_or_house_present')
    base, raw_section = road.get('road_base'), road.get('section_raw')
    _need(isinstance(base, str) and base and isinstance(raw_section, str), 'invalid_road_strings')
    _need(_NEIGHBOUR_PREFIX.match(base) is None, 'neighbour_prefix_contamination')
    section = ''
    if raw_section:
        match = _SECTION.fullmatch(raw_section)
        _need(match is not None, 'unsupported_section_token')
        number = strict_number(match[1])
        _need(number > 0, 'nonpositive_section')
        section = str(number)
    canonical = base + (section + '段' if section else '')
    _need(road.get('section_number') == section and road.get('road_name') == canonical,
          'canonical_road_section_mismatch')
    return base + raw_section, canonical


def _eligibility(row):
    for column, reason in (
        ('is_embedded_header', 'embedded_header'), ('land_parcel_context', 'land_context'),
        ('near_or_opposite_or_unassigned', 'near_or_unassigned'),
        ('house_number_range', 'number_range'), ('district_prefix_conflict', 'admin_conflict'),
        ('road_parse_uncertain', 'road_parse_uncertain'),
    ):
        if row[column]:
            return _record(reason)
    if not row['admin_query_eligible'] or not row['road_query_eligible']:
        return _record('legacy_admin_or_road_ineligible')
    if row['road_count'] != 2 or row['road_mentions_count'] != 2:
        return _record('not_exactly_two_roads_and_mentions')
    rest = legacy.normalize_text(row['address_remainder'])
    if any(legacy.HOUSE.finditer(rest)):
        return _record('raw_house_token_present')
    try:
        roads = json.loads(row['road_candidates_json'])
        _need(isinstance(roads, list) and len(roads) == 2, 'invalid_road_records')
        raw_and_canonical = [_road(road) for road in roads]
        names = [item[1] for item in raw_and_canonical]
        _need(names == [row['road1_name'], row['road2_name']], 'road_field_mismatch')
        _need(len(set(names)) == 2, 'not_two_distinct_complete_road_names')
        body = _body(row)
        left, right = [item[0] for item in raw_and_canonical]
        reconstructions = sum(body == left + connector + right + suffix
                              for connector in CONNECTORS for suffix in SUFFIXES)
        _need(reconstructions == 1, 'not_unique_complete_pair_grammar')
        names = tuple(sorted(names))
        key = json.dumps([row['city'], row['district_full'], *names],
                         ensure_ascii=False, separators=(',', ':'))
        return _record('eligible', names, key)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        known = str(exc)
        return _record(known if known in {
            'postal_prefix_mismatch', 'missing_admin', 'city_prefix_mismatch',
            'repeated_city_prefix_mismatch', 'district_prefix_mismatch',
            'repeated_district_prefix_mismatch', 'administrative_remainder_mismatch',
            'locality_prefix_mismatch', 'neighbour_prefix_contamination', 'invalid_road_record',
            'road_context_uncertain', 'lane_alley_or_house_present', 'invalid_road_strings',
            'unsupported_section_token', 'nonpositive_section', 'canonical_road_section_mismatch',
            'invalid_road_records', 'road_field_mismatch', 'not_two_distinct_complete_road_names',
            'not_unique_complete_pair_grammar',
        } else 'invalid_number_or_parsed_record')


def eligibility_details(P):
    """Preserve index/order; rejected rows have roads=() and key=__MISSING__."""
    _need(isinstance(P, pd.DataFrame) and P.columns.is_unique, 'Parsed columns must be unique')
    _need(set(_REQUIRED).issubset(P.columns), 'Missing parsed address columns')
    rows = [_eligibility(dict(zip(_REQUIRED, values)))
            for values in P[_REQUIRED].itertuples(index=False, name=None)]
    result = pd.DataFrame(rows, index=P.index, columns=DETAIL_COLUMNS)
    result['eligible'] = result['eligible'].astype(bool)
    for column in ('reason', 'roads', 'key'):
        result[column] = result[column].astype(object)
    return result


def features_from_parsed(P):
    details = eligibility_details(P)
    return pd.DataFrame({FEATURE_COLUMNS[0]: details['key'].copy()}, index=P.index)


def features_from_X(X):
    _need(isinstance(X, pd.DataFrame) and X.columns.is_unique, 'X columns must be unique')
    _need({'address_raw', 'district'}.issubset(X.columns), 'Required X fields: address_raw,district')
    if not len(X):
        return pd.DataFrame({FEATURE_COLUMNS[0]: pd.Series(index=X.index, dtype=object)}, index=X.index)
    raw = X[['address_raw', 'district']].copy(deep=True)
    raw.columns = ['土地位置建物門牌', '鄉鎮市區']
    return features_from_parsed(legacy.parse_geo_queries(raw))


def self_check():
    checks = []
    def ok(name, condition):
        _need(bool(condition), 'SELF_CHECK: ' + name)
        checks.append(name)
    addresses = [
        '臺北市大安區中山路一段與民生路二段交叉路口',
        '臺北市大安區民生路２段與中山路１段交叉路口',
        '臺北市大安區中山一路及和平路交口',
        '臺北市大安區中山路壹段、民生路貳段路口',
        '臺北市大安區中山路/民生路交會處',
        '臺北市大安區中山路與民生路交叉處',
        '臺北市大安區中山路與民生路交叉口',
        '臺北市大安區中山路與民生路口',
        '臺北市大安區保安里中山路與民生路口',
    ]
    X = pd.DataFrame({'address_raw': addresses, 'district': ['大安區'] * len(addresses),
                      'untouched_numeric': np.arange(len(addresses), dtype='float32')},
                     index=pd.Index(range(101, 101 + len(addresses)), name='original_row'))
    before = X.copy(deep=True)
    raw = X[['address_raw', 'district']].rename(columns={'address_raw': '土地位置建物門牌', 'district': '鄉鎮市區'})
    P = legacy.parse_geo_queries(raw)
    old = P.copy(deep=True)
    D, F = eligibility_details(P), features_from_parsed(P)
    ok('allowed_connectors_suffixes_fullwidth_and_financial', D.eligible.all())
    ok('unordered_pair_and_strict_section_equivalence', F.iloc[0, 0] == F.iloc[1, 0] == F.iloc[3, 0])
    ok('numbers_inside_road_name_and_和平_preserved', D.roads.iloc[2] == ('中山一路', '和平路'))
    ok('single_object_category_only', list(F) == FEATURE_COLUMNS and str(F.dtypes.iloc[0]) == 'object')
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_frame_equal(P, old)
    ok('input_values_dtypes_index_order_unchanged', True)
    pd.testing.assert_frame_equal(F, features_from_X(X))
    ok('X_adapter_same_two_columns_only', True)
    pd.testing.assert_frame_equal(F.iloc[::-1], features_from_X(X.iloc[::-1]))
    pd.testing.assert_frame_equal(F, pd.concat([features_from_X(X.iloc[[i]]) for i in range(len(X))]))
    ok('reorder_and_batch_sequence_equal', True)
    duplicates = X.iloc[:2].copy(); duplicates.index = pd.Index([7, 7], name='duplicates')
    ok('duplicate_index_preserved', features_from_X(duplicates).index.equals(duplicates.index))
    ok('empty_frame_object_schema', features_from_X(X.iloc[:0]).dtypes.iloc[0] == object)
    bad = [
        ('single_road', '中山路口'), ('three_roads', '中山路與民生路及和平路交叉口'),
        ('duplicate_mentions', '中山路與民生路與中山路交叉口'),
        ('same_road', '中山路與中山路交叉口'), ('omitted_connector', '中山路民生路交叉口'),
        ('unknown_connector', '中山路暨民生路交叉口'), ('extra_tail', '中山路與民生路交叉口附近'),
        ('unrecognized_near_tail', '中山路與民生路口旁'),
        ('parcel_mixture', '中山路與民生路交叉口仁愛段12地號'),
        ('repeated_same_house', '中山路10號及民生路10號路口'),
        ('lane', '中山路12巷與民生路路口'), ('alley', '中山路2弄與民生路路口'),
        ('range', '中山路10至20號與民生路交叉口'),
        ('mixed_section_digits', '中山路10一段與民生路交叉口'),
        ('noncanonical_section', '中山路一百二段與民生路交叉口'),
        ('zero_section', '中山路零段與民生路交叉口'),
        ('directional_section', '中山路北段與民生路交叉口'),
        ('elided_section', '中山路一段與二段路口'),
        ('neighbour_prefix', '保安里12鄰中山路與民生路口'),
        ('legacy_和_absorbed_is_not_repaired', '中山路和民生路交叉口'),
    ]
    B = pd.DataFrame({'address_raw': ['臺北市大安區' + a for _, a in bad], 'district': '大安區'})
    rejected = features_from_X(B).iloc[:, 0].eq(MISSING)
    for (name, _), rejected_value in zip(bad, rejected):
        ok('reject_' + name, rejected_value)
    conflict = X.iloc[:1].copy(); conflict.loc[:, 'district'] = '信義區'
    ok('admin_conflict', features_from_X(conflict).iloc[0, 0] == MISSING)
    corrupted = P.iloc[:1].copy(); corrupted.loc[:, 'address_remainder'] += '不明'
    ok('original_text_alignment', not eligibility_details(corrupted).eligible.iloc[0])
    corrupted = P.iloc[:1].copy(); corrupted.loc[:, 'road_mentions_count'] = 3
    ok('mention_count_before_dedup', not eligibility_details(corrupted).eligible.iloc[0])
    corrupted = P.iloc[:1].copy(); corrupted.loc[:, 'is_embedded_header'] = True
    ok('embedded_header', not eligibility_details(corrupted).eligible.iloc[0])
    missing = pd.DataFrame({'address_raw': [None, '__MISSING__'], 'district': ['大安區', None]})
    ok('missing_values', features_from_X(missing).iloc[:, 0].eq(MISSING).all())
    return {'all_checks_passed': True, 'check_count': len(checks), 'checks': checks,
            'synthetic_only': True, 'real_data_read': False, 'model_fit_or_score': False,
            'legacy_parser_changed': False, 'R25_full_self_check_rerun': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check-only', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=True), flush=True)
