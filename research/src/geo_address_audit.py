"""Pure-X address parsing and offline OSM query preparation.

Imports require only stdlib, pandas and numpy. Workbook reading is CLI-only.
No price column, external data, model or prediction is loaded by this module.
Original rows/IDs (including embedded headers and duplicates) are preserved.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs/geography/address_audit'
GEO_COLUMNS = ['編號', '鄉鎮市區', '交易標的', '土地位置建物門牌']
CN = '零〇一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰'
NUM = r'[0-9' + CN + r']+'
CITY = re.compile(r'^([\u4e00-\u9fff]{2,3}?[縣市])')
DISTRICT = re.compile(r'^([\u4e00-\u9fff]{1,5}?區|[\u4e00-\u9fff]{1,5}?[鄉鎮市])')
ROAD = re.compile(r'([\u4e00-\u9fffA-Za-z0-9·]{1,24}?(?:大道|路|街))')
SECTION = re.compile(r'^(' + NUM + r')段')
LANE = re.compile(r'^(' + NUM + r'(?:[-之]' + NUM + r')?)巷')
ALLEY = re.compile(r'^(' + NUM + r'(?:[-之]' + NUM + r')?)弄')
HOUSE = re.compile(r'(' + NUM + r'(?:[-之]' + NUM + r')?)號(?:之(' + NUM + r'))?')
MISSING = {'', '__MISSING__', 'nan', 'NaN', 'None', 'null', '<NA>', 'NaT'}
RULES = {
    'normalization': 'NFKC; remove whitespace; 台 to 臺; no historical city/district renaming',
    'district': 'full supplied prefix first; complete address suffix fallback; conflicts flagged',
    'road_section': 'normalize Chinese numbers only after 大道/路/街 immediately before 段',
    'road_in_name_numbers': 'preserved: 中山一路 is not 中山1路 or 中山路1段',
    'multi_road': 'retain every distinct full road/section; unordered pair only for exactly two',
    'parcel': 'separate cadastral section/subsection; never treat land number as house number',
    'matching_status': 'query preparation only; no OSM feature matched by this script',
    'row_policy': 'keep original rows, embedded headers, duplicate rows and IDs',
    'target_policy': 'read_excel usecols explicitly restricts to four non-price columns',
}


def normalize_text(value) -> str:
    """Matching text only; preserve original text in separate provenance columns."""
    if value is None or pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).strip()
    if text in MISSING:
        return ''
    return re.sub(r'\s+', '', text).replace('台', '臺')


def normalize_admin_name(value) -> str:
    """Keep the complete administrative suffix and do not guess historical aliases."""
    return normalize_text(value)


def _number(text: str) -> str:
    if text.isdigit():
        return str(int(text))
    digit = dict(zip('零〇一二三四五六七八九', [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    digit.update(dict(zip('兩壹貳參肆伍陸柒捌玖', [2, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    units = {'十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000}
    if not text or any(c not in digit and c not in units for c in text):
        return text
    if not any(c in units for c in text):
        return str(int(''.join(str(digit[c]) for c in text)))
    total, current = 0, 0
    for char in text:
        if char in digit:
            current = digit[char]
        else:
            total += (current or 1) * units[char]
            current = 0
    return str(total + current)


def normalize_road_name(value) -> str:
    """Canonical OSM/source name; 中山路一段 == 中山路1段, 中山一路 unchanged."""
    text = normalize_text(value)
    return re.sub(r'((?:大道|路|街))(' + NUM + r')段',
                  lambda m: m[1] + _number(m[2]) + '段', text)


def _number_suffix(text: str) -> str:
    return '-'.join(_number(part) for part in re.split('[-之]', text))


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


@lru_cache(maxsize=160000)
def _parse(address: str, supplied: str) -> dict:
    rest = address
    postal = re.match(r'^\d{3,6}(?=[\u4e00-\u9fff])', rest)
    if postal:
        rest = rest[postal.end():]
    city_match = CITY.match(rest)
    city = city_match[1] if city_match else ''
    if city_match:
        rest = rest[city_match.end():]
    repeated_city = bool(city and rest.startswith(city))
    if repeated_city:
        rest = rest[len(city):]
    supplied_is_city = bool(city and supplied == city)
    supplied_ok = bool(re.fullmatch(r'[\u4e00-\u9fff]{1,6}[區鄉鎮市]', supplied)) and not supplied_is_city
    dm = DISTRICT.match(rest)
    address_district = dm[1] if dm else ''
    prefix_match = bool(supplied_ok and rest.startswith(supplied))
    conflict = bool(supplied_ok and not prefix_match and address_district and address_district != supplied)
    if prefix_match:
        district, evidence = supplied, 'supplied_prefix_exact'
        rest = rest[len(supplied):]
    elif supplied_ok:
        district, evidence = supplied, 'supplied_conflict' if conflict else 'supplied_column_only'
        if dm:
            rest = rest[dm.end():]
    elif dm:
        district, evidence = address_district, 'address_suffix_only'
        rest = rest[dm.end():]
    else:
        district, evidence = '', 'unknown'
    repeated_district = bool(district and rest.startswith(district))
    if repeated_district:
        rest = rest[len(district):]
    locality = re.match(r'^([\u4e00-\u9fff]{1,8}?[里村])(?=.{2,}(?:大道|路|街))', rest)
    locality_prefix = locality[1] if locality else ''
    road_text = rest[locality.end():] if locality else rest
    land_explicit = bool(re.search(r'地號|小段', rest))
    # A named segment without an immediately preceding road is cadastral, not road section.
    parcel = re.match(r'^([\u4e00-\u9fff0-9]{1,15}?段)(?:([\u4e00-\u9fff0-9]{1,12}?小段))?', rest)
    parcel_has_road = bool(parcel and re.search(r'大道|路|街', parcel[1]))
    parcel_road_word_is_name = bool(parcel and len(re.findall(r'大道|路|街', parcel[1])) == 1
                                  and re.search(r'(?:大道|路|街)段$', parcel[1]))
    parcel_section = parcel[1] if parcel and (not parcel_has_road or parcel_road_word_is_name) else ''
    parcel_subsection = (parcel[2] or '') if parcel_section else ''
    land_context = bool(land_explicit or parcel_section)
    starts_cadastral = bool(parcel_section)
    intersection_explicit = bool(re.search(r'交叉|交會|交匯|交接|路口|街口', road_text))
    near_or_opposite = bool(re.search(r'附近|對面|旁邊|周邊|鄰近|近旁|無門牌|未編釘|暫編', rest))
    # Stop road extraction at a leading parcel name; embedded land descriptions remain context only.
    scan = '' if starts_cadastral else road_text
    scan = re.sub(r'交叉路口|交叉口|交會處|交匯處|交接處', '；', scan)
    roads = []
    position = 0
    uncertain = False
    while position < len(scan):
        match = ROAD.search(scan, position)
        if not match:
            break
        # Strip one explicit connector, never road-name characters such as 和 in 和平路.
        base = re.sub(r'^(?:與|及|暨|鄰近|近旁|位於)', '', match[1], count=1) if roads else match[1]
        end = match.end()
        section = SECTION.match(scan[end:])
        section_raw = section[0] if section else ''
        section_number = _number(section[1]) if section else ''
        if section:
            end += section.end()
        # Do not silently drop directional sections or elided lists/ranges of sections.
        # A following complete road name is another road, not an unsupported section.
        qualifier = re.match(r'^([^();；]{1,18}?段)', scan[end:])
        section_unsupported = bool(qualifier and not re.search(r'大道|路|街', qualifier[1]))
        uncertain |= section_unsupported
        lane = LANE.match(scan[end:])
        lane_raw = lane[0] if lane else ''
        if lane:
            end += lane.end()
        alley = ALLEY.match(scan[end:])
        alley_raw = alley[0] if alley else ''
        if alley:
            end += alley.end()
        house = HOUSE.match(scan[end:])
        house_number = (_number_suffix(house[1]) + ('-' + _number(house[2]) if house[2] else '')) if house else ''
        if house:
            end += house.end()
        position = end
        if base in {'路', '街', '大道', '交叉路', '路口', '交會路'} or len(base) < 2:
            continue
        bad_context = bool(re.search(r'地號|小段|[區鄉鎮]|號|巷|弄|[0-9]段', base))
        uncertain |= bad_context
        item = {'road_base': base, 'section_raw': section_raw, 'section_number': section_number,
                'road_name': normalize_road_name(base + section_raw),
                'lane_raw': lane_raw, 'lane_number': _number_suffix(lane[1]) if lane else '',
                'alley_raw': alley_raw, 'alley_number': _number_suffix(alley[1]) if alley else '',
                'house_number': house_number, 'section_unsupported': section_unsupported,
                'context_uncertain': bad_context or section_unsupported}
        roads.append(item)
    mentions = len(roads)
    distinct = []
    observed = set()
    for road in roads:
        key = (road['road_name'], road['lane_number'], road['alley_number'])
        if key not in observed:
            distinct.append(road)
            observed.add(key)
    house_numbers = []
    if not starts_cadastral:
        for m in HOUSE.finditer(rest):
            number = _number_suffix(m[1]) + ('-' + _number(m[2]) if m[2] else '')
            if number not in house_numbers:
                house_numbers.append(number)
    number_range = bool(re.search(NUM + r'(?:至|到|~|～)' + NUM + r'號', rest))
    count = len(distinct)
    multi = count > 1
    inferred_intersection = bool(count == 2 and not house_numbers and re.search(r'與|及|暨|和', rest))
    admin_ok = bool(city and district and not conflict)
    road_ready = bool(admin_ok and count and not uncertain)
    house_ready = bool(road_ready and count == 1 and len(house_numbers) == 1 and
                       distinct[0]['house_number'] and not land_context and not near_or_opposite and
                       not intersection_explicit and not number_range)
    status = ('admin_unknown_or_conflict' if not admin_ok else 'cadastral_only' if starts_cadastral else
              'road_unknown' if not count else 'road_parse_uncertain' if uncertain else
              'multi_road_more_than_two' if count > 2 else 'intersection_candidate' if count == 2 else
              'house_number_candidate' if house_ready else 'road_context_with_parcel' if land_context else
              'road_only')
    first = distinct[0] if distinct else {}
    second = distinct[1] if count >= 2 else {}
    return {'city': city, 'district_full': district, 'district_in_address': address_district,
            'district_evidence': evidence, 'district_prefix_conflict': conflict,
            'supplied_district_is_city': supplied_is_city, 'repeated_city_prefix': repeated_city,
            'repeated_district_prefix': repeated_district, 'postal_prefix_present': bool(postal),
            'locality_prefix': locality_prefix, 'address_remainder': rest,
            'parcel_section': parcel_section, 'parcel_subsection': parcel_subsection,
            'land_parcel_explicit': land_explicit, 'land_parcel_context': land_context,
            'starts_with_cadastral_section': starts_cadastral,
            'road_candidates_json': _json(distinct), 'road_count': count, 'road_mentions_count': mentions,
            'road1_base': first.get('road_base', ''), 'road1_section': first.get('section_number', ''),
            'road1_name': first.get('road_name', ''), 'road1_lane': first.get('lane_number', ''),
            'road1_alley': first.get('alley_number', ''), 'road2_name': second.get('road_name', ''),
            'house_numbers_json': _json(house_numbers), 'house_number': house_numbers[0] if len(house_numbers) == 1 else '',
            'house_number_count': len(house_numbers), 'missing_explicit_house_number': not bool(house_numbers),
            'multiple_house_numbers': len(house_numbers) > 1, 'house_number_range': number_range,
            'multi_road': multi, 'more_than_two_roads': count > 2, 'road_parse_uncertain': uncertain,
            'intersection_explicit': intersection_explicit, 'intersection_inferred': inferred_intersection,
            'near_or_opposite_or_unassigned': near_or_opposite,
            'admin_query_eligible': admin_ok, 'road_query_eligible': road_ready,
            'house_query_eligible': house_ready, 'query_precision_limit': status}


def build_osm_query_keys(parsed: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with exact road/section keys, explicit fallback keys and pair keys."""
    result = parsed.copy()
    full_keys, base_keys, pairs = [], [], []
    for row in parsed[['city', 'district_full', 'road_candidates_json', 'road_query_eligible']].itertuples(index=False, name=None):
        city, district, encoded, eligible = row
        roads = json.loads(encoded)
        prefix = city + '|' + district + '|'
        names = sorted(set(r['road_name'] for r in roads))
        full_keys.append(_json([prefix + name for name in names]) if eligible else '[]')
        base_keys.append(_json(sorted(set(prefix + r['road_base'] for r in roads))) if eligible else '[]')
        pairs.append(prefix + ' & '.join(names) if eligible and len(names) == 2 else '')
    result['road_query_keys_json'] = full_keys
    result['road_base_fallback_keys_json'] = base_keys
    result['intersection_query_key'] = pairs
    return result


def parse_geo_queries(raw_geo: pd.DataFrame) -> pd.DataFrame:
    """Pure parse; prices are rejected, IDs are provenance only, input/index remain unchanged."""
    if '總價元' in raw_geo or 'y' in raw_geo:
        raise ValueError('Pass only geographic X columns; price columns are forbidden.')
    required = {'土地位置建物門牌', '鄉鎮市區'}
    if not required.issubset(raw_geo):
        raise ValueError('Required columns: 土地位置建物門牌, 鄉鎮市區')
    addresses = raw_geo['土地位置建物門牌'].map(normalize_text)
    supplied = raw_geo['鄉鎮市區'].map(normalize_admin_name)
    records = [dict(_parse(a, d)) for a, d in zip(addresses, supplied)]
    result = pd.DataFrame.from_records(records, index=raw_geo.index)
    result.insert(0, 'source_row_index', np.arange(len(raw_geo), dtype='int64'))
    result.insert(1, 'source_excel_row', np.arange(len(raw_geo), dtype='int64') + 2)
    source_ids = raw_geo['編號'].copy(deep=True) if '編號' in raw_geo else pd.Series(None, index=raw_geo.index, dtype=object)
    result.insert(2, 'source_id', source_ids)
    result['source_address'] = raw_geo['土地位置建物門牌'].to_numpy(copy=True)
    result['source_district'] = raw_geo['鄉鎮市區'].to_numpy(copy=True)
    result['source_transaction_type'] = raw_geo['交易標的'].to_numpy(copy=True) if '交易標的' in raw_geo else None
    result['address_normalized'] = addresses
    result['supplied_district_normalized'] = supplied
    ids = raw_geo['編號'].fillna('').astype(str).str.strip().str.lower() if '編號' in raw_geo else pd.Series('', index=raw_geo.index)
    header = ids.eq('serial number') | addresses.str.lower().isin(['landsectorpositionbuildingsectorhousenumber','土地位置建物門牌'])
    result['is_embedded_header'] = header
    result['literal_tai_to_tai_normalization'] = raw_geo['土地位置建物門牌'].fillna('').astype(str).str.contains('台', regex=False)
    result.loc[header, ['admin_query_eligible', 'road_query_eligible', 'house_query_eligible']] = False
    result.loc[header, 'query_precision_limit'] = 'embedded_header_preserved'
    return build_osm_query_keys(result)


def self_check() -> dict:
    cases = [
        ('新市區', '臺南市新市區中正路一段10號'),
        ('前鎮區', '高雄市前鎮區一心一路10號'),
        ('平鎮區', '桃園市平鎮區中豐路二段10號'),
        ('大安區', '台北市大安區市民大道四段１００巷二弄１０號之１'),
        ('大園區', '桃園市大園區領航南路二段及致善路一段交叉路口'),
        ('清水區', '臺中市清水區大勇路大勇北路港都五路口市鎮北段399地號'),
        ('中正區', '臺北市中正區中山路段123-4地號'),
        ('大安區', '臺北市大安區中山路一段附近'),
        ('北區', '新竹市新竹市北區中正路10號'),
        ('新竹市', '新竹市新竹市東光段238-2地號'),
        ('新莊區', '新北市新莊區福前街63巷、中原東路89巷交叉路口'),
        ('大安區', '臺北市信義區松仁路10號'),
        ('大安區', '臺北市大安區中山路一段1號及中山路二段2號'),
        ('大安區', '臺北市大安區中山路一段1至99號'),
        ('大安區', '臺北市大安區和平路一段99號'),
        ('大安區', '臺北市大安區中正路與和平路二段交叉口'),
        ('大安區', '臺北市大安區近江路10號'),
        ('大安區', '臺北市大安區仁愛段一小段123-4地號'),
        ('大安區', '臺北市大安區及第路10號'),
        ('大安區', '臺北市大安區中山路一段與二段'),
        ('大安區', '臺北市大安區中山路一、二段'),
        ('大安區', '臺北市大安區中華路北段10號'),
        ('大安區', '臺北市大安區中華路南段10號'),
        ('新店區', '新北市新店區及人路10號'),
    ]
    raw = pd.DataFrame({'編號': list(range(len(cases))), '鄉鎮市區': [x[0] for x in cases],
                        '土地位置建物門牌': [x[1] for x in cases]}, index=np.arange(len(cases)) + 8)
    original = raw.copy(deep=True)
    p = parse_geo_queries(raw).reset_index(drop=True)
    assert p.district_full.iloc[:3].tolist() == ['新市區', '前鎮區', '平鎮區']
    assert p.road1_name.iloc[:4].tolist() == ['中正路1段', '一心一路', '中豐路2段', '市民大道4段']
    assert p.iloc[3].house_number == '10-1' and p.iloc[3].road1_lane == '100' and p.iloc[3].road1_alley == '2'
    assert p.iloc[4].road_count == 2 and not p.iloc[4].house_query_eligible
    assert p.iloc[5].road_count == 3 and p.iloc[5].more_than_two_roads
    assert p.iloc[6].starts_with_cadastral_section and p.iloc[6].road_count == 0 and p.iloc[6].house_number_count == 0
    assert p.iloc[7].road_query_eligible and not p.iloc[7].house_query_eligible
    assert p.iloc[8].district_full == '北區' and p.iloc[8].repeated_city_prefix
    assert not p.iloc[9].admin_query_eligible and p.iloc[9].supplied_district_is_city
    assert p.iloc[10].road_count == 2 and p.iloc[10].road1_lane == '63'
    assert p.iloc[11].district_prefix_conflict and not p.iloc[11].road_query_eligible
    assert p.iloc[12].road_count == 2 and p.iloc[12].multiple_house_numbers
    assert p.iloc[13].house_number_range and not p.iloc[13].house_query_eligible
    assert p.iloc[14].road1_name == '和平路1段'
    assert p.iloc[15].road2_name == '和平路2段'
    assert p.iloc[16].road1_name == '近江路'
    assert p.iloc[17].parcel_section == '仁愛段' and p.iloc[17].parcel_subsection == '一小段'
    assert not p.iloc[17].road_query_eligible and p.iloc[17].house_number_count == 0
    assert p.iloc[18].road1_name == '及第路'
    assert p.iloc[19:23].road_parse_uncertain.all() and not p.iloc[19:23].road_query_eligible.any()
    assert p.iloc[19:23].road_query_keys_json.eq('[]').all()
    assert p.iloc[23].road1_name == '及人路' and p.iloc[23].road_query_eligible
    assert normalize_road_name(' 中山路 一段 ') == normalize_road_name('中山路1段')
    assert normalize_road_name('中山一路') == '中山一路' != normalize_road_name('中山路一段')
    pd.testing.assert_frame_equal(raw, original)
    alternate = raw.copy(); alternate['編號'] += 10000
    q = parse_geo_queries(alternate).reset_index(drop=True)
    pd.testing.assert_frame_equal(p.drop(columns='source_id'), q.drop(columns='source_id'))
    return {'status': 'passed', 'cases': len(cases), 'input_unchanged': True, 'id_not_a_parser_feature': True,
            'all_roads_retained': True, 'land_and_house_numbers_separate': True, 'model_training': False}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_geo_workbook(path: Path) -> pd.DataFrame:
    """Read four explicitly selected X/provenance columns; openpyxl needed only here."""
    frame = pd.read_excel(path, usecols=GEO_COLUMNS, dtype=object, engine='openpyxl')
    if set(frame.columns) != set(GEO_COLUMNS):
        raise ValueError('Unexpected workbook geographic columns')
    return frame[GEO_COLUMNS]


def _counts(values: pd.Series) -> list[dict]:
    return [{'value': str(key), 'rows': int(count)} for key, count in values.value_counts(dropna=False).items()]


def audit_split(split: str, source: Path, output: Path) -> tuple[dict, pd.DataFrame]:
    from house_features import _cat, _text
    from round3_features import ADDRESS_COLUMNS, correct_address_features
    print(json.dumps({'loading_geographic_columns': split}, ensure_ascii=True), flush=True)
    source_before = file_sha(source)
    geo_input_path = output / f'{split}_geographic_input.pkl'
    geo_input_meta = output / f'{split}_geographic_input.json'
    geography_input_reused = geo_input_path.exists() and geo_input_meta.exists()
    if geography_input_reused:
        saved = json.loads(geo_input_meta.read_text(encoding='utf-8'))
        if saved['source_sha256'] != source_before or saved['columns'] != GEO_COLUMNS or saved['input_sha256'] != file_sha(geo_input_path):
            raise ValueError('Stale geography-only input cache; use a new output directory')
        raw = pd.read_pickle(geo_input_path)
    else:
        raw = read_geo_workbook(source)
        raw.to_pickle(geo_input_path)
        geo_input_meta.write_text(json.dumps({'source_sha256': source_before, 'columns': GEO_COLUMNS,
            'rows': len(raw), 'input_sha256': file_sha(geo_input_path), 'price_column_loaded': False},
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert list(raw.columns) == GEO_COLUMNS
    parsed = parse_geo_queries(raw)
    assert len(parsed) == len(raw) and parsed.source_row_index.tolist() == list(range(len(raw)))
    pd.testing.assert_series_equal(parsed.source_id, raw['編號'], check_names=False)
    header = parsed.is_embedded_header.to_numpy()
    clean_geo = raw.loc[~header].reset_index(drop=True)
    cache_path = ROOT / f'data_cache/{split}_features.pkl'
    cache = pd.read_pickle(cache_path)  # This file contains X only, never ID or y.
    assert len(cache) == len(clean_geo) and '總價元' not in cache and '編號' not in cache
    expected_address = _cat(_text(clean_geo['土地位置建物門牌']).str.replace('台', '臺', regex=False))
    pd.testing.assert_series_equal(expected_address, cache.address_raw, check_names=False)
    pd.testing.assert_series_equal(_cat(clean_geo['鄉鎮市區']), cache.district, check_names=False)
    pd.testing.assert_series_equal(_cat(clean_geo['交易標的']), cache.transaction_type, check_names=False)
    cache_index = np.full(len(raw), -1, dtype='int64'); cache_index[~header] = np.arange(len(cache))
    parsed['existing_feature_cache_row_index'] = cache_index
    mini = cache[['district', *ADDRESS_COLUMNS]].copy()
    corrected = correct_address_features(clean_geo, mini)
    old_vs_r3 = {name: int((mini[name] != corrected[name]).sum()) for name in ADDRESS_COLUMNS}
    usable = parsed.loc[~header].reset_index(drop=True)
    difference = usable.district_full.ne(corrected.address_district.replace('__MISSING__', '')).to_numpy()
    flags = [c for c in parsed if pd.api.types.is_bool_dtype(parsed[c])]
    all_flag_counts = {c: int(parsed[c].sum()) for c in flags}
    candidate_rows = []
    for row in parsed.loc[~header, ['city', 'district_full', 'road_candidates_json', 'road_query_eligible', 'land_parcel_context']].itertuples(index=False, name=None):
        city, district, encoded, eligible, land = row
        if eligible:
            for road in json.loads(encoded):
                candidate_rows.append({'city': city, 'district_full': district, 'road_base': road['road_base'],
                    'road_name': road['road_name'], 'section_number': road['section_number'],
                    'query_key': city + '|' + district + '|' + road['road_name'], 'split': split,
                    'land_context': int(land)})
    query_table = pd.DataFrame(candidate_rows)
    query_counts = query_table.groupby(['split', 'city', 'district_full', 'road_base', 'road_name', 'section_number', 'query_key'], dropna=False).agg(
        rows=('query_key', 'size'), land_context_rows=('land_context', 'sum')).reset_index() if len(query_table) else query_table
    single_exact = parsed.road_count.eq(1) & parsed.road_query_eligible & ~parsed.land_parcel_context & ~parsed.is_embedded_header
    single_exact_keys = parsed.loc[single_exact, 'city'] + '|' + parsed.loc[single_exact, 'district_full'] + '|' + parsed.loc[single_exact, 'road1_name']
    path = output / f'{split}_parsed.pkl'
    parsed.to_pickle(path)
    # Verify saved row provenance, the full schema and actual values, without touching source/cache.
    reloaded = pd.read_pickle(path)
    pd.testing.assert_frame_equal(reloaded, parsed)
    assert file_sha(source) == source_before
    report = {'split': split, 'source': str(source.relative_to(ROOT)), 'source_sha256': source_before,
        'read_columns': GEO_COLUMNS, 'price_column_loaded': False, 'original_rows': len(raw),
        'geography_input_reused': geography_input_reused,
        'geography_input_path': str(geo_input_path.relative_to(ROOT)),
        'geography_input_sha256': file_sha(geo_input_path),
        'single_road_exact_nonparcel_rows': int(single_exact.sum()),
        'single_road_exact_nonparcel_distinct_keys': int(single_exact_keys.nunique()),
        'embedded_header_rows': int(header.sum()), 'embedded_header_excel_rows': parsed.loc[header, 'source_excel_row'].tolist(),
        'rows_preserved_in_output': len(parsed), 'existing_feature_cache_rows': len(cache),
        'exact_alignment_to_header_clean_feature_cache': True,
        'cache_alignment_columns': ['address_raw', 'district', 'transaction_type'],
        'post_training_dedup_applied': False, 'duplicate_ids_preserved_excess': int(parsed.source_id.duplicated().sum()),
        'source_id_missing': int(parsed.source_id.isna().sum()),
        'flags': all_flag_counts, 'precision_status_counts': _counts(parsed.query_precision_limit),
        'city_counts': _counts(parsed.city), 'supplied_district_counts': _counts(parsed.supplied_district_normalized),
        'district_evidence_counts': _counts(parsed.district_evidence),
        'road_count_distribution': _counts(parsed.road_count),
        'query_road_distinct_keys': int(query_table.query_key.nunique()) if len(query_table) else 0,
        'legacy_to_round3_address_changes': old_vs_r3,
        'new_district_differs_from_round3_rows': int(difference.sum()),
        'new_vs_round3_district_difference_groups': usable.loc[difference, ['city', 'supplied_district_normalized', 'district_full', 'district_evidence']].value_counts().reset_index(name='rows').head(30).to_dict(orient='records'),
        'parsed_path': str(path.relative_to(ROOT)), 'parsed_sha256': file_sha(path),
        'parsed_columns': list(parsed.columns), 'parsed_dtypes': {c: str(t) for c, t in parsed.dtypes.items()},
        'house_candidate_is_not_a_verified_coordinate': True, 'osm_match_performed': False}
    print(json.dumps({'split_completed': split, 'rows': len(parsed), 'flags': all_flag_counts,
                      'unique_road_keys': report['query_road_distinct_keys']}, ensure_ascii=True), flush=True)
    _parse.cache_clear()
    return report, query_counts


def write_report(result: dict, path: Path) -> None:
    lines = ['# 離線地理查詢前置：地址純 X 稽核', '',
    '本次只解析 supplied 原始活頁簿的編號、鄉鎮市區、交易標的、土地位置建物門牌四欄；編號僅供回連原列。未載入價格欄、未訓練模型或計分，也未下載外部成交資料。下列成功數是「能形成查詢鍵」，尚未執行 OSM 道路匹配，不能當成定位正確率。', '',
    '## 原列保留與可用查詢', '',
    '| split | 原始列數 | 保留表頭列 | 可組行政區＋道路鍵 | 單一門牌候選 | 多道路 | 三條以上道路 | 地籍內容 | 無明確門牌 | 不同道路鍵 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['splits']:
        f=r['flags']; lines.append(f"| {r['split']} | {r['original_rows']:,} | {r['embedded_header_rows']:,} | {f['road_query_eligible']:,} | {f['house_query_eligible']:,} | {f['multi_road']:,} | {f['more_than_two_roads']:,} | {f['land_parcel_context']:,} | {f['missing_explicit_house_number']:,} | {r['query_road_distinct_keys']:,} |")
    lines += ['', '| split | 單一道路、非地籍精確鍵候選列 | 占原始列比例 | 不同精確鍵 |', '|---|---:|---:|---:|']
    for r in result['splits']:
        lines.append(f"| {r['split']} | {r['single_road_exact_nonparcel_rows']:,} | {100*r['single_road_exact_nonparcel_rows']/r['original_rows']:.2f}% | {r['single_road_exact_nonparcel_distinct_keys']:,} |")
    lines += ['', '第二表採用後續首輪 matcher 的固定條件：road_count==1、road_query_eligible、非 land_parcel_context、非表頭。它仍然只是道路名精確鍵候選，沒有門牌插值。', '', '分母保留所有原始資料列，包含夾入表頭；這些表頭列不會成為查詢。完全重複資料與重複 ID 均未刪除，沒有重新排序。輸出 source_row_index 從 0 起算，source_excel_row＝source_row_index＋2，source_id 保留原編號。existing_feature_cache_row_index 對應既有清除英文表頭、尚未移除訓練重複列的 X cache；表頭為 -1。',
    '已逐列核對地址、原行政區、交易標的與既有純 X cache 完全一致。root 可先用 existing_feature_cache_row_index 回連原 cache，再套既有的訓練去重／時間切分索引；本程式不讀 y，因此不自行重建包含價格的 529 列去重規則。', '',
    '## 行政區、縣市名稱與舊解析', '',
    'NFKC、移除空白與台→臺只作用於匹配文字；原始地址與原始行政區另欄保留。优先以 supplied 鄉鎮市區的完整前綴切分地址，所以新市區、前鎮區、平鎮區不會截成新市、前鎮、平鎮。地址缺行政區時可保留 supplied 完整欄位，但另標 supplied_column_only；地址與欄位衝突不輸出可信道路查詢。',
    '重複縣市／行政區前綴會單獨標記；supplied 行政區其實等於縣市（如新竹市）不冒充鄉鎮市區。沒有證據的完整行政區保持未知。舊縣市文字保留原名，不自行將桃園縣及舊鄉鎮市改成現今行政名稱；須待離線地理資料提供可核對 alias／邊界再處理。', '',
    '| split | 舊87欄→R3修正行政區差異 | 新解析與R3行政區差異 | supplied/地址衝突 | supplied為縣市 | 重複縣市前綴 |',
    '|---|---:|---:|---:|---:|---:|']
    for r in result['splits']:
        f=r['flags']; lines.append(f"| {r['split']} | {r['legacy_to_round3_address_changes']['address_district']:,} | {r['new_district_differs_from_round3_rows']:,} | {f['district_prefix_conflict']:,} | {f['supplied_district_is_city']:,} | {f['repeated_city_prefix']:,} |")
    lines += ['', '新解析僅輸出地理研究檔，未改既有 house_features.py、round3_features.py、87 欄 cache 或任何第五輪檔案。這些變更未作用到已訓練房價模型。', '',
    '## 道路、交叉口、地段與門牌', '',
    '- 道路基名、段、巷、弄分欄保存。中山路一段與中山路1段同鍵；中山一路保持中山一路，不能轉成中山1路或中山路1段。不同段不直接合併。',
    '- 保留全部不同道路／巷弄候選，不挑第一路冒充唯一地址。恰好兩個不同道路名時另外輸出無序道路對；三條以上保留完整清單並標記，不任取前兩條。交叉口文字明示與僅由與／及推測分開旗標；是否真的相交仍須幾何核對。',
    '- 地號、小段、地址開頭的地籍段與道路的數字段分開。地號不能當門牌；以中山路段等帶「路」地籍名開頭時不誤認中山路。地址中同時含道路與地籍說明，只能視為帶地籍的道路參考。',
    '- 無門牌、附近、對面、尚未編釘與暫編只支持較粗定位。單門牌候選也只是語法完整，須有 OSM 門牌資料佐證；不能將道路代表點當成精確住宅座標。多門牌／號碼範圍與行政區衝突另列歧義。', '',
    '## 後續離線 OSM 匹配規則', '',
    '1. 先依已保存的城市與完整行政區限縮候選，匹配 OSM name／name:zh／name:zh-Hant／alt_name 等實際存在的名稱；alias 需記錄來源、使用欄位與正規化後碰撞，不能用房價選 alias。',
    '2. 優先以完整路名＋段比對；帶巷／弄的查詢仍保留細節。無段 base key 只供明示降級候選，遇到多段或同名不連通道路，回傳 ambiguous，不能挑第一個或只取全路中點。',
    '3. 多道路要先各自成功命名，再做真實相交／足夠近的幾何檢查；交點不唯一或道路不相交時留 ambiguity。缺門牌的單道路只回傳道路範圍與粗精度，不宣稱門牌匹配。',
    '4. 地籍段無道路證據時保留 unknown/cadastral_only；不將地段文字硬套成 OSM 道路，也不使用外部成交價格補答案。',
    '5. 建議匹配結果另存 osm_candidate_count、matched_name_field、alias_used、admin_fallback、section_missing_in_osm、disconnected_components、coordinate_precision、ambiguous、unknown_reason 與地理快照日期。候選不唯一與不存在須分開，不能填同一個看似有效的零座標。', '',
    '## 前置檢查與實際修正紀錄', '',
    '- 首版合成檢查攔到三道路後附地籍段被當成開頭地籍段的錯誤；修正地段判斷後，全部道路仍保留。',
    '- 首次真實稽核被 ID dtype 一致性斷言攔下：pandas 對 ndarray 自動推斷 string dtype。改為原 Series 原樣傳遞後，ID 值與型別核對通過；失敗版本未交付 parsed 成果。',
    '- 自行 review 發現連接詞前綴規則會把和平路裁成平路；現保留和／近等路名字元，第一道路也不剝除「及第路」的及。已加入和平路、近江路、及第路合成斷言。',
    '- 地籍段與小段改用非貪婪分拆，增加仁愛段一小段案例；此類資料仍不當道路或門牌定位。',
    '- 獨立唯讀 review 找到多段省略（中山路一段與二段／一、二段）與非數字段（中華路北段／南段）會丟失段別的問題。最後加入保守 uncertain gate，這些情況不再輸出精確道路鍵；未猜測替代路名。及人路首字亦以合成例核對保留。',
    '- 在 outputs/geography/address_audit 另存僅四欄的地理輸入快取，核對原 workbook SHA、欄位白名單與快取 SHA 後可重用。沒有覆写原 data_cache；最後一次稽核使用這些純 X 快取重跑解析及對齊。這是資料工程檢查與修正，未使用模型誤差挑選規則。', '',
    '## API、查詢鍵與重跑', '',
    '```python', 'from geo_address_audit import normalize_text, normalize_admin_name, normalize_road_name',
    'from geo_address_audit import parse_geo_queries, build_osm_query_keys',
    "normalize_road_name('中山路一段')  # 中山路1段",
    'parsed = parse_geo_queries(raw_geo)  # 僅接受地理X欄，回傳新DataFrame', '```', '',
    '道路鍵：city|district_full|road_name，例如「臺北市|大安區|忠孝東路4段」。road_candidates_json 保留 road_base、section_raw、section_number、lane/alley 及候選門牌；road_query_keys_json 保存全部行政區道路鍵，road_base_fallback_keys_json 明確標示缺段降級鍵。intersection_query_key 是同一行政區中排序後的兩個路名，不是已驗證交點。', '',
    '```text', 'python src\\geo_address_audit.py', 'python src\\geo_address_audit.py --self-check-only', '```', '',
    '純函式 import 只需 pandas/numpy，沒有 openpyxl、CatBoost 或 model_common 載入副作用；CLI 讀原 workbook 才需要 openpyxl。本次執行使用 bundled Python。', '',
    '- [完整解析稽核 JSON](../outputs/geography/address_audit/audit.json)',
    '- [道路查詢鍵分布](../outputs/geography/address_audit/road_query_counts.csv)',
    '- [純 X 解析與稽核程式](../src/geo_address_audit.py)']
    for r in result['splits']:
        lines.append(f"- [{r['split']} 原列對齊解析檔](../{r['parsed_path'].replace(chr(92), '/')})")
    lines += ['', f"程式 SHA256：`{result['source_sha256']}`。輸入活頁簿、原特徵 cache 與舊解析程式於稽核前後雜湊一致；完整來源與輸出 SHA 见 JSON。", '']
    path.write_text('\n'.join(lines).replace('优先','優先').replace('见 JSON','見 JSON').replace('覆写','覆寫'), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    checked = self_check()
    if args.self_check_only:
        print(json.dumps(checked, ensure_ascii=True)); return
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must stay within workspace')
    output.mkdir(parents=True, exist_ok=True)
    folder = ROOT/'台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression'
    sources = {'train': folder/'train-v2.xlsx', 'valid': folder/'valid-v2.xlsx', 'test': folder/'test-reindex-test-v2.2.xlsx'}
    protected = [ROOT/'src/house_features.py', ROOT/'src/round3_features.py'] + list(sources.values())
    protected += [ROOT/f'data_cache/{s}_features.pkl' for s in sources]
    before = {str(p.relative_to(ROOT)): file_sha(p) for p in protected}
    results, queries = [], []
    for split, path in sources.items():
        r, q = audit_split(split, path, output); results.append(r); queries.append(q)
    counts = pd.concat(queries, ignore_index=True)
    counts.to_csv(output/'road_query_counts.csv', index=False, encoding='utf-8-sig')
    assert before == {str(p.relative_to(ROOT)): file_sha(p) for p in protected}
    result = {'purpose': 'pure-X address parsing; candidate query coverage is not OSM match accuracy',
              'rules': RULES, 'self_check': checked, 'source_sha256': file_sha(Path(__file__)),
              'splits': results, 'preflight_history': ['three-road-plus-cadastral synthetic assertion failed then fixed', 'real source ID dtype assertion failed before parsed output then fixed', 'self-review fixed 和平路/近江路 prefix stripping and preserved first-road 及第路', 'cadastral subsection greedy split fixed with synthetic assertion', 'independent review: elided multi-section and nonnumeric section examples blocked as uncertain rather than downgraded silently', 'four-column geography cache hash-checked and reused for final audit'], 'protected_source_sha256': before, 'protected_sources_unchanged': True,
              'road_query_counts_sha256': file_sha(output/'road_query_counts.csv'),
              'constraints': {'price_column_loaded': False, 'model_trained': False, 'model_scored': False,
                              'external_price_data_used': False, 'osm_downloaded_or_matched': False,
                              'old_source_cache_or_round5_files_modified': False}}
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_report(result, ROOT/'reports/geography_address_audit.md')
    print(json.dumps({'completed': True, 'output': str(output), 'report': 'reports/geography_address_audit.md'}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()