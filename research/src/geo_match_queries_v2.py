"""Exact single-road joins plus explicitly stated, unambiguous planar intersections.

Geographic X only. No target, model, nearest-road guess, or intersection buffer.
All rejected rows keep missing coordinates; row/ID/cache provenance is preserved.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import shape, mapping, LineString
from shapely.ops import transform
from shapely.errors import GEOSException
from pyproj import Transformer
from geo_match_queries import sha

ROOT = Path(__file__).resolve().parents[1]
PARSER_SHA = 'e0672ff80e0a9558f4a09051a5f02edfa120d841cf8d91c8d89467e80a985bf9'
PROJECT = Transformer.from_crs(4326, 3826, always_xy=True).transform
POINT_GROUP_METRES = 50.0
ROAD_COLUMNS = ['latitude', 'longitude', 'quality', 'road_length_m', 'extent_diagonal_m',
                'connected_components_with_50m_gap_tolerance', 'components_with_1mm_gap_tolerance',
                'nearby_fragment_grouping', 'source_way_count']
LEGACY_REQUIRED = ['source_id', 'source_row_index', 'source_excel_row', 'is_embedded_header',
                   'city', 'district_full', 'road1_name', 'road_count', 'road_query_eligible', 'land_parcel_context']
GATE_COLUMNS = ['intersection_explicit', 'road_parse_uncertain', 'multiple_house_numbers',
                'house_number_count', 'house_number_range', 'road_candidates_json']
POLICY = {
    'single_road': 'Unchanged exact city|district|road join and quality policy from geo_match_queries.py',
    'intersection_candidates': 'Exactly two different complete road names, same supplied city/district, explicit intersection flag',
    'exclude': ['embedded headers', 'land context', 'parse/candidate context uncertainty',
                'multiple house numbers or number range', 'any lane or alley on either road'],
    'geometry_intersection': 'Exact intersection of original EPSG:4326 line geometries; no buffer, distance tolerance, or nearest-road selection',
    'point_group_distance_metres': POINT_GROUP_METRES,
    'accept_point_support': 'One 50m graph group AND every pair of intersection points at most 50m apart',
    'overlap': 'Any nonempty linear overlap withholds coordinates, even if points also exist',
    'representative': 'Actual intersection point closest to the centroid of intersection points; deterministic lexicographic tie order',
    'quality': 'planar_intersection; grade-separated connectivity and turn accessibility are unverified',
    'extent_diagonal_m': 'Bounding-box diagonal of accepted intersection-point support; single point 0 is not house-location error 0',
    'road_length_m': 'Sum of the two district-clipped source road lengths',
    'nearby_fragment_grouping': 'Logical OR of the two source road flags',
    'source_road_quality': 'Disconnected/broad source roads may yield a unique qualifying intersection',
}


def _check_parsed(parsed: pd.DataFrame) -> None:
    required = set(LEGACY_REQUIRED + GATE_COLUMNS + ['existing_feature_cache_row_index'])
    missing = required - set(parsed.columns)
    if missing:
        raise ValueError('Missing required parsed columns: ' + ', '.join(sorted(missing)))
    if not parsed.index.is_unique or not parsed.source_row_index.is_unique:
        raise ValueError('Parsed index and original row positions must be unique')
    if '總價元' in parsed or 'y' in parsed:
        raise ValueError('Price columns forbidden')
    flags = ['is_embedded_header', 'road_query_eligible', 'land_parcel_context',
             'intersection_explicit', 'road_parse_uncertain', 'multiple_house_numbers', 'house_number_range']
    if parsed[flags + ['road_count', 'house_number_count']].isna().any().any():
        raise ValueError('Missing eligibility flag or count')


def intersection_gate(row: dict) -> tuple[str, str, str]:
    """Return a fixed rejection reason or two exact keys. Never infer omitted names."""
    if bool(row['is_embedded_header']): return 'embedded_header', '', ''
    if bool(row['land_parcel_context']): return 'land_parcel', '', ''
    if int(row['road_count']) != 2: return 'intersection_requires_exactly_two_roads', '', ''
    if bool(row['road_parse_uncertain']): return 'intersection_parse_uncertain', '', ''
    if not bool(row['road_query_eligible']) or not row['city'] or not row['district_full']:
        return 'intersection_admin_or_parse_ineligible', '', ''
    if not bool(row['intersection_explicit']): return 'intersection_not_explicit', '', ''
    if bool(row['multiple_house_numbers']) or int(row['house_number_count']) > 1 or bool(row['house_number_range']):
        return 'intersection_multiple_house_objects', '', ''
    candidates = json.loads(row['road_candidates_json'])
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError('road_count=2 contradicts serialized candidate count')
    if any(bool(c.get('context_uncertain', False)) or bool(c.get('section_unsupported', False)) for c in candidates):
        return 'intersection_candidate_context_uncertain', '', ''
    if any(c.get('lane_number') or c.get('alley_number') for c in candidates):
        return 'intersection_lane_or_alley_unsupported', '', ''
    names = [c.get('road_name', '') for c in candidates]
    if any(not isinstance(n, str) or not n or n == '__MISSING__' or '|' in n for n in names) or len(set(names)) != 2:
        return 'intersection_requires_two_distinct_names', '', ''
    names.sort()
    prefix = str(row['city']) + '|' + str(row['district_full']) + '|'
    return 'eligible', prefix + names[0], prefix + names[1]


def classify_intersections(parsed: pd.DataFrame) -> pd.DataFrame:
    _check_parsed(parsed)
    result = pd.DataFrame({'reason': 'not_two_roads', 'key_1': '', 'key_2': ''}, index=parsed.index)
    subset = parsed.loc[parsed.road_count.eq(2), list(dict.fromkeys(LEGACY_REQUIRED + GATE_COLUMNS))]
    records = [intersection_gate(row) for row in subset.to_dict(orient='records')]
    if records:
        result.loc[subset.index, ['reason', 'key_1', 'key_2']] = records
    return result


def _point_groups(points_xy: np.ndarray) -> tuple[int, float, np.ndarray]:
    count = len(points_xy)
    if not count: return 0, 0.0, np.zeros((0, 0))
    distances = np.sqrt(np.square(points_xy[:, None, :] - points_xy[None, :, :]).sum(axis=2))
    pending = set(range(count)); groups = 0
    while pending:
        groups += 1
        todo = [min(pending)]; pending.remove(todo[0])
        while todo:
            current = todo.pop()
            adjacent = {i for i in pending if distances[current, i] <= POINT_GROUP_METRES}
            pending.difference_update(adjacent); todo.extend(sorted(adjacent))
    return groups, float(distances.max()), distances


def planar_intersection_summary(first: dict, second: dict) -> dict:
    """Pure geometry result. Coordinates are actual planar intersections, never door locations."""
    result = {'status': 'intersection_geometry_missing', 'latitude': None, 'longitude': None,
              'quality': 'intersection_geometry_missing', 'road_length_m': None, 'extent_diagonal_m': None,
              'nearby_fragment_grouping': bool(first.get('nearby_fragment_grouping', False) or second.get('nearby_fragment_grouping', False)),
              'source_way_count': None, 'point_count': 0, 'point_group_count': 0,
              'point_max_pair_distance_m': None, 'overlap_length_m': 0.0,
              'source_quality_1': first.get('quality'), 'source_quality_2': second.get('quality'),
              'grade_connectivity_verified': False, 'coordinate_interpretation': 'No usable planar intersection'}
    def reject(status):
        result['status'] = result['quality'] = status
        return result
    if (first.get('city'), first.get('district')) != (second.get('city'), second.get('district')):
        return reject('intersection_admin_mismatch')
    if first.get('road_name') == second.get('road_name'):
        return reject('intersection_same_road_name')
    if not first.get('geometry') or not second.get('geometry'):
        return result
    try:
        a, b = shape(first['geometry']), shape(second['geometry'])
        if a.is_empty or b.is_empty: return result
        if a.geom_type not in ('LineString', 'MultiLineString') or b.geom_type not in ('LineString', 'MultiLineString'):
            return reject('intersection_non_linear_source_geometry')
        if not a.is_valid or not b.is_valid: return reject('intersection_invalid_source_geometry')
        # EPSG:4326 exact planar topology first. Metric projection is used only for support statistics.
        overlap = a.intersection(b)
    except (ValueError, GEOSException):
        return reject('intersection_geometry_error')
    result['road_length_m'] = float(first.get('road_length_m', transform(PROJECT, a).length)) + float(second.get('road_length_m', transform(PROJECT, b).length))
    if 'source_way_ids' in first and 'source_way_ids' in second:
        result['source_way_count'] = len(set(first['source_way_ids']) | set(second['source_way_ids']))
    points, linear_parts, unexpected = [], [], []
    def visit(geometry):
        if geometry.is_empty: return
        if geometry.geom_type == 'Point': points.append((float(geometry.x), float(geometry.y)))
        elif geometry.geom_type in ('LineString', 'LinearRing'): linear_parts.append(geometry)
        elif hasattr(geometry, 'geoms'):
            for child in geometry.geoms: visit(child)
        else: unexpected.append(geometry.geom_type)
    visit(overlap)
    points = sorted(set(points))
    result['point_count'] = len(points)
    projected = np.asarray([PROJECT(x, y) for x, y in points], dtype='float64').reshape((-1, 2))
    groups, max_distance, _ = _point_groups(projected)
    result['point_group_count'] = groups
    result['point_max_pair_distance_m'] = max_distance if points else None
    if linear_parts:
        result['overlap_length_m'] = float(sum(transform(PROJECT, g).length for g in linear_parts))
        return reject('intersection_linear_overlap')
    if unexpected: return reject('intersection_non_point_result')
    if not points: return reject('intersection_no_geometric_intersection')
    if groups != 1 or max_distance > POINT_GROUP_METRES:
        return reject('intersection_multiple_distant_points')
    center = projected.mean(axis=0)
    choice = int(np.square(projected - center).sum(axis=1).argmin())
    longitude, latitude = points[choice]
    result.update({'status': 'planar_intersection', 'quality': 'planar_intersection',
                   'latitude': latitude, 'longitude': longitude,
                   'extent_diagonal_m': float(np.linalg.norm(projected.max(axis=0) - projected.min(axis=0))),
                   'coordinate_interpretation': 'Actual planar intersection point representing the unique accepted point group; not a house location; grade connectivity unverified'})
    return result


def load_index(index_path: Path, needed_geometry_keys: set[str]) -> tuple[pd.DataFrame, dict]:
    metadata, geometry = [], {}
    with (index_path / 'road_index.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['key'] != '|'.join([row['city'], row['district'], row['road_name']]):
                raise ValueError('Inconsistent road index key')
            metadata.append({key: row[key] for key in ['key', *ROAD_COLUMNS]})
            if row['key'] in needed_geometry_keys:
                geometry[row['key']] = row
    roads = pd.DataFrame(metadata)
    if not roads.key.is_unique: raise ValueError('Road index keys must be unique')
    return roads.set_index('key'), geometry


def match_parsed(parsed: pd.DataFrame, lookup: pd.DataFrame, geometry: dict, pair_cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    _check_parsed(parsed)
    pair_cache = {} if pair_cache is None else pair_cache
    gate = classify_intersections(parsed)
    keep = list(dict.fromkeys(LEGACY_REQUIRED + ['existing_feature_cache_row_index'] + [x for x in GATE_COLUMNS if x != 'road_candidates_json']))
    frame = parsed[keep].copy()
    key = frame[['city', 'district_full', 'road1_name']].fillna('').astype(str).agg('|'.join, axis=1)
    eligible = (frame.road_count.eq(1) & frame.road_query_eligible.fillna(False).astype(bool)
                & ~frame.land_parcel_context.fillna(False).astype(bool) & ~frame.is_embedded_header.fillna(False).astype(bool))
    # This single-road block preserves the original matcher join, masking and status precedence.
    frame['geo_key'] = key
    for column in ROAD_COLUMNS: frame['geo_' + column] = key.map(lookup[column]).where(eligible)
    frame['geo_match_status'] = 'ineligible_address'
    frame.loc[frame.is_embedded_header.astype(bool), 'geo_match_status'] = 'embedded_header'
    frame.loc[~frame.is_embedded_header.astype(bool) & frame.land_parcel_context.astype(bool), 'geo_match_status'] = 'land_parcel'
    frame.loc[~frame.is_embedded_header.astype(bool) & ~frame.land_parcel_context.astype(bool) & frame.road_count.gt(1), 'geo_match_status'] = 'multiple_roads'
    frame.loc[eligible, 'geo_match_status'] = 'road_key_not_found'
    found = eligible & frame.geo_quality.notna()
    frame.loc[found, 'geo_match_status'] = frame.loc[found, 'geo_quality'].astype(str)
    single_result = frame.loc[eligible, ['geo_' + c for c in ROAD_COLUMNS] + ['geo_match_status']].copy(deep=True)
    frame['geo_query_method'] = np.where(eligible, 'single_road', 'none')
    frame['geo_intersection_gate_reason'] = gate.reason
    frame['geo_intersection_key_1'] = gate.key_1
    frame['geo_intersection_key_2'] = gate.key_2
    frame['geo_intersection_point_count'] = np.nan
    frame['geo_intersection_point_group_count'] = np.nan
    frame['geo_intersection_point_max_pair_distance_m'] = np.nan
    frame['geo_intersection_overlap_length_m'] = np.nan
    frame['geo_intersection_source_quality_1'] = pd.Series(None, index=frame.index, dtype=object)
    frame['geo_intersection_source_quality_2'] = pd.Series(None, index=frame.index, dtype=object)
    frame['geo_grade_connectivity_verified'] = False
    frame['geo_coordinate_interpretation'] = 'No usable coordinate'
    frame.loc[frame.geo_match_status.eq('road_level'), 'geo_coordinate_interpretation'] = 'Original road-level representative; not a house location'
    two = frame.road_count.eq(2)
    frame.loc[two, 'geo_query_method'] = 'intersection_rejected_before_geometry'
    rejected_gate = two & ~gate.reason.eq('eligible')
    frame.loc[rejected_gate, 'geo_match_status'] = gate.loc[rejected_gate, 'reason']
    frame.loc[rejected_gate, 'geo_quality'] = gate.loc[rejected_gate, 'reason']
    attempted = gate.reason.eq('eligible')
    frame.loc[attempted, 'geo_query_method'] = 'planar_intersection'
    for index, row in gate.loc[attempted].iterrows():
        pair = (row.key_1, row.key_2)
        if pair not in pair_cache:
            if pair[0] not in geometry or pair[1] not in geometry:
                missing = [k for k in pair if k not in geometry]
                pair_cache[pair] = {'status': 'intersection_road_key_not_found', 'missing_keys': missing}
            else:
                pair_cache[pair] = planar_intersection_summary(geometry[pair[0]], geometry[pair[1]])
        result = pair_cache[pair]
        frame.at[index, 'geo_match_status'] = result['status']
        frame.at[index, 'geo_quality'] = result.get('quality', result['status'])
        frame.at[index, 'geo_key'] = pair[0] + ' & ' + pair[1]
        for name in ['latitude', 'longitude', 'road_length_m', 'extent_diagonal_m', 'nearby_fragment_grouping', 'source_way_count']:
            frame.at[index, 'geo_' + name] = result.get(name, np.nan)
        for name in ['point_count', 'point_group_count', 'point_max_pair_distance_m', 'overlap_length_m', 'source_quality_1', 'source_quality_2']:
            frame.at[index, 'geo_intersection_' + name] = result.get(name, np.nan)
        frame.at[index, 'geo_coordinate_interpretation'] = result.get('coordinate_interpretation', 'No usable coordinate')
    pd.testing.assert_frame_equal(single_result, frame.loc[eligible, single_result.columns])
    usable_single = frame.geo_match_status.eq('road_level')
    usable_pair = frame.geo_match_status.eq('planar_intersection')
    usable = usable_single | usable_pair
    frame['geo_result_group'] = np.select([usable_single, usable_pair], ['single_road', 'dual_road_planar_intersection'], default='rejected')
    assert frame.loc[~usable, ['geo_latitude', 'geo_longitude']].isna().all().all()
    assert np.isfinite(frame.loc[usable, ['geo_latitude', 'geo_longitude']].to_numpy(dtype=float)).all()
    assert frame.loc[usable, 'geo_latitude'].between(20, 28).all() and frame.loc[usable, 'geo_longitude'].between(116, 125).all()
    pd.testing.assert_frame_equal(frame[['source_id', 'source_row_index', 'source_excel_row', 'existing_feature_cache_row_index']], parsed[['source_id', 'source_row_index', 'source_excel_row', 'existing_feature_cache_row_index']])
    normal = ~frame.is_embedded_header.astype(bool)
    def counts(series): return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}
    by_city = []
    for city, group in frame.loc[normal].groupby('city', dropna=False):
        n = len(group); active = group.geo_match_status.isin(['road_level', 'planar_intersection'])
        by_city.append({'city': None if pd.isna(city) else str(city), 'rows': n, 'usable_coordinates': int(active.sum()),
                        'fraction': float(active.mean()), 'result_group_counts': counts(group.geo_result_group), 'status_counts': counts(group.geo_match_status)})
    audit = {'rows_including_headers': len(frame), 'non_header_rows': int(normal.sum()),
             'single_road_eligible_rows': int(eligible.sum()), 'single_road_key_found_rows': int(found.sum()),
             'explicit_two_road_eligible_rows': int(attempted.sum()), 'unique_eligible_road_pairs': int(gate.loc[attempted, ['key_1', 'key_2']].drop_duplicates().shape[0]),
             'usable_single_road_coordinates': int(usable_single.sum()), 'usable_planar_intersection_coordinates': int(usable_pair.sum()),
             'usable_coordinates': int(usable.sum()), 'usable_fraction_of_non_header_rows': float(usable.sum() / normal.sum()),
             'result_group_counts': counts(frame.geo_result_group), 'status_counts': counts(frame.geo_match_status),
             'two_road_gate_reason_counts': counts(gate.loc[two, 'reason']), 'by_city': by_city,
             'single_road_results_unchanged': True, 'row_id_order_and_cache_index_unchanged': True,
             'all_rejections_have_missing_coordinates': True}
    return frame, audit


def self_check() -> dict:
    def record(name, coordinates, city='甲市', district='甲區', quality='road_level'):
        g = LineString(coordinates)
        return {'city': city, 'district': district, 'road_name': name, 'key': city + '|' + district + '|' + name,
                'geometry': mapping(g), 'quality': quality, 'latitude': 24.0 if quality == 'road_level' else None,
                'longitude': 121.0 if quality == 'road_level' else None, 'road_length_m': transform(PROJECT, g).length,
                'extent_diagonal_m': 10.0, 'connected_components_with_50m_gap_tolerance': 1,
                'components_with_1mm_gap_tolerance': 1, 'nearby_fragment_grouping': False,
                'source_way_ids': [name], 'source_way_count': 1}
    a = record('甲路', [(121, 24), (121.003, 24)])
    b = record('乙路', [(121.001, 23.999), (121.001, 24.001)])
    one = planar_intersection_summary(a, b)
    assert one['status'] == 'planar_intersection' and one['point_count'] == one['point_group_count'] == 1
    assert one['extent_diagonal_m'] == 0 and not one['grade_connectivity_verified']
    different_city = dict(b, city='乙市')
    assert planar_intersection_summary(a, different_city)['status'] == 'intersection_admin_mismatch'
    far = record('遠交點路', [(121.0002, 23.999), (121.0002, 24.001), (121.0012, 24.001), (121.0012, 23.999)])
    distant = planar_intersection_summary(a, far)
    assert distant['status'] == 'intersection_multiple_distant_points' and distant['point_group_count'] == 2
    near = record('近交點路', [(121.0002, 23.999), (121.0002, 24.001), (121.0004, 24.001), (121.0004, 23.999)])
    close = planar_intersection_summary(a, near)
    assert close['status'] == 'planar_intersection' and close['point_count'] == 2 and 0 < close['extent_diagonal_m'] < 50
    chain = record('鏈狀交點路', [(121.0002, 23.999), (121.0002, 24.001), (121.0006, 24.001), (121.0006, 23.999), (121.0010, 23.999), (121.0010, 24.001)])
    chained = planar_intersection_summary(a, chain)
    assert chained['point_group_count'] == 1 and chained['point_max_pair_distance_m'] > 50 and chained['status'] == 'intersection_multiple_distant_points'
    overlap = record('重疊路', [(121.0005, 24), (121.0015, 24)])
    assert planar_intersection_summary(a, overlap)['status'] == 'intersection_linear_overlap'
    absent = record('不交路', [(121, 24.001), (121.003, 24.001)])
    assert planar_intersection_summary(a, absent)['status'] == 'intersection_no_geometric_intersection'
    almost = record('近而未交路', [(121.001, 23.999), (121.001, 23.999999)])
    assert planar_intersection_summary(a, almost)['status'] == 'intersection_no_geometric_intersection'
    broad = dict(a, quality='broad_road', latitude=None, longitude=None, nearby_fragment_grouping=True)
    disjoint_summary = planar_intersection_summary(broad, dict(b, quality='disconnected', latitude=None, longitude=None))
    assert disjoint_summary['status'] == 'planar_intersection' and disjoint_summary['nearby_fragment_grouping']
    roads = [{'road_name': '甲路', 'lane_number': '', 'alley_number': '', 'context_uncertain': False},
             {'road_name': '乙路', 'lane_number': '', 'alley_number': '', 'context_uncertain': False}]
    base = {'source_id': 'x', 'source_row_index': 0, 'source_excel_row': 2, 'existing_feature_cache_row_index': 0,
            'city': '甲市', 'district_full': '甲區', 'road1_name': '甲路', 'road_count': 2,
            'road_query_eligible': True, 'land_parcel_context': False, 'is_embedded_header': False,
            'road_parse_uncertain': False, 'intersection_explicit': True, 'multiple_house_numbers': False,
            'house_number_count': 0, 'house_number_range': False, 'road_candidates_json': json.dumps(roads)}
    assert intersection_gate(base)[0] == 'eligible'
    assert intersection_gate(dict(base, land_parcel_context=True))[0] == 'land_parcel'
    assert intersection_gate(dict(base, multiple_house_numbers=True))[0] == 'intersection_multiple_house_objects'
    assert intersection_gate(dict(base, house_number_range=True))[0] == 'intersection_multiple_house_objects'
    assert intersection_gate(dict(base, intersection_explicit=False))[0] == 'intersection_not_explicit'
    lane = [dict(roads[0], lane_number='63'), roads[1]]
    assert intersection_gate(dict(base, road_candidates_json=json.dumps(lane)))[0] == 'intersection_lane_or_alley_unsupported'
    alley = [roads[0], dict(roads[1], alley_number='2')]
    assert intersection_gate(dict(base, road_candidates_json=json.dumps(alley)))[0] == 'intersection_lane_or_alley_unsupported'
    ambiguous = [dict(roads[0], context_uncertain=True), roads[1]]
    assert intersection_gate(dict(base, road_candidates_json=json.dumps(ambiguous)))[0] == 'intersection_candidate_context_uncertain'
    same = [roads[0], dict(roads[1], road_name='甲路')]
    assert intersection_gate(dict(base, road_candidates_json=json.dumps(same)))[0] == 'intersection_requires_two_distinct_names'
    cases = [base, dict(base, road_count=1), dict(base, land_parcel_context=True),
             dict(base, road_candidates_json=json.dumps(lane)), dict(base, multiple_house_numbers=True),
             dict(base, intersection_explicit=False), dict(base, is_embedded_header=True)]
    for i, row in enumerate(cases): row.update(source_row_index=i, source_excel_row=i + 2, existing_feature_cache_row_index=i)
    frame = pd.DataFrame(cases, index=np.arange(len(cases)) + 11)
    lookup = pd.DataFrame([{c: r[c] for c in ['key', *ROAD_COLUMNS]} for r in (a, b)]).set_index('key')
    result, audit = match_parsed(frame, lookup, {a['key']: a, b['key']: b})
    assert audit['usable_planar_intersection_coordinates'] == 1 and audit['usable_single_road_coordinates'] == 1
    assert result.iloc[0].geo_match_status == 'planar_intersection' and result.iloc[1].geo_match_status == 'road_level'
    assert result.iloc[1].geo_latitude == a['latitude'] and result.iloc[1].geo_road_length_m == a['road_length_m']
    assert result.iloc[2:][['geo_latitude', 'geo_longitude']].isna().all().all()
    return {'status': 'passed', 'geometry_cases': 9, 'eligibility_cases': 9,
            'end_to_end_rows': len(frame), 'single_road_unchanged': True, 'provenance_unchanged': True,
            'no_tolerance_for_nonintersecting_roads': True, 'no_price_or_model_used': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--parsed', nargs='+', help='split=path pairs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    checked = self_check()
    if args.self_check_only:
        print(json.dumps(checked, ensure_ascii=True)); return
    if args.index is None or args.parsed is None or args.output is None:
        parser.error('--index, --parsed and --output are required unless --self-check-only')
    index = args.index.resolve(); out = args.output.resolve()
    if out.exists(): raise FileExistsError('Refuse overwrite: ' + str(out))
    if not out.is_relative_to(ROOT): raise ValueError('Output must be inside workspace')
    files = {}
    for spec in args.parsed:
        split, name = spec.split('=', 1)
        if split not in ('train', 'valid', 'test') or split in files: raise ValueError('Invalid or duplicate split')
        path = Path(name).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        files[split] = path
    sources = [Path(__file__), ROOT/'src/geo_match_queries.py', ROOT/'src/geo_address_audit.py']
    if sha(sources[2]) != PARSER_SHA: raise ValueError('Frozen parser source changed')
    required_files = [index/'road_index.jsonl', index/'audit.json', *files.values()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources + required_files}
    wanted = set()
    for path in files.values():
        parsed = pd.read_pickle(path)
        gate = classify_intersections(parsed); eligible = gate.reason.eq('eligible')
        wanted.update(gate.loc[eligible, 'key_1']); wanted.update(gate.loc[eligible, 'key_2'])
        del parsed, gate
    lookup, geometry = load_index(index, wanted)
    out.mkdir(parents=True)
    audit = {'status': 'completed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'purpose': 'Pure X exact-road and planar-intersection coverage; no price/model evaluation',
             'policy': POLICY, 'self_check': checked, 'source_and_input_sha256': before,
             'index_directory': index.relative_to(ROOT).as_posix(), 'index_keys': len(lookup),
             'requested_intersection_road_geometry_keys': len(wanted), 'found_intersection_road_geometry_keys': len(geometry),
             'target_values_read': False, 'model_fitting_or_scoring_performed': False,
             'no_external_website_access': True, 'house_coordinates_claimed': False,
             'grade_connectivity_verified': False, 'splits': {}}
    pair_cache = {}
    for split, path in files.items():
        parsed = pd.read_pickle(path)
        matched, summary = match_parsed(parsed, lookup, geometry, pair_cache)
        filename = out / f'{split}_matched.pkl'; matched.to_pickle(filename)
        pd.testing.assert_frame_equal(pd.read_pickle(filename), matched)
        summary.update(output_path=filename.relative_to(ROOT).as_posix(), output_sha256=sha(filename))
        audit['splits'][split] = summary
        print(json.dumps({'split': split, 'rows': len(matched), 'result_groups': summary['result_group_counts'],
                          'eligible_intersection_rows': summary['explicit_two_road_eligible_rows']}, ensure_ascii=True), flush=True)
    audit['unique_computed_intersection_pairs'] = len(pair_cache)
    for relative, expected in before.items():
        if sha(ROOT/relative) != expected: raise AssertionError('Input/source changed during matching: ' + relative)
    (out/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'completed': True, 'output': out.relative_to(ROOT).as_posix()}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()