"""Synthetic geographic checks only; no housing data, labels, or models.

Runs the real road-index builder on tiny temporary GeoJSON fixtures inside the
workspace. The persistent outputs are this script and an explicit JSON report.
Use --output with a new filename for another source revision; never overwrite.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/geography/index_self_check.json')
    args = parser.parse_args()
    output = args.output.resolve()
    base = (ROOT/'outputs/geography').resolve()
    assert base.is_relative_to(ROOT.resolve()), 'Temporary area must stay inside workspace'
    assert output.is_relative_to(base), 'Report must stay inside outputs/geography'
    assert not output.exists(), 'Use a new report filename; existing evidence is not overwritten'
    sources = [ROOT/'src/geo_road_index.py', ROOT/'src/geo_address_audit.py', Path(__file__)]
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    import geo_road_index as geo
    from shapely.geometry import LineString, Point, box, shape, mapping
    from shapely.ops import transform

    # Deliberately fictional district/road names; these are not real addresses.
    city_a, city_b = '\u81fa\u5317\u5e02', '\u65b0\u5317\u5e02'
    unknown_city = '\u672a\u660e\u5e02'
    district, north = '\u5408\u6210\u5340', '\u5317\u6e2c\u5340'
    unknown_district, crossing_district = '\u672a\u77e5\u5340', '\u8de8\u754c\u5340'
    shared, crossing = '\u5408\u6210\u8def', '\u7a7f\u754c\u8def'
    section_one, alias_one = '\u5206\u6bb5\u8def\u4e00\u6bb5', '\u5206\u6bb5\u8def1\u6bb5'
    section_two, alias_two = '\u5206\u6bb5\u8def\u4e8c\u6bb5', '\u5206\u6bb5\u8def2\u6bb5'
    bare_section, bent = '\u5206\u6bb5\u8def', '\u5f4e\u6298\u8def'
    assert city_a in geo.CITIES and city_b in geo.CITIES and unknown_city not in geo.CITIES
    area_a = box(121.500, 25.000, 121.510, 25.020)
    area_b = box(121.510, 25.000, 121.520, 25.020)
    lower_a = box(121.500, 25.000, 121.510, 25.010)
    upper_a = box(121.500, 25.010, 121.510, 25.020)
    area_unknown = box(121.530, 25.000, 121.540, 25.020)

    def boundary(identifier, name, geometry, level):
        return {'type': 'Feature', 'id': identifier, 'properties': {'name': name, 'admin_level': str(level)}, 'geometry': mapping(geometry)}

    def road(identifier, name, coordinates, alias=None):
        props = {'name': name, 'highway': 'residential'}
        if alias is not None:
            props['alt_name'] = alias
        return {'type': 'Feature', 'id': 'w'+str(identifier), 'osm_id': identifier,
                'properties': props, 'geometry': mapping(LineString(coordinates))}

    areas = [boundary('a1', city_a, area_a, 4), boundary('a2', city_b, area_b, 4),
             boundary('a3', district, lower_a, 8), boundary('a4', north, upper_a, 8),
             boundary('a5', district, area_b, 8), boundary('a6', unknown_city, area_unknown, 4),
             boundary('a7', unknown_district, area_unknown, 8),
             boundary('a8', crossing_district, box(121.509, 25.016, 121.511, 25.018), 8)]
    roads = [road(1001, shared, [(121.502,25.005),(121.518,25.005)]),
             road(1002, crossing, [(121.506,25.002),(121.506,25.018)]),
             road(1003, section_one, [(121.501,25.003),(121.504,25.003)], alias_one),
             road(1004, section_two, [(121.501,25.004),(121.504,25.004)], alias_two),
             road(1005, bent, [(121.501,25.001),(121.504,25.001),(121.504,25.004)]),
             road(1006, section_one, [(121.513,25.003),(121.517,25.003)], alias_one),
             road(1007, shared, [(121.531,25.005),(121.539,25.005)])]
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({'name': name, 'passed': True, 'detail': detail})
        except Exception as exc:
            checks.append({'name': name, 'passed': False, 'error': type(exc).__name__+': '+str(exc)})

    def on_projected_road(row):
        assert row['latitude'] is not None and row['longitude'] is not None
        p = transform(geo.PROJECT, Point(row['longitude'], row['latitude']))
        g = transform(geo.PROJECT, shape(row['geometry']))
        error = p.distance(g)
        assert error < 1e-5, f'Representative is {error} metres off the projected road'
        return error

    base.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.index-self-check-', dir=base)).resolve()
    try:
        source, destination = temp/'source', temp/'index'
        source.mkdir()
        for filename, features in [('roads.jsonl', roads), ('administrative_areas.jsonl', areas)]:
            with (source/filename).open('x', encoding='utf-8', newline='\n') as handle:
                for feature in features:
                    handle.write(json.dumps(feature, ensure_ascii=False)+'\n')
        (source/'audit.json').write_text(json.dumps({'synthetic_fixture': True, 'targets_read': False})+'\n', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            geo.build(source, destination)
        rows = [json.loads(line) for line in (destination/'road_index.jsonl').read_text(encoding='utf-8').splitlines()]
        index = {(row['city'],row['district'],row['road_name']): row for row in rows}
        assert len(index) == len(rows), 'Duplicate index key'
        audit = json.loads((destination/'audit.json').read_text(encoding='utf-8'))

        def normalized(name):
            return geo.normalize_road_name(name)

        def clipped_across_city():
            a, b = index[(city_a,district,normalized(shared))], index[(city_b,district,normalized(shared))]
            assert lower_a.buffer(1e-12).covers(shape(a['geometry']))
            assert area_b.buffer(1e-12).covers(shape(b['geometry']))
            assert a['bbox'][2] <= 121.510+1e-12 and b['bbox'][0] >= 121.510-1e-12
            assert a['source_way_ids'] == b['source_way_ids'] == [1001]
            assert a['longitude'] < 121.510 < b['longitude']
            for row in (a,b):
                on_projected_road(row)
            original_length = shape(roads[0]['geometry']).length
            total = shape(a['geometry']).length + shape(b['geometry']).length
            assert abs(total-original_length) < 1e-10
            return {'distinct_city_keys': 2, 'same_district_name_kept_city_scoped': True, 'source_geometry_partitioned': True}
        check('cross_city_road_is_clipped_and_city_scoped', clipped_across_city)

        def clipped_across_district():
            a = index[(city_a,district,normalized(crossing))]
            b = index[(city_a,north,normalized(crossing))]
            assert lower_a.buffer(1e-12).covers(shape(a['geometry']))
            assert upper_a.buffer(1e-12).covers(shape(b['geometry']))
            assert a['bbox'][3] <= 25.010+1e-12 and b['bbox'][1] >= 25.010-1e-12
            assert a['latitude'] < 25.010 < b['latitude']
            assert a['source_way_ids'] == b['source_way_ids'] == [1002]
            return {'district_fragments': 2, 'coordinates_inside_correct_district': True}
        check('cross_district_road_is_clipped', clipped_across_district)

        def sections_and_aliases():
            one = set(geo.names(roads[2]['properties'], road=True))
            two = set(geo.names(roads[3]['properties'], road=True))
            assert one and two and one.isdisjoint(two), 'Different sections collapsed'
            for name in one:
                assert index[(city_a,district,name)]['source_way_ids'] == [1003]
                assert index[(city_b,district,name)]['source_way_ids'] == [1006]
            for name in two:
                assert index[(city_a,district,name)]['source_way_ids'] == [1004]
                assert (city_b,district,name) not in index
            assert (city_a,district,normalized(bare_section)) not in index
            return {'section_aliases_disjoint': True, 'no_unqualified_section_fallback': True, 'same_alias_in_other_city_not_merged': True}
        check('section_aliases_do_not_merge_or_cross_city', sections_and_aliases)

        def point_on_bend():
            row = index[(city_a,district,normalized(bent))]
            geometry = transform(geo.PROJECT, shape(row['geometry']))
            assert geometry.centroid.distance(geometry) > 1, 'Fixture centroid must lie off road'
            error = on_projected_road(row)
            return {'centroid_off_road_metres': geometry.centroid.distance(geometry), 'representative_off_road_metres': error,
                    'is_road_representative_not_promised_arclength_midpoint': True}
        check('bent_road_representative_is_on_road', point_on_bend)

        def missing_city():
            assert not any(row['city'] == unknown_city or row['district'] == unknown_district for row in rows)
            assert index.get((unknown_city,district,normalized(shared))) is None
            assert index.get(('',district,normalized(shared))) is None
            assert all(1007 not in row['source_way_ids'] for row in rows)
            rejected = {row['source_id'] for row in audit['administration']['rejected_district_assignments']}
            assert 'a7' in rejected and 'a8' in rejected
            return {'unrecognized_city_has_no_index_coordinate': True, 'half_in_two_cities_district_rejected': True,
                    'scope': 'Exact index-key behavior; a future address matcher must also prohibit nationwide fallback'}
        check('unknown_city_and_ambiguous_city_assignment_fail_closed', missing_city)

        def metric_summary(gap):
            lines = [LineString([(300000,2770000),(300100,2770000)]),
                     LineString([(300100+gap,2770000),(300200+gap,2770000)])]
            items = [(transform(geo.UNPROJECT,line),2001+i,'residential') for i,line in enumerate(lines)]
            return geo.road_summary((city_a,district,'synthetic_gap'), items)

        def far_fragments():
            row = metric_summary(300)
            assert row['quality'] == 'disconnected'
            assert row['latitude'] is None and row['longitude'] is None
            assert row['connected_components_with_50m_gap_tolerance'] == 2
            assert row['components_with_1mm_gap_tolerance'] == 2
            assert row['nearby_fragment_grouping'] is False
            return {'gap_metres': 300, 'coordinates_withheld': True}
        check('distant_same_name_fragments_withhold_coordinates', far_fragments)

        def near_fragments():
            row = metric_summary(20)
            assert row['quality'] == 'road_level'
            assert row['connected_components_with_50m_gap_tolerance'] == 1
            assert row['components_with_1mm_gap_tolerance'] == 2
            assert row['nearby_fragment_grouping'] is True
            error = on_projected_road(row)
            return {'gap_metres': 20, 'representative_allowed_with_explicit_grouping_flag': True,
                    'physical_connectivity_not_proven': True, 'representative_off_road_metres': error}
        check('near_fragments_are_explicitly_spatial_not_topological_group', near_fragments)

        def connected_fragments():
            row = metric_summary(0)
            assert row['quality'] == 'road_level'
            assert row['connected_components_with_50m_gap_tolerance'] == 1
            assert row['components_with_1mm_gap_tolerance'] == 1
            assert row['nearby_fragment_grouping'] is False
            on_projected_road(row)
            return {'shared_endpoint': True, 'nearby_fragment_grouping': False}
        check('truly_joined_fragments_have_no_nearby_grouping_flag', connected_fragments)

        def audit_wording():
            rules = audit['fixed_geographic_rules']
            assert 'all_disconnected_matches_withhold_coordinates' not in rules
            assert rules['all_disconnected_after_50m_gap_tolerance_withhold_coordinates'] is True
            assert rules['nearby_fragments_can_share_a_road_representative'] is True
            assert rules['physical_road_connectivity_is_not_proven'] is True
            return {'50m_policy_is_named': True, 'does_not_claim_physical_connectivity': True}
        check('audit_discloses_gap_tolerance_policy', audit_wording)
        row_count = len(rows)
    finally:
        # Verify the absolute target stays within this task's temporary area
        # immediately before recursive cleanup. No input/output archive is moved.
        assert temp.parent == base and temp.name.startswith('.index-self-check-')
        assert temp.is_relative_to(base) and temp.is_relative_to(ROOT.resolve()) and not temp.is_symlink()
        shutil.rmtree(temp)

    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    checks.append({'name': 'sources_unchanged_during_check', 'passed': before == after})
    report = {'status': 'passed' if all(c['passed'] for c in checks) else 'failed',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'source_sha256': before, 'source_sha256_after': after,
              'versions': {name: version(name) for name in ('shapely','pyproj','numpy','pandas')},
              'python_executable': sys.executable, 'synthetic_boundaries': len(areas), 'synthetic_roads': len(roads),
              'synthetic_index_rows': row_count, 'labels_read': False, 'housing_data_read': False,
              'external_data_downloaded': False, 'models_trained': False,
              'geography_policy_note': 'Within 50m gaps may form one spatial group; 1mm grouping is diagnostic, not proof of real-world road connectivity',
              'scope_limitations': ['Synthetic geometry/administration/index checks, not measured real-data coverage.',
                                    'No address lookup API exists in geo_road_index; unknown-city behavior is checked at the exact-key index boundary.',
                                    'The representative is on projected EPSG:3826 road geometry; it is not a building coordinate or necessarily an arclength midpoint.']}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'passed_checks': sum(c['passed'] for c in checks), 'checks': len(checks),
                      'output': output.relative_to(ROOT).as_posix(), 'failed': [c for c in checks if not c['passed']]}, ensure_ascii=False))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
