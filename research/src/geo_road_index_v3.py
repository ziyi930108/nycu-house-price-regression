"""Annual parent-aware road index: verify existing districts, add missing coverage.

Same frozen clipping and road summary policies as v2; no housing inputs/targets.
Existing named-city geometry is retained. Parent subareas never form a city union.
"""
from __future__ import annotations
import argparse
import collections
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from geo_road_index import (ROOT, boundaries, digest, names, read_features,
                            road_summary, valid_polygon)
from geo_road_index_v2 import (CITY, PARENT_ID, POLICY, require, load_json,
                              write_json, snapshot, verify_snapshot,
                              load_parent_districts, merge_districts,
                              district_clips, self_check as v2_self_check)


def existing_city_geometry(areas_path):
    """Only explicit named city features, never an inferred union of districts."""
    parts = []
    for feature in read_features(areas_path):
        if CITY in names(feature['properties']):
            geometry = valid_polygon(shape(feature['geometry']))
            if not geometry.is_empty:
                parts.append(geometry)
    return unary_union(parts) if parts else None


def classify_parent_coverage(base_districts, parent_districts, complete_proof,
                             city_geometry=None):
    """Keep a complete ref proof, separating verified existing from added rows.

    Existing key must have the same single source relation ID. Its geometry must
    equal that same original district clipped by the same exact city geometry,
    or equal the original district directly. No existing geometry is overwritten.
    """
    by_key = {(d['city'], d['district']): d for d in base_districts}
    require(len(by_key) == len(base_districts), 'Duplicate base district key')
    available = set(complete_proof['available_subarea_ids'])
    missing = set(complete_proof['missing_subarea_ids'])
    observed = {d['administration_evidence']['subarea_id'] for d in parent_districts}
    require(len(observed) == len(parent_districts) and observed == available,
            'Complete available parent inventory required before coverage classification')
    require(available.isdisjoint(missing), 'Available/missing parent refs overlap')
    if 'parent_subarea_ids' in complete_proof:
        require(available | missing == set(complete_proof['parent_subarea_ids']),
                'Complete parent proof is not exhaustive')
    supplemental, covered = [], []
    for district in parent_districts:
        key = (district['city'], district['district'])
        ev = district['administration_evidence']
        sid = ev['subarea_id']
        require(ev['parent_id'] == PARENT_ID and district['city'] == CITY and
                ev['member_type'] == 'r' and ev['member_role'] == 'subarea',
                'Invalid parent lineage before classification')
        require(ev['source_pbf_sha256'] == complete_proof['source_pbf_sha256'] and
                ev['source_replication_timestamp'] == complete_proof['source_replication_timestamp'],
                'Mixed annual snapshots before classification')
        existing = by_key.get(key)
        if existing is None:
            supplemental.append(district)
            continue
        require(set(existing['sources']) == set(district['sources']) == {f'relation/{sid}'},
                'An existing district key has a different or ambiguous administrative ID')
        original = district['geometry']
        geometry = existing['geometry']
        if geometry.equals(original):
            rule = 'topological equality with same original parent district polygon'
            coverage = 1.0
        else:
            require(city_geometry is not None and not city_geometry.is_empty,
                    'Existing clipped district requires its exact named city geometry')
            expected = original.intersection(city_geometry)
            coverage = expected.area / original.area if original.area else 0.0
            require(coverage >= POLICY['district_city_overlap_minimum'],
                    'Existing district did not meet original city-overlap policy')
            require(geometry.equals(expected),
                    'Existing district geometry differs from same source polygon under original clipping')
            rule = 'topological equality with same original district clipped to exact named city polygon'
        covered.append({'city': CITY, 'district': district['district'], 'subarea_id': sid,
                        'source_feature_id': f'relation/{sid}', 'geometry_check': rule,
                        'original_polygon_fraction_retained': coverage,
                        'base_geometry_preserved': True})
    supplement_ids = {d['administration_evidence']['subarea_id'] for d in supplemental}
    covered_ids = {r['subarea_id'] for r in covered}
    require(supplement_ids.isdisjoint(covered_ids) and supplement_ids | covered_ids == available,
            'Available parent geometry not partitioned exactly once')
    proof = dict(complete_proof)
    proof.update({'available_subarea_ids': sorted(supplement_ids),
                  'complete_available_subarea_ids': sorted(available),
                  'complete_parent_subarea_ids': sorted(available | missing),
                  'already_covered_subarea_ids': sorted(covered_ids),
                  'already_covered_districts': covered,
                  'available_subarea_ids_semantics': 'Supplemental subset actually passed to add-only merge; complete inventory separately retained'})
    return supplemental, proof, covered


def self_check():
    result = v2_self_check()
    ids = [2106257, 2106258]
    def district(sid, name, geometry):
        return {'city': CITY, 'district': name, 'geometry': geometry,
                'sources': [f'relation/{sid}'], 'admin_levels': ['5'],
                'administration_evidence': {'parent_id': PARENT_ID, 'subarea_id': sid,
                    'member_type': 'r', 'member_role': 'subarea',
                    'source_pbf_sha256': 'synthetic', 'source_replication_timestamp': 'synthetic'}}
    parents = [district(ids[0], '甲區', box(120, 22, 120.01, 22.01)),
               district(ids[1], '乙區', box(120.01, 22, 120.02, 22.01))]
    proof = {'parent_id': PARENT_ID, 'city': CITY,
             'available_subarea_ids': ids, 'missing_subarea_ids': [2106668],
             'parent_subarea_ids': ids + [2106668],
             'source_pbf_sha256': 'synthetic', 'source_replication_timestamp': 'synthetic'}
    added, subset, covered = classify_parent_coverage([], parents, proof)
    require(len(merge_districts([], added, subset)) == 2 and not covered,
            'Absent city coverage should add both districts')
    added, subset, covered = classify_parent_coverage(parents, parents, proof)
    require(not added and len(covered) == 2 and len(merge_districts(parents, added, subset)) == 2,
            'Already complete city should not duplicate districts')
    added, subset, covered = classify_parent_coverage(parents[:1], parents, proof)
    require(len(added) == len(covered) == 1 and len(merge_districts(parents[:1], added, subset)) == 2,
            'Partial city coverage should add only its missing district')
    city = box(120, 22, 120.0199, 22.01)
    clipped = [{**d, 'geometry': d['geometry'].intersection(city)} for d in parents]
    added, subset, covered = classify_parent_coverage(clipped, parents, proof, city)
    require(not added and all(r['base_geometry_preserved'] for r in covered),
            'Original justified city clipping should be preserved')
    def rejected(base):
        try:
            classify_parent_coverage(base, parents, proof)
        except ValueError:
            return True
        return False
    require(rejected([{**parents[0], 'sources': ['relation/999']}]),
            'Conflicting source ID was accepted')
    require(rejected([{**parents[0], 'geometry': box(120, 22, 120.005, 22.01)}]),
            'Conflicting existing geometry was accepted')
    require(subset['missing_subarea_ids'] == [2106668], 'Missing geometry was relabeled as existing')
    result.update({'annual_no_duplicate_cases': 6,
                   'absent_partial_complete_city_cases_passed': True,
                   'existing_id_and_geometry_mismatch_rejected': True,
                   'original_city_clipping_preserved': True})
    return result


def compare_baseline_roads(new_path, baseline_path):
    """All old rows, including any existing Kaohsiung districts, stay identical."""
    old = iter(read_features(baseline_path))
    previous = next(old, None)
    count = 0
    for row in read_features(new_path):
        if previous is not None and row['key'] == previous['key']:
            require(row == previous, f'Baseline road changed: {row["key"]}')
            previous = next(old, None)
            count += 1
        else:
            require(row['city'] == CITY, f'Unexpected new non-parent road: {row["key"]}')
    require(previous is None, 'Original baseline roads missing or reordered')
    return count


def build_index(extraction_dir, parent_verified_dir, output_dir, *,
                baseline_dir=None, expected_base_districts=None,
                expected_added_districts=None, equivalent_index=None):
    started = time.perf_counter()
    source, parent_dir, out = map(lambda p: Path(p).resolve(),
                                  (extraction_dir, parent_verified_dir, output_dir))
    staging = out.with_name(out.name + '.building')
    require(not out.exists() and not staging.exists(), 'Refusing to overwrite output')
    checks = self_check()
    complete_extra, complete_proof, inputs = load_parent_districts(source, parent_dir)
    sources = snapshot([Path(__file__), ROOT/'src/geo_road_index_v2.py', ROOT/'src/geo_road_index.py',
                        ROOT/'src/geo_address_audit.py'])
    base, original_admin = boundaries(source/'administrative_areas.jsonl')
    city_geometry = existing_city_geometry(source/'administrative_areas.jsonl')
    extra, proof, covered = classify_parent_coverage(base, complete_extra, complete_proof, city_geometry)
    if expected_base_districts is not None:
        require(len(base) == expected_base_districts, 'Unexpected original district count')
    if expected_added_districts is not None:
        require(len(extra) == expected_added_districts, 'Unexpected additional district count')
    districts = merge_districts(base, extra, proof)
    preserved = {}
    for directory in (ROOT/'outputs/geography/road_index_2022',
                      ROOT/'outputs/geography/road_index_2022_stable'):
        if directory.exists():
            preserved.update(snapshot(p for p in directory.rglob('*') if p.is_file()))
    baseline = None
    if baseline_dir is not None:
        baseline = Path(baseline_dir).resolve()
        stable = load_json(baseline/'audit.json')
        require(stable['status'] == 'completed', 'Baseline index incomplete')
        require(stable['fixed_geographic_rules'] == POLICY, 'Baseline policy differs')
        for relative, sha in stable['source_sha256'].items():
            require(digest(ROOT/relative) == sha, 'Baseline helper source changed')
        for relative, sha in stable['input_sha256'].items():
            require(digest(ROOT/relative) == sha, 'Baseline input changed')
        for filename, sha in stable['output_sha256'].items():
            require(digest(baseline/filename) == sha, 'Baseline artifact changed')
        preserved.update(snapshot(p for p in baseline.rglob('*') if p.is_file()))
    print(json.dumps({'stage': 'inputs_verified', 'base_districts': len(base),
                      'added_districts': len(extra), 'districts': len(districts),
                      'pbf_time': proof['source_replication_timestamp']}), flush=True)
    tree = STRtree([d['geometry'] for d in districts])
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    for road in read_features(source/'roads.jsonl'):
        counts['source_road_ways'] += 1
        aliases = names(road['properties'], True)
        geometry = shape(road['geometry'])
        if not aliases or geometry.is_empty:
            continue
        matched = False
        for d, clipped in district_clips(geometry, districts, tree):
            matched = True
            for name in aliases:
                groups[(d['city'], d['district'], name)].append(
                    (clipped, road['osm_id'], road['properties']['highway']))
            counts['road_district_fragments'] += 1
            if d['city'] == CITY:
                counts['parent_verified_road_district_fragments'] += 1
        if matched:
            counts['ways_assigned_to_district'] += 1
        if counts['source_road_ways'] % 50000 == 0:
            print(json.dumps(dict(counts)), flush=True)
    staging.mkdir(parents=True)
    quality = collections.Counter()
    bycity = collections.Counter()
    parent_quality = collections.Counter()
    with (staging/'road_index.jsonl').open('w', encoding='utf-8') as handle:
        for key, items in sorted(groups.items()):
            row = road_summary(key, items)
            quality[row['quality']] += 1
            bycity[row['city']] += 1
            if row['city'] == CITY:
                parent_quality[row['quality']] += 1
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    inventory = [{k: v for k, v in d.items() if k != 'geometry'} for d in districts]
    write_json(staging/'district_inventory.json', inventory)
    write_json(staging/'self_check.json', checks)
    original_road_count = (compare_baseline_roads(staging/'road_index.jsonl',
                          baseline/'road_index.jsonl') if baseline else None)
    equivalence = None
    if equivalent_index is not None:
        equivalent = Path(equivalent_index).resolve()
        old_roads = equivalent/'road_index.jsonl'
        require(digest(staging/'road_index.jsonl') == digest(old_roads), 'Annual wrapper road output differs from equivalent previous index')
        old_inventory = load_json(equivalent/'district_inventory.json')
        fields = ('city', 'district', 'sources', 'admin_levels')
        core = lambda rows: [{k: row[k] for k in fields} for row in rows]
        require(core(old_inventory) == core(inventory), 'Equivalent district identity inventory differs')
        inputs.update(snapshot([old_roads, equivalent/'district_inventory.json', equivalent/'audit.json']))
        equivalence = {'index_directory': str(equivalent), 'road_index_bytes_identical': True,
                       'district_identity_inventory_identical': True,
                       'parent_audit_lineage_can_differ': True, 'road_index_sha256': digest(old_roads)}
    verify_snapshot(inputs)
    verify_snapshot(sources)
    verify_snapshot(preserved)
    administration = dict(original_admin)
    administration.update({
        'districts': len(districts), 'original_districts': len(base),
        'parent_verified_additional_districts': len(extra),
        'parent_verified_already_covered_districts': len(covered),
        'parent_verified_complete_available_districts': len(complete_extra),
        'districts_by_city': dict(collections.Counter(d['city'] for d in districts)),
        'cities_with_districts': sorted({d['city'] for d in districts}),
        'complete_city_boundary_count': len(original_admin['cities_found']),
        'parent_only_partial_cities': [CITY] if CITY not in original_admin['cities_found'] else [], 'city_union_created': False,
        'parent_evidence': proof,
        'method': original_admin['method'] + '; add exact original district polygons only when same-snapshot parent relation explicitly lists them as subarea; no city union',
        'original_rejected_district_assignments': original_admin['rejected_district_assignments'],
        'parent_resolved_original_rejections': [r for r in original_admin['rejected_district_assignments']
             if r['source_id'] in {s for d in extra for s in d['sources']}],
    })
    payload = {
        'status': 'completed', 'format_version': 3,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter() - started,
        'input_sha256': inputs, 'source_sha256': sources,
        'preserved_original_artifact_sha256': preserved,
        'source_replication_timestamp': proof['source_replication_timestamp'],
        'versions': {p: version(p) for p in ('osmium', 'shapely', 'pyproj', 'numpy', 'pandas')},
        'administration': administration, 'counts': dict(counts),
        'road_keys': len(groups), 'quality_counts': dict(quality),
        'road_keys_by_city': dict(bycity), 'parent_verified_quality_counts': dict(parent_quality),
        'baseline_original_road_rows_exactly_equal': original_road_count,
        'equivalence_check': equivalence,
        'fixed_geographic_rules': POLICY, 'self_check': checks,
        'projection': 'EPSG:3826 approximate metric geometry; stored output coordinates EPSG:4326',
        'targets_read': False, 'housing_inputs_read': False,
        'house_coordinates_claimed': False, 'price_models_trained': False,
        'license': 'ODbL-1.0; OpenStreetMap contributors; Geofabrik extract',
        'limitations': ['Road representatives are not dwelling coordinates.',
                       'Parent-listed districts do not establish a complete city boundary.',
                       'Unassembled or missing district geometries remain missing.',
                       'PBF actual header time is not an earlier as-of map or road opening date.'],
        'output_sha256': {p.name: digest(p) for p in staging.iterdir() if p.is_file()},
    }
    write_json(staging/'audit.json', payload)
    (staging/'audit.sha256').write_text(digest(staging/'audit.json') + '  audit.json\n', encoding='utf-8')
    staging.rename(out)
    print(json.dumps({'status': 'completed', 'output': str(out), 'districts': len(districts),
                      'road_keys': len(groups), 'quality_counts': dict(quality),
                      'audit_sha256': digest(out/'audit.json')}, ensure_ascii=False), flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--parent-verified', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--expected-base-districts', type=int)
    parser.add_argument('--expected-added-districts', type=int)
    parser.add_argument('--equivalent-index', type=Path)
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else:
        if not all((args.source, args.parent_verified, args.output)):
            parser.error('--source, --parent-verified and --output are required')
        build_index(args.source, args.parent_verified, args.output,
                    baseline_dir=args.baseline,
                    expected_base_districts=args.expected_base_districts,
                    expected_added_districts=args.expected_added_districts,
                    equivalent_index=args.equivalent_index)
