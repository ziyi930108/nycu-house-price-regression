"""Add only parent-verified districts to an annual road index; no price inputs."""
from __future__ import annotations
import argparse
import collections
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import osmium
from shapely.geometry import LineString, Point, box, shape
from shapely.strtree import STRtree
from geo_road_index import (ROOT, boundaries, digest, linear, names, norm,
                            read_features, road_summary)

PARENT_ID = 2127079
CITY = '高雄市'
POLICY = {
    'district_city_overlap_minimum': .98,
    'component_buffer_metres': 25,
    'maximum_usable_diagonal_metres': 10000,
    'exact_alias_only': True,
    'all_disconnected_after_50m_gap_tolerance_withhold_coordinates': True,
    'nearby_fragments_can_share_a_road_representative': True,
    'physical_road_connectivity_is_not_proven': True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def checked_audit(directory):
    path = Path(directory) / 'audit.json'
    receipt = Path(directory) / 'audit.sha256'
    require(receipt.read_text(encoding='utf-8').split()[0] == digest(path),
            f'Audit receipt mismatch: {path}')
    audit = load_json(path)
    require(audit['status'] == 'complete', f'Incomplete evidence: {path}')
    return audit


def snapshot(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def verify_snapshot(hashes):
    for path, sha in hashes.items():
        require(digest(path) == sha, f'Input/source changed: {path}')


def fresh_parent(pbf):
    """Copy the parent relation while its osmium view is alive."""
    found = []
    class ParentReader(osmium.SimpleHandler):
        def relation(self, relation):
            if relation.id == PARENT_ID:
                found.append({
                    'id': int(relation.id),
                    'tags': {str(t.k): str(t.v) for t in relation.tags},
                    'members': [{'type': str(m.type), 'ref': int(m.ref),
                                 'role': str(m.role)} for m in relation.members],
                })
    ParentReader().apply_file(str(pbf), locations=False)
    require(len(found) == 1, 'Expected exactly one parent relation in PBF')
    parent = found[0]
    require(CITY in names(parent['tags']) and
            parent['tags'].get('boundary') == 'administrative',
            'Parent identity is not the named Kaohsiung administrative boundary')
    refs = [m['ref'] for m in parent['members']
            if m['type'] == 'r' and m['role'] == 'subarea']
    require(len(refs) == len(set(refs)), 'Duplicate parent subarea refs')
    parent['subarea_ids'] = sorted(refs)
    return parent


def load_parent_districts(extraction_dir, parent_verified_dir):
    """Validate annual evidence, then return district records, proof and hashes.

    No snapshot, district count, or permanent missing district is hard-coded.
    An annual parent artifact must bind to that same year's extracted geometry.
    """
    source = Path(extraction_dir).resolve()
    parent_dir = Path(parent_verified_dir).resolve()
    extracted = checked_audit(source)
    audit = checked_audit(parent_dir)
    require(audit['parent_id'] == PARENT_ID and audit['city'] == CITY,
            'Wrong administrative parent')
    require(audit['crs'] == extracted['crs'] == 'EPSG:4326', 'Unexpected CRS')
    require(audit['original_geometry_json_values_preserved'] is True and
            audit['city_union_created'] is False and
            audit['failed_geometry_reconstructed'] is False,
            'Parent geometry must be unchanged original district geometry')
    bound_paths = [source/'audit.json', source/'audit.sha256',
                   parent_dir/'audit.json', parent_dir/'audit.sha256']
    for record in audit['inputs'].values():
        path = Path(record['path'])
        require(digest(path) == record['sha256'], f'Parent input mismatch: {path}')
        bound_paths.append(path)
    for filename, record in audit['files'].items():
        path = parent_dir/filename
        require(digest(path) == record['sha256'], f'Parent output mismatch: {path}')
        bound_paths.append(path)
    for key, filename in [('roads', 'roads.jsonl'),
                          ('administrative_areas', 'administrative_areas.jsonl')]:
        require(digest(source/filename) == extracted['files'][key]['sha256'],
                f'Extraction file mismatch: {filename}')
        bound_paths.append(source/filename)
    pbf = Path(extracted['source']['path'])
    pbf_sha = digest(pbf)
    require(pbf_sha == extracted['source']['sha256'] ==
            audit['inputs']['pbf']['sha256'], 'Different PBF snapshots')
    with osmium.io.Reader(str(pbf), osmium.osm.NOTHING) as reader:
        timestamp = reader.header().get('osmosis_replication_timestamp')
    require(bool(timestamp) and timestamp ==
            extracted['source']['header']['replication_timestamp'] ==
            audit['source_replication_timestamp'], 'PBF header time mismatch')
    areas_sha = digest(source/'administrative_areas.jsonl')
    require(areas_sha == audit['inputs']['administrative_areas']['sha256'],
            'Parent and index administrative geometries differ')
    require(digest(source/'audit.json') ==
            audit['inputs']['source_extraction_audit']['sha256'],
            'Parent and index extraction audit differ')
    parent = fresh_parent(pbf)
    refs = set(parent['subarea_ids'])
    declared_refs = set(audit['explicit_subarea_ids'])
    require(refs == declared_refs and len(refs) == audit['explicit_subarea_count'],
            'Fresh PBF parent refs disagree with evidence')
    missing = audit['missing_subareas']
    missing_ids = {r['subarea_id'] for r in missing}
    require(len(missing_ids) == len(missing) == audit['missing_subarea_count'],
            'Duplicate or inconsistent missing subareas')
    original = {}
    for feature in read_features(source/'administrative_areas.jsonl'):
        if feature['osm_type'] == 'relation' and feature['osm_id'] in refs:
            require(feature['osm_id'] not in original, 'Duplicate source district')
            original[feature['osm_id']] = feature
    require(set(original) == refs - missing_ids,
            'Missing/available declarations differ from original area features')
    rows = list(read_features(parent_dir/'districts.jsonl'))
    available_ids = {r['subarea_id'] for r in rows}
    require(len(available_ids) == len(rows) == audit['exported_district_count'],
            'Duplicate or inconsistent exported districts')
    require(available_ids.isdisjoint(missing_ids) and
            available_ids | missing_ids == refs, 'Subarea evidence not exhaustive')
    extra = []
    for feature in rows:
        sid = feature['subarea_id']
        raw = original[sid]
        prop = feature['properties']
        require(feature['osm_type'] == 'relation' and feature['osm_id'] == sid and
                feature['id'] == raw['id'] == f'relation/{sid}', 'ID join mismatch')
        require(feature['geometry'] == raw['geometry'] and
                feature['bbox'] == raw['bbox'], 'Source geometry was modified')
        require(prop['city'] == feature['city'] == CITY and
                prop['parent_id'] == feature['parent_id'] == PARENT_ID and
                prop['subarea_id'] == sid and prop['parent_member_type'] == 'r' and
                prop['parent_member_role'] == 'subarea', 'Missing parent evidence')
        require(prop['source_pbf_sha256'] == pbf_sha and
                prop['source_replication_timestamp'] == timestamp and
                prop['source_areas_sha256'] == areas_sha and
                prop['source_feature_id'] == raw['id'] and
                prop['source_extraction_audit_sha256'] == digest(source/'audit.json'),
                'District lineage mismatch')
        require(all(prop.get(k) == v for k, v in raw['properties'].items()),
                'Original area properties were modified')
        district = norm(feature['name'])
        require(district in names(raw['properties']), 'District name is not an exact alias')
        geometry = shape(raw['geometry'])
        require(geometry.geom_type in ('Polygon', 'MultiPolygon') and
                geometry.is_valid and not geometry.is_empty,
                'Parent district geometry must be valid without repair')
        extra.append({'city': CITY, 'district': district, 'geometry': geometry,
                      'sources': [raw['id']],
                      'admin_levels': [str(raw['properties'].get('admin_level'))],
                      'administration_evidence': {
                          'method': 'explicit_parent_relation_subarea',
                          'parent_id': PARENT_ID, 'subarea_id': sid,
                          'member_type': 'r', 'member_role': 'subarea',
                          'source_pbf_sha256': pbf_sha,
                          'source_replication_timestamp': timestamp,
                          'source_areas_sha256': areas_sha,
                          'parent_audit_sha256': digest(parent_dir/'audit.json')}})
    proof = {'parent_id': PARENT_ID, 'city': CITY,
             'available_subarea_ids': sorted(available_ids),
             'missing_subarea_ids': sorted(missing_ids),
             'missing_subareas': missing, 'parent_subarea_ids': sorted(refs),
             'source_pbf_sha256': pbf_sha, 'source_replication_timestamp': timestamp,
             'source_areas_sha256': areas_sha,
             'parent_audit_sha256': digest(parent_dir/'audit.json')}
    return extra, proof, snapshot(bound_paths + [pbf])


def merge_districts(base_districts, additional_districts, parent_evidence):
    """Explicit merge API: unnamed or unevidenced supplemental districts fail."""
    require(bool(parent_evidence), 'Supplemental districts require parent evidence')
    require(parent_evidence['parent_id'] == PARENT_ID and
            parent_evidence['city'] == CITY, 'Wrong parent proof')
    allowed = set(parent_evidence['available_subarea_ids'])
    missing = set(parent_evidence['missing_subarea_ids'])
    require(allowed.isdisjoint(missing), 'Available and missing refs overlap')
    result = list(base_districts)
    keys = {(d['city'], d['district']) for d in result}
    require(len(keys) == len(result), 'Duplicate original district key')
    seen_ids = set()
    for d in additional_districts:
        ev = d.get('administration_evidence', {})
        sid = ev.get('subarea_id')
        require(d['city'] == CITY and ev.get('parent_id') == PARENT_ID and
                ev.get('member_type') == 'r' and ev.get('member_role') == 'subarea' and
                sid in allowed and sid not in missing and sid not in seen_ids,
                'Supplemental district has no available parent subarea evidence')
        require(ev.get('source_pbf_sha256') == parent_evidence['source_pbf_sha256'] and
                ev.get('source_replication_timestamp') == parent_evidence['source_replication_timestamp'],
                'Supplemental district belongs to another snapshot')
        key = (d['city'], d['district'])
        require(key not in keys, 'Supplemental district would replace an existing key')
        keys.add(key)
        seen_ids.add(sid)
        result.append(d)
    require(seen_ids == allowed, 'Not all evidenced available districts supplied')
    return sorted(result, key=lambda d: (d['city'], d['district']))


def district_clips(geometry, districts, tree):
    for index in tree.query(geometry, predicate='intersects'):
        district = districts[int(index)]
        clipped = linear(geometry.intersection(district['geometry']))
        if not clipped.is_empty and clipped.length > 0:
            yield district, clipped


def self_check():
    """Small synthetic cross-city geometry; no housing data or real estimates."""
    proof = {'parent_id': PARENT_ID, 'city': CITY,
             'available_subarea_ids': [2127080], 'missing_subarea_ids': [2106668],
             'source_pbf_sha256': 'synthetic', 'source_replication_timestamp': 'synthetic'}
    base = [{'city': '臺南市', 'district': '南區',
             'geometry': box(120, 22.9, 120.01, 22.91), 'sources': ['synthetic']}]
    ev = dict(parent_id=PARENT_ID, subarea_id=2127080, member_type='r',
              member_role='subarea', source_pbf_sha256='synthetic',
              source_replication_timestamp='synthetic')
    extra = [{'city': CITY, 'district': '茄萣區',
              'geometry': box(120, 22.89, 120.01, 22.9),
              'sources': ['synthetic'], 'administration_evidence': ev}]
    districts = merge_districts(base, extra, proof)
    crossing = LineString([(120.005, 22.895), (120.005, 22.905)])
    clips = list(district_clips(crossing, districts,
                                STRtree([d['geometry'] for d in districts])))
    require(len(clips) == 2, 'Cross-city road should produce exactly two fragments')
    keys = set()
    for d, g in clips:
        require(d['geometry'].covers(g) and g.length < crossing.length,
                'Road fragment escaped its assigned district')
        row = road_summary((d['city'], d['district'], '合成跨界路'), [(g, 1, 'residential')])
        require(d['geometry'].buffer(1e-10).covers(Point(row['longitude'], row['latitude'])),
                'Road representative escaped its assigned district')
        keys.add(row['key'])
    require(len(keys) == 2 and clips[0][1].intersection(clips[1][1]).length == 0,
            'Cross-city road keys or interior fragments merged')
    def rejected(call):
        try:
            call()
        except ValueError:
            return True
        return False
    require(rejected(lambda: merge_districts(base, extra, None)),
            'Missing parent proof was accepted')
    unproved = [{**extra[0], 'administration_evidence': {}}]
    require(rejected(lambda: merge_districts(base, unproved, proof)),
            'Unevidenced supplemental district was accepted')
    cijin = [{**extra[0], 'district': '旗津區',
              'administration_evidence': {**ev, 'subarea_id': 2106668}}]
    require(rejected(lambda: merge_districts(base, cijin, proof)),
            'Missing Cijin geometry was filled')
    return {'status': 'passed', 'synthetic_only': True,
            'cross_city_road_separate_clipped_keys': True,
            'representatives_stay_in_assigned_district': True,
            'no_parent_evidence_rejected': True, 'missing_cijin_rejected': True,
            'shared_road_summary_function': True}


def compare_original_roads(new_path, baseline_path):
    count = 0
    prior = iter(read_features(baseline_path))
    for row in read_features(new_path):
        if row['city'] == CITY:
            continue
        old = next(prior, None)
        require(old == row, f'Original non-Kaohsiung road changed: {row["key"]}')
        count += 1
    require(next(prior, None) is None, 'Original roads missing from new index')
    return count


def build_index(extraction_dir, parent_verified_dir, output_dir, *,
                baseline_dir=None, expected_base_districts=None,
                expected_added_districts=None):
    started = time.perf_counter()
    source, parent_dir, out = map(lambda p: Path(p).resolve(),
                                  (extraction_dir, parent_verified_dir, output_dir))
    staging = out.with_name(out.name + '.building')
    require(not out.exists() and not staging.exists(), 'Refusing to overwrite output')
    checks = self_check()
    extra, proof, inputs = load_parent_districts(source, parent_dir)
    sources = snapshot([Path(__file__), ROOT/'src/geo_road_index.py',
                        ROOT/'src/geo_address_audit.py'])
    base, original_admin = boundaries(source/'administrative_areas.jsonl')
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
    original_road_count = (compare_original_roads(staging/'road_index.jsonl',
                          baseline/'road_index.jsonl') if baseline else None)
    verify_snapshot(inputs)
    verify_snapshot(sources)
    verify_snapshot(preserved)
    administration = dict(original_admin)
    administration.update({
        'districts': len(districts), 'original_districts': len(base),
        'parent_verified_additional_districts': len(extra),
        'districts_by_city': dict(collections.Counter(d['city'] for d in districts)),
        'cities_with_districts': sorted({d['city'] for d in districts}),
        'complete_city_boundary_count': len(original_admin['cities_found']),
        'parent_only_partial_cities': [CITY], 'city_union_created': False,
        'parent_evidence': proof,
        'method': original_admin['method'] + '; add exact original district polygons only when same-snapshot parent relation explicitly lists them as subarea; no city union',
        'original_rejected_district_assignments': original_admin['rejected_district_assignments'],
        'parent_resolved_original_rejections': [r for r in original_admin['rejected_district_assignments']
             if r['source_id'] in {s for d in extra for s in d['sources']}],
    })
    payload = {
        'status': 'completed', 'format_version': 2,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter() - started,
        'input_sha256': inputs, 'source_sha256': sources,
        'preserved_original_artifact_sha256': preserved,
        'source_replication_timestamp': proof['source_replication_timestamp'],
        'versions': {p: version(p) for p in ('osmium', 'shapely', 'pyproj', 'numpy', 'pandas')},
        'administration': administration, 'counts': dict(counts),
        'road_keys': len(groups), 'quality_counts': dict(quality),
        'road_keys_by_city': dict(bycity), 'parent_verified_quality_counts': dict(parent_quality),
        'baseline_non_parent_road_rows_exactly_equal': original_road_count,
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
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    else:
        parser.error('--source, --parent-verified and --output are required') if not all(
            (args.source, args.parent_verified, args.output)) else None
        build_index(args.source, args.parent_verified, args.output,
                    baseline_dir=args.baseline,
                    expected_base_districts=args.expected_base_districts,
                    expected_added_districts=args.expected_added_districts)