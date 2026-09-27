"""Export verified Kaohsiung district geometry using explicit OSM parent refs.

Only the fixed 2022 geography snapshot is accepted. This is a provenance join,
not a reconstructed city boundary: no union, repair, clipping, interpolation,
road-index update, housing input, target, or model is involved.

python src/geo_parent_coverage.py --pbf SOURCE.osm.pbf --areas administrative_areas.jsonl --output NEW_DIRECTORY
Use the separate .venv-geo environment; importing this module does no I/O.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PARENT_ID = 2127079
CITY = '高雄市'
EXPECTED_PBF_SHA = 'b7650b8c77deecb27592782cada8241f88b7cdbc2838bf8e14b20c7957e63ff9'
EXPECTED_TIME = '2022-01-01T21:21:26Z'
EXPECTED_MISSING = {2106668}
NAME_KEYS = ('name', 'name:zh', 'name:zh-Hant', 'alt_name', 'short_name')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False)+'\n', encoding='utf-8')


def geographic_tags(tags):
    return {key: str(tags.get(key)) if tags.get(key) is not None else None
            for key in NAME_KEYS+('boundary', 'admin_level')}


def read_relations(pbf, osmium, ids):
    result = {}
    for obj in (osmium.FileProcessor(str(pbf), osmium.osm.RELATION)
                .with_filter(osmium.filter.IdFilter(sorted(ids)))):
        oid = int(obj.id)
        require(oid not in result, 'Duplicate relation version in snapshot.')
        record = {'id': oid, 'tags': geographic_tags(obj.tags)}
        if oid == PARENT_ID:
            # Copy scalar fields now; never retain OSM views beyond this iteration.
            record['members'] = [{'type': str(member.type), 'ref': int(member.ref),
                                  'role': str(member.role)} for member in obj.members]
            record['affiliation_tags'] = {str(k): str(v) for k, v in obj.tags
                                          if k.startswith(('is_in', 'ISO3166', 'admin:'))}
        result[oid] = record
    require(set(result) == set(ids), 'A required parent/subarea relation is absent from the PBF.')
    return result


def run(pbf, areas, output):
    import osmium
    import shapely
    require(version('osmium') == '4.3.1', 'Use osmium==4.3.1 in the isolated geography environment.')
    pbf, areas, output = Path(pbf).resolve(), Path(areas).resolve(), Path(output).resolve()
    require(pbf.is_file() and areas.is_file(), 'PBF and areas JSONL must exist locally.')
    require(not output.exists(), 'Refusing to overwrite an existing output directory.')
    require(output.is_relative_to(ROOT/'outputs/geography') and output != ROOT/'outputs/geography', 'Output must be a new subdirectory of outputs/geography.')
    require(not output.is_relative_to(areas.parent), 'Do not write into the existing extraction directory.')
    source_audit_path = areas.parent/'audit.json'
    receipt = source_audit_path.with_suffix('.sha256')
    require(source_audit_path.is_file() and receipt.is_file(), 'Source extraction audit and SHA receipt are required.')
    source_audit_sha = sha256(source_audit_path)
    require(receipt.read_text(encoding='ascii').strip() == source_audit_sha, 'Source extraction audit receipt differs.')
    source_audit = json.loads(source_audit_path.read_text(encoding='utf-8'))
    pbf_sha, areas_sha = sha256(pbf), sha256(areas)
    require(pbf_sha == EXPECTED_PBF_SHA == source_audit['source']['sha256'], 'PBF differs from the fixed source extraction.')
    require(pbf.stat().st_size == source_audit['source']['bytes'], 'PBF size differs from source audit.')
    declared = source_audit['files']['administrative_areas']
    require(areas_sha == declared['sha256'] and areas.stat().st_size == declared['bytes'], 'Administrative geometry differs from source audit.')
    require(source_audit['status'] == 'complete' and source_audit['crs'] == 'EPSG:4326', 'Source geometry audit must be complete WGS84 extraction.')
    with osmium.io.Reader(str(pbf), osmium.osm.NOTHING) as reader:
        header = reader.header()
        timestamp = str(header.get('osmosis_replication_timestamp', ''))
        multiple_versions = bool(header.has_multiple_object_versions)
    require(not multiple_versions and timestamp == EXPECTED_TIME, 'Actual PBF timestamp or snapshot type differs.')
    require(timestamp == source_audit['source']['header']['replication_timestamp'], 'PBF header and extraction timestamp differ.')
    parent = read_relations(pbf, osmium, {PARENT_ID})[PARENT_ID]
    require(parent['tags']['boundary'] == 'administrative' and
            (parent['tags']['name:zh'] or parent['tags']['name']) == CITY, 'Wrong administrative parent.')
    subareas = [m['ref'] for m in parent['members'] if m['type'] == 'r' and m['role'] == 'subarea']
    require(len(subareas) == len(set(subareas)) == 38, 'Expected 38 unique explicit relation/subarea refs.')
    require(all(m['role'] == 'subarea' for m in parent['members'] if m['type'] == 'r'), 'Unexpected non-subarea relation membership.')
    child_meta = read_relations(pbf, osmium, set(subareas))
    role_counts = [{'type': t, 'role': r, 'count': n} for (t,r),n in sorted(
        Counter((m['type'],m['role']) for m in parent['members']).items())]
    existing = {}
    with areas.open(encoding='utf-8') as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            if record.get('osm_type') != 'relation' or record.get('osm_id') not in child_meta:
                continue
            oid = int(record['osm_id'])
            require(oid not in existing and record['id'] == f'relation/{oid}', 'Ambiguous or inconsistent child geometry ID.')
            require(record['type'] == 'Feature' and record['geometry']['type'] in {'Polygon','MultiPolygon'}, 'Expected polygon subarea feature.')
            # Names/admin tags must agree with the same original PBF, not a name-based guess.
            require(record['properties'] == child_meta[oid]['tags'], 'PBF tags and extracted child feature differ.')
            geometry = shapely.from_geojson(json.dumps(record['geometry']))
            require(not geometry.is_empty and geometry.is_valid, 'Existing child geometry is not valid.')
            existing[oid] = record
    missing = set(subareas)-set(existing)
    require(len(existing) == 37 and missing == EXPECTED_MISSING, 'Expected 37 usable subareas and only Cijin missing.')
    require((child_meta[2106668]['tags']['name:zh'] or child_meta[2106668]['tags']['name']) == '旗津區', 'Missing child is not the expected Cijin relation.')
    inputs = {'pbf': {'path': str(pbf), 'sha256': pbf_sha, 'bytes': pbf.stat().st_size},
              'administrative_areas': {'path': str(areas), 'sha256': areas_sha, 'bytes': areas.stat().st_size},
              'source_extraction_audit': {'path': str(source_audit_path), 'sha256': source_audit_sha},
              'source_extraction_receipt': {'path': str(receipt), 'sha256': sha256(receipt)},
              'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)}}
    provenance = {'source_pbf_sha256': pbf_sha, 'source_replication_timestamp': timestamp,
                  'source_areas_sha256': areas_sha, 'source_extraction_audit_sha256': source_audit_sha}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent/('.'+output.name+'-staging-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        exported, missing_records = [], []
        geometry_path = stage/'districts.jsonl'
        with geometry_path.open('w', encoding='utf-8', newline='\n') as handle:
            for oid in sorted(subareas):
                name = child_meta[oid]['tags']['name:zh'] or child_meta[oid]['tags']['name']
                lineage = {'city': CITY, 'parent_id': PARENT_ID, 'subarea_id': oid,
                           'name': name, 'parent_member_type': 'r', 'parent_member_role': 'subarea',
                           'source_feature_id': f'relation/{oid}' if oid in existing else None, **provenance}
                if oid in missing:
                    missing_records.append({**lineage, 'status': 'missing_geometry',
                        'reason': 'Explicit child relation exists, but no successful feature exists in the source areas JSONL.',
                        'exact_geometry_failure_cause_confirmed': False})
                    continue
                original = existing[oid]
                record = {**original, 'city': CITY, 'parent_id': PARENT_ID, 'subarea_id': oid,
                          'name': name, 'source_feature_id': original['id'],
                          'properties': {**original['properties'], **lineage}}
                require(record['geometry'] == original['geometry'] and record['bbox'] == original['bbox'], 'Geometry or bounds changed during provenance export.')
                handle.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
                exported.append(lineage)
        write_json(stage/'missing_subareas.json', missing_records)
        # Check the serialized geometry is an exact JSON-value copy of the audited source.
        saved = [json.loads(line) for line in geometry_path.read_text(encoding='utf-8').splitlines()]
        require(len(saved) == 37 and {r['subarea_id'] for r in saved} == set(existing), 'Saved district inventory differs.')
        for record in saved:
            require(record['geometry'] == existing[record['subarea_id']]['geometry'], 'Serialized geometry differs from original.')
        for info in inputs.values():
            require(sha256(info['path']) == info['sha256'], 'An input changed during verification/export.')
        audit = {'format_version': 1, 'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'Explicit OSM parent/subarea provenance join, not a city boundary reconstruction or predictive model.',
            'inputs': inputs, 'source_replication_timestamp': timestamp,
            'city': CITY, 'parent_id': PARENT_ID, 'parent_tags': parent['tags'],
            'parent_affiliation_tags': parent['affiliation_tags'], 'parent_member_count': len(parent['members']),
            'parent_member_type_role_counts': role_counts, 'explicit_subarea_ids': sorted(subareas),
            'explicit_subarea_count': 38, 'exported_district_count': 37, 'missing_subarea_count': 1,
            'exported_subareas': exported, 'missing_subareas': missing_records,
            'crs': 'EPSG:4326', 'coordinate_order': ['longitude','latitude'],
            'original_geometry_json_values_preserved': True, 'city_union_created': False,
            'failed_geometry_reconstructed': False, 'road_index_or_matching_rules_modified': False,
            'housing_inputs_or_targets_read': False, 'price_model_trained_or_scored': False,
            'current_scope': 'Separate provenance artifact only; the existing 21-city road index is unchanged.',
            'limitations': ['37 district geometries do not establish the complete city boundary; Cijin remains missing.',
                           'No claim is made about the exact cause of the unavailable original geometry.',
                           'This source snapshot must not be presented as available before its actual 2022 timestamp.'],
            'license': source_audit['license'],
            'versions': {'python': platform.python_version(), 'osmium': version('osmium'), 'shapely': version('shapely')},
            'files': {p.name: {'path': str(output/p.name), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                      for p in [geometry_path, stage/'missing_subareas.json']}}
        write_json(stage/'audit.json', audit)
        (stage/'audit.sha256').write_text(sha256(stage/'audit.json')+'\n', encoding='ascii')
        require(not output.exists(), 'Output appeared during export; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        print('Incomplete provenance export; staging retained: '+str(stage), file=sys.stderr, flush=True)
        raise
    result = {'status': 'complete', 'output': str(output), 'subareas': 38, 'exported': 37, 'missing': 1,
              'missing_ids': sorted(missing), 'audit_sha256': sha256(output/'audit.json'),
              'districts_sha256': sha256(output/'districts.jsonl'), 'source_sha256': inputs['script']['sha256']}
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pbf', type=Path, required=True)
    parser.add_argument('--areas', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.pbf, args.areas, args.output)


if __name__ == '__main__':
    main()