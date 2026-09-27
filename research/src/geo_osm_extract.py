"""Extract geography only from a local OSM PBF; never download or load ML data.

Use the separately provisioned pyosmium==4.3.1/Shapely environment:
  python src/geo_osm_extract.py --pbf INPUT.osm.pbf --output NEW_DIRECTORY
  python src/geo_osm_extract.py --self-check --output NEW_SYNTHETIC_DIRECTORY

Output is GeoJSON Features in roads.jsonl / administrative_areas.jsonl.
The source WGS84 coordinates remain longitude, latitude (EPSG:4326). No
centroids, metre distances, geocoding, prices, buildings, or POIs are emitted.
All OSM views are consumed immediately; only copied strings/ints/geometries
survive an iterator step. No data processing takes place at import time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time
import uuid

NAME_KEYS = ('name', 'name:zh', 'name:zh-Hant', 'alt_name', 'short_name')
CRS = 'EPSG:4326'
DOCS = [
    'https://docs.osmcode.org/pyosmium/latest/user_manual/03-Working-with-Geometries/',
    'https://docs.osmcode.org/pyosmium/latest/user_manual/02-Extracting-Object-Data/',
    'https://docs.osmcode.org/pyosmium/latest/reference/File-Processing/',
    'https://docs.osmcode.org/pyosmium/latest/reference/IO/',
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


def emit_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def dependencies():
    import osmium
    import shapely
    require(version('osmium') == '4.3.1', 'Use the isolated environment with osmium==4.3.1.')
    return osmium, shapely


def names(tags):
    # Values are copied while the OSM view is alive, without normalizing aliases.
    return {key: str(tags.get(key)) if tags.get(key) is not None else None for key in NAME_KEYS}


def source_header(pbf, osmium, pool):
    with osmium.io.Reader(str(pbf), osmium.osm.NOTHING, thread_pool=pool) as reader:
        header = reader.header()
        fields = {key: str(header.get(key, '')) for key in [
            'osmosis_replication_timestamp', 'osmosis_replication_sequence_number',
            'osmosis_replication_base_url', 'generator', 'source']}
        multiple_versions = bool(header.has_multiple_object_versions)
    actual = fields['osmosis_replication_timestamp']
    require(bool(actual), 'PBF header has no osmosis_replication_timestamp; do not infer it from the filename.')
    stamp = datetime.fromisoformat(actual.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None, 'Replication timestamp must specify its time zone.')
    require(not multiple_versions, 'History/diff PBF is unsupported; use a single snapshot.')
    return {'replication_timestamp': stamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'replication_timestamp_raw': actual, 'replication_timestamp_header_key': 'osmosis_replication_timestamp',
            'has_multiple_object_versions': multiple_versions, 'options': fields,
            'timestamp_interpretation': 'Actual PBF replication header, not road opening dates or filename date.'}


def geometry_failure(geometry, expected):
    if geometry.is_empty:
        return 'empty_geometry'
    if geometry.geom_type not in expected:
        return 'unexpected_geometry_type_' + geometry.geom_type
    bounds = geometry.bounds
    if len(bounds) != 4 or not all(math.isfinite(float(x)) for x in bounds):
        return 'nonfinite_geometry'
    if not (-180 <= bounds[0] <= bounds[2] <= 180 and -90 <= bounds[1] <= bounds[3] <= 90):
        return 'outside_wgs84_coordinate_range'
    if not geometry.is_valid:
        return 'invalid_geometry'
    if geometry.geom_type == 'LineString' and geometry.length <= 0:
        return 'zero_length_line'
    return None


def feature(osm_type, osm_id, properties, geometry, shapely):
    # shapely.to_geojson copies all coordinate arrays. No OSM view is retained.
    return {'type': 'Feature', 'id': f'{osm_type}/{osm_id}', 'osm_type': osm_type,
            'osm_id': int(osm_id), 'bbox': [float(x) for x in geometry.bounds],
            'properties': properties, 'geometry': json.loads(shapely.to_geojson(geometry))}


def extract(pbf, output):
    pbf, output = Path(pbf).resolve(), Path(output).resolve()
    require(pbf.is_file() and pbf.suffix.lower() == '.pbf', 'Expected an existing local .pbf file.')
    require(not output.exists(), 'Refusing to overwrite an existing output directory: ' + str(output))
    require(pbf != output and not pbf.is_relative_to(output), 'Output cannot contain the source PBF.')
    osmium, shapely = dependencies()
    started = time.perf_counter()
    pool = osmium.io.ThreadPool(4, 8)
    header = source_header(pbf, osmium, pool)
    print(json.dumps({'event': 'source_header', **header}, ensure_ascii=True), flush=True)
    initial_sha, initial_bytes = sha256(pbf), pbf.stat().st_size
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '-staging-' + uuid.uuid4().hex)
    stage.mkdir()
    counts, error_counts, highway_counts, admin_counts = Counter(), Counter(), Counter(), Counter()
    expected_areas, assembled_areas, emitted_areas = set(), set(), set()
    errors_sample = []
    geom_factory = osmium.geom.WKBFactory()
    paths = {'roads': stage/'roads.jsonl', 'administrative_areas': stage/'administrative_areas.jsonl',
             'errors': stage/'errors.jsonl'}
    try:
        with paths['roads'].open('w', encoding='utf-8', newline='\n') as roads_file, \
             paths['administrative_areas'].open('w', encoding='utf-8', newline='\n') as areas_file, \
             paths['errors'].open('w', encoding='utf-8', newline='\n') as errors_file:
            def failure(osm_type, osm_id, reason, detail=None):
                counts['errors'] += 1
                error_counts[reason] += 1
                item = {'osm_type': str(osm_type), 'osm_id': int(osm_id), 'reason': reason}
                if detail:
                    item['detail'] = str(detail)[:300]
                emit_jsonl(errors_file, item)
                if len(errors_sample) < 100:
                    errors_sample.append(item)

            # Do not restrict source entity types: boundary members and highway
            # nodes must reach location/area processors before final tag filters.
            processor = (osmium.FileProcessor(str(pbf), thread_pool=pool)
                         .with_locations('flex_mem')
                         .with_areas(osmium.filter.TagFilter(('boundary', 'administrative')))
                         .with_filter(osmium.filter.KeyFilter('highway', 'boundary')))
            progress_at = time.perf_counter()
            for obj in processor:
                counts['tag_filtered_objects'] += 1
                if obj.is_way():
                    osm_id = int(obj.id)
                    boundary = obj.tags.get('boundary') == 'administrative'
                    if boundary:
                        counts['administrative_source_ways'] += 1
                        if obj.is_closed():
                            expected_areas.add(('way', osm_id))
                        else:
                            counts['administrative_open_ways_not_standalone_areas'] += 1
                    highway = obj.tags.get('highway')
                    if highway is None:
                        continue
                    counts['highway_ways'] += 1
                    props = names(obj.tags)
                    if not any(value and value.strip() for value in props.values()):
                        counts['unnamed_highway_ways_excluded'] += 1
                        continue
                    counts['named_highway_candidates'] += 1
                    props['highway'] = str(highway)
                    # Never connect across a missing node: reject the entire way.
                    missing_nodes = [int(node.ref) for node in obj.nodes if not node.location.valid()]
                    if missing_nodes:
                        counts['named_highway_missing_node_references'] += len(missing_nodes)
                        failure('way', osm_id, 'missing_node_location', {'node_ids': missing_nodes[:20], 'total': len(missing_nodes)})
                        continue
                    try:
                        geometry = shapely.from_wkb(geom_factory.create_linestring(obj))
                    except (RuntimeError, ValueError, shapely.errors.GEOSException) as exc:
                        failure('way', osm_id, 'line_geometry_exception', exc)
                        continue
                    reason = geometry_failure(geometry, {'LineString'})
                    if reason:
                        failure('way', osm_id, reason)
                        continue
                    emit_jsonl(roads_file, feature('way', osm_id, props, geometry, shapely))
                    counts['roads_written'] += 1
                    highway_counts[props['highway']] += 1
                elif obj.is_relation() and obj.tags.get('boundary') == 'administrative':
                    counts['administrative_source_relations'] += 1
                    if obj.tags.get('type') in ('boundary', 'multipolygon'):
                        expected_areas.add(('relation', int(obj.id)))
                    else:
                        counts['administrative_relations_without_supported_area_type'] += 1
                elif obj.is_area() and obj.tags.get('boundary') == 'administrative':
                    osm_type, osm_id = ('way' if obj.from_way() else 'relation'), int(obj.orig_id())
                    key = (osm_type, osm_id)
                    counts['administrative_areas_assembled'] += 1
                    assembled_areas.add(key)
                    props = names(obj.tags)
                    props['boundary'] = 'administrative'
                    admin = obj.tags.get('admin_level')
                    props['admin_level'] = str(admin) if admin is not None else None
                    try:
                        geometry = shapely.from_wkb(geom_factory.create_multipolygon(obj))
                    except (RuntimeError, ValueError, shapely.errors.GEOSException) as exc:
                        failure(osm_type, osm_id, 'area_geometry_exception', exc)
                        continue
                    reason = geometry_failure(geometry, {'Polygon', 'MultiPolygon'})
                    if reason:
                        failure(osm_type, osm_id, reason)
                        continue
                    if key in emitted_areas:
                        failure(osm_type, osm_id, 'duplicate_administrative_area')
                        continue
                    emit_jsonl(areas_file, feature(osm_type, osm_id, props, geometry, shapely))
                    counts['administrative_areas_written'] += 1
                    admin_counts[props['admin_level'] if props['admin_level'] is not None else '__MISSING__'] += 1
                    emitted_areas.add(key)
                if time.perf_counter() - progress_at >= 20:
                    print(json.dumps({'event': 'progress', 'elapsed_seconds': round(time.perf_counter()-started, 1),
                                      'counts': dict(counts)}, ensure_ascii=True), flush=True)
                    progress_at = time.perf_counter()
            for osm_type, osm_id in sorted(expected_areas - assembled_areas):
                failure(osm_type, osm_id, 'administrative_area_not_assembled',
                        'Candidate closed way or boundary/multipolygon relation did not yield an area; member geometry may be incomplete or invalid.')
        require(sha256(pbf) == initial_sha and pbf.stat().st_size == initial_bytes, 'Source PBF changed during extraction.')
        audit = {'format_version': 1, 'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': {'path': str(pbf), 'bytes': initial_bytes, 'sha256': initial_sha, 'header': header},
            'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
            'versions': {'python': platform.python_version(), 'osmium': version('osmium'),
                         'shapely': version('shapely'), 'geos': str(shapely.geos_version_string)},
            'crs': CRS, 'coordinate_order': ['longitude', 'latitude'], 'geometry_format': 'GeoJSON',
            'location_cache': 'flex_mem', 'io_threads': 4,
            'name_keys': list(NAME_KEYS), 'name_values_unnormalized': True,
            'administrative_levels_filtered': False,
            'selection': {'roads': 'ways with highway and at least one nonblank allowed name tag',
                          'administrative_areas': 'valid area geometry with boundary=administrative, every admin_level including missing'},
            'allowed_output_tags': {'roads': list(NAME_KEYS)+['highway'],
                                    'administrative_areas': list(NAME_KEYS)+['boundary', 'admin_level']},
            'counts': dict(sorted(counts.items())), 'highway_counts': dict(sorted(highway_counts.items())),
            'admin_level_counts': dict(sorted(admin_counts.items())),
            'error_counts': dict(sorted(error_counts.items())), 'error_samples_first_100': errors_sample,
            'candidate_administrative_areas': len(expected_areas),
            'unassembled_candidate_administrative_areas': len(expected_areas-assembled_areas),
            'geometry_policy': 'Keep complete original lines/polygons and holes; skip missing/invalid geometry, never repair, interpolate missing nodes, or assign a road to one district.',
            'spatial_index_notes': 'A road may intersect several administrative polygons. Clip with the actual geometry; a bbox hit is only a candidate. Distances and lengths in these coordinates are degrees, not metres.',
            'historical_limitations': 'Replication time is a map snapshot; OSM mapping history is not a verified road opening or administrative legal history. Missing names/geometry do not establish absence.',
            'license': {'name': 'Open Database License 1.0', 'attribution': 'OpenStreetMap contributors',
                        'url': 'https://www.openstreetmap.org/copyright'},
            'official_documentation': DOCS, 'external_price_or_transaction_data_used': False,
            'ml_data_or_predictions_read': False,
            'files': {key: {'path': str(output/path.name), 'sha256': sha256(path), 'bytes': path.stat().st_size}
                      for key, path in paths.items()},
            'elapsed_seconds': time.perf_counter()-started}
        write_json(stage/'audit.json', audit)
        (stage/'audit.sha256').write_text(sha256(stage/'audit.json')+'\n', encoding='ascii')
        require(not output.exists(), 'Output directory appeared during extraction; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        print('Extraction incomplete; staging retained: '+str(stage), file=sys.stderr, flush=True)
        raise
    print(json.dumps({'event': 'complete', 'output': str(output), 'counts': audit['counts'],
                      'admin_level_counts': audit['admin_level_counts'],
                      'audit_sha256': sha256(output/'audit.json')}, ensure_ascii=True), flush=True)
    return audit


def self_check(output):
    """Small integration fixture: crossing districts, a missing node, aliases, and excluded tags."""
    osmium, shapely = dependencies()
    output = Path(output).resolve()
    require(not output.exists(), 'Refusing to overwrite an existing synthetic directory.')
    output.mkdir(parents=True)
    xml_path, pbf_path = output/'fixture.osm', output/'fixture.osm.pbf'
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="geography-self-check">
<node id="1" lat="23" lon="120"/><node id="2" lat="23" lon="121"/>
<node id="3" lat="24" lon="121"/><node id="4" lat="24" lon="120"/>
<node id="5" lat="23" lon="122"/><node id="6" lat="24" lon="122"/>
<node id="7" lat="23.5" lon="120.25"/><node id="8" lat="23.5" lon="121.75"/>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
<way id="11"><nd ref="2"/><nd ref="5"/><nd ref="6"/><nd ref="3"/><nd ref="2"/></way>
<way id="20"><nd ref="7"/><nd ref="8"/><tag k="highway" v="residential"/>
<tag k="name" v="Crossing Road"/><tag k="name:zh" v="跨界路"/>
<tag k="name:zh-Hant" v="跨界路"/><tag k="alt_name" v="Alias One;Alias Two"/>
<tag k="short_name" v="Crossing"/><tag k="shop" v="must_not_export"/></way>
<way id="21"><nd ref="7"/><nd ref="9999"/><nd ref="8"/>
<tag k="highway" v="residential"/><tag k="name" v="Missing Node Road"/></way>
<way id="22"><nd ref="7"/><nd ref="8"/><tag k="highway" v="service"/></way>
<way id="23"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="building" v="yes"/><tag k="name" v="Excluded Building"/></way>
<relation id="100"><member type="way" ref="10" role="outer"/>
<tag k="type" v="boundary"/><tag k="boundary" v="administrative"/>
<tag k="admin_level" v="8"/><tag k="name" v="West District"/></relation>
<relation id="101"><member type="way" ref="11" role="outer"/>
<tag k="type" v="boundary"/><tag k="boundary" v="administrative"/>
<tag k="admin_level" v="7"/><tag k="name" v="East District"/></relation>
</osm>'''
    xml_path.write_text(xml, encoding='utf-8')
    header = osmium.io.Header()
    header.set('osmosis_replication_timestamp', '2022-01-01T00:00:00Z')
    pool = osmium.io.ThreadPool(2, 4)
    with osmium.io.Reader(str(xml_path), thread_pool=pool) as reader, \
         osmium.SimpleWriter(str(pbf_path), header=header, thread_pool=pool) as writer:
        osmium.apply(reader, writer)
    result = extract(pbf_path, output/'extracted')
    roads = [json.loads(line) for line in (output/'extracted/roads.jsonl').read_text(encoding='utf-8').splitlines()]
    areas = [json.loads(line) for line in (output/'extracted/administrative_areas.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(roads) == 1 and roads[0]['osm_id'] == 20, 'Named complete highway selection failed.')
    require(len(areas) == 2 and {x['properties']['admin_level'] for x in areas} == {'7','8'}, 'All administrative levels must remain available.')
    require(result['error_counts'].get('missing_node_location') == 1, 'Missing node must reject the whole way, not bridge its gap.')
    require(result['source']['header']['replication_timestamp'] == '2022-01-01T00:00:00Z', 'Actual PBF header timestamp failed.')
    require(roads[0]['properties']['alt_name'] == 'Alias One;Alias Two' and roads[0]['properties']['name:zh-Hant'] == '跨界路', 'Complete name values were altered.')
    require(set(roads[0]['properties']) == set(NAME_KEYS)|{'highway'}, 'Disallowed tags were exported.')
    line = shapely.from_geojson(json.dumps(roads[0]['geometry']))
    polygons = [shapely.from_geojson(json.dumps(item['geometry'])) for item in areas]
    clipped = [line.intersection(polygon) for polygon in polygons]
    require(all(part.geom_type == 'LineString' and math.isclose(part.length, .75, abs_tol=1e-12) for part in clipped), 'Cross-boundary road geometry was lost or assigned to only one district.')
    require(math.isclose(sum(part.length for part in clipped), line.length, abs_tol=1e-12), 'Clipping did not preserve the line.')
    audit = {'status': 'passed', 'synthetic_only': True, 'source_header_roundtrip': True,
             'crossing_road_clips_to_both_districts': True, 'missing_node_rejected_without_interpolation': True,
             'all_name_fields_preserved': True, 'non_geographic_tags_excluded': True,
             'every_admin_level_preserved': True, 'osm_view_not_used_after_iteration': True,
             'script_sha256': sha256(__file__), 'extraction_audit_sha256': sha256(output/'extracted/audit.json')}
    write_json(output/'self_check.json', audit)
    print(json.dumps(audit, ensure_ascii=True), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pbf', type=Path, help='Existing local PBF; never downloaded by this script.')
    group.add_argument('--self-check', action='store_true', help='Generate and validate a tiny synthetic PBF.')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are rejected.')
    args = parser.parse_args()
    if args.self_check:
        self_check(args.output)
    else:
        extract(args.pbf, args.output)


if __name__ == '__main__':
    main()