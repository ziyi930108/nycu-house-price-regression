"""Acquire only prespecified annual public Taiwan OSM snapshots.

Acquisition is pure geographic source preparation, not permission to extract,
match, train, or score confirmation data. Development acceptance remains a
separate gate. Existing files, including the 2022 snapshot, are never replaced.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data_external/geography'
SPECS = {
    2021: {'filename': 'taiwan-210101.osm.pbf', 'bytes': 84556154},
    2023: {'filename': 'taiwan-230101.osm.pbf', 'bytes': 114165920},
    2024: {'filename': 'taiwan-240101.osm.pbf', 'bytes': 242830461},
}
OFFICIAL_BASE = 'https://download.geofabrik.de/asia/'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def actual_header(path, year):
    import osmium
    with osmium.io.Reader(osmium.io.File(str(path), 'pbf'), osmium.osm.NOTHING) as reader:
        header = reader.header()
        stamp = str(header.get('osmosis_replication_timestamp', ''))
        multiple = bool(header.has_multiple_object_versions)
        options = {key: str(header.get(key, '')) for key in (
            'generator', 'osmosis_replication_timestamp',
            'osmosis_replication_sequence_number', 'osmosis_replication_base_url')}
    require(stamp and not multiple, 'PBF must be a single snapshot with a real replication timestamp')
    parsed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.year == year,
            'Actual PBF year differs from the prespecified annual source')
    require(parsed < datetime(year, 8, 10, 16, tzinfo=timezone.utc),
            'Actual PBF timestamp is later than the annual forecast origin')
    return {'replication_timestamp': stamp, 'header_key': 'osmosis_replication_timestamp',
            'has_multiple_object_versions': multiple, 'options': options,
            'interpretation': 'Actual map replication timestamp; not a road opening date or a house coordinate'}


def verify_existing(path, meta, year, url, expected):
    require(path.stat().st_size == expected, 'Existing PBF size differs; refusing replacement')
    require(meta.is_file(), 'Existing PBF has no download receipt; inspect without replacing it')
    record = json.loads(meta.read_text(encoding='utf-8'))
    digest = sha256(path)
    require(record['sha256'] == digest and record['bytes'] == expected and record['url'] == url,
            'Existing download receipt does not match; refusing replacement')
    receipt = meta.with_suffix('.sha256')
    if receipt.exists():
        require(receipt.read_text(encoding='ascii').split()[0] == sha256(meta),
                'Existing metadata SHA receipt differs')
    header = actual_header(path, year)
    if record.get('actual_pbf_header'):
        require(record['actual_pbf_header'] == header, 'Existing header differs from receipt')
    return {'status': 'existing_verified_read_only', 'year': year,
            'path': str(path), 'bytes': expected, 'sha256': digest,
            'source_sha256': sha256(__file__), 'actual_pbf_header': header,
            'download_receipt_sha256': sha256(meta), 'files_written': False}


def acquire(year, *, verify_only=False):
    require(year in SPECS, 'Only the fixed 2021, 2023 and 2024 sources are allowed')
    specification = SPECS[year]
    filename, expected = specification['filename'], specification['bytes']
    url = OFFICIAL_BASE + filename
    path = DIRECTORY/filename
    meta = DIRECTORY/filename.replace('.osm.pbf', '.download.json')
    receipt = meta.with_suffix('.sha256')
    if path.exists():
        result = verify_existing(path, meta, year, url, expected)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    require(not verify_only, 'Requested annual snapshot has not been downloaded')
    part = path.with_suffix(path.suffix + '.part')
    meta_part = meta.with_suffix(meta.suffix + '.part')
    require(not any(p.exists() for p in (part, meta, meta_part, receipt)),
            'Partial file or receipt already exists; inspect before retry; no overwrite')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    preserved = {str(p): sha256(p) for p in DIRECTORY.iterdir() if p.is_file()}
    source_sha = sha256(__file__)
    request = urllib.request.Request(url, headers={'User-Agent': 'NYCU-local-geography-research/1.0'})
    count, last = 0, 0
    stream_digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=45) as response, part.open('xb') as handle:
            require(response.geturl() == url and response.status == 200,
                    'Unexpected redirect or HTTP status from fixed official source')
            headers = dict(response.headers.items())
            advertised = response.headers.get('Content-Length')
            if advertised is not None:
                require(int(advertised) == expected, 'Advertised length differs from verified official listing')
            while chunk := response.read(1024*1024):
                count += len(chunk)
                require(count <= expected, 'Download exceeds verified official byte length')
                handle.write(chunk)
                stream_digest.update(chunk)
                if count-last >= 20*1024*1024:
                    print(json.dumps({'stage': 'downloading', 'year': year,
                                      'bytes': count, 'expected_bytes': expected}), flush=True)
                    last = count
            require(count == expected, f'Download incomplete: {count}/{expected} bytes')
        require(part.stat().st_size == expected and sha256(part) == stream_digest.hexdigest(),
                'Saved PBF differs from received bytes')
        header = actual_header(part, year)
        require(sha256(__file__) == source_sha, 'Downloader source changed during acquisition')
        for old_path, old_sha in preserved.items():
            require(sha256(old_path) == old_sha, 'A pre-existing geographic file changed')
        payload = {
            'url': url, 'official_index': 'https://download.geofabrik.de/asia/taiwan.html',
            'official_listed_bytes': expected, 'bytes': count, 'sha256': stream_digest.hexdigest(),
            'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'http_headers': headers, 'data_timestamp_verified': True,
            'actual_pbf_header': header, 'year': year,
            'purpose': 'Pre-stage a fixed public geographic snapshot; no transaction or price data',
            'development_gate_status_at_request': 'not_yet_passed',
            'acquisition_does_not_authorize_confirmation_processing': True,
            'extraction_performed': False, 'address_matching_performed': False,
            'model_training_or_scoring_performed': False, 'local_addresses_transmitted': False,
            'license': 'ODbL-1.0', 'attribution': 'OpenStreetMap contributors; extract by Geofabrik',
            'license_url': 'https://www.openstreetmap.org/copyright',
            'source_file': path.relative_to(ROOT).as_posix(),
            'downloader_source': {'path': Path(__file__).resolve().relative_to(ROOT).as_posix(),
                                  'sha256': source_sha},
            'preserved_existing_geography_sha256': preserved,
            'versions': {'osmium': version('osmium')},
            'hash_interpretation': 'SHA256 of the downloaded bytes, recorded for later integrity verification; official listing supplied byte length',
        }
        with meta_part.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        require(not path.exists() and not meta.exists() and not receipt.exists(),
                'Final output appeared during acquisition; refusing replacement')
        part.rename(path)
        meta_part.rename(meta)
        with receipt.open('x', encoding='ascii') as handle:
            handle.write(sha256(meta)+'\n')
    except BaseException:
        print(json.dumps({'status': 'incomplete_acquisition', 'year': year,
                          'partial_file_retained_if_present': str(part),
                          'automatic_retry_or_overwrite': False}), flush=True)
        raise
    result = {'status': 'downloaded', 'year': year, 'path': str(path), 'bytes': count,
              'sha256': payload['sha256'], 'source_sha256': source_sha,
              'actual_pbf_header': header, 'download_receipt_sha256': sha256(meta),
              'source_receipt': str(meta), 'model_training_or_scoring_performed': False}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=sorted(SPECS), required=True)
    parser.add_argument('--verify-only', action='store_true')
    arguments = parser.parse_args()
    acquire(arguments.year, verify_only=arguments.verify_only)