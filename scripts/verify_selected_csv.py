"""Check the selected submission's exact bytes and format; no model or Kaggle access."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json


def check(csv_path: Path, selection_path: Path, sample_path: Path | None = None):
    selection = json.loads(selection_path.read_text(encoding='utf-8-sig'))
    payload = csv_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != selection['csv_sha256']:
        raise ValueError('CSV bytes differ from the selected submission; no output was changed.')
    rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig'), newline='')))
    if not rows or rows[0] != selection['columns'] or len(rows) - 1 != selection['rows']:
        raise ValueError('Header or row count differs from the selected submission.')
    identifiers = []
    for row in rows[1:]:
        if len(row) != 2 or not row[0] or not row[1].isascii() or not row[1].isdecimal():
            raise ValueError('Expected a nonempty ID and a positive integer price.')
        if not 0 < int(row[1]) <= 2**53 - 1:
            raise ValueError('Price is outside the selected CSV integer contract.')
        identifiers.append(row[0])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate submission IDs.')
    if sample_path is not None:
        with sample_path.open(encoding='utf-8-sig', newline='') as f:
            sample = list(csv.reader(f))
        if not sample or sample[0] != selection['columns']:
            raise ValueError('The supplied sample header differs.')
        if any(len(row) != 2 for row in sample[1:]) or [row[0] for row in sample[1:]] != identifiers:
            raise ValueError('IDs or order differ from the supplied sample.')
    return {'status': 'passed', 'selected_submission': selection['selected_submission'],
            'rows': len(identifiers), 'csv_sha256': digest,
            'sample_ids_and_order_checked': sample_path is not None,
            'public_score_source': 'Human report in the selection evidence, not calculated by this tool',
            'private_score_calculated': False, 'model_inference': False, 'Kaggle_actions': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--sample', type=Path)
    args = parser.parse_args()
    try:
        result = check(args.csv, args.selection, args.sample)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f'Submission check failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
