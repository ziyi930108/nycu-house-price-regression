"""Read local path settings and run the preserved six-model predictor.

--describe checks source identities without loading models.
--check-inputs also checks paths, but does not deserialize or hash private data.
Relative paths in JSON are always relative to the repository root.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'inference/v25/source'
ENTRY = SOURCE / 'src/predict_v25.py'
EXPECTED_CLOSURE = 'a781bc2d3c08464c69e591d60ff7deb344aac946c55bdd405ce41da884f0dc30'
PATH_KEYS = ('artifacts', 'test', 'sample', 'nn_python', 'output_dir')
KEYS = set(PATH_KEYS) | {'device'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate setting: ' + key)
        result[key] = value
    return result


def load_settings(path):
    settings = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)
    if not isinstance(settings, dict) or set(settings) != KEYS:
        raise ValueError('Settings must contain exactly: ' + ', '.join(sorted(KEYS)))
    if any(not isinstance(v, str) or not v.strip() for v in settings.values()):
        raise ValueError('Every setting must be a nonempty string')
    if settings['device'] not in {'cuda:0', 'cpu'}:
        raise ValueError('device must be cuda:0 or cpu; only CUDA replay has been verified')
    for key in PATH_KEYS:
        value = Path(settings[key]).expanduser()
        settings[key] = (value if value.is_absolute() else ROOT / value).absolute()
    return settings


def describe_source():
    closure = SOURCE / 'closure.json'
    if hashlib.sha256(closure.read_bytes()).hexdigest() != EXPECTED_CLOSURE:
        raise ValueError('Preserved source closure identity differs')
    result = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8', str(ENTRY), '--describe'],
                            check=True, capture_output=True, text=True, encoding='utf8', cwd=ROOT)
    value = json.loads(result.stdout)
    if value.get('closure_sha256') != EXPECTED_CLOSURE or value.get('models_loaded') != 0:
        raise ValueError('Source-only description did not pass')
    return value


def check_inputs(settings):
    problems = []
    for key in ('test', 'sample', 'nn_python'):
        if not settings[key].is_file():
            problems.append(key + ': file not found')
    if not settings['artifacts'].is_dir():
        problems.append('artifacts: directory not found')
    else:
        manifest = json.loads((SOURCE / 'artifacts/manifest.json').read_text(encoding='utf8'))
        for entry in manifest['files']:
            if not (settings['artifacts'] / entry['path']).is_file():
                problems.append('artifacts: missing ' + entry['path'])
    output = settings['output_dir'].resolve()
    if output.exists():
        problems.append('output_dir already exists; choose a new directory')
    if output.is_relative_to(ROOT):
        problems.append('output_dir must be outside the public repository')
    if problems:
        raise ValueError('; '.join(problems))


def build_command(settings):
    command = [sys.executable, '-I', '-B', '-X', 'utf8', str(ENTRY)]
    for key in (*PATH_KEYS, 'device'):
        command += ['--' + key.replace('_', '-'), str(settings[key])]
    return command + ['--expected-closure-sha', EXPECTED_CLOSURE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/predict.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    try:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        settings = load_settings(config_path)
        source_info = describe_source()
        if args.describe:
            print(json.dumps({'mode': 'describe', 'settings': {k: str(v) for k, v in settings.items()},
                              'source': source_info, 'private_inputs_checked': False}, ensure_ascii=False, indent=2))
            return 0
        check_inputs(settings)
        if args.check_inputs:
            print(json.dumps({'status': 'paths_ready', 'models_loaded': 0, 'new_fits': 0,
                              'private_bytes_verified': False, 'inference_completed': False}))
            return 0
        return subprocess.run(build_command(settings), cwd=ROOT, check=False).returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, 'Error: ' + str(error) + '\n')


if __name__ == '__main__':
    sys.exit(main())
