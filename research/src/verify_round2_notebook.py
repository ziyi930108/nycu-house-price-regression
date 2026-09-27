"""Execute the round-two notebook and verify every released submission is unchanged."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from prepare_data import ROOT, REPORT_DIR, sha256


def release_submissions(config: dict) -> dict[str, Path]:
    """Use the same variant naming contract as predict_round2.py."""
    variants = config.get('variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError('Release configuration must contain a nonempty variants list')
    paths = {}
    for variant in variants:
        name = variant.get('name') if isinstance(variant, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError(f'Invalid release variant name: {name!r}')
        if name in paths:
            raise ValueError(f'Duplicate release variant: {name}')
        paths[name] = ROOT / 'outputs' / f'submission_{name}.csv'
    return paths


def main() -> None:
    notebook_path = ROOT / 'solution_round2.ipynb'
    config_path = ROOT / 'models' / 'round2' / 'release_config.json'
    kernel_python = ROOT / '.venv' / 'Scripts' / 'python.exe'
    if not kernel_python.is_file():
        raise FileNotFoundError(f'Project virtual-environment Python is missing: {kernel_python}')

    config_before = sha256(config_path)
    submissions = release_submissions(json.loads(config_path.read_text(encoding='utf-8')))
    before = {name: sha256(path) for name, path in submissions.items()}
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    # Stale outputs must not make a skipped cell appear executed in this run.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []

    runtime_dir = ROOT / 'outputs' / 'round2' / 'notebook_runtime'
    ipython_dir = runtime_dir / 'ipython'
    ipython_dir.mkdir(parents=True, exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime_dir)
    os.environ['IPYTHONDIR'] = str(ipython_dir)
    os.environ['PYTHONUTF8'] = '1'
    manager = KernelManager(
        kernel_name='python3',
        connection_file=str(runtime_dir / f'kernel-round2-{uuid.uuid4().hex}.json'),
    )
    manager.kernel_spec.argv = [
        str(kernel_python), '-m', 'ipykernel_launcher', '-f', '{connection_file}',
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=1200, allow_errors=False,
        resources={'metadata': {'path': str(ROOT)}},
    )
    started = time.perf_counter()
    execution_error = None
    cleanup_errors = []
    try:
        client.execute()
    except BaseException as exc:
        execution_error = exc
    finally:
        # These cleanup steps are independent: channels must stop even if shutdown fails.
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        except Exception as exc:
            cleanup_errors.append(f'Kernel shutdown: {type(exc).__name__}: {exc}')
        finally:
            try:
                if client.kc is not None:
                    client.kc.stop_channels()
            except Exception as exc:
                cleanup_errors.append(f'Channel shutdown: {type(exc).__name__}: {exc}')

    executed = client.nb
    code_cells = [cell for cell in executed.cells if cell.cell_type == 'code']
    executable_cells = [cell for cell in code_cells if cell.source.strip()]
    executed_count = sum(cell.execution_count is not None for cell in executable_cells)
    cell_errors = [
        output for cell in code_cells for output in cell.get('outputs', [])
        if output.output_type == 'error'
    ]
    all_cells_passed = (
        execution_error is None and not cell_errors
        and executed_count == len(executable_cells)
    )
    after = {
        name: sha256(path) if path.is_file() else None
        for name, path in submissions.items()
    }
    csv_checks = {
        name: {
            'path': path.relative_to(ROOT).as_posix(),
            'before_sha256': before[name],
            'after_sha256': after[name],
            'unchanged': before[name] == after[name],
        }
        for name, path in submissions.items()
    }
    submissions_unchanged = all(item['unchanged'] for item in csv_checks.values())
    config_after = sha256(config_path) if config_path.is_file() else None
    config_unchanged = config_before == config_after
    nbformat.validate(executed)
    nbformat.write(executed, notebook_path)

    result = {
        'notebook_path': notebook_path.relative_to(ROOT).as_posix(),
        'notebook_sha256': sha256(notebook_path),
        'executed_code_cells': executed_count,
        'expected_executable_code_cells': len(executable_cells),
        'empty_code_cells': len(code_cells) - len(executable_cells),
        'all_cells_passed': all_cells_passed,
        'all_submissions_reproduced_exactly': submissions_unchanged,
        'csv_sha256': csv_checks,
        'submission_sha256': after,
        'release_config_before_sha256': config_before,
        'release_config_after_sha256': config_after,
        'release_config_unchanged': config_unchanged,
        'elapsed_seconds': time.perf_counter() - started,
        'kernel_name': 'python3',
        'kernel_python': str(kernel_python),
        'runtime_directory': runtime_dir.relative_to(ROOT).as_posix(),
        'cleanup_errors': cleanup_errors,
    }
    if execution_error is not None:
        result['execution_error'] = f'{type(execution_error).__name__}: {execution_error}'
    result['verification_passed'] = (
        all_cells_passed and submissions_unchanged and config_unchanged
        and not cleanup_errors
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / 'round2_notebook_checks.json'
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if execution_error is not None:
        raise execution_error
    if not result['verification_passed']:
        raise AssertionError(f'Round-two notebook verification failed; inspect {report_path}')


if __name__ == '__main__':
    main()