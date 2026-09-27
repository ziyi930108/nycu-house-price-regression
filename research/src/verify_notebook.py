"""Execute every notebook cell and verify saved-model prediction reproducibility."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from prepare_data import ROOT, REPORT_DIR, sha256


def main():
    path = ROOT / 'solution.ipynb'
    submission = ROOT / 'outputs' / 'submission.csv'
    before = sha256(submission)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    runtime_dir = ROOT / 'outputs' / 'notebook_runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime_dir)
    os.environ['IPYTHONDIR'] = str(runtime_dir / 'ipython')
    os.environ['PYTHONUTF8'] = '1'
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False,
                            resources={'metadata': {'path': str(ROOT)}})
    start = time.perf_counter()
    try:
        executed = client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
    code_cells = [cell for cell in executed.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
    after = sha256(submission)
    if before != after:
        raise AssertionError('Notebook prediction changed the saved-model submission')
    nbformat.validate(executed)
    nbformat.write(executed, path)
    result = {'executed_code_cells': len(code_cells), 'all_cells_passed': True,
              'submission_reproduced_exactly': True, 'submission_sha256': after,
              'notebook_sha256': sha256(path),
              'seconds': time.perf_counter() - start, 'kernel_python': sys.executable}
    (REPORT_DIR / 'notebook_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()