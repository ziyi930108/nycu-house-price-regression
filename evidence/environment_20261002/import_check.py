"""Import only: verify isolated origins and pinned versions without loading models."""
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import site
import sys

assert sys.flags.isolated and not site.ENABLE_USER_SITE
prefix = Path(sys.prefix).resolve()
assert prefix != Path(sys.base_prefix).resolve()
expected = {
    'numpy': ('numpy', '2.3.5'), 'pandas': ('pandas', '3.0.1'),
    'scipy': ('scipy', '1.18.1'), 'scikit-learn': ('sklearn', '1.9.1'),
    'catboost': ('catboost', '1.2.10'), 'lightgbm': ('lightgbm', '4.7.0'),
    'openpyxl': ('openpyxl', '3.1.5'), 'torch': ('torch', '2.13.0'),
}
records = {}
for distribution, (module_name, version) in expected.items():
    module = importlib.import_module(module_name)
    installed = md.version(distribution)
    assert installed.split('+')[0] == version, (distribution, installed, version)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix), path
    metadata_path = Path(md.distribution(distribution)._path).resolve()
    assert metadata_path.is_relative_to(prefix), metadata_path
    records[distribution] = {'version': installed, 'module_version': getattr(module, '__version__', None),
                             'module_file': str(path), 'metadata_directory': str(metadata_path)}

from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from catboost import CatBoostRegressor, Pool
from torch import nn
import torch

# Imported dependencies must come from this environment, not inherited site packages.
origins = {}
for name, module in tuple(sys.modules.items()):
    source = getattr(module, '__file__', None)
    if source and 'site-packages' in [part.casefold() for part in Path(source).parts]:
        source_path = Path(source).resolve()
        assert source_path.is_relative_to(prefix), (name, source_path)
        origins[name] = str(source_path)

print(json.dumps({
    'executable': sys.executable, 'prefix': str(prefix),
    'direct_packages': records, 'third_party_module_count': len(origins),
    'third_party_module_origins': origins,
    'threadpoolctl_version': md.version('threadpoolctl'),
    'extra_imports': ['scipy.spatial.cKDTree', 'threadpoolctl.threadpool_limits',
                      'catboost.CatBoostRegressor', 'catboost.Pool', 'torch.nn'],
    'torch': {'version': torch.__version__, 'compiled_cuda': torch.version.cuda,
              'matches_original_build': torch.__version__ == '2.13.0+cu126'},
    'models_loaded': 0, 'new_fits': 0, 'inference_performed': False,
    'gpu_computation': False,
}, indent=2))
