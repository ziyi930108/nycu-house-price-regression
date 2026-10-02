"""Observe the NN child launch mode; import/device query only, no tensor or model."""
import importlib.metadata as md
import json, os, site, sys
from pathlib import Path
import numpy as np
import torch

prefix=Path(sys.prefix).resolve()
assert prefix != Path(sys.base_prefix).resolve()
assert not site.ENABLE_USER_SITE and sys.flags.no_user_site==1
assert 'PYTHONPATH' not in os.environ and 'PYTHONHOME' not in os.environ
assert np.__version__=='2.3.5' and torch.__version__=='2.13.0+cu126'
assert torch.version.cuda=='12.6'
origins={}
for name, module in tuple(sys.modules.items()):
    source=getattr(module,'__file__',None)
    if source and 'site-packages' in [part.casefold() for part in Path(source).parts]:
        path=Path(source).resolve()
        assert path.is_relative_to(prefix),(name,path)
        origins[name]=str(path)
assert torch.cuda.is_available()
print(json.dumps({
    'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
    'sys_path':sys.path,'isolated_flag':sys.flags.isolated,'ignore_environment_flag':sys.flags.ignore_environment,
    'no_user_site_flag':sys.flags.no_user_site,'user_site_enabled':site.ENABLE_USER_SITE,
    'pythonpath_absent':True,'pythonhome_absent':True,
    'numpy':np.__version__,'torch':str(torch.__version__),'torch_metadata':md.version('torch'),
    'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
    'device_count':torch.cuda.device_count(),
    'third_party_module_origins':origins,'all_third_party_modules_inside_venv':True,
    'models_loaded':0,'new_fits':0,'gpu_tensor_computation':False,
    'scope':'Separate diagnostic under same interpreter flags/env as NN child; script directory differs from nn_predict.py',
},ensure_ascii=False,indent=2))
