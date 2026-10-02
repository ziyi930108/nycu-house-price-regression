"""Read-only package/interpreter facts for a new venv; no model or GPU computation."""
import importlib.metadata as md
import importlib.util
import json,os,platform,site,sys
from pathlib import Path
packages=['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','openpyxl','torch','pip']
result={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
        'isolated':sys.flags.isolated,'user_site_enabled':site.ENABLE_USER_SITE,'sys_path':sys.path,
        'platform':platform.platform(),'packages':{},'model_files_read':False,'training':False}
for name in packages:
    try:
        d=md.distribution(name)
        result['packages'][name]={'version':d.version,'metadata_directory':str(d._path)}
    except md.PackageNotFoundError:result['packages'][name]=None
print(json.dumps(result,ensure_ascii=False,indent=2))
