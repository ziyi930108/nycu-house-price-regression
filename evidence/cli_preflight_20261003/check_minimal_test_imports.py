"""Run the documented lightweight tests with model-package imports forbidden.

Existing Python environment is used; this is not a fresh install or Actions run.
"""
from pathlib import Path
import importlib.abc
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT if (ROOT/'tests/test_prediction_preflight.py').is_file() else ROOT/'deliverables/v22_github'
class NoModelPackages(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'catboost','lightgbm','torch','sklearn','scipy','openpyxl'}:
            raise ModuleNotFoundError(f'Model package deliberately unavailable in lightweight test: {fullname}',name=fullname)
        return None

sys.meta_path.insert(0,NoModelPackages())
suite=unittest.defaultTestLoader.discover(str(REPO/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
