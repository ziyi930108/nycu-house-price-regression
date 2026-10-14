"""Compatibility entry for the current report; never train a model."""
from pathlib import Path
import argparse, subprocess, sys
r=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,default=r/'build/report_current')
a=p.parse_args()
subprocess.run([sys.executable,'-B','-X','utf8',str(r/'report/source/build_report.py'),
                '--output-dir',str(a.output_dir.absolute())],check=True,cwd=r)
