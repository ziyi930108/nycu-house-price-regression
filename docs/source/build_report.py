"""Rebuild current report; no model training."""
from pathlib import Path
import subprocess, sys
r=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(r/'docs/source/course_completion/build_report.py'),'--output-dir',str(r/'docs'),'--markdown-path',str(r/'REPORT.md'),'--source-map-path',str(r/'docs/REPORT_SOURCES.md'),'--qa-dir',str(r/'build/report_course')],check=True,cwd=r)
