"""Render the native Word-exported PDF through the canonical DOCX renderer.

Windows has no bundled LibreOffice. Export DOCX with Word 16 first; this
adapter reuses render_docx.py pagination/DPI/PNG workflow with that native PDF.
"""
from pathlib import Path
import importlib.util, os, sys
SOURCE=Path(__file__).resolve().parent
OUT=SOURCE.parent
iteration=sys.argv[1] if len(sys.argv)>1 else 'r1'
skill=Path('C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/documents/26.1007.11041/skills/documents/render_docx.py')
pdf=OUT/'qa'/f'word_{iteration}.pdf'
assert pdf.exists()
spec=importlib.util.spec_from_file_location('docx_renderer',skill)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.convert_to_pdf=lambda *a,**kw:(str(pdf),'Microsoft Word 16 native PDF export; Windows bundled LibreOffice unavailable')
os.environ['PATH']='C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'+os.pathsep+os.environ['PATH']
sys.argv=[str(skill),str(OUT/'台灣房屋銷售價格預測挑戰賽_正式報告.docx'),'--output_dir',str(OUT/'qa'/f'render_{iteration}'),'--dpi','120']
mod.main()
