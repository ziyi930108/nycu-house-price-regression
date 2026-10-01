from pathlib import Path
import subprocess,sys,json,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NODE='C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
steps=[]
for style in ('B','C'):
 for stage,args in [('edit',[NODE,str(HERE/'build/edit_notes.mjs'),style]),('preserve',[sys.executable,'-X','utf8',str(HERE/'preserve_package.py'),style]),('finalize',[NODE,str(HERE/'build/finalize_notes.mjs'),style,'preserved'])]:
  t=time.perf_counter();p=subprocess.run(args,cwd=ROOT,capture_output=True);duration=time.perf_counter()-t
  log=HERE/'build'/style;log.mkdir(exist_ok=True)
  (log/(stage+'.stdout.txt')).write_bytes(p.stdout);(log/(stage+'.stderr.txt')).write_bytes(p.stderr)
  steps.append({'style':style,'stage':stage,'command':args,'seconds':duration,'exit_code':p.returncode})
  with (HERE/'build/bc_execution.json').open('w',encoding='utf-8') as f:json.dump(steps,f,ensure_ascii=False,indent=2)
  print(style,stage,p.returncode,round(duration,3),flush=True)
  if p.returncode:print(p.stderr.decode('utf-8',errors='replace'));sys.exit(p.returncode)
