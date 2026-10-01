from pathlib import Path
import subprocess,time,json,sys
HERE=Path(__file__).resolve().parent
NODE='C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
results=[];start=time.perf_counter()
for key in ('A','B','C'):
 for label,cmd in [('edit',[NODE,str(HERE/'build/edit_notes.mjs'),key]),('preserve',[sys.executable,'-B',str(HERE/'preserve_package.py'),key]),('finalize',[NODE,str(HERE/'build/finalize_notes.mjs'),key,'preserved'])]:
  step=time.perf_counter();p=subprocess.run(cmd,capture_output=True)
  (HERE/f'{key}_{label}.stdout.log').write_bytes(p.stdout);(HERE/f'{key}_{label}.stderr.log').write_bytes(p.stderr)
  results.append({'style':key,'step':label,'exit_code':p.returncode,'elapsed_seconds':time.perf_counter()-step})
  (HERE/'execution.json').write_text(json.dumps({'steps':results,'elapsed_seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
  if p.returncode:print(p.stderr.decode('utf-8',errors='replace')[-3000:]);sys.exit(p.returncode)
print(json.dumps({'steps':results,'elapsed_seconds':time.perf_counter()-start}))
