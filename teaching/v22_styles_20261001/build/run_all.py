from pathlib import Path
import subprocess,json,time

D=Path(__file__).resolve().parents[1]
NODE=Path('C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
receipt=D/'build/execution.json'
assert not receipt.exists(),'Preserve earlier execution evidence'
records=[]
for key in ['A','B','C']:
    for step in ['build_deck.mjs','finalize_deck.mjs']:
        p=D/'build'/key/step
        start=time.perf_counter()
        result=subprocess.run([str(NODE),str(p)],cwd=D.parents[1],capture_output=True)
        elapsed=time.perf_counter()-start
        (p.parent/(step+'.stdout.log')).write_bytes(result.stdout)
        (p.parent/(step+'.stderr.log')).write_bytes(result.stderr)
        records.append({'style':key,'step':step,'exit_code':result.returncode,'elapsed_seconds':elapsed})
        receipt.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(records[-1]),flush=True)
        if result.returncode:
            print(result.stderr.decode('utf-8',errors='replace')[-4000:],flush=True)
            raise SystemExit(result.returncode)
print('Three revised decks and 69 renders complete.',flush=True)
