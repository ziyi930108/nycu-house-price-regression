"""Copy only hash-verified required private files from the original workspace."""
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,required=True)
    ap.add_argument("--destination",type=Path,default=ROOT/"private/artifacts")
    a=ap.parse_args()
    entries=json.loads((ROOT/"artifacts/manifest.json").read_text(encoding="utf8"))["files"]
    source=a.workspace.resolve();dest=a.destination.resolve()
    if dest == source: raise ValueError("Use a separate destination")
    for e in entries:
        rel=Path(e["path"]); src=(source/rel).resolve(); out=(dest/rel).resolve()
        if not src.is_relative_to(source) or not out.is_relative_to(dest):
            raise ValueError("Manifest path escapes root")
        if digest(src)!=e["sha256"]:raise ValueError(f"Source hash differs: {rel}")
        if out.exists():
            if digest(out)!=e["sha256"]:raise FileExistsError(f"Existing artifact differs: {out}")
        else:
            out.parent.mkdir(parents=True,exist_ok=True)
            with src.open("rb") as f,out.open("xb") as g:shutil.copyfileobj(f,g)
            if digest(out)!=e["sha256"]:raise RuntimeError(f"Copy verification failed: {out}")
    print(json.dumps({"verified_files":len(entries),"destination":str(dest),"remote_upload":False}))
if __name__=="__main__":main()
