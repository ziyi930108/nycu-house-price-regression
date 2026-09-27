"""Verify the release snapshot without model weights, course data, or ML dependencies."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def main():
    manifest=json.loads((ROOT/"SOURCE_MANIFEST.json").read_text(encoding="utf8"))
    for e in manifest["files"]:
        p=(ROOT/e["path"]).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=e["sha256"]:
            raise ValueError("Release file missing or changed: "+e["path"])
    historical=json.loads((ROOT/"research/source_manifest.json").read_text(encoding="utf8"))
    for e in historical:
        p=(ROOT/e["copy"]).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=e["sha256"]:
            raise ValueError("Historical source bytes changed: "+e["copy"])
    if (ROOT/".git").exists():
        tracked=subprocess.check_output(["git","-C",str(ROOT),"ls-files","-z"]).decode("utf8").split("\0")
        expected={e["path"] for e in manifest["files"]}|{"SOURCE_MANIFEST.json"}
        unknown=set(filter(None,tracked))-expected
        if unknown:raise ValueError("Tracked files missing from source manifest: "+str(sorted(unknown)))
        for rel in filter(None,tracked):
            p=Path(rel)
            if p.parts[0] in {"private","data","runs"} or p.suffix.lower() in {".xlsx",".csv",".npy",".npz",".pkl",".cbm",".pt"}:
                raise ValueError("Private input/output must not be tracked: "+rel)
    print(json.dumps({"status":"passed","snapshot_files":len(manifest["files"]),
      "historical_sources":len(historical),"models_loaded":0,"new_fits":0}))
if __name__=="__main__":main()
