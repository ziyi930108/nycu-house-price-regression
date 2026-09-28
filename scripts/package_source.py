"""Package only public code/report files; never include private artifacts or input data."""
from pathlib import Path
import argparse, hashlib, json, zipfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_DIRS={"src","scripts","tests","config","artifacts","research","evidence","docs","presentation",".github"}
ALLOWED_TOP={".gitignore",".gitattributes","AGENTS.md","HANDOFF.md","README.md","REPORT.md","DATA.md","requirements-inference.txt",
             "requirements-torch.txt","requirements-report.txt"}
SUFFIXES={".py",".md",".json",".txt",".pdf",".sha256",".pptx",".mjs",".png",".jpg",".yml",".yaml",".toml",".patch"}
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def included():
    for p in sorted(ROOT.rglob("*")):
        rel=p.relative_to(ROOT)
        if not p.is_file() or "__pycache__" in rel.parts or p.is_symlink():continue
        if len(rel.parts)==1 and rel.name in ALLOWED_TOP:yield p
        elif rel.parts[0] in ALLOWED_DIRS and (p.suffix.lower() in SUFFIXES or p.name == "LICENSE"):yield p
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    paths=list(included())
    forbidden={".xlsx",".csv",".npy",".npz",".pkl",".cbm",".pt"}
    if any(p.suffix in forbidden for p in paths):raise RuntimeError("Private artifact in public package")
    entries=[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in paths]
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"scope":"code, formal report and presentation, no model/data bundle",
              "files":entries,"private_artifact_manifest":"artifacts/manifest.json",
              "private_artifacts_included":False,"raw_course_data_included":False,"publication_target":{"repository":"https://github.com/ziyi930108/nycu-house-price-regression","visibility":"private","status_source":"Git history; this is a local file snapshot"}}
    raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode("utf8")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,"x",zipfile.ZIP_DEFLATED) as z:
        z.writestr("v22_github/SOURCE_MANIFEST.json",raw)
        for p,e in zip(paths,entries):
            if sha(p)!=e["sha256"]:raise RuntimeError("File changed while packaging")
            z.write(p,"v22_github/"+e["path"])
    with zipfile.ZipFile(a.output) as z:
        if z.testzip() is not None:raise RuntimeError("ZIP CRC failure")
        for e in entries:
            if hashlib.sha256(z.read("v22_github/"+e["path"])).hexdigest()!=e["sha256"]:
                raise RuntimeError("ZIP member hash failure")
    (ROOT/"SOURCE_MANIFEST.json").write_bytes(raw)
    receipt={"archive":a.output.name,"bytes":a.output.stat().st_size,"sha256":sha(a.output),
             "files":len(entries)+1,"all_member_hashes_verified":True,"excluded":[".git","private","data","runs","__pycache__"]}
    a.output.with_suffix(".zip.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=="__main__":main()
