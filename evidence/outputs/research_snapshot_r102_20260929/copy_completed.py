"""Copy completed R101/R102 code and aggregate evidence; never train or publish."""
from pathlib import Path
import json,hashlib,re,urllib.parse
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'deliverables/v22_github'
HERE=Path(__file__).resolve().parent
DEST=REPO/'evidence/outputs/research_snapshot_r102_20260929'
NAV=['README.md','HANDOFF.md','research/README.md','docs/RESEARCH_UPDATES_20260929_PRIOR_INPUT.md']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,r):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
def inspect(x):
 if isinstance(x,dict):
  assert not {'global_median','reference_global_median','full_fit_global_median','cells'}.intersection(x),'Raw prior state forbidden'
  for v in x.values():inspect(v)
 elif isinstance(x,list):
  assert not (len(x)>100 and all(isinstance(v,(int,float)) for v in x)),'Numeric row vector forbidden'
  for v in x:inspect(v)
def links(p):return [s.strip('<>') for s in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8'))]
def main():
 assert not DEST.exists(),'Preserve snapshot receipt'
 for s in ['outputs/round101/execution.json','outputs/round102/execution.json','outputs/round102/evaluation/execution.json']:
  assert read(ROOT/s)['exit_code']==0,s
 baseline=read(REPO/'SOURCE_MANIFEST.json')
 for e in baseline['files']:assert sha(REPO/e['path'])==e['sha256'],e['path']
 paths=set(HERE.glob("*.py"))
 for r in (101,102):
  for pat in [f'round{r}_*.py',f'summarize_round{r}.py']:paths.update((ROOT/'src').glob(pat))
  paths.update((ROOT/'reports').glob(f'round{r}_*'))
  for folder in [f'round{r}',f'round{r}_launcher_20260929',f'round{r}_driver_prefreeze',f'round{r}_evaluator_prefreeze',f'round{r}_summary',f'round{r}_results_review']:
   paths.update((ROOT/'outputs'/folder).rglob('*'))
 paths=sorted(p for p in paths if p.is_file() and p.suffix in {'.md','.py','.json','.sha256','.patch'} and '__pycache__' not in p.parts)
 for p in paths:
  assert not {'prefix_states','states','private','data'}.intersection(p.relative_to(ROOT).parts)
  if p.suffix=='.json':inspect(read(p))
 copied=[]
 DEST.mkdir(parents=True)
 save(DEST/'baseline.json',{'files':baseline['files'],'manifest_sha256':sha(REPO/'SOURCE_MANIFEST.json')})
 for n in NAV[:3]:
  q=DEST/'navigation_before'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((REPO/n).read_bytes())
 for p in paths:
  name=p.relative_to(ROOT).as_posix();rel=('research/' if name.startswith('src/') else 'evidence/')+name;q=REPO/rel
  assert not q.exists(),rel
  q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());assert sha(q)==sha(p)
  copied.append({'source':name,'destination':rel,'sha256':sha(p),'bytes':p.stat().st_size})
 save(DEST/'copy_manifest.json',{'files':copied,'copied_count':len(copied),'copied_bytes':sum(x['bytes'] for x in copied),'excluded':['raw data','labels','arrays','models','price prior states'],'new_model_fits':0})
 print(json.dumps({'copied_count':len(copied),'copied_bytes':sum(x['bytes'] for x in copied)}))
if __name__=='__main__':main()
