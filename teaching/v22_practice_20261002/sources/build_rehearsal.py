"""Rebuild the frozen 20-page rehearsal HTML, without model or network dependencies."""
import argparse,hashlib,json,os,re
from pathlib import Path
PIN='b907f4e71a1f4f912f951f7c7780366e3b2aee9268aca44586611a3144f944b6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('narrative','template','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if os.path.lexists(a.output):p.error('Output exists; use a new file.')
    if sha(a.narrative)!=PIN:raise ValueError('Frozen narrative SHA mismatch.')
    n=json.loads(a.narrative.read_text(encoding='utf8'));slides=n['slides'][:20]
    if len(n['slides'])!=23 or [s['number'] for s in slides]!=list(range(1,21)):raise ValueError('Expected 20 main slides.')
    if any(s['is_appendix'] for s in slides):raise ValueError('Appendix in main speech.')
    if sum(len(re.findall('[\u4e00-\u9fff]',s['script'])) for s in slides)!=4065:raise ValueError('Main script changed.')
    if sum(s['seconds'] for s in slides)!=885:raise ValueError('Planned timing changed.')
    template=a.template.read_text(encoding='utf8')
    if template.count('__DECK_JSON__')!=1:raise ValueError('Expected one data placeholder.')
    source_before={str(a.narrative.resolve()):sha(a.narrative),str(a.template.resolve()):sha(a.template)}
    payload={'source_sha256':PIN,'slides':[{k:s[k] for k in ('number','title','takeaway','script','seconds')} for s in slides]}
    text=template.replace('__DECK_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    # The reviewed Windows artifact uses CRLF; preserve those bytes on every host.
    raw=text.replace('\r\n','\n').replace('\n','\r\n').encode('utf8')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(raw)
    if any(sha(Path(k))!=v for k,v in source_before.items()):raise ValueError('Source changed during build.')
    print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'source_sha256':source_before,'main_han':4065,'planned_seconds':885,'new_model_fits':0},ensure_ascii=False))
if __name__=='__main__':main()
