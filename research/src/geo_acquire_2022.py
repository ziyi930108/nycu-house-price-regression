from pathlib import Path
import json, hashlib, urllib.request, os
from datetime import datetime,timezone
ROOT=Path.cwd(); out=ROOT/'data_external/geography';out.mkdir(parents=True,exist_ok=True)
url='https://download.geofabrik.de/asia/taiwan-220101.osm.pbf'
expected=102877382
p=out/'taiwan-220101.osm.pbf'
meta=out/'taiwan-220101.download.json'
if p.exists():
    assert p.stat().st_size==expected and meta.exists()
    old=json.loads(meta.read_text(encoding='utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==old['sha256']
    print('existing download verified',flush=True)
else:
    part=p.with_suffix(p.suffix+'.part')
    assert not part.exists(), 'partial file present; inspect before retry'
    request=urllib.request.Request(url,headers={'User-Agent':'NYCU-local-geography-research/1.0'})
    digest=hashlib.sha256();n=0;last=0
    with urllib.request.urlopen(request,timeout=45) as response, part.open('xb') as f:
        assert response.geturl()==url and response.status==200
        headers=dict(response.headers.items())
        advertised=response.headers.get('Content-Length')
        if advertised is not None: assert int(advertised)==expected
        while chunk:=response.read(1024*1024):
            n+=len(chunk)
            assert n<=expected, 'source length exceeds recorded official listing'
            f.write(chunk);digest.update(chunk)
            if n-last>=20*1024*1024:
                print(f'downloaded {n}/{expected} bytes',flush=True);last=n
        assert n==expected,(n,expected)
    part.rename(p)
    payload={'url':url,'official_index':'https://download.geofabrik.de/asia/taiwan.html','official_listed_bytes':expected,'bytes':n,'sha256':digest.hexdigest(),'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),'http_headers':headers,'data_timestamp_verified':False,'purpose':'Historical OSM named roads and administrative boundaries; no price data','license':'ODbL-1.0','attribution':'OpenStreetMap contributors; extract by Geofabrik','license_url':'https://www.openstreetmap.org/copyright','source_file':p.relative_to(ROOT).as_posix()}
    meta.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'ATTRIBUTION.md').write_text('# 地理資料來源與授權\n\n© OpenStreetMap contributors，臺灣擷取檔由 Geofabrik 提供。\n\n[OSM 著作權／ODbL 授權](https://www.openstreetmap.org/copyright)；[Geofabrik 臺灣](https://download.geofabrik.de/asia/taiwan.html)。\n\n本地使用固定日期快照；每個檔案的 URL、bytes、SHA256、取得日期記在同名 download.json。不得把道路代表點描述成住宅的精確門牌座標。未取得內政部成交答案或外部房價。\n',encoding='utf-8')
    print(json.dumps({'status':'downloaded','bytes':n,'sha256':digest.hexdigest()}),flush=True)
