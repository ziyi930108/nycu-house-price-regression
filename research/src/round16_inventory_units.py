"""Bounded target-free inventory of the existing train/test feature caches.
No raw target cache, Excel, model, predictions, scores, fitting or parser edits.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, re, unicodedata, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEST_JSON=ROOT/'outputs/round16_research/units_inventory.json'
DEST_MD=ROOT/'reports/round16_unit_inventory.md'
SOURCES=('src/round16_inventory_units.py','src/house_features.py','src/round3_features.py','src/round4_augmented.py')
INPUTS={'train':'data_cache/train_features.pkl','test':'data_cache/test_features.pkl'}
MISSING='__MISSING__'
CN='零〇一二三四五六七八九十百千兩'
CODE_FLOOR=re.compile(r'^(?P<code>[A-Z]{1,3}[0-9]{0,3})-(?P<floor>B?[0-9]{1,3})[F樓層](?:-?(?P<unit>[A-Z]?[0-9]{1,4})戶?)?$')
UNIT_CODE=re.compile(r'^[A-Z]{1,3}[0-9]{0,4}戶$')
CODE_NUMBER=re.compile(r'^[A-Z]{1,3}[0-9]{0,3}-[0-9]{1,4}$')
BARE_CODE=re.compile(r'^[A-Z]{1,3}[0-9]{0,4}$')
FLOOR_MARKED=re.compile(r'(?:地下|地上|B)?[0-9'+CN+r']+[F樓層]')
CLASSES=('missing','code_dash_explicit_floor','explicit_block_marker','explicit_unit_code',
         'code_dash_number_ambiguous','bare_code_ambiguous','other_explicit_floor','other')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def normalize(value):
    if pd.isna(value):return MISSING
    s=unicodedata.normalize('NFKC',str(value)).strip().upper()
    s=re.sub(r'\s+','',s).replace('－','-').replace('—','-').replace('–','-')
    return MISSING if s in {'','NAN','NONE','NULL','<NA>',MISSING} else s


def describe(value):
    s=normalize(value)
    if s==MISSING:return ('missing',None,None,False)
    multi=bool(re.search(r'[,，、;；/／]|及|與',s))
    match=CODE_FLOOR.fullmatch(s)
    if match:
        token=match.group('floor')
        floor=-int(token[1:]) if token.startswith('B') else int(token)
        return ('code_dash_explicit_floor',match.group('code'),floor,multi)
    if '棟' in s or '幢' in s:return ('explicit_block_marker',None,None,multi)
    if UNIT_CODE.fullmatch(s):return ('explicit_unit_code',s[:-1],None,multi)
    if CODE_NUMBER.fullmatch(s):return ('code_dash_number_ambiguous',None,None,multi)
    if BARE_CODE.fullmatch(s):return ('bare_code_ambiguous',None,None,multi)
    if FLOOR_MARKED.search(s):return ('other_explicit_floor',None,None,multi)
    return ('other',None,None,multi)


def signature(value):
    s=normalize(value)
    if s==MISSING:return MISSING
    s=re.sub(r'[A-Z]+',lambda m:m.group() if m.group() in {'F','B'} else 'L',s)
    s=re.sub(r'[0-9]+','#',s)
    s=re.sub('['+CN+']+','CN',s)
    keep=set('棟幢戶樓層地下上之號室全屋頂附夾')
    s=''.join(c if not ('\u4e00'<=c<='\u9fff') or c in keep else '字' for c in s)
    return re.sub('字+','字',s)


def finite(series):
    return np.isfinite(pd.to_numeric(series,errors='coerce').to_numpy(dtype='float64'))


def missing(series):
    return series.map(normalize).eq(MISSING).to_numpy()


def examples(norm,mask,limit=3):
    # Aggregated unit strings only: no street addresses, project names or row IDs.
    values=norm[mask].value_counts()
    return [{'text':str(k),'rows':int(v)} for k,v in values.head(limit).items() if len(str(k))<=60]


def analyze(path):
    frame=pd.read_pickle(path)
    need(isinstance(frame,pd.DataFrame) and frame.shape[1]==87,'Expected frozen target-free original87 cache')
    need(frame.columns.is_unique and frame.index.is_unique,'Feature schema/index not unique')
    need(not {'編號','總價元','y','target','ID','id'}&set(frame),'Forbidden ID/target column in feature cache')
    core=['building_and_unit','building_block','unit_floor','transfer_floor_raw']
    floors=[c for c in frame if c.startswith('floor_') or c in {
        'total_floors','relative_floor','floors_below_top','is_top_floor','is_basement','is_entire_building','is_rooftop'}]
    need(set(core+['floor_mean','floor_count']).issubset(frame),'Missing existing unit/floor features')
    X=frame[core+floors]
    norm=X.building_and_unit.map(normalize)
    counts=norm.value_counts()
    fact={s:describe(s) for s in counts.index}
    classes=norm.map(lambda s:fact[s][0]).to_numpy()
    codes=norm.map(lambda s:fact[s][1])
    explicit_floor=norm.map(lambda s:fact[s][2]).to_numpy(dtype='float64')
    multi=norm.map(lambda s:fact[s][3]).to_numpy(dtype=bool)
    unit_missing=norm.eq(MISSING).to_numpy()
    block_missing=missing(X.building_block)
    old_unit=X.unit_floor.to_numpy(dtype='float64')
    old_floor=X.floor_mean.to_numpy(dtype='float64')
    old_count=X.floor_count.to_numpy(dtype='float64')
    out={'rows':len(X),'feature_count':frame.shape[1],
         'index_type':type(frame.index).__name__,'index_unique':bool(frame.index.is_unique),
         'selected_columns':list(X),'dtypes':{c:str(X[c].dtype) for c in X},
         'unit_missing_rows':int(unit_missing.sum()),'unit_nonmissing_rows':int((~unit_missing).sum()),
         'unit_normalized_unique_nonmissing':int((counts.index!=MISSING).sum()),
         'building_block_missing_rows':int(block_missing.sum()),
         'unit_floor_finite_rows':int(np.isfinite(old_unit).sum()),
         'transfer_floor_mean_finite_rows':int(np.isfinite(old_floor).sum()),
         'multiple_object_separator_rows':int(multi.sum()),
         'numeric_fields':{},'classes':{},'top30_unit_strings':[],
         'top30_format_signatures':[]}
    for c in floors+['unit_floor']:
        values=pd.to_numeric(X[c],errors='coerce').to_numpy(dtype='float64');valid=np.isfinite(values)
        out['numeric_fields'][c]={'finite_rows':int(valid.sum()),'missing_rows':int(np.isnan(values).sum()),
            'nonfinite_nonmissing_rows':int((~valid&~np.isnan(values)).sum()),
            'minimum':float(values[valid].min()) if valid.any() else None,
            'maximum':float(values[valid].max()) if valid.any() else None}
    for label in CLASSES:
        mask=classes==label
        out['classes'][label]={'rows':int(mask.sum()),'fraction_all_rows':float(mask.mean()),
            'block_missing_rows':int((mask&block_missing).sum()),
            'unit_floor_finite_rows':int((mask&np.isfinite(old_unit)).sum()),
            'transfer_floor_mean_finite_rows':int((mask&np.isfinite(old_floor)).sum()),
            'examples_aggregate':examples(norm,mask)}
    need(sum(v['rows'] for v in out['classes'].values())==len(X),'Classes must partition the cache rows')
    clear=np.isfinite(explicit_floor)
    comparable=clear&np.isfinite(old_unit)
    single=np.isfinite(old_unit)&np.isfinite(old_floor)&(old_count==1)
    out['code_dash_explicit_floor_check']={
        'rows':int(clear.sum()),'old_unit_floor_missing':int((clear&~np.isfinite(old_unit)).sum()),
        'old_unit_floor_equal':int((comparable&(old_unit==explicit_floor)).sum()),
        'old_unit_floor_different':int((comparable&(old_unit!=explicit_floor)).sum()),
        'old_building_block_missing':int((clear&block_missing).sum()),
        'prefix_code_unique':int(codes[clear].nunique()),
        'zero_or_negative_explicit_floor_rows':int((clear&(explicit_floor<=0)).sum()),
    }
    out['unit_vs_single_transfer_floor']={
        'comparable_rows':int(single.sum()),'equal_rows':int((single&(old_unit==old_floor)).sum()),
        'different_rows':int((single&(old_unit!=old_floor)).sum()),
        'not_an_error_label':'Different fields may refer to different objects/floors; do not auto-repair.'}
    blockmark=classes=='explicit_block_marker'
    out['explicit_block_not_extracted']={'rows':int((blockmark&block_missing).sum()),
        'examples_aggregate':examples(norm,blockmark&block_missing)}
    # Count fixed illustrative formats directly; this does not change the classification rules.
    out['requested_examples']={s:int(counts.get(s,0)) for s in ['A1-5F','A棟5F','A1戶']}
    out['top30_unit_strings']=[{'text':str(k),'rows':int(v),'format':signature(k),'class':fact[k][0]}
                               for k,v in counts.items() if k!=MISSING][:30]
    sig_counts=Counter()
    sig_example={}
    for text,count in counts.items():
        if text==MISSING:continue
        key=signature(text);sig_counts[key]+=int(count)
        if key not in sig_example:sig_example[key]=text if len(text)<=60 else '(long unit string omitted)'
    out['top30_format_signatures']=[{'format':key,'rows':count,'example':sig_example[key]}
        for key,count in sorted(sig_counts.items(),key=lambda kv:(-kv[1],kv[0]))[:30]]
    transfer=X.transfer_floor_raw.map(normalize).value_counts()
    out['top30_transfer_floor_strings']=[{'text':str(k),'rows':int(v)} for k,v in transfer.items()][:30]
    out['input_feature_frame_unchanged']=bool(frame[core+floors].equals(X))
    return out


def escape(value):return str(value).replace('|','／').replace('\n',' ')


def render(data):
    a,b=data['splits']['train'],data['splits']['test']
    lines=['# R16 棟及號與移轉樓層：純 X 格式盤點','',
      '這是資料可用性研究，不是正式候選、訓練或成績報告。只讀既有 train_features.pkl 與 test_features.pkl；沒有讀 train.pkl、valid.pkl、原始 Excel、y、模型、預測或 MAE。查閱 test X 已獲授權，因此不能把本研究稱為完全未接觸測試分布的盲測。','',
      '## 既有處理與邊界','',
      '- 原 87 欄包含完整 building_and_unit 與 transfer_floor_raw 類別，字串資訊沒有被刪除。building_block 只抽取開頭 12 字內且以「棟／幢」結尾的片段；unit_floor 已抽取帶「樓／層／F/f」的數字與部分地下標記。',
      '- 移轉層次已有 floor_min/max/mean/count/span、地下／整棟／屋頂旗標，以及相對樓層、距頂樓與是否頂樓。跨地下至地上範圍排除不存在的 0 樓修正已在原 parser 中。',
      '- round3_features.py 修正地址行政區；round4_augmented.py 增加道路與建案前綴的 10 欄，沒有另外拆解棟戶字串。這僅確認所讀來源範圍，跨輪方法新穎性另見獨立歷史盤點。',
      '- train_features 是既有去除 3 列嵌入表頭後的 X cache；本次不重做含 target 的完整列去重，也不把本次 cache 筆數當成正式年度 fit 筆數。test 也未刪列或重排。','',
      '## 覆蓋概況','',
      '| 項目 | train X | test X |','|---|---:|---:|']
    for label,key in [('cache 列數','rows'),('棟及號非缺失','unit_nonmissing_rows'),('棟及號正規化後相異字串','unit_normalized_unique_nonmissing'),('building_block 缺失','building_block_missing_rows'),('unit_floor 有有限數值','unit_floor_finite_rows'),('移轉 floor_mean 有有限數值','transfer_floor_mean_finite_rows'),('含多物件／分隔符訊號','multiple_object_separator_rows')]:
        lines.append(f'| {label} | {a[key]:,} | {b[key]:,} |')
    lines+=['','## 固定少量格式分類','','只用下列預先寫入的八類；不以價格錯誤新增規則。`code` 只表示字串中的代碼，不能直接宣稱是真實棟別或戶型。','',
            '| 格式類別 | train 列數 | test 列數 | train 未抽 block | test 未抽 block |','|---|---:|---:|---:|---:|']
    for key in CLASSES:
        x,y=a['classes'][key],b['classes'][key]
        lines.append(f"| {key} | {x['rows']:,} | {y['rows']:,} | {x['block_missing_rows']:,} | {y['block_missing_rows']:,} |")
    lines+=['','## 可明確量化的缺口與已處理部分','']
    for label,s in [('train',a),('test',b)]:
        v=s['code_dash_explicit_floor_check'];u=s['classes']['explicit_unit_code'];z=s['unit_vs_single_transfer_floor'];e=s['explicit_block_not_extracted']
        lines.append(f"- {label}：如 `A1-5F` 的「代碼－明示樓層」有 {v['rows']:,} 列；既有 unit_floor 相等 {v['old_unit_floor_equal']:,}、缺失 {v['old_unit_floor_missing']:,}、不同 {v['old_unit_floor_different']:,} 列。這些列 building_block 缺失 {v['old_building_block_missing']:,} 列，可額外拆出 {v['prefix_code_unique']:,} 種前綴代碼，但代碼含義仍有歧義。")
        lines.append(f"- {label}：如 `A1戶` 的明示戶碼有 {u['rows']:,} 列，其中 building_block 缺失 {u['block_missing_rows']:,} 列；戶碼不是樓層。含棟／幢但未抽 block 共 {e['rows']:,} 列，具體聚合例子保存在 JSON。")
        lines.append(f"- {label}：unit_floor 與單一移轉樓層同時可比較 {z['comparable_rows']:,} 列，相等 {z['equal_rows']:,}、不同 {z['different_rows']:,} 列；差異不等同標錯，不能據此自動覆蓋任一欄。")
    lines+=['','## 前 30 個常見格式','',
      '格式摘要用 L 表示英文字串、# 表示阿拉伯數字、CN 表示中文數字；F/B 與棟戶樓層標記保留。表內例子是聚合的棟戶字串，不是逐筆地址。JSON 另保存前 30 個完整棟戶字串及移轉樓層字串，方便重現。']
    for label,s in [('train',a),('test',b)]:
        lines+=['',f'### {label}','','| 格式 | 列數 | 聚合例子 |','|---|---:|---|']
        for item in s['top30_format_signatures']:
            lines.append(f"| {escape(item['format'])} | {item['rows']:,} | {escape(item['example'])} |")
    lines+=['','## 可用方向與限制','',
      '可供後續考慮的是把明示樓層前的代碼、帶戶字的戶碼及格式旗標分開，幫助模型在未見完整棟戶字串間共享資訊。原樓層已大致可由既有 unit_floor／移轉樓層取得，應先看上述實際缺失與一致性，不應將整個方向描述成「補上原本完全沒有樓層」。目前沒有定案候選、沒有採用新 parser，也沒有證據證明可降低 MAE。','',
      '`A1` 可能同時表示棟、戶型、平面單元或銷售編碼；無 F／樓／層的 `A1-5` 不應擅自視為五樓。含多棟／多戶、範圍、地下、夾層、特殊字母或超過 12 字的前綴可能誤判。相同代碼跨建案通常不代表同一物理物件，新建案也可能採不同命名；只憑 test X 格式頻率不證明預測有效。後續若使用建案內統計，仍需 fit-only 與未見建案回退，不能用價格或 query 標籤推定代碼含義。','',
      '## 重現與來源','',
      '- [盤點程式](../src/round16_inventory_units.py)；執行 `python src/round16_inventory_units.py self-check` 或 `python src/round16_inventory_units.py run`。run 拒絕覆寫既有產物。',
      '- [完整 JSON 與來源 SHA](../outputs/round16_research/units_inventory.json)。所有來源與 X cache 均做前後 SHA 核對；不修改既有來源、cache 或模型。','',
      '| 來源 | SHA256 |','|---|---|']
    for name,digest in data['source_sha256'].items():lines.append(f'| {name} | `{digest}` |')
    for name,item in data['inputs'].items():lines.append(f"| {item['path']} | `{item['sha256']}` |")
    return '\n'.join(lines)+'\n'


def self_check():
    cases={'A1-5F':('code_dash_explicit_floor','A1',5,False),
           'Ａ１－５ｆ':('code_dash_explicit_floor','A1',5,False),
           'A1-B1F':('code_dash_explicit_floor','A1',-1,False),
           'A棟五樓':('explicit_block_marker',None,None,False),
           'A1戶':('explicit_unit_code','A1',None,False),
           'A1-5':('code_dash_number_ambiguous',None,None,False),
           'A1':('bare_code_ambiguous',None,None,False),
           MISSING:('missing',None,None,False)}
    for text,expected in cases.items():need(describe(text)==expected,'Synthetic format differs: '+text)
    need(describe('A1-5F、A2-6F')[0]!='code_dash_explicit_floor' and describe('A1-5F、A2-6F')[3],'Do not treat multiple units as one exact code-floor')
    need(signature('A1-5F')=='L#-#F','Signature digits/letters changed')
    need(normalize(None)==MISSING,'Missing normalization differs')
    return {'checks':len(cases)+3,'passed':True,'real_data_read':False}


def run():
    need(not DEST_JSON.exists() and not DEST_MD.exists(),'Refusing to overwrite inventory/report')
    checks=self_check()
    sources={name:sha(ROOT/name) for name in SOURCES}
    inputs={split:{'path':name,'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for split,name in INPUTS.items()}
    result={'kind':'round16_target_free_unit_inventory','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'formal_candidate_selected':False,'model_training_performed':False,'price_or_prediction_values_read':False,
      'query_X_was_inspected':True,'historically_untouched':False,
      'read_boundary':'Only original87 train_features.pkl/test_features.pkl and named source files; no target cache, raw Excel, models, predictions or metrics.',
      'source_sha256':sources,'inputs':inputs,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
      'rules':{'normalization':'NFKC uppercase whitespace removal; inventory only, existing features unchanged',
               'code_floor_pattern':CODE_FLOOR.pattern,'unit_code_pattern':UNIT_CODE.pattern,'ambiguous_code_number_pattern':CODE_NUMBER.pattern,
               'bare_code_pattern':BARE_CODE.pattern,'classes':list(CLASSES),'top_formats_limit':30},
      'synthetic_checks':checks,'splits':{split:analyze(ROOT/name) for split,name in INPUTS.items()}}
    for name,digest in sources.items():need(sha(ROOT/name)==digest,'Source changed during inventory')
    for item in inputs.values():need(sha(ROOT/item['path'])==item['sha256'],'X cache changed during inventory')
    result['input_and_source_hashes_unchanged']=True
    encoded=(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    md=render(result).encode('utf-8')
    destinations=[(DEST_JSON,encoded),(DEST_MD,md)]
    staged=[]
    for path,content in destinations:
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        with tmp.open('xb') as f:f.write(content)
        staged.append((tmp,path))
    need(not any(path.exists() for _,path in staged),'Concurrent inventory output appeared; staging retained')
    for tmp,path in staged:tmp.rename(path)
    print(json.dumps({'status':'completed','rows':{k:v['rows'] for k,v in result['splits'].items()},
      'json_sha256':sha(DEST_JSON),'report_sha256':sha(DEST_MD),'source_sha256':sources[SOURCES[0]],
      'code_floor':{k:v['code_dash_explicit_floor_check'] for k,v in result['splits'].items()},
      'explicit_unit':{k:v['classes']['explicit_unit_code'] for k,v in result['splits'].items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['self-check','run'])
    args=parser.parse_args()
    if args.action=='self-check':print(json.dumps(self_check(),ensure_ascii=False))
    else:run()
