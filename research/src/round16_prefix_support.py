"""One bounded pure-X supplement using the frozen two-rule unit-prefix parser.
Only original87 train/test feature caches are read; never labels or predictions.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import round16_unit_prefix as parser

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/round16_research/prefix_support.json'
REPORT=ROOT/'reports/round16_prefix_support.md'
FIRST=ROOT/'outputs/round16_research/units_inventory.json'
EXPECTED_FIRST_SHA='9d331dd8f303d9f5b00e72cefe20bac3fa54d13bd97c8256a4212649e5cc6664'
EXPECTED_PARSER_SHA='69d9903f57b1b4a7d7edd00f03a3499cc3e26c52b8d2ba97ae24fb8fc9c426ad'
SOURCE_NAMES=('src/round16_prefix_support.py','src/round16_unit_prefix.py')
INPUT_NAMES={'train':'data_cache/train_features.pkl','test':'data_cache/test_features.pkl'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def aggregate_examples(series,mask):
    return [{'text':str(k),'rows':int(v)} for k,v in series[mask].value_counts().head(3).items() if len(str(k))<=60]


def inspect_cache(path,previous):
    frame=pd.read_pickle(path)
    need(frame.shape[1]==87 and frame.columns.is_unique and frame.index.is_unique,'Expected original87 unique schema/index')
    need(not {'編號','總價元','y','target','ID','id'}&set(frame),'Target/ID column forbidden')
    need(len(frame)==previous['rows'],'Inventory row identity differs')
    raw=frame['building_and_unit'];oldblock=frame['building_block']
    normalized=raw.map(parser.normalize)
    detail={s:parser.prefix_details(s) for s in normalized.unique()}
    prefix=normalized.map(lambda s:detail[s]['prefix_code'])
    source=normalized.map(lambda s:detail[s]['source'])
    reason=normalized.map(lambda s:detail[s]['reason'])
    nonmissing=normalized.ne(parser.MISSING)
    missing_block=oldblock.map(parser.normalize).eq(parser.MISSING)
    eligible=prefix.ne(parser.MISSING)
    need(int(nonmissing.sum())==previous['unit_nonmissing_rows'],'Nonmissing X count differs')
    need(int(missing_block.sum())==previous['building_block_missing_rows'],'Existing block count differs')
    counts_by_source={}
    code_counts={}
    for origin in ['code_floor','explicit_block']:
        mask=source.eq(origin)
        count=prefix[mask].value_counts()
        code_counts[origin]={str(k):int(v) for k,v in count.items()}
        counts_by_source[origin]={
            'rows':int(mask.sum()),'fraction_all_rows':float(mask.mean()),
            'old_block_missing_rows':int((mask&missing_block).sum()),
            'unique_prefix_codes':len(count),'examples_aggregate':aggregate_examples(normalized,mask),
            'raw_strings_with_terminal_number_marker':int((mask&normalized.str.endswith('號')).sum()),
        }
    common=set(code_counts['code_floor'])&set(code_counts['explicit_block'])
    lexical={code:{'code_floor_rows':code_counts['code_floor'][code],
                   'explicit_block_rows':code_counts['explicit_block'][code]} for code in sorted(common)}
    blockmask=source.eq('explicit_block')
    old=oldblock.map(parser.normalize)
    old_agrees=old.eq(prefix+'棟')|old.eq(prefix+'幢')
    recovered=eligible&missing_block
    out={'rows':len(frame),'feature_count':87,'unit_nonmissing_rows':int(nonmissing.sum()),
         'building_block_missing_rows':int(missing_block.sum()),
         'nonmissing_unit_with_missing_block_rows':int((nonmissing&missing_block).sum()),
         'accepted_rows':int(eligible.sum()),'accepted_fraction_all_rows':float(eligible.mean()),
         'accepted_unique_codes':int(prefix[eligible].nunique()),
         'recovered_missing_block_rows':int(recovered.sum()),
         'recovered_fraction_of_all_rows':float(recovered.mean()),
         'recovered_fraction_of_all_missing_block':float(recovered.sum()/missing_block.sum()) if missing_block.any() else None,
         'recovered_fraction_of_nonmissing_unit_missing_block':float(recovered.sum()/(nonmissing&missing_block).sum()) if (nonmissing&missing_block).any() else None,
         'counts_by_source':counts_by_source,
         'reason_counts':{str(k):int(v) for k,v in reason.value_counts().items()},
         'rejected_examples_by_reason':{str(k):aggregate_examples(normalized,reason.eq(k)) for k in reason.unique() if k not in {'accepted','missing'}},
         'shared_code_unique_count':len(common),
         'code_floor_rows_whose_code_also_occurs_in_explicit_block':int((source.eq('code_floor')&prefix.isin(common)).sum()),
         'cross_source_lexical_code_counts':lexical,
         'explicit_block_old_prefix_agreement_rows':int((blockmask&old_agrees).sum()),
         'explicit_block_old_prefix_disagreement_rows':int((blockmask&~old_agrees).sum()),
         'top30_prefix_codes': [{'code':str(k),'rows':int(v)} for k,v in prefix[eligible].value_counts().head(30).items()],
         'existing95_or_reference_or_scoring_features_changed':False}
    need(sum(out['reason_counts'].values())==len(frame),'Reasons do not partition original rows')
    need(out['accepted_rows']==sum(v['rows'] for v in counts_by_source.values()),'Accepted sources do not partition rows')
    return out,code_counts


def percentage(value):return '—' if value is None else f'{100*value:.2f}%'


def render(result):
    a,b=result['splits']['train'],result['splits']['test'];cross=result['cross_split_lexical_overlap']
    lines=['# R16 字串前綴碼：一次限定純 X 補充','',
      '首版 inventory 與 source 保持不變。首版精確 code-floor 文法沒有接受常見末尾「號」，因此該窄類別為 0；前 30 格式表已顯示 `A1-5F號` 等實際支援。本補充只採事先固定的兩條規則，不讀價格、不訓練、不計 MAE，也不挑新門檻。','',
      '## 固定解析規則','',
      '- NFKC、移除空白、大寫；缺失與歧義回傳 `__MISSING__`。',
      '- 明示樓層：`'+parser.RULES['code_floor_regex']+'`。',
      '- 明示棟／幢：`'+parser.RULES['explicit_block_regex']+'`。',
      '- 共同拒絕逗號、頓號、分號、斜線、&、及／與／暨等多物件訊號，至／到／~ 範圍，或多於一個棟／幢標記。不將 `A1-5號` 當成五樓，不加中文數字或反向 `5F-A號` 規則。',
      '- 輸出只是字串前綴碼。`A1-5F號` 的 `A1` 與 `A1棟5F號` 的 `A1` 共用文字類別，不代表同一種類或同一個物理物件；兩種來源仍分開記錄以便稽核。','',
      '## 實際覆蓋','',
      '| 項目 | train X | test X |','|---|---:|---:|']
    for label,key in [('原 cache 列數','rows'),('棟及號非缺失','unit_nonmissing_rows'),('既有 building_block 缺失','building_block_missing_rows'),('有字串但缺 building_block','nonmissing_unit_with_missing_block_rows'),('兩規則總接受列數','accepted_rows'),('接受的相異前綴码','accepted_unique_codes'),('缺 block 但可提供前綴碼','recovered_missing_block_rows')]:
        lines.append(f'| {label} | {a[key]:,} | {b[key]:,} |')
    for label,key in [('補出前綴／所有列','recovered_fraction_of_all_rows'),('補出前綴／全部缺block列','recovered_fraction_of_all_missing_block'),('補出前綴／有字串但缺block列','recovered_fraction_of_nonmissing_unit_missing_block')]:
        lines.append(f'| {label} | {percentage(a[key])} | {percentage(b[key])} |')
    lines+=['','| 規則來源 | train 列數／unique code | test 列數／unique code |','|---|---:|---:|']
    for origin in ['code_floor','explicit_block']:
        x,y=a['counts_by_source'][origin],b['counts_by_source'][origin]
        lines.append(f"| {origin} | {x['rows']:,}／{x['unique_prefix_codes']:,} | {y['rows']:,}／{y['unique_prefix_codes']:,} |")
    lines+=['','## 跨來源文字重疊與拒絕','',
      f"- train 中兩種來源共用 {a['shared_code_unique_count']:,} 種碼，涵蓋明示 code-floor 的 {a['code_floor_rows_whose_code_also_occurs_in_explicit_block']:,} 列；test 為 {b['shared_code_unique_count']:,} 種、{b['code_floor_rows_whose_code_also_occurs_in_explicit_block']:,} 列。",
      f"- test 明示 code-floor 有 {cross['test_code_floor_rows_with_code_seen_in_train_explicit_block']:,} 列的碼也出現在 train 明示棟／幢格式。這是整份 cache 的純文字交集，並非正式年度 fit-only 詞表，也不能作實體相同的證據。",
      f"- 明示棟／幢來源與原 building_block 去除棟／幢後一致：train {a['explicit_block_old_prefix_agreement_rows']:,} 列、test {b['explicit_block_old_prefix_agreement_rows']:,} 列；不一致分別 {a['explicit_block_old_prefix_disagreement_rows']:,}/{b['explicit_block_old_prefix_disagreement_rows']:,}。",'',
      '| 拒絕原因 | train | test |','|---|---:|---:|']
    reasons=sorted(set(a['reason_counts'])|set(b['reason_counts']))
    for reason in reasons:
        if reason!='accepted':lines.append(f"| {reason} | {a['reason_counts'].get(reason,0):,} | {b['reason_counts'].get(reason,0):,} |")
    lines+=['','## 解讀與限制','',
      '完整 building_and_unit 本來就是類別特徵；這裡的增量是可共享的拆解資訊未顯式給出，不是原文字遺失。原 unit_floor 已能解析絕大多數這類樓層標記，因此本補充不覆寫樓層，也不更動原 95 欄、reference、建案分組或評分鍵。','',
      '相同碼在不同建案可能表示不同棟、戶型或銷售代碼；跨來源同碼會形成語意碰撞。多物件拒絕是保守的文字訊號，不能保證查出所有語意歧義。限定 ASCII 英數碼也會留下中文棟名、數字開頭、反向樓層等格式。本研究未為提高覆蓋率增加規則，也沒有證據顯示新類別能改善 MAE。','',
      'train X 是去除 3 列表頭後、尚未 canonical 529 列完整去重的 feature cache。本次不讀含 target 的去重資料、不建立建案統計、不處理 valid X；test X 的分布已被檢視，不能稱為未接觸盲測。','',
      '## 來源與重現','',
      '- [固定純 parser](../src/round16_unit_prefix.py)、[補充盤點程式](../src/round16_prefix_support.py)。`python src/round16_prefix_support.py run` 拒絕覆寫結果。',
      '- [補充 JSON](../outputs/round16_research/prefix_support.json) 與同名 SHA 收據包含來源、X 檔及首版證據的完整身份。',
      '- [保留的首版報告](round16_unit_inventory.md)；首版 JSON 的 receipt 在此次補充新增，只證明原 bytes SHA，沒有聲稱首版當時已存在 receipt。','',
      '| 新執行來源 | SHA256 |','|---|---|']
    for name,digest in result['source_sha256'].items():lines.append(f'| {name} | `{digest}` |')
    return '\n'.join(lines)+'\n'


def run():
    outputs=[DEST,DEST.with_suffix('.sha256'),REPORT]
    need(not any(p.exists() for p in outputs),'Refusing to overwrite supplemental artifacts')
    need(sha(FIRST)==EXPECTED_FIRST_SHA,'Original inventory changed')
    first=json.loads(FIRST.read_text(encoding='utf-8'))
    need(sha(ROOT/'src/round16_unit_prefix.py')==EXPECTED_PARSER_SHA,'Frozen parser changed')
    checks=parser.self_check()
    receipt=FIRST.with_suffix('.sha256')
    added=not receipt.exists()
    if added:
        with receipt.open('x',encoding='ascii') as f:f.write(EXPECTED_FIRST_SHA+'\n')
    need(receipt.read_text(encoding='ascii').strip()==EXPECTED_FIRST_SHA,'First byte receipt differs')
    evidence=dict(first['source_sha256'])
    for path in [FIRST,receipt,ROOT/'reports/round16_unit_inventory.md']:
        evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in evidence.items():need(sha(ROOT/name)==digest,'Original source/evidence changed')
    sources={name:sha(ROOT/name) for name in SOURCE_NAMES}
    inputs={}
    for split,name in INPUT_NAMES.items():
        digest=sha(ROOT/name)
        need(first['inputs'][split]['path']==name and first['inputs'][split]['sha256']==digest,'X cache differs from first inventory')
        inputs[split]={'path':name,'sha256':digest,'bytes':(ROOT/name).stat().st_size}
    splits={};code_counts={}
    for split,name in INPUT_NAMES.items():splits[split],code_counts[split]=inspect_cache(ROOT/name,first['splits'][split])
    trainblock=set(code_counts['train']['explicit_block'])
    cross={'test_code_floor_rows_with_code_seen_in_train_explicit_block':sum(n for c,n in code_counts['test']['code_floor'].items() if c in trainblock),
      'test_code_floor_unique_codes_seen_in_train_explicit_block':len(set(code_counts['test']['code_floor'])&trainblock),
      'not_a_physical_identity_or_formal_training_vocabulary':True}
    result={'kind':'round16_fixed_two_rule_prefix_support','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'source_sha256':sources,'inputs':inputs,'evidence_sha256':evidence,
      'original_inventory_receipt_added_during_supplement':added,
      'rules':parser.RULES,'synthetic_checks':checks,
      'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
      'data_access':{'only_target_free_train_test_feature_caches':True,'valid_cache_read':False,'raw_target_cache_or_Excel_read':False,
                     'price_prediction_or_MAE_read':False,'query_X_inspected':True,'formal_fit_or_score_performed':False},
      'original95_reference_scoring_keys_changed':False,'physical_identity_claimed':False,
      'splits':splits,'cross_split_lexical_overlap':cross}
    for mapping in [sources,evidence]:
        for name,digest in mapping.items():need(sha(ROOT/name)==digest,'Source/evidence changed during inventory')
    for item in inputs.values():need(sha(ROOT/item['path'])==item['sha256'],'X cache changed during inventory')
    result['inputs_sources_and_original_evidence_unchanged']=True
    raw=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    contents=[raw,(hashlib.sha256(raw).hexdigest()+'\n').encode('ascii'),render(result).encode('utf-8')]
    staged=[]
    for path,content in zip(outputs,contents):
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('xb') as f:f.write(content)
        staged.append((temporary,path))
    need(not any(path.exists() for _,path in staged),'Concurrent output appeared; staging retained')
    for temporary,path in staged:temporary.rename(path)
    print(json.dumps({'status':'completed','source_sha256':sources,'json_sha256':sha(DEST),'report_sha256':sha(REPORT),
      'counts':{split:{key:value[key] for key in ['rows','accepted_rows','accepted_unique_codes','recovered_missing_block_rows','recovered_fraction_of_nonmissing_unit_missing_block','counts_by_source','shared_code_unique_count','code_floor_rows_whose_code_also_occurs_in_explicit_block','explicit_block_old_prefix_disagreement_rows','reason_counts']} for split,value in splits.items()},
      'cross_split':cross},ensure_ascii=False),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=['self-check','run'])
    args=cli.parse_args()
    if args.action=='self-check':print(json.dumps(parser.self_check(),ensure_ascii=False))
    else:run()
