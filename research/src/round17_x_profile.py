"""R17 bounded target-free category profile; only train/test feature caches."""
from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round17_research'
JSON_PATH=OUT/'x_profile.json'
MD_PATH=ROOT/'reports/round17_x_profile.md'
SOURCE_PATH='src/round17_x_profile.py'
INPUTS={'train':'data_cache/train_features.pkl','test':'data_cache/test_features.pkl'}
SOURCES=(SOURCE_PATH,'src/house_features.py')
CATEGORIES=('transaction_type','building_type','main_use')
IDENTIFIERS=('project_name','building_and_unit','address_raw','building_block')
LABELS={'transaction_type':'交易標的','building_type':'建物型態','main_use':'主要用途'}
SPLITS={'full_train':'全 train X','development_fit':'開發 fit X','development_query':'開發 query X','test':'全 test X'}
EPOCH=pd.Timestamp('2000-01-01')
START=pd.Timestamp('2022-08-11')
STOP=pd.Timestamp('2023-08-11')
MISSING='__MISSING__'
TOP_N=30


def need(value,message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def nonmissing(series):
    # Exact cached strings are retained for set comparison; only blank detection strips.
    text=series.astype('string')
    return series.notna() & text.ne(MISSING).fillna(False) & text.str.strip().ne('').fillna(False)


def dates(frame):
    day=pd.to_numeric(frame['trade_day_index'],errors='raise')
    values=day.to_numpy(dtype='float64')
    valid=np.isfinite(values)
    need(np.all(valid | np.isnan(values)),'Infinite cached date')
    need(np.equal(values[valid],np.floor(values[valid])).all(),'Noninteger cached day index')
    return pd.Series(pd.to_datetime(day,unit='D',origin=EPOCH,errors='coerce'),index=frame.index)


def masks(date):
    return date.notna() & date.lt(START),date.ge(START) & date.lt(STOP)


def category_profile(series):
    mask=nonmissing(series); selected=series.loc[mask].astype(str)
    counts=selected.value_counts(dropna=False)
    items=sorted(((str(k),int(v)) for k,v in counts.items()),key=lambda x:(-x[1],x[0]))
    top=items[:TOP_N]
    return {'rows':int(len(series)),'missing_rows':int((~mask).sum()),'nonmissing_rows':int(mask.sum()),
        'nonmissing_unique_values':len(items),'top_limit':TOP_N,
        'top_categories':[{'value':k,'rows':v,'share_of_all_rows':v/len(series) if len(series) else None} for k,v in top],
        'other_nonmissing_rows':int(mask.sum())-sum(v for _,v in top)}


def identifier_profile(series):
    mask=nonmissing(series)
    return {'rows':int(len(series)),'missing_rows':int((~mask).sum()),'nonmissing_rows':int(mask.sum()),
            'nonmissing_unique_values':int(series.loc[mask].nunique(dropna=True))}


def unseen_profile(reference,query):
    rmask,qmask=nonmissing(reference),nonmissing(query)
    known=set(reference.loc[rmask].astype(str))
    values=query.loc[qmask].astype(str)
    unknown=~values.isin(known)
    unique=set(values)
    return {'reference_nonmissing_rows':int(rmask.sum()),'reference_unique_values':len(known),
        'query_rows':len(query),'query_missing_rows':int((~qmask).sum()),'query_nonmissing_rows':len(values),
        'unseen_nonmissing_rows':int(unknown.sum()),
        'unseen_row_rate_among_nonmissing':float(unknown.mean()) if len(values) else None,
        'query_nonmissing_unique_values':len(unique),'unseen_unique_values':len(unique-known),
        'unseen_unique_rate':len(unique-known)/len(unique) if unique else None,
        'missing_excluded_from_unseen_denominator':True}


def date_audit(frame):
    result=dates(frame);good=result.notna()
    need((result.loc[good].dt.year.between(1990,2030)).all(),'Date outside frozen feature bounds')
    for column,component in [('trade_year',result.dt.year),('trade_month',result.dt.month),('trade_day',result.dt.day)]:
        given=pd.to_numeric(frame[column],errors='raise')
        need(given.isna().equals(component.isna()) and given.loc[good].eq(component.loc[good]).all(),
             'Cached date components disagree: '+column)
    flag=pd.to_numeric(frame['trade_date_invalid'],errors='raise')
    need(flag.eq((~good).astype('int64')).all(),'Cached invalid-date flag disagrees')
    return result,{'valid_rows':int(good.sum()),'missing_or_invalid_rows':int((~good).sum()),
        'minimum':None if not good.any() else result.min().date().isoformat(),
        'maximum':None if not good.any() else result.max().date().isoformat(),
        'cached_year_month_day_invalid_flag_consistent':True,
        'out_of_range_flag_rows':int(pd.to_numeric(frame['trade_date_out_of_range'],errors='raise').eq(1).sum())}


def pct(value):return '—' if value is None else f'{100*value:.2f}%'
def escaped(value):return str(value).replace('|','\\|').replace('\n',' ')


def render(value):
    lines=['# 第十七輪前：純 X 類別輪廓','','本文件只盤點既有特徵字串與日期，不提出已有效的價格方法，也沒有建立正式候選、訓練或計算 MAE。','',
        '## 資料與時間口徑','',
        '只讀 train_features.pkl 與 test_features.pkl。train X 已去除 3 列嵌入英文表頭，尚未做包含價格的 529 列 canonical 完整去重；本次不載入 target，也不重現該去重。因此下列窗口列數不可直接當作正式模型 240,316／81,234 列。沒有讀 valid、原始 Excel、帶 y 的快取、模型或預測。','',
        'trade_day_index 是距 2000-01-01 的整數日數；來源將解析後落在 1990–2030 外的日期轉成缺值。盤點已核對快取的年、月、日及 invalid 旗標。fit 使用日期 <2022-08-11；query 使用 2022-08-11≤日期<2023-08-11（含 2023-08-10）。無效日期只留在全 train 的描述，不進時間窗口。','',
        '| 資料集合 | 列數 | 日期缺失／無效 |','|---|---:|---:|']
    for key,title in SPLITS.items():
        item=value['profiles'][key]
        lines.append(f"| {title} | {item['rows']:,} | {item['date_missing_or_invalid_rows']:,} |")
    lines+=['','## 交易、型態與用途','','所有類別沿用快取原字串，不合併歷史行政名稱、不重新分群。下表每欄最多列出 30 個常見非缺失值，另列缺失與其他類別筆數。']
    for key,title in SPLITS.items():
        for column in CATEGORIES:
            item=value['profiles'][key]['categories'][column]
            lines += ['',f"### {title}：{LABELS[column]}（{column}）",'',
                f"缺失 {item['missing_rows']:,} 列；非缺失 {item['nonmissing_rows']:,} 列、{item['nonmissing_unique_values']:,} 種值。",'',
                '| 類別 | 列數 | 占該集合 |','|---|---:|---:|']
            lines += [f"| {escaped(row['value'])} | {row['rows']:,} | {pct(row['share_of_all_rows'])} |" for row in item['top_categories']]
            if item['other_nonmissing_rows']:lines.append(f"| 其他未列非缺失類別 | {item['other_nonmissing_rows']:,} | — |")
    lines+=['','## 名稱欄位缺失與唯一值','','沒有展示原始地址或棟戶字串；只保留聚合數。','',
            '| 集合 | 欄位 | 非缺失列 | 缺失列 | 非缺失唯一值 |','|---|---|---:|---:|---:|']
    for key,title in SPLITS.items():
        for column,item in value['profiles'][key]['identifiers'].items():
            lines.append(f"| {title} | {column} | {item['nonmissing_rows']:,} | {item['missing_rows']:,} | {item['nonmissing_unique_values']:,} |")
    lines+=['','## 純字串集合未見率','','development query 對 development fit；test 對完整 train X。分母只含 query/test 的非缺失列；缺失另列，不算作新名稱。比較是快取字串精確相等，不建立 city|district|project 實體鍵，所以相同建案名跨地區碰撞仍可能被當已見，棟碼／戶號跨建案共享也不代表相同物件。這不是模型正式評分的三組建案口徑。','',
            '| 比較 | 欄位 | 非缺失列 | 未見列 | 列未見率 | 未見唯一值／全部唯一值 | 缺失列 |','|---|---|---:|---:|---:|---:|---:|']
    for key,comparison in value['unseen_comparisons'].items():
        for column,item in comparison['fields'].items():
            lines.append(f"| {comparison['query']} 對 {comparison['reference']} | {column} | {item['query_nonmissing_rows']:,} | {item['unseen_nonmissing_rows']:,} | {pct(item['unseen_row_rate_among_nonmissing'])} | {item['unseen_unique_values']:,}／{item['query_nonmissing_unique_values']:,} | {item['query_missing_rows']:,} |")
    lines+=['','## 可解讀範圍','','類別的支援量、稀有類別與未見字串可說明後續研究需要處理的資料覆蓋；它們不能證明拆成不同房屋型態的模型會較好，也不能證明新編碼能降低 MAE。本盤點不選門檻、不改既有特徵或評估規約。',
            '', '已使用 development query X 與 test X 描述分布，故不能稱完全未接觸的盲測；沒有使用 query/test 價格，也沒有計算任何新價格成績。', '',
            '- [可重跑程式](../src/round17_x_profile.py)',
            '- [完整機器統計與來源 SHA](../outputs/round17_research/x_profile.json)',
            '- [既有日期與類別定義](../src/house_features.py)', '']
    return '\n'.join(lines)


def run():
    destinations=(JSON_PATH,JSON_PATH.with_suffix('.sha256'),MD_PATH)
    need(not any(p.exists() for p in destinations),'Refusing to overwrite profile artifacts')
    need(not list(OUT.glob('.x_profile-stage-*')),'Prior partial staging must be retained and inspected')
    inputs={k:{'path':p,'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for k,p in INPUTS.items()}
    source_hashes={p:sha(ROOT/p) for p in SOURCES}
    frames={k:pd.read_pickle(ROOT/p) for k,p in INPUTS.items()}
    required=set(CATEGORIES+IDENTIFIERS+('trade_day_index','trade_year','trade_month','trade_day','trade_date_invalid','trade_date_out_of_range'))
    for key,frame in frames.items():
        need(isinstance(frame,pd.DataFrame) and frame.columns.is_unique and frame.index.is_unique,'Invalid X schema/index')
        need(len(frame.columns)==87 and required<=set(frame.columns),'Unexpected original feature schema')
        need(not {'編號','總價元','target','y','ID','id'}&set(frame.columns),'Forbidden identifier/target column')
        need(all(str(frame[c].dtype)=='object' for c in CATEGORIES+IDENTIFIERS),'Unexpected category dtype')
    need(frames['train'].columns.equals(frames['test'].columns) and frames['train'].dtypes.equals(frames['test'].dtypes),'Train/test feature schema differs')
    date_values={};date_summary={}
    for key,frame in frames.items():date_values[key],date_summary[key]=date_audit(frame)
    fit,query=masks(date_values['train'])
    need(not (fit & query).any(),'Fit/query overlap')
    selections={'full_train':frames['train'],'development_fit':frames['train'].loc[fit],
                'development_query':frames['train'].loc[query],'test':frames['test']}
    profile={}
    for key,frame in selections.items():
        profile[key]={'rows':len(frame),'date_missing_or_invalid_rows':int(frame['trade_date_invalid'].eq(1).sum()),
            'categories':{c:category_profile(frame[c]) for c in CATEGORIES},
            'identifiers':{c:identifier_profile(frame[c]) for c in IDENTIFIERS}}
    comparisons={}
    for key,reference,queryname in [('development_query_vs_fit','development_fit','development_query'),('test_vs_full_train','full_train','test')]:
        comparisons[key]={'reference':reference,'query':queryname,
            'fields':{c:unseen_profile(selections[reference][c],selections[queryname][c]) for c in IDENTIFIERS}}
    value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'research_only':True,'round17_candidate_created':False,
        'source_sha256':source_hashes,'inputs':inputs,'runtime':{'python_note':'project .venv','pandas':pd.__version__,'numpy':np.__version__},
        'schema':{'columns':list(frames['train'].columns),'dtypes':{c:str(t) for c,t in frames['train'].dtypes.items()},
                  'feature_count':87,'actual_categorical_count':int(sum(frames['train'].dtypes.eq(object)))},
        'date_definition':{'epoch':'2000-01-01','unit':'days','valid_source_years':[1990,2030],
            'fit_before':'2022-08-11','query_start_inclusive':'2022-08-11','query_stop_exclusive':'2023-08-11',
            'invalid_dates_excluded_from_temporal_windows':True},'date_summary':date_summary,
        'row_basis':{'train_after_three_embedded_header_rows_removed':True,'canonical_529_full_row_dedup_applied':False,
            'target_involved_historical_dedup_repeated_here':False,'model_canonical_fit_query_counts_not_claimed':True,
            'train_rows_outside_fit_query':int(len(frames['train'])-fit.sum()-query.sum())},
        'missing_policy':'actual null, exact __MISSING__, or whitespace-only; other strings retained exactly',
        'profiles':profile,'unseen_comparisons':comparisons,
        'data_access':{'only_two_target_free_feature_caches_read':True,'query_and_test_X_read':True,
            'raw_target_cache_or_Excel_or_valid_read':False,'model_or_prediction_read':False,
            'price_or_MAE_computed':False,'formal_fit_or_score_performed':False,'historically_untouched':False},
        'synthetic_checks':self_check(),
        'execution_notes':['Initial synthetic date roundtrip stopped before any real X read because pandas datetime us/ns resolution made Series.equals dtype-sensitive; the synthetic assertion now compares both at ns precision. No date boundaries or profile rules changed.']}
    for key,item in inputs.items():need(sha(ROOT/item['path'])==item['sha256'] and (ROOT/item['path']).stat().st_size==item['bytes'],'Input changed')
    need(all(sha(ROOT/p)==h for p,h in source_hashes.items()),'Source changed')
    value['inputs_and_sources_unchanged']=True
    encoded=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    contents=(encoded,(hashlib.sha256(encoded).hexdigest()+'\n').encode('ascii'),render(value).encode('utf-8'))
    OUT.mkdir(parents=True,exist_ok=True);staging=OUT/('.x_profile-stage-'+uuid.uuid4().hex);staging.mkdir()
    for path,content in zip(destinations,contents):
        with (staging/path.name).open('xb') as f:f.write(content)
    need(not any(p.exists() for p in destinations),'Concurrent profile exists')
    for path in destinations:(staging/path.name).rename(path)
    staging.rmdir()
    print(json.dumps({'status':'profile_written','paths_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in destinations},
        'source_sha256':source_hashes,'rows':{k:v['rows'] for k,v in profile.items()},'unseen':comparisons},ensure_ascii=False),flush=True)


def self_check():
    checks=[]
    def check(name,value):need(value,name);checks.append(name)
    d=pd.Series(pd.to_datetime(['2022-08-10','2022-08-11','2023-08-10','2023-08-11',None]))
    f,q=masks(d)
    check('fixed_fit_and_inclusive_query_end',[bool(v) for v in f]==[True,False,False,False,False]
          and [bool(v) for v in q]==[False,True,True,False,False])
    day=(d-EPOCH).dt.days
    check('day_index_2000_epoch_roundtrip',dates(pd.DataFrame({'trade_day_index':day})).astype('datetime64[ns]').equals(d.astype('datetime64[ns]')))
    s=pd.Series(['A','A',None,MISSING,' ','B'])
    out=category_profile(s)
    check('missing_and_top_counts',out['missing_rows']==3 and out['nonmissing_rows']==3
          and out['top_categories'][0]['value']=='A' and out['top_categories'][0]['rows']==2)
    u=unseen_profile(s,pd.Series(['A','C','C',MISSING,None]))
    check('unseen_nonmissing_row_and_unique_denominators',u['unseen_nonmissing_rows']==2
          and u['query_nonmissing_rows']==3 and u['unseen_row_rate_among_nonmissing']==2/3
          and u['unseen_unique_values']==1 and u['query_nonmissing_unique_values']==2)
    check('all_missing_query_rate_is_null',unseen_profile(s,pd.Series([None,MISSING]))['unseen_row_rate_among_nonmissing'] is None)
    check('exact_cached_strings_not_aliased',unseen_profile(pd.Series(['A']),pd.Series([' A ']))['unseen_nonmissing_rows']==1)
    check('read_scope_is_exactly_two_X_caches',set(INPUTS.values())=={'data_cache/train_features.pkl','data_cache/test_features.pkl'})
    return {'passed':True,'checks':checks,'count':len(checks),'real_data_used_in_checks':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    else:run()