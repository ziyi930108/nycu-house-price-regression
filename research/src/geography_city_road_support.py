"""One fixed, target-free 2024 city/road support inventory; no coordinate release."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/geography_city_road_support_2024'
REPORT=ROOT/'reports/geography_city_road_support_2024.md'
PARSED='outputs/geography/address_audit/test_parsed.pkl'
INDEX='outputs/geography/road_index_2024_annual_v3/road_index.jsonl'
AUDIT='outputs/geography/road_index_2024_annual_v3/audit.json'
MANIFEST='outputs/round6/geography/final/manifest.json'
MEMO='reports/geography_city_road_feasibility.md'
EXPECTED={PARSED:'1a1352023da75aafb86dd23ff5c5f074ef93eafb3bb8d90b902478bfe97886ec',
 INDEX:'2ab7f655685123ac652be142d03bca50e352c2b463b87dcdb9c3a50435cc04e2',
 AUDIT:'5924b8c5971290dc639d66f9d4226a4ed8670839b7235a77eb4687c28f3a7a0c',
 MANIFEST:'f8c2b44161ea9588d460cbdbb82f003b63f7d1c4001660a8ad32f42c36b5a6ee',
 MEMO:'e7ad23a98994c933d438f150b2d20c98795905bbb6e4e0a277f9b5d7e88f5c98'}
SOURCE_PATHS=['src/geo_address_audit.py','src/geo_match_queries_v2.py','src/geo_road_index.py','src/geo_road_index_v3.py']
CN='零〇一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰'
EXCLUDE=['land_parcel_context','near_or_opposite_or_unassigned','house_number_range','road_parse_uncertain','intersection_explicit','intersection_inferred','multiple_house_numbers']
RULES={'candidate_definition':'Exactly the prefilter in geography_city_road_feasibility.md; no new address repair',
 'expected_test_rows':47073,'expected_candidate_rows':710,'expected_city_road_pairs':72,
 'admin':'explicit city, empty district_full and district_in_address, no district_prefix_conflict, supplied normalized value blank or supplied_district_is_city',
 'road':'road_count=1, road_mentions_count=1, nonempty road1_name; candidate JSON exactly one same road_name with no context_uncertain or section_unsupported',
 'excluded_flags':EXCLUDE,'neighbor_prefix_rejection':'road1_base starts with [0-9'+CN+']+鄰; no renaming',
 'join':'exact (city, full road_name), collect all original district keys before considering quality',
 'zero':'no exact city/full-road record','unique':'exactly one original district road key','multiple':'more than one original district road key; never choose by quality',
 'duplicate_full_key':'hard error, matching original index key uniqueness contract; no silent deduplication',
 'unique_quality':'original quality road_level; original matcher finite latitude in [20,28], longitude in [116,125] checked, coordinates not exported',
 'geometry_recomputed':False,'fuzzy_matching':False,'base_road_fallback':False,'historical_name_alias_added':False,
 'other_years_processed':False,'coordinate_release':False,'price_or_model_use':False}

def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def lock_write(path,value):
 write(path,value)
 with Path(path).with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')

def candidate_counts(parsed):
 need(not {'y','總價元'}.intersection(parsed.columns),'Target column forbidden')
 flags=['is_embedded_header','district_prefix_conflict',*EXCLUDE]
 need(not parsed[flags+['road_count','road_mentions_count']].isna().any().any(),'Missing parser gate')
 n=parsed.loc[~parsed.is_embedded_header].copy()
 need(len(n)==RULES['expected_test_rows'],'Test row count changed')
 explicit=n.city.fillna('').ne('');absent=n.district_full.fillna('').eq('');conflict=n.district_prefix_conflict.astype(bool)
 blank=n.supplied_district_normalized.fillna('').eq('');city_only=n.supplied_district_is_city.astype(bool)
 admin=explicit & absent & n.district_in_address.fillna('').eq('') & ~conflict & (blank|city_only)
 single=admin & n.road_count.eq(1) & n.road_mentions_count.eq(1) & n.road1_name.fillna('').ne('')
 clean=single.copy()
 for flag in EXCLUDE:clean &= ~n[flag].astype(bool)
 neighbor=n.road1_base.fillna('').astype(str).str.match(r'^[0-9'+CN+r']+鄰');clean &= ~neighbor
 context_ok=pd.Series(False,index=n.index)
 for i,row in n.loc[clean].iterrows():
  cs=json.loads(row.road_candidates_json)
  context_ok.loc[i]=len(cs)==1 and cs[0].get('road_name')==row.road1_name and not cs[0].get('context_uncertain',False) and not cs[0].get('section_unsupported',False)
 clean &= context_ok
 selected=n.loc[clean]
 pairs=collections.Counter(zip(selected.city,selected.road1_name))
 need(int(clean.sum())==710 and len(pairs)==72,'Fixed 710/72 candidate set changed')
 need(selected.groupby('city').size().to_dict()=={'嘉義市':130,'新竹市':580},'Fixed city counts changed')
 need(not selected.road_query_eligible.any(),'Candidate already eligible under original matcher')
 counts={'non_header_rows':len(n),'explicit_city_missing_parsed_district':int((explicit&absent).sum()),'district_conflicts_excluded':int(conflict.sum()),
  'admin_prefilter_rows':int(admin.sum()),'single_road_and_mention_rows':int(single.sum()),'candidate_rows':int(clean.sum()),'candidate_city_road_pairs':len(pairs),
  'source_city_only_rows':int((clean&city_only).sum()),'source_blank_rows':int((clean&blank).sum())}
 return pairs,counts

def inspect_record(row):
 # Validate the original quality declaration, without constructing geometry or choosing new coordinates.
 quality=row['quality'];components=row['connected_components_with_50m_gap_tolerance'];extent=row['extent_diagonal_m']
 need(isinstance(components,int) and components>=1 and math.isfinite(extent) and extent>=0,'Malformed original geometry summary')
 expected='disconnected' if components>1 else 'broad_road' if extent>10000 else 'road_level'
 need(quality==expected,'Original road quality inconsistent with its recorded policy')
 lat,lon=row['latitude'],row['longitude']
 if quality=='road_level':need(lat is not None and lon is not None and math.isfinite(lat) and math.isfinite(lon) and 20<=lat<=28 and 116<=lon<=125,'Original usable coordinate outside matcher contract')
 else:need(lat is None and lon is None,'Original rejected road unexpectedly contains coordinates')
 return {k:row[k] for k in ['key','district','quality','extent_diagonal_m','connected_components_with_50m_gap_tolerance','nearby_fragment_grouping','source_way_count']} | {'original_coordinate_admissible':quality=='road_level'}

def classify(pair,rows,records):
 records=sorted(records,key=lambda x:x['key']);count=len(records)
 status='zero' if count==0 else 'unique' if count==1 else 'multiple'
 usable=count==1 and records[0]['original_coordinate_admissible']
 reason='exact_city_full_road_not_in_2024_index' if not count else 'multiple_district_road_keys' if count>1 else 'unique_original_road_level' if usable else 'unique_original_quality_'+records[0]['quality']
 return {'city':pair[0],'road_name':pair[1],'candidate_rows':rows,'matching_original_key_count':count,'status':status,'unique_usable':usable,'reason':reason,'records':records}

def totals(items):
 return {'candidate_rows':sum(v['candidate_rows'] for v in items),'city_road_pairs':len(items),
  **{status+'_rows':sum(v['candidate_rows'] for v in items if v['status']==status) for status in ['zero','unique','multiple']},
  **{status+'_pairs':sum(v['status']==status for v in items) for status in ['zero','unique','multiple']},
  'unique_usable_rows':sum(v['candidate_rows'] for v in items if v['unique_usable']),
  'unique_usable_pairs':sum(v['unique_usable'] for v in items),
  'unique_unusable_rows':sum(v['candidate_rows'] for v in items if v['status']=='unique' and not v['unique_usable']),
  'unique_unusable_pairs':sum(v['status']=='unique' and not v['unique_usable'] for v in items)}

def render(d):
 t=d['total'];lines=['# 2024 城市＋完整路名：缺區候選的純 X 支援盤點','',
 f"固定的 710 筆／72 個 city-road 候選中，**{t['unique_usable_rows']:,} 筆、{t['unique_usable_pairs']} 鍵**在 2024 歷史索引恰有一個行政區道路鍵，且原品質為 road_level；{t['zero_rows']:,} 筆無精確鍵，{t['multiple_rows']:,} 筆多鍵歧義，{t['unique_unusable_rows']:,} 筆雖唯一但原品質不可用。這是索引支援數，不是已發布新增座標或房價效益。",'',
 '| 城市 | 候選列／鍵 | 零鍵列 | 唯一鍵列 | 多鍵列 | 唯一且原品質可用列 | 唯一但不可用列 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for city,v in d['by_city'].items():lines.append(f"| {city} | {v['candidate_rows']}/{v['city_road_pairs']} | {v['zero_rows']} | {v['unique_rows']} | {v['multiple_rows']} | {v['unique_usable_rows']} | {v['unique_unusable_rows']} |")
 lines += ['', '候選定義與 [前置備忘](geography_city_road_feasibility.md) 完全相同：明示城市、parsed 區名及地址區名皆空、無衝突、供應欄空或只填城市；1 路／1 mention，排除既定地籍、near、range、解析不確定、多門牌／交叉口等旗標，並核單一候選 JSON。未擴詞、修地名或重新挑 710 的分母。47,073 個 test 原列保留重複列，不重算 target 去重；沒有讀 train/valid 或價格。','',
 f"本次單次 stream 讀取 {d['index_scan']['rows']:,} 個原道路 record（{d['index_scan']['bytes']:,} bytes），每個完整 key 都核為 city|district|road_name 且全檔唯一；沒有同 key 多列，未做去重合併。全檔 SHA 對原 final manifest／index audit 一致。只對目標 72 個精確 city-road 保留 key/品質摘要，沒有輸出座標或原始房屋地址。",'',
 '排歧先使用全部品質的 district keys，才對恰一鍵套原 road_level；多键即使只有一條品質較好，也不選它。保留原 disconnected／broad_road 政策（50m 間隙組群後多組、extent 超過10km），核原 road_level 座標有限且在既有 matcher 邊界內。這不是新半徑或新品質門檻。','',
 '## 無鍵與歧義明細','', '| 城市 | 完整路名字串 | 候選列 | 原鍵數 | 原行政區／品質 | 原因 |','| --- | --- | ---: | ---: | --- | --- |']
 for v in d['pairs']:
  if v['unique_usable']:continue
  display='；'.join(x['district']+'/'+x['quality'] for x in v['records']) or '—'
  lines.append(f"| {v['city']} | {v['road_name']} | {v['candidate_rows']} | {v['matching_original_key_count']} | {display} | {v['reason']} |")
 lines += ['', '完整 72 鍵（包括可用者）的城市、路名、行數、所有匹配區鍵及品質保存在 JSON。零鍵只表示此年度精確索引沒有該字串，不能斷言現實道路不存在；多鍵可能是同路跨區，也可能是不同道路，這次不推論或合併。','',
 '## 判斷及界線','',
 f"本次可支持的上限是 {t['unique_usable_rows']:,} 筆，占固定候選 {t['unique_usable_rows']/710:.2%}、占全部 test 原列 {t['unique_usable_rows']/47073:.2%}。下一步若繼續，應先對這一固定缺區 fallback 設計來源標記與原座標不變檢查，再以各自年份索引盤點訓練／回測支援；本次不做其他年份，也不建立價格候選。",
 '索引唯一不等於現實世界唯一，OSM 缺漏仍可能造成表面唯一。原 parser 的正規化／路名字串也不是全文與地標／門牌語意认证；此計數不擴充它的資格保證。原代表點只是區內道路上靠近長度加權中心的點，不是門牌；50m 碎片群組不等於物理道路連通。新 inferred district 不能直接當成已知真實行政區，更不能靜默覆寫原地址欄。',
 '本次未調用舊 matcher、未寫 parsed/matched/features、未載模型／價格／預測／score，未下載資料或使用外部成交答案。沒有 coordinate release、CSV、registry 或 CURRENT_STATUS 變更。','',
 '第一次命令在資料載入前因本程式僅接受純 SHA 收據而中止；舊 index audit 的收據實為 SHA 加 audit.json 檔名，JSON 身分未變。已保留原程式與失敗執行紀錄，只擴充收據讀取以接受並核對該明確檔名；候選／排歧／品質規則全未改。此次成功執行才做唯一完整索引掃描。原紀錄見 [receipt failure](../outputs/geography_city_road_support_2024_receipt_failure_01/execution.json)。','',
 f"執行開始 {d['started_at_utc']}，程式內耗時 {d['elapsed_seconds']:.3f} 秒；固定規則、來源SHA及完整聚合見 [support.json](../outputs/geography_city_road_support_2024/support.json)，[實際腳本](../src/geography_city_road_support.py)；命令／exit／stdout另存 [execution.json](../outputs/geography_city_road_support_2024/execution.json)。",'']
 return '\n'.join(lines)

def run():
 start=time.perf_counter();need(not OUT.exists() and not REPORT.exists(),'Output already exists; no overwrite')
 OUT.mkdir(parents=True,exist_ok=False);write(OUT/'run_claim.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'rules':RULES,'source_sha256':sha(__file__)})
 source_before={name:sha(ROOT/name) for name in [*SOURCE_PATHS,MEMO]};source_before['src/geography_city_road_support.py']=sha(__file__)
 for name,digest in EXPECTED.items():
  if name!=INDEX:need(sha(ROOT/name)==digest,'Fixed input differs: '+name)
 manifest=read(ROOT/MANIFEST);audit=read(ROOT/AUDIT)
 for name in [MANIFEST,AUDIT]:
  receipt=(ROOT/name).with_suffix('.sha256').read_text(encoding='ascii').split()
  need(len(receipt) in (1,2) and receipt[0]==sha(ROOT/name) and (len(receipt)==1 or receipt[1]==Path(name).name),'Receipt differs')
 need(manifest['snapshot_year']==2024 and manifest['geo_index_audit_path']==AUDIT and manifest['source_sha256'][INDEX]==EXPECTED[INDEX],'Historical year/index differs')
 need(audit['output_sha256']['road_index.jsonl']==EXPECTED[INDEX],'Index expected hash disagrees')
 for name in SOURCE_PATHS:need(source_before[name]==manifest['source_sha256'][name],'Recorded geographic source changed')
 parsed=pd.read_pickle(ROOT/PARSED);pairs,prefilter=candidate_counts(parsed);del parsed
 wanted={key:[] for key in pairs};all_keys=set();digest=hashlib.sha256();rows=0;index_stat=(ROOT/INDEX).stat()
 with (ROOT/INDEX).open('rb') as handle:
  for line in handle:
   digest.update(line);row=json.loads(line);key=row['key'];rows+=1
   need(key=='|'.join([row['city'],row['district'],row['road_name']]),'Malformed original key')
   need(key not in all_keys,'Duplicate original index key: '+key);all_keys.add(key)
   pair=(row['city'],row['road_name'])
   if pair in wanted:wanted[pair].append(inspect_record(row))
 need(digest.hexdigest()==EXPECTED[INDEX] and rows==audit['road_keys'],'Index full-stream SHA/count mismatch')
 need((ROOT/INDEX).stat().st_size==index_stat.st_size and (ROOT/INDEX).stat().st_mtime_ns==index_stat.st_mtime_ns,'Index changed during read')
 items=[classify(pair,count,wanted[pair]) for pair,count in sorted(pairs.items())];total=totals(items)
 need(total['zero_rows']+total['unique_rows']+total['multiple_rows']==710,'Row partition differs')
 need(total['unique_usable_rows']+total['unique_unusable_rows']==total['unique_rows'],'Unique quality partition differs')
 for name,value in source_before.items():need(sha(ROOT/name)==value,'Source changed during run')
 need(sha(ROOT/PARSED)==EXPECTED[PARSED],'Parsed source changed during run')
 d={'status':'completed_pure_X_support_only','started_at_utc':read(OUT/'run_claim.json')['started_at_utc'],'completed_at_utc':datetime.now(timezone.utc).isoformat(),
  'rules':RULES,'source_sha256':source_before,'input_sha256':{name:digest for name,digest in EXPECTED.items() if name!=MEMO},'prefilter':prefilter,
  'index_scan':{'rows':rows,'bytes':index_stat.st_size,'sha256':digest.hexdigest(),'all_original_full_keys_unique':True,'full_scans':1,'quality_filtered_before_disambiguation':False},
  'original_index_quality_rules':audit['fixed_geographic_rules'],'total':total,'by_city':{city:totals([v for v in items if v['city']==city]) for city in sorted({v['city'] for v in items})},'pairs':items,
  'previous_pre_data_receipt_failure':{'execution_path':'outputs/geography_city_road_support_2024_receipt_failure_01/execution.json','source_path':'outputs/geography_city_road_support_2024_receipt_failure_01/original_source.py','exit_code':1,'parsed_or_index_data_read':False,'candidate_disambiguation_quality_functions_changed':False},
  'raw_addresses_exported':False,'coordinates_exported':False,'price_values_read':False,'models_loaded':False,'features_or_matching_rules_modified':False,'elapsed_seconds':time.perf_counter()-start}
 report=render(d)
 with REPORT.open('x',encoding='utf-8') as f:f.write(report)
 d['report_sha256']=sha(REPORT);lock_write(OUT/'support.json',d)
 print(json.dumps({'status':d['status'],'total':total,'by_city':d['by_city'],'support_sha256':sha(OUT/'support.json'),'report_sha256':sha(REPORT),'elapsed_seconds':d['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True);parser.parse_args();run()
