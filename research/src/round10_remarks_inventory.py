"""R10 fit-period-only, target-free remarks inventory; never trains a model.

ZIP bytes inevitably cover the workbook. Raw byte fragments are selected by
row/column/shared-string ID before Unicode or XML value decoding. Only selected
X is decoded; future remarks and prices are never decoded or analyzed.
"""
from __future__ import annotations
import argparse, ast, bisect, csv, hashlib, io, json, platform, re, time, unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd
from house_features import _date_parts
ROOT=Path(__file__).resolve().parents[1]
CUTOFF='2022-08-11'
SOURCE=ROOT/'台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression/train-v2.xlsx'
OUT=ROOT/'outputs/round10/research'
REPORT=ROOT/'reports/round10_remarks_inventory.md'
SELECTED=('交易年月日','備註','建案名稱','鄉鎮市區','土地位置建物門牌')
CITIES=set('基隆市 臺北市 新北市 桃園市 新竹市 新竹縣 苗栗縣 臺中市 彰化縣 南投縣 雲林縣 嘉義市 嘉義縣 臺南市 高雄市 屏東縣 宜蘭縣 花蓮縣 臺東縣 澎湖縣 金門縣 連江縣'.split())
RULES={
 'lease':{'mention':r'租約|租賃|出租|承租|租用|租金|租客|租戶',
  'positive':r'帶租約|附租約|含租約|租賃中|出租中|承租中|現有租約|租約(?:尚)?未到期|有租約|已出租|已承租|出租予|出租於|租予|租給|租用中|承租人|租賃契約',
  'negative':r'無租約|沒有租約|並無租約|未出租|尚未出租|未承租|無人承租|(?:無|沒有|並無|非|不含)(?:.{0,3})租賃',
  'ambiguous':r'預計|擬|可供|得供|可能|如有|若有|倘有|可出租'},
 'possession':{'mention':r'占用|佔用|占有|佔有|點交|交屋|交付|騰空|搬遷',
  'positive':r'不點交|不予點交|未點交|占用|佔用|占有|佔有',
  'negative':r'(?:無|沒有|並無|未遭|未被)(?:.{0,3})(?:占用|佔用|占有|佔有)|已騰空|空屋點交|點交空屋',
  'ambiguous':r'預計|擬|可能|如有|若有|倘有'},
 'rights':{'mention':r'地上權|使用權|承租土地|租用土地|租賃土地|借地|租地|地租|分管|約定專用',
  'positive':r'地上權|使用權|租地建屋|土地承租|承租土地|租用土地|租賃土地',
  'negative':r'非地上權|無地上權|非使用權|不含使用權|沒有使用權',
  'ambiguous':r'預計|擬|可能|如有|若有|倘有'}
}
SAFE_TERMS=tuple(sorted(set('帶租約 附租約 含租約 租賃中 出租中 承租中 現有租約 租約未到期 租約尚未到期 有租約 已出租 已承租 出租予 出租於 租予 租給 租用中 承租人 租賃契約 無租約 沒有租約 並無租約 未出租 尚未出租 未承租 無人承租 非租賃 不含租賃 租約 租賃 出租 承租 租用 租金 租客 租戶 不點交 不予點交 未點交 占用 佔用 占有 佔有 無占用 無佔用 無占有 無佔有 已騰空 空屋點交 點交空屋 現況交屋 現況點交 交屋 點交 交付 騰空 搬遷 地上權 使用權 租地建屋 土地承租 承租土地 租用土地 租賃土地 非地上權 無地上權 非使用權 不含使用權 沒有使用權 借地 租地 地租 分管 約定專用 車位 停車 公設 公共 約定專用 土地 房屋 建物 預計 擬 可能 如有 若有 倘有 可出租'.split()),key=lambda x:(-len(x),x)))


def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def legacy_text(value):
 value='' if value is None else str(value).strip().replace('\u3000','')
 return '' if value in {'','nan','NaN','None','null','<NA>'} else value
def text(value):
 return unicodedata.normalize('NFKC',legacy_text(value))
def write_json(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def xml_fragments(stream,tag):
 """Select raw XML element bytes before any Unicode/XML value decoding."""
 token=tag.encode('ascii');opening=re.compile(b'<'+token+br'(?=[\s/>])');closing=b'</'+token+b'>';buffer=b''
 while True:
  chunk=stream.read(1024*1024)
  if not chunk:break
  buffer+=chunk
  while True:
   match=opening.search(buffer)
   if not match:
    buffer=buffer[-64:];break
   if match.start():buffer=buffer[match.start():]
   finish=buffer.find(b'>')
   if finish<0:break
   if buffer[:finish].rstrip().endswith(b'/'):stop=finish+1
   else:
    stop=buffer.find(closing,finish+1)
    if stop<0:break
    stop+=len(closing)
   fragment=buffer[:stop];buffer=buffer[stop:];yield fragment
  require(len(buffer)<64*1024*1024,'Unexpectedly large XML element')
 require(not opening.search(buffer),'Unterminated XML element')


def display_text(elem):
 # OOXML phonetic rPh is metadata, not displayed text.
 parts=[]
 for child in elem:
  tag=child.tag.rsplit('}',1)[-1]
  if tag=='t':parts.append(child.text or '')
  elif tag=='r':parts.extend(n.text or '' for n in child if n.tag.rsplit('}',1)[-1]=='t')
 return ''.join(parts)


def sheet_cells(z,columns=None,eligible=None,header_only=False):
 """Only selected row+column fragments are decoded as XML/Unicode."""
 values={}
 with z.open('xl/worksheets/sheet1.xml') as stream:
  for fragment in xml_fragments(stream,'row'):
   row_match=re.search(br'\br="([0-9]+)"',fragment[:fragment.find(b'>')]);require(row_match is not None,'Missing row index')
   row=int(row_match.group(1))
   if not (row==1 if header_only else row>1 and (eligible is None or row in eligible)):continue
   current={}
   for cell_bytes in xml_fragments(io.BytesIO(fragment),'c'):
    opening=cell_bytes[:cell_bytes.find(b'>')];coordinate=re.search(br'\br="([A-Z]+)([0-9]+)"',opening);require(coordinate is not None,'Missing cell coordinate')
    col=coordinate.group(1).decode('ascii');require(int(coordinate.group(2))==row,'Cell/row mismatch')
    if columns is not None and col not in columns:continue
    elem=ET.fromstring(cell_bytes)
    require(not any(n.tag.rsplit('}',1)[-1]=='f' for n in elem.iter()),'Selected X formula is unsupported')
    kind=elem.attrib.get('t','n')
    if kind=='inlineStr':parts=[display_text(n) for n in elem if n.tag.rsplit('}',1)[-1]=='is']
    else:parts=[n.text or '' for n in elem if n.tag.rsplit('}',1)[-1]=='v']
    current[col]=(kind,''.join(parts))
   values[row]=current
   if header_only:break
 return values


def shared_strings(z,wanted):
 """Count every si, including empty si; decode only requested si bytes."""
 wanted=set(wanted);found={}
 if not wanted:return found
 with z.open('xl/sharedStrings.xml') as stream:
  for index,fragment in enumerate(xml_fragments(stream,'si')):
   if index not in wanted:continue
   elem=ET.fromstring(fragment)
   found[index]=display_text(elem)
 require(set(found)==wanted,'Missing required shared string')
 return found


def decode_selected(z,rows):
 wanted={int(v) for row in rows.values() for kind,v in row.values() if kind=='s'}
 strings=shared_strings(z,wanted)
 out={r:{c:(strings[int(v)] if kind=='s' else v) for c,(kind,v) in cells.items()} for r,cells in rows.items()}
 return out,len(wanted)


def current_patterns():
 tree=ast.parse((ROOT/'src/house_features.py').read_text(encoding='utf-8-sig'))
 matches=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remark_patterns' for t in n.targets)]
 require(len(matches)==1,'Existing remarks pattern location differs')
 result=ast.literal_eval(matches[0].value);require(len(result)==8,'Existing flag count differs');return result


def classify(value,axis):
 rule=RULES[axis];states=[]
 for clause in re.split(r'[，,。；;！!？?\n\r]+',text(value)):
  if not re.search(rule['mention'],clause):continue
  negative=bool(re.search(rule['negative'],clause));remaining=re.sub(rule['negative'],' ',clause)
  positive=bool(re.search(rule['positive'],remaining))
  uncertain=bool(re.search(rule['ambiguous'],clause))
  if uncertain:states.append('ambiguous')
  elif positive and negative:states.append('conflict')
  elif positive:states.append('affirmed')
  elif negative:states.append('negated')
  else:states.append('ambiguous')
 if not states:return 'not_mentioned'
 if 'conflict' in states or ('affirmed' in states and 'negated' in states):return 'conflict'
 if 'ambiguous' in states:return 'ambiguous'
 return states[0]


def semantic_projection(value):
 """All output tokens are fixed literals; no source substring is exported."""
 value=text(value);positions=[]
 for term in SAFE_TERMS:
  for match in re.finditer(re.escape(term),value):positions.append((match.start(),match.end(),term))
 occupied=set();result=[]
 for start,end,term in sorted(positions,key=lambda v:(v[0],-(v[1]-v[0]))):
  if not any(i in occupied for i in range(start,end)):
   occupied.update(range(start,end));result.append(term)
 require(all(t in SAFE_TERMS for t in result),'Unexpected example token')
 return ' … '.join(result[:14])+' [其他文字全數移除]'


def concentration(values):
 counts=Counter(values);n=sum(counts.values());ranked=counts.most_common()
 return {'rows':n,'distinct':len(counts),'largest_group_rows':ranked[0][1] if ranked else 0,'largest_group_fraction':ranked[0][1]/n if n else None,'top5_fraction':sum(v for k,v in ranked[:5])/n if n else None,'effective_groups_inverse_hhi':1/sum((v/n)**2 for v in counts.values()) if n else None,'top10_hashed_groups':[{'sha256':k,'rows':v} for k,v in ranked[:10]]}


def template_hash(value,project,city,district,address):
 value=text(value)
 for term in [project,address,city,district]:
  if term:value=value.replace(term,'[IDENTIFIER]')
 value=re.sub(r'[0-9零〇○一二三四五六七八九十百千萬億壹貳參肆伍陸柒捌玖拾佰仟]+(?:[.,．][0-9]+)*','[NUMBER]',value)
 value=re.sub(r'\s+','',value)
 return hashlib.sha256(value.encode('utf-8')).hexdigest()


def self_check():
 checks=[]
 def test(name,condition):require(condition,'Synthetic failure: '+name);checks.append(name)
 for axis,value,expected in [('lease','本件帶租約','affirmed'),('lease','本件無租約','negated'),('lease','可出租','ambiguous'),('lease','帶租約；無租約','conflict'),('possession','他人占用不點交','affirmed'),('possession','並無占用','negated'),('possession','現況交屋','ambiguous'),('rights','地上權住宅','affirmed'),('rights','非地上權','negated'),('rights','主要用途為住家、都市土地使用分區','not_mentioned')]:test(axis+':'+value,classify(value,axis)==expected)
 test('empty_unknown',all(classify('',a)=='not_mentioned' for a in RULES))
 test('amount_change_invariant',classify('帶租約，租金123456元','lease')==classify('帶租約，租金999元','lease'))
 test('projection_identifier_and_digits_absent',semantic_projection('王大明0912345678臺北市私人路88號租金999萬元帶租約')=='租金 … 帶租約 [其他文字全數移除]')
 fixture=io.BytesIO()
 with ZipFile(fixture,'w') as z:
  z.writestr('xl/worksheets/sheet1.xml','<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>date</t></is></c></row><row r="2"><c r="A2"><v>1110101</v></c><c r="B2"><v>SECRET_PRICE</v></c><c r="C2" t="s"><v>0</v></c></row><row r="3"><c r="A3"><v>1120101</v></c><c r="C3" t="s"><v>1</v></c></row></sheetData></worksheet>')
  z.writestr('xl/sharedStrings.xml','<sst><si><t>帶租約</t></si><si><t>SECRET_FUTURE_TEXT</t></si></sst>')
 fixture.seek(0)
 with ZipFile(fixture) as z:
  selected=sheet_cells(z,{'A'});test('price_column_not_selected',all(set(v)=={'A'} for v in selected.values()))
  rows=sheet_cells(z,{'C'},{2});decoded,n=decode_selected(z,rows);test('only_eligible_text_reconstructed',decoded=={2:{'C':'帶租約'}} and n==1)
 extra=io.BytesIO()
 with ZipFile(extra,'w') as z:
  z.writestr('xl/sharedStrings.xml',b'<sst><si/><si><r><t>\xe5\xb8\xb6\xe7\xa7\x9f\xe7\xb4\x84</t></r><r><t>&amp;X</t></r></si><si><t>\xffSECRET_FUTURE</t></si></sst>'.decode('unicode_escape').encode('latin1'))
  z.writestr('xl/worksheets/sheet1.xml',b'<worksheet><sheetData><row r="2"><c r="A2" t="inlineStr"><is><t>SAFE&amp;X</t></is></c><c r="B2"><v>\xffSECRET_PRICE</v></c></row></sheetData></worksheet>'.replace(b'\\xff',b'\xff'))
 extra.seek(0)
 with ZipFile(extra) as z:
  test('empty_shared_si_rich_text_escape_and_nonselected_invalid_utf8',shared_strings(z,{0,1})=={0:'',1:'帶租約&X'})
  test('inline_escape_and_price_bytes_never_decoded',sheet_cells(z,{'A'},{2})=={2:{'A':('inlineStr','SAFE&X')}})
 class TinyChunks(io.BytesIO):
  def read(self,size=-1):return super().read(min(size,7) if size>=0 else 7)
 test('fragment_crosses_chunk_boundaries',list(xml_fragments(TinyChunks(b'<sst><si/><si><t>x</t></si></sst>'),'si'))==[b'<si/>',b'<si><t>x</t></si>'])
 test('phonetic_text_not_displayed',display_text(ET.fromstring('<si><r><t>帶</t></r><rPh><t>IGNORE</t></rPh><r><t>租約</t></r></si>'))=='帶租約')
 test('legacy_missing_semantics',legacy_text(' NaN ')=='' and legacy_text('未裝潢')=='未裝潢')
 test('concentration_not_row_independence' ,concentration(['x']*9+['y'])['effective_groups_inverse_hhi']<2)
 return checks


def run(source,output,report):
 require(not output.exists() and not report.exists(),'Refusing to overwrite inventory outputs/report')
 started=time.monotonic();before={Path(__file__).relative_to(ROOT).as_posix():sha(__file__),'src/house_features.py':sha(ROOT/'src/house_features.py'),'reports/round10_remarks_feasibility.md':sha(ROOT/'reports/round10_remarks_feasibility.md')};source_hash=sha(source);checks=self_check()
 print(json.dumps({'event':'date_only_scan_started'},ensure_ascii=False),flush=True)
 with ZipFile(source) as z:
  headers,_=decode_selected(z,sheet_cells(z,header_only=True));columns={v:k for k,v in headers[1].items()};require(all(k in columns for k in SELECTED),'Required X column absent')
  date_col=columns['交易年月日'];tokens=sheet_cells(z,{date_col});dates_raw,date_strings=decode_selected(z,tokens);del tokens
  rows=sorted(dates_raw);require(rows==list(range(2,max(rows)+1)),'Noncontiguous source rows')
  date_values=pd.Series([dates_raw[r].get(date_col,'') for r in rows],index=rows,dtype=object);del dates_raw
  parsed=_date_parts(date_values);valid=parsed.notna() & parsed.dt.year.between(1990,2030);fit=valid & parsed.lt(pd.Timestamp(CUTOFF));fit_rows=parsed.index[fit].astype(int).tolist()
  embedded=[int(r) for r,v in date_values.items() if re.fullmatch(r'(?i)transaction[ a-z_-]*(?:date|day)[ a-z_-]*',text(v))]
  require(len(embedded)==3,'Expected three X-identifiable embedded header rows; do not guess cache mapping')
  require(not set(embedded)&set(fit_rows),'Embedded header entered fit')
  print(json.dumps({'event':'fit_rows_selected','source_rows':len(rows),'fit_rows_no_target_dedup':len(fit_rows),'excluded_embedded_header_rows':embedded},ensure_ascii=False),flush=True)
  del date_values
  allowed={columns[k] for k in SELECTED if k!='交易年月日'}
  x_tokens=sheet_cells(z,allowed,set(fit_rows));selected,text_strings=decode_selected(z,x_tokens);del x_tokens
 print(json.dumps({'event':'fit_text_only_decoded','rows':len(selected),'shared_string_ids_reconstructed':text_strings},ensure_ascii=False),flush=True)
 patterns=current_patterns();records=[];base=Counter();joint=Counter();old_counts=Counter();examples=Counter();city_year=Counter();alignment=[]
 for row in fit_rows:
  values=selected.pop(row);legacy_remark=legacy_text(values.get(columns['備註'],''));remark=text(legacy_remark);project=text(values.get(columns['建案名稱'],''));district=text(values.get(columns['鄉鎮市區'],''));address=text(values.get(columns['土地位置建物門牌'],'')).replace('台','臺');m=re.match(r'^(.{2,3}?[縣市])',address);city=m.group(1) if m and m.group(1) in CITIES else '__UNKNOWN__';year=int(parsed.at[row].year)
  cache_index=row-2-bisect.bisect_left(embedded,row);alignment.append((row,row-2,cache_index));base['rows']+=1;base['remarks_nonempty']+=bool(remark)
  old={k:bool(re.search(pattern,legacy_remark)) for k,pattern in patterns.items()};old_any=any(old.values());base['old_any']+=old_any;base['nonempty_old_unmatched']+=bool(remark) and not old_any
  old_counts.update(k for k,v in old.items() if v)
  project_key=hashlib.sha256((city+'|'+district+'|'+project).encode('utf-8')).hexdigest() if project else None
  template=template_hash(remark,project,city,district,address) if remark else None
  states={axis:classify(remark,axis) for axis in RULES}
  broad=any(s!='not_mentioned' for s in states.values());base['any_new_axis_mention']+=broad;base['new_axis_old_unmatched']+=broad and not old_any
  scope='parking' if re.search('車位|停車',remark) else 'common_area' if re.search('公設|公共|約定專用',remark) else 'land' if re.search('土地|租地|借地',remark) else 'residential_or_unspecified'
  for axis,state in states.items():
   city_year[(axis,state,year,city)]+=1
   if state!='not_mentioned':
    joint[(axis,state,scope,old_any)]+=1;examples[(axis,state,scope,semantic_projection(remark))]+=1
  if remark:records.append({'project':project_key,'template':template,'city':city,'year':year,'old':old,'states':states,'scope':scope})
 del selected
 axis_summary={}
 for axis in RULES:
  hit=[r for r in records if r['states'][axis]!='not_mentioned'];statuses=Counter(r['states'][axis] for r in hit);statuses['not_mentioned']=base['rows']-len(hit)
  axis_summary[axis]={'status_counts':dict(statuses),'mentioned_rows':len(hit),'affirmed_rows':statuses['affirmed'],'old_any_rows':sum(any(r['old'].values()) for r in hit),'old_flag_overlap':{k:sum(r['old'][k] for r in hit) for k in patterns},'named_project_support':concentration([r['project'] for r in hit if r['project']]),'missing_project_rows':sum(r['project'] is None for r in hit),'affirmed_named_project_support':concentration([r['project'] for r in hit if r['project'] and r['states'][axis]=='affirmed']),'affirmed_template_support':concentration([r['template'] for r in hit if r['states'][axis]=='affirmed']),'affirmed_without_parking_common_terms':{'rows':sum(r['states'][axis]=='affirmed' and r['scope'] not in {'parking','common_area'} for r in hit),'named_project_support':concentration([r['project'] for r in hit if r['project'] and r['states'][axis]=='affirmed' and r['scope'] not in {'parking','common_area'}])},'template_support':concentration([r['template'] for r in hit]),'scope_counts':dict(Counter(r['scope'] for r in hit)),'affirmed_scope_counts':dict(Counter(r['scope'] for r in hit if r['states'][axis]=='affirmed')),'year_counts':dict(sorted(Counter(str(r['year']) for r in hit).items())),'city_counts':dict(sorted(Counter(r['city'] for r in hit).items()))}
  require(sum(axis_summary[axis]['status_counts'].values())==base['rows'],'Axis count does not reconcile')
 output.mkdir(parents=True)
 with (output/'fit_row_alignment.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['source_excel_row','raw_body_index','existing_feature_cache_row_index']);w.writerows(alignment)
 with (output/'by_city_year.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['axis','status','year','city','rows']);w.writerows([(*k,v) for k,v in sorted(city_year.items())])
 with (output/'status_scope_overlap.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['axis','status','heuristic_scope','has_old_flag','rows']);w.writerows([(*k,v) for k,v in sorted(joint.items())])
 projected=[]
 for axis in RULES:
  for state in ['affirmed','negated','ambiguous','conflict']:
   matching=[(k,v) for k,v in examples.items() if k[:2]==(axis,state)]
   for k,v in sorted(matching,key=lambda item:(-item[1],item[0]))[:8]:projected.append({'axis':axis,'status':state,'scope':k[2],'semantic_projection':k[3],'rows':v})
 write_json(output/'redacted_semantic_examples.json',{'kind':'allowlisted semantic projections, not verbatim source excerpts','all_other_characters_removed':True,'examples':projected})
 write_json(output/'self_check.json',{'checks':checks,'passed':True,'real_model_training_or_scoring':False})
 source_after=sha(source);require(source_after==source_hash,'Workbook changed during inventory')
 require(all(sha(ROOT/p)==d for p,d in before.items()),'Source code changed during inventory')
 inventory={'created_at_utc':datetime.now(timezone.utc).isoformat(),'kind':'research_only_not_formal_candidate','cutoff_exclusive':CUTOFF,'source_workbook':source.relative_to(ROOT).as_posix(),'source_workbook_sha256':source_hash,'source_code_sha256':before,'selected_columns':list(SELECTED),'price_column_selected':False,'price_values_converted_or_retained':False,'raw_target_cache_read':False,'future_or_valid_or_test_remarks_analyzed':False,'target_dedup_performed':False,'bottom_level_reading':'ZIP decompression and byte-fragment scanning traverse bytes containing all columns and shared strings. Only selected row+column cell fragments enter Unicode/XML decoding; only requested shared-string si IDs enter Unicode/XML decoding. All si elements including empty elements increment storage IDs. Dates are selected for all train rows; remarks/context strings only for valid-date rows strictly before cutoff. No claim that compressed/storage bytes for prices/future text were never traversed.','date_scan':{'source_data_rows':len(rows),'valid_date_rows':int(valid.sum()),'invalid_or_outside_1990_2030_rows':int((~valid).sum()),'nonfit_valid_date_rows':int((valid & ~fit).sum()),'embedded_header_excel_rows':embedded,'shared_date_string_ids':date_strings,'shared_fit_x_string_ids':text_strings},'alignment':{'source_excel_row':'one-based worksheet row; row1 is header','raw_body_index':'source_excel_row minus 2','existing_feature_cache_row_index':'raw_body_index minus count of three date-identifiable embedded header rows preceding it','model_training_rows_equivalent':False,'downstream':'Use saved R6 train_source_cache_indices to select canonical train; apply stage fit/query indices afterwards. full_source_cache_indices acts on canonical train+valid, not original Excel rows.'},'base_counts':dict(base),'old_patterns':patterns,'old_flag_counts':dict(old_counts),'research_rules':RULES,'classification_limit':'Fixed exploratory regex triage, not manually verified legal status or information availability. Ambiguous/conflict are distinct. Parking/common-area scope is whole-remark heuristic, not entity resolution.','axis_summary':axis_summary,'nonempty_named_project_support':concentration([r['project'] for r in records if r['project']]),'nonempty_template_support':concentration([r['template'] for r in records]),'template_definition':'NFKC, selected project/address/city/district literal removal, Arabic/Chinese numeric-character replacement, whitespace removal, SHA256; no raw template exported. Repetitions retained.','outputs_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'elapsed_seconds':time.monotonic()-started,'model_training_or_scoring':False,'formal_candidate_selected':False}
 write_json(output/'inventory.json',inventory);(output/'inventory.sha256').write_text(sha(output/'inventory.json')+'\n',encoding='utf-8')
 lines=['# 第十輪 development fit 備註純 X 盤點','','本次為研究盤點，沒有價格、候選分數、模型訓練或正式選定候選。只探索 train 中有效日期早於 2022-08-11 的文字；沒有用後段、valid 或 test 備註補詞。','','## 計數口徑','',f"原 Excel 有 {len(rows):,} 個資料列；以交易日期辨識並排除 {len(embedded)} 個嵌入英文表頭。日期有效且早於切點者 {base['rows']:,} 列，未作價格去重；因此不等於正式模型 fit 筆數。備註非空 {base['remarks_nonempty']:,} 列，既有八旗標任一命中 {base['old_any']:,} 列，非空但八旗標都未命中 {base['nonempty_old_unmatched']:,} 列。",'',f"三軸任一研究字詞命中 {base['any_new_axis_mention']:,} 列，其中未命中舊八旗標者 {base['new_axis_old_unmatched']:,} 列。廣泛命中不代表肯定條件、可預測性或資訊時點合格。",'','| 語意軸 | 肯定 | 否定 | 含混 | 衝突 | 任一提及 | 具名建案數 | 最大案占比 | 最大模板占比 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for axis,a in axis_summary.items():
  s=a['status_counts'];p=a['named_project_support'];t=a['template_support'];pct=lambda x:'—' if x is None else f'{x:.2%}'
  lines.append(f"| {axis} | {s.get('affirmed',0):,} | {s.get('negated',0):,} | {s.get('ambiguous',0):,} | {s.get('conflict',0):,} | {a['mentioned_rows']:,} | {p['distinct']:,} | {pct(p['largest_group_fraction'])} | {pct(t['largest_group_fraction'])} |")
 lines+=['','最大案占比的分母是具名建案命中列，不含缺案名列；模板占比分母是該軸所有命中列。每案／模板可重複很多次，不能把列數當作獨立支持數。年度、城市、各軸 scope 與舊旗標重疊完整數值保存在 research 聚合檔。','','## 語意與隱私限制','','分類是探索用規則分流，不是人工確認的法律／租用狀態。一般「現況交屋」只有含混提及；「可出租」不認定現有租約；否定先移除再查肯定，矛盾另列。rights 使用權仍可能只指車位／公設，scope 表是整段文字的粗分類，不能當作房屋權利種類。出租金額、人物、電話、地址不進輸出；例句檔只保留事前白名單條件詞，其他字全部刪除，是語意投影而非原文引句。','','原八旗標從 house_features.py AST 讀出，對同一 fit 原備註套用以計算重疊，沒有載入完整特徵快取或價格。特殊關係／持分已實作，不是本輪新增訊號。','','## 底層讀取與重現','','直接掃 XLSX 的 ZIP/XML。底層 ZIP 解壓與 byte-fragment 掃描必須遍歷含其他欄／共享字串的位元組；不能聲稱這些位元組都未讀。先按原列／欄及 shared-string ID 選原始 bytes，只有選定片段才作 Unicode/XML 解碼，價格欄值與非 fit 備註不作解碼、数值轉換或保留。第一 pass 只解碼所有 train 日期；第二 pass 只解碼合格 fit 列的備註與必要 X。只還原所需共享字串 ID，不載入整個字串表或含 y pickle。原始地址只取城市與內部模板正規化，不輸出地址。','','`fit_row_alignment.csv` 保存 Excel row、raw body index、去三個嵌入表頭後的 existing_feature_cache_row_index。沒有重算 529 筆價格相關完全重複。後續可用已凍結 R6 train_source_cache_indices 對接 canonical train，再套每期 fi/qi；full_source_cache_indices 是 canonical train+valid 的去重，不可直接索引 Excel。','','執行：`.venv/Scripts/python.exe -B src/round10_remarks_inventory.py`。拒覆寫既有輸出。`--self-check-only` 只做合成檢查。','','[機器可讀聚合](../outputs/round10/research/inventory.json)、[年度城市分布](../outputs/round10/research/by_city_year.csv)、[語意／scope／舊旗標重疊](../outputs/round10/research/status_scope_overlap.csv)、[去識別語意投影](../outputs/round10/research/redacted_semantic_examples.json)。來源、程式、輸出 SHA 及真實產生時間保存在 inventory。','','## 可行性判斷','','本程式只提供無標籤支持證據，未用自動頻率門檻假裝選定模型。須結合肯定列的案別集中、車位／公設混淆與明確租約／交付條件的覆蓋作研究結論。尚未訓練、評分或凍結，不能把存在未編碼文字當作必然達到 1% 改善。']
 report.write_text('\n'.join(lines).replace('数值','數值')+'\n',encoding='utf-8')
 print(json.dumps({'event':'inventory_complete','base_counts':dict(base),'axes':{k:{'status_counts':v['status_counts'],'projects':v['named_project_support']['distinct'],'largest_project_fraction':v['named_project_support']['largest_group_fraction'],'scope_counts':v['scope_counts']} for k,v in axis_summary.items()},'inventory_sha256':sha(output/'inventory.json'),'report_sha256':sha(report),'script_sha256':sha(__file__)},ensure_ascii=False,indent=2),flush=True)
 return inventory


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--report',type=Path,default=REPORT);p.add_argument('--self-check-only',action='store_true');a=p.parse_args()
 if a.self_check_only:print(json.dumps({'passed':True,'checks':self_check()},ensure_ascii=False));return
 run(a.source.resolve(),a.output.resolve(),a.report.resolve())
if __name__=='__main__':main()
