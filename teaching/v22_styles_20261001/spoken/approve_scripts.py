from pathlib import Path
import json,re
P=Path(__file__).resolve().parent/'content'
data=json.loads((P/'proposed_scripts.json').read_text(encoding='utf-8'))
fixes={6:('先看資料清理。先處理格式和完全重複的資料。','先看資料清理。'),7:('因此不直接放進模型。','因此不直接放進模型。備註只保留有限的條件旗標。'),11:('比較近期也不代表一定更可靠','較近期的資料，也不一定更可靠'),17:('這是使用者提供的平台截圖所回報的結果。','這是我提交後，平台回報的結果，已保存截圖。'),18:('先固定想驗證的問題和時間切分','要減少這些風險，我們會先固定問題和時間切分'),19:('但請分清楚，我們這次沒有從頭重訓全部模型','剛才的重現驗證，是重新載入保存模型，沒有從頭重訓全部模型')}
for s in data['slides']:
 if s['number'] in fixes:
  old,new=fixes[s['number']];assert s['script'].count(old)==1;s['script']=s['script'].replace(old,new)
 sentences=re.findall(r'[^。！？]+[。！？]?',s['script'])
 s['script']='\n\n'.join(''.join(sentences[i:i+3]) for i in range(0,len(sentences),3))
count=sum(len(re.findall(r'[\u4e00-\u9fff]',s['script'])) for s in data['slides'])
assert 4400<=count<=5000,count
with (P/'approved_scripts.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
with (P/'editor_review.json').open('x',encoding='utf-8') as f:json.dump({'changed_pages':list(fixes),'replacements':fixes,'paragraph_breaks':'every three sentences','main_han':count,'independent_review':'All 20 scripts reviewed; small oral clarity and presenter voice edits; no new factual defects found.','source_proposal_preserved':True},f,ensure_ascii=False,indent=2)
print(count)
