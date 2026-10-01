"""Prepare a new edition; old decks, scripts and research remain unchanged."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'outputs/v22_spoken_20261001'
def write(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
prepare=(OLD/'prepare_content.py').read_text(encoding='utf-8')
prepare=prepare.replace('/285*60','/275*60').replace('4400<=count<=5000','4050<=count<=4250')
prepare=prepare.replace("'edition':'v22_spoken_20261001'","'edition':'v22_15min_20261001'")
prepare=prepare.replace("'user_requested_change':'讓講解更白話、容易上台說'","'user_requested_change':'15分鐘上下，不要超過太多或是太少', 'target_main_seconds':900, 'duration_basis':'核准稿純漢字數除以250至300字/分鐘，主要規劃以275字/分四捨五入到5秒；目標約15分鐘，仍需真人計時。'")
prepare=prepare.replace("'user_request':'讓講解更白話、容易上台說'","'user_request':'15分鐘上下，不要超過太多或是太少'")
prepare=prepare.replace('# 臺灣房屋總價預測：口語講稿','# 臺灣房屋總價預測：15分鐘口語講稿')
prepare=prepare.replace("'estimated_minutes_at_300':count/300,'estimated_minutes_at_250':count/250,'human_rehearsal_measured':False,","'estimated_minutes_at_300':count/300,'estimated_minutes_at_250':count/250,'estimated_minutes_at_275':count/275,'target_main_seconds':900,'human_rehearsal_measured':False,")
write(HERE/'prepare_content.py',prepare)
for name in ('edit_notes.mjs','finalize_notes.mjs'):
 s=(OLD/'build'/name).read_text(encoding='utf-8').replace('_口語講稿版.pptx','_15分鐘版.pptx')
 write(HERE/'build'/name,s)
write(HERE/'preserve_package.py',(OLD/'preserve_package.py').read_text(encoding='utf-8'))
print(json.dumps({'source_prepared':True}))
