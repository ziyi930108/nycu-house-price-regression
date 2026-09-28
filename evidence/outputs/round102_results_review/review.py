"""Finite aggregate review; no labels, numeric arrays, models or scoring."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
start=time.monotonic()
r=c.read('outputs/round102/evaluation/results.json')
sources=r['source_sha256'];c.verify_hashes(sources)
c.need(c.protected_hashes()==r['protected_after'],'Protected files changed')
counts={'negative':0,'crossing':0,'positive':0}
intervals=[]
for s in c.STAGES:
 for control in ['v22','C']:
  for family in ['project','month']:
   for key in ['overall','weighted']:
    v=r['stages'][s]['paired'][control]['methods'][family]['summary'][key]
    lo,hi=v['percentile_2_5'],v['percentile_97_5']
    c.need(lo<=hi and v['valid_replicates']==2000 and v['invalid_replicates']==0,'Saved interval contract')
    kind='negative' if hi<0 else 'positive' if lo>0 else 'crossing'
    counts[kind]+=1;intervals.append({'stage':s,'control':control,'family':family,'metric':key,'lower':lo,'upper':hi,'kind':kind})
c.need(counts=={'negative':3,'crossing':17,'positive':4},'All saved CI counts')
c.need(not r['research_gates']['primary']['adoption_research_gate_passed'] and not r['research_gates']['mechanism']['mechanism_gate_passed'],'Gate report differs')
native={s:r['stages'][s]['layer_metrics']['T']['native']['overall_mae']-r['stages'][s]['layer_metrics']['C']['native']['overall_mae'] for s in c.STAGES}
c.need(all(x<0 for x in native.values()),'Native mechanism evidence differs')
execution={p:c.read(p) for p in ['outputs/round102/execution.json','outputs/round102/evaluation/execution.json','outputs/round102_summary/execution.json']}
c.need(all(v['exit_code']==0 and v.get('failure') is None for v in execution.values()),'Incomplete execution')
report="""# R102有限結果核對

2026-09-29。跨作者子代理先只讀source與完成aggregate，回報無矛盾；主線接手將既有結論與有限JSON核對落檔。不是完全獨立審查、不是從零重訓或原query真價重算；沒有載NPY/NPZ/model/y、重算MAE或bootstrap。

- 三stage、六NN及獨立scorer均成功終止，全部保存模型重載exact；source審查另見[紀錄](round102_source_review.md)。
- T相對新zero C原生NN三期改善304,378.87／386,280.18／298,184.14元，native必要條件通過。
- 完整T−C為+4,295.72／−902.66／+6,982.99；完整T−v22為+934.33／+7,219.36／−9,619.90，故不能把原生改善冒稱完整成功。Primary四項皆false；mechanism的native項true而完整overall/weighted兩項false，兩gate均false。
- 24保存區間全部核對端點順序與2000有效／0無效replicates：3全負、17跨零、4全正。未重新抽樣或讀replicate NPY，不是Private勝率。
- aux嘉義市731筆+6.5078%、屏東縣1240筆+5.4725%反證保留；dev/conf沒有本輪固定大群>5%的T−v22違例。
- 最新期2–5千萬元價帶對全窗差額貢獻+6,910.26元、億元+1,900.60，不能只歸咎尾部。
- 最近期在base已翻退、dev在median翻退；層間MAE差不能相加為因果貢獻，不能直接推論移除area NN或R/S。

讀取來源與保存24CI見[小型收據](../outputs/round102_results_review/receipt.json)。核對範圍是來源SHA、九個保護檔、三份terminal metadata和保存aggregate；未重新核全部模型與705 prefixstate。v22唯一提交／Public1,998,568.5706／Private未知，無新CSV或Kaggle操作。
"""
p=ROOT/'reports/round102_results_review.md'
with p.open('x',encoding='utf8',newline='\n') as f:f.write(report)
c.write(Path(__file__).parent/'receipt.json',{'status':'passed_limited_aggregate_review','source':c.record(__file__),
 'review':c.record(p),'input_results':c.record('outputs/round102/evaluation/results.json'),
 'execution_metadata':{p:c.record(p) for p in execution},'source_sha256':sources,'protected_sha256':c.protected_hashes(),
 'interval_counts':counts,'saved_intervals':intervals,'native_T_minus_C':native,'elapsed_seconds':time.monotonic()-start,
 'authors':'Peer read-only feedback; root recorded finite aggregate verification. Not fully independent.',
 'array_y_model_loaded':False,'new_fit_MAE_bootstrap':False,'full_artifact_audit':False})
print(json.dumps({'status':'passed','interval_counts':counts,'elapsed_seconds':time.monotonic()-start}))
