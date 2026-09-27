"""JSON-only R76/R77 delivery report and current human-document summaries; no selector writes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/round76_77_delivery.md'
HUMAN = ('CURRENT_STATUS.md', 'PRIVATE_SELECTION.md', 'reports/model_selection_explanation.md')
STAGES = ('development', 'confirmation', 'auxiliary')
NAMES = {'development':'開發／中間年度', 'confirmation':'確認／最新年度', 'auxiliary':'輔助／最舊年度'}
PLAN_SHA = '04612c7c862e634c3554331fffe31e64e63400a1c6fd68242b5ecb4aaed194a4'
PINS = {
 'outputs/round76/results.json':'a7ea0219deec6aacaf43be9abd9b62e41b0cd720d09274d6fa81c5fa34d81631',
 'outputs/round77/results.json':'6e9b837a9bfe5918cb5200831ae31268d3ad204aec21430da1b0577de2be8337',
 'outputs/round76/results_review.json':'c3107e796bf96905c5d4374241d74889d22e81578794d4157e045018b8862765',
 'outputs/round77/results_review.json':'90aa860b08de99751dbbe0d00550ab0bbfcf3c40cba2ee21662bbaad914fdf68',
 'outputs/round77/decision.json':'f03b510afa6c73bdc4badfe2d522a4627c8332005bd34c68ca6f985e8ac7e924',
}
MARKER = '<!-- R76_R77_VERIFIED_DELIVERY -->'


def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(release_sha):
 assert not REPORT.exists(), 'No duplicate delivery report'
 inputs = {}
 def capture(n, expected=None):
  p=(ROOT/n).resolve()
  assert p.is_relative_to(ROOT) and p.is_file() and p.suffix in ('.json','.sha256','.md','.py','.mjs')
  assert p.stat().st_size < 30_000_000, 'Small JSON/text only'
  h=sha(p);assert expected is None or h==expected,n
  assert n not in inputs or inputs[n]==h
  inputs[n]=h;return p
 def small(n, expected=None):
  p=capture(n,expected if expected is not None else PINS.get(n))
  receipt=p.with_suffix('.sha256').relative_to(ROOT).as_posix()
  assert capture(receipt).read_text(encoding='ascii').strip()==sha(p),n
  return json.loads(p.read_text(encoding='utf8'))
 hist={n:small(n) for n in PINS}
 r76=hist['outputs/round76/results.json'];r77=hist['outputs/round77/results.json']
 v76=hist['outputs/round76/results_review.json'];v77=hist['outputs/round77/results_review.json']
 decision=hist['outputs/round77/decision.json']
 assert decision['decision']=='advance_two_separately_identified_final_candidates_R76_priority_R77_alternative'
 assert not decision['final_version_chosen'] and decision['private_mae'] is None
 capture('reports/round76_77_final_plan.md',PLAN_SHA)
 release=small('outputs/round76_77/final_export/release.json',release_sha)
 ee=small('outputs/round76_77/final_export/execution.json')
 assert release['status']=='experimental_v20_v21_trials_generated' and set(release['candidates'])=={'76','77'}
 assert ee['exit_code']==0 and ee['failure'] is None
 assert release['plan_sha256']==PLAN_SHA and release['plan_path']=='reports/round76_77_final_plan.md'
 assert release['protected_before']==release['protected_after'] and len(release['protected_before'])==21
 assert release['old_CSV_unchanged'] and not release['selector_updated'] and not release['registry_updated']
 assert release['selector_sha256_before']==release['selector_sha256_after']
 assert not release['uploaded'] and not release['final_version_chosen']
 joint=small(release['joint_vector_evidence_path'],release['joint_vector_evidence_sha256'])
 je=small('outputs/round76_77/final/execution.json')
 area=small('outputs/round77/final_area/results.json')
 ae=small('outputs/round77/final_area/execution.json')
 assert je['exit_code']==ae['exit_code']==0 and je['failure'] is None and ae['failure'] is None
 assert joint['status']=='final_vectors_and_export_payloads_ready_no_CSV'
 assert joint['plan_sha256']==area['plan_sha256']==PLAN_SHA
 assert joint['new_fits_in_this_composer']==joint['tree_NN_model_predict_calls_in_this_composer']==0
 assert joint['separately_completed_new_NN_fits']==area['new_NN_fits']==area['new_fits']==1
 assert area['status']=='completed' and area['fit_rows']==joint['fit_rows']==467221
 assert area['query_rows']==joint['query_rows']==area['reload_rows']==47073 and area['saved_model_reload_exact']
 assert area['numeric_count']==603 and area['seed']==42 and area['epochs']==30
 assert not area['query_labels_used'] and not area['encoder_prior_ramp_or_weights_refit']
 assert area['preparation']['original_eight_arrays_bitwise_unchanged']
 assert not area['preparation']['sample_weights_recomputed'] and not area['preparation']['sample_weights_renormalized']
 assert joint['canonical_frames_sha256']==area['input_frame_sha256'] and joint['old_v19_CSV_exact']
 assert joint['reference_predict_calls']==joint['spatial_predict_calls']==3
 assert joint['reference_or_spatial_refits']==0
 for proof in (release,joint,area):
  for n,h in proof['source_sha256'].items():capture(n,h)
 for number in (76,77):
  item=release['candidates'][str(number)];version=number-56
  assert item['round']==number and item['rows']==47073
  assert item['path']==f'outputs/submission_v{version}_trial.csv'
  for flag in ('positive_int64_full_roundtrip','original_unrounded_prediction_comparison_exact','UTF8_BOM','all_sample_IDs_and_order_exact','exact_headers'):
   assert item[flag],(number,flag)
  cp=small(item['candidate_proof_path'],item['candidate_proof_sha256'])
  small(item['authoring_validation_path'],item['authoring_validation_sha256'])
  assert cp['round']==number and cp['query_rows']==47073 and cp['plan_sha256']==PLAN_SHA
  assert cp['status']=='final_vector_and_export_payload_ready_no_CSV' and not cp['new_CSV']
  assert cp['prediction_path']==item['prediction_path'] and cp['prediction_sha256']==item['prediction_sha256']
  assert cp['export_payload_path']==item['payload_path'] and cp['export_payload_sha256']==item['payload_sha256']
  assert cp['sample_ID_path']==item['sample_ID_path'] and cp['sample_ID_sha256']==item['sample_ID_sha256']
  assert item['prediction_min_TWD']>0 and item['prediction_max_TWD']>=item['prediction_min_TWD']
 for num,main,review in ((76,r76,v76),(77,r77,v77)):
  assert main['checked_stages']==review['checked_stages']==list(STAGES)
  assert main['status']=='three_periods_completed_no_adoption_decision'
  for name in (f'reports/round{num}_diagnostics.json',f'outputs/round{num}/paired_vs_'+('v19' if num==76 else 'R76')+'/results.json'):
   obj=small(name,review['input_sha256'][name])
   assert obj['candidate']==f'R{num}' and obj['status'] in ('completed_saved_predictions_only','completed')
  for s in STAGES:assert main['stages'][s]['metrics']==review['stages'][s]['metrics']
 selector_path=capture('outputs/current_private_candidate.json')
 selection=json.loads(selector_path.read_text(encoding='utf8'))
 # Root publishes the validated local priority first; this script never changes it.
 assert selection['private_priority']==release['candidates']['76']['path']
 assert selection['sha256']==release['candidates']['76']['sha256']
 assert selection['private_mae'] is None and not selection['final_version_chosen']
 assert selection['known_public_best_filename']=='outputs/submission_v18_trial.csv' and selection['known_public_best_mae']==2388386.4613
 original={n:(ROOT/n).read_bytes() for n in HUMAN}
 assert all(MARKER.encode() not in raw for raw in original.values()), 'No duplicate summary'
 report=render(release,r77,v76,v77,area,ae,joint,je,ee,release_sha)
 updates={}
 titles={'CURRENT_STATUS.md':'# 房價預測：v20 優先試交、v21 替代候選',
         'PRIVATE_SELECTION.md':'# Private 選型建議：v20 有限本機優先、v21 替代',
         'reports/model_selection_explanation.md':'# 版本比較：v20／v21 新候選與已知 Public 最佳 v18'}
 for n,body in original.items():
  prefix='' if n.startswith('reports/') else 'reports/'
  csvprefix='../' if n.startswith('reports/') else ''
  intro=summary(release,v77,prefix,csvprefix,area)
  heading=titles[n]+'\n\n'+MARKER+'\n'+intro+'\n\n## 以下為原文歷史紀錄\n\n以下內容逐字保留各輪當時狀態；其中「目前／尚未完成／維持v19」等敘述不覆蓋上方已核驗的最新摘要。\n\n'
  updates[n]=heading.encode('utf8')+body
  assert updates[n].endswith(body)
 assert all(sha(ROOT/n)==h for n,h in inputs.items()), 'Evidence changed while rendering'
 assert all((ROOT/n).read_bytes()==b for n,b in original.items()), 'Human document changed concurrently'
 temps={n:(ROOT/n).with_name(Path(n).name+'.round76_77.tmp') for n in HUMAN}
 assert not any(p.exists() for p in temps.values())
 with REPORT.open('x',encoding='utf8',newline='\n') as f:f.write(report)
 for n,temp in temps.items():
  with temp.open('xb') as f:f.write(updates[n])
 for n,temp in temps.items():os.replace(temp,ROOT/n)
 assert all((ROOT/n).read_bytes().endswith(b) for n,b in original.items())
 assert sha(selector_path)==inputs['outputs/current_private_candidate.json']
 print(json.dumps({'status':'verified_delivery_report_and_human_summaries_written','report_sha256':sha(REPORT),
   'human_documents':{n:{'before_sha256':hashlib.sha256(original[n]).hexdigest(),'after_sha256':sha(ROOT/n),'old_body_preserved_exact':True} for n in HUMAN},
   'selector_updated':False,'array_or_model_loads':0,'new_scores':0,'source_sha256':sha(Path(__file__)),
   'release_sha256':release_sha,'completed_at_utc':datetime.now(timezone.utc).isoformat()},ensure_ascii=False))


def summary(release,v77,prefix,csvprefix,area):
 a,b=release['candidates']['76'],release['candidates']['77'];cf=v77['stages']['confirmation']
 return '\n\n'.join([
  f"兩份新試交檔已在本機完成全部47,073列核驗：**[v20／R76]({csvprefix}{a['path']}) 暫列優先，[v21／R77]({csvprefix}{b['path']}) 為替代候選**。v19與所有舊CSV保留。這是有限歷史證據下的本機順序，不是已選定最終版本；沒有上傳Kaggle或更改平台勾選。",
  f"最新確認年度v20 MAE **1,410,654.84元**，v21 **1,412,874.62元**；v21比v20高 **{cf['differences']['R76']['overall']['delta_TWD']:,.2f}元**，支持test組成高 **{cf['test_composition_differences']['R76']['delta_TWD']:,.2f}元**。v21在開發／最舊年度則分別更低5,451.60／15,846.12元，兩版有年際取捨。",
  '限制不能略過：v20最新期約78%收益來自至少5,000萬元的2,044列；v21相對v20最新新案與≥1億元群退步，最新四個配對區間皆跨零。開發高價尾端與部分較早季度仍有退步；窗口反覆使用、單seed，不是Private改善保證或逐ID修價依據。',
  f"此次只有v21新增1次固定area NN全量訓練（{area['fit_seconds']:.3f}秒），v20沒有新fit。原模型／來源／失敗歷程保留，47,073列保存重載與CSV格式、sample ID順序、正int64、BOM及未取整向量對照皆通過。[完整交付與比較]({prefix}round76_77_delivery.md)。",
  '**v18仍是已回報Public最佳，MAE 2,388,386.4613。** v19／v20／v21尚無Public回報，所有版本Private未知、最終未選。GitHub程式與報告待定版後整理，現在未發布。',
 ])


def render(release,r77,v76,v77,area,ae,joint,je,ee,release_sha):
 fmt=lambda x:f'{x:,.2f}'
 row=lambda a:'| '+' | '.join(str(v) for v in a)+' |'
 a,b=release['candidates']['76'],release['candidates']['77']
 lines=['# R76／R77 交付：v20 優先試交，v21 替代候選','',
  summary(release,v77,'','../',area),'','## 同一歷史窗口比較','',
  '以下是已保存的歷史驗證MAE（元、越低越好），不是此次test或Private的實測誤差。v20對應R76；v21對應R77。',
  '',row(['期別','v19整體／加權','v20整體／加權','v21整體／加權']),row(['---','---:','---:','---:'])]
 for s in STAGES:
  m=r77['stages'][s]['metrics']
  lines.append(row([NAMES[s]]+[fmt(m[k]['overall_mae'])+'／'+fmt(m[k]['weighted_mae']) for k in ('v19','R76','R77')]))
 lines += ['',row(['期別','v20−v19整體','v21−v19整體','v21−v20整體','v21−v20支持test組成']),row(['---']+['---:']*4)]
 for s in STAGES:
  d=v77['stages'][s]
  lines.append(row([NAMES[s],fmt(v76['stages'][s]['differences']['v19']['overall']['delta_TWD']),
   fmt(d['differences']['v19']['overall']['delta_TWD']),fmt(d['differences']['R76']['overall']['delta_TWD']),fmt(d['test_composition_differences']['R76']['delta_TWD'])]))
 lines += ['', 'v20只改單價分支：移除該base中舊總價模型子槽，以保存的R19方法單價模型總價預測占75%、舊unit占25%，再各做一次原reference／spatial。NN與日期加權log分支保持v19，最外仍取三支中位數。v21沿同一新unit，另把NN中的area模型換成固定日期加權訓練版本；其餘架構、603個數值通道、seed42、30epochs、batch2048及融合公式固定。',
  '', '## 收益與風險一起保留','',
  'v20最新期相對v19整體、加權、新舊案、四季、五價帶皆改善；建案與月份配對的雙指標區間全負。但約78%整體收益集中在至少5,000萬元的2,044列。開發期支持test組成退3,082.14元，≥1億元182列MAE退約352.3萬元，Q1退；最舊期中間三價帶及後三季退。單價分支單獨三期都變差，完整median卻改善，不能把單支品質直接當最終品質。',
  'v21相對v20最新期新案+5,202.39元、舊案−1,720.52元，Q1／Q4+5,269.51／+8,327.83元；≥1億元319列MAE再退733,056.36元。最新四個配對區間全跨零。開發四季改善但高價尾端仍退，建案區間跨零、月份區間全負；最舊期新舊、四季、五價帶皆優於v20，兩方法雙指標區間全負。',
  '支持test組成只在歷史有支持的city×新舊案群內正規化；真價帶僅供事後診斷，不能在test按未知答案分流。四格交互是保存MAE的描述性差額，不是因果估計；單seed、重用窗口、多輪選型與未來移轉仍限制結論，配對負向比例不是Private勝率。',
  '', '## 實際訓練、保存重載與交付','',
  f"唯一新fit使用完整467,221列，原R55八個陣列及R63日期權重原樣重用，8列無有效日期者保留既有中性權重1；沒有重建encoder/prior或重選epoch。area fit {area['fit_seconds']:.3f}秒，完整area流程{area['elapsed_seconds']:.3f}秒／outer {ae['elapsed_seconds']:.3f}秒，exit0；全部47,073列保存模型重載逐位核對。",
  f"Composer 0新fit／0樹或NN推論，完整{joint['elapsed_seconds']:.3f}秒／outer {je['elapsed_seconds']:.3f}秒。包含舊R40控制一次、新unit一次、新area-NN分支一次reference/spatial；v19組合及舊CSV控制一致。匯出核驗outer {ee['elapsed_seconds']:.3f}秒，exit0。",
  '每份CSV皆47,073列，sample ID唯一且順序完全相同、兩欄名稱正確、正int64、UTF-8 BOM，對原未取整TWD向量只rint一次。驗證證據確認原20份CSV與registry不变。本報告只讀小JSON與收據，不再讀CSV／陣列／模型，不重新評分或bootstrap。',
  '',row(['版本','檔案SHA256','預測最小／最大TWD']),row(['---','---','---:']),
  row(['v20',a['sha256'],fmt(a['prediction_min_TWD'])+'／'+fmt(a['prediction_max_TWD'])]),
  row(['v21',b['sha256'],fmt(b['prediction_min_TWD'])+'／'+fmt(b['prediction_max_TWD'])]),
  '', '保存的相對v19預測變動僅描述出價差，不是逐筆誤差改善；沒有據此加價帽或逐ID修正。Private未知，兩份都尚無Public回報；v18仍為已回報Public最佳。舊CSV、模型、依賴、完整程式及失敗attempt都保留，GitHub待選定版本後再整理，不在本輪發布。',
  '', '[事前final計畫](round76_77_final_plan.md) · [R76歷史報告](round76_report.md) · [R77四格報告](round77_report.md) · [候選準備決策](../outputs/round77/decision.json)',
  '[area fit證據](../outputs/round77/final_area/results.json) · [composer證據](../outputs/round76_77/final/trial_release.json) · [CSV完整核驗](../outputs/round76_77/final_export/release.json) · [交付報告程式](../src/round76_77_delivery_report.py)',
  f"\n本報告依已完成release `{release_sha}`，僅更新人類說明，不寫selector或registry。", '']
 return '\n'.join(lines)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',action='store_true',required=True)
 parser.add_argument('--release-sha',required=True)
 args=parser.parse_args();run(args.release_sha)
