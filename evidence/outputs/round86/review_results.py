"""Summarize fixed R86 JSON evidence only. No model loading, training or price generation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STAGES=("development","confirmation","auxiliary")
LABELS={"development":"開發／中間","confirmation":"確認／最新","auxiliary":"輔助／最舊"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.perf_counter();inputs={}
 def read(name):
  p=ROOT/name;h=sha(p);inputs[name]=h
  side=p.with_suffix(".sha256")
  if side.exists():
   assert side.read_text("utf8").strip()==h,name
   inputs[side.relative_to(ROOT).as_posix()]=sha(side)
  return json.loads(p.read_text("utf8"))
 result=read("outputs/round86/results.json")
 control=read("outputs/round79/results.json")
 diag=read("reports/round86_diagnostics.json")
 paired={c:read(f"outputs/round86/paired_vs_{c}/results.json") for c in ("v22","v24")}
 execution=read("outputs/round86/execution.json")
 diag_exec=read("outputs/round86/diagnostics/execution.json")
 plan=read("outputs/round86/plan.json")
 metadata={s:read(f"outputs/round86/{s}/model/metadata.json") for s in STAGES}
 priors={s:read(f"outputs/round86/{s}/prior_verification.json") for s in STAGES}
 fit_exec={s:read(f"outputs/round86/{s}/model/execution.json") for s in STAGES}
 assert execution["exit_code"]==diag_exec["exit_code"]==0
 assert result["checked_stages"]==list(STAGES) and result["new_fits"]==3
 assert result["protected_before"]==result["protected_after"]
 assert not any(result[k] for k in ["automatic_adoption","new_CSV","new_test_predictions","new_NN_fits","new_reference_or_spatial_states","Public_feedback_used_for_tuning"])
 assert result["query_labels_used_only_for_evaluation"]
 for s in STAGES:
  v=result["stages"][s];m=metadata[s]
  prior=priors[s]
  assert all(prior[k] for k in ["fit_only_OOF_rebuild_exact","full_fit_state_exact","query_prediction_exact","R63_factor_values_exact"])
  assert not prior["chronological_oof"] and not prior["query_labels_used"]
  assert m["added_feature_names"]==["log_recency_oof_unit_prior"]
  assert not m["prior_provenance_verified_in_helper"] and m["prior_uses_fit_labels"]
  assert v["controls_full_metrics_exact"] and v["v22_base_reference_spatial_median_full_vectors_exact"]
  assert v["full_query_raw_TWD_reload_exact"] and v["date_weight_provenance_verified"]
  assert diag["stages"][s]["all_full_saved_metrics_exact"]
  assert m["reload_TWD_exact"] and m["reload_raw_exact"] and m["reload_all_query_rows_verified"]
  assert m["actual_trees"]==2999 and fit_exec[s]["exit_code"]==0
  assert m["feature_count"]==96 and m["categorical_feature_count"]==23
  for k in ["fit_uses_eval_set","early_stopping_used","date_weights_recomputed","date_weights_renormalized","query_labels_used"]:
   assert not m[k],(s,k)
 intervals=[];tail={};summary={};cities={}
 for c,p in paired.items():
  assert p["status"]=="completed" and p["replicates"]==2000 and p["seed"]==20260925
  for s in STAGES:
   assert p["stages"][s]["point_metrics_reproduced"]
   for method in ("project","month"):
    for metric in ("overall","weighted"):
     v=p["stages"][s][method]["summary"][metric];lo=v["percentile_2_5"];hi=v["percentile_97_5"]
     assert v["valid_replicates"]==2000 and v["invalid_replicates"]==0
     intervals.append({"control":c,"stage":s,"method":method,"metric":metric,"point_delta":v["point_delta"],"lower":lo,"upper":hi,
       "classification":"entirely_positive" if lo>0 else "entirely_negative" if hi<0 else "crosses_zero",
       "valid_replicates":v["valid_replicates"],"invalid_replicates":v["invalid_replicates"]})
 for s in STAGES:
  v=result["stages"][s];n=v["query_rows"];bands=diag["stages"][s]["price_bands"]
  tail[s]={}
  for c in ("v22","v24"):
   contributions={b["band"]:(b["R86"]-b[c])*b["rows"]/n for b in bands}
   delta=v["metrics"]["R86"]["overall_mae"]-v["metrics"][c]["overall_mae"]
   assert abs(sum(contributions.values())-delta)<1e-7
   tail[s][c]={"band_contributions_TWD":contributions,"all_bands_sum":sum(contributions.values()),
     "ge_100m_contribution":contributions[">=100m"],"all_other_bands_contribution":sum(x for k,x in contributions.items() if k!=">=100m")}
  summary[s]={"query_rows":n,"fit_rows":v["fit_rows"],"fit_seconds":v["fit_seconds"],"period_seconds":v["elapsed_seconds"],
    "overall_mae":v["metrics"]["R86"]["overall_mae"],"weighted_mae":v["metrics"]["R86"]["weighted_mae"],
    "vs_v22":v["differences"]["v22"],"vs_v24":v["differences"]["v24"],
    "testmix_delta_vs_v22":diag["stages"][s]["test_composition"]["models"]["R86"]-diag["stages"][s]["test_composition"]["models"]["v22"],
    "unit_layer_differences_vs_v22":{k:v["unit_branch_metrics"][k]["overall_mae"]-control["stages"][s]["unit_branch_metrics"][k]["overall_mae"] for k in ["native","base","reference","spatial"]}}
  cities[s]={city:{"rows":g["rows"],"candidate_mae":g["mae"],"v22_mae":v["metrics"]["v22"]["cities"][city]["mae"],
        "delta_vs_v22":g["mae"]-v["metrics"]["v22"]["cities"][city]["mae"],
        "delta_vs_v24":g["mae"]-v["metrics"]["v24"]["cities"][city]["mae"]} for city,g in v["metrics"]["R86"]["cities"].items()}
 count={c:{cl:sum(i["control"]==c and i["classification"]==cl for i in intervals) for cl in ["entirely_positive","crosses_zero","entirely_negative"]} for c in paired}
 f=lambda x:f"{x:,.2f}"
 sign=lambda x:f"{x:+,.2f}"
 lines=["# R86 單價CatBoost加入分組OOF價格先驗：結果與採用審查","",
 "日期：2026-09-28。建議保留研究，不進全量 fit、不新增提交 CSV。這份文件提出研究審查建議，沒有更動 selector、使用者選定的 v22 或 Kaggle。Private 未知。","",
 "最新年度完整median的MAE比v22改善1,051.01元（0.0760%），固定加權改善1,682.07元；但四個配對區間均跨零。中間／最舊年度整體MAE各退步32,262.95／12,956.06元，對v24三期皆退步。5,000萬至1億元价帶三期均改善，是必須保留的有利證據，但目前不足支持新的全量fit。","",
 "## 固定假設與流程","",
 "目標與權重沿R79：有效面積A、target=y/(A×1e4)、Pool權重=A/100×保存日期權重。只加一個數值欄位log_recency_oof_unit_prior=log1p(prior_TWD_per_m2/1e4)，共96欄／23類別，其餘GPU CatBoost MAE、2999樹、depth8、learning_rate .065、seed42不變。log1p只作用先驗特徵，不是log回歸目標。三期無query eval_set或early stopping。","",
 "訓練欄位來自同窗R60五折建案分組OOF，held fold只使用其他fold的fit標籤和日期；query由完整fit prior state提供，不讀query標籤。這不是時間OOF，不能聲稱訓練交易只看自己之前的成交。之前NN已有相關先驗，本輪把訊息新增到單價CatBoost，可能改善地區尺度，也可能與類別統計或NN重複。","",
 "新 native 還原為元後，先與既有 LGB 以 .75/.25 混合，再用同一期保存 reference/spatial狀態重算此分支，最後與固定NN、log分支取逐筆median。只有三次新CatBoost fit，沒有新NN／reference／spatial state fit，沒有test預測。","",
 "## 完整 median 結果","",
 "| 期別 | query列數 | v22 MAE | v24 MAE | R86 MAE | R86−v22 | R86−v24 |",
 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s];m=v["metrics"]
  lines.append(f"| {LABELS[s]} | {v['query_rows']:,} | {f(m['v22']['overall_mae'])} | {f(m['v24']['overall_mae'])} | {f(m['R86']['overall_mae'])} | {sign(summary[s]['vs_v22']['overall']['delta_TWD'])} | {sign(summary[s]['vs_v24']['overall']['delta_TWD'])} |")
 lines+=["","| 期別 | v22固定加權 | v24固定加權 | R86固定加權 | 對v22變化 | 對v24變化 |","| --- | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  m=result["stages"][s]["metrics"];lines.append(f"| {LABELS[s]} | {f(m['v22']['weighted_mae'])} | {f(m['v24']['weighted_mae'])} | {f(m['R86']['weighted_mae'])} | {sign(m['R86']['weighted_mae']-m['v22']['weighted_mae'])} | {sign(m['R86']['weighted_mae']-m['v24']['weighted_mae'])} |")
 lines+=["","固定加權沿用新案／已見建案組成。最新年度點估計略有利，但中間、最舊不利；對v24三期整體與固定加權皆退步。R86在最新／最舊仍優於v11/v12/v5；中間期甚至略差於v11，不能用較弱基準取代主要對照。三期交易不同，不是逐年進步曲線。","",
 "## 原生模型與完整組合必須分開看","",
 "| 期別 | 層 | v22對應單價分支MAE | R86 MAE | 差 |","| --- | --- | ---: | ---: | ---: |"]
 for s in STAGES:
  for k in ["native","base","reference","spatial"]:
   a=control["stages"][s]["unit_branch_metrics"][k]["overall_mae"];b=result["stages"][s]["unit_branch_metrics"][k]["overall_mae"]
   lines.append(f"| {LABELS[s]} | {k} | {f(a)} | {f(b)} | {sign(b-a)} |")
 lines+=["",
 "最新期native改善40,227.14元，reference/spatial後單分支仍改善23,466.99元，但完整median只改善1,051.01元。逐筆median依分支相對排序選中間值，單支改善不能直接當成整體收益。本輪沒有搜尋融合比例。",
 "中間期從native、base、reference、spatial到median都退步。最舊期native改善1,848.73元，與LGB混合後轉差，reference/spatial與median亦退步。新增先驗不保證維持分支互補性；此差異尚不能隔離成單一欄位的因果效果。","",
 "## 五個真價帶與尾部的整體貢獻","",
 "| 期別 | 真價帶 | 列數 | v22 MAE | R86 MAE | 差vs v22 | 差vs v24 | 對全期MAE差貢獻vs v22 |",
 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  for b in diag["stages"][s]["price_bands"]:
   lines.append(f"| {LABELS[s]} | {b['band']} | {b['rows']:,} | {f(b['v22'])} | {f(b['R86'])} | {sign(b['R86']-b['v22'])} | {sign(b['R86']-b['v24'])} | {sign(tail[s]['v22']['band_contributions_TWD'][b['band']])} |")
 lines+=["","| 期別 | ≥1億貢獻 | 其餘價帶合計貢獻 | 全期MAE差 |","| --- | ---: | ---: | ---: |"]
 for s in STAGES:
  g=tail[s]["v22"];lines.append(f"| {LABELS[s]} | {sign(g['ge_100m_contribution'])} | {sign(g['all_other_bands_contribution'])} | {sign(g['all_bands_sum'])} |")
 lines+=["",
 "最強有利反證：5,000萬至1億元價帶三期均改善，802／1,725／1,382列的群MAE各下降323,693.54／313,510.09／173,716.87元；最舊期≥1億元的171列也改善2,513,129.45元，故不能說尾部全都退步。中間期≥1億元182列卻退步13,243,454.07元，對全期差貢獻+29,671.18元，幾乎涵蓋整體+32,262.95元退步。最新期20–50m及50–100m改善，被較低價帶與≥100m退步抵銷，淨收益僅−1,051.01元。最舊期反而<50m三帶退步、尾部有利，顯示跨期不穩定。",
 "有號貢獻＝群MAE差×群列數／全期列數；它不是平均prediction−truth。五帶合計精確回到整體MAE差。真價分群是事後診斷，不能用未知test真價把低價列切給R86、高價列切回v22。","",
 "## 新舊建案與季度","",
 "| 期別 | 建案組 | 列數 | R86 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s]
  for k,g in v["metrics"]["R86"]["strata"].items():
   if not g["rows"]:continue
   lines.append(f"| {LABELS[s]} | {k} | {g['rows']:,} | {f(g['mae'])} | {sign(v['differences']['v22']['strata'][k]['delta_TWD'])} | {sign(v['differences']['v24']['strata'][k]['delta_TWD'])} |")
 lines+=["","三期missing建案群均0列。中間／最舊已見建案小幅改善，新建案退步；最新期新建案改善5,256.03元、已見案退步4,504.18元，沒有跨年度一致有利的群組。","",
 "| 期別 | 預測季度 | 列數 | R86 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s]
  for k,g in v["metrics"]["R86"]["forecast_quarters"].items():
   lines.append(f"| {LABELS[s]} | {k} | {g['rows']:,} | {f(g['mae'])} | {sign(v['differences']['v22']['forecast_quarters'][k]['delta_TWD'])} | {sign(v['differences']['v24']['forecast_quarters'][k]['delta_TWD'])} |")
 lines+=["","最新期Q2、Q3與最舊期Q1改善，其他九季度退步；中間期Q1的17,608列增加101,379.08元。季度沿query起點的預測距離定義，不是新的calendar切分。","",
 "## 全部城市，保留改善與退步","",
 "| 期別 | 城市 | 列數 | R86 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  for city,g in cities[s].items():
   lines.append(f"| {LABELS[s]} | {city} | {g['rows']:,} | {f(g['candidate_mae'])} | {sign(g['delta_vs_v22'])} | {sign(g['delta_vs_v24'])} |")
 lines+=["",
 "有利城市也具規模：最舊期新北17,191列改善50,795.68元；最新期高雄16,443列改善22,561.66元、臺北4,957列改善71,302.08元、苗栗4,001列改善48,614.75元。中間期基隆737列改善409,513.41元。另有城市退步需看全表，澎湖／金門小樣本不可單独當成穩健收益。",
 "上述列舉不替代全表，也沒有据此按城市挑模型。各城市含不同建案、時間及價带，描述性分群不能證明某欄位改動的因果效果。","",
 "## 固定test組成估計","",
 "| 期別 | 支持質量 | 未支持test列 | v22估計 | v24估計 | R86估計 | 差vs v22 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  q=diag["stages"][s]["test_composition"];m=q["models"]
  lines.append(f"| {LABELS[s]} | {q['test_mass_covered']:.4%} | {q['unsupported_test_rows']:,} | {f(m['v22'])} | {f(m['v24'])} | {f(m['R86'])} | {sign(m['R86']-m['v22'])} |")
 lines+=["","沿用R49固定city×seen/new test-X權重，歷史cell至少200列，僅有支持質量重正規化。最新期對v22改善3,363.18元，中間／最舊退步76,651.28／4,467.65元；對v24三期皆較差。這不是test／Private真實MAE。","",
 "## 全部24個配對區間","",
 "固定建案群集與三個月循環區塊，每項2,000次、seed20260925。差值為R86−對照；所有區間valid=2000、invalid=0。以下一併呈現，不以某一抽法代替其餘。","",
 "| 對照 | 期別 | 抽法 | 指標 | 點差 | 2.5% | 97.5% | 區間 |","| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
 for i in intervals:
  cl={"entirely_positive":"全正","crosses_zero":"跨零","entirely_negative":"全負"}[i["classification"]]
  lines.append(f"| {i['control']} | {LABELS[i['stage']]} | {i['method']} | {i['metric']} | {sign(i['point_delta'])} | {sign(i['lower'])} | {sign(i['upper'])} | {cl} |")
 lines+=["",
 "相對v22：最新期4區間全部跨零，沒有支持小幅改善已穩健成立；最舊期4項也跨零；中間期project兩項跨零、month兩項全正。相對v24：中間與最新各為project兩項跨零、month兩項全正；最舊4項全正。24項沒有全負區間，但不能說已證明每個情況都較差。",
 "這些區間以反覆查看的窗口、固定保存模型為條件，未涵蓋選型偏差、重訓隨機性或未來市場移轉；不是Private勝率、p值或新盲測。","",
 "## 真實執行、保存與審查範圍","",
 "| 期別 | fit/query列 | fit秒 | 全期秒 | 保存後全query重載 |","| --- | --- | ---: | ---: | --- |"]
 for s in STAGES:
  v=result["stages"][s]
  lines.append(f"| {LABELS[s]} | {v['fit_rows']:,}/{v['query_rows']:,} | {v['fit_seconds']:.3f} | {v['elapsed_seconds']:.3f} | raw與TWD位元相同 |")
 prior_sum=sum(priors[s]["elapsed_seconds"] for s in STAGES)
 lines += ["",f"三次fit-only OOF重播合計{prior_sum:.3f}秒，已含主流程耗時，不能重複加成GPU訓練时间。"]
 fit_sum=sum(result["stages"][s]["fit_seconds"] for s in STAGES)
 lines+=["",
 f"三次fit合計{fit_sum:.3f}秒；主流程內層{result['elapsed_seconds']:.3f}秒、外層{execution['elapsed_seconds']:.3f}秒。generic診斷內層{diag['elapsed_seconds']:.3f}秒、外層{diag_exec['elapsed_seconds']:.3f}秒；paired vs v22內層{paired['v22']['elapsed_seconds']:.3f}秒，vs v24內層{paired['v24']['elapsed_seconds']:.3f}秒。未把聊天等待時間列為訓練時間。",
 "三個fit、主流程與generic execution皆exit0；paired JSON皆completed。日期權重、同切分索引／特徵、對照完整metrics與v22各層向量重播通過，原保護清單前後相同。模型保存CBM與96欄state後重載全query，raw與TWD位元相同；這是主程式證據，本審查沒有另載模型或產生價格。",
 "三期GPU訓練前以fit資料重新建構OOF先驗，OOF向量、fold、fullfit state、query prior及R63 float32 price factor全部核對相同。helper metadata的prior_provenance_verified_in_helper=False表示helper不負責來源驗證；驗證在driver完成，不能把此欄解讀成洩漏失敗或直接改True。本審查讀三份prior_verification及sidecar SHA，沒有自行重建先驗。",
 "本審查只讀JSON並核sidecar SHA與加總，沒有重跑模型／價格／bootstrap。GPU固定seed不保證位元決定性，v22未同期重訓，最新期僅0.0760%改善尤其不能當純因果證據。歷史窗口反覆選型，只測一組配置，不能推論OOF先驗永遠沒有價值。",
 "",
 "## 建議與下一步","",
 "建議保留R86研究，不做final fit或提交CSV。最新期收益微小且區間跨零，中間／最舊有點估計代價，對v24三期皆退步，目前不足支持額外全量fit。這不是要求每一期都顯著勝出；5,000萬–1億元、若干城市与建案群的局部改善已保存。不得按query真價或最佳群直接拼接模型。",
 "使用者選定v22保持不變，Private仍未知。後續新候選應另立固定假設與驗證計畫，保留這次反證；本文件不更改任何採用登錄。",
 "",
 "[固定計畫](round86_plan.md) · [主結果](../outputs/round86/results.json) · [generic完整診斷](round86_diagnostics.json) · [vs v22](round86_paired_vs_v22.md) · [vs v24](round86_paired_vs_v24.md)",
 "[決策審查JSON](../outputs/round86/decision_review.json) · [可重跑審查程式](../outputs/round86/review_results.py) · [前置誤差盤點](r85_error_audit.md)",""]
 review={"review_type":"recommendation_only_not_selection_mutation","timestamp_utc":datetime.now(timezone.utc).isoformat(),
  "round":86,"candidate":"OOF_prior_feature_in_unit_CatBoost","recommendation":"retain_research_do_not_final_fit_or_export",
  "reason":"Tiny confirmation benefit with all paired intervals crossing zero; development and auxiliary worsen; all periods worse than v24. Some price-band and city gains are preserved.",
  "selector_changed":False,"csv_generated":False,"private_score":None,"new_fits_in_review":0,"prices_recomputed_in_review":False,
  "stage_summary":summary,"true_price_tail_decomposition":tail,"paired_interval_counts":count,"all_24_intervals":intervals,
  "favorable_evidence":["Confirmation native improves 40,227.14 TWD, spatial improves 23,466.99 TWD, final median improves only 1,051.01.",
    "50-100m TWD band improves in all three periods: 802, 1725 and 1382 rows.",
    "Auxiliary >=100m improves 2,513,129.45 TWD on 171 rows while lower price bands worsen.",
    "Several sizeable cities and confirmation Q2/Q3 plus auxiliary Q1 improve."],
  "OOF_prior_replay_evidence":priors,"prior_replay_seconds_sum":sum(p["elapsed_seconds"] for p in priors.values()),
  "complete_city_comparison":cities,
  "fit_seconds_sum":fit_sum,"main_inner_seconds":result["elapsed_seconds"],"main_outer_seconds":execution["elapsed_seconds"],
  "limitations":["Repeatedly inspected historical windows, not blind","No same-run v22 retrain; GPU randomness not isolated",
    "Bootstrap not corrected for selection or future shift; not Private probability","True-price groups cannot be inference routing",
    "This review does not rescore per-row area/type candidate groups or reload models","Grouped project OOF is not temporal OOF"],
  "input_sha256":inputs}
 source=Path(__file__);inputs[source.relative_to(ROOT).as_posix()]=sha(source)
 assert all(sha(ROOT/n)==h for n,h in inputs.items())
 review["inputs_unchanged"]=True
 review["review_elapsed_seconds"]=time.perf_counter()-t
 report_path=ROOT/"reports/round86_report.md"
 with report_path.open("x",encoding="utf8",newline="\n") as fmd:fmd.write("\n".join(lines))
 review["report_sha256"]=sha(report_path)
 with (OUT/"decision_review.json").open("x",encoding="utf8") as fjson:json.dump(review,fjson,ensure_ascii=False,indent=2,allow_nan=False)
 with (OUT/"decision_review.sha256").open("x",encoding="utf8") as fsha:fsha.write(sha(OUT/"decision_review.json")+"\n")
 print(json.dumps({"report":"reports/round86_report.md","review":"outputs/round86/decision_review.json","fit_seconds_sum":fit_sum,"interval_counts":count,"input_files":len(inputs)}))
if __name__=="__main__":main()
