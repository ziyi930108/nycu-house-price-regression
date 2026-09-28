"""Summarize fixed R85 JSON evidence only. No model loading, training or price generation."""
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
 result=read("outputs/round85/results.json")
 control=read("outputs/round79/results.json")
 diag=read("reports/round85_diagnostics.json")
 paired={c:read(f"outputs/round85/paired_vs_{c}/results.json") for c in ("v22","v24")}
 execution=read("outputs/round85/execution.json")
 diag_exec=read("outputs/round85/diagnostics/execution.json")
 plan=read("outputs/round85/plan.json")
 metadata={s:read(f"outputs/round85/{s}/model/metadata.json") for s in STAGES}
 fit_exec={s:read(f"outputs/round85/{s}/model/execution.json") for s in STAGES}
 assert execution["exit_code"]==diag_exec["exit_code"]==0
 assert result["checked_stages"]==list(STAGES) and result["new_fits"]==3
 assert result["protected_before"]==result["protected_after"]
 assert not any(result[k] for k in ["automatic_adoption","new_CSV","new_test_predictions","new_NN_fits","new_reference_or_spatial_states","Public_feedback_used_for_tuning"])
 assert result["query_labels_used_only_for_evaluation"]
 for s in STAGES:
  v=result["stages"][s];m=metadata[s]
  assert v["controls_full_metrics_exact"] and v["v22_base_reference_spatial_median_full_vectors_exact"]
  assert v["full_query_raw_TWD_reload_exact"] and v["date_weight_provenance_verified"]
  assert diag["stages"][s]["all_full_saved_metrics_exact"]
  assert m["reload_TWD_exact"] and m["reload_raw_exact"] and m["reload_all_query_rows_verified"]
  assert m["actual_trees"]==2999 and m["fixed_area_exponent"]==0.5 and fit_exec[s]["exit_code"]==0
  assert m["feature_count"]==95 and m["categorical_feature_count"]==23
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
   contributions={b["band"]:(b["R85"]-b[c])*b["rows"]/n for b in bands}
   delta=v["metrics"]["R85"]["overall_mae"]-v["metrics"][c]["overall_mae"]
   assert abs(sum(contributions.values())-delta)<1e-7
   tail[s][c]={"band_contributions_TWD":contributions,"all_bands_sum":sum(contributions.values()),
     "ge_100m_contribution":contributions[">=100m"],"all_other_bands_contribution":sum(x for k,x in contributions.items() if k!=">=100m")}
  summary[s]={"query_rows":n,"fit_rows":v["fit_rows"],"fit_seconds":v["fit_seconds"],"period_seconds":v["elapsed_seconds"],
    "overall_mae":v["metrics"]["R85"]["overall_mae"],"weighted_mae":v["metrics"]["R85"]["weighted_mae"],
    "vs_v22":v["differences"]["v22"],"vs_v24":v["differences"]["v24"],
    "testmix_delta_vs_v22":diag["stages"][s]["test_composition"]["models"]["R85"]-diag["stages"][s]["test_composition"]["models"]["v22"],
    "unit_layer_differences_vs_v22":{k:v["unit_branch_metrics"][k]["overall_mae"]-control["stages"][s]["unit_branch_metrics"][k]["overall_mae"] for k in ["native","base","reference","spatial"]}}
  cities[s]={city:{"rows":g["rows"],"candidate_mae":g["mae"],"v22_mae":v["metrics"]["v22"]["cities"][city]["mae"],
        "delta_vs_v22":g["mae"]-v["metrics"]["v22"]["cities"][city]["mae"],
        "delta_vs_v24":g["mae"]-v["metrics"]["v24"]["cities"][city]["mae"]} for city,g in v["metrics"]["R85"]["cities"].items()}
 count={c:{cl:sum(i["control"]==c and i["classification"]==cl for i in intervals) for cl in ["entirely_positive","crosses_zero","entirely_negative"]} for c in paired}
 f=lambda x:f"{x:,.2f}"
 sign=lambda x:f"{x:+,.2f}"
 lines=["# R85 平方根面積尺度回歸：結果與採用審查","",
 "日期：2026-09-28。建議保留研究，不進全量 fit、不新增提交 CSV。這份文件提出研究審查建議，沒有更動 selector、使用者選定的 v22 或 Kaggle。Private 未知。","",
 "三期完整組合相對 v22 的整體與固定加權 MAE 皆退步，但不是每個模型層或分群都退步：中間期原生單支與修正分支改善；中間、最新期低於5,000萬元的三個價帶也改善。高價尾部抵銷了這些收益，完整 median 沒有繼承單支的平均改善。","",
 "## 固定假設與流程","",
 "有效面積 A 與 R79 相同。F=sqrt(A/100)，target=(y/1e6)/F，Pool權重=F×原保存日期權重，逆轉換=max(z×F×1e6,0)。相較舊 F=A/100，這是總價與單價之間的固定表示；面積冪次沒有搜尋。95欄／23類別、GPU CatBoost MAE、2999樹、depth8、learning_rate .065、seed42及其餘參數保留。三期均未使用query eval_set或early stopping。","",
 "未截零前，F×d×|y/(1e6×F)−z|=d×|y−1e6×F×z|/1e6，仍對應日期加權總價MAE；權重正規化、正則與函數表示卻不等同，因此有必要實測。","",
 "新 native 還原為元後，先與既有 LGB 以 .75/.25 混合，再用同一期保存 reference/spatial狀態重算此分支，最後與固定NN、log分支取逐筆median。只有三次新CatBoost fit，沒有新NN／reference／spatial state fit，沒有test預測。","",
 "## 完整 median 結果","",
 "| 期別 | query列數 | v22 MAE | v24 MAE | R85 MAE | R85−v22 | R85−v24 |",
 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s];m=v["metrics"]
  lines.append(f"| {LABELS[s]} | {v['query_rows']:,} | {f(m['v22']['overall_mae'])} | {f(m['v24']['overall_mae'])} | {f(m['R85']['overall_mae'])} | {sign(summary[s]['vs_v22']['overall']['delta_TWD'])} | {sign(summary[s]['vs_v24']['overall']['delta_TWD'])} |")
 lines+=["","| 期別 | v22固定加權 | v24固定加權 | R85固定加權 | 對v22變化 | 對v24變化 |","| --- | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  m=result["stages"][s]["metrics"];lines.append(f"| {LABELS[s]} | {f(m['v22']['weighted_mae'])} | {f(m['v24']['weighted_mae'])} | {f(m['R85']['weighted_mae'])} | {sign(m['R85']['weighted_mae']-m['v22']['weighted_mae'])} | {sign(m['R85']['weighted_mae']-m['v24']['weighted_mae'])} |")
 lines+=["","固定加權沿用新案／已見建案組成；三期評分的是不同交易，不能把表格視為逐年模型改進。R85仍全面優於較早v11、v12、v5的整體MAE，但本輪基準是已完成的v22與v24，不能用較弱對照取代主要結論。","",
 "## 原生模型與完整組合必須分開看","",
 "| 期別 | 層 | v22對應單價分支MAE | R85 MAE | 差 |","| --- | --- | ---: | ---: | ---: |"]
 for s in STAGES:
  for k in ["native","base","reference","spatial"]:
   a=control["stages"][s]["unit_branch_metrics"][k]["overall_mae"];b=result["stages"][s]["unit_branch_metrics"][k]["overall_mae"]
   lines.append(f"| {LABELS[s]} | {k} | {f(a)} | {f(b)} | {sign(b-a)} |")
 lines+=["",
 "中間期 native 改善99,982.11元，經reference/spatial後單分支仍改善24,646.54元；但完整median退步9,458.39元。逐筆median對每筆採用不同分支的排序位置，單支平均MAE降低不保證組合MAE降低。這裡沒有重新最佳化融合比例。",
 "最新期 native 改善11,848.92元、base改善25,692.77元，但reference/spatial後轉差。最舊期從native到完整組合皆退步，代表新表示也有跨年度不穩定性；不能只以最新或中間原生支判定成功。","",
 "## 五個真價帶與尾部的整體貢獻","",
 "| 期別 | 真價帶 | 列數 | v22 MAE | R85 MAE | 差vs v22 | 差vs v24 | 對全期MAE差貢獻vs v22 |",
 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  for b in diag["stages"][s]["price_bands"]:
   lines.append(f"| {LABELS[s]} | {b['band']} | {b['rows']:,} | {f(b['v22'])} | {f(b['R85'])} | {sign(b['R85']-b['v22'])} | {sign(b['R85']-b['v24'])} | {sign(tail[s]['v22']['band_contributions_TWD'][b['band']])} |")
 lines+=["","| 期別 | ≥1億貢獻 | 其餘價帶合計貢獻 | 全期MAE差 |","| --- | ---: | ---: | ---: |"]
 for s in STAGES:
  g=tail[s]["v22"];lines.append(f"| {LABELS[s]} | {sign(g['ge_100m_contribution'])} | {sign(g['all_other_bands_contribution'])} | {sign(g['all_bands_sum'])} |")
 lines+=["",
 "中間期<5,000萬的80,250列、最新期<5,000萬的131,653列，三個價帶各自都比v22改善。相對地，≥1億只182／319列，卻各貢獻+17,892.67／+32,450.03元的全期MAE差，抵銷其餘價帶收益。最舊期5,000萬–1億的1,382列改善107,295.25元，但其餘四個價帶均退步，不能把整體問題完全歸因極端尾部。",
 "有號貢獻＝群MAE差×群列數／全期列數；它不是平均prediction−truth。五帶合計精確回到整體MAE差。真價分群是事後診斷，不能用未知test真價把低價列切給R85、高價列切回v22。","",
 "## 新舊建案與季度","",
 "| 期別 | 建案組 | 列數 | R85 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s]
  for k,g in v["metrics"]["R85"]["strata"].items():
   if not g["rows"]:continue
   lines.append(f"| {LABELS[s]} | {k} | {g['rows']:,} | {f(g['mae'])} | {sign(v['differences']['v22']['strata'][k]['delta_TWD'])} | {sign(v['differences']['v24']['strata'][k]['delta_TWD'])} |")
 lines+=["","三期missing建案群均0列。新建案與已見建案兩群三期皆退步，表示不能靠改新舊案權重把本輪描述成普遍改善。","",
 "| 期別 | 預測季度 | 列數 | R85 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  v=result["stages"][s]
  for k,g in v["metrics"]["R85"]["forecast_quarters"].items():
   lines.append(f"| {LABELS[s]} | {k} | {g['rows']:,} | {f(g['mae'])} | {sign(v['differences']['v22']['forecast_quarters'][k]['delta_TWD'])} | {sign(v['differences']['v24']['forecast_quarters'][k]['delta_TWD'])} |")
 lines+=["","中間期Q3、Q4與最新期Q3對v22改善；其餘九個季度退步。季度按既有query起點的預測距離定義，不是新的calendar切分。","",
 "## 全部城市，保留改善與退步","",
 "| 期別 | 城市 | 列數 | R85 MAE | 差vs v22 | 差vs v24 |","| --- | --- | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  for city,g in cities[s].items():
   lines.append(f"| {LABELS[s]} | {city} | {g['rows']:,} | {f(g['candidate_mae'])} | {sign(g['delta_vs_v22'])} | {sign(g['delta_vs_v24'])} |")
 lines+=["",
 "局部改善例子：中間期新竹市1,250列改善554,191.46元、新竹縣3,639列改善310,872.07元；最新期桃園26,327列改善15,342.36元、高雄16,443列改善18,639.43元。這些是完整median結果，規模並非全是零星交易。澎湖、金門等小樣本群的波動需另看列數，不把它們當穩健收益。",
 "上述列舉不替代全表，也沒有据此按城市挑模型。各城市含不同建案、時間及價带，描述性分群不能證明某欄位改動的因果效果。","",
 "## 固定test組成估計","",
 "| 期別 | 支持質量 | 未支持test列 | v22估計 | v24估計 | R85估計 | 差vs v22 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for s in STAGES:
  q=diag["stages"][s]["test_composition"];m=q["models"]
  lines.append(f"| {LABELS[s]} | {q['test_mass_covered']:.4%} | {q['unsupported_test_rows']:,} | {f(m['v22'])} | {f(m['v24'])} | {f(m['R85'])} | {sign(m['R85']-m['v22'])} |")
 lines+=["","沿用R49固定city×seen/new test-X權重，歷史cell至少200列，僅對有支持質量重正規化；三期亦退步。這不是test／Private真實MAE，也不是完整未來泛化預測。","",
 "## 全部24個配對區間","",
 "固定建案群集與三個月循環區塊，每項2,000次、seed20260925。差值為R85−對照；所有區間valid=2000、invalid=0。以下一併呈現，不以某一抽法代替其餘。","",
 "| 對照 | 期別 | 抽法 | 指標 | 點差 | 2.5% | 97.5% | 區間 |","| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
 for i in intervals:
  cl={"entirely_positive":"全正","crosses_zero":"跨零","entirely_negative":"全負"}[i["classification"]]
  lines.append(f"| {i['control']} | {LABELS[i['stage']]} | {i['method']} | {i['metric']} | {sign(i['point_delta'])} | {sign(i['lower'])} | {sign(i['upper'])} | {cl} |")
 lines+=["",
 "相對v22：中間期4區間都跨零；最新期只有month overall全正，其餘3項跨零；最舊期4項全正。不能說v22在所有期別均顯著勝出，但沒有任何區間整段支持R85改善。相對v24：中間期4項跨零，最新與最舊各4項全正。",
 "這些區間以反覆查看的窗口、固定保存模型為條件，未涵蓋選型偏差、重訓隨機性或未來市場移轉；不是Private勝率、p值或新盲測。","",
 "## 真實執行、保存與審查範圍","",
 "| 期別 | fit/query列 | fit秒 | 全期秒 | 保存後全query重載 |","| --- | --- | ---: | ---: | --- |"]
 for s in STAGES:
  v=result["stages"][s]
  lines.append(f"| {LABELS[s]} | {v['fit_rows']:,}/{v['query_rows']:,} | {v['fit_seconds']:.3f} | {v['elapsed_seconds']:.3f} | raw與TWD位元相同 |")
 fit_sum=sum(result["stages"][s]["fit_seconds"] for s in STAGES)
 lines+=["",
 f"三次fit合計{fit_sum:.3f}秒；主流程內層{result['elapsed_seconds']:.3f}秒、外層{execution['elapsed_seconds']:.3f}秒。generic診斷內層{diag['elapsed_seconds']:.3f}秒、外層{diag_exec['elapsed_seconds']:.3f}秒；paired vs v22內層{paired['v22']['elapsed_seconds']:.3f}秒，vs v24內層{paired['v24']['elapsed_seconds']:.3f}秒。未把聊天等待時間列為訓練時間。",
 "三個fit、主流程與generic execution均exit0；paired JSON均completed。日期權重來源、同切分索引／特徵、對照完整metrics與v22 base/reference/spatial/median向量重播通過，原保護清單前後相同。模型各自保存CBM及metadata後重載全query核驗。這些為主程式已有證據，本審查器沒有另載模型或產生價格。",
 "本審查只讀JSON、核其sidecar SHA、檢查數字一致及加總，沒有重跑訓練／推論／bootstrap。缺少本輪新的面積、型態逐列候選診斷，不能據先前v22誤差盤點斷言R85改善廠辦或面積尾部；現有真價帶已顯示不利尾部證據。",
 "GPU固定seed不保證位元決定性，v22未同期重新訓練，不能把全部差異當已隔離的面積冪次因果效果。舊reference/spatial配方未重調，也可能與新表示不匹配，但本輪沒有搜尋新修正強度。",
 "",
 "## 建議與下一步","",
 "建議保留R85完整研究，不做final fit或提交CSV。理由是三期整體、固定加權、新舊案及test組成估計一致不利，加上億級尾部代價；並非臨時要求每一期必須顯著勝出。中低價與若干城市／季度的改善保存為後續特徵及誤差研究線索，不能直接按query最佳群拼接模型。",
 "使用者選定v22保持不變，Private仍未知。後續新候選應另立固定假設與驗證計畫，保留這次反證；本文件不更改任何採用登錄。",
 "",
 "[固定計畫](round85_plan.md) · [主結果](../outputs/round85/results.json) · [generic完整診斷](round85_diagnostics.json) · [vs v22](round85_paired_vs_v22.md) · [vs v24](round85_paired_vs_v24.md)",
 "[決策審查JSON](../outputs/round85/decision_review.json) · [可重跑審查程式](../outputs/round85/review_results.py) · [前置誤差盤點](r85_error_audit.md)",""]
 review={"review_type":"recommendation_only_not_selection_mutation","timestamp_utc":datetime.now(timezone.utc).isoformat(),
  "round":85,"candidate":"sqrt_area_scale","recommendation":"retain_research_do_not_final_fit_or_export",
  "reason":"All three overall/weighted/testmix and seen/new strata worse than v22; high-price tail offsets genuine lower-price gains.",
  "selector_changed":False,"csv_generated":False,"private_score":None,"new_fits_in_review":0,"prices_recomputed_in_review":False,
  "stage_summary":summary,"true_price_tail_decomposition":tail,"paired_interval_counts":count,"all_24_intervals":intervals,
  "favorable_evidence":["Development native improves 99,982.11 TWD and spatial branch improves 24,646.54 TWD although median worsens.",
    "Development and confirmation each improve all three bands below 50m TWD; 80,250 and 131,653 rows respectively.",
    "Development Q3/Q4 and confirmation Q3 improve versus v22; several substantial city groups improve.",
    "Auxiliary 50-100m band improves on 1,382 rows; other four bands worsen."],
  "complete_city_comparison":cities,
  "fit_seconds_sum":fit_sum,"main_inner_seconds":result["elapsed_seconds"],"main_outer_seconds":execution["elapsed_seconds"],
  "limitations":["Repeatedly inspected historical windows, not blind","No same-run v22 retrain; GPU randomness not isolated",
    "Bootstrap not corrected for selection or future shift; not Private probability","True-price groups cannot be inference routing",
    "This review does not rescore per-row area/type candidate groups or reload models"],
  "input_sha256":inputs}
 source=Path(__file__);inputs[source.relative_to(ROOT).as_posix()]=sha(source)
 assert all(sha(ROOT/n)==h for n,h in inputs.items())
 review["inputs_unchanged"]=True
 review["review_elapsed_seconds"]=time.perf_counter()-t
 with (OUT/"decision_review.json").open("x",encoding="utf8") as fjson:json.dump(review,fjson,ensure_ascii=False,indent=2,allow_nan=False)
 with (ROOT/"reports/round85_report.md").open("x",encoding="utf8",newline="\n") as fmd:fmd.write("\n".join(lines))
 print(json.dumps({"report":"reports/round85_report.md","review":"outputs/round85/decision_review.json","fit_seconds_sum":fit_sum,"interval_counts":count,"input_files":len(inputs)}))
if __name__=="__main__":main()
