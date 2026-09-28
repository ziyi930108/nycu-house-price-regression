"""Summarize completed R97 and the three original fixed windows; no labels/models."""
from pathlib import Path
import hashlib,json,time
from datetime import datetime,timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/round97_summary"
REPORT=ROOT/"reports/round97_stability_summary.md"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,d):
 with p.open("x",encoding="utf-8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def main():
 tick=time.monotonic()
 assert not OUT.exists() and not REPORT.exists(),"Keep original summary evidence"
 sources={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in [
  ROOT/"outputs/round79/results.json",ROOT/"outputs/round83/results.json",
  ROOT/"outputs/round97/sealed_manifest.json",ROOT/"outputs/round97/execution.json",
  ROOT/"outputs/round97/evaluation/results.json",ROOT/"outputs/round97/evaluation/execution.json",
  ROOT/"reports/round97_plan.md",Path(__file__)]}
 old22=read(ROOT/"outputs/round79/results.json");old24=read(ROOT/"outputs/round83/results.json")
 seal=read(ROOT/"outputs/round97/sealed_manifest.json");ev=read(ROOT/"outputs/round97/evaluation/results.json")
 assert seal["status"]=="sealed_predictions_without_query_labels" and seal["new_native_fits"]==7
 assert ev["status"]=="completed_sealed_prediction_evaluation"
 assert ev["sealed_manifest_sha256"]==sha(ROOT/"outputs/round97/sealed_manifest.json")
 assert all(read(ROOT/p)["exit_code"]==0 for p in ["outputs/round97/execution.json","outputs/round97/evaluation/execution.json"])
 rows=[]
 for stage,label in [("auxiliary","較早年度"),("development","中間年度"),("confirmation","最近年度")]:
  a=old22["stages"][stage];b=old24["stages"][stage]
  rows.append(dict(window=label,stage=stage,dates=a["dates"],fit_rows=a["fit_rows"],query_rows=a["query_rows"],
    v22=a["metrics"]["R79"],v24=b["metrics"]["R83"],newly_retrained_here=False))
 rows.append(dict(window="錯開切點（本次）",stage="offset_20230211",dates=["2023-02-11","2024-02-10"],
   fit_rows=seal["fit_rows"],query_rows=seal["query_rows"],v22=ev["metrics"]["v22"],v24=ev["metrics"]["v24"],newly_retrained_here=True))
 compact=[]
 for r in rows:
  compact.append({k:r[k] for k in ["window","stage","dates","fit_rows","query_rows","newly_retrained_here"]}|
    {v+"_"+m:r[v][m] for v in ["v22","v24"] for m in ["overall_mae","weighted_mae"]}|
    {"delta_"+m:r["v24"][m]-r["v22"][m] for m in ["overall_mae","weighted_mae"]})
 OUT.mkdir()
 font=FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
 for ax,metric,title in zip(axes,["overall_mae","weighted_mae"],["整體 MAE 差值","固定群組加權 MAE 差值"]):
  v=[r["delta_"+metric] for r in compact]
  bars=ax.barh(range(4),v,color=["#16827c" if x<0 else "#c35145" for x in v],height=.55)
  ax.axvline(0,color="#475569",linewidth=1)
  ax.set_yticks(range(4),[r["window"] for r in compact],fontproperties=font)
  ax.invert_yaxis();ax.set_title(title,fontproperties=font,fontsize=15)
  ax.set_xlabel("v24 − v22（元）；負值表示 v24 較低",fontproperties=font)
  span=max(abs(x) for x in v) or 1
  ax.set_xlim(min(min(v),0)-.3*span,max(max(v),0)+.3*span)
  for bar,x in zip(bars,v):
   ax.text(x-(.02*span) if x<0 else x+.02*span,bar.get_y()+bar.get_height()/2,
    f"{x:+,.0f}",ha="right" if x<0 else "left",va="center",fontsize=10)
  ax.spines[["top","right"]].set_visible(False)
 fig.suptitle("固定完整配方的時間切分穩定性",fontproperties=font,fontsize=18)
 fig.text(.5,-.025,"四個窗口有重疊，不是四次獨立盲測；這些數值都不是 Private 分數。",ha="center",fontproperties=font,fontsize=11)
 fig.savefig(OUT/"four_window_mae_delta.png",dpi=170,bbox_inches="tight");plt.close(fig)
 lines=["# R97：換一個時間切點，完整回歸配方是否仍穩定？","",
  "本輪重新訓練完整 v22／v24 所需的七個模型及全部學得狀態，保留原配方。這是時間穩定性證據；沒有提出或採用新的提交版本。","",
  "## 四個窗口一起看","",
  "| 評分窗口 | fit／query 列數 | v22整體MAE | v24整體MAE | v24−v22 | 加權差值 |",
  "|---|---:|---:|---:|---:|---:|"]
 for r in compact:
  lines.append(f'| {r["dates"][0]}～{r["dates"][1]} | {r["fit_rows"]:,}／{r["query_rows"]:,} | {r["v22_overall_mae"]:,.2f} | {r["v24_overall_mae"]:,.2f} | {r["delta_overall_mae"]:+,.2f} | {r["delta_weighted_mae"]:+,.2f} |')
 lines+=["","原三個8月起點來自既有保存結果；只有2023-02-11起點是本輪完整重訓。窗口與訓練資料重疊，不能相加當成四組獨立證據。","",
  "![四窗口MAE差值](../outputs/round97_summary/four_window_mae_delta.png)","",
  "## 本次配對區間與分群代價","",
  "| 抽法 | 指標 | 差值 | 2.5% | 97.5% |","|---|---|---:|---:|---:|"]
 for family,d in ev["paired"]["methods"].items():
  for metric,m in d["summary"].items():
   lines.append(f'| {family} | {metric} | {m["point_delta"]:+,.2f} | {m["percentile_2_5"]:+,.2f} | {m["percentile_97_5"]:+,.2f} |')
 c=ev["comparison"]
 lines+=["",f'本窗口整體與固定加權點估計是否同時改善：{c["both_point_metrics_improve"]}。',
  "所有城市／月份／新舊建案／五價帶及各層指標均保存，不能只挑平均值。","",
  "超過5%退步的季度或至少500列分群："]
 for x in c["quarter_or_large_group_deterioration_above_5pct"]:
  lines.append(f'- {x["section"]}／{x.get("group",x.get("band"))}：{x["rows"]:,}列，差{x["delta"]:+,.2f}元（{x["relative_change_pct"]:+.2f}%）。')
 if not c["quarter_or_large_group_deterioration_above_5pct"]:lines.append("- 本窗口沒有達到此固定通報條件的分群；不等於每一群都改善。")
 lines+=["","## 本次實際執行與限制","",
  f'5個樹模型、2個NN；各模型 fit 計時段合計{seal["fit_seconds"]["native_total"]:,.3f}秒，包含在main {read(ROOT/"outputs/round97/execution.json")["elapsed_seconds"]:,.3f}秒內，不重複相加。另建立新encoder／ramps、group OOF/full prior、reference/spatial。所有query保存模型／state重載一致。',
  "模型driver只接隔離輸出的fit標籤，全部預測封存後才由獨立scorer載query價格。底層價格pickle在隔離extractor與scorer各會反序列化原表，此限制已明記。",
  "固定配方先前已使用重疊資料選型；本輪沒有新的盲測，bootstrap不校正累積選型、重新訓練或未來分布改變，更不是Private勝率。",
  "v22仍是使用者唯一選定提交。Public 1,998,568.5706，Private未知；沒有上傳Kaggle、產生新CSV或更改勾選。",
  "保存模型重載／本機七模型新窗口重訓，不等於全新環境重現最終test模型。","",
  "[固定計畫](round97_plan.md) · [主程式](../src/round97_offset_experiment.py) · [評分程式](../src/round97_offset_evaluate.py) · [完整分群報告](../outputs/round97/evaluation/report.md) · [模型封存收據](../outputs/round97/sealed_manifest.json)",""]
 with REPORT.open("x",encoding="utf-8") as f:f.write("\n".join(lines))
 summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),windows=compact,
  source_sha256=sources,fit_seconds=seal["fit_seconds"],new_native_fits=7,new_CSV=False,Private_score=None,
  repeated_overlapping_windows=True,selection_changed=False,no_labels_or_prediction_arrays_loaded=True)
 write(OUT/"summary.json",summary)
 write(OUT/"receipt.json",dict(status="completed",source_sha256=sources,
  artifact_sha256={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in [REPORT,OUT/"summary.json",OUT/"four_window_mae_delta.png"]},
  elapsed_seconds=time.monotonic()-tick))
 print(json.dumps(dict(status="completed",windows=4,artifact=str(REPORT))))
if __name__=="__main__":main()

