"""Create the final comparison report and reproducibility manifest."""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"outputs"/"matplotlib_cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_common import ROOT, REPORT_DIR, EXPERIMENT_DIR, load_data, mae
from prepare_data import sha256, DATA_DIR, FILENAMES, SAMPLE_FILENAME

def money(value):
    return f"{float(value):,.0f}"

def main():
    result_path=REPORT_DIR/"results.json"
    result=json.loads(result_path.read_text(encoding="utf-8"))
    config=json.loads((ROOT/"models"/"final_config.json").read_text(encoding="utf-8"))
    training=json.loads((REPORT_DIR/"final_training.json").read_text(encoding="utf-8"))
    checks=json.loads((ROOT/"outputs"/"submission_checks.json").read_text(encoding="utf-8"))
    result["final_submission"]={**checks,"path":"outputs/submission.csv"}
    result["final_training"]=training
    notebook_path=REPORT_DIR/"notebook_checks.json"
    notebook_checks=json.loads(notebook_path.read_text(encoding="utf-8")) if notebook_path.exists() else None
    if notebook_checks and (notebook_checks.get("submission_sha256") != checks["sha256"]
                            or notebook_checks.get("notebook_sha256") != sha256(ROOT/"solution.ipynb")):
        notebook_checks=None
    if notebook_checks:
        result["notebook_verification"]=notebook_checks
    else:
        result.pop("notebook_verification",None)
    result["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    data=load_data()
    pred=sum(c["weight"]*np.load(EXPERIMENT_DIR/f"{c['name']}_valid.npy") for c in config["components"])
    sel=result["selection"]
    if not np.isclose(mae(data["y_valid"],pred),sel["validation_mae"],rtol=0,atol=1e-5):
        raise ValueError("Reported MAE differs from archived predictions")
    unique=np.load(ROOT/"data_cache"/"valid_model_unique_mask.npy")
    pd.DataFrame({"編號":data["ids_valid"],"實際總價元":data["y_valid"],"預測總價元":pred,
                  "絕對誤差元":np.abs(data["y_valid"]-pred),"原始欄位未與訓練重疊":data["valid_clean_mask"],
                  "模型特徵未與訓練重疊":unique}).to_csv(
        ROOT/"outputs"/"validation_predictions.csv",index=False,encoding="utf-8-sig")
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    temporal={r["name"]:r for r in result["temporal_validation"]}
    candidates=result["validation"]
    short={"catboost_raw":"CatBoost total","catboost_unit":"CatBoost unit",
           "catboost_structural":"CatBoost structural","lightgbm_l1_raw63":"LightGBM total",
           "lightgbm_l1_unit63":"LightGBM unit"}
    names=[r["name"] for r in candidates]
    labels=[short.get(n,n) for n in names]+["Selected"]
    values=[r["mae"]/1e6 for r in candidates]+[sel["validation_mae"]/1e6]
    times=[temporal[n]["mae"]/1e6 for n in names]+[sel["temporal_mae"]/1e6]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,vals,title in zip(axes,[values,times],["Provided validation set","Chronological holdout: Apr-Aug 2024"]):
        bars=ax.bar(labels,vals,color=["#597a9a"]*len(names)+["#23856d"],width=.65)
        ax.set_title(title); ax.set_ylabel("MAE (million TWD)"); ax.set_ylim(0,max(vals)*1.22)
        ax.tick_params(axis="x",rotation=30,labelsize=8)
        for tick in ax.get_xticklabels(): tick.set_horizontalalignment("right")
        ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in vals],padding=3,fontsize=9)
    fig.suptitle("Local validation: lower error is better",fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_DIR/"model_comparison.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    table=["| 模型 | 原驗證 MAE | 排除原始重疊 MAE | 時間驗證 MAE | 時間驗證：未見建案 MAE |",
           "|---|---:|---:|---:|---:|"]
    for r in candidates:
        t=temporal[r["name"]]
        table.append(f"| {r['name']} | {money(r['mae'])} | {money(r['clean_mae'])} | {money(t['mae'])} | {money(t['unseen_project_mae'])} |")
    components="、".join(f"{c['name']} {c['weight']:.0%}" for c in config["components"])
    iterations="\n".join(f"- {c['name']}：{c['iterations']:,} 輪，權重 {c['weight']:.0%}。" for c in config["components"])
    reduction=1-sel["validation_mae"]/result["baseline"]["mae"]
    notebook_note=(f"Notebook 的 {notebook_checks['executed_code_cells']} 個程式格已實際執行通過，"
                   "由保存模型重新產生的提交 CSV 與原檔完全相同（SHA-256 一致）。"
                   "詳見 [Notebook 執行檢查](notebook_checks.json)。" if notebook_checks else "")
    text=f"""# 台灣房屋銷售價格預測：完成報告

## 提交檔與主要結果

- [submission.csv](../outputs/submission.csv)：{checks['rows']:,} 筆，欄位「編號、總價元」。
- 選定模型：{components}。
- 原驗證集 MAE：**{money(sel['validation_mae'])} 元**。
- 排除 36 筆原始欄位重疊後 MAE：**{money(sel['clean_validation_mae'])} 元**。
- 排除 38 筆模型特徵重疊後 MAE：**{money(sel['model_unique_validation_mae'])} 元**。
- 時間驗證 MAE：**{money(sel['temporal_mae'])} 元**。
- 相較訓練價格中位數基準 MAE {money(result['baseline']['mae'])} 元，原驗證誤差降低 {reduction:.1%}。

這些是最終重訓之前的本機驗證結果，不是 Kaggle 排行榜分數。最終模型加入驗證集重新訓練後，沒有再拿已參與訓練的標籤宣稱測試成績。

## 資料與規定

僅使用題目提供的 train、valid、test 與提交範例。沒有查詢內政部不動產成交案件實際資訊資料供應系統，也沒有從外部資料比對測試答案。下載套件與閱讀模型官方文件不作為房價資料來源。

訓練檔原有 455,787 列，移除 3 列英文欄位說明後剩 455,784 筆。依全部原始欄位（排除編號、保留價格）去除 529 筆完全重複記錄，模型比較使用 455,255 筆訓練資料；驗證集為 11,994 筆。不同價格的交易不會只因編號相同而任意刪除。

最終合併 train 與 valid，再依相同規則去重，實際使用 **{training['training_rows']:,} 筆**有標籤資料。原始檔均保留。

## 特徵與模型

固定 87 個特徵：21 個類別與 66 個數值欄位，包含地區／地址拆分、建案與棟號、交易日期、面積、樓層、格局、車位及有限的備註條件旗標。

價格與編號不作為特徵。沒有可用日期的建築完成年月、常數管理組織、完整備註與可能事後才知道的解約情形未納入。無效日期設缺失並保留旗標，極端房價與面積保留。「建物面積減車位面積」只是近似衍生量，不代表登記室內坪數。

比較 CatBoost 與 LightGBM 的總價及面積加權單價模型。單價模型以有效面積 A 定義目標 y/A，使用 A 作權重，滿足 A × |y/A − 預測單價| = |y − A × 預測單價|，仍對應總價絕對誤差。有效面積優先用正的建物面積，其次正的車位面積，否則用 1。訓練另作固定數值縮放，輸出全部換回元。

structural 候選忽略建案名稱、完整地址與完整棟戶號，檢查對新建案的泛化能力；特徵 schema 保持一致。

## 模型比較

所有 MAE 單位均為元。

{chr(10).join(table)}

![模型比較](model_comparison.png)

## 時間驗證與選型

以 2024-04-01 之前的有效日期資料訓練，保留 2024-04-01～2024-08-10 的 58,206 筆去重交易作時間驗證，訓練側為 397,041 筆。8 筆無效或不合理日期不參與此時間切分。

原 valid 幾乎都是已見建案，但 test 相對最終有標籤資料的未見建案比例為 **{sel['test_unseen_project_rate']:.2%}**。另比較時間驗證中的已見／未見建案誤差，再依測試集兩組比例重新加權；只使用測試集的無標籤建案名稱，沒有使用測試價格。

- 已見建案 MAE：{money(sel['seen_project_mae'])} 元。
- 未見建案 MAE：{money(sel['unseen_project_mae'])} 元。
- 依測試組成重新加權的時間驗證 MAE：{money(sel['reweighted_temporal_mae'])} 元。

在單模型與兩模型 25%／50%／75% 權重的粗網格中選型，詳見 [ensemble_search.csv](ensemble_search.csv)。兩份驗證資料都參與模型選擇，這些數字是選型依據；未來市場與新建案的實際誤差仍可能不同。

## 最終訓練與提交檢查

{iterations}

已核對 47,073 個編號的完整性、唯一性及與範例一致的順序，確認欄位、有限正價格和存檔後讀回一致。價格四捨五入到元，最低為 1 元。預測範圍為 {checks['minimum']:,}～{checks['maximum']:,} 元，中位數 {money(checks['median'])} 元。

## 重現與檔案

- [操作說明](../README.md)、[Notebook](../solution.ipynb)。
- [完整結果](results.json)、[最終訓練紀錄](final_training.json)。
- [逐筆驗證誤差](../outputs/validation_predictions.csv)、[提交檢查](../outputs/submission_checks.json)。
- [資料品質](data_quality.json)、[切分稽核](split_audit.json)、[特徵重疊](model_feature_overlap.json)。
- [版本資訊](runtime.json)、[來源與程式雜湊](run_manifest.json)。

使用保存模型執行 src/predict.py 可精確重現提交檔。固定亂數種子的 GPU 重新訓練仍可能有微小數值差異，不保證逐位相同。

{notebook_note}
"""
    (REPORT_DIR/"model_report.md").write_text(text,encoding="utf-8")
    quality=json.loads((REPORT_DIR/"data_quality.json").read_text(encoding="utf-8"))
    inputs={}
    for split,filename in FILENAMES.items():
        digest=sha256(DATA_DIR/filename)
        if digest!=quality[split]["source_sha256"]: raise ValueError("Original input changed")
        inputs[filename]=digest
    digest=sha256(DATA_DIR/SAMPLE_FILENAME)
    if digest!=quality["submission_template"]["source_sha256"]: raise ValueError("Template changed")
    inputs[SAMPLE_FILENAME]=digest
    paths=list((ROOT/"src").glob("*.py"))+[ROOT/"requirements.txt",ROOT/"README.md",ROOT/"solution.ipynb",
          ROOT/"models"/"final_config.json",ROOT/"outputs"/"submission.csv"]
    paths+=list((ROOT/"data_cache").glob("*.pkl"))
    paths+=[p for p in REPORT_DIR.iterdir() if p.is_file() and p.name != "run_manifest.json"]
    paths.append(ROOT/"outputs"/"submission_checks.json")
    for component in config["components"]:
        paths.append(ROOT/component["model_path"])
        if "category_maps_path" in component: paths.append(ROOT/component["category_maps_path"])
    manifest={"inputs_unchanged":True,"input_sha256":inputs,
              "artifacts_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in paths},
              "source_policy":"Only supplied files; no external property prices or government answer lookup."}
    (REPORT_DIR/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"validation_mae":sel["validation_mae"],"temporal_mae":sel["temporal_mae"],
                      "submission_rows":checks["rows"],"inputs_unchanged":True}),flush=True)

if __name__=="__main__":
    main()
