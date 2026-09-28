"""Summarize completed research receipts only; never fit, predict, or read labels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/regression_after_r95_20260929"
STAGES = ("auxiliary", "development", "confirmation")
STAGE_NAMES = ("較早年度", "中間年度", "最新年度")
EXPERIMENTS = (
    (85, "平方根面積尺度", 3, 0, 0),
    (86, "單價樹加入價格先驗", 3, 0, 0),
    (87, "固定混合兩種面積表示", 0, 0, 0),
    (88, "低支持群調整參考修正", 0, 0, 0),
    (89, "分群時間趨勢", 0, 0, 3),
    (91, "先驗上的總價殘差", 3, 0, 0),
    (92, "較淺的單價樹", 3, 0, 0),
    (93, "建案密度權重", 3, 0, 0),
    (95, "欄位 Attention 先驗回歸", 0, 3, 0),
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    evidence = {}

    def read(relative):
        p = ROOT / relative
        evidence[relative] = sha(p)
        return json.loads(p.read_text(encoding="utf-8"))

    baseline = read("outputs/round79/results.json")
    rows = []
    for n, label, trees, nn, wrappers in EXPERIMENTS:
        r = read(f"outputs/round{n}/results.json")
        execution = read(f"outputs/round{n}/execution.json")
        decision = read(f"outputs/round{n}/decision_review.json")
        assert r["status"] == "completed" and execution["exit_code"] == 0
        assert set(r["checked_stages"]) == set(STAGES)
        stages = {}
        for s in STAGES:
            m = r["stages"][s]["metrics"]
            b = baseline["stages"][s]["metrics"]["R79"]
            assert all(m["v22"][k] == b[k] for k in ("overall_mae", "weighted_mae"))
            stages[s] = {
                "v22_mae": m["v22"]["overall_mae"],
                "candidate_mae": m[f"R{n}"]["overall_mae"],
                "delta_mae": m[f"R{n}"]["overall_mae"] - m["v22"]["overall_mae"],
                "delta_weighted_mae": m[f"R{n}"]["weighted_mae"] - m["v22"]["weighted_mae"],
            }
        rows.append({"round": n, "hypothesis": label, "stages": stages,
                     "tree_fits": trees, "NN_fits": nn, "CPU_wrapper_fits": wrappers,
                     "fit_seconds": sum(r["stages"][s].get("fit_seconds", 0) for s in STAGES),
                     "main_elapsed_seconds": execution["elapsed_seconds"],
                     "saved_recommendation": decision["recommendation"],
                     "decision_path": f"outputs/round{n}/decision_review.json"})

    failed = read("outputs/round94/execution.json")
    assert failed["exit_code"] == 1
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Nine scientific hypotheses; R94/R95 are one recovered hypothesis.",
               "selected_submission": "v22", "Private_score": None,
               "new_fit_or_predict_calls_in_this_summary": 0,
               "input_price_or_prediction_arrays_loaded": False,
               "experiments": rows,
               "total_tree_fits": sum(x["tree_fits"] for x in rows),
               "total_NN_fits": sum(x["NN_fits"] for x in rows),
               "total_CPU_wrapper_fits": sum(x["CPU_wrapper_fits"] for x in rows),
               "total_fit_seconds": sum(x["fit_seconds"] for x in rows),
               "total_successful_main_seconds": sum(x["main_elapsed_seconds"] for x in rows),
               "R94_prefit_failed_seconds": failed["elapsed_seconds"],
               "timing_note": "Fit is included in main. Development, review, synthetic tests, diagnostics and publishing are not included. Not whole-chat elapsed time.",
               "limitations": ["Same historical windows reused; no correction for adaptive model selection.",
                               "Different hypotheses change different components and are not independent replications.",
                               "Selection reasoning and favorable counterevidence remain in each original report.",
                               "Lower historical MAE does not establish lower unknown Private MAE."],
               "input_sha256": evidence,
               "source_sha256": sha(Path(__file__))}

    # All required completed evidence is checked before creating exclusive new outputs.
    assert not OUT.exists(), "Refusing to overwrite a prior summary"
    report_path = ROOT / "reports/regression_after_r95_20260929.md"
    assert not report_path.exists()
    OUT.mkdir()
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    font = FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
    data = np.array([[x["stages"][s]["delta_mae"] / 1000 for s in STAGES] for x in rows])
    bound = max(abs(data.min()), abs(data.max()))
    fig, ax = plt.subplots(figsize=(12.2, 7.5), dpi=160)
    fig.patch.set_facecolor("#fafafa")
    im = ax.imshow(data, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(3), STAGE_NAMES, fontproperties=font, fontsize=12)
    ax.set_yticks(range(len(rows)), [f'R{x["round"]}  {x["hypothesis"]}' for x in rows], fontproperties=font, fontsize=11)
    ax.tick_params(length=0, pad=12)
    for i in range(len(rows)):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:+.2f}", ha="center", va="center", fontsize=12, color="#141414")
    ax.set_title("九項回歸假設：相對 v22 的 MAE 差額", fontproperties=font, fontsize=19, pad=28)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.04)
    cb.set_label("千元；負值較好", fontproperties=font, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.04, .035, "完整集成的離線結果；三期窗口反覆使用，不能視為 Private 表現。R94 工程失敗併入 R95 同一假設。", fontproperties=font, fontsize=10)
    fig.subplots_adjust(left=.31, right=.90, top=.84, bottom=.11)
    fig.savefig(OUT / "mae_delta_comparison.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    lines = ["# 九項回歸改良研究：R95 完成後彙整", "",
             "本頁由已完成的結果與決策 JSON 產生，沒有重訓、推論或重新讀取價格標籤。選定版本仍為 v22；Private 未知。各輪的採用理由和有利反證仍以原報告為準。", "",
             "![三期差額](../outputs/regression_after_r95_20260929/mae_delta_comparison.png)", "",
             "差額＝新候選 MAE − v22 MAE，單位為新臺幣元；負值代表該期改善。", "",
             "|假設|較早年度|中間年度|最新年度|新 fit：樹／NN／wrapper|", "|---|---:|---:|---:|---:|"]
    for x in rows:
        ds = "|".join(f'{x["stages"][s]["delta_mae"]:+,.2f}' for s in STAGES)
        lines.append(f'|[R{x["round"]} {x["hypothesis"]}](round{x["round"]}_report.md)|{ds}|{x["tree_fits"]}/{x["NN_fits"]}/{x["CPU_wrapper_fits"]}|')
    lines += ["", f'共 {summary["total_tree_fits"]} 次樹模型、{summary["total_NN_fits"]} 次 NN、{summary["total_CPU_wrapper_fits"]} 次 CPU 趨勢 wrapper fit。真 fit 合計 {summary["total_fit_seconds"]:.3f} 秒，已包含於各成功 main 合計 {summary["total_successful_main_seconds"]:.3f} 秒；兩者不可相加。', "",
              f'R94 在真 fit 前因舊 JSON 讀取契約失敗，耗時 {summary["R94_prefit_failed_seconds"]:.3f} 秒；原檔保留，R95 是相同科學配置的工程恢復，不能重算一項新假設。合成、開發、審查、診斷、發布時間不在上述 main 合計，整段聊天時間沒有實測。', "",
              "多轮重用年度窗口帶來選型偏差；各假設亦非獨立重複試驗，不能從這張表推論 Private 勝率。不能只挑某一期改善、某個原生分支勝出或訓練 loss 下降就宣布新提交更好。", "",
              "[機器可讀彙整與來源 SHA](../outputs/regression_after_r95_20260929/summary.json)；[R95 完整分析](round95_report.md)。"]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"summary_sha256": sha(OUT / "summary.json"), "report_sha256": sha(report_path),
               "plot_sha256": sha(OUT / "mae_delta_comparison.png"), "source_sha256": sha(Path(__file__)),
               "input_hashes_still_match": all(sha(ROOT / p) == d for p, d in evidence.items())}
    assert receipt["input_hashes_still_match"]
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": len(rows), "fits": [summary["total_tree_fits"], summary["total_NN_fits"], summary["total_CPU_wrapper_fits"]], "output": str(OUT)}))


if __name__ == "__main__":
    main()
