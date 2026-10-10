"""Read completed diagnostic aggregates, render a scientific plot, and explain it.

--build reads JSON/source identities only. --check-vectors separately recomputes
the existing 60 prediction-vector MAEs, without loading a model or fitting one.
Every output is a new file; existing diagnostic sources/results stay immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/v25_training_curve_20261010'
CURVE = BASE + '/curve.json'
REPORT = 'reports/v25_training_curve_20261010.md'
FIGURE = BASE + '/learning_curve.png'
EPOCHS = list(range(1, 31))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    path = (ROOT / name).resolve()
    need(path.is_relative_to(ROOT), 'Path outside this workspace')
    return path


def sha(name):
    return hashlib.sha256(safe(name).read_bytes()).hexdigest()


def record(name):
    return {'path': name, 'sha256': sha(name)}


def read(name):
    need(safe(name).suffix == '.json', 'JSON metadata only')
    return json.loads(safe(name).read_text(encoding='utf8'))


def write(name, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')
    with safe(name).open('xb') as stream:
        stream.write(data)


def contract(curve):
    exact = {'format': 'v25_prior_nn_learning_curve_v1', 'component': 'R107_prior_NN_native',
        'metric': 'unweighted_total_price_MAE', 'unit': 'TWD', 'evaluation_mode': 'eval',
        'epoch_selection': 'fixed_last_epoch', 'query_labels_used_for_fitting': False,
        'fit_rows': 321550, 'query_rows': 133697, 'diagnostic_refit_only': True,
        'original_v25_training_trajectory': False, 'full_v25_median_curve': False,
        'model_capacity_varied': False, 'Private_score': None, 'new_CSV': False,
        'selection_changed': False, 'fit_prior': 'group-fold OOF', 'query_prior': 'full outer-fit',
        'gap_also_reflects_prior_construction_and_time_shift': True,
        'historical_query_reused_not_blind': True}
    need(all(curve.get(key) == value for key, value in exact.items()), 'Curve scope/configuration differs')
    need(curve['epochs'] == EPOCHS and 'best_epoch' not in curve, 'Fixed 30 epochs required')
    need(curve['fit_period_label'] == '成交日期 < 2023-08-11'
         and curve['query_period_label'] == '2023-08-11～2024-08-10', 'Temporal split differs')
    for key in ('train_mae_twd', 'validation_mae_twd'):
        values = curve[key]
        need(len(values) == 30 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                     for v in values), 'Thirty finite nonnegative TWD MAEs required')
    need(bool(curve['sources']) and all(type(r) is dict and type(r.get('path')) is str
         and type(r.get('sha256')) is str and len(r['sha256']) == 64
         and set(r['sha256']) <= set('0123456789abcdef') for r in curve['sources']), 'Source identities missing')


def bundle():
    curve = read(CURVE)
    contract(curve)
    for rec in curve['sources']:
        need(record(rec['path']) == rec, 'Aggregate source identity changed: ' + rec['path'])
    receipts = {action: read(BASE + '/' + action + '_execution.json')
                for action in ('freeze', 'train', 'evaluate')}
    for action, receipt in receipts.items():
        need(receipt['exit_code'] == 0 and receipt['failure'] is None
             and receipt['action'] == action and receipt['new_CSV'] is False
             and receipt['selection_changed'] is False, 'Completed root execution required: ' + action)
        need(record(receipt['source']['path']) == receipt['source'], 'Root execution source changed')
    need(read(BASE + '/scoring_execution.json')['exit_code'] == 0, 'Completed scoring required')
    seal = read(BASE + '/seal.json')
    need(seal['status'] == 'thirty_snapshots_sealed_before_query_labels'
         and seal['checkpoint_epochs'] == EPOCHS and len(seal['snapshots']) == 30
         and seal['query_labels_used_for_fitting'] is False
         and seal['query_scores_computed'] is False, 'Pre-label 30-checkpoint seal required')
    request = read(BASE + '/request.json')
    for name, digest in request['protected_before'].items():
        need(sha(name) == digest, 'Protected historic artifact changed: ' + name)
    return curve, receipts


def relative_from_report(name):
    return '../' + name


def make_report(curve, receipts):
    train, query = curve['train_mae_twd'], curve['validation_mae_twd']
    rows = '\n'.join(f'| {epoch} | {t:,.6f} | {q:,.6f} | {q-t:+,.6f} |'
                     for epoch, t, q in zip(EPOCHS, train, query))
    sources = '\n'.join(f"- [{rec['path']}]({relative_from_report(rec['path'])})：`{rec['sha256']}`"
                        for rec in curve['sources'])
    return f'''# v25 先驗神經網路：實際訓練曲線與講解

日期：2026-10-10。這份圖來自**另一次固定設定的診斷重跑**，以 v25 的先驗神經網路分量觀察訓練過程。它補足原 R107 未保存逐回驗證 MAE 的缺口；不是原 v25 那次訓練的逐回紀錄，也不是六模型完整中位數融合的曲線。

![先驗神經網路的實際訓練曲線]({relative_from_report(FIGURE)})

## 可以直接上台說的版本

「這張圖是我們自己實際訓練出來的結果。橫軸是訓練回數，從第一回到第三十回；模型大小一直相同。縱軸是平均預測誤差，單位是元，越低越好。藍線看模型對練習資料的表現，橘線看它對後面一年交易的表現。

練習資料的誤差由約 {train[0]/10000:.1f} 萬元，降到約 {train[-1]/10000:.1f} 萬元；後面一年資料的誤差，則由約 {query[0]/10000:.1f} 萬元降到約 {query[-1]/10000:.1f} 萬元，但中間有波動。這提醒我們：模型對練習資料越來越熟，不代表對未來資料也會等幅進步。

這次沒有出現教科書常畫的明顯 U 形，所以我們照實呈現，不能為了符合示意圖去改結果。兩條線有差距，也不能只用差距大小就證明過擬合或證明沒有過擬合，因為兩段資料的時間、房屋組成和先驗建立方式也有差異。我們在看驗證價格以前就固定訓練三十回，沒有看完圖再挑誤差最低的一回。這張圖只描述其中一個神經網路；最終提交還有其他模型與校正融合。」

## 兩條線怎麼產生

| 項目 | 本次實際設定 |
| --- | --- |
| 訓練交易 | 成交日期嚴格早於 2023-08-11；{curve['fit_rows']:,} 筆 |
| 時間留出驗證交易 | 2023-08-11 至 2024-08-10（含首尾）；{curve['query_rows']:,} 筆 |
| 模型 | R107 prior NN 分量；原架構、seed 42、30 回、batch 2048、日期加權 SmoothL1，beta = 0.1 百萬元 |
| 橫軸 | 固定架構下的 epoch；沒有增加模型容量、層數或 dVC |
| 藍線／橘線共同量尺 | 每回快照均 `model.eval()`、關閉 dropout，換回總價後計算**不加權總價 MAE（新臺幣元）** |
| 真價精度 | 兩邊都用原 canonical raw 的 float64 元；fit 真價與原保存 fit_y 逐值一致，並核對 `train_y_million == float32(raw_fit_y / 1e6)` |
| 權重的角色 | 日期權重只用於原訓練損失。圖上兩條線都不加權，沒有把 SmoothL1 訓練損失與 MAE 混在同一張圖 |
| 標籤順序 | 先保存完整 30 個快照、封存身分並確認 train exit 0；之後才讀取本切分的驗證真價、離線評分 |
| 選回數方式 | 事前固定最後第 30 回；不 early stop、不選最低誤差 epoch、不掃參數 |

新增 trainer 與原 trainer 的科學流程相同，差異僅每回 `schedule.step()` 後保存一次 checkpoint。原九鍵 prepared、原訓練 metadata、四組 canonical/raw 列軸與日期來源皆以 SHA 綁定。真訓練不讀 query y；評分階段因原格式是 pickle，會一次 materialize 整張原價格表，再只取這個 confirmation 切分的 fit/query 列使用，不能把這稱為只載入驗證價格。完整流程見[事前計畫](v25_training_curve_plan_20261010.md)與[執行工具](../src/v25_training_curve_20261010.py)。

## 真實觀察，以及不能推論的事

| 量測 | 第 1 回 | 固定第 30 回 | 第 30 回減第 1 回 |
| --- | ---: | ---: | ---: |
| 訓練 MAE（元） | {train[0]:,.6f} | {train[-1]:,.6f} | {train[-1]-train[0]:+,.6f} |
| 時間留出驗證 MAE（元） | {query[0]:,.6f} | {query[-1]:,.6f} | {query[-1]-query[0]:+,.6f} |

兩條線的末回差為 **{query[-1]-train[-1]:,.6f} 元**。訓練誤差整體下降得多，驗證誤差下降較少且有波動；第 21–30 回驗證 MAE 落在 {min(query[20:]):,.2f}–{max(query[20:]):,.2f} 元。本次沒有明顯「後段持續上升」的典型 U 形。這是當次觀察，不足以宣稱模型完全不會過擬合；第 30 回也不是經由本圖選出的最佳回數。

老師原本用欠擬合／過擬合圖解釋模型容量與誤差的關係。這裡呈現的是**容量固定、隨訓練回數變化**的實測，所以不能把橫軸改稱模型複雜度或 dVC，也不能將這張圖當作容量實驗。藍線可以白話叫練習誤差，橘線叫時間留出驗證誤差；它們不是理論上的精確 E_in／母體 E_out。

fit 的價格先驗採 group-fold OOF，query 的先驗由完整 outer-fit 建立；因此兩邊除了時間、交易分布不同，先驗生成方式也不同。曲線差距同時含這些因素，不能全部歸因於過擬合。此歷史驗證窗先前已反覆用於研究，不是全新盲測；本次只做單 seed 診斷重跑，不能當作多 seed 穩健性檢查，亦不保證在其他硬體逐 bit 重現。

**1,777,133.98 元是這次先驗 NN 單一分量的驗證 MAE。** 原 R107 同窗口 v25 完整融合約 1,381,702.87 元，是不同預測物的離線結果，不能混稱本圖末點。Kaggle Public 與未知 Private 也不是這張曲線的驗證資料。本次沒有新提交 CSV，原 v25 CSV、原選定、舊 release、原離線 gate 及歷史來源保持；新圖不構成改選模型或 epoch 的依據。[目前方法與版本界線](v25_method_and_code_20261010.md)。

## 執行與程式入口

主執行緒完成 freeze、train、evaluate；保存的 phase 牆鐘時間分別為 {receipts['freeze']['elapsed_seconds']:.3f}、{receipts['train']['elapsed_seconds']:.3f}、{receipts['evaluate']['elapsed_seconds']:.3f} 秒。train phase 含驗證輸入、單次 fit、30 快照檢查與封存；evaluate phase 含取得本切分標籤及逐快照評分，不能把整個 phase 說成純 GPU 訓練時間。

- [事前固定計畫](v25_training_curve_plan_20261010.md)：先定切分、原設定、30 快照與解讀限制。
- [獨立 trainer](../src/round107_training_curve_train_20261010.py)：原科學碼加 6 行快照保存，未改原 trainer。
- [freeze／train／evaluate 工具](../src/v25_training_curve_20261010.py)：三階段入口；標籤讀取受 30 快照 seal 和 train exit 0 限制。
- [本圖與白話報告 builder](../src/summarize_v25_training_curve_20261010.py)：`--build` 只讀聚合 JSON 和身分／收據，不讀模型、原資料或逐筆標籤；`--check-vectors` 是另外一次只讀的 60 向量 MAE 核對，不做 fit。
- [曲線聚合 JSON]({relative_from_report(CURVE)})、[根 train 收據]({relative_from_report(BASE + '/train_execution.json')})、[根 evaluate 收據]({relative_from_report(BASE + '/evaluate_execution.json')})、[圖表建置收據]({relative_from_report(BASE + '/report_build_execution.json')})、[60 向量核對收據]({relative_from_report(BASE + '/curve_vector_review.json')})。

30 個模型快照、逐筆預測與標籤留在本機私有位置，不納 GitHub；可交付來源、聚合 30 點、此圖、說明及彙總身分收據。這不是從 raw 到乾淨環境的一鍵重訓驗證。

## 完整 30 點（同為不加權總價 MAE，元）

| 回數 | 訓練 MAE | 時間留出驗證 MAE | 驗證減訓練 |
| ---: | ---: | ---: | ---: |
{rows}

## 聚合來源身分

曲線 JSON SHA256：`{sha(CURVE)}`。來源依本機固定紀錄核對如下；列出 JSON 身分不代表私有逐筆資料將上傳。

{sources}
'''


def draw(curve):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import font_manager, ticker
    font = Path('C:/Windows/Fonts/msjh.ttc')
    need(font.is_file(), 'Expected Traditional Chinese plot font missing')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=180)
    fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.83)
    ax.plot(EPOCHS, curve['train_mae_twd'], color='#2266b0', marker='o', markersize=3.5,
            linewidth=2.0, label='訓練資料（eval，關閉 dropout）')
    ax.plot(EPOCHS, curve['validation_mae_twd'], color='#d06a19', marker='o', markersize=3.5,
            linewidth=2.0, label='後一年驗證資料（eval，關閉 dropout）')
    ax.axvline(30, color='#666666', linewidth=1, linestyle='--', alpha=.65)
    ax.set(xlabel='訓練回數（epoch；架構固定）', ylabel='平均絕對誤差 MAE（新臺幣元；越低越好）',
           xlim=(.5, 30.7), ylim=(0, max(curve['train_mae_twd'] + curve['validation_mae_twd']) * 1.09))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis='y', alpha=.23)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=10.5)
    fig.suptitle('先驗神經網路：固定架構的實際訓練曲線', fontsize=19, x=.12, ha='left', y=.96)
    fig.text(.12, .875, '321,550 筆訓練  ·  133,697 筆時間留出驗證  ·  事前固定第 30 回', fontsize=11)
    fig.text(.12, .09, '單一 NN 分量的診斷重跑；不是原 v25 訓練紀錄、完整融合或 Private 分數。', fontsize=10)
    fig.text(.12, .055, '兩線皆不加權總價 MAE；差距也含 OOF／full-fit 先驗差異及時間變動。', fontsize=10)
    with safe(FIGURE).open('xb') as stream:
        fig.savefig(stream, format='png', facecolor='white')
    plt.close(fig)


def build():
    need(not safe(REPORT).exists() and not safe(FIGURE).exists(), 'Preserve existing report/figure')
    curve, receipts = bundle()
    report = make_report(curve, receipts)
    draw(curve)
    with safe(REPORT).open('xb') as stream:
        stream.write(report.encode('utf8'))
    return {'outputs': [record(REPORT), record(FIGURE)], 'curve': record(CURVE),
        'points_per_line': 30, 'raw_data_or_labels_or_models_loaded': False, 'new_fits': 0,
        'epoch_selection': 'fixed_last_epoch', 'source': record('src/' + Path(__file__).name)}


def check_vectors():
    import numpy as np
    curve, _ = bundle()
    labels, manifest = read(BASE + '/evaluation_labels.json'), read(BASE + '/private_prediction_manifest.json')
    need(manifest['all_predictions_eval_mode'] and manifest['query_labels_used_for_fitting'] is False
         and manifest['new_fits'] == 0 and len(manifest['epochs']) == 30, 'Saved prediction provenance differs')
    loaded = []

    def load_once(rec):
        need(safe(rec['path']).suffix == '.npy', 'Only sealed private NPY arrays allowed')
        payload = safe(rec['path']).read_bytes()
        need(hashlib.sha256(payload).hexdigest() == rec['sha256'], 'NPY file identity differs')
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        need(str(array.dtype) == rec['dtype'] == 'float64' and list(array.shape) == rec['shape']
             and array.ndim == 1 and np.isfinite(array).all(), 'NPY shape/dtype/finite check differs')
        need(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == rec['values_sha256'],
             'NPY values identity differs')
        loaded.append(rec)
        return array

    fit_y, query_y = load_once(labels['fit_y']), load_once(labels['query_y'])
    need(len(fit_y) == curve['fit_rows'] and len(query_y) == curve['query_rows']
         and (fit_y > 0).all() and (query_y > 0).all(), 'Truth dimensions/values differ')
    points = []
    for epoch, entry in zip(EPOCHS, manifest['epochs']):
        need(entry['epoch'] == epoch, 'Saved vector epoch order differs')
        train, query = load_once(entry['train']), load_once(entry['query'])
        need(train.shape == fit_y.shape and query.shape == query_y.shape
             and (train >= 0).all() and (query >= 0).all(), 'Prediction/target alignment differs')
        t = float(np.mean(np.abs(train - fit_y), dtype=np.float64))
        q = float(np.mean(np.abs(query - query_y), dtype=np.float64))
        need(t == curve['train_mae_twd'][epoch-1] and q == curve['validation_mae_twd'][epoch-1],
             'Recomputed MAE not exactly equal at epoch ' + str(epoch))
        points.append({'epoch': epoch, 'train_mae_twd': t, 'validation_mae_twd': q, 'exact_equal': True})
    return {'status': 'all_60_saved_vector_MAEs_exact', 'curve': record(CURVE), 'points': points,
        'saved_prediction_arrays_loaded_once_each': 60, 'label_arrays_loaded_once_each': 2,
        'file_and_values_SHA_dtype_shape_verified': True, 'inputs': loaded,
        'new_fits': 0, 'model_weights_loaded': 0, 'GPU_calls': 0, 'raw_pickle_materializations': 0,
        'new_CSV': False, 'selection_changed': False, 'source': record('src/' + Path(__file__).name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', action='store_true')
    group.add_argument('--check-vectors', action='store_true')
    args = parser.parse_args()
    action = 'report_build' if args.build else 'curve_vector_review'
    output = BASE + '/' + (action + '_execution.json' if args.build else action + '.json')
    need(not safe(output).exists(), 'Preserve prior execution receipt; no overwrite or automatic retry')
    tick, failure, result = time.monotonic(), None, None
    try:
        result = build() if args.build else check_vectors()
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        write(output, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'action': action,
            'exit_code': 1 if failure else 0, 'failure': failure, 'elapsed_seconds': time.monotonic()-tick,
            'automatic_retry': False, 'result': result})
    print(json.dumps({'action': action, 'exit_code': 0, 'receipt': record(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
