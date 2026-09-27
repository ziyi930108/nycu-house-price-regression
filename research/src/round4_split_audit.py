"""Audit one auxiliary annual split without loading or evaluating any model."""
from __future__ import annotations
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = '2021-08-11', '2022-08-10'
EPOCH = pd.Timestamp('2000-01-01')
GROUP_COLUMNS = ['address_city', 'address_district', 'project_name']
OUTPUT = ROOT / 'outputs/round4/audit/auxiliary_split_design.json'
REPORT = ROOT / 'reports/round4_validation_plan.md'
MISSING = '__MISSING__'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def index_hash(index):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(index, dtype='<i8'), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def groups(frame):
    result = frame[GROUP_COLUMNS].copy()
    for column in GROUP_COLUMNS:
        values = result[column].astype('string').fillna(MISSING).str.strip()
        result[column] = values.mask(values.isin(['', MISSING]), MISSING)
    return result


def stratify(fit, query):
    fg, qg = groups(fit), groups(query)
    fit_named = fg['project_name'].ne(MISSING).to_numpy()
    query_named = qg['project_name'].ne(MISSING).to_numpy()
    known = pd.MultiIndex.from_frame(fg.loc[fit_named]).unique()
    query_keys = pd.MultiIndex.from_frame(qg)
    seen = query_named & query_keys.isin(known)
    new = query_named & ~seen
    missing = ~query_named
    masks = {'seen': seen, 'new': new, 'missing': missing}
    counts = {name: {'rows': int(mask.sum()), 'fraction': float(mask.mean()),
                     'unique_geographic_projects': int(query_keys[mask].nunique()) if name != 'missing' else None}
              for name, mask in masks.items()}
    new_fit_overlap = int(query_keys[new].isin(known).sum())
    named_count = int(query_named.sum())
    assert not new_fit_overlap and sum(item['rows'] for item in counts.values()) == len(query)
    name_known = set(fg.loc[fit_named, 'project_name'])
    audit = {
        'new_rows_with_identical_geographic_project_in_fit': new_fit_overlap,
        'new_rows_with_name_seen_in_other_geography': int((new & qg['project_name'].isin(name_known).to_numpy()).sum()),
        'named_query_rows_with_missing_city_or_district': int((query_named & qg[['address_city','address_district']].eq(MISSING).any(axis=1).to_numpy()).sum()),
        'named_query_rows': named_count,
        'fit_unique_named_geographic_projects': int(known.nunique()),
        'fit_missing_project_rows': int((~fit_named).sum()),
    }
    return counts, masks, audit


def price_summary(y):
    y = np.asarray(y, dtype='float64')
    if not len(y):
        return {'rows': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None,
                'p99': None, 'minimum': None, 'maximum': None, 'at_least_100m_rows': 0}
    assert np.isfinite(y).all() and (y > 0).all()
    return {'rows': len(y), 'mean': float(y.mean()), 'median': float(np.median(y)),
            'p10': float(np.quantile(y, .1)), 'p90': float(np.quantile(y, .9)),
            'p99': float(np.quantile(y, .99)), 'minimum': float(y.min()), 'maximum': float(y.max()),
            'at_least_100m_rows': int((y >= 100_000_000).sum())}


def distribution(frame, column):
    values = frame[column].astype('string').fillna(MISSING)
    return {str(key): {'rows': int(value), 'fraction': float(value / len(frame))}
            for key, value in values.value_counts(dropna=False).sort_index().items()}


def total_variation(left, right):
    return float(.5 * sum(abs(left.get(key, {}).get('fraction', 0.) - right.get(key, {}).get('fraction', 0.))
                          for key in set(left) | set(right)))


def main():
    input_paths = [ROOT / 'data_cache/train.pkl', ROOT / 'data_cache/train_features.pkl',
                   ROOT / 'data_cache/valid_features.pkl', ROOT / 'data_cache/test_features.pkl',
                   ROOT / 'reports/feature_schema.json', ROOT / 'src/house_features.py',
                   ROOT / 'src/model_common.py', ROOT / 'reports/round3_results.json']
    input_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in input_paths}
    schema = read(ROOT / 'reports/feature_schema.json')
    assert schema['feature_source_sha256'] == input_hashes['src/house_features.py']
    # Read trusted local caches directly: no cache regeneration or model_common side effects.
    raw = pd.read_pickle(ROOT / 'data_cache/train.pkl')
    features = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
    valid = pd.read_pickle(ROOT / 'data_cache/valid_features.pkl')
    test = pd.read_pickle(ROOT / 'data_cache/test_features.pkl')
    assert len(raw) == len(features) and list(features.columns) == schema['columns']
    for split, frame in [('train', features), ('valid', valid), ('test', test)]:
        metadata = read(ROOT / f'data_cache/{split}.json')
        assert schema['splits'][split]['input_source_sha256'] == metadata['source_sha256']
        assert list(frame.columns) == schema['columns']
    keep = ~raw.duplicated(subset=[column for column in raw if column != '編號']).to_numpy()
    y = raw.loc[keep, '總價元'].to_numpy(dtype='float64')
    x = features.loc[keep].reset_index(drop=True)
    days = x['trade_day_index'].to_numpy(dtype='float64')
    start = (pd.Timestamp(START) - EPOCH).days
    end = (pd.Timestamp(END) - EPOCH).days
    fi = np.flatnonzero(np.isfinite(days) & (days < start))
    qi = np.flatnonzero(np.isfinite(days) & (days >= start) & (days <= end))
    assert len(fi) > 10_000 and len(qi) > 10_000 and not np.intersect1d(fi, qi).size
    fit, query = x.iloc[fi], x.iloc[qi]
    strata, masks, leak_audit = stratify(fit, query)
    test_strata, _, test_group_audit = stratify(pd.concat([x, valid], ignore_index=True), test)
    dates = EPOCH + pd.to_timedelta(days[qi], unit='D')
    all_days = pd.date_range(START, END, freq='D')
    observed = pd.DatetimeIndex(dates).unique()
    quarters = np.minimum(((days[qi] - start) // 92).astype(int), 3)
    city_query, city_test = distribution(query, 'address_city'), distribution(test, 'address_city')
    geo_query = query['address_city'].astype(str) + '|' + query['address_district'].astype(str)
    geo_test = test['address_city'].astype(str) + '|' + test['address_district'].astype(str)
    district_query = distribution(pd.DataFrame({'key': geo_query}), 'key')
    district_test = distribution(pd.DataFrame({'key': geo_test}), 'key')
    old_splits = {'development': ('2022-08-11','2023-08-10'), 'confirmation': ('2023-08-11','2024-08-10')}
    overlap = {}
    for name, (lower, upper) in old_splits.items():
        check = np.flatnonzero(np.isfinite(days) & (days >= (pd.Timestamp(lower) - EPOCH).days)
                              & (days <= (pd.Timestamp(upper) - EPOCH).days))
        old_fit = np.flatnonzero(np.isfinite(days) & (days < (pd.Timestamp(lower) - EPOCH).days))
        overlap[name] = {'auxiliary_check_vs_historical_check_rows': int(np.intersect1d(qi, check).size),
                         'auxiliary_check_previously_in_historical_fit_rows': int(np.intersect1d(qi, old_fit).size)}
    history = read(ROOT / 'reports/round3_results.json')
    controls = {'source': 'reports/round3_results.json', 'historically_used': True,
                'development': next(row for row in history['development'] if row['candidate'] == 'v2_control'),
                'confirmation': history['confirmation_control']}
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'One fixed auxiliary annual confirmation window; design audit only, no model evaluation.',
        'split_name': 'auxiliary_2021', 'recommended_extra_split_count': 1,
        'start_inclusive': START, 'end_inclusive': END, 'fit_rule': 'finite trade_day_index < start; deduplicated train only',
        'fit_rows': len(fi), 'check_rows': len(qi), 'total_deduplicated_train_rows': len(x),
        'excluded_invalid_date_rows': int((~np.isfinite(days)).sum()),
        'fit_first_date': str((EPOCH + pd.to_timedelta(days[fi].min(), unit='D')).date()),
        'fit_last_date': str((EPOCH + pd.to_timedelta(days[fi].max(), unit='D')).date()),
        'check_first_date': str(dates.min().date()), 'check_last_date': str(dates.max().date()),
        'calendar_days': len(all_days), 'observed_days': len(observed),
        'unobserved_dates': [str(day.date()) for day in all_days.difference(observed)],
        'check_rows_by_month': {str(k): int(v) for k, v in pd.Series(dates.strftime('%Y-%m')).value_counts().sort_index().items()},
        'check_rows_by_horizon_quarter': {str(q+1): int((quarters == q).sum()) for q in range(4)},
        'index_reference': 'zero-based positions in load_data deduplicated X_train; SHA is little-endian int64 NPY bytes',
        'fit_indices_sha256': index_hash(fi), 'check_indices_sha256': index_hash(qi),
        'group_definition': GROUP_COLUMNS, 'strata': strata, 'test_strata_relative_train_plus_valid': test_strata,
        'group_leakage_checks': leak_audit, 'test_group_checks': test_group_audit,
        'row_overlap_with_historical_splits': overlap, 'historical_reference_controls': controls,
        'price_unit': 'TWD', 'fit_price_summary': price_summary(y[fi]), 'check_price_summary': price_summary(y[qi]),
        'price_summary_by_stratum': {name: price_summary(y[qi][mask]) for name, mask in masks.items()},
        'city_distributions': {'auxiliary': city_query, 'test': city_test},
        'city_total_variation_vs_test': total_variation(city_query, city_test),
        'district_total_variation_vs_test': total_variation(district_query, district_test),
        'adopted_auxiliary_veto_protocol': {
            'freeze_one_candidate_before_auxiliary_model_fit': True,
            'freeze_sources_parameters_features_iteration_counts_and_formulas': True,
            'baseline': 'refit v2_control using only auxiliary fit rows',
            'max_relative_deterioration_overall_and_fixed_test_strata_weighted_mae': .03,
            'horizon_quarter_metrics_are_diagnostic_not_auxiliary_veto': True,
            'max_relative_deterioration_each_stratum_with_at_least_500_rows': .05,
            'if_rejected': 'retain v2; do not test another candidate on this window in the same round',
            'missing_stratum_zero_weight_and_empty_mae_null': True,
            'auxiliary_prices_not_for_tuning_clipping_or_target_transforms': True,
        },
        'limitations': [
            'All historical labels have previously participated in older fitting; not a historically untouched test.',
            'This earlier annual window differs from future test in price level and project/city composition.',
            'Seen-project overlap is intentional chronological forecasting; only absent named geographic keys are called new.',
            'Project aliases or location parsing errors can affect entity identity; same name in another geography is separately counted.',
            'One extra period is a veto/sensitivity check, not an independent guarantee of Private performance.',
        ],
        'input_sha256': input_hashes, 'script_sha256': sha256(Path(__file__)),
        'evaluated_models': [], 'loaded_prediction_files': [], 'test_labels_read': False, 'external_property_data_used': False,
    }
    # Guard against source/cache changes during this read-only audit.
    assert all(sha256(ROOT / path) == digest for path, digest in input_hashes.items())
    proportions = '／'.join(f"{strata[name]['rows']:,}（{100*strata[name]['fraction']:.2f}%）" for name in ['seen','new','missing'])
    test_props = '／'.join(f"{100*test_strata[name]['fraction']:.2f}%" for name in ['seen','new','missing'])
    prices = result['check_price_summary']
    text = f'''# 第四輪輔助驗證計畫

**唯一推薦：新增 2021-08-11～2022-08-10 的完整年度 rolling-origin 確認切分。** 它的檢查列與第三輪兩個檢查年度均不重疊，訓練量較小，可在候選凍結後增加一次不同年度的敏感度檢查。此稽核未訓練模型、未載入預測、未查看 test 標籤；價格概況只用於說明資料。

## 切分與資料規模

| 項目 | 固定定義／數量 |
|---|---|
| Fit | 去重 train 中有效交易日期早於 2021-08-11；{len(fi):,} 筆 |
| Check | 2021-08-11～2022-08-10，含首尾；{len(qi):,} 筆 |
| 覆蓋 | {len(all_days)} 天中有交易資料 {len(observed)} 天；不把缺失日期補進切分 |
| 不使用 | 原 valid 不混入此年度 fit；切分後或 test 的標籤不用於訓練 |

按「縣市＋行政區＋建案名稱」定義，Check 的 **已見／有名稱的新建案／缺名** 為 **{proportions}**。test 相對完整 train＋valid 的比例為 **{test_props}**。計分時另報固定 test 比例加權 MAE；缺名權重為 0，空組 MAE 應記 null，不造出 0 分。

Check 價格中位數約 **{prices['median']:,.0f} 元**，P10／P90／P99 約 **{prices['p10']:,.0f}／{prices['p90']:,.0f}／{prices['p99']:,.0f} 元**；至少一億元的交易有 **{prices['at_least_100m_rows']:,} 筆**。這些數值不是截斷、排除高價或校正倍率的依據。縣市／行政區分布對 test 的總變異距離為 **{result['city_total_variation_vs_test']:.4f}／{result['district_total_variation_vs_test']:.4f}**，因此額外年度只作穩健性檢查，不當作未来分數預報。

## 防止同建案資料造成假新案

先依日期切分，再只從 **Fit** 建立非缺名建案集合；Check 中相同完整鍵出現在 Fit 的列必須算「已見」。標為新案的地理鍵在 Fit 中的交集已核對為 **{leak_audit['new_rows_with_identical_geographic_project_in_fit']} 列**。不能先抽建案的一部分列再把其餘列留在 Fit，卻聲稱做了跨建案留出。本方案沒有額外人工 group holdout；同案過去成交可以合法用來預測其未來成交，但如實列為已見組。

另有 **{leak_audit['new_rows_with_name_seen_in_other_geography']:,} 筆**「新地理鍵但 Fit 見過相同案名」，完整鍵中的地理欄缺失則涉及 **{leak_audit['named_query_rows_with_missing_city_or_district']:,} 筆**具名 Check。Fit 本身另有 **{leak_audit["fit_missing_project_rows"]:,} 筆（{100*leak_audit["fit_missing_project_rows"]/len(fi):.2f}%）缺案名**，因此「新」精確地表示 Fit 未見的具名地理鍵，並非已證明現實中的建案從未成交。這些是實體辨識限制，不把相同案名跨縣市硬合併；實際建案別名仍可能影響分類。

## 固定執行方式與成本

1. 保留第三輪 **2022-08-11～2023-08-10** 與 **2023-08-11～2024-08-10** 指標，明確標為「歷史已使用」。可以在它們上面開發，但不得重新包裝為新的獨立確認。
2. 在讀取本額外年度的任何模型分數前，只凍結 **一個** 候選，以及特徵、公式、超參數、輪數與來源雜湊。不要用此年度 early stopping、選混合權重、改日期窗、學價格截斷值或人工參考價。
3. 對照與候選都只使用上述 Fit 重新估計：包括類別編碼、補值、鄰里價格參考、趨勢、分位界限。不可套用曾看過 Check 標籤的全量 v1／v2 模型。兩個共用底模各訓練一次；若候選只改參考層，可重用這兩份 **Fit 專用** 預測，避免整個候選網格重訓。
4. 主流程採納的額外否決門檻：整體與固定三組加權 MAE 相對同 Fit 的 v2_control，各自最多退步 **3%**；樣本至少 **500** 的建案組最多退步 **5%**。四個預測季度保留診斷，不另作此窗口的否決門檻。3% 容忍度在評分前決定，用來承認本期 85.70% 新案及較多早期缺名資料的不同市場組成；不因此把微幅改善當成獲勝。分母、空組與季度定義沿用第三輪，正式規則須在評分前凍結，不能看結果後放寬。
5. 未通過就保留 v2；同一輪不再拿第二候選重試此窗口。若未來反覆使用，它就變成另一個開發年度，須在紀錄中降級其確認地位。

這不是歷史上完全未見的測試：本 Check 的 **{len(qi):,} 筆**標籤都曾落入第三輪開發年度的 Fit，且先前全量訓練亦已使用歷史資料。優點是把新的候選決策與另一段尚未評分的年度回測分開；無法消除所有歷史選擇偏差，也不能保證 Private 成績。相較新增隨機地理建案留出，這個唯一額外窗口保留接近一年的預測長度，同時自然包含大量新建案，不把價格年代差异藏在隨機拆分中。

完整月份、季度、三組、價格摘要、索引 SHA 與資料 SHA 見 [auxiliary_split_design.json](../outputs/round4/audit/auxiliary_split_design.json)。第三輪既有對照數字另存於該 JSON 的 historical_reference_controls，原始 [round3_results.json](round3_results.json) 不變；可另讀 [第三輪完整報告](optimization_round3_report.md)。本次僅新增本計畫、資料稽核 JSON 與 [重跑稽核程式](../src/round4_split_audit.py)。
'''.replace('未来','未來').replace('须','須').replace('差异','差異')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    REPORT.write_text(text, encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['fit_rows','check_rows','strata','check_price_summary',
          'city_total_variation_vs_test','district_total_variation_vs_test','group_leakage_checks','row_overlap_with_historical_splits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
