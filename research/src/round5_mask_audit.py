"""Pure-X round-five mask distribution audit; never fit or score price models.

The unchanged shared loaders transiently load supplied train/valid labels solely
for their existing price-aware exact-row deduplication. Those labels/raw frames
are discarded before the mask audit. Test has no price input. The script does
not open confirmation/auxiliary scores or prediction files.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key]="3"
import sys
sys.dont_write_bytecode=True
from datetime import datetime,timezone
import hashlib
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model_common import ROOT,load_data,full_training_data,temporal_indices
from round5_project_mask import (MASK_RULES,MASK_TOKEN,IDENTIFIER_COLUMNS,
                                fit_feature_state,transform_features,project_keys,query_mask)

HELPER=ROOT/'src/round5_project_mask.py'
EXPECTED_HELPER_SHA='78509888884c658510108887a54a99356515ac381569662363db74b4315c56d0'
OUT=ROOT/'outputs/round5/audit/mask_distribution.json'
DOC=ROOT/'reports/round5_preprocessing_record.md'
SPLITS={'development':('2022-08-11','2023-08-10'),
        'confirmation':('2023-08-11','2024-08-10'),
        'auxiliary':('2021-08-11','2022-08-10')}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def frame_sha(frame):
    h=hashlib.sha256()
    h.update(json.dumps(list(frame.columns),ensure_ascii=False).encode('utf-8'))
    h.update(json.dumps([str(t) for t in frame.dtypes]).encode('ascii'))
    h.update(pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype='uint64').tobytes())
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


def proportion(number,denominator):
    return float(number/denominator) if denominator else None



def scorer_key_function():
    """Read only the two X-only scorer definitions; never import the evaluator."""
    source=ROOT/'src/round3_evaluate.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
           and node.name in ('norm','project_keys')]
    assert {node.name for node in nodes}=={'norm','project_keys'}
    namespace={'pd':pd}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['project_keys']


SCORER_PROJECT_KEYS=scorer_key_function()


def compare_scorer_keys(X,label):
    helper_keys,helper_missing=project_keys(X)
    scorer_keys,scorer_missing=SCORER_PROJECT_KEYS(X)
    key_diff=helper_keys.to_numpy()!=scorer_keys.to_numpy()
    missing_diff=helper_missing.to_numpy()!=scorer_missing.to_numpy()
    different=key_diff|missing_diff
    examples=[]
    for position in np.flatnonzero(different)[:5]:
        examples.append({'row_position':int(position),
                        'address_city':str(X.iloc[position]['address_city']),
                        'address_district':str(X.iloc[position]['address_district']),
                        'project_name':str(X.iloc[position]['project_name']),
                        'helper_key':str(helper_keys.iloc[position]),
                        'scorer_key':str(scorer_keys.iloc[position]),
                        'helper_missing':bool(helper_missing.iloc[position]),
                        'scorer_missing':bool(scorer_missing.iloc[position])})
    report={'scope':label,'rows':len(X),'different_keys':int(key_diff.sum()),
            'different_missing_flags':int(missing_diff.sum()),
            'different_rows':int(different.sum()),'examples':examples,
            'comparison_is_rowwise_ignoring_index_labels':True}
    assert not different.any(), report
    return report


def checked_transform(X,state):
    """Check a real feature frame without labels, model prediction, or side effects."""
    input_sha=frame_sha(X)
    state_sha=content_sha(state)
    flags=query_mask(X,state)
    transformed=transform_features(X,state)
    other=[name for name in X if name not in IDENTIFIER_COLUMNS]
    categories=[name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
    assert len(X.columns)==87 and len(categories)==21 and len(other)==83
    assert list(transformed.columns)==list(X.columns)
    assert transformed.index.equals(X.index) and transformed.dtypes.equals(X.dtypes)
    pd.testing.assert_frame_equal(transformed[other],X[other])
    pd.testing.assert_frame_equal(transformed.loc[~flags],X.loc[~flags])
    assert transformed.loc[flags,list(IDENTIFIER_COLUMNS)].eq(MASK_TOKEN).all().all()
    assert frame_sha(X)==input_sha and content_sha(state)==state_sha
    assert all(str(X[name].dtype)=='float32' for name in X if name not in categories)
    return flags,{'feature_count':87,'categorical_count':21,'numeric_count':66,
                  'numeric_dtype':'float32','column_order_unchanged':True,'dtypes_unchanged':True,
                  'index_unchanged':True,'other_83_columns_unchanged':True,
                  'unmasked_rows_unchanged':True,'all_four_identifiers_masked_where_required':True,
                  'input_unchanged':True,'feature_state_unchanged':True,
                  'input_feature_values_sha256':input_sha}


def audit_stage(name,Xfit,Xquery,dates=None):
    scorer_checks={'fit':compare_scorer_keys(Xfit,name+'_fit'),
                   'query':compare_scorer_keys(Xquery,name+'_query')}
    state=fit_feature_state(Xfit)
    assert not state['target_used'] and not state['query_rows_used']
    original_state=content_sha(state)
    fit_mask,fit_checks=checked_transform(Xfit,state)
    qmask,query_checks=checked_transform(Xquery,state)
    fk,fmissing=project_keys(Xfit)
    qk,qmissing=project_keys(Xquery)
    fit_known=set(state['project_mask_flags'])
    assert fit_known==set(fk.loc[~fmissing])
    assert not any(key.endswith('|__MISSING__') for key in fit_known)
    fm=fmissing.to_numpy()
    qm=qmissing.to_numpy()
    qseen=~qm & qk.isin(fit_known).to_numpy()
    qnew=~qm & ~qk.isin(fit_known).to_numpy()
    assert qmask[qnew|qm].all()
    assert np.array_equal(qmask[qseen],qk[qseen].map(state['project_mask_flags']).to_numpy(dtype=bool))
    fit_named=fk.loc[~fmissing]
    masked_fit_keys={k for k,v in state['project_mask_flags'].items() if v}
    assert np.array_equal(fit_mask[~fm],fit_named.isin(masked_fit_keys).to_numpy())
    # A JSON wire-format round trip must preserve transformation, including unseen keys.
    restored=json.loads(json.dumps(state,ensure_ascii=False,allow_nan=False))
    pd.testing.assert_frame_equal(transform_features(Xquery,state),transform_features(Xquery,restored))
    assert content_sha(state)==original_state
    groups={}
    for group,mask in [('seen_named',qseen),('new_named',qnew),('missing_name',qm)]:
        groups[group]={'rows':int(mask.sum()),'row_fraction':proportion(mask.sum(),len(Xquery)),
                       'masked_rows':int((mask&qmask).sum()),'mask_fraction':proportion((mask&qmask).sum(),mask.sum()),
                       'distinct_named_projects':int(qk[mask].nunique()) if group!='missing_name' else None}
    summary=state['fit_summary']
    report={'stage':name,'query_dates_inclusive':dates,
        'fit':{'rows':len(Xfit),'named_projects':summary['named_projects'],
               'masked_named_projects':summary['masked_named_projects'],
               'masked_named_project_fraction':proportion(summary['masked_named_projects'],summary['named_projects']),
               'named_rows':int((~fm).sum()),'missing_name_rows':int(fm.sum()),
               'masked_named_rows':int((fit_mask&~fm).sum()),
               'masked_named_row_fraction':proportion((fit_mask&~fm).sum(),(~fm).sum()),
               'masked_rows':int(fit_mask.sum()),'masked_row_fraction':float(fit_mask.mean()),
               'missing_names_not_grouped_as_one_project':True},
        'query':{'rows':len(Xquery),'masked_rows':int(qmask.sum()),'masked_row_fraction':float(qmask.mean()),'groups':groups},
        'fit_checks':fit_checks,'query_checks':query_checks,
        'scorer_key_consistency':scorer_checks,
        'state_checks':{'fit_project_set_exact':True,'all_named_fit_project_rows_share_flag':True,
                        'seen_query_reuses_fit_flags':True,'all_new_named_queries_masked':True,
                        'all_missing_queries_masked':True,'json_roundtrip_exact':True,
                        'query_cannot_change_state':True,'no_label_argument_passed_to_state':True,
                        'state_content_sha256':state['state_content_sha256'],'state_saved_as_model_artifact':False}}
    print(json.dumps({'stage':name,'fit_rows':len(Xfit),'query_rows':len(Xquery),
                      'fit_mask_rate':report['fit']['masked_row_fraction'],
                      'query_mask_rate':report['query']['masked_row_fraction'],'feature_checks':'passed'},ensure_ascii=True),flush=True)
    return report


def write_doc(result):
    pct=lambda v:'無樣本' if v is None else f'{100*v:.2f}%'
    clean=result['existing_preprocessing']
    lines=['# 第五輪整建案遮罩：資料處理與純特徵稽核紀錄','',
      '本文件記錄本輪固定遮罩方法與真實資料的 X-only 檢查。没有訓練價格模型、產生房價預測或計算 MAE；下列比例不能解讀為模型改善或候選已獲採用。','',
      '## 沿用資料清理，價格僅供原有去重規則','',
      f"原 train 455,787 列移除 3 列夾入英文欄名後，特徵快取為 {clean['train_rows_before_dedup']:,} 列。原 load_data() 以排除編號、包含價格的其餘原始欄位判斷完全重複，再移除 {clean['removed_exact_train_duplicates_without_id']:,} 列，實際 train 為 {clean['train_rows']:,} 列。valid {clean['valid_rows']:,} 列、test {clean['test_rows']:,} 列保持原樣。",
      f"全量 fit 仍按原 full_training_data() 合併去重：train＋valid 共 {clean['train_rows']+clean['valid_rows']:,} 列，再排除 {clean['combined_duplicates_removed']:,} 列，得到 {clean['full_fit_rows']:,} 列。年度切分在原去重後 train 上依日期進行，無效日期不進入年度 fit/query；沒有因本輪遮罩新增刪列。",'',
      '為完全沿用上述規則，既有載入器會暫時讀入 supplied train／valid 的價格；本稽核只允許其參與原有完全重複判定，沒有取價格均值、分位数、誤差或建案 target encoding。取出 X 後即從本稽核的資料字典移除 y 與 raw frames。test 沒有提供價格，也未讀取任何 confirmation／auxiliary 評分或預測檔。',
      '遮罩狀態的函式只接收 fit X；每個年度與全量 fit 都各自建立，沒有重用包含較晚列的全量字典。先前研究的 feature-cache 描述性計數尚含 529 列去重前重複；本文件改用正式去重後 X，因此數字不應直接混用。','',
      '## 本輪唯一新處理：整案一致的四欄遮罩','',
      '非缺失建案鍵沿用原 address_city｜address_district｜完整 project_name。固定計算 SHA256("round5|42|" + key)，取前 8 bytes 為 unsigned big-endian 整數，小於 2**63 的整個建案遮罩。50% 是每個具名建案的固定 hash 規則，並非要求每批成交列恰好一半；不能為湊齊比例而改 seed。',
      '被遮罩的四欄為 project_name、address_raw、building_and_unit、building_block，統一替換成新的保留字 __R5_PROJECT_MASK__，與原始缺失值 __MISSING__ 區分。原 87 欄及 21 類別／66 個 float32 數值的 schema 不變，其他 83 欄、欄序、dtype、index 與原 X 均保持。',
      'query 已見建案沿用 fit 的遮罩狀態；具名新建案一律遮罩。缺案名的 fit/query 列也一律遮罩，但不將所有缺案名列合併成一個實體建案或加入建案字典。每次 query 僅查詢，不能增加或修改 state。v2 reference 與舊模型仍必須使用原 X，不接收遮罩版本。','',
      '## 實際 fit 遮罩分布','',
      '| fit 範圍 | fit 列數 | 具名建案數 | 遮罩具名建案 | 遮罩建案比例 | 缺案名列數 | 全部遮罩列數 | 全部遮罩列比例 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    names={'development':'開發年度以前','confirmation':'確認年度以前','auxiliary':'輔助年度以前','full_fit_to_test':'全 train＋valid'}
    for stage in result['stages']:
        f=stage['fit']
        lines.append(f"| {names[stage['stage']]} | {f['rows']:,} | {f['named_projects']:,} | {f['masked_named_projects']:,} | {pct(f['masked_named_project_fraction'])} | {f['missing_name_rows']:,} | {f['masked_rows']:,} | {pct(f['masked_row_fraction'])} |")
    lines+=['','## query 分組與遮罩分布','',
      '| query 範圍 | query 列數 | 已見具名列 | 新具名列 | 缺案名列 | 遮罩列數 | 遮罩列比例 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    qnames={'development':'2022-08-11～2023-08-10','confirmation':'2023-08-11～2024-08-10','auxiliary':'2021-08-11～2022-08-10','full_fit_to_test':'test（僅特徵）'}
    for stage in result['stages']:
        q=stage['query'];g=q['groups']
        lines.append(f"| {qnames[stage['stage']]} | {q['rows']:,} | {g['seen_named']['rows']:,} | {g['new_named']['rows']:,} | {g['missing_name']['rows']:,} | {q['masked_rows']:,} | {pct(q['masked_row_fraction'])} |")
    test=result['stages'][-1]['query'];g=test['groups']
    lines+=['',f"test 的具名新案為 {g['new_named']['rows']:,} 列，占 {pct(g['new_named']['row_fraction'])}，已見具名案 {g['seen_named']['rows']:,} 列，缺案名 {g['missing_name']['rows']:,} 列。已見案的遮罩比例為 {pct(g['seen_named']['mask_fraction'])}；新案與有樣本的缺案名組都依規則 100% 遮罩。這是識別資訊可用性的特徵統計，不涉及 test 房價。",'',
      '各時段已驗證：fit 的同案所有列一致、query 已見案完全重用 fit flag、新案／缺案名必遮罩；JSON 保存／重載的特徵轉換逐值一致；query 前後 state 內容雜湊相同。四欄外的 83 欄與未遮罩列全部逐值相同，原輸入未被就地更動。完整分組遮罩列數、state 雜湊與檢查結果見 JSON。','',
      '## 方法界線與風險','',
      '- 50% 按建案決定，案量大小不同且缺案名必遮罩，所以按列比例可能明顯偏離 50%；本稽核沒有因此調整規則。',
      '- 城市＋行政區＋案名是資料中的文字鍵，不保證唯一識別實體建案；沿用原行政區解析，沒有在此候選修正舊 87 欄。其他道路與格局仍可能間接識別建案，遮罩不等於完全消除記憶。',
      '- 遮住已見案也可能損害其精度；新案資訊較少，遮罩本身不會補上地點座標，也不負責一年後的價格趨勢外推。收益必須由凍結規約的價格評估判定。',
      '- 這次可以查看確認／輔助年度的 X 分布，沒有查看該年度 y 的誤差；資料年份曾在歷史訓練／評估出現，不能稱為全新、完全未見的測試資料。',
      '- 僅一個固定 raw CatBoost 遮罩候選；建案密度權重未納入，沒有重試單價 CatBoost。原 raw 參數固定 2,999 輪、seed 42、GPU、4 threads，y/1e6 均等權重，無 eval_set／early stopping。Public 不用於選遮罩率、seed 或融合係數。','',
      '## 追溯與重現','',
      f"凍結 helper SHA256：`{result['helper_sha256']}`。稽核前後所列既有 source/cache 雜湊全部相同；只建立本輪稽核輸出，沒有保存新价格模型或以全量 state 取代年度 state。",'',
      '```text','.venv\\Scripts\\python.exe src\\round5_mask_audit.py','```','',
      '- [完整純 X 遮罩統計](../outputs/round5/audit/mask_distribution.json)',
      '- [可重現稽核程式](../src/round5_mask_audit.py)',
      '- [凍結遮罩 helper](../src/round5_project_mask.py)',
      '- [研究建議與既有方法比較](round5_research_features.md)',
      '- [既有資料清理紀錄](round3_preprocessing_record.md)',
      '- [既有去重與切分程式](../src/model_common.py)','']
    preflight=result['normalization_preflight']
    consistency=['## 訓練前正規化口徑修正與逐列核對','',
      '在任何本輪真實價格模型訓練／評分前，發現 helper 曾額外將字面字串 `NaN`、`null` 視為缺案名，而既有 scorer 不採此定義。已先保存舊 helper 原始位元並核對舊 SHA，只移除這兩個額外別名；遮罩率、seed、四欄、hash 規則與其他演算法均未改。這是預先檢查修正，不是另一個候選。',
      '現在與 scorer 一致：真正空值、去除空白後空字串，以及 `nan`、`None`、`<NA>`、`NaT` 視為缺失；字面 `NaN`／`null` 保留為文字。helper 保留原 index，scorer 會重設 index，逐列比較以相同行的位置核對。',
      '修正前已比較原始特徵快取 train 455,784／valid 11,994／test 47,073 列：key 與 missing flag 差異均為 0。修正後再次比較這三份快取，以及每個年度 fit/query、全量 fit/test：所有逐列差異仍為 0，沒有需要列出的異常文字範例。四組實際遮罩分布與保留的舊版本稽核完全一致，未因這次口徑修正改變任何列的遮罩。',
      '新版合成檢查另涵蓋 Python 空值與所有上述字面別名，確認 key／missing flag 與 scorer 相同。同案一致、JSON 重載、四欄移除與其餘 83 欄不變也均通過。舊 synthetic preflight 原檔保留，新增 v2 紀錄，不覆寫歷史證據。','',
      f"舊 helper SHA256：`{preflight['previous_helper_sha256']}`。最終 helper SHA256：`{result['helper_sha256']}`。",'',
      '- [訓練前口徑修正與新版合成稽核](../outputs/round5/audit/mask_synthetic_preflight_v2.json)',
      '- [修正前 helper 原始位元](../outputs/round5/audit/helper_before_normalization_fix.py)',
      '- [修正前純 X 分布紀錄](../outputs/round5/audit/mask_distribution_before_normalization_fix.json)',
      '- [保留的原合成稽核](../outputs/round5/audit/mask_synthetic_preflight.json)','']
    position=lines.index('## 方法界線與風險')
    lines[position:position]=consistency
    DOC.write_text('\n'.join(lines).replace('没有','沒有').replace('分位数','分位數').replace('价格','價格'),encoding='utf-8')


def main():
    assert file_sha(HELPER)==EXPECTED_HELPER_SHA
    protected=['src/round5_project_mask.py','src/house_features.py','src/model_common.py','src/prepare_data.py',
               'reports/feature_schema.json','data_cache/valid_clean_mask.npy','data_cache/valid_clean_mask.json']
    protected += ['src/round3_evaluate.py','outputs/round5/audit/helper_before_normalization_fix.py',
                  'outputs/round5/audit/mask_synthetic_preflight.json',
                  'outputs/round5/audit/mask_synthetic_preflight_v2.json',
                  'outputs/round5/audit/mask_distribution_before_normalization_fix.json']
    protected += [f'data_cache/{split}{suffix}' for split in ('train','valid','test') for suffix in ('.pkl','_features.pkl','.json')]
    before={p:file_sha(ROOT/p) for p in protected}
    preflight=json.loads((ROOT/'outputs/round5/audit/mask_synthetic_preflight_v2.json').read_text(encoding='utf-8-sig'))
    assert preflight['helper_sha256']==EXPECTED_HELPER_SHA
    assert preflight['backup_sha256']==file_sha(ROOT/preflight['backup_path'])==preflight['previous_helper_sha256']
    cache_consistency=[]
    for split in ('train','valid','test'):
        feature_cache=pd.read_pickle(ROOT/f'data_cache/{split}_features.pkl')
        cache_consistency.append(compare_scorer_keys(feature_cache,split+'_original_feature_cache'))
        del feature_cache
    data=load_data()
    Xfull,_discarded_target=full_training_data(data)
    del _discarded_target
    existing=dict(data['metadata'])
    existing['full_fit_rows']=len(Xfull)
    existing['combined_duplicates_removed']=existing['train_rows']+existing['valid_rows']-len(Xfull)
    existing['price_use']='unchanged exact-row deduplication only; not used for masking, statistics, prediction, or scoring'
    for key in list(data):
        if key.startswith(('y_','raw_','ids_')):
            data.pop(key)
    stages=[]
    for name,(start,end) in SPLITS.items():
        fi,qi=temporal_indices(data,start,end=end)
        assert not np.intersect1d(fi,qi).size
        fit=data['X_train'].iloc[fi].reset_index(drop=True)
        query=data['X_train'].iloc[qi].reset_index(drop=True)
        stage=audit_stage(name,fit,query,[start,end])
        stage['fit_indices_content_sha256']=hashlib.sha256(np.asarray(fi,dtype='<i8').tobytes()).hexdigest()
        stage['query_indices_content_sha256']=hashlib.sha256(np.asarray(qi,dtype='<i8').tobytes()).hexdigest()
        stages.append(stage)
    stages.append(audit_stage('full_fit_to_test',Xfull.reset_index(drop=True),data['X_test'].reset_index(drop=True)))
    old_audit=json.loads((ROOT/'outputs/round5/audit/mask_distribution_before_normalization_fix.json').read_text(encoding='utf-8-sig'))
    assert old_audit['helper_sha256']==preflight['previous_helper_sha256']
    assert len(old_audit['stages'])==len(stages)
    for old,new in zip(old_audit['stages'],stages):
        assert old['stage']==new['stage']
        assert old['fit']==new['fit'] and old['query']==new['query']
    after={p:file_sha(ROOT/p) for p in protected}
    assert before==after and after['src/round5_project_mask.py']==EXPECTED_HELPER_SHA
    result={'audit_type':'feature-only real-split project-mask distribution; no model fit/prediction/error',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),'helper_sha256':EXPECTED_HELPER_SHA,
        'audit_script_sha256':file_sha(Path(__file__)),'mask_rules':MASK_RULES,
        'existing_preprocessing':existing,'stages':stages,
        'normalization_preflight':preflight,
        'original_feature_cache_scorer_consistency':cache_consistency,
        'scorer_source_sha256':file_sha(ROOT/'src/round3_evaluate.py'),
        'scorer_functions_loaded_without_importing_or_executing_evaluation':True,
        'actual_mask_distributions_identical_before_and_after_normalization_fix':True,
        'protected_artifact_sha256':after,'protected_artifacts_unchanged':True,
        'constraints':{'real_price_model_trained':False,'model_predictions_computed':False,'prediction_errors_computed':False,
                       'confirmation_or_auxiliary_scores_read':False,'test_prices_read':False,
                       'valid_prices_used_outside_existing_deduplication':False,
                       'target_used_for_mask_state_or_statistics':False,'feature_state_fit_on_query':False,
                       'mask_probability_changed_after_audit':False,'external_property_data_used':False,
                       'helper_modified_during_final_distribution_audit':False,'states_saved_for_price_inference':False},
        'risk_notes':['Project-key hash masks projects, not exactly 50 percent of rows.',
                      'Missing names always mask but never form a physical-project group.',
                      'Geographic text keys can contain aliases/collisions or old parser limitations.',
                      'No score or accepted-improvement conclusion follows from this feature-only audit.']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_doc(result)
    print(json.dumps({'completed':True,'audit':str(OUT.relative_to(ROOT)),'document':str(DOC.relative_to(ROOT)),
                      'helper_unchanged':True,'all_feature_checks':'passed'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()