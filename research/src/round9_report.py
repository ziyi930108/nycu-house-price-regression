"""Create the R9 human report only from finalized, verified experiment artifacts."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import round9_experiment as experiment
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round9'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def number(value,digits=2):return '—' if value is None else f'{value:,.{digits}f}'
def gain(base,new):return (base-new)/base*100 if base else None

def generate():
    specification=experiment.plan()
    decision=experiment.locked(OUT/'accepted_candidate.json')
    require(decision['round']==9 and decision['plan_sha256']==experiment.sha(OUT/'candidate_plan.json'),'Final decision refers to another plan')
    frozen=experiment.locked(OUT/'frozen_candidate.json')
    require(decision['frozen_candidate_sha256']==experiment.sha(OUT/'frozen_candidate.json'),'Final decision freeze differs')
    results={stage:experiment.read_result(stage) for stage in decision['result_sha256']}
    for stage,digest in decision['result_sha256'].items():require(experiment.sha(OUT/'evaluation'/stage/'results.json')==digest,'Final result hash changed')
    require(results and 'development' in results,'No development result')
    all_pass=len(results)==3 and all(value['decision']['passed'] for value in results.values())
    require(decision['accepted']==all_pass,'Final acceptance differs from results')
    require(frozen['passed']==results['development']['decision']['passed'],'Development freeze differs')
    control,candidate=experiment.CONTROL,experiment.CANDIDATE
    selected='候選已通過三期，須完成最終訓練及發布核對才可提供新版提交檔。' if all_pass else '本輪未採用，維持 submission_v3.csv；沒有用本輪方法另產生提交檔。'
    inputs={}
    def bind(path):
        path=Path(path);inputs[path.relative_to(ROOT).as_posix()]=experiment.sha(path)
    for path in [OUT/'candidate_plan.json',OUT/'candidate_plan.sha256',OUT/'frozen_candidate.json',OUT/'frozen_candidate.sha256',OUT/'accepted_candidate.json',OUT/'accepted_candidate.sha256',ROOT/'reports/round9_validation_plan.md',ROOT/'reports/round9_poi_design_review.md',ROOT/'reports/round9_model_design_review.md',OUT/'audit/preflight.json',OUT/'audit/poi_preparation_self_check.json',OUT/'audit/prepared_independent_review.json']:
        bind(path)
    bind(Path(__file__).resolve())
    stage_details={}
    for stage,value in results.items():
        bind(OUT/'evaluation'/stage/'results.json');bind(OUT/'evaluation'/stage/'results.sha256')
        model=experiment.locked(OUT/'models'/stage/'metadata.json');bind(OUT/'models'/stage/'metadata.json')
        manifest=experiment.locked(OUT/'poi_features'/stage/'manifest.json');bind(OUT/'poi_features'/stage/'manifest.json')
        index=experiment.locked(OUT/'poi_index'/str(manifest['snapshot_year'])/'audit.json');bind(OUT/'poi_index'/str(manifest['snapshot_year'])/'audit.json')
        stage_details[stage]={'metrics':value['metrics'],'decision':value['decision'],'fit_rows':value['fit_rows'],'query_rows':value['query_rows'],'snapshot_year':manifest['snapshot_year'],'source_pbf_sha256':manifest['source_pbf_sha256'],'replication_timestamp':manifest['replication_timestamp'],'geometry_records_by_class':manifest['geometry_records_by_class'],'feature_count':model['feature_count'],'categorical_feature_count':model['categorical_feature_count'],'fit_uses_eval_set':model['fit_uses_eval_set'],'early_stopping_used':model['early_stopping_used']}
    payload={'round':9,'created_at_utc':datetime.now(timezone.utc).isoformat(),'accepted':all_pass,'candidate':candidate,'control':control,'current_released_recommendation':'v3','release_not_performed_by_this_report':True,'stages':stage_details,'unevaluated_stages':decision['unevaluated_stages'],'formulas':experiment.FORMULAS,'thresholds':experiment.THRESHOLDS,'source_input_sha256':inputs,'external_property_answers_used':False,'public_score_used_for_selection':False,'private_score_known':False,'historically_untouched':False,'public_score_record':{'submission_v2.csv':2883250.4138,'submission_v3.csv':None}}
    lines=['# 第九輪結果：歷史設施距離','','**'+selected+'**','','本輪事前固定 13 類設施的距離及 1 個位置旗標，只替換 v3 裡 37.5% 的地理 CatBoost。未搜尋權重或在看成績後改特徵，未混入 R7／R8 未採用的方案。','','## 固定窗口結果','','改善率為正代表 MAE 降低；兩種 MAE 的單位都是新臺幣元。','','| 窗口 | v3 整體 MAE | 候選整體 MAE | 改善 | v3 加權 MAE | 候選加權 MAE | 改善 | 門檻 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for stage,value in results.items():
        a,b=value['metrics'][control],value['metrics'][candidate]
        lines.append(f"| {stage} | {number(a['overall_mae'])} | {number(b['overall_mae'])} | {number(gain(a['overall_mae'],b['overall_mae']),4)}% | {number(a['weighted_mae'])} | {number(b['weighted_mae'])} | {number(gain(a['weighted_mae'],b['weighted_mae']),4)}% | {'通過' if value['decision']['passed'] else '未通過'} |")
    if decision['unevaluated_stages']:lines+=['','未進入模型評分的窗口：'+ '、'.join(decision['unevaluated_stages'])+'。其地圖資料可能已按同一規則準備，這不代表曾查看模型分數。']
    for stage,value in results.items():
        lines+=['',f'### {stage}：分組與季度','','| 分組 | 筆數 | v3 MAE | 候選 MAE | 改善 |','|---|---:|---:|---:|---:|']
        a,b=value['metrics'][control],value['metrics'][candidate]
        for group in a['strata']:
            left,right=a['strata'][group],b['strata'][group]
            improvement=None if left['mae'] is None else gain(left['mae'],right['mae'])
            lines.append(f"| {group} | {left['rows']:,} | {number(left['mae'])} | {number(right['mae'])} | {number(improvement,4)}{'%' if improvement is not None else ''} |")
        for group in a['forecast_quarters']:
            left,right=a['forecast_quarters'][group],b['forecast_quarters'][group]
            improvement=None if left['mae'] is None else gain(left['mae'],right['mae'])
            lines.append(f"| {group} | {left['rows']:,} | {number(left['mae'])} | {number(right['mae'])} | {number(improvement,4)}{'%' if improvement is not None else ''} |")
        lines+=['','逐項條件：']
        for key,passed in value['decision']['checks'].items():lines.append(f"- `{key}`：{'通過' if passed else '未通過'}。")
    lines+=['','## 實際資料處理','','- 沿用原始表頭清理、完整資料去重與列對齊，原 87 欄及 R6 的 8 個地理欄保持不變。','- 只從事前指定年度的本地 OSM PBF 取設施；用實際 header 與 SHA 核對年份，不下載或查詢任何外部成交價格。','- 納入合法原始點及正常組裝的面；內洞保留。開放線、非面關係、缺失或不合法幾何跳過並留下原因，不補面或換成質心。','- 分別保留鐵路站、捷運出入口、客運站場、學校、幼兒園、高等教育、醫院、門診中心、藥局、超市、便利商店、公園及交流道出口節點。醫療只用指定 amenity 標籤，未擴充 healthcare-only。','- 使用固定 AEQD 投影計算點到幾何的最短距離，再作 log1p；不找半徑、截斷值或密度最佳參數。','- 原地理位置不可用或其範圍超過 2 公里時，新距離保留 NaN；新位置旗標只表示通過計算資格，並非定位精度或設施營業保證。','- 共 109 欄／23 類別欄；raw y/1e6、等權重、原有正則化、2,999 棵樹，不用驗證集早停。','- 兩個混合式均套同一份 fit-only v2 reference，trend 一次。保存模型、列索引、特徵、參數、來源与預測；正式評估前核對重載與所有查詢列重播。','','## 過擬合與適用限制','','development 要求整體及 test-X 建案比例加權 MAE 都改善至少 1%，並限制季度與大樣本建案組別惡化。通過後才准許 confirmation 與 auxiliary，後兩期只確認或否決、不調參。完整條件見驗證計畫。','','三個年度窗口在先前輪次已經使用，不能稱為從未看過的測試集。跨輪反覆嘗試仍可能造成選型過擬合；預設 1% 門檻不是統計顯著性或 Private 改善保證。測試集城市、月份、建案及地圖覆蓋也可能與歷史驗證不同。','','POI 完整度會隨年度與地區改變；快照時間不等於開業日，未出現的設施不代表不存在。道路代表點不等於住宅門牌，距離不等於交通時間或可達性。每個帶自身標籤的 OSM 物件保留自己的有效幾何，未作關係成員的設施語意去重；重複或錯位描述仍可能改變最近距離。','','使用者回報的 Public MAE 2,883,250.4138 屬 submission_v2.csv。v3 與本輪沒有已知新 Public 分數，Private 成績未知；不由本機驗證推定排行榜分數。','','## 檔案與重跑','','- 驗證計畫：`reports/round9_validation_plan.md`。','- 處理與決策日誌：`reports/round9_work_log.md`。','- 機器可讀結果：`reports/round9_results.json`。','- 固定計畫、模型、預測及判定：`outputs/round9/`。','- 目前已發布推薦仍為 `outputs/submission_v3.csv`；本報告本身不訓練或發布模型。']
    text='\n'.join(lines).replace('來源与','來源與')+'\n'
    report=ROOT/'reports/round9_report.md';result_path=ROOT/'reports/round9_results.json';readme=ROOT/'README_round9.md'
    for path in (report,result_path,readme):require(not path.exists(),'Refusing to replace existing report: '+str(path))
    experiment.write(result_path,payload);report.write_text(text,encoding='utf-8')
    readme.write_text('# 第九輪：歷史 POI 距離\n\n'+selected+'\n\n結果與限制見 [報告](reports/round9_report.md)，全部操作見 [工作紀錄](reports/round9_work_log.md)，固定規約見 [驗證計畫](reports/round9_validation_plan.md)。\n\n模型程式為 `src/round9_geo_poi.py` 與 `src/round9_experiment.py`。原始來源與處理後特徵以 SHA 綁定；既有輸出不能覆寫。已存在的實驗只能核對與重播，不能把重訓當成原模型。\n\n本輪程式不會查詢外部成交答案，不會上傳 Kaggle，也不改動 v2／v3 提交檔或版本表。\n',encoding='utf-8')
    for name,digest in inputs.items():require(experiment.sha(ROOT/name)==digest,'Report source changed while writing')
    print(json.dumps({'status':'report_written','accepted':all_pass,'report_sha256':experiment.sha(report),'results_sha256':experiment.sha(result_path),'readme_sha256':experiment.sha(readme)}),flush=True)
    return payload
if __name__=='__main__':generate()
