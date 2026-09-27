"""Round-three provenance and readable action log; archives stay immutable."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
from prepare_data import ROOT,DATA_DIR,sha256

OUT=ROOT/'outputs/round3'
LOG=OUT/'work_log.json'

def now(): return datetime.now(timezone.utc).isoformat()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def record(step,category,action,reason,result,safeguards,artifacts=()):
    doc=json.loads(LOG.read_text(encoding='utf-8')) if LOG.exists() else {'round':3,'events':[]}
    existing=next((e for e in doc['events'] if e['step']==step),None)
    if existing: return
    doc['events'].append({'step':step,'logged_at_utc':now(),'category':category,'action':action,
        'reason':reason,'result':result,'safeguards':safeguards,'artifacts':list(artifacts)})
    write(LOG,doc)
    text=['# 第三輪優化工作紀錄','',
          '每一步記錄做了什麼、原因、結果與避免過擬合的措施。原始資料與前兩輪封存檔不修改。',
          '所有模型只使用提供的本地資料；不查詢內政部成交答案。Public 只作外部參考，Private 分數目前未知。','']
    for e in doc['events']:
        text.extend([f"## {e['step']}｜{e['category']}",'',f"時間（UTC）：{e['logged_at_utc']}",'',
                     f"**動作：** {e['action']}",'',f"**原因：** {e['reason']}",'',
                     f"**結果：** {e['result']}",'',f"**避免過擬合／資料保護：** {e['safeguards']}",''])
        for artifact in e['artifacts']:
            text.append(f'- [{artifact}](../{artifact})')
        text.append('')
    (ROOT/'reports/round3_work_log.md').write_text('\n'.join(text),encoding='utf-8')


def initialize():
    submission=ROOT/'outputs/submission_v2.csv'
    digest=sha256(submission)
    assert digest=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
    assert digest==sha256(ROOT/'runs/v2/outputs/submission_v2.csv')
    path=ROOT/'reports/kaggle_submission_history.json'
    history=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'submissions':[]}
    if not any(r['submission_sha256']==digest and r['public_mae']==2883250.4138 for r in history['submissions']):
        history['submissions'].append({'version':'v2','submission':'outputs/submission_v2.csv',
            'submission_sha256':digest,'reported_at_utc':now(),'public_mae':2883250.4138,
            'public_test_fraction_approx':.30,'private_test_fraction_approx':.70,'private_mae':None,
            'score_direction':'lower_is_better','source':'User screenshot and explicit filename confirmation',
            'rank_in_screenshot':5,'final_grade_basis':'private'})
    write(path,history)
    index_path=ROOT/'runs/versions.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    item=next(v for v in index['versions'] if v['version']=='v2')
    item.update(kaggle_score=2883250.4138,kaggle_score_split='public',kaggle_public_mae=2883250.4138,
                kaggle_private_mae=None,public_test_fraction_approx=.30,private_test_fraction_approx=.70)
    write(index_path,index)
    record('001','公開分數對應','將使用者確認的 submission_v2.csv 對應至 Public MAE 2,883,250.4138。',
           '避免混淆本機驗證、公榜與不同提交版本。','CSV SHA 與第二輪封存完全相同；Private 成績未知，最終依其餘約 70% 的測試資料評分。',
           '不把 Public 當作訓練標籤或唯一選型依據；不改寫已封存的第二輪報告。',['reports/kaggle_submission_history.json','runs/versions.json'])
    manifest=json.loads((ROOT/'reports/run_manifest.json').read_text(encoding='utf-8'))
    checked={name:sha256(DATA_DIR/name) for name in manifest['input_sha256']}
    assert checked==manifest['input_sha256']
    write(OUT/'input_integrity.json',{'input_sha256':checked,'inputs_unchanged':True,'checked_at_utc':now()})
    record('002','資料處理基準','沿用前兩輪可追溯的清理與 87 個特徵，先確認原始檔案未變更。',
           '先固定資料版本，再判断新模型差異，避免把前處理變動誤當模型改善。',
           '原 train 455,787 列，移除 3 列夾入的英文欄名；再移除排除編號後的 529 列完全重複資料，留下 455,255 列。valid 11,994 列、test 47,073 列。最終 train＋valid 去重後 467,221 列。四份原始檔雜湊全部相同。',
           '原檔不修改；編號及總價不作特徵；缺失與不合理值沿用既有標記方式；額外修正如有採用將另記。',['outputs/round3/input_integrity.json','src/prepare_data.py','src/house_features.py','reports/data_quality.json'])
    record('003','驗證設計','啟動長時間回測、異常大額誤差與特徵分布三項獨立稽核。',
           '原本約四個月時間驗證短於近一年的 test，原 valid 又多為已見建案；本機 MAE 可能偏樂觀。',
           '先檢查兩段接近一年的 rolling-origin 切分是否可行，再凍結小量候選與接受條件。診斷結果尚未用來承諾改善。',
           '候選開發期與確認期分開；確認後不再按其結果反覆調參；不以 Kaggle 公榜挑係數。',['outputs/round3/audit'])
    print(json.dumps({'public_mae':2883250.4138,'version':'v2','inputs_unchanged':True,'events':len(json.loads(LOG.read_text(encoding='utf-8'))['events'])}))

if __name__=='__main__': initialize()
