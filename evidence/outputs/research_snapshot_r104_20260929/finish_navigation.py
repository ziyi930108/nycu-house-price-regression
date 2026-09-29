"""Finish R103/R104 navigation from aggregate JSON after evidence copy; no ML."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'deliverables/v22_github'
DEST=REPO/'evidence/outputs/research_snapshot_r104_20260929'
BASE='cca69847befe8f123f19ff7351be0e23cc1ec7f2'
PAGE='docs/RESEARCH_UPDATES_20260929_BASE_OBJECTIVE.md'
NAV=['README.md','HANDOFF.md','research/README.md',PAGE]
STAGES=['development','confirmation','auxiliary']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,value):
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def interval_counts(result):
    counts=dict(negative=0,crossing=0,positive=0)
    for s in STAGES:
        for control in ['v22','C']:
            for family in ['project','month']:
                for metric in ['overall','weighted']:
                    v=result['stages'][s]['paired'][control]['methods'][family]['summary'][metric]
                    assert v['valid_replicates']==2000 and v['invalid_replicates']==0
                    lo,hi=v['percentile_2_5'],v['percentile_97_5'];assert lo<=hi
                    counts['negative' if hi<0 else 'positive' if lo>0 else 'crossing']+=1
    assert sum(counts.values())==result['conditional_intervals_reported']==24
    return counts

def build_documents(result,seal,teacher,costs):
    g=result['research_gates'];counts=interval_counts(result)
    assert not g['primary']['adoption_research_gate_passed'] and not g['mechanism']['mechanism_gate_passed']
    short='／'.join(f'{result["stages"][s]["metrics"]["T"]["overall_mae"]-result["stages"][s]["metrics"]["v22"]["overall_mae"]:+,.2f}' for s in STAGES)
    intro=f'R103 誤差互補診斷與 R104 兩教師／六學生、三期隔離評分均已完成。完整 T−v22 MAE 為 **{short} 元**（development／confirmation／auxiliary）；兩項固定門檻未通過，**不採用、無新 CSV；唯一提交仍為 v22，Private 未知。**'
    page=['# R103／R104：從誤差互補診斷到固定 base 目標','','2026-09-29。'+intro,'',
        f'本頁於發布前核驗，基底 {BASE}；實際發布身分以 Git main／SOURCE_MANIFEST 為準。Public 1,998,568.5706 元是使用者回報；本頁其餘分數均為離線回測。','',
        '## 先診斷，再固定一個假設','',
        '[R103 診斷報告](../evidence/reports/round103_report.md)以已保存預測做代數分解，沒有新模型或 bootstrap。最近窗 native 改善後，固定 base 的誤差抵銷減少；development 有 49,017 筆落在相同側外，spatial 的淨改善被 median 截住。median 也會阻擋退步，不能说它一律有害，更不能將事後分解當作因果。','',
        '[R104 計畫](../evidence/reports/round104_plan.md)只測一個假設：訓練 prior NN 時，把固定 area／LGB 分支納入損失，是否能改善完整回歸？兩組同合法 forward teacher 覆蓋列、145 欄、factor、原日期權重、初始化 bias、seed 42、30 epochs；不按 y 或誤差選列，不搜融合比例。','',
        '|組別|每批損失（價格先還原成百萬元）|','|---|---|',
        '|C：同列、梯度尺度對照|0.375 × mean(w × abs(p−y))|',
        '|T：固定 base 目標|mean(w × abs(0.375p + 0.375A + 0.25L − y))|','',
        'A／L 為較早 origin 的完整 forward 預測。新增 2020-08-11 origin 的 area NN／unit LGB；auxiliary、development、confirmation 學生依序使用 1／2／3 個老師期間，與原 eligible fit 列按 canonical 索引交集。學生不重建 prior／encoder／R/S，query 沿原同窗 A/L、R/S、H/U 與三支 median。所有八個 native fit、全 query 重載封存後才隔離評分。','',
        '## 完整模型與機制結果','','單位：元；差值負數較好。本輪 C 不是 R100 或 R102 的舊對照。','',
        '|窗口|v22 MAE|C MAE|T MAE|T−v22|T−C|T−v22 weighted|base T−C|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in STAGES:
        a=result['stages'][s];m=a['metrics'];lm=a['layer_metrics']
        values=[m['v22']['overall_mae'],m['C']['overall_mae'],m['T']['overall_mae'],m['T']['overall_mae']-m['v22']['overall_mae'],
                m['T']['overall_mae']-m['C']['overall_mae'],m['T']['weighted_mae']-m['v22']['weighted_mae'],lm['T']['base']['overall_mae']-lm['C']['base']['overall_mae']]
        assert values[-1]>0
        page.append('|'+s+'|'+'|'.join(f'{v:,.2f}' if i<3 else f'{v:+,.2f}' for i,v in enumerate(values))+'|')
    page+=['','Primary 要求三窗完整 overall／weighted 改善、最近窗至少改善 0.5%，且無季度或至少 500 列固定群退步超過 5%；mechanism 要求 T 相對 C 的 base 與 median、兩種 MAE 三窗均改善。原生 NN 單獨改善不是這次 mechanism 的必要條件。','',
           '|門檻|條件鍵（對應完整 JSON）|通過|','|---|---|---|']
    for gate in ['primary','mechanism']:
        for name,value in g[gate]['components'].items():page.append(f'|{gate}|{name}|{"是" if value else "否"}|')
    page+=['','T−C 的 base 三窗均退步，不能把只有最舊窗口完整改善當作研究成功。各層、分群／價帶、尾部貢獻與最強反證見[R104 摘要](../evidence/reports/round104_report.md)及[全部聚合 JSON](../evidence/outputs/round104/evaluation/results.json)。原 R102T 僅為固定背景，不額外搜模型。','',
        '## 配對區間、耗時與限制','',
        f'保存全部 24 個 CI：{counts["negative"]} 全負、{counts["crossing"]} 跨零、{counts["positive"]} 全正。T−v22／T−C 各三窗，project 與連續三個月 block，overall／weighted，固定 seed 20260925、每種 2,000 次。窗口反覆使用且只有單一 seed，區間不是 Private 勝率。','',
        '|步驟|秒數|包含關係|','|---|---:|---|',
        f'|R103 診斷 main|{costs["diagnostic"]:.3f}|0 新 fit／bootstrap|',
        f'|兩教師 fit|{teacher["fit_seconds"]:.3f}|已含於教師 main|',
        f'|教師 main|{costs["teacher_main"]:.3f}|已含於教師五步 launcher|',
        f'|教師五步 launcher|{costs["teacher_launcher"]:.3f}|凍結、準備、隔離 fit 標籤、2 fit|',
        f'|六學生 fit|{seal["student_fit_seconds"]:.3f}|已含於學生 main|',
        f'|學生 main|{costs["main"]:.3f}|準備、6 fit、重載、封存|',
        f'|獨立 scorer|{costs["evaluate"]:.3f}|指標與 24 CI|',
        f'|聚合摘要|{costs["summary"]:.3f}|僅 JSON，不新評分|','',
        '以上是程序實測，不是聊天／開發工時；內含時間不能再相加。2023 靜態非價格地理晚於早期 origin，因此價格標籤與 learned state 雖按時間隔離，仍不屬完整 historical-as-of 驗證；教師與 query 模型分布也不同。沒有因果或 Private 最佳保證。','',
        '## 程式、證據與交接界線','',
        '[R103 helper](../research/src/round103_complementarity_math.py)／[driver](../research/src/round103_complementarity_diagnostics.py)；'
        '[教師準備](../research/src/round104_teacher_preparation.py)／[教師訓練](../research/src/round104_teacher_fit.py)；'
        '[學生準備](../research/src/round104_student_preparation.py)／[loss trainer](../research/src/round104_base_aware_train.py)／'
        '[main](../research/src/round104_experiment.py)／[組合](../research/src/round104_composition.py)／[隔離 scorer](../research/src/round104_evaluate.py)。','',
        '[有限結果審查](../evidence/reports/round104_results_review.md)及[下一方向盤點](../evidence/reports/round104_next_direction_inventory.md)保留反證；後者只有 source／report 盤點，尚無新計畫或 fit。','',
        '[本批精確複製清單](../evidence/outputs/research_snapshot_r104_20260929/copy_manifest.json)保留原碼、人工測例的各次來源／收據、計畫、模型 metadata、聚合結果與來源 SHA。'
        '不包原交易表、逐筆標籤／向量、模型、完整 encoder／ramp／category／prior states。部分原碼及報告連結仍依賴本機工作區；這是文件與程式快照，不是完整重訓備份或乾淨環境重現。','',
        '[R101／R102 歷史頁](RESEARCH_UPDATES_20260929_PRIOR_INPUT.md)中提出的診斷已完成，歷史頁原 bytes 保留。正式 v22 報告／PDF／簡報與歷史 source 不變；不新增提交，也不承諾下一個改良。','']
    latest='## 最新組合目標研究\n\n[R103／R104 完成頁]('+PAGE+')：'+intro+'\n\n'+f'24 CI：{counts["negative"]} 全負／{counts["crossing"]} 跨零／{counts["positive"]} 全正。T−C base 三窗均退步，較早窗口完整改善也保留；2023 地理非 as-of、單 seed 與重用窗口限制見完成頁。正式報告及簡報仍聚焦 v22。\n\n'
    documents={};s=(REPO/NAV[0]).read_text(encoding='utf8');start=s.index('## 最新先驗輸入研究');end=s.index('## 簡報與專案更新',start)
    documents[NAV[0]]=s[:start]+latest+s[end:]
    s=(REPO/NAV[1]).read_text(encoding='utf8').replace('R85–R102','R85–R104')
    updates={'最新研究':f'R103 診斷＋R104 八 native 完成；T−v22 {short} 元；兩 gate 未過、不採用、無新 CSV',
             '本批發布':f'本頁於發布前核驗，base {BASE}；當前身分以 Git main／SOURCE_MANIFEST 為準',
             '後續界線':'R103／R104 已完成；v22 唯一提交，Private 未知；後續需另固定假設，不自動開新訓練'}
    for key,value in updates.items():
        s,n=re.subn(r'^\| '+key+r' \|.*$', '| '+key+' | '+value+' |',s,flags=re.M);assert n==1
    s=s.replace('- [最新 R101／R102]', '- [最新 R103／R104]('+PAGE+')：誤差互補、固定 base 目標、兩項門檻、24 CI 與地理時間限制。\n- [前批 R101／R102]')
    start=s.index('## 最新研究與後續工作');end=s.index('使用者已選定v22，最新紀錄',start)
    s=s[:start]+'## 最新研究與後續工作\n\n[R103／R104 完成頁]('+PAGE+')是最新入口。'+intro+'\n\n'+(
        '先以保存預測完成誤差抵銷／median 路由診斷，再固定 C 同尺度 native loss 與 T base loss。兩教師加六學生、全部封存後才獨立評分；C 使用相同 teacher-covered 子集，不能混稱舊 R102 的 zero 欄對照。\n\n'
        '兩項固定 gate 均未通過；全部層、群、24 CI 與時間見[摘要](evidence/reports/round104_report.md)。2023 靜態地理晚於早期 origin，因此不冒稱完全歷史時點可用；保存重載也不等於全新環境重訓。\n\n'
        '[下一方向盤點](evidence/reports/round104_next_direction_inventory.md)只讀 source／report，沒有新計畫或 fit。歷史頁保留，此頁不新增下一候選或背景訓練承諾。\n\n')+s[end:]
    documents[NAV[1]]=s
    s=(REPO/NAV[2]).read_text(encoding='utf8').replace('R85–R102','R85–R104').replace('## 最新追加：R101','## 歷史追加：R101')
    note='## 最新追加：R103 診斷與 R104 固定 base 目標\n\n[完成頁](../'+PAGE+')：'+intro+'\n\n'+(
        '[R103 計畫](../evidence/reports/round103_plan.md)／[診斷](../evidence/reports/round103_report.md)；[R104 計畫](../evidence/reports/round104_plan.md)／[摘要](../evidence/reports/round104_report.md)。'
        '新來源、人工歷史與聚合證據見[精確複製清單](../evidence/outputs/research_snapshot_r104_20260929/copy_manifest.json)。不包逐筆資料、模型、完整 encoder／價格 state；舊研究的待辦是當時狀態。\n\n')
    split=s.find('\n')+1;documents[NAV[2]]=s[:split]+'\n'+note+s[split:];documents[PAGE]='\n'.join(page)
    return documents,counts

def main():
    assert (DEST/'copy_manifest.json').is_file() and not (DEST/'final_local_verification.json').exists() and not (REPO/PAGE).exists()
    baseline=read(DEST/'baseline.json');copied=read(DEST/'copy_manifest.json');inputs={}
    assert sha(REPO/'SOURCE_MANIFEST.json')==baseline['manifest_sha256']
    for row in baseline['files']:assert sha(REPO/row['path'])==row['sha256'],row['path']
    for row in copied['files']:assert sha(ROOT/row['source'])==row['sha256']==sha(REPO/row['destination']),row['source']
    for name in NAV[:3]:assert (DEST/'navigation_before'/name).read_bytes()==(REPO/name).read_bytes()
    def bound(name):
        p=ROOT/name;v=read(p);inputs[name]=sha(p);return v
    paths={'diagnostic':'outputs/round103/execution.json','teacher_main':'outputs/round104_teacher/models/execution.json',
           'teacher_launcher':'outputs/round104_teacher_launcher_20260929/execution.json','main':'outputs/round104/execution.json',
           'evaluate':'outputs/round104/evaluation/execution.json','summary':'outputs/round104_summary/execution.json'}
    costs={}
    for key,path in paths.items():
        v=bound(path);assert v['exit_code']==0;costs[key]=v['elapsed_seconds']
    result=bound('outputs/round104/evaluation/results.json');seal=bound('outputs/round104/sealed_manifest.json');teacher=bound('outputs/round104_teacher/models/sealed_manifest.json')
    bound('outputs/round104_summary/summary.json');bound('outputs/round104_summary/receipt.json')
    assert result['status']=='completed_three_window_evaluation' and set(result['stages'])==set(STAGES)
    assert result['selected_submission']=='v22' and result['Private_score'] is None and result['new_CSV'] is False
    assert seal['new_native_fits_total']==8 and seal['new_NN_fits']==6 and teacher['new_native_fits']==2
    assert seal['all_query_reload_exact'] and teacher['all_query_reload_exact']
    docs,counts=build_documents(result,seal,teacher,costs);checks=[]
    for name,text in docs.items():
        for href in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',text):
            href=href.strip('<>')
            if re.match(r'^[a-z]+:|^#',href,re.I):continue
            q=((REPO/name).parent/urllib.parse.unquote(href.split('#')[0])).resolve()
            exists=q.exists() or q==(REPO/PAGE).resolve();checks.append({'document':name,'href':href,'exists':exists});assert exists,(name,href)
    changes=[]
    for name,text in docs.items():
        p=REPO/name;before=sha(p) if p.exists() else None
        with p.open('w' if p.exists() else 'x',encoding='utf8',newline='\n') as f:f.write(text)
        changes.append({'path':name,'before_sha256':before,'after_sha256':sha(p),'bytes':p.stat().st_size})
    changed=[r['path'] for r in baseline['files'] if sha(REPO/r['path'])!=r['sha256']];assert set(changed)==set(NAV[:3])
    for row in copied['files']:assert sha(REPO/row['destination'])==row['sha256']
    assert sha(REPO/'SOURCE_MANIFEST.json')==baseline['manifest_sha256']
    save(DEST/'document_changes.json',{'files':changes,'input_sha256':inputs})
    save(DEST/'navigation_links.json',{'status':'passed','navigation_count':4,'checks':checks})
    receipt={'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'published_base':BASE,
       'copied_count':copied['copied_count'],'copied_bytes':copied['copied_bytes'],'baseline_files':len(baseline['files']),
       'baseline_changed_only':changed,'navigation_changes':changes,'navigation_local_links_checked':len(checks),
       'all_navigation_links_resolve':True,'all_original_copies_exact':True,'interval_counts':counts,'input_sha256':inputs,
       'source_sha256':sha(Path(__file__)),'SOURCE_MANIFEST_unchanged':True,'historical_sources_and_formal_REPORT_PDF_PPTX_unchanged':True,
       'new_fit_score_calls':0,'package_commit_push_performed':False,'full_retraining_backup':False}
    save(DEST/'final_local_verification.json',receipt)
    print(json.dumps({'status':'passed','copied_count':copied['copied_count'],'navigation_links':len(checks),'receipt_sha256':sha(DEST/'final_local_verification.json')}))

if __name__=='__main__':main()
