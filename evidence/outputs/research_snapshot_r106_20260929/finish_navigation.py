"""Finish exactly three navigation files and one new R106 page; no ML or publish."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import re
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'deliverables/v22_github'
DEST=REPO/'evidence/outputs/research_snapshot_r106_20260929'
BASE='bde2a9221ea6beaad2297e97e12900e322b3872c'
PAGE='docs/RESEARCH_UPDATES_20260929_JOINT_DROPOUT.md'
NAV=['README.md','HANDOFF.md','research/README.md']
STAGES=('development','confirmation','auxiliary')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,value):
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def build_documents(result,costs):
    g=result['research_gates'];primary=g['primary']['adoption_research_gate_passed'];mechanism=g['mechanism']['mechanism_gate_passed']
    counts=dict(negative=0,crosses_zero=0,positive=0,unavailable=0)
    for s in STAGES:
        for control in ('v22','C'):
            for family in ('project','month'):
                for metric in ('overall','weighted'):
                    v=result['stages'][s]['paired'][control]['methods'][family]['summary'][metric]
                    assert v['valid_replicates']+v['invalid_replicates']==2000
                    lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    if lo is None or hi is None:key='unavailable'
                    else:
                        assert lo<=hi
                        key='negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero'
                    counts[key]+=1
    assert sum(counts.values())==result['conditional_intervals_reported']==24
    short='／'.join(f"{result['stages'][s]['comparison']['overall_mae']['delta']:+,.2f}" for s in STAGES)
    outcome='兩項研究門檻通過，但不自動採用' if primary and mechanism else '研究門檻未全通過，不採用'
    intro=f'R106六次NN與三窗評分完成；完整T−v22 MAE為 **{short} 元**（development／confirmation／auxiliary）。{outcome}；**唯一提交仍v22、Private未知，無新CSV。依使用者要求，本輪完成後暫停研究。**'
    page=['# R106：四個身分類別的聯動 dropout','','2026-09-29。'+intro,'',
        'C依原R63逐欄獨立0.1遮罩重訓；T令案名、完整門牌、棟及戶、棟別共用案名原mask。仍抽完整23欄，其他19欄與eval不變；同144數值欄、原prior、日期權重、初始化及30epochs。新C不要求等於保存v22；六fit全部重載封存後才隔離評分。','',
        '|窗口|T−v22整體／加權|T−C整體／加權|C−v22整體／加權|','|---|---:|---:|---:|']
    for s in STAGES:
        a=result['stages'][s]
        values=['／'.join(f"{a[p][k]['delta']:+,.2f}" for k in ('overall_mae','weighted_mae')) for p in ('comparison','comparison_vs_C','C_vs_v22')]
        page.append('|'+s+'|'+'|'.join(values)+'|')
    page+=['',f'Primary：{primary}；機制對照：{mechanism}。24 CI：{counts["negative"]}全負／{counts["crosses_zero"]}跨零／{counts["positive"]}全正。所有五層及分群反證見[完整報告](../evidence/reports/round106_report.md)。',
        f'六fit {costs["fit"]:.3f}秒已含main {costs["main"]:.3f}秒；scorer {costs["evaluation"]:.3f}秒。launcher {costs["launcher"]:.3f}秒含主流程，不能重複相加；非聊天工時。','',
        'unknown embedding原已訓練；聯動不遮numeric prior或其他地址，並非真新案模擬。單seed、重用窗口與2023地理可用性限制保留，CI不是Private勝率。Public仍為使用者回報1,998,568.5706元。',
        '依要求停止後續研究，不開R107或背景訓練；再次開始須使用者指示。正式v22報告與簡報保持。','',
        '[計畫](../evidence/reports/round106_plan.md) · [結果](../evidence/outputs/round106/evaluation/results.json) · [trainer](../research/src/round106_joint_dropout_train.py) · [driver](../research/src/round106_joint_dropout_experiment.py) · [scorer](../research/src/round106_joint_dropout_evaluate.py) · [複製清單](../evidence/outputs/research_snapshot_r106_20260929/copy_manifest.json)',
        '原碼、人工attempt與聚合收據保留；不包原交易表、逐筆array、模型及完整learned states，部分路徑仍依賴本機工作區，非完整重訓備份。',
        f'發布前base {BASE}；當前身分以Git main／SOURCE_MANIFEST為準。','']
    latest='## 最新聯動dropout研究\n\n[R106完成頁]('+PAGE+')：'+intro+'\n\nC為原R63設定重訓，T只改四欄mask聯動；C−v22偏移、24CI及反證全部保留。正式報告／簡報聚焦v22。研究於本輪完成後暫停，不安排下一輪。\n\n'
    docs={};s=(REPO/NAV[0]).read_text(encoding='utf8');start=s.index('## 最新原生平均研究');end=s.index('## 簡報與專案更新',start)
    docs[NAV[0]]=s[:start]+latest+s[end:]
    s=(REPO/NAV[1]).read_text(encoding='utf8').replace('R85–R105','R85–R106')
    updates={'最新研究':f'R106六NN完成；T−v22 {short} 元；{outcome}，無新CSV',
        '本批發布':f'本頁於發布前核驗，base {BASE}；當前身分以Git main／SOURCE_MANIFEST為準',
        '後續界線':'使用者要求本輪跑完後暫停；R106完成後停止研究，不開R107或背景訓練；v22唯一提交'}
    for key,value in updates.items():s,n=re.subn(r'^\| '+key+r' \|.*$','| '+key+' | '+value+' |',s,flags=re.M);assert n==1
    s=s.replace('- [最新 R105]', '- [最新 R106]('+PAGE+')：六NN、四欄聯動、C重訓偏移、兩門檻與24CI；完成後暫停。\n- [前批 R105]')
    start=s.index('## 最新研究與後續工作');end=s.index('使用者已選定v22，最新紀錄',start)
    s=s[:start]+'## 最新研究與後續工作\n\n[R106完成頁]('+PAGE+')是最新入口。'+intro+'\n\n'+(
        'C是原R63設定的新重訓，T只聯動四個類別遮罩；同23欄完整抽樣、其他19欄及eval不變。原v22五層先exact重播，六NN與全部query封存後才獨立評分。\n\n'
        'C−v22重訓偏移、完整層次、群別正反證與24CI見[報告](evidence/reports/round106_report.md)。單seed及舊窗口並非盲測；未知Private、2023地理及乾淨環境尚未重訓的限制保留。\n\n'
        '使用者要求跑完本輪後暫停；目前R106已完成，停止研究，不新增R107或排程／背景訓練。後續僅待新的明確指示。\n\n')+s[end:]
    docs[NAV[1]]=s
    s=(REPO/NAV[2]).read_text(encoding='utf8').replace('R85–R105','R85–R106').replace('## 最新追加：R105','## 歷史追加：R105')
    note='## 最新追加：R106聯動dropout，完成後暫停\n\n[完成頁](../'+PAGE+')：'+intro+'\n\n'+(
        '[計畫](../evidence/reports/round106_plan.md)／[報告](../evidence/reports/round106_report.md)／[精確複製清單](../evidence/outputs/research_snapshot_r106_20260929/copy_manifest.json)。六NN與全部人工attempt保存，不包原始交易、逐筆向量、模型或完整states。使用者要求本輪後暫停，不留下一輪待跑。\n\n')
    end=s.find('\n')+1;docs[NAV[2]]=s[:end]+'\n'+note+s[end:];docs[PAGE]='\n'.join(page)
    return docs,counts


def main():
    assert (DEST/'copy_manifest.json').is_file() and not (DEST/'final_local_verification.json').exists() and not (REPO/PAGE).exists()
    baseline=read(DEST/'baseline.json');copied=read(DEST/'copy_manifest.json');inputs={}
    assert sha(REPO/'SOURCE_MANIFEST.json')==baseline['manifest_sha256']
    for row in baseline['files']:assert sha(REPO/row['path'])==row['sha256'],row['path']
    for row in copied['files']:assert sha(ROOT/row['source'])==row['sha256']==sha(REPO/row['destination']),row['source']
    for name in NAV:assert (DEST/'navigation_before'/name).read_bytes()==(REPO/name).read_bytes()
    def bound(name):
        p=ROOT/name;value=read(p);inputs[name]=sha(p);return value
    paths={'main':'outputs/round106/execution.json','evaluation':'outputs/round106/evaluation/execution.json',
           'summary':'outputs/round106_summary/execution.json','launcher':'outputs/round106_launcher_20260929/execution.json'}
    costs={}
    for key,path in paths.items():v=bound(path);assert v['exit_code']==0;costs[key]=v['elapsed_seconds']
    result=bound('outputs/round106/evaluation/results.json');seal=bound('outputs/round106/sealed_manifest.json')
    bound('outputs/round106_summary/summary.json');bound('outputs/round106_summary/receipt.json')
    assert result['status']=='completed_three_window_evaluation' and set(result['stages'])==set(STAGES)
    assert result['selected_submission']=='v22' and result['Private_score'] is None and not result['new_CSV']
    assert seal['new_NN_fits']==6 and seal['new_native_tree_fits']==0 and seal['prior_or_R_S_build_calls']==0
    assert seal['all_query_reload_exact'] and seal['original_v22_all_five_layers_replay_exact'] and seal['R_S_state_and_frames_unchanged']
    costs['fit']=seal['fit_seconds']
    docs,counts=build_documents(result,costs);links=[]
    for name,text in docs.items():
        for href in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',text):
            href=href.strip('<>')
            if re.match(r'^[a-z]+:|^#',href,re.I):continue
            q=((REPO/name).parent/urllib.parse.unquote(href.split('#')[0])).resolve()
            exists=q.exists() or q==(REPO/PAGE).resolve();links.append({'document':name,'href':href,'exists':exists});assert exists,(name,href)
    changes=[]
    for name,text in docs.items():
        p=REPO/name;before=sha(p) if p.exists() else None
        with p.open('w' if p.exists() else 'x',encoding='utf8',newline='\n') as f:f.write(text)
        changes.append({'path':name,'before_sha256':before,'after_sha256':sha(p),'bytes':p.stat().st_size})
    changed=[row['path'] for row in baseline['files'] if sha(REPO/row['path'])!=row['sha256']];assert set(changed)==set(NAV)
    for row in copied['files']:assert sha(REPO/row['destination'])==row['sha256']
    assert sha(REPO/'SOURCE_MANIFEST.json')==baseline['manifest_sha256']
    save(DEST/'document_changes.json',{'files':changes,'input_sha256':inputs})
    save(DEST/'navigation_links.json',{'status':'passed','navigation_count':4,'checks':links,
        'scope':'Four new/edited navigation documents; immutable copied source/report links retain workspace-layout limitations'})
    save(DEST/'final_local_verification.json',{'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'published_base':BASE,'copied_count':copied['copied_count'],'copied_bytes':copied['copied_bytes'],
        'baseline_files':len(baseline['files']),'baseline_changed_only':changed,'navigation_changes':changes,
        'navigation_local_links_checked':len(links),'all_navigation_links_resolve':True,'all_original_copies_exact':True,
        'interval_counts':counts,'input_sha256':inputs,'source_sha256':sha(Path(__file__)),
        'SOURCE_MANIFEST_unchanged':True,'historical_sources_and_formal_REPORT_PDF_PPTX_unchanged':True,
        'new_fit_score_calls':0,'package_commit_push_performed':False,'full_retraining_backup':False})
    print(json.dumps({'status':'passed','copied_count':copied['copied_count'],'navigation_links':len(links),
                     'receipt_sha256':sha(DEST/'final_local_verification.json')}))


if __name__=='__main__':main()
