"""Finish exactly three navigation files and one new R105 page; no ML or publish."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import re
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'deliverables/v22_github'
DEST=REPO/'evidence/outputs/research_snapshot_r105_20260929'
BASE='374d488b2d1cfac5833bf685016522ad683d8949'
PAGE='docs/RESEARCH_UPDATES_20260929_NATIVE_AVERAGE.md'
NAV=['README.md','HANDOFF.md','research/README.md']
STAGES=('development','confirmation','auxiliary')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,value):
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def build_documents(result,costs):
    gates=result['research_gates'];primary=gates['primary']['adoption_research_gate_passed']
    endpoint=gates['supplementary_endpoint']['supplementary_endpoint_gate_passed']
    counts=dict(negative=0,crosses_zero=0,positive=0,unavailable=0)
    for s in STAGES:
        for control in ('v22','C'):
            for family in ('project','month'):
                for measure in ('overall','weighted'):
                    v=result['stages'][s]['paired'][control]['methods'][family]['summary'][measure]
                    assert v['valid_replicates']+v['invalid_replicates']==2000
                    lo,hi=v['percentile_2_5'],v['percentile_97_5']
                    if lo is None or hi is None:key='unavailable'
                    else:
                        assert lo<=hi
                        key='negative' if hi<0 else 'positive' if lo>0 else 'crosses_zero'
                    counts[key]+=1
    assert sum(counts.values())==result['conditional_intervals_reported']==24
    short='／'.join(f"{result['stages'][s]['comparison']['overall_mae']['delta']:+,.2f}" for s in STAGES)
    outcome='研究門檻皆通過，但不自動採用' if primary and endpoint else '研究門檻未全通過，不採用'
    intro=f'R105固定原生50/50平均與三期隔離評分已完成；完整T−v22 MAE為 **{short} 元**（development／confirmation／auxiliary）。{outcome}；**唯一提交仍v22、Private未知，無新CSV。**'
    page=['# R105：固定原生平均','','2026-09-29。'+intro,'',
        '只測一次float64 `M=.5*P+.5*Q`；P是v22原prior NN，Q是R102T。先重播兩端五層exact，再沿原base／reference／spatial／median。C指原R102T完整端點。0新fit、0原生模型或NPZ載入，但有載入已學R/S狀態；三窗全部封存後才獨立評分。','',
        '|窗口|T−v22 MAE|T−C MAE|T−v22 weighted|T−C weighted|','|---|---:|---:|---:|---:|']
    for stage in STAGES:
        a=result['stages'][stage]
        values=[a[p][k]['delta'] for k in ('overall_mae','weighted_mae') for p in ('comparison','comparison_vs_C')]
        page.append('|'+stage+'|'+'|'.join(f'{v:+,.2f}' for v in values)+'|')
    page+=['',f'Primary：{primary}；補充端點門檻：{endpoint}。24 CI為{counts["negative"]}全負／{counts["crosses_zero"]}跨零／{counts["positive"]}全正；完整門檻、分群、五層及區間見[報告](../evidence/reports/round105_report.md)。',
        f'實測main {costs["main"]:.3f}秒、評分{costs["evaluation"]:.3f}秒、摘要{costs["summary"]:.3f}秒；launcher {costs["launcher"]:.3f}秒包含主流程，不可重複加總。這不是聊天工時。','',
        '反覆窗口並非盲測；沒有多seed方差估計，平均不保證完整流程受益。2023非價格地理存在歷史可用性限制；bootstrap不是Private勝率。唯一提交v22與使用者回報Public 1,998,568.5706元不變。','',
        '[固定計畫](../evidence/reports/round105_plan.md) · [結果JSON](../evidence/outputs/round105/evaluation/results.json) · [driver](../research/src/round105_native_average.py) · [scorer](../research/src/round105_native_average_evaluate.py) · [複製清單](../evidence/outputs/research_snapshot_r105_20260929/copy_manifest.json)','',
        '[下一方向盤點](../evidence/reports/round105_next_direction_inventory.md)尚無新計畫或fit。[R104歷史頁](RESEARCH_UPDATES_20260929_BASE_OBJECTIVE.md)保持原bytes；正式報告及簡報不變。快照排除交易表、逐筆向量、模型及learned states，部分原碼路徑依賴本機布局，不是完整重訓備份。',
        f'發布前基底：{BASE}；實際身分以Git main／SOURCE_MANIFEST為準。','']
    latest='## 最新原生平均研究\n\n[R105完成頁]('+PAGE+')：'+intro+'\n\n'+f'0新fit，兩端全五層重播後套固定平均；24 CI為{counts["negative"]}全負／{counts["crosses_zero"]}跨零／{counts["positive"]}全正。正式報告與簡報仍聚焦v22。\n\n'
    docs={};s=(REPO/NAV[0]).read_text(encoding='utf8');start=s.index('## 最新組合目標研究');end=s.index('## 簡報與專案更新',start)
    docs[NAV[0]]=s[:start]+latest+s[end:]
    s=(REPO/NAV[1]).read_text(encoding='utf8').replace('R85–R104','R85–R105')
    updates={'最新研究':f'R105原生固定平均完成；T−v22 {short} 元；{outcome}，無新CSV',
        '本批發布':f'本頁於發布前核驗，base {BASE}；當前身分以Git main／SOURCE_MANIFEST為準',
        '後續界線':'R105已完成；v22唯一提交、Private未知；後續需另固定假設，不自動新訓練'}
    for key,value in updates.items():s,n=re.subn(r'^\| '+key+r' \|.*$','| '+key+' | '+value+' |',s,flags=re.M);assert n==1
    s=s.replace('- [最新 R103／R104]', '- [最新 R105]('+PAGE+')：固定原生平均、兩端重播、兩項門檻及24CI。\n- [前批 R103／R104]')
    start=s.index('## 最新研究與後續工作');end=s.index('使用者已選定v22，最新紀錄',start)
    s=s[:start]+'## 最新研究與後續工作\n\n[R105完成頁]('+PAGE+')是最新入口。'+intro+'\n\n'+(
        '候選只固定50/50原生平均；兩端五層完全重播、同窗R/S未變，三窗全部封存後獨立評分。C為原R102T完整端點；没有新native模型fit，但有套用保存learned R/S狀態。\n\n'
        '完整層次、分群正反證與24CI見[報告](evidence/reports/round105_report.md)。反覆窗口及靜態2023地理限制保留，不宣稱variance reduction、Private最好或全新環境重現。\n\n'
        '[下一方向盤點](evidence/reports/round105_next_direction_inventory.md)僅來源研究，尚無新計畫或訓練；R104提出的固定平均則已由本輪完成驗證。\n\n')+s[end:]
    docs[NAV[1]]=s
    s=(REPO/NAV[2]).read_text(encoding='utf8').replace('R85–R104','R85–R105').replace('## 最新追加：R103','## 歷史追加：R103')
    note='## 最新追加：R105固定原生平均\n\n[完成頁](../'+PAGE+')：'+intro+'\n\n'+(
        '[固定計畫](../evidence/reports/round105_plan.md)／[報告](../evidence/reports/round105_report.md)／[精確複製清單](../evidence/outputs/research_snapshot_r105_20260929/copy_manifest.json)。保留舊來源與限制，0新fit，不包原始交易表、逐筆向量或learned states。\n\n')
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
    paths={'main':'outputs/round105/execution.json','evaluation':'outputs/round105/evaluation/execution.json',
           'summary':'outputs/round105_summary/execution.json','launcher':'outputs/round105_launcher_20260929/execution.json'}
    costs={}
    for key,path in paths.items():v=bound(path);assert v['exit_code']==0;costs[key]=v['elapsed_seconds']
    result=bound('outputs/round105/evaluation/results.json');seal=bound('outputs/round105/sealed_manifest.json')
    bound('outputs/round105_summary/summary.json');bound('outputs/round105_summary/receipt.json')
    assert result['status']=='completed_three_window_evaluation' and set(result['stages'])==set(STAGES)
    assert result['selected_submission']=='v22' and result['Private_score'] is None and not result['new_CSV']
    assert all(seal[k]==0 for k in ('new_fit_calls','new_native_fits','new_NN_fits','new_native_tree_fits'))
    assert seal['all_endpoints_replay_exact'] and seal['all_saved_vectors_reload_exact'] and seal['learned_R_S_states_loaded']
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
