"""Prepare a new review folder with clearer rehearsal and interpreter guidance.

Report typography repair is integrated separately after its visual QA.
"""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OLD=ROOT/'deliverables/v22_審查資料夾_20261002';OUT=ROOT/'deliverables/v22_審查資料夾_20261002_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
assert all(sha(OLD/x['path'])==x['sha256'] for x in manifest['files'])
assert not OUT.exists()
shutil.copytree(OLD,OUT,ignore=lambda d,n: ['manifest.json'] if Path(d)==OLD else [])
note='4,065字只計第1至20頁的口述正文。「先記住」與「設計理由」是排練提示，意思已融入正文，不需要再逐字重複朗讀；附錄與問答也不計入主講。'
p=OUT/'02_講稿與排練/完整口語講稿.md';s=p.read_text('utf-8')
assert s.count('全文4,065純漢字。')==1
s=s.replace('全文4,065純漢字。','主講正文4,065純漢字。')
anchor='三款簡報共用。主講第1至20頁，附錄第21至23頁不計時。'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n\n'+note)
p.write_text(s,encoding='utf-8')
p=OUT/'02_講稿與排練/上台口述提示.md';s=p.read_text('utf-8')
anchor='下面每頁的提醒是找回思路用，完整說明仍在口語講稿。'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n\n'+note);p.write_text(s,encoding='utf-8')
p=OUT/'02_講稿與排練/排練與問答.md';s=p.read_text('utf-8')
anchor='## 排練方法';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n\n'+note);p.write_text(s,encoding='utf-8')
p=OUT/'index.html';s=p.read_text('utf-8').replace('v22_review_20261002','v22_review_20261002_r2')
anchor='<h2>逐頁口語講稿</h2>';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'<p>每頁粗體重點與「設計理由」是排練提示，中間段落才是完整口述正文；不必三部分都逐字再念。4,065字與配速只計20頁主講正文。</p>')
anchor='這次改善影片字幕斷行，讓英文名稱和中文術語更容易閱讀。';assert s.count(anchor)==1
s=s.replace(anchor,'字幕換行已修正；本次再補清楚講稿怎麼念、程式該用哪個Python，並修復正式報告的負號字形。')
anchor='<h2>字幕更新與講稿修改</h2>';assert s.count(anchor)==1
s=s.replace(anchor,'<h2>本次使用說明與字形修正</h2><p>這次讓主講正文與排練提示更容易區分，補上固定Python路徑的操作方式，並修正報告第10頁負號缺字。模型、數值、PPTX和影片保持。</p>')
p.write_text(s,encoding='utf-8')
p=OUT/'00_審查指南.md';s=p.read_text('utf-8')
anchor='## 建議這樣看';assert s.count(anchor)==1
s=s.replace(anchor,'## 這份r2先修正什麼\n\n'+note+'\n\n正式PDF修復一個顯示成方框的負號；原數值與方法不變。程式操作另外附[Python環境說明](04_程式與證據/執行程式先選對Python.md)，避免安裝與推論用了不同的Python。這些都不改投影片主稿、影片、模型或提交價格。\n\n'+anchor)
p.write_text(s,encoding='utf-8')
(OUT/'04_程式與證據/執行程式先選對Python.md').write_text('''# 執行程式時，固定使用同一個Python

程式ZIP中README示範將套件安裝進`.venv`，後面的裸`python`卻可能指向系統的另一套Python。沒有先啟用虛擬環境時，照抄這些命令可能遇到找不到套件，或得到不同的運算結果。以下補充不修改封存的程式ZIP。

先完整解壓`v22_程式與研究紀錄.zip`，在其`v22_github`目錄打開PowerShell。保留原README的依賴與artifact要求，所有安裝、收集、推論及測試命令都明確使用同一個ML環境執行檔。**下列是操作指引，並未在全新環境重新安裝或訓練。**

若已按原README建立`.venv`，先確認路徑：

```powershell
$v22Python = (Resolve-Path -LiteralPath '.venv/Scripts/python.exe').Path
& $v22Python -c 'import sys; print(sys.executable); print(sys.version)'
```

如果要使用已有ML環境，將`$v22Python`改成該環境的Python完整路徑，不要指向Node、FFmpeg或其他程式。不要只複製虛擬環境資料夾就假設換電腦可用。原實測為Python3.12.14，使用ML與Torch兩套既有環境，版本及硬體詳見程式包`docs/REPRODUCIBILITY.md`。

原README的相關命令應一致寫成：

```powershell
# 要安裝時，安裝進指定的ML環境；新安裝尚未端到端驗證。
& $v22Python -m pip install -r requirements-inference.txt

# 僅在有原研究工作區時，依manifest複製17項私有artifact。
& $v22Python scripts/collect_artifacts.py --workspace C:/NYCU_ML

# 格式測試；通過不等於完成模型推論或重訓。
& $v22Python -m unittest discover -s tests -v
```

`--workspace`是你真正保存資料、模型與狀態的原研究目錄；換電腦需要自行移轉相應檔案，程式ZIP不含約1.45GB私有artifact。不要拿舊預測向量代替缺少的模型或自行放寬SHA檢查。

原始test與sample要按程式包DATA/README放到`data/`。若NN使用另一套Torch環境，推論時明確指定：

```powershell
$v22NnPython = Read-Host 'Torch環境的Python完整路徑'
if (-not (Test-Path -LiteralPath $v22NnPython -PathType Leaf)) { throw '找不到Torch Python。' }
& $v22Python src/predict_v22.py --test data/test-reindex-test-v2.2.xlsx --sample data/test-reindex-test-sample-submission-v2.csv --output-dir runs/v22_reproduction_new --device cuda:0 --nn-python $v22NnPython
if ($LASTEXITCODE -ne 0) { throw '推論未通過，請保留輸出與日誌後查錯。' }
```

ML與NN用同一個環境時可省略`--nn-python`，但須先備妥相容的Torch/CUDA。輸出目錄必须是尚不存在的新目錄。這個入口只對固定競賽test的已保存狀態重現，不能當任意房屋估價API。

原保存模型推論已在受測GPU環境完成47,073筆，CSV與v22一致；本補充沒有重跑推論。CPU或其他硬體的數值差異不是真實MAE，也不能據此判斷Private好壞。GPU重訓與乾淨電腦安裝仍未驗證。
''',encoding='utf-8')
p=OUT/'04_程式與證據/先讀我.md';s=p.read_text('utf-8')
s=s.replace('# 程式與證據','# 程式與證據\n\n執行前先看[固定Python環境的補充](執行程式先選對Python.md)，確保安裝套件與推論使用同一套ML環境。',1)
p.write_text(s,encoding='utf-8')
p=OUT/'本次修改.md';p.write_text('''# 10月2日r2：閱讀與操作修正

主講仍為4,065字正文；新增提醒，說明「先記住／設計理由」不需重複朗讀。程式指南統一使用明確的Python執行檔。正式報告以新副本修復負號字形，原模型結果與數值不改。PPTX、主稿正文、影片與v22價格均保持。

以下保留前版修改紀錄。

'''+p.read_text('utf-8'),encoding='utf-8')
before=(OLD/'02_講稿與排練/完整口語講稿.md').read_text('utf-8').split('## 第01頁',1)[1]
after=(OUT/'02_講稿與排練/完整口語講稿.md').read_text('utf-8').split('## 第01頁',1)[1]
assert before==after
assert all(sha(OLD/x['path'])==x['sha256'] for x in manifest['files'])
result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared_pending_pdf',
    'folder':OUT.relative_to(ROOT).as_posix(),'spoken_body_and_appendices_bytes_unchanged':True,
    'new_models':0,'human_rehearsal_measured':False}
with (WORK/'preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
