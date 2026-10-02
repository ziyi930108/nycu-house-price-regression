from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/v22_審查資料夾_20261002_r2'
SOURCE=ROOT/'outputs/v22_reader_review_20261002/pdf_revision'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((SOURCE/'final_receipt.json').read_text('utf-8'))
assert receipt['status']=='passed' and receipt['per_page_normalized_text_matches_declared_repair']
pdf=SOURCE/receipt['artifact'];assert sha(pdf)==receipt['output_sha256']
target=OUT/'03_正式報告/臺灣房價預測_v22_正式報告.pdf'
old_sha=sha(target);assert old_sha=='e8470b323cdbcf6b340b06fd8106215cdaa301912a7bcbc1f69649139dc104ca'
shutil.copy2(pdf,target)
code=OUT/'04_程式與證據/正式報告字型修正';code.mkdir(exist_ok=False)
for name in ('build_report_pdf_font_fixed.py','REPORT_font_fixed.txt','build_receipt.json','final_receipt.json','verification.json'):
    shutil.copy2(SOURCE/name,code/name)
(code/'README.md').write_text('''# 正式報告負號字形修正

[本包正式PDF](../../03_正式報告/臺灣房價預測_v22_正式報告.pdf)是原報告的衍生修正版。第10頁「最新年度 -1,999.31 元」原負號顯示成方框，現在改為可顯示的ASCII減號；數值與模型方法沒有變。

11頁全部重新渲染，僅第10頁有像素差異，其餘10頁RGB像素完全相同。第10頁已檢視，負號正常、沒有裁切或重疊；逐頁正文核對符合宣告修補。[最終收據](final_receipt.json)、[來源與生成記錄](build_receipt.json)、[機器核对](verification.json)可供查證。機器核對當時的待看圖標記屬先前階段，最終以final_receipt為準。

`REPORT_font_fixed.txt`內容仍為Markdown格式，為保留字元身分而以文字來源存放。裡面的相對證據路徑以原程式包的`v22_github`根目錄為基準；要點來源連結，請解壓程式ZIP後閱讀原REPORT.md。這份txt是生成輸入，不是獨立的證據導覽頁。

修正版builder沿用原A4版型、字型與表格，只新增Unicode負號正規化和拒絕覆寫。生成依賴已裝有ReportLab與pypdf的Python，以及Windows Microsoft JhengHei字型。從本資料夾執行：

```powershell
$pdfPython = Read-Host '具備ReportLab與pypdf的Python完整路徑'
& $pdfPython -B -X utf8 ./build_report_pdf_font_fixed.py --input ./REPORT_font_fixed.txt --output ./new-output/report_font_fixed.pdf --font-dir C:/Windows/Fonts
if ($LASTEXITCODE -ne 0) { throw 'PDF生成失敗，請保留新輸出與終端訊息。' }
```

輸出PDF和同名build.json都必須不存在。再次生成的PDF時間欄位可能不同，不承諾整檔SHA相同。原實測只用既有環境，不是新電腦安裝測試，也沒有重跑機器學習。原封存PDF、REPORT.md、原builder均保留；本次修改只進入新r2資料夾。
''',encoding='utf-8')
p=OUT/'00_審查指南.md';s=p.read_text('utf-8').replace('## 10月2日修正版','## 前版10月2日字幕修正版（r1）')
s=s.replace('本次重點是把第12頁的','原審查版重點是把第12頁的',1);p.write_text(s,encoding='utf-8')
p=OUT/'02_講稿與排練/排練與問答.md';s=p.read_text('utf-8').replace('日期：2026-10-01。','講稿來源：2026-10-01；使用說明更新：2026-10-02。',1);p.write_text(s,encoding='utf-8')
p=OUT/'04_程式與證據/先讀我.md';s=p.read_text('utf-8')
s+='\n\n本包03正式PDF已修復負號字形，來源和核對見[報告字型修正](正式報告字型修正/README.md)。程式ZIP裡的REPORT.md仍是原文字來源，PDF保留發布時舊版；需要繳交閱讀時優先使用03內的修正版。\n'
p.write_text(s,encoding='utf-8')
result={'status':'integrated','old_pdf_sha256':old_sha,'new_pdf_sha256':sha(target),
    'declared_visible_change':'One missing minus sign on page 10; other ten rendered pages pixel-identical',
    'source_folder':SOURCE.relative_to(ROOT).as_posix(),'target':target.relative_to(ROOT).as_posix(),
    'text_and_numbers_preserved_except_declared_glyph_repair':True}
with (WORK/'report_integration.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
