import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const QA=path.resolve(process.argv[2]||path.join(import.meta.dirname,'..','rebuild'));
const MODULES=process.env.RUNTIME_NODE_MODULES||'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SKILL=process.env.PRESENTATIONS_SKILL_DIR||'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(MODULES+'/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const N=JSON.parse(await fs.readFile(path.join(import.meta.dirname,'content.json'),'utf8'));
const curve=JSON.parse(await fs.readFile(path.join(import.meta.dirname,'training_curve.json'),'utf8'));
const FONT='Microsoft JhengHei';
const C={bg:'#FCFCFA',ink:'#23343A',text:'#3A4A4E',muted:'#718082',teal:'#276E68',blue:'#315D8A',orange:'#BA6548',line:'#E1E7E4',light:'#EEF3F0',white:'#FFFFFF',cover:'#1F373B'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
function tx(s,text,x,y,w,h,size=28,o={}){
 const a=s.shapes.add({name:o.name||'body',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=text;a.text.style={typeface:o.mono?'Consolas':FONT,fontSize:size,color:o.color||C.text,bold:!!o.bold,alignment:o.align||'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};return a;
}
function slide(n,title){const s=p.slides.add();s.background.fill=C.bg;if(title)N.slides[n-1].title=title;
 tx(s,title||N.slides[n-1].title,80,48,1120,76,44,{bold:true,color:C.ink,name:'slide-title'});
 tx(s,String(n).padStart(2,'0'),1150,674,50,25,17,{color:C.muted,align:'right'});
 const a=N.slides[n-1];s.speakerNotes.textFrame.setText(a.script+'\n\n補充理由\n'+a.why+'\n\n資料依據\n課程交易資料、保存的程式與發行證據、方法與程式導覽。');return s;}
function text(s,t,x,y,w=1120,h=100,size=28,o={}){return tx(s,t,x,y,w,h,size,o);}
function caption(s,t,y=627){tx(s,t,80,y,1110,60,23,{color:C.muted});}
function heading(s,t,x,y,w=500){tx(s,t,x,y,w,45,28,{bold:true,color:C.teal});}
function table(s,values,{x=80,y=190,w=1120,h=330,widths,font=26,highlight=-1,numberCols=[]}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({outside:{fill:C.bg,width:1,style:'solid'},insideVertical:{fill:C.bg,width:1,style:'solid'},insideHorizontal:{fill:C.line,width:0.8,style:'solid'}});
 for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;
  for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?C.light:(r===highlight?'#E4EFEB':C.bg);
   a.text.style={typeface:FONT,fontSize:font,color:r===0?C.ink:C.text,bold:r===0,alignment:numberCols.includes(c)?'right':'left',verticalAlignment:'middle',autoFit:'none',insets:{left:14,right:16,top:12,bottom:10}};
  }
 }return t;
}
function bar(s,cats,vals,{x=600,y=192,w=590,h=350,max,min=0,format='0',horizontal=false,points,fill=C.teal,labels=true}={}){
 const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cats,series:[{name:'數值',values:vals,fill,points,valuesFormatCode:format}],barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:140},hasLegend:false,chartFill:'none',plotAreaFill:'none',
  xAxis:{textStyle:{typeface:FONT,fontSize:24,fill:C.text},line:{fill:C.line,width:1},majorGridlines:horizontal?{fill:C.line,width:1}:null},
  yAxis:{min,max,numberFormatCode:format,textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},
  dataLabels:{showValue:labels,position:'outEnd',numberFormatCode:format,textStyle:{typeface:FONT,fontSize:28,fill:C.ink}}});applyPresentationChartFont(c,{fontFamily:FONT});return c;
}

// A restrained typographic cover. All foreground text stays editable.
{const s=slide(1);s.background.fill=C.cover;s.shapes.items[0].delete();
 tx(s,'台灣房屋銷售價格\n預測挑戰賽',88,178,1120,204,64,{bold:true,color:C.white,name:'slide-title'});
 text(s,'姓名：廖子頤\n學號：315581048',94,548,850,90,27,{color:'#E5EEEB'});}
{const s=slide(2,'房屋條件與價格答案');
 heading(s,'特徵',80,181);text(s,'地點、面積、樓層\n建物類型、交易日期',80,245,500,130,34);
 heading(s,'標籤',700,181);text(s,'已知成交總價',700,245,500,100,38,{bold:true,color:C.ink});
 text(s,'訓練：用條件和答案調整規則\n推論：用保存的規則估計新交易',80,465,1120,115,29);
 s.images.add({blob:new Uint8Array(await fs.readFile(path.join(import.meta.dirname,'assets/house_illustration.png'))),contentType:'image/png',alt:'青綠色房屋示意插圖',fit:'contain',position:{left:740,top:330,width:450,height:275}});
 caption(s,'回歸預測連續金額。未知交易仍需要獨立驗證。');}
{const s=slide(3,'估價誤差與 MAE');
 text(s,'每筆相差金額取絕對值，再求平均',80,141,1120,55,29);
 table(s,[['教學假設','真實總價','預測總價','絕對誤差'],['交易 A','1,000萬','1,100萬','100萬'],['交易 B','1,000萬','950萬','50萬']],{x:80,y:249,w:720,h:225,widths:[150,190,190,190],font:25,numberCols:[1,2,3]});
 heading(s,'MAE',900,221,300);text(s,'75',895,281,230,120,90,{bold:true,color:C.teal});text(s,'萬元',1048,339,140,55,32);
 text(s,'（100 + 50）÷ 2',899,437,290,56,25,{color:C.muted});
 caption(s,'MAE 越低越好。高估和低估不會互相抵銷，也不表示每筆都相差75萬元。');}
{const s=slide(4,'文字轉換的實際例子');
 text(s,'原始文字',270,151,450,40,23,{color:C.muted});text(s,'轉換結果',815,151,385,40,23,{color:C.muted});
 const rows=[['數值','1,200.5','1200.5'],['日期','112年8月11日','2023-08-11'],['樓層','十二層／地下二層','12／−2']];
 for(let i=0;i<3;i++){let y=215+i*100;heading(s,rows[i][0],80,y,150);text(s,rows[i][1],270,y,450,62,30);text(s,rows[i][2],815,y,385,62,34,{bold:true,color:C.teal});}
 text(s,'地址拆地區與道路，交易筆棟數拆土地、建物及車位數量',80,528,1120,60,26);
 caption(s,'缺失與異常標記保留。填補、尺度及類別編碼只由當期訓練資料建立。');}
{const s=slide(5,'輸入特徵與排除欄位');
 heading(s,'模型可用的線索',80,181);text(s,'面積、用途與樓層\n地區、交易時間\n不含價格的道路位置',80,249,520,170,31);
 heading(s,'排除的欄位',700,181);text(s,'編號與成交總價\n完整備註、解約資訊\n不可用日期與常數欄位',700,249,500,170,31);
 text(s,'87 基礎欄位 + 8 地理欄位 = 95 欄',80,509,1120,66,34,{bold:true,color:C.ink});
 caption(s,'其中23欄為類別。道路代表點提供大致位置，並非精確門牌。');}
{const s=slide(6,'用較早交易預測下一年');
 table(s,[['訓練資料截止','後一年評分期間'],['2021-08-10','2021-08-11 至 2022-08-10'],['2022-08-10','2022-08-11 至 2023-08-10'],['2023-08-10','2023-08-11 至 2024-08-10'],['2023-02-10','2023-02-11 至 2024-02-10']],{y:182,h:350,widths:[400,720],font:28});
 caption(s,'各起點重建模型、編碼及價格參考。窗口已反覆使用且重疊，無法視為四次獨立盲測。');}
{const s=slide(7,'樹模型與神經網路');
 heading(s,'樹模型',80,187);text(s,'學習條件與門檻',80,252,520,70,36,{bold:true,color:C.ink});text(s,'許多小樹逐步修正誤差\nCatBoost、LightGBM',80,354,520,124,29);
 heading(s,'神經網路',700,187);text(s,'學習條件的共同影響',700,252,500,100,34,{bold:true,color:C.ink});text(s,'數值與類別轉成可學習表示\n經多層計算調整估價',700,354,500,124,29);
 caption(s,'兩種學法可能互補。模型增加是否有效，仍以完整方法的驗證結果判斷。');}
{const s=slide(8,'面積與總價誤差');
 text(s,'總價 = 單價 × 有效面積',80,179,530,100,36,{bold:true,color:C.ink});
 text(s,'單價同樣估錯\n每平方公尺1萬元',80,332,490,100,31);
 text(s,'面積越大，總價誤差越大',80,499,500,78,29,{bold:true,color:C.teal});
 bar(s,['50 m²','100 m²'],[50,100],{x:635,y:215,w:565,h:350,max:120});
 text(s,'總價誤差，單位：萬元',655,158,530,45,23,{color:C.muted});
 caption(s,'面積權重對齊總價差距。另保留直接總價與對數學法，最後皆還原為元。');}
{const s=slide(9,'日期權重的影響');
 text(s,'365 日半衰期',80,183,900,65,36,{bold:true,color:C.ink});
 bar(s,['最新日期','早一年','早兩年'],[1,.5,.25],{x:80,y:228,w:740,h:338,max:1.2,format:'0.00'});
 heading(s,'相差一年',920,275,280);text(s,'1 : 2',911,340,290,100,65,{bold:true,color:C.teal});text(s,'較舊與較新交易\n原始權重的比例',920,459,280,95,26);
 caption(s,'圖示為正規化前相對權重。權重改變學習影響，房價本身不乘權重。');}
{const s=slide(10,'參考單價與分組先驗');
 text(s,'參考單價 × 有效面積 × 預測倍率',80,180,1120,80,36,{bold:true,color:C.ink});
 heading(s,'參考單價',80,320);text(s,'依地區、交易類型與建物類型\n資料不足時逐層放寬',80,385,540,116,29);
 heading(s,'建案分組 OOF',700,320);text(s,'分成5組，每組只參考其他4組\nquery 使用當期完整 fit',700,385,500,116,29);
 caption(s,'建案分組與時間隔離處理不同問題。NN固定30輪，SmoothL1轉折為10萬元。');}
{const s=slide(11,'六模型組成三條分支');
 text(s,'所有輸出先換回總價元',80,141,1120,55,28,{color:C.muted});
 table(s,[['分支','主要部分 75%','共同部分 25%'],['單價','單價 CatBoost','單價 LightGBM'],['神經網路','面積 NN、參考單價 NN 平均','同一個 LightGBM'],['總價／對數','總價、對數 CatBoost 平均','同一個 LightGBM']],{y:236,h:288,widths:[225,535,360],font:25});
 caption(s,'假例：1,000萬 × 75% + 1,200萬 × 25% = 1,050萬。固定比例未證明最佳。');}
{const s=slide(12,'成交參考與空間修正');
 heading(s,'成交參考 R',80,202,530);text(s,'同建案、地址或道路的成交',80,274,530,94,33,{bold:true,color:C.ink});text(s,'結合物件條件\n與當期學到的時間趨勢',80,403,530,120,29);
 heading(s,'空間修正 S',700,202,500);text(s,'附近且相容的交易資訊',700,274,500,94,33,{bold:true,color:C.ink});text(s,'同時讀取初估與 R 的結果\n參考不足時回退',700,403,500,120,29);
 caption(s,'三條分支各自完成修正。價格參考只取自當期訓練資料。');}
{const s=slide(13,'逐筆取三路中位數');
 text(s,'同一間房，三個修正後總價',80,151,1120,55,29,{color:C.muted});
 const xs=[80,490,900],vals=['1,000','1,100','1,600'];
 for(let i=0;i<3;i++){heading(s,'分支 '+String.fromCharCode(65+i),xs[i],268,300);text(s,vals[i],xs[i]-4,342,330,120,68,{bold:i===1,color:i===1?C.teal:C.text});text(s,'萬元',xs[i],467,300,48,27,{color:C.muted});}
 caption(s,'中間值為1,100萬元，可減少單一路線極端值的影響，仍可能共同高估或低估。');}
{const s=slide(14,'四個時間窗口的完整結果');
 text(s,'總價 MAE，單位：元，越低越好',80,141,1120,50,27,{color:C.muted});
 const t=table(s,[['評分期間','基準方法','此方法','此方法 − 基準'],['2021–2022 八月起點','2,309,352.36','2,310,236.02','+883.66'],['2022–2023 八月起點','1,407,387.07','1,405,971.19','−1,415.88'],['2023–2024 八月起點','1,383,228.41','1,381,702.87','−1,525.54'],['2023–2024 二月起點','1,336,410.67','1,336,107.62','−303.06']],{y:225,h:335,widths:[365,260,260,235],font:24,numberCols:[1,2,3],highlight:3});
 for(let r=1;r<5;r++)t.getCell(r,3).text.style={typeface:FONT,fontSize:24,color:r===1?C.orange:C.teal,alignment:'right',verticalAlignment:'middle',autoFit:'none',insets:{left:14,right:16,top:12,bottom:10}};
 caption(s,'差值由未取整 MAE 計算。三期改善、一期退步，且未同期配對重訓基準。');}
{const s=slide(15,'分群退步與不利對照');
 heading(s,'最近年度的 MAE 差值',80,175,650);text(s,'此方法減基準，單位：元',80,225,650,48,23,{color:C.muted});
 const t=table(s,[['群別','差值'],['既有建案','+213.10'],['新建案','−2,841.61']],{x:80,y:306,w:620,h:195,widths:[345,275],font:30,numberCols:[1]});
 for(let r=1;r<3;r++)t.getCell(r,1).text.style={typeface:FONT,fontSize:30,color:r===1?C.orange:C.teal,alignment:'right',verticalAlignment:'middle',autoFit:'none',insets:{left:14,right:16,top:12,bottom:10}};
 heading(s,'另一對照方法',835,178,365);text(s,'208.30',830,277,370,100,65,{bold:true,color:C.orange});text(s,'萬元 Public MAE',840,392,360,50,27);
 text(s,'離線較好，公開成績反而退步',835,478,365,94,26);
 caption(s,'最早年退步，原強推薦條件未通過。重抽樣區間約[−3,595, +523]元，跨過0。');}
{const s=slide(16,'Public 成績與證據限制');
 heading(s,'Public MAE',80,191,660);text(s,'199.58',73,263,710,145,100,{bold:true,color:C.teal});text(s,'萬元',80,420,180,60,28,{color:C.muted});
 heading(s,'相對基準',850,204,350);text(s,'−0.1367%',842,281,370,95,54,{bold:true,color:C.ink});text(s,'約降低2,732元\n改善幅度有限',854,409,340,110,28);
 text(s,'精確值：1,995,836.2136 元，使用者文字回報',80,553,1120,55,25,{color:C.muted});
 caption(s,'Public 約占test 30%。其餘約70%的 Private 成績仍未知。');}
{const s=slide(17,'實際訓練與時間驗證曲線');
 text(s,'單一參考單價神經網路，另行固定設定診斷',80,132,1120,45,25,{color:C.muted});
 tx(s,'未\n加\n權\n總\n價\nMAE',35,282,74,255,24,{color:C.text,align:'center'});
 tx(s,'百萬元',93,184,150,36,22,{color:C.muted});
 const cv=v=>Number(v.toPrecision(15));
 const c=s.charts.add('scatter',{position:{left:102,top:190,width:1098,height:412},series:[{name:'訓練資料',xValues:curve.epochs,values:curve.train_mae_twd.map(cv),valuesFormatCode:'0',line:{fill:C.blue,width:3.5,style:'solid'},marker:{symbol:'none'}},{name:'時間驗證',xValues:curve.epochs,values:curve.validation_mae_twd.map(cv),valuesFormatCode:'0',line:{fill:C.orange,width:3.5,style:'solid'},marker:{symbol:'none'}}],scatterOptions:{style:'line'},hasLegend:true,legend:{position:'top',overlay:false,textStyle:{typeface:FONT,fontSize:24,fill:C.text}},chartFill:'none',plotAreaFill:'none',xAxis:{title:{text:'訓練輪數 Epoch',textStyle:{typeface:FONT,fontSize:23,fill:C.text}},min:1,max:30,majorUnit:5,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:0,max:2500000,majorUnit:500000,numberFormatCode:'0.0,,',textStyle:{typeface:FONT,fontSize:24,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(c,{fontFamily:FONT});
 caption(s,'固定30輪，兩線皆關閉 dropout 後評分。未依最低點選輪數，不能證明沒有過擬合。',618);
 text(s,'此圖涵蓋單一網路。Epoch為學習輪數，不代表模型容量或VC維度。',80,660,1045,37,21,{color:C.muted});}
{const s=slide(18,'最終訓練與提交檔案');
 heading(s,'最終有答案資料',80,195,530);text(s,'467,221',73,268,560,118,77,{bold:true,color:C.teal});text(s,'筆交易用於最終訓練',80,411,530,70,30);
 heading(s,'未知交易估價',700,195,500);text(s,'47,073',692,268,510,118,77,{bold:true,color:C.ink});text(s,'筆預測完整對齊編號與順序',700,411,500,70,29);
 text(s,'新增1支參考單價 NN，其餘5模型沿用保存預測向量',80,549,1120,55,27);
 caption(s,'新 NN 重載輸出一致。全部六模型的乾淨環境端到端重現尚未驗證。');}
{const s=slide(19,'結論');
 text(s,'完整流程以未知交易的\n總價誤差檢查估價能力',80,174,1120,155,49,{bold:true,color:C.ink});
 text(s,'整理資料並建立價格表示，六模型組成三分支\n各自加入成交與空間參考，再逐筆取中位數',80,380,1120,118,30);
 text(s,'Public 小幅改善，仍有跨期退步與反例\nPrivate 未知，尚不能保證未來表現',80,552,1120,100,28,{color:C.teal});}
{const s=slide(20,'附錄　完整融合公式');
 text(s,'各符號皆為已還原成元的總價',80,142,1120,50,26,{color:C.muted});
 text(s,'B_unit = .75*C_unit + .25*L_unit\n\nB_nn = .75*(.5*N_area + .5*N_prior)\n       + .25*L_unit\n\nB_log = .75*(.5*C_raw + .5*C_log)\n        + .25*L_unit',80,242,730,310,26,{mono:true,color:C.ink});
 text(s,'U = S(B_unit, R(B_unit))\nN = S(B_nn, R(B_nn))\nH = S(B_log, R(B_log))',850,242,350,165,22,{mono:true,color:C.text});
 text(s,'median(N, H, U)',845,467,355,65,29,{mono:true,bold:true,color:C.teal});
 caption(s,'R為成交參考與時間趨勢，S同時讀取初估與R結果。最後取中位數，不再乘面積。');}
{const s=slide(21,'附錄　固定訓練設定');
 table(s,[['參考單價 NN','固定內容'],['隱藏層','512、256、128，LayerNorm 與 ReLU'],['損失','日期加權 SmoothL1，β = 0.1 百萬元'],['輪數／批次／種子','30／2,048／42，採最後一輪'],['dropout／category dropout','0.1／0.1'],['最佳化','AdamW，lr 0.001，weight decay 0.0001']],{y:180,h:372,widths:[435,685],font:24});
 caption(s,'基準 NN 採日期加權 MAE。本次未同期配對重訓基準，其餘模型保留各自損失。');}
{const s=slide(22,'附錄　名詞對照');
 const terms=[['特徵／標籤','房屋條件／已知成交總價'],['訓練／推論','學習規則／用保存規則估價'],['先驗／分組 OOF','參考單價／使用其他組的標籤'],['權重','每筆誤差在學習時的影響'],['過擬合','太貼合練習資料，新交易可能失準'],['中位數','由小到大排序後的中間值']];
 for(let i=0;i<6;i++){const x=i<3?80:700,y=181+(i%3)*150;heading(s,terms[i][0],x,y,500);text(s,terms[i][1],x,y+60,500,87,27);}}

if(p.slides.items.length!==22)throw Error('Slide count');
await fs.mkdir(QA,{recursive:true});
await fs.writeFile(QA+'/narrative.json',JSON.stringify(N,null,2));
await fs.writeFile(QA+'/presentation.json',JSON.stringify(p.toProto()));
await fs.writeFile(QA+'/source_data.json',JSON.stringify({curve,mainHan:3758,mainSlides:19,appendixSlides:3,nativeTableSlides:[3,6,11,14,15,21],nativeChartSlides:[8,9,17],retainedNotesUnchanged:true,modelChanged:false},null,2));
await (await PresentationFile.exportPptx(p)).save(QA+'/candidate.pptx');
console.log(JSON.stringify({candidate:QA+'/candidate.pptx',slides:22}));
