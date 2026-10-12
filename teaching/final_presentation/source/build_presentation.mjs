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
 const a=N.slides[n-1];s.speakerNotes.textFrame.setText(a.script+'\n\n補充理由\n'+a.why+'\n\n資料依據\n'+([9,10].includes(n)?a.sources.join('\n'):'課程交易資料、保存的程式與發行證據、方法與程式導覽。'));return s;}
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
{const s=slide(4,'原始文字的處理步驟');
 table(s,[['原始欄位','我怎麼處理','整理後的資訊'],['數值','去除逗號，再轉成數值','1,200.5 變成 1200.5'],['日期','民國年加1911，再拆年月','112年變成2023年'],['樓層','中文數字轉數值，地下層記負數','十二層=12／地下二層=−2'],['地址','台統一為臺，依名稱結尾拆欄','縣市／行政區／道路／段別'],['交易筆棟數','依土地、建物、車位擷取數字','土地1、建物1、車位0']],{y:178,h:420,widths:[220,500,400],font:24});
 caption(s,'空白及解析失敗另留標記。填補與類別編碼，只從訓練資料建立規則。');}
{const s=slide(5,'模型輸入欄位的選擇');
 text(s,'先確認每個欄位在估價時能否使用',80,143,1120,48,29,{color:C.muted});
 table(s,[['資料欄位','實際做法','選擇的原因'],['面積、樓層、地點、日期','整理成模型輸入','描述房屋與交易條件'],['成交總價','另存為訓練答案','避免答案混入輸入'],['編號','保留對齊，排除訓練','識別交易，不代表價格'],['完整備註、解約等資訊','排除全文與不可用欄位','避免價格或事後資訊']],{y:227,h:330,widths:[405,350,365],font:25});
 caption(s,'數值與類別採不同處理。所有需學習的轉換規則，只用當期訓練資料建立。');}
{const s=slide(6,'用較早交易預測下一年');
 table(s,[['訓練資料截止','後一年評分期間'],['2021-08-10','2021-08-11 至 2022-08-10'],['2022-08-10','2022-08-11 至 2023-08-10'],['2023-08-10','2023-08-11 至 2024-08-10'],['2023-02-10','2023-02-11 至 2024-02-10']],{y:182,h:350,widths:[400,720],font:28});
 caption(s,'各起點重建模型、編碼及價格參考。窗口已反覆使用且重疊，無法視為四次獨立盲測。');}
{const s=slide(7,'模型的價格目標與訓練步驟');
 text(s,'單價樹模型：訓練答案＝已知總價 ÷ 有效面積',80,143,1120,40,25,{color:C.teal});
 table(s,[['模型路線','先預測什麼','總價如何產生'],['單價樹模型、面積網路','每平方公尺單價','預測單價 × 有效面積'],['直接總價樹模型','總價','直接換回總價元'],['對數總價樹模型','取對數後的總價','將預測值還原為總價'],['參考單價網路','相對參考價的倍率','參考單價 × 有效面積 × 倍率']],{x:80,y:198,w:1120,h:325,widths:[410,300,410],font:24});
 text(s,'共同訓練：輸入條件、產生預測、依目標計算損失、更新模型',80,550,1120,42,27,{color:C.ink});
 text(s,'樹模型：增加小樹修正誤差　神經網路：反向更新參數',80,596,1120,35,25);
 text(s,'各模型先換回總價元，再進行分支融合與修正。驗證時固定參數。',80,645,1045,36,22,{color:C.muted});}
{const s=slide(8,'面積與總價誤差');
 text(s,'總價 = 單價 × 有效面積',80,179,530,100,36,{bold:true,color:C.ink});
 text(s,'單價同樣估錯\n每平方公尺1萬元',80,332,490,100,31);
 text(s,'面積越大，總價誤差越大',80,499,500,78,29,{bold:true,color:C.teal});
 bar(s,['50 m²','100 m²'],[50,100],{x:635,y:215,w:565,h:350,max:120});
 text(s,'總價誤差，單位：萬元',655,158,530,45,23,{color:C.muted});
 caption(s,'面積權重對齊總價差距。另保留直接總價與對數學法，最後皆還原為元。');}
{const s=slide(9,'新舊交易的學習影響');
 text(s,'房價會隨時間改變，因此讓近期成交有較大影響',80,140,1120,43,29,{color:C.muted});
 text(s,'權重：一筆交易對模型學習的影響程度',80,190,1120,45,29,{bold:true,color:C.teal});
 bar(s,['最新日期','早一年','早兩年'],[1,.5,.25],{x:80,y:250,w:740,h:320,max:1.2,format:'0.00'});
 text(s,'圖中的數字代表\n相對學習影響',900,280,300,90,28);
 text(s,'每早一年\n影響降為一半',900,415,300,110,36,{bold:true,color:C.teal});
 caption(s,'圖為正規化前的相對比例。權重乘在訓練損失上，成交價本身不變。',622);}
{const s=slide(10,'相似交易提供參考單價');
 text(s,'用同類交易的中位單價作參考，近期交易影響較大',80,153,1120,46,29,{color:C.muted});
 text(s,'教學假例',80,235,1120,35,23,{color:C.muted});
 text(s,'20萬/m²',80,285,275,70,46,{bold:true,color:C.ink});
 text(s,'× 50m²',355,285,265,70,46,{bold:true,color:C.ink});
 text(s,'× 1.1',620,285,280,70,46,{bold:true,color:C.teal});
 text(s,'＝1,100萬',900,285,300,70,46,{bold:true,color:C.teal});
 text(s,'參考單價',80,362,275,44,26);
 text(s,'有效面積',390,362,230,44,26);
 text(s,'模型學到的倍率',650,362,260,44,26);
 text(s,'預測總價',940,362,260,44,26);
 heading(s,'參考價的分類',80,465,535);
 text(s,'依地區、交易及建物類型整理\n同類不足30筆，就逐層放寬',80,519,535,95,27);
 heading(s,'訓練時隔開答案',700,465,500);
 text(s,'同建案放同組，分成5組\n每組只參考其他4組的交易',700,519,500,95,27);
 caption(s,'這份參考單價稱為「分組先驗」，由當期訓練交易建立。',640);}
{const s=slide(11,'用後一年交易檢查訓練過程');
 text(s,'較早交易學習：321,550筆　後一年驗證：133,697筆',80,132,1120,38,25,{color:C.muted});
 text(s,'訓練截至2023-08-10　驗證：2023-08-11 至 2024-08-10',80,169,1120,32,22,{color:C.muted});
 tx(s,'未\n加\n權\n總\n價\nMAE',35,282,74,255,24,{color:C.text,align:'center'});
 tx(s,'百萬元',93,210,150,36,22,{color:C.muted});
 const cv=v=>Number(v.toPrecision(15));
 const c=s.charts.add('scatter',{position:{left:102,top:215,width:1098,height:387},series:[{name:'較早訓練交易',xValues:curve.epochs,values:curve.train_mae_twd.map(cv),valuesFormatCode:'0',line:{fill:C.blue,width:3.5,style:'solid'},marker:{symbol:'none'}},{name:'後一年交易（未參與訓練）',xValues:curve.epochs,values:curve.validation_mae_twd.map(cv),valuesFormatCode:'0',line:{fill:C.orange,width:3.5,style:'solid'},marker:{symbol:'none'}}],scatterOptions:{style:'line'},hasLegend:true,legend:{position:'top',overlay:false,textStyle:{typeface:FONT,fontSize:24,fill:C.text}},chartFill:'none',plotAreaFill:'none',xAxis:{title:{text:'訓練輪數 Epoch',textStyle:{typeface:FONT,fontSize:23,fill:C.text}},min:1,max:30,majorUnit:5,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:0,max:2500000,majorUnit:500000,numberFormatCode:'0.0,,',textStyle:{typeface:FONT,fontSize:24,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(c,{fontFamily:FONT});
 caption(s,'保存各輪模型後，固定參數分別計算兩批 MAE。此圖是另跑的一支網路診斷。',618);
 text(s,'固定30輪，不依最低點選輪數。曲線不能證明沒有過擬合，輪數也不代表模型容量。',80,660,1045,37,21,{color:C.muted});}
{const s=slide(12,'六模型組成三條分支');
 text(s,'所有輸出先換回總價元',80,141,1120,55,28,{color:C.muted});
 table(s,[['分支','主要部分 75%','共同部分 25%'],['單價','單價 CatBoost','單價 LightGBM'],['神經網路','面積 NN、參考單價 NN 平均','同一個 LightGBM'],['總價／對數','總價、對數 CatBoost 平均','同一個 LightGBM']],{y:236,h:288,widths:[225,535,360],font:25});
 caption(s,'假例：1,000萬 × 75% + 1,200萬 × 25% = 1,050萬。固定比例未證明最佳。');}
{const s=slide(13,'成交參考與空間修正');
 heading(s,'成交參考 R',80,202,530);text(s,'同建案、地址或道路的成交',80,274,530,94,33,{bold:true,color:C.ink});text(s,'結合物件條件\n與當期學到的時間趨勢',80,403,530,120,29);
 heading(s,'空間修正 S',700,202,500);text(s,'附近且相容的交易資訊',700,274,500,94,33,{bold:true,color:C.ink});text(s,'同時讀取初估與 R 的結果\n參考不足時回退',700,403,500,120,29);
 caption(s,'三條分支各自完成修正。價格參考只取自當期訓練資料。');}
{const s=slide(14,'逐筆取三路中位數');
 text(s,'同一間房，三個修正後總價',80,151,1120,55,29,{color:C.muted});
 const xs=[80,490,900],vals=['1,000','1,100','1,600'];
 for(let i=0;i<3;i++){heading(s,'分支 '+String.fromCharCode(65+i),xs[i],268,300);text(s,vals[i],xs[i]-4,342,330,120,68,{bold:i===1,color:i===1?C.teal:C.text});text(s,'萬元',xs[i],467,300,48,27,{color:C.muted});}
 caption(s,'中間值為1,100萬元，可減少單一路線極端值的影響，仍可能共同高估或低估。');}
{const s=slide(15,'跨年度檢查同一套流程');
 text(s,'每個時間起點重建流程，再計算總價 MAE（元）',80,141,1120,50,27,{color:C.muted});
 const t=table(s,[['評分期間','基準方法','此方法','此方法 − 基準'],['2021–2022 八月起點','2,309,352.36','2,310,236.02','+883.66'],['2022–2023 八月起點','1,407,387.07','1,405,971.19','−1,415.88'],['2023–2024 八月起點','1,383,228.41','1,381,702.87','−1,525.54'],['2023–2024 二月起點','1,336,410.67','1,336,107.62','−303.06']],{y:225,h:335,widths:[365,260,260,235],font:24,numberCols:[1,2,3],highlight:3});
 for(let r=1;r<5;r++)t.getCell(r,3).text.style={typeface:FONT,fontSize:24,color:r===1?C.orange:C.teal,alignment:'right',verticalAlignment:'middle',autoFit:'none',insets:{left:14,right:16,top:12,bottom:10}};}
{const s=slide(16,'驗證之後的分組檢查');
 const rows=[['01　建立分組','比對當期訓練資料，標記建案是否見過'],['02　分別評分','既有、新建案與缺失組，各自計算 MAE'],['03　查看退步','檢查整體平均是否掩蓋某一群的問題'],['04　保留反證','一起記錄跨期退步與不利對照']];
 for(let i=0;i<4;i++){let y=190+i*105;heading(s,rows[i][0],80,y,320);text(s,rows[i][1],425,y,775,80,29);}
 caption(s,'實際曾出現分群退步。歷史窗口已反覆使用，Private 成績仍未知。',638);}
if(p.slides.items.length!==16)throw Error('Slide count');
await fs.mkdir(QA,{recursive:true});
await fs.writeFile(QA+'/narrative.json',JSON.stringify(N,null,2));
await fs.writeFile(QA+'/presentation.json',JSON.stringify(p.toProto()));
await fs.writeFile(QA+'/source_data.json',JSON.stringify({curve,mainHan:N.meta.mainHan,mainSlides:16,appendixSlides:0,nativeTableSlides:[3,4,5,6,7,12,15],nativeChartSlides:[8,9,11],retainedNotesUnchanged:false,modelChanged:false},null,2));
await (await PresentationFile.exportPptx(p)).save(QA+'/candidate.pptx');
console.log(JSON.stringify({candidate:QA+'/candidate.pptx',slides:16}));
