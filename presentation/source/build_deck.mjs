import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = 'C:/NYCU_ML';
const DIR = ROOT + '/outputs/presentation_v22_20260928';
const BUILD = DIR + '/build';
const OUT = DIR + '/deliverables';
const SKILL = 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const { resolvePresentationFont, applyPresentationChartFont } = await import(pathToFileURL(SKILL + '/container_tools/artifact_tool_utils.mjs').href);
const FONT = resolvePresentationFont({fontFamily:'Microsoft JhengHei', availableFonts:['Microsoft JhengHei']});
const MONO = 'Consolas';
const C = {bg:'#F5F3EC', ink:'#203D35', body:'#34483F', muted:'#69776D', green:'#31644F', lime:'#C8DC9E', pale:'#DEE8D5', gray:'#ABB7AC', rust:'#AF553C', white:'#FFFFFF'};
const p = Presentation.create({slideSize:{width:1280,height:720}});
const ledger=[];
const feedback=JSON.parse(await fs.readFile(ROOT+'/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json','utf8'));
const results=JSON.parse(await fs.readFile(ROOT+'/outputs/round79/results.json','utf8'));
const publicScores=Object.values(feedback.screenshotted_scores);
const publicLabels=Object.keys(feedback.screenshotted_scores).map(x=>x.match(/v\d+/)[0]);
const stages=['auxiliary','development','confirmation'];
const oldMAE=stages.map(s=>results.stages[s].metrics.v20.overall_mae);
const newMAE=stages.map(s=>results.stages[s].metrics.R79.overall_mae);
const improvement=100*(1-publicScores[3]/publicScores[1]);
if(Math.abs(publicScores[3]-1998568.5706)>1e-6) throw Error('Score identity');
if(455787-3-529+11994-28!==467221) throw Error('Data arithmetic');

function text(s,txt,x,y,w,h,size=28,opt={}){
 const z=s.shapes.add({name:opt.name||'text',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=txt;
 z.text.style={typeface:opt.mono?MONO:FONT,fontSize:size,color:opt.color||C.body,bold:opt.bold||false,
  alignment:opt.align||'left',verticalAlignment:opt.valign||'top',autoFit:'none',wrap:'square',
  insets:{left:0,right:0,top:0,bottom:0},...opt.style};
 return z;
}
function slide(title,n,{dark=false}={}){
 const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;
 text(s,title,72,48,1136,72,44,{bold:true,color:dark?C.white:C.ink,name:'slide-title'});
 text(s,n>13?'附錄／v22':'臺灣房屋總價預測／v22',72,676,650,24,17,{color:dark?C.pale:C.muted});
 text(s,String(n).padStart(2,'0'),1150,673,58,28,19,{align:'right',color:dark?C.pale:C.muted});
 return s;
}
function note(s,title,sec,script,sources){
 const t=`建議時間：${sec?sec+' 秒':'問答時使用'}\n\n${script}\n\n資料來源（不必口述）\n${sources.join('\n')}\n\n本簡報整理 v22 已完成的方法與結果。Private 成績未知，最終候選尚未定案。`;
 s.speakerNotes.textFrame.setText(t);ledger.push({title,seconds:sec,script,sources});
}
function table(s,values,x,y,w,h,widths,{font=25,highlightRows=[]}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:C.bg,width:2,style:'solid'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=h/values.length;
  for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.ink:highlightRows.includes(r)?C.pale:(r%2?C.white:'#EDEFE6');
   cell.text.style={typeface:FONT,fontSize:font,color:r===0?C.white:C.body,bold:r===0||highlightRows.includes(r),
    verticalAlignment:'middle',autoFit:'none',insets:{left:16,right:12,top:8,bottom:8}};
  }
 }
 return t;
}
const axis={textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:C.gray,width:1},majorGridlines:{fill:'#D8DED2',width:1}};
function chartFont(c){applyPresentationChartFont(c,{fontFamily:FONT});return c;}
const sourceMethod='REPORT.md，第 2 至 4 節';
const sourceReproduction='evidence/reproduction/integration_verified.json';
const sourceV24='evidence/reports/round83_delivery.md';
const sourceData='REPORT.md，第 2 節';
const sourceR79='evidence/reports/round79_report.md';
const sourceValidation='REPORT.md，第 5、7 節';
const sourcePublic='evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json';

// 01 Cover: one photograph and a minimal editable title.
{
 const s=p.slides.add();s.background.fill=C.bg;
 const photo=await fs.readFile(BUILD+'/assets/taipei_markus_winkler.jpg');
 s.images.add({blob:new Uint8Array(photo),contentType:'image/jpeg',alt:'台北住宅建築情境照片，並非競賽中的個別交易',fit:'cover',position:{left:764,top:56,width:444,height:608}});
 text(s,'臺灣房屋\n總價預測',72,164,650,190,74,{bold:true,color:C.ink,name:'cover-title'});
 text(s,'近期成交加權與多模型組合',76,384,650,56,31,{color:C.green});
 text(s,'v22 研究成果報告',76,486,620,42,27);
 text(s,'監督式回歸專題',76,544,620,35,23,{color:C.muted});
 text(s,'2026.09.28',76,634,250,28,21,{color:C.muted});
 note(s,'臺灣房屋總價預測',20,'今天分享臺灣房屋總價預測。這是一個監督式回歸問題，輸入房屋與交易屬性，輸出連續的總價數值。我會說明 v22 的資料處理、模型方法、實驗結果與推論重現。Private 成績仍未知。',[sourceMethod,'Cover photograph: Markus Winkler, Taipei, Taiwan. https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs','Photo used under the Unsplash License as contextual imagery only.']);
}
// 02 Metric and final scoring population.
{
 const s=slide('評分看每筆總價的平均絕對誤差',2);
 text(s,'MAE = 平均 | 預測總價 − 真實總價 |',72,164,735,70,32,{bold:true,color:C.green});
 text(s,'誤差越低，預測越接近真價',72,248,705,48,28);
 text(s,'示意交易',72,354,660,36,23,{color:C.muted});
 text(s,'真價 1,000 萬　預測 1,100 萬',72,410,675,58,31);
 text(s,'這筆絕對誤差為 100 萬元',72,486,675,50,30,{bold:true});
 const c=s.charts.add('doughnut',{position:{left:830,top:154,width:368,height:376},categories:['Public 約 30%','Private 約 70%'],
  series:[{name:'測試資料占比',values:[30,70],points:[{idx:0,fill:C.lime},{idx:1,fill:'#86B09A'}]}],doughnutOptions:{holeSize:64},
  hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:22,fill:C.body}},
  dataLabels:{showPercent:true,position:'center',textStyle:{typeface:FONT,fontSize:26,fill:C.ink,bold:true}},chartFill:'none',chartLine:{fill:'none',width:0}});chartFont(c);
 text(s,'最終成績以 Private 為準',792,560,430,46,27,{bold:true,color:C.green});
 text(s,'Public 與 Private 都切自 test',72,616,800,34,23,{color:C.muted});
 note(s,'評分與 Public／Private',40,'先說明分數的意義。例如真價一千萬元，預測一千一百萬元，絕對誤差就是一百萬元。把所有評分交易的誤差平均，就是 MAE。公開分數約使用 test 的三成，最終看的 Private 則是另外七成。這與課程另外提供的 valid 不同。分數是總價誤差，不是準確率，也不表示每筆都錯同樣的金額。',[sourceMethod,sourcePublic,'User-provided competition evaluation and leaderboard screenshots in this conversation.']);
}
// 03 Input cleaning.
{
 const s=slide('清理後使用 467,221 筆最終訓練資料',3);
 table(s,[['處理步驟','保留筆數'],['原始 train','455,787'],['移除 3 列英文表頭','455,784'],['移除 529 列完全重複','455,255'],['加入 11,994 筆 valid','467,249'],['移除 28 列跨集重複','467,221']],72,173,680,390,[472,208],{font:25,highlightRows:[5]});
 text(s,'467,221',826,191,380,102,64,{bold:true,color:C.green});
 text(s,'最終訓練筆數',832,299,370,40,27);
 text(s,'47,073',828,412,380,86,58,{bold:true,color:C.ink});
 text(s,'test 全數保留',834,507,370,40,27);
 text(s,'按 sample 的編號順序輸出，編號只用於對齊',72,604,1136,39,24,{color:C.muted});
 note(s,'資料清理與列數',45,'原 train 有四十五萬五千七百八十七列，先去掉混入的三列英文表頭，再去掉五百二十九列完全重複交易。最終加入 valid，移除二十八筆跨集重複，得到四十六萬七千二百二十一筆。重複判定包含價格但排除編號，不是按 ID 刪除。這張表是最終 fit 的資料，年度回測另按時間切分且不混入 valid。test 四萬七千零七十三筆全數保留，依 sample 順序輸出。',[sourceData,'research/src/model_common.py:25','research/src/round79_final.py:83']);
}
// 04 Features.
{
 const s=slide('房屋描述轉成可學習的特徵',4);
 table(s,[['特徵內容','代表欄位'],['房屋屬性','面積、樓層、格局、主要用途'],['交易資訊','交易日期、車位、交易筆棟數'],['地理資訊','地址、道路位置、空間網格']],72,199,812,310,[205,607],{font:27});
 text(s,'95',961,205,232,114,84,{bold:true,color:C.green});
 text(s,'單價 CatBoost\n使用的特徵數',930,335,280,90,27);
 text(s,'其中 23 欄類別',930,464,290,43,26,{bold:true});
 text(s,'價格標籤與成交參考來自課程資料，外部地圖僅提供地理資訊',72,569,1136,74,25,{color:C.muted});
 note(s,'特徵設計',35,'共同基礎有八十七個特徵，包括民國日期、中文樓層、地址與面積比例等轉換。單價 CatBoost 再加八個道路地理欄位，共九十五個，其中二十三個類別欄。這個維度只適用這支樹模型，神經網路另有自己的編碼。編號、目標價格、完整備註與解約情形不作輸入。道路位置也不等同精確門牌。外部地圖沒有成交價格，我們沒有去內政部查 test 的答案。',[sourceMethod,'research/src/house_features.py:99','research/src/round6_geo.py:9']);
}
// 05 Architecture at the branch level.
{
 const s=slide('三條價格分支逐筆取中位數',5);
 table(s,[['價格分支','分支內的主要模型','預測方式'],['單價分支','CatBoost + LightGBM','單價乘回面積'],['對數分支','總價與 log CatBoost + LightGBM','以不同價格尺度學習'],['神經網路分支','兩個表格 NN + LightGBM','面積連結與參考價倍率']],72,166,1136,328,[213,534,389],{font:26});
 text(s,'各分支先換回「元」，再做成交參考與空間修正',72,523,1136,45,28);
 text(s,'最終總價 = median(NN 分支, log 分支, 單價分支)',72,590,1136,48,32,{bold:true,color:C.green});
 note(s,'模型組合',50,'v22 有三條價格分支，分支內還包含多個模型。單價分支學每單位面積價格，再乘回面積。對數分支用另一種尺度表示總價，神經網路提供不同的非線性預測。每條分支先還原成元，再以訓練資料中的同建案、地址、道路等可比成交與時間趨勢修正，必要時補上空間修正。最後逐筆取三個價格的中位數。這可以降低某一條分支極端輸出的影響，但三條一起錯時仍會錯。完整係數放在附錄。',[sourceMethod,'research/src/round79_final.py:153','research/src/round71_final.py:115','research/src/round46_trial_release.py:243']);
}
// 06 Recency is a native chart of a declared mathematical rule.
{
 const s=slide('近期成交的訓練權重較高',6);
 text(s,'相對日期權重（正規化前）',72,157,760,42,25,{color:C.muted});
 const c=s.charts.add('line',{position:{left:72,top:215,width:752,height:357},categories:['近期','早 1 年','早 2 年','早 3 年'],
  series:[{name:'相對權重',values:[1,.5,.25,.125],line:{fill:C.green,width:4},marker:{symbol:'circle',size:9},valuesFormatCode:'0.###'}],hasLegend:false,
  xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:1,majorUnit:.25,numberFormatCode:'0.##'},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:23,fill:C.green,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}});chartFont(c);
 text(s,'365 天',902,220,302,92,61,{bold:true,color:C.green});
 text(s,'日期權重半衰期',902,329,300,42,27);
 text(s,'面積權重 × 日期權重',877,429,342,42,27,{bold:true});
 text(s,'單價學習仍對齊\n近期成交的總價誤差',877,493,345,85,26);
 text(s,'實際日期權重會正規化，缺失或異常日期採中性權重 1',72,610,1136,39,24,{color:C.muted});
 note(s,'近期成交加權',50,'v22 相對 v20 的主要改動，是在單價 CatBoost 原有面積權重上，乘上日期權重。日期採三百六十五天半衰期，所以相差一年的有效成交，較早一筆的日期權重約是一半。圖中的一、零點五、零點二五是正規化前的相對值，實際資料還會正規化再乘面積權重。這不是只留下最近一年，而是仍保留可用舊資料。面積權重讓單價損失對齊總價 MAE，日期權重則讓近期成交更有影響力。',[sourceMethod,'research/src/round79_recency_unit.py:78','research/src/round60_recency_price_prior.py:49']);
}
// 07 Time splits.
{
 const s=slide('每一期用過去資料預測下一年',7);
 table(s,[['fit 截止日','評分日期','fit 筆數','評分筆數'],
  ['2021/08/10','2021/08/11\n至 2022/08/10','134,805','105,511'],
  ['2022/08/10','2022/08/11\n至 2023/08/10','240,316','81,234'],
  ['2023/08/10','2023/08/11\n至 2024/08/10','321,550','133,697']],72,174,1136,355,[250,416,235,235],{font:27});
 text(s,'模型、編碼與價格參考只使用該期 fit 資料',72,561,1136,44,28,{bold:true,color:C.green});
 text(s,'這些歷史窗口已反覆使用，不能視為全新盲測',72,618,1136,35,24,{color:C.muted});
 note(s,'時間驗證設計',45,'我們用三個完整年度做時間回測，每次只用評分起點以前的 train，預測接下來一年。模型、編碼與價格參考都必須依該期 fit 建立，不能把看過評分價格的全量模型拿回來測。這比把年份隨機混在一起，更接近用過去推估未來。不過這些年度已反覆用於研究，晚期訓練也包含較早的評分年度，因此不是三個獨立試驗，也不是新盲測。',[sourceValidation,'evidence/outputs/round79/plan.json']);
}
// 08 Offline evidence.
{
 const s=slide('相對 v20，v22 三期整體 MAE 都較低',8);
 text(s,'總價 MAE（萬元，越低越好）',72,146,1136,37,25,{color:C.muted});
 const c=s.charts.add('bar',{position:{left:72,top:196,width:1136,height:373},categories:['2021–2022','2022–2023','2023–2024'],
  series:[{name:'v20',values:oldMAE.map(x=>Number((x/1e4).toFixed(2))),fill:C.gray,valuesFormatCode:'0.00'},
          {name:'v22',values:newMAE.map(x=>Number((x/1e4).toFixed(2))),fill:C.green,valuesFormatCode:'0.00'}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:true,
  legend:{position:'top',textStyle:{typeface:FONT,fontSize:24,fill:C.body}},xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:260,majorUnit:50,numberFormatCode:'0'},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:22,fill:C.body}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}});chartFont(c);
 text(s,'改善 2.65%',208,587,220,41,26,{bold:true,color:C.green,align:'center'});
 text(s,'改善 0.62%',556,587,220,41,26,{bold:true,color:C.green,align:'center'});
 text(s,'改善 1.94%',904,587,220,41,26,{bold:true,color:C.green,align:'center'});
 text(s,'中間年度的配對區間跨 0，不能稱為三期皆有顯著改善',72,632,1136,31,23,{color:C.muted});
 note(s,'歷史回測結果',45,'這張圖按時間順序比較 v20 與 v22，單位是萬元。v22 三期的整體 MAE 都較低，改善幅度依序約二點六五、零點六二、一點九四百分比。最小的改善出現在中間年度，配對抽樣區間也跨過零，所以不能說三期都證明有穩定優勢。這些結果提供跨時間的比較依據，但窗口重用造成的選型偏差仍然存在。',[sourceR79,'evidence/outputs/round79/results.json','evidence/outputs/round79/paired_vs_v20/results.json']);
}
// 09 Public evidence, kept separate from offline data.
{
 const s=slide('v22 的 Public MAE 為 199.86 萬元',9);
 text(s,'Public MAE（萬元，越低越好）',72,151,834,39,25,{color:C.muted});
 const c=s.charts.add('bar',{position:{left:62,top:217,width:826,height:371},categories:publicLabels,
  series:[{name:'Public MAE',values:publicScores.map(x=>Number((x/1e4).toFixed(2))),fill:C.gray,points:publicLabels.map((_,i)=>({idx:i,fill:i===3?C.green:C.gray})),valuesFormatCode:'0.00'}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:false,
  xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:260,majorUnit:50,numberFormatCode:'0'},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:22,fill:C.body}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}});chartFont(c);
 text(s,improvement.toFixed(2)+'%',934,236,290,100,64,{bold:true,color:C.green});
 text(s,'相對 v20\n平均誤差降低',935,352,267,86,29);
 text(s,'平均誤差少\n約 7.13 萬元',930,470,291,100,27,{bold:true});
 text(s,'來源：使用者提供的 Kaggle 提交截圖，Private 尚未公布',72,621,1136,35,24,{color:C.muted});
 note(s,'Kaggle Public 成績',45,'這是使用者實際提交後提供的六個 Public 分數，圖上以萬元顯示。v22 的完整 MAE 是一百九十九萬八千五百六十八點五七零六元，在這六版中最低。相對 v20，平均每筆誤差少約七萬一千兩百六十九元，降低三點四四百分比。這是 MAE 降低，不是準確率增加。較新的 v23 和 v24 在這次 Public 沒有超過 v22，但最終七成 test 的 Private 仍未知，不能把公開結果當成 Private 保證。',[sourcePublic,'evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.png']);
}
// 10 Candidate counterevidence is shown alongside Public.
{
 const s=slide('v24 的三期本機整體 MAE 更低',10);
 text(s,'總價 MAE（萬元，越低越好）',72,148,1136,40,25,{color:C.muted});
 table(s,[['評分年度','v22','v24'],['2021–2022','230.94','227.54'],['2022–2023','140.74','140.73'],['2023–2024','138.32','137.32']],72,210,1136,280,[540,298,298],{font:27});
 text(s,'v24 仍有代價',72,520,430,45,29,{bold:true,color:C.rust});
 text(s,'最新年度的億元以上群共 319 筆\n群內 MAE 相對 v22 增加約 90.97 萬元',500,512,700,92,27);
 text(s,'本機優勢與 Public 排序不同，Private 仍需保留不確定性',72,623,1136,35,24,{color:C.muted});
 note(s,'v24 的有利與不利證據',45,'不能只挑支持 v22 的結果。v24 在三個年度的整體 MAE 都比 v22 更低，但 Public 排序相反。v24 單價分支採兩個 CatBoost seed 各半，已移除這條分支原有的 LightGBM 比例，其他兩條分支保持。它在最新年度真價至少一億的三百一十九筆資料中，群內 MAE 又比 v22 增加約九十點九七萬元。這是局部群組的差值，不是全體變差。我們因此保留 v24 作互補候選，不能宣稱 v22 每個評估都最好。',[sourceV24,'REPORT.md，第 5、6 節','research/src/round83_final.py']);
}

// 10 Counterevidence and model-selection uncertainty.
{
 const s=slide('平均改善仍伴隨高價物件風險',11);
 text(s,'實際退步的群組',72,171,720,44,30,{bold:true,color:C.rust});
 text(s,'最新回測年度\n總價 5,000 萬至 1 億',72,247,700,111,35,{bold:true});
 text(s,'此價格群共 1,725 筆交易',72,394,670,48,28);
 text(s,'價格群用於事後分析，推論時不能按真價分流',72,467,665,82,25,{color:C.muted});
 text(s,'+20.47 萬',812,239,408,109,60,{bold:true,color:C.rust});
 text(s,'此群 MAE 相對 v20 增加',810,374,395,80,28);
 text(s,'反覆使用驗證資料會累積選型偏差，分群與多期檢查仍不能保證 Private 表現',72,587,1136,72,26,{bold:true,color:C.green});
 note(s,'不利結果與過擬合風險',40,'平均分數會掩蓋不同族群的差異。例如最新回測年度，真價五千萬到一億這一群有一千七百二十五筆，v22 的群內平均誤差仍比 v20 多約二十點四七萬元。這不是全體 MAE 增加二十萬，也不能在預測時按未知真價決定用哪個模型。另一個風險是反覆看同一批驗證資料，方法會逐漸適應這些窗口。因此我們保留不利結果，不把 bootstrap 解讀成 Private 勝率。',[sourceR79,sourceValidation]);
}
// 12 Saved-model inference has now been reproduced on all test rows.
{
 const s=slide('原始 test 重新推論可重現 v22',12);
 text(s,'47,073 筆',72,177,700,102,67,{bold:true,color:C.green});
 text(s,'六模型重新預測，與保存結果完全一致',75,294,1110,52,30,{bold:true});
 text(s,'原 test Excel ＋ 保存地理、模型與 fit 狀態',72,391,1136,45,29);
 text(s,'未取整預測最大差 0 元，CSV 的 SHA256 相同',72,454,1136,45,29);
 text(s,'單一入口 84.74 秒',72,527,725,53,33,{bold:true,color:C.green});
 text(s,'同機既有環境／RTX 4060 Ti',763,537,452,38,24,{color:C.muted});
 text(s,'全新安裝、從零重訓與地理重建尚未驗證',72,614,1136,40,26,{color:C.rust});
 note(s,'保存模型推論重現',50,'這次新增的是可執行的推論入口。我們從原 test Excel 重新產生特徵，搭配保存的地理資料、編碼與價格先驗，真正載入四個樹模型和兩個神經網路，再重算參考與空間修正。四萬七千零七十三筆的未取整結果完全一致，最大差為零，最終 CSV 的完整 SHA256 也相同。單一入口耗時八十四點七四秒，這是同一台電腦既有環境的實測，不是重訓時間。全新安裝、從零重訓與地理資料重建還沒有驗證。',[sourceReproduction,'docs/REPRODUCIBILITY.md','src/predict_v22.py']);
}
// 13 Main-talk conclusion.
{
 const s=slide('研究結論與程式交付',13,{dark:true});
 text(s,'v22 Public MAE 約 199.86 萬元',72,172,1120,57,38,{bold:true,color:C.lime});
 text(s,'相對 v20 降低 3.44%，三期回測也有整體改善',72,254,1136,51,31,{color:C.white});
 text(s,'v24 本機優勢與高價群退步，是保留候選的理由',72,346,1136,66,29,{color:C.white});
 text(s,'程式入口  src/predict_v22.py',72,449,1136,43,30,{color:C.pale});
 text(s,'github.com/ziyi930108/nycu-house-price-regression',72,510,1136,39,24,{color:C.pale});
 text(s,'Private 未知，最終提交版本仍未定',72,605,1110,44,31,{bold:true,color:C.lime});
 note(s,'結論與程式交付',35,'v22 把近期成交權重加入單價學習，再與對數與神經網路分支組合，Public MAE 約一百九十九點八六萬元。程式與正式報告已整理成獨立交付目錄，推論入口已完成全量核對。但 v24 的本機優勢與高價群的退步提醒我們，較低的 Public 不能保證 Private 較好。後續研究方向是固定候選配方做時間切分穩定性檢查。畫面是私人 GitHub 儲存庫，閱覽需要授權。',['REPORT.md','README.md','https://github.com/ziyi930108/nycu-house-price-regression']);
}

// 12 Appendix: exact recipe, all variables are predictions in TWD.
{
 const s=slide('附錄 A　完整組合公式',14);
 text(s,'所有模型輸出先還原成總價「元」',72,145,1110,43,28,{bold:true,color:C.green});
 const code=[
  'B_unit = .75 * C_unit + .25 * L_unit',
  'B_nn   = .75 * (.5 * N_area + .5 * N_prior) + .25 * L_unit',
  'B_log  = .75 * (.5 * C_raw  + .5 * C_log)   + .25 * L_unit',
  '',
  'U = S(B_unit, R(B_unit))',
  'N = S(B_nn,   R(B_nn))',
  'H = S(B_log,  R(B_log))',
  '',
  'prediction = median(N, H, U)',
 ].join('\n');
 text(s,code,74,222,1132,314,27,{mono:true,color:C.ink});
 text(s,'R：成交參考與時間趨勢　　S：固定空間修正',72,570,1136,43,27);
 text(s,'省略共同特徵與保存狀態引數，修正發生在每條分支內',72,622,1136,35,23,{color:C.muted});
 note(s,'完整模型公式',0,'這頁供模型細節問答。C_unit 是 R79 單價 CatBoost，L_unit 是既有單價 LightGBM，C_raw 是直接總價 CatBoost，C_log 是近期加權對數 CatBoost。N_area 是面積連結 NN，N_prior 是近期加權先驗倍率 NN。所有變數先還原成元。R 為 reference trend，S 同時接收原基準與 reference 結果。最外層取逐筆中位數，沒有在 median 後再乘面積或再次做趨勢。',[sourceMethod,'research/src/round79_final.py:153','research/src/round71_final.py:115','research/src/round46_trial_release.py:243']);
}
// 13 Appendix: interpretable code and the main training configuration.
{
 const s=slide('附錄 B　核心程式與單價模型設定',15);
 text(s,'訓練核心（省略來源與保存檢查）',72,152,680,43,27,{bold:true,color:C.green});
 text(s,['A = effective_area(X)','target = y / (A * 1e4)','weight = A / 100 * date_w','','model.fit(Pool(','    X_geo, target,','    weight=weight,','    cat_features=cat_cols', '))'].join('\n'),72,223,630,320,28,{mono:true,color:C.ink});
 table(s,[['設定','值'],['iterations','2,999'],['depth','8'],['learning_rate','0.065'],['loss_function','MAE'],['random_seed','42']],746,180,462,354,[282,180],{font:25});
 text(s,'round79_recency_unit.py   /   round79_final.py',72,578,1136,42,25,{mono:true,color:C.green});
 text(s,'推論入口：src/predict_v22.py，完整從零重訓尚未驗證',72,628,1136,31,23,{color:C.muted});
 note(s,'程式與參數',0,'R79 是實驗編號，v22 是提交檔版本名。這段只保留核心概念，完整來源還有日期權重來源、列序、特徵、保存與重載檢查。主要 CatBoost 使用 GPU、兩千九百九十九棵樹、深度八、學習率零點零六五與 seed 四十二，沒有 query eval_set 或 early stopping。資料特徵在 house_features.py，全量 fit 在 round79_final_unit.py，組合在 round79_final.py，CSV 檢查在 round79_release.py。交付已新增 src/predict_v22.py，使用原 test 加十七項保存 artifact 真正重跑六模型。這是保存模型推論，不是從零重訓。',[sourceMethod,'research/src/round79_recency_unit.py:78','evidence/outputs/round79/final_model/model/metadata.json']);
}
// 14 Appendix: likely classroom questions.
{
 const s=slide('附錄 C　問答準備',16);
 text(s,'為什麼不只用 Public 最低分決定？',72,161,1136,45,31,{bold:true,color:C.green});
 text(s,'Public 約占 test 三成，其餘七成與未來市場差異仍未知',72,221,1136,64,27);
 text(s,'如何降低過擬合與資料洩漏風險？',72,321,1136,45,31,{bold:true,color:C.green});
 text(s,'使用時間回測與 fit-only 狀態，保留分群退步與窗口重用限制',72,381,1136,67,27);
 text(s,'程式可以完整重現嗎？',72,487,1136,45,31,{bold:true,color:C.green});
 text(s,'原 test 加保存狀態的推論已重現，從零重訓尚未驗證',72,550,1136,66,27);
 note(s,'課堂問答',0,'若被問到外部資料：價格標籤只來自課程 train 與 valid，外部 OSM 道路資訊只提供地理特徵，沒有查成交答案。若問 OOF：價格先驗採五折建案分組，避免本建案標籤直接進入其先驗，這不是時間 OOF，不能聲稱每筆先驗都只用當日以前的交易。若問 median：它降低單支極端輸出的影響，但不能保證沒有偏誤。若問正式交付：原 test 加保存狀態的推論已核對全部四萬七千零七十三筆，程式與報告已整理。模型與課程交易資料留在本機，GitHub 不包含它們。新環境安裝、其他硬體與從零重訓仍未測。CPU 神經網路可能因數值運算產生不同結果，不能保證位元相同。',[sourceMethod,sourceValidation,'docs/REPRODUCIBILITY.md']);
}

await fs.mkdir(BUILD+'/renders',{recursive:true});
await fs.mkdir(OUT,{recursive:true});
await fs.writeFile(BUILD+'/source_data.json',JSON.stringify({publicLabels,publicScores,oldMAE,newMAE,improvement,slides:ledger},null,2));
await fs.writeFile(BUILD+'/presentation.json',JSON.stringify(p.toProto()));
await (await PresentationFile.exportPptx(p)).save(BUILD+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 const img=await p.export({slide:s,format:'png',scale:1.25});
 await fs.writeFile(BUILD+`/renders/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await img.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(BUILD+`/renders/slide-${String(i+1).padStart(2,'0')}.json`,await layout.text());
 console.log('RENDERED',i+1);
}
const notes=['# 臺灣房屋總價預測：v22 正式報告講稿','',
 'v22 研究成果。主講 13 頁，另有 3 頁問答附錄，約 9 分 5 秒。Private 未知，最終候選未定。\n\n程式與報告：https://github.com/ziyi930108/nycu-house-price-regression（私人儲存庫，需閱覽權限）。','',
 ...ledger.flatMap((a,i)=>[`## ${String(i+1).padStart(2,'0')}　${a.title}`,'',a.seconds?`建議 ${a.seconds} 秒。`:'問答時使用，不列入主講時間。','',a.script,'','資料來源：',...a.sources.map(x=>'- '+x),''])].join('\n');
await fs.writeFile(OUT+'/講稿與問答.md',notes,'utf8');
console.log(JSON.stringify({slides:p.slides.items.length,mainTalkSeconds:ledger.reduce((a,x)=>a+x.seconds,0),candidate:BUILD+'/candidate.pptx'}));
