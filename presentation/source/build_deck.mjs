import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const ROOT='C:/NYCU_ML';
const DIR=ROOT+'/outputs/presentation_v22_focused_20260928';
const BUILD=DIR+'/build',OUT=DIR+'/deliverables';
const SKILL='C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const {resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const FONT=resolvePresentationFont({fontFamily:'Microsoft JhengHei',availableFonts:['Microsoft JhengHei']});
const MONO='Consolas';
const C={bg:'#F5F3EC',ink:'#203D35',body:'#34483F',muted:'#69776D',green:'#31644F',lime:'#C8DC9E',pale:'#DEE8D5',gray:'#ABB7AC',rust:'#AF553C',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ledger=[];
const saved=JSON.parse(await fs.readFile(ROOT+'/outputs/round79/results.json','utf8'));
const mae=['auxiliary','development','confirmation'].map(s=>saved.stages[s].metrics.R79.overall_mae);
const feedback=JSON.parse(await fs.readFile(ROOT+'/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json','utf8'));
const publicScore=feedback.screenshotted_scores['submission_v22_trial.csv'];
if(publicScore!==1998568.5706 || 455787-3-529+11994-28!==467221)throw Error('Source identity');
const sourceMethod='REPORT.md，第 2 至 4 節';
const sourceValidation='REPORT.md，第 5、7 節';
const sourcePublic='evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json';
const sourceRepro='evidence/reproduction/integration_verified.json';
function text(s,txt,x,y,w,h,size=28,opt={}){
 const z=s.shapes.add({name:opt.name||'text',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=txt;z.text.style={typeface:opt.mono?MONO:FONT,fontSize:size,color:opt.color||C.body,bold:opt.bold||false,alignment:opt.align||'left',verticalAlignment:opt.valign||'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0},...opt.style};return z;
}
function slide(title,n,{dark=false}={}){
 const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;
 text(s,title,72,48,1136,72,44,{bold:true,color:dark?C.white:C.ink,name:'slide-title'});
 text(s,n>13?'技術附錄':'臺灣房屋總價預測',72,676,650,24,17,{color:dark?C.pale:C.muted});
 text(s,String(n).padStart(2,'0'),1150,673,58,28,19,{align:'right',color:dark?C.pale:C.muted});return s;
}
function note(s,title,sec,script,sources){
 s.speakerNotes.textFrame.setText(`建議時間：${sec?sec+' 秒':'問答時使用'}\n\n${script}\n\n資料來源（不必口述）\n${sources.join('\n')}`);ledger.push({title,seconds:sec,script,sources});
}
function table(s,values,x,y,w,h,widths,{font=25,highlightRows=[]}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.bg,width:2,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[0].length;c++){const z=t.getCell(r,c);z.fill=r===0?C.ink:highlightRows.includes(r)?C.pale:(r%2?C.white:'#EDEFE6');z.text.style={typeface:FONT,fontSize:font,color:r===0?C.white:C.body,bold:r===0||highlightRows.includes(r),verticalAlignment:'middle',autoFit:'none',insets:{left:16,right:12,top:8,bottom:8}};}}return t;
}
const axis={textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:C.gray,width:1},majorGridlines:{fill:'#D8DED2',width:1}};
function chartFont(c){applyPresentationChartFont(c,{fontFamily:FONT});return c;}
// 1 Single chosen submission, no experiment-history narrative.
{
 const s=p.slides.add();s.background.fill=C.bg;
 const photo=await fs.readFile(BUILD+'/assets/taipei_markus_winkler.jpg');
 s.images.add({blob:new Uint8Array(photo),contentType:'image/jpeg',alt:'台北住宅情境照片，非競賽中的個別交易',fit:'cover',position:{left:764,top:56,width:444,height:608}});
 text(s,'臺灣房屋\n總價預測',72,164,650,190,74,{bold:true,color:C.ink});
 text(s,'近期成交加權與多模型回歸',76,384,650,56,31,{color:C.green});
 text(s,'本次選定提交模型 v22',76,486,620,42,27);
 text(s,'機器學習專題報告',76,544,620,35,23,{color:C.muted});
 text(s,'2026.09.28',76,634,250,28,21,{color:C.muted});
 note(s,'臺灣房屋總價預測',20,'本專題預測每一筆房屋交易的總價。我們選定一套模型作為最終提交，接下來依序說明資料處理、預測方法、評估結果，以及程式如何重新產生提交檔。目標是預測連續的價格數字，因此這是監督式回歸問題。',[sourceMethod,'Cover photograph: Markus Winkler, Taipei, Taiwan. https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs','Photo used under the Unsplash License as contextual imagery only.']);
}
// 2 Metric and scoring population.
{
 const s=slide('評分看總價的平均絕對誤差',2);
 text(s,'MAE = 平均 | 預測總價 − 真實總價 |',72,164,735,70,32,{bold:true,color:C.green});
 text(s,'單位為新臺幣元，越低越好',72,248,705,48,28);
 text(s,'假設例子',72,354,660,36,23,{color:C.muted});
 text(s,'真價 1,000 萬　預測 1,100 萬',72,410,675,58,31);
 text(s,'這筆絕對誤差為 100 萬元',72,486,675,50,30,{bold:true});
 chartFont(s.charts.add('doughnut',{position:{left:830,top:154,width:368,height:376},categories:['Public 約 30%','Private 約 70%'],series:[{name:'測試資料占比',values:[30,70],points:[{idx:0,fill:C.lime},{idx:1,fill:'#86B09A'}]}],doughnutOptions:{holeSize:64},hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:22,fill:C.body}},dataLabels:{showPercent:true,position:'center',textStyle:{typeface:FONT,fontSize:26,fill:C.ink,bold:true}},chartFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'最終成績以 Private 為準',792,560,430,46,27,{bold:true,color:C.green});
 text(s,'Public 與 Private 都切自 test',72,616,800,34,23,{color:C.muted});
 note(s,'總價回歸與 MAE',40,'例如一筆交易真價一千萬元，我們預測一千一百萬元，絕對誤差就是一百萬元。把所有評分交易的誤差平均，就是 MAE，越低越好。這是金額誤差，不是準確率。公開分數約使用 test 的三成，Private 使用其餘七成，與另外提供的 valid 檔案不同。',['REPORT.md，第 1 節',sourcePublic]);
}
// 3 Data cleaning.
{
 const s=slide('清理後使用 467,221 筆訓練資料',3);
 table(s,[['處理步驟','保留筆數'],['原始 train','455,787'],['移除 3 列英文表頭','455,784'],['移除 529 列完全重複','455,255'],['加入 11,994 筆 valid','467,249'],['移除 28 列跨集重複','467,221']],72,173,680,390,[472,208],{font:25,highlightRows:[5]});
 text(s,'467,221',826,191,380,102,64,{bold:true,color:C.green});text(s,'最終訓練筆數',832,299,370,40,27);
 text(s,'47,073',828,412,380,86,58,{bold:true,color:C.ink});text(s,'test 全數保留',834,507,370,40,27);
 text(s,'按 sample 的編號順序輸出，編號只用於對齊',72,604,1136,39,24,{color:C.muted});
 note(s,'資料清理',45,'原 train 有四十五萬五千七百八十七列，移除混入的三列英文表頭，再排除五百二十九列完全重複交易。最終加入 valid 並移除二十八列跨集重複，得到四十六萬七千二百二十一筆。重複判定包含價格、排除編號，不是按照相同 ID 刪除。年度回測則另按時間切分且不混 valid。test 四萬七千零七十三筆全部保留，依 sample 順序輸出。',[sourceMethod,'research/src/prepare_data.py','research/src/model_common.py']);
}
// 4 Feature construction.
{
 const s=slide('房屋描述轉成可學習的特徵',4);
 table(s,[['特徵內容','代表欄位'],['房屋屬性','面積、樓層、格局、主要用途'],['交易資訊','交易日期、車位、交易筆棟數'],['地理資訊','地址、道路位置、空間網格']],72,199,812,310,[205,607],{font:27});
 text(s,'95',961,205,232,114,84,{bold:true,color:C.green});text(s,'單價 CatBoost\n使用的特徵數',930,335,280,90,27);text(s,'其中 23 欄類別',930,464,290,43,26,{bold:true});
 text(s,'價格標籤來自課程資料，外部地圖僅提供地理資訊',72,569,1136,74,25,{color:C.muted});
 note(s,'特徵設計',40,'共同基礎有八十七個特徵，包括民國日期、中文樓層、地址與面積比例等。單價 CatBoost 再加八個道路地理欄位，共九十五欄，其中二十三欄為類別。其他模型有自己的編碼維度。編號、目標價格、完整備註與解約情形不作輸入。價格標籤只來自課程資料，OSM 道路地圖沒有提供成交價格，也沒有查詢內政部的 test 答案。',[sourceMethod,'research/src/house_features.py','evidence/data_external/geography/ATTRIBUTION.md']);
}
// 5 Six unique models, shared LightGBM appears in all bases.
{
 const s=slide('六個模型組成三條價格分支',5);
 text(s,'單價 × 面積 = 總價，另以不同價格尺度補充預測',72,145,1136,44,29,{bold:true,color:C.green});
 table(s,[['價格分支','主要模型','學習的價格表示'],['單價分支','單價 CatBoost + LightGBM','單價乘回面積'],['對數分支','總價及 log CatBoost + LightGBM','直接總價與對數總價'],['神經網路分支','兩個表格 NN + LightGBM','面積連結與先驗倍率']],72,213,1136,301,[213,545,378],{font:25});
 text(s,'各分支還原成「元」，再做成交參考與空間修正',72,552,1136,44,28);
 text(s,'3 個 CatBoost ＋ 1 個 LightGBM ＋ 2 個 NN，LightGBM 由三分支共用',72,619,1136,35,24,{color:C.muted});
 note(s,'六模型與三分支',55,'總價可以從單價乘上面積來理解，所以我們先建立單價分支，學習單位面積價格後乘回面積。對數分支結合直接總價與對數總價，神經網路則用面積連結和價格先驗倍率提供不同的非線性預測。共有三個 CatBoost、一個 LightGBM 和兩個 NN，其中同一個 LightGBM 在三條分支共用，所以模型總數是六個。所有模型先換回總價元，再利用 fit 中的建案、地址或道路等可比成交與時間趨勢修正，支持較弱時補充固定的空間修正。最後才組合三條分支。',[sourceMethod,'src/predict_v22.py']);
}
// 6 Recency weighting, not a claim that all other components lack date weighting.
{
 const s=slide('單價 CatBoost 提高近期成交權重',6);
 text(s,'相對日期權重（正規化前）',72,157,760,42,25,{color:C.muted});
 chartFont(s.charts.add('line',{position:{left:72,top:215,width:752,height:357},categories:['近期','早 1 年','早 2 年','早 3 年'],series:[{name:'相對權重',values:[1,.5,.25,.125],line:{fill:C.green,width:4},marker:{symbol:'circle',size:9},valuesFormatCode:'0.###'}],hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:1,majorUnit:.25,numberFormatCode:'0.##'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:23,fill:C.green,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'365 天',902,220,302,92,61,{bold:true,color:C.green});text(s,'日期權重半衰期',902,329,300,42,27);text(s,'面積權重 × 日期權重',877,429,342,42,27,{bold:true});text(s,'面積對齊總價誤差\n日期提高近期影響',877,493,345,85,26);
 text(s,'實際日期權重會正規化，缺失或異常日期採中性權重 1',72,610,1136,39,24,{color:C.muted});
 note(s,'近期成交權重',50,'房價會隨時間改變，單價 CatBoost 因此採三百六十五天半衰期的日期權重。同一 fit 內相差一年的有效交易，較早一筆日期權重約是一半。圖上是正規化前的比例，實際權重會正規化後再乘面積權重。面積權重把單價誤差對齊總價誤差，日期權重提高近期成交在訓練中的影響。舊資料仍保留。這張圖只在說明單價模型的設計，其他分支各有其訓練目標。',[sourceMethod,'research/src/round79_recency_unit.py']);
}
// 7 Clearly illustrative median demonstration.
{
 const s=slide('三條總價預測逐筆取中位數',7);
 text(s,'假設例子：各分支修正後的總價，單位為萬元',72,148,1136,43,25,{color:C.muted});
 chartFont(s.charts.add('bar',{position:{left:72,top:222,width:758,height:339},categories:['單價分支','對數分支','神經網路分支'],series:[{name:'假設預測',values:[1200,1320,1900],fill:C.gray,points:[{idx:0,fill:C.gray},{idx:1,fill:C.green},{idx:2,fill:C.gray}],valuesFormatCode:'#,##0'}],barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:2200,majorUnit:500,numberFormatCode:'#,##0'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:24,fill:C.body,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'1,320 萬',897,258,316,102,59,{bold:true,color:C.green});text(s,'這筆的最終預測',891,391,322,46,28);text(s,'取三個數字\n排序後的中間值',891,472,322,88,27);
 text(s,'中位數可降低單一分支極端輸出的影響，三條共同偏誤仍可能保留',72,615,1136,47,25,{color:C.muted});
 note(s,'總價中位數的假設例子',45,'這是方便理解的假設例子，不是真實交易。假設三條分支修正後分別預測一千兩百萬、一千三百二十萬和一千九百萬，排序後的中間值是一千三百二十萬，所以它成為這一筆的最終預測。下一筆交易的中間值可能來自另一條分支。這樣能降低單條極端輸出的影響，但若三條一起偏高或偏低，中位數也無法消除共同錯誤。',[sourceMethod,'Illustrative numbers only; not model predictions or evaluation evidence.']);
}
// 8 Time split.
{
 const s=slide('每一期用過去資料預測下一年',8);
 table(s,[['fit 截止日','評分日期','fit 筆數','評分筆數'],['2021/08/10','2021/08/11\n至 2022/08/10','134,805','105,511'],['2022/08/10','2022/08/11\n至 2023/08/10','240,316','81,234'],['2023/08/10','2023/08/11\n至 2024/08/10','321,550','133,697']],72,174,1136,355,[250,416,235,235],{font:27});
 text(s,'模型、編碼與價格參考只使用該期 fit 資料',72,561,1136,44,28,{bold:true,color:C.green});text(s,'歷史時間窗已多次用於研究，不能視為全新盲測',72,618,1136,35,24,{color:C.muted});
 note(s,'時間驗證設計',45,'我們用三個完整年度做時間回測。每次只用評分起點以前的 train，再預測接下來一年。模型、編碼、價格先驗與成交參考都必須按同一期 fit 建立，避免未來的 query 價格洩漏。回測不加入另外提供的 valid。這些歷史時間窗曾多次參與研究，較晚的訓練也包含較早的評分年度，因此它們不是三次獨立盲測。',[sourceValidation,'evidence/outputs/round79/plan.json']);
}
// 9 One selected model, distinct chronological scoring sets.
{
 const s=slide('三個年度的歷史回測結果',9);
 text(s,'總價 MAE（萬元，越低越好）',72,146,1136,37,25,{color:C.muted});
 chartFont(s.charts.add('bar',{position:{left:72,top:218,width:1136,height:373},categories:['2021–2022','2022–2023','2023–2024'],series:[{name:'提交模型的時間回測',values:mae.map(x=>Number((x/1e4).toFixed(2))),fill:C.green,valuesFormatCode:'0.00'}],barOptions:{direction:'column',grouping:'clustered',gapWidth:125},hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:260,majorUnit:50,numberFormatCode:'0'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:27,fill:C.body,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'每根柱代表不同年度的交易，年度差值不能解讀為改版進步',72,620,1136,42,25,{color:C.muted});
 note(s,'歷史回測 MAE',40,'固定同一套方法，在三個時間切分分別重建模型與狀態後，整體 MAE 依序約二百三十點九四萬、一百四十點七四萬與一百三十八點三二萬元。這三根柱子的評分交易不同，房屋組成與市場狀態也不同，所以不能把後兩期較低直接解釋成模型改版的進步。它們描述這套方法在不同年度的誤差範圍。',[sourceValidation,'evidence/outputs/round79/results.json']);
}
// 10 A single public result, kept separate from offline time windows.
{
 const s=slide('Kaggle Public 的平均誤差約 199.86 萬元',10);
 text(s,'199.86 萬元',72,194,1120,133,83,{bold:true,color:C.green});
 text(s,'Public MAE = 1,998,568.5706 元',76,360,1110,49,32);
 text(s,'約 30% test 交易的平均絕對總價誤差',72,452,1110,53,31);
 text(s,'Private 約占其餘 70%，目前成績未知',72,554,1136,57,32,{bold:true});
 text(s,'來源為 Kaggle 提交成績截圖，不與不同評分集的歷史 MAE 混算',72,632,1136,31,23,{color:C.muted});
 note(s,'Kaggle Public 成績',40,'這套模型提交到 Kaggle 後，Public MAE 是一百九十九萬八千五百六十八點五七零六元，約一百九十九點八六萬元。意思是在公開評分的交易中，平均絕對總價誤差約為這個金額，不代表每筆都錯一樣多。這是 test 的一部分，與前頁三個歷史年度不同，不能把它們的分數直接混算。Private 使用其餘約七成，最終分數還不知道。',[sourcePublic]);
}
// 11 Limits without internal candidate comparisons.
{
 const s=slide('高價尾部與驗證資料重用仍是限制',11);
 text(s,'高價與大面積交易',72,179,1136,45,31,{bold:true,color:C.rust});
 text(s,'少量大額誤差可能拉高總價 MAE，平均值會掩蓋群組差異',72,242,1136,82,28);
 text(s,'時間窗曾多次參與模型選擇',72,361,1136,45,31,{bold:true,color:C.green});
 text(s,'歷史回測可能樂觀，重抽樣區間也不能保證 Private 表現',72,424,1136,82,28);
 text(s,'成交參考與地理資訊仍有適用範圍',72,541,1136,45,31,{bold:true,color:C.green});
 text(s,'新建案、缺失地址與粗略道路位置，可能降低可比交易的支持',72,604,1136,59,27);
 note(s,'限制與過擬合風險',45,'這個方法仍有幾項限制。總價 MAE 容易受高價或大面積交易的巨大誤差影響，因此不能只看全體平均，還要檢查價格群和新舊建案。其次，歷史時間窗已多次參與選型，會累積對驗證資料的適應，bootstrap 也不能消除這個偏差。最後，道路位置不是精確門牌，新建案或缺失地址可能找不到可靠參考。這些限制不會因為我們已決定提交模型而消失。',[sourceValidation]);
}
// 12 Actual saved-model inference replay, not from-scratch training.
{
 const s=slide('原始 test 重新推論可重現提交檔',12);
 text(s,'47,073 筆',72,177,700,102,67,{bold:true,color:C.green});text(s,'六模型重新預測，與保存結果完全一致',75,294,1110,52,30,{bold:true});
 text(s,'原 test Excel ＋ 保存地理、模型與 fit 狀態',72,391,1136,45,29);text(s,'未取整預測最大差 0 元，CSV 的 SHA256 相同',72,454,1136,45,29);
 text(s,'單一入口 84.74 秒',72,527,725,53,33,{bold:true,color:C.green});text(s,'同機既有環境／RTX 4060 Ti',763,537,452,38,24,{color:C.muted});
 text(s,'全新安裝、從零重訓與地理重建尚未驗證',72,614,1136,40,26,{color:C.rust});
 note(s,'保存模型推論重現',50,'我們已把推論整理成單一程式入口。從原 test Excel 重新產生特徵，搭配保存的地理、編碼與價格先驗，真正載入四個樹模型及兩個神經網路，再重算三條分支。四萬七千零七十三筆未取整預測全部一致，最大差是零，最終 CSV 的完整 SHA256 也相同。這個入口在同一台電腦既有環境耗時八十四點七四秒，並非訓練時間。全新安裝、從零重訓與地理重建尚未驗證。',[sourceRepro,'docs/REPRODUCIBILITY.md','src/predict_v22.py']);
}
// 13 Conclusion for the chosen model.
{
 const s=slide('研究結論與程式交付',13,{dark:true});
 text(s,'以房屋屬性與歷史成交預測總價',72,181,1110,58,38,{bold:true,color:C.lime});
 text(s,'近期加權的單價學習，搭配三分支中位數組合',72,273,1136,63,31,{color:C.white});
 text(s,'Public MAE 約 199.86 萬元，Private 尚未知',72,371,1136,58,31,{color:C.white});
 text(s,'程式入口  src/predict_v22.py',72,473,1136,45,30,{color:C.pale});
 text(s,'github.com/ziyi930108/nycu-house-price-regression',72,538,1136,39,24,{color:C.pale});
 text(s,'程式、正式報告與逐頁講稿均已保存',72,612,1110,44,29,{bold:true,color:C.lime});
 note(s,'結論與交付',30,'本研究將房價預測視為總價回歸，結合特徵處理、近期成交加權和不同價格表示，再逐筆取三條分支的中位數。目前 Public MAE 約一百九十九點八六萬元，Private 尚未知。所選模型的推論程式、正式報告及提交檔身分都有保存。畫面是私人 GitHub 儲存庫，閱覽需要授權。謝謝。',['REPORT.md','README.md','https://github.com/ziyi930108/nycu-house-price-regression']);
}
// 14 Complete composition, all outputs in TWD.
{
 const s=slide('附錄 A　完整組合公式',14);
 text(s,'所有模型輸出先還原成總價「元」',72,145,1110,43,28,{bold:true,color:C.green});
 text(s,['B_unit = .75 * C_unit + .25 * L_unit','B_nn   = .75 * (.5 * N_area + .5 * N_prior) + .25 * L_unit','B_log  = .75 * (.5 * C_raw  + .5 * C_log)   + .25 * L_unit','','U = S(B_unit, R(B_unit))','N = S(B_nn,   R(B_nn))','H = S(B_log,  R(B_log))','','prediction = median(N, H, U)'].join('\n'),74,222,1132,314,27,{mono:true,color:C.ink});
 text(s,'R：成交參考與時間趨勢　　S：固定空間修正',72,570,1136,43,27);text(s,'省略共同特徵與保存狀態引數，修正發生在每條分支內',72,622,1136,35,23,{color:C.muted});
 note(s,'完整模型公式',0,'C_unit 是近期加權單價 CatBoost，L_unit 是單價 LightGBM 還原總價。C_raw 與 C_log 是直接總價和對數 CatBoost，N_area 與 N_prior 是面積連結和先驗倍率神經網路。所有符號已換回元。R 是 fit 中成交參考與時間趨勢，S 接收原基準及參考結果做空間修正。逐分支修正後才取中位數，最後不再乘面積或重複修正。',[sourceMethod,'src/predict_v22.py']);
}
// 15 Core training design, no internal round or comparison labels.
{
 const s=slide('附錄 B　單價模型核心程式與設定',15);
 text(s,'訓練核心（省略來源與保存檢查）',72,152,680,43,27,{bold:true,color:C.green});
 text(s,['A = effective_area(X)','target = y / (A * 1e4)','weight = A / 100 * date_w','','model.fit(Pool(','    X_geo, target,','    weight=weight,','    cat_features=cat_cols', '))'].join('\n'),72,223,630,320,28,{mono:true,color:C.ink});
 table(s,[['設定','值'],['iterations','2,999'],['depth','8'],['learning_rate','0.065'],['loss_function','MAE'],['random_seed','42']],746,180,462,354,[282,180],{font:25});
 text(s,'A：建物面積優先，其次車位面積，皆無效時取 1',72,584,1136,40,26,{color:C.green});text(s,'推論入口：src/predict_v22.py，從零重訓尚未驗證',72,632,1136,31,23,{color:C.muted});
 note(s,'單價模型設定',0,'有效面積先用正且有限的建物面積，再用車位面積，仍無效才取一。這是數值縮放規則，不是宣稱實際室內面積等於一。CatBoost 使用兩千九百九十九次迭代、深度八、學習率零點零六五、MAE 和 seed 四十二。它不使用 query 價格作訓練、eval_set 或 early stopping。模型推論入口已全量驗證，完整從零重訓仍是不同的工作範圍。',[sourceMethod,'research/src/round79_recency_unit.py','src/predict_v22.py']);
}
// Traditional Chinese copy normalization and version-narrative check.
for(const a of ledger){if(/v20|v24|R79/.test(a.title+' '+a.script))throw Error('Internal comparison label in spoken copy');}
await fs.mkdir(BUILD+'/renders',{recursive:true});await fs.mkdir(OUT,{recursive:true});
await fs.writeFile(BUILD+'/source_data.json',JSON.stringify({selectedSubmission:'submission_v22_trial.csv',publicMAE:publicScore,historicalMAE:mae,historicalPeriods:['2021-08-11 to 2022-08-10','2022-08-11 to 2023-08-10','2023-08-11 to 2024-08-10'],medianIllustration:{hypothetical:true,unit:'10,000 TWD',branchTotals:[1200,1320,1900],median:1320},slides:ledger},null,2));
await fs.writeFile(BUILD+'/presentation.json',JSON.stringify(p.toProto()));
await (await PresentationFile.exportPptx(p)).save(BUILD+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){const s=p.slides.items[i],stem=String(i+1).padStart(2,'0');const img=await p.export({slide:s,format:'png',scale:1.25});await fs.writeFile(BUILD+'/renders/slide-'+stem+'.png',new Uint8Array(await img.arrayBuffer()));const layout=await s.export({format:'layout'});await fs.writeFile(BUILD+'/renders/slide-'+stem+'.json',await layout.text());console.log('RENDERED',i+1);}
const total=ledger.reduce((a,x)=>a+x.seconds,0);
const extraFAQ=['## 延伸問答','','**這是回歸還是分類？** 預測的是連續數值的總價，因此是監督式回歸。','','**為什麼選擇 MAE？** 競賽以總價元的平均絕對誤差評分，報告與模型評估沿用同一單位。','','**有查外部成交答案嗎？** 沒有。價格標籤與成交參考來自課程資料，外部 OSM 道路資訊只提供非價格地理特徵。','','**價格先驗如何避免直接使用自身標籤？** 訓練先驗採五折建案分組 OOF，避免同建案的標籤直接成為自己的先驗。這不是時間 OOF，不能宣稱每筆先驗都只看當日以前。','','**完整重現的範圍是什麼？** 已測的是原 test 加保存地理、模型與 fit 狀態的六模型推論重現。全新安裝、從零重訓、其他硬體及重建 OSM 地理尚未驗證。','','**GitHub 裡有模型與交易資料嗎？** 儲存庫包含程式、正式報告和說明，課程交易資料、模型及價格參考狀態留在本機。','','**Private 一定會一樣好嗎？** 無法保證。最終選擇這套模型不代表已知 Private 真實誤差。'];
const notes=['# 臺灣房屋總價預測：上台講稿','','以單一最終提交模型說明方法與結果。主講 13 頁，另有 2 頁技術附錄。建議配速 '+Math.floor(total/60)+' 分 '+total%60+' 秒。','','程式與報告：https://github.com/ziyi930108/nycu-house-price-regression（私人儲存庫，需閱覽權限）。','',...ledger.flatMap((a,i)=>[`## ${String(i+1).padStart(2,'0')}　${a.title}`,'',a.seconds?`建議 ${a.seconds} 秒。`:'問答時使用，不列入主講時間。','',a.script,'','資料來源：',...a.sources.map(x=>'- '+x),'']),...extraFAQ].join('\n');
await fs.writeFile(OUT+'/講稿與問答.md',notes,'utf8');console.log(JSON.stringify({slides:p.slides.items.length,mainTalkSeconds:total,candidate:BUILD+'/candidate.pptx'}));
