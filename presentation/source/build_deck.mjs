import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT='C:/NYCU_ML';
const DIR=ROOT+'/outputs/presentation_v22_fullflow_20260928';
const BUILD=DIR+'/build',OUT=DIR+'/deliverables';
const SKILL='C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const {resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const FONT=resolvePresentationFont({fontFamily:'Microsoft JhengHei',availableFonts:['Microsoft JhengHei']});
const MONO='Consolas';
const C={bg:'#F5F3EC',ink:'#203D35',body:'#34483F',muted:'#69776D',green:'#31644F',lime:'#C8DC9E',pale:'#DEE8D5',gray:'#ABB7AC',rust:'#AF553C',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const narrative=JSON.parse(await fs.readFile(DIR+'/narrative.json','utf8'));
const ledger=narrative.slides;
if(ledger.length!==22 || ledger.filter(x=>!x.is_appendix).length!==20)throw Error('Narrative slide count');
const saved=JSON.parse(await fs.readFile(ROOT+'/outputs/round79/results.json','utf8'));
const mae=['auxiliary','development','confirmation'].map(s=>saved.stages[s].metrics.R79.overall_mae);
const feedback=JSON.parse(await fs.readFile(ROOT+'/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json','utf8'));
const publicScore=feedback.screenshotted_scores['submission_v22_trial.csv'];
if(publicScore!==1998568.5706 || 455787-3-529+11994-28!==467221)throw Error('Source identity');
function text(s,txt,x,y,w,h,size=28,opt={}){
 const z=s.shapes.add({name:opt.name||'text',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=txt;z.text.style={typeface:opt.mono?MONO:FONT,fontSize:size,color:opt.color||C.body,bold:opt.bold||false,alignment:opt.align||'left',verticalAlignment:opt.valign||'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0},...opt.style};return z;
}
function slide(n,{dark=false,title=null}={}){
 const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;
 text(s,title??ledger[n-1].title,72,48,1136,86,43,{bold:true,color:dark?C.white:C.ink,name:'slide-title'});
 text(s,n>20?'技術附錄':'臺灣房屋總價預測',72,676,850,24,17,{color:dark?C.pale:C.muted});
 text(s,String(n).padStart(2,'0'),1150,673,58,28,19,{align:'right',color:dark?C.pale:C.muted});
 attachNote(s,n);return s;
}
function attachNote(s,n){const a=ledger[n-1];s.speakerNotes.textFrame.setText(`建議配速：${a.is_appendix?'問答時使用':a.seconds+' 秒'}\n\n${a.script}\n\n資料來源（不必口述）\n${a.sources.join('\n')}`);}
function table(s,values,x,y,w,h,widths,{font=25,highlightRows=[]}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.bg,width:2,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[0].length;c++){const z=t.getCell(r,c);z.fill=r===0?C.ink:highlightRows.includes(r)?C.pale:(r%2?C.white:'#EDEFE6');z.text.style={typeface:FONT,fontSize:font,color:r===0?C.white:C.body,bold:r===0||highlightRows.includes(r),verticalAlignment:'middle',autoFit:'none',insets:{left:16,right:12,top:8,bottom:8}};}}return t;
}
const axis={textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:C.gray,width:1},majorGridlines:{fill:'#D8DED2',width:1}};
function chartFont(c){applyPresentationChartFont(c,{fontFamily:FONT});return c;}
function pair(s,y,heading,body,{color=C.green}={}){text(s,heading,72,y,390,55,31,{bold:true,color});text(s,body,502,y+1,706,104,27);}
// 01 Cover.
{
 const s=p.slides.add();s.background.fill=C.bg;
 const photo=await fs.readFile(BUILD+'/assets/taipei_markus_winkler.jpg');
 s.images.add({blob:new Uint8Array(photo),contentType:'image/jpeg',alt:'台北住宅情境照片，非競賽中的個別交易',fit:'cover',position:{left:764,top:56,width:444,height:608}});
 text(s,'臺灣房屋\n總價預測',72,164,650,190,74,{bold:true,color:C.ink});
 text(s,'資料處理、模型設計與驗證流程',76,384,650,66,30,{color:C.green});
 text(s,'本次選定提交模型 v22',76,486,620,42,27);
 text(s,'機器學習專題報告',76,544,620,35,23,{color:C.muted});
 text(s,'2026.09.28',76,634,250,28,21,{color:C.muted});attachNote(s,1);
}
// 02 Total-price regression and the loss metric.
{
 const s=slide(2);
 text(s,'輸入房屋與交易屬性，輸出連續的總價數字',72,158,1136,54,31,{bold:true,color:C.green});
 text(s,'MAE = 平均 | 預測總價 − 真實總價 |',72,263,1136,68,36,{bold:true});
 table(s,[['假設交易','真價','預測','絕對誤差'],['A','1,000 萬元','1,100 萬元','100 萬元'],['B','1,000 萬元','950 萬元','50 萬元']],72,370,1136,202,[230,302,302,302],{font:28});
 text(s,'平均絕對誤差 = 75 萬元，越低越好',72,610,1136,44,31,{bold:true,color:C.green});
}
// 03 End-to-end work, preserving temporal fit/query boundaries.
{
 const s=slide(3);
 table(s,[['步驟','做什麼','為什麼'],['檔案與資料角色','確認欄位、列數與標籤用途','避免把格式範例當答案'],['清理與特徵','處理重複，解析日期、樓層與地址','保留欄位意義，防止資料錯位'],['時間回測與學習','依當期 fit 建模型與價格狀態','檢查以過去預測未來'],['最終模型與提交','最終 fit、test 推論及核驗','保存可追溯的模型與提交']],72,170,1136,395,[251,412,473],{font:24});
 text(s,'回測的 query 價格不參與同一期模型與價格參考',72,606,1136,48,29,{bold:true,color:C.green});
}
// 04 Distinct data roles.
{
 const s=slide(4);
 table(s,[['資料','原始列數','是否有總價','使用角色'],['train','455,787','有','歷史回測與模型學習'],['valid','11,994','有','最終全量 fit 併入'],['test','47,073','無','全部保留，等待模型預測'],['sample','47,073','範例價格','提供欄名與編號順序']],72,173,1136,384,[218,182,245,491],{font:26});
 text(s,'Public 約 30% 與 Private 約 70% 都來自 test',72,602,1136,53,28,{bold:true,color:C.green});
}
// 05 Cleaning and exact row counts.
{
 const s=slide(5);
 table(s,[['處理步驟','保留筆數'],['原始 train','455,787'],['移除 3 列英文表頭','455,784'],['移除 529 列完全重複','455,255'],['加入 11,994 筆 valid','467,249'],['移除 28 列跨集重複','467,221']],72,173,680,390,[472,208],{font:25,highlightRows:[5]});
 text(s,'467,221',826,191,380,102,64,{bold:true,color:C.green});text(s,'最終 fit 的交易筆數',825,299,380,75,27);
 text(s,'去重包含價格\n但排除編號',825,429,378,96,28,{bold:true});
 text(s,'移除格式錯列與重複交易，test 不刪列、不去重',72,606,1136,48,28,{color:C.muted});
}
// 06 Parsing raw columns into meaningful predictors.
{
 const s=slide(6);
 table(s,[['原始欄位','解析後資訊','設計目的'],['民國交易日期','年、月、季及時間索引','讓模型辨識交易時點'],['中文樓層與總樓層','數值樓層與相關旗標','保留樓層位置及不規則情況'],['土地、建物與車位面積','有效數值與面積比例','描述交易規模與組成'],['地址、道路與建案','類別及粗略地理資訊','連結區位與可比成交']],72,181,1136,362,[303,365,468],{font:24});
 text(s,'共同基礎 87 欄，單價 CatBoost 加地理後 95 欄',72,592,1136,47,29,{bold:true,color:C.green});
 text(s,'該模型含 23 個類別欄，NN 另使用自己的編碼維度',72,638,1136,30,22,{color:C.muted});
}
// 07 Exclusions and leakage discipline.
{
 const s=slide(7);
 pair(s,175,'編號與總價','編號只用來對齊輸出\n總價只作學習標籤，不作輸入特徵');
 pair(s,329,'完整備註與解約情形','避免文字價格或交易後資訊進入模型\n僅保留有限的非價格備註旗標');
 pair(s,488,'不可用或不變的欄位','建築完成年月不可用，管理組織是常數\n不將缺乏有效訊息的欄位當作主要依據');
}
// 08 Chronological validation, with all learned states fit only.
{
 const s=slide(8);
 table(s,[['fit 截止日','評分期間','fit 筆數','評分筆數'],['2021/08/10','2021/08/11\n至 2022/08/10','134,805','105,511'],['2022/08/10','2022/08/11\n至 2023/08/10','240,316','81,234'],['2023/08/10','2023/08/11\n至 2024/08/10','321,550','133,697']],72,174,1136,355,[250,416,235,235],{font:27});
 text(s,'每一期的模型、編碼、先驗與參考只用該期 fit',72,561,1136,48,28,{bold:true,color:C.green});
 text(s,'模擬以過去預測未來，這些時間窗仍曾多次參與研究',72,619,1136,39,25,{color:C.muted});
}
// 09 Six distinct models, three complementary price representations.
{
 const s=slide(9);
 text(s,'單價 × 面積 = 總價，另以不同尺度補充預測',72,145,1136,44,29,{bold:true,color:C.green});
 table(s,[['價格分支','主要模型','學習的價格表示'],['單價分支','單價 CatBoost + LightGBM','單價乘回面積'],['對數分支','總價及 log CatBoost + LightGBM','直接總價與對數總價'],['神經網路分支','兩個表格 NN + LightGBM','面積連結與先驗倍率']],72,213,1136,301,[213,545,378],{font:25});
 text(s,'3 個 CatBoost ＋ 1 個 LightGBM ＋ 2 個 NN',72,554,1136,45,29,{bold:true});
 text(s,'LightGBM 在三分支共用，互補性是設計動機，不等同消融證明',72,617,1136,42,25,{color:C.muted});
}
// 10 Unit representation and the role of area in the training weights.
{
 const s=slide(10);
 text(s,'總價 = 單價 × 面積',72,151,1136,61,37,{bold:true,color:C.green});
 text(s,'假設例子：單價 CatBoost 出現相同的單價誤差',72,244,1136,40,25,{color:C.muted});
 table(s,[['面積','單價誤差','對應的總價誤差'],['100 平方公尺','1 萬元／平方公尺','100 萬元'],['200 平方公尺','1 萬元／平方公尺','200 萬元']],72,307,1136,220,[338,411,387],{font:28});
 text(s,'面積權重讓單價損失對齊總價誤差',72,565,1136,50,31,{bold:true,color:C.green});
 text(s,'target = y / (A × 10,000)，weight 含 A / 100',72,633,1136,31,24,{color:C.muted});
}
// 11 Date weights.
{
 const s=slide(11);
 text(s,'相對日期權重（正規化前）',72,157,760,42,25,{color:C.muted});
 chartFont(s.charts.add('line',{position:{left:72,top:215,width:752,height:357},categories:['近期','早 1 年','早 2 年','早 3 年'],series:[{name:'相對權重',values:[1,.5,.25,.125],line:{fill:C.green,width:4},marker:{symbol:'circle',size:9},valuesFormatCode:'General'}],hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:1,majorUnit:.25,numberFormatCode:'General'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:23,fill:C.green,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'365 天',902,220,302,92,61,{bold:true,color:C.green});text(s,'日期權重半衰期',902,329,300,42,27);text(s,'面積權重 × 日期權重',877,429,342,42,27,{bold:true});text(s,'提高近期成交影響\n仍保留可用舊資料',877,493,345,85,26);
 text(s,'權重正規化後使用，無效日期採 1；365 天是設計選擇，不是最優保證',72,610,1136,49,24,{color:C.muted});
}
// 12 A baseline prior plus grouped out-of-fold training preparation.
{
 const s=slide(12);
 pair(s,170,'表格神經網路','面積 NN 學單價再換回總價\n先驗倍率 NN 以區域參考單價為起點調整');
 pair(s,326,'五折建案分組 OOF','同一建案放在同一折\n每折的訓練先驗由其他折的 fit 標籤建立');
 pair(s,485,'避免直接使用自己的價格','減少自身或同建案標籤直接流入訓練先驗\nquery 的先驗只由 fit 狀態計算');
 text(s,'這是建案分組 OOF，不是逐筆時間 OOF',72,631,1136,34,24,{bold:true,color:C.rust});
}
// 13 Price-reference and spatial adjustments only from fit prices.
{
 const s=slide(13);
 pair(s,168,'可比成交與時間趨勢','利用 fit 中建案、地址、道路等成交支持\n把基準預測與可比價格資訊結合');
 pair(s,327,'固定空間修正','當參考支持較弱時，補充附近空間資訊\n支持不足則回退，並非強制逐筆修正');
 pair(s,485,'地圖與價格分開','OSM 道路提供粗略地理資訊\n價格標籤與成交參考來自課程 fit 資料');
 text(s,'逐分支修正後才取中位數，不查詢外部 test 真價',72,632,1136,34,25,{bold:true,color:C.green});
}
// 14 Explicitly hypothetical median demonstration.
{
 const s=slide(14);
 text(s,'假設例子：各分支修正後的總價，單位為萬元',72,148,1136,43,25,{color:C.muted});
 chartFont(s.charts.add('bar',{position:{left:72,top:222,width:758,height:339},categories:['單價分支','對數分支','神經網路分支'],series:[{name:'假設預測',values:[1000,1100,3000],fill:C.gray,points:[{idx:0,fill:C.gray},{idx:1,fill:C.green},{idx:2,fill:C.gray}],valuesFormatCode:'#,##0'}],barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:3500,majorUnit:1000,numberFormatCode:'#,##0'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:24,fill:C.body,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'1,100 萬',897,258,316,102,59,{bold:true,color:C.green});text(s,'這筆的最終預測',891,391,322,46,28);text(s,'取三個數字\n排序後的中間值',891,472,322,88,27);
 text(s,'降低單一分支極端輸出的影響，三條共同偏誤仍可能保留',72,615,1136,47,25,{color:C.muted});
}
// 15 Same method, different historical scoring sets.
{
 const s=slide(15);
 text(s,'總價 MAE（萬元，越低越好）',72,146,1136,37,25,{color:C.muted});
 chartFont(s.charts.add('bar',{position:{left:72,top:218,width:1136,height:373},categories:['2021–2022','2022–2023','2023–2024'],series:[{name:'提交方法的時間回測',values:mae.map(x=>Number((x/1e4).toFixed(2))),fill:C.green,valuesFormatCode:'0.00'}],barOptions:{direction:'column',grouping:'clustered',gapWidth:125},hasLegend:false,xAxis:{...axis,majorGridlines:null},yAxis:{...axis,min:0,max:260,majorUnit:50,numberFormatCode:'0'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:27,fill:C.body,bold:true}},chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'評分交易與市場時期不同，年度差值不能解讀為模型逐年變好',72,620,1136,42,25,{color:C.muted});
}
// 16 Public and unknown Private, separated from historical validation.
{
 const s=slide(16);
 text(s,'199.86 萬元',72,206,726,113,63,{bold:true,color:C.green});text(s,'Public MAE = 1,998,568.5706 元',75,350,745,58,30);
 text(s,'在公開評分交易上\n平均絕對總價誤差約為此金額',72,457,740,111,30);
 chartFont(s.charts.add('doughnut',{position:{left:847,top:169,width:347,height:363},categories:['Public 約 30%','Private 約 70%'],series:[{name:'test 評分分割',values:[30,70],points:[{idx:0,fill:C.lime},{idx:1,fill:'#86B09A'}]}],doughnutOptions:{holeSize:64},hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:22,fill:C.body}},dataLabels:{showPercent:true,position:'center',textStyle:{typeface:FONT,fontSize:26,fill:C.ink,bold:true}},chartFill:'none',chartLine:{fill:'none',width:0}}));
 text(s,'Private 成績未知',842,565,371,46,30,{bold:true,color:C.green});text(s,'Public 與 Private 都切自 test，與前頁歷史回測不是同一批交易',72,633,1136,32,23,{color:C.muted});
}
// 17 Honest risks and boundaries.
{
 const s=slide(17);
 pair(s,177,'高價與大面積交易','少量大額誤差會拉高總價 MAE\n整體平均可能掩蓋特定族群的退步',{color:C.rust});
 pair(s,332,'驗證時間窗重複使用','多輪選型可能逐漸適應這些時間窗\n重抽樣區間不能消除此偏差或保證 Private');
 pair(s,488,'可比資料有限','新建案、缺失地址與市場改變\n可能讓訓練中的價格參考不再適用');
}
// 18 Final training and submission integrity.
{
 const s=slide(18);
 table(s,[['階段','實際處理','要避免的問題'],['最終 fit','合併清理後 train 與 valid\n以 467,221 筆建立最終狀態','把歷史回測模型誤當最終模型'],['推論所有 test','保留全部 47,073 筆\n按 sample 列序對齊','漏列、重複編號或排序不一致'],['CSV 輸出','兩欄：編號、總價元\n最後只取整一次，價格至少 1 元','重複縮放、重複取整或欄名錯誤'],['檔案核對','列數、編號、正整數\nUTF-8 BOM、CSV SHA256','價格相同但提交格式或位元不同']],72,173,1136,384,[207,523,406],{font:24});
 text(s,'評分模型與提交檔要一起保存，格式檢查不等於泛化保證',72,606,1136,48,26,{color:C.muted});
}
// 19 Reproduction scope and actual local inference evidence.
{
 const s=slide(19);
 text(s,'47,073 筆完全一致',72,176,1136,89,58,{bold:true,color:C.green});
 text(s,'原 test Excel ＋ 17 項保存模型、地理與 fit 狀態',72,291,1136,54,30);
 text(s,'真正載入六模型，重算三分支及最終中位數',72,373,1136,48,29);
 text(s,'src/predict_v22.py',72,454,1136,48,32,{mono:true,bold:true,color:C.green});
 text(s,'同機既有環境 84.74 秒，未取整最大差 0 元，CSV SHA 相同',72,537,1136,62,26);
 text(s,'全新安裝、從零重訓與地理重建尚未驗證',72,621,1136,38,26,{color:C.rust});
}
// 20 Conclude with the completed selected model, not version competition.
{
 const s=slide(20,{dark:true});
 text(s,'用過去的房屋與交易資料估計總價',72,179,1110,59,38,{bold:true,color:C.lime});
 text(s,'資料解析、近期加權與三分支組合都有明確設計動機',72,278,1136,65,30,{color:C.white});
 text(s,'Public MAE 約 199.86 萬元，Private 仍未知',72,381,1136,57,31,{color:C.white});
 text(s,'程式、正式報告、模型身分與提交檔均可追溯',72,476,1136,50,29,{color:C.pale});
 text(s,'github.com/ziyi930108/nycu-house-price-regression',72,553,1136,45,25,{color:C.pale});
 text(s,'本次選定提交模型 v22',72,622,1110,42,29,{bold:true,color:C.lime});
}
// 21 Exact formula appendix.
{
 const s=slide(21);
 text(s,'所有模型輸出先還原成總價「元」',72,145,1110,43,28,{bold:true,color:C.green});
 text(s,['B_unit = .75 * C_unit + .25 * L_unit','B_nn   = .75 * (.5 * N_area + .5 * N_prior) + .25 * L_unit','B_log  = .75 * (.5 * C_raw  + .5 * C_log)   + .25 * L_unit','','U = S(B_unit, R(B_unit))','N = S(B_nn,   R(B_nn))','H = S(B_log,  R(B_log))','','prediction = median(N, H, U)'].join('\n'),74,222,1132,314,27,{mono:true,color:C.ink});
 text(s,'R：成交參考與時間趨勢　　S：固定空間修正',72,570,1136,43,27);text(s,'省略共同特徵與保存狀態引數，修正發生在每條分支內',72,622,1136,35,23,{color:C.muted});
}
// 22 Main settings, all saved evidence.
{
 const s=slide(22);
 table(s,[['模型設定','內容'],['單價 CatBoost','MAE，2,999 iterations，depth 8\nlearning rate 0.065，seed 42'],['單價目標與權重','target = y / (A × 10,000)\nweight = A / 100 × date_weight'],['日期權重','365 日半衰期，合法日期正規化\n無效日期採中性權重 1'],['表格 NN','embedding MLP，512／256／128\n30 epochs，batch 2048，seed 42']],72,174,1136,399,[329,807],{font:26});
 text(s,'受測推論硬體：RTX 4060 Ti、CUDA 12.6；完整環境見重現文件',72,616,1136,47,24,{color:C.muted});
}
for(const a of ledger){if(/v20|v24|R79/.test(a.title+' '+a.script))throw Error('Internal version label in spoken copy');}
await fs.mkdir(BUILD+'/renders',{recursive:true});await fs.mkdir(OUT,{recursive:true});
await fs.writeFile(BUILD+'/source_data.json',JSON.stringify({selectedSubmission:'submission_v22_trial.csv',publicMAE:publicScore,historicalMAE:mae,medianIllustration:{hypothetical:true,unit:'10,000 TWD',branchTotals:[1000,1100,3000],median:1100},narrative_source:'../narrative.json',slides:ledger},null,2));
await fs.writeFile(BUILD+'/presentation.json',JSON.stringify(p.toProto()));
await (await PresentationFile.exportPptx(p)).save(BUILD+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){const s=p.slides.items[i],stem=String(i+1).padStart(2,'0');const img=await p.export({slide:s,format:'png',scale:1.25});await fs.writeFile(BUILD+'/renders/slide-'+stem+'.png',new Uint8Array(await img.arrayBuffer()));const layout=await s.export({format:'layout'});await fs.writeFile(BUILD+'/renders/slide-'+stem+'.json',await layout.text());console.log('RENDERED',i+1);}
const total=ledger.filter(x=>!x.is_appendix).reduce((a,x)=>a+x.seconds,0);
const notes=['# 臺灣房屋總價預測：完整流程上台講稿','','共 20 頁主講、2 頁技術附錄。逐頁配速合計 '+Math.floor(total/60)+' 分 '+total%60+' 秒，不含問答與附錄。這是純漢字字數按 285 字／分鐘的配速估計，尚未實際彩排計時；不靠停頓、附錄或問答補足 12 分鐘。','','程式與報告：https://github.com/ziyi930108/nycu-house-price-regression（私人儲存庫，需閱覽權限）。','',...ledger.flatMap((a,i)=>[`## ${String(i+1).padStart(2,'0')}　${a.title}`,'',a.is_appendix?'問答時使用，不列入主講時間。':`建議配速 ${a.seconds} 秒。`,'',a.script,'','資料來源：',...a.sources.map(x=>'- '+x),''])].join('\n');
await fs.writeFile(OUT+'/完整流程講稿.md',notes,'utf8');console.log(JSON.stringify({slides:p.slides.items.length,mainTalkSeconds:total,candidate:BUILD+'/candidate.pptx'}));
