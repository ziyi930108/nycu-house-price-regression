import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const DIR=path.resolve(import.meta.dirname,'..');
const ROOT=path.resolve(DIR,'../..');
const BUILD=path.join(DIR,'build');
const SKILL='C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const {resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const FONT=resolvePresentationFont({fontFamily:'Microsoft JhengHei',availableFonts:['Microsoft JhengHei']});
const C={bg:'#F7F5EF',ink:'#173C38',body:'#314B46',muted:'#627570',green:'#167364',pale:'#E4EDE6',white:'#FFFFFF',rust:'#B04F35',gray:'#B5C5BC'};
const narrative=JSON.parse(await fs.readFile(DIR+'/content/narrative.json','utf8'));
const slides=narrative.slides;
if(slides.length!==23||slides.filter(s=>!s.is_appendix).length!==20)throw Error('Expected twenty spoken slides and three appendices');
const scores=JSON.parse(await fs.readFile(ROOT+'/outputs/round79/results.json','utf8'));
const historicalTWD=['auxiliary','development','confirmation'].map(k=>scores.stages[k].metrics.R79.overall_mae);
const historical=historicalTWD.map(v=>Number((v/10000).toFixed(2))); 
const feedback=JSON.parse(await fs.readFile(ROOT+'/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json','utf8'));
const publicMAE=feedback.screenshotted_scores['submission_v22_trial.csv'];
if(publicMAE!==1998568.5706)throw Error('Unexpected public source');
const p=Presentation.create({slideSize:{width:1280,height:720}});
function tx(s,v,x,y,w,h,size=30,o={}){const sh=s.shapes.add({name:o.name||'body',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=v;sh.text.style={typeface:o.mono?'Consolas':FONT,fontSize:size,color:o.color||C.body,bold:!!o.bold,alignment:o.align||'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};return sh;}
function note(s,n){const a=slides[n-1];s.speakerNotes.textFrame.setText(a.script+'\n\n資料來源\n'+a.sources.join('\n')+(n===1?'\nCover photo: Markus Winkler. https://unsplash.com/photos/gray-concrete-building-vztbnMlTDNs\nUnsplash License. Taipei contextual photo, not a competition transaction.':''));}
function slide(n,dark=false){const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;tx(s,slides[n-1].title,68,44,1144,96,46,{name:'slide-title',bold:true,color:dark?C.white:C.ink});tx(s,String(n).padStart(2,'0'),1150,672,60,28,19,{color:dark?C.pale:C.muted,align:'right'});note(s,n);return s;}
function bottom(s,str,o={}){tx(s,str,70,603,1120,61,o.size||27,{bold:!!o.bold,color:o.color||C.green});}
function table(s,rows,x,y,w,h,widths,font=26){const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,values:rows,columnWidths:widths});t.borders.assign({fill:C.bg,width:2,style:'solid'});for(let r=0;r<rows.length;r++){t.rows[r].height=h/rows.length;for(let c=0;c<rows[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:r%2?C.white:C.pale;cell.text.style={typeface:FONT,fontSize:font,color:r===0?C.white:C.body,bold:r===0,verticalAlignment:'middle',autoFit:'none',insets:{left:16,right:12,top:9,bottom:9}};}}return t;}
function pair(s,y,a,b){tx(s,a,70,y,360,64,33,{bold:true,color:C.green});tx(s,b,463,y,735,114,30);}
function bar(s,cats,vals,{x=70,y=205,w=1120,h=345,format='0.00',max,points}={}){const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cats,series:[{name:'數值',values:vals,fill:C.green,valuesFormatCode:format,points}],barOptions:{direction:'column',grouping:'clustered',gapWidth:125},hasLegend:false,chartFill:'none',plotAreaFill:'none',xAxis:{textStyle:{typeface:FONT,fontSize:25,fill:C.body},line:{fill:C.gray,width:1}},yAxis:{minimumScale:0,maximumScale:max,numberFormatCode:format,textStyle:{typeface:FONT,fontSize:24,fill:C.muted},majorGridlines:{fill:'#D8E0D7',width:1},line:{fill:'none',width:0}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{typeface:FONT,fontSize:29,fill:C.ink}}});applyPresentationChartFont(c,{fontFamily:FONT});return c;}
// 01 Opening photo and editable title.
{const s=p.slides.add();s.background.fill=C.bg;const img=await fs.readFile(BUILD+'/assets/taipei_markus_winkler.jpg');s.images.add({blob:new Uint8Array(img),contentType:'image/jpeg',alt:'台北住宅情境照片，非競賽交易個案',fit:'cover',position:{left:775,top:48,width:435,height:624}});tx(s,'臺灣房屋\n總價預測',70,150,650,205,76,{bold:true,color:C.ink});tx(s,'資料、模型與驗證的完整流程',73,401,664,72,31,{color:C.green});tx(s,'v22 專題報告',73,515,630,45,28);tx(s,'2026.10',73,629,620,30,22,{color:C.muted});note(s,1);}
// 02 Fundamental supervised regression, then inference.
{const s=slide(2);tx(s,'房屋條件',72,172,410,65,45,{bold:true,color:C.green});tx(s,'估計總價',712,172,480,65,45,{bold:true,color:C.green});tx(s,'地點、面積、樓層、交易日期',72,268,530,106,32);tx(s,'例如 1,020 萬元\n教學假設',712,268,480,106,32);pair(s,422,'學習','用有成交價的歷史資料找規律');pair(s,524,'推論','把已學規律用在尚未知道價格的交易');}
// 03 The same example in notes and report.
{const s=slide(3);tx(s,'MAE = 每筆「差多少錢」的平均',70,150,1125,63,37,{bold:true,color:C.green});table(s,[['教學假設','真實總價','預測總價','差額絕對值'],['交易 A','1,000 萬','1,100 萬','100 萬'],['交易 B','1,000 萬','950 萬','50 萬']],70,259,1138,258,[240,299,299,300],29);bottom(s,'（100 萬 + 50 萬）÷ 2 = 75 萬元，越低越好',{bold:true,size:31});}
// 04 Course split and leaderboard split are different.
{const s=slide(4);table(s,[['檔案','原始列數','已知總價','實際用途'],['train','455,787','有','學習與歷史回測'],['valid','11,994','有','最終全量訓練時併入'],['test','47,073','無','全部產生價格預測'],['sample','47,073','格式示例','提供欄名和編號順序']],70,174,1138,385,[215,205,218,500],27);bottom(s,'Kaggle 將 test 約分成 Public 30% 與 Private 70%',{size:28});}
// 05 A native table maps each action to its reason.
{const s=slide(5);table(s,[['步驟','做什麼','為什麼'],['01 資料整理','確認欄位、清理與轉換','讓每一筆資料有一致的意義'],['02 模型學習','用較早交易學習不同價格表示','找出房屋條件與價格的關係'],['03 歷史回測','預測較晚交易，再與答案比較','檢查遇到未見資料時的誤差'],['04 最終交付','全量訓練、預測與核對CSV','讓提交內容對應到保存模型']],70,177,1138,389,[240,420,478],25);bottom(s,'每一步都保留程式與結果，方便核對和重現',{bold:true});}
// 06 Exact arithmetic from preparation evidence.
{const s=slide(6);table(s,[['整理動作','保留列數'],['原始 train','455,787'],['移除 3 列英文表頭','455,784'],['移除 529 列完全重複','455,255'],['加入 11,994 筆 valid，再去重 28 筆','467,221']],70,176,740,389,[550,190],25);tx(s,'467,221',865,215,346,90,57,{bold:true,color:C.green});tx(s,'最終有標籤交易',865,319,333,80,29);tx(s,'去重比對整列\n包含價格、排除編號',865,449,344,114,27);bottom(s,'修正格式與重複資訊，test 的 47,073 筆全數保留',{size:27});}
// 07 Features make input meanings usable; price remains the target.
{const s=slide(7);table(s,[['原始資料','交給模型的特徵','用途'],['民國日期、中文樓層','交易時點、樓層數值與旗標','保留時間與樓層意義'],['土地、建物、車位面積','面積與比例','描述交易的規模與組成'],['地址、道路、建案','區域類別與粗略地理資訊','描述區位'],['編號、總價','編號對齊，總價作學習答案','不把這兩欄當輸入特徵']],70,176,1138,386,[335,453,350],25);bottom(s,'單價 CatBoost 使用 95 欄，其中 23 欄是類別；其他模型另有編碼',{size:25});}
// 08 Fixed chronological windows.
{const s=slide(8);tx(s,'先學過去，再預測後一年',70,159,1125,60,37,{bold:true,color:C.green});table(s,[['回測','學習資料的截止日','評分交易期間'],['較早窗口','2021-08-10','2021-08-11 至 2022-08-10'],['中間窗口','2022-08-10','2022-08-11 至 2023-08-10'],['較新窗口','2023-08-10','2023-08-11 至 2024-08-10']],70,274,1138,284,[230,360,548],27);bottom(s,'模型、編碼與價格參考都只從當期學習資料建立',{bold:true});}
// 09 Two model families explained before acronyms.
{const s=slide(9);pair(s,168,'樹模型','依資料條件逐步分群，再估計價格\n例如同區、面積、樓層共同影響估價');tx(s,'CatBoost 與 LightGBM 會讓多棵樹逐步修正誤差',463,302,737,88,27,{color:C.muted});pair(s,424,'表格神經網路','把數值和類別轉成可學習的表示\n再透過多層計算估計價格');bottom(s,'使用多種模型的目的是嘗試不同估計方式，效果仍要靠回測',{size:26});}
// 10 Unit scaling and the area weight are demonstrated.
{const s=slide(10);tx(s,'總價 = 單價 × 面積',70,157,1138,65,40,{bold:true,color:C.green});table(s,[['教學假設','單價估錯','造成的總價誤差'],['50 平方公尺','每平方公尺 1 萬元','50 萬元'],['100 平方公尺','每平方公尺 1 萬元','100 萬元']],70,259,1138,240,[360,389,389],29);tx(s,'面積權重讓單價模型更貼近「總共差多少錢」',70,541,1138,55,30,{bold:true,color:C.green});tx(s,'另一種表示使用 log 壓縮價格尺度，最後都還原成總價元',70,615,1138,51,26,{color:C.muted});}
// 11 Literal teaching data, relative weight before normalization.
{const s=slide(11);tx(s,'相對日期權重，未正規化的示意',70,156,1138,43,29,{color:C.muted});bar(s,['最新日期','早一年','早兩年'],[1,.5,.25],{format:'0.00',max:1.15,y:226,h:311});bottom(s,'365 日半衰期：較近的成交在訓練中占較大比重',{bold:true,size:29});}
// 12 Group OOF example represented as an editable table.
{const s=slide(12);tx(s,'價格先驗：先從已知成交，估一個參考單價',70,160,1138,60,34,{bold:true,color:C.green});table(s,[['建立誰的價格先驗','參考價格來自','避開的問題'],['第 1 組建案','其餘 4 組的已知成交','把本組答案塞回自己的特徵'],['第 2 組建案','其餘 4 組的已知成交','同樣排除自己的建案組'],['新的待預測交易','當期學習資料的已知成交','使用待預測交易的未知價格']],70,274,1138,282,[310,400,428],25);bottom(s,'五折建案分組 OOF 會輪流排除一組；它不等於時間切分',{size:27});}
// 13 Six unique models share the LightGBM prediction.
{const s=slide(13);table(s,[['分支','主要模型','共同加入'],['單價分支','單價 CatBoost','單價 LightGBM'],['神經網路分支','面積 NN、價格先驗 NN','同一個 LightGBM 預測'],['對數分支','總價 CatBoost、log CatBoost','同一個 LightGBM 預測']],70,203,1138,297,[278,494,366],26);tx(s,'3 個 CatBoost + 1 個 LightGBM + 2 個 NN = 6 個模型',70,544,1138,53,29,{bold:true,color:C.green});tx(s,'每條分支先組合模型、再修正價格，精確係數放在附錄',70,618,1138,47,27,{color:C.muted});}
// 14 Explain fit-based corrections without implying external price lookup.
{const s=slide(14);pair(s,170,'成交參考','從當期資料找同建案、同地址、同道路的成交\n再按物件條件比較，結合模型估價與參考價');pair(s,386,'空間修正','參考支持較弱時，補充附近空間資訊\n支持不足就回退，避免勉強套用少量樣本');bottom(s,'成交價格只來自課程的已知標籤；外部道路資料不含成交價',{size:26});}
// 15 Hypothetical prices after each branch has been corrected.
{const s=slide(15);tx(s,'同一間房子的三個修正後估價，單位：萬元（教學假設）',70,151,1138,54,28,{color:C.muted});bar(s,['分支 A','分支 B','分支 C'],[980,1020,1800],{format:'0',max:2200,y:238,h:298,points:[{idx:1,fill:C.rust}]});bottom(s,'排序後取中間值 1,020 萬元，降低單一極端估價的影響',{bold:true,size:28});}
// 16 Actual saved out-of-time results, unchanged data windows.
{const s=slide(16);tx(s,'MAE，單位：萬元，越低越好',70,153,1138,46,28,{color:C.muted});bar(s,['2021–2022','2022–2023','2023–2024'],historical,{max:280,format:'0.00',y:231,h:302});bottom(s,'三個年度是不同評分集合，不能把下降解讀成同一批資料持續進步',{size:25});}
// 17 Public score as an error amount, not accuracy.
{const s=slide(17);tx(s,'199.86 萬元',70,174,1138,124,79,{bold:true,color:C.green});tx(s,'v22 的 Public 平均絕對誤差',75,316,1115,60,34);tx(s,'完整分數 1,998,568.5706 元，依留存的 Kaggle 提交結果截圖',75,409,1115,91,26,{color:C.muted});tx(s,'Private 約占 test 的 70%，目前分數未知',75,542,1115,65,34,{bold:true,color:C.ink});}
// 18 Limits remain in the main talk.
{const s=slide(18);pair(s,167,'驗證重複使用','研究多次看過同一批歷史窗口\n選擇方法時仍可能逐漸貼近那些資料');pair(s,333,'整體平均的盲點','某些高價群或新建案仍可能更差\n中位數也無法消除三條分支共同的偏差');pair(s,499,'地理資訊的精度','各年度使用對應的非價格道路快照\n道路代表點無法等同精確的住宅位置');}
// 19 Real entry point and proven reproduction scope.
{const s=slide(19);tx(s,'467,221 筆有標籤資料完成最終學習',70,162,1138,70,38,{bold:true,color:C.green});tx(s,'47,073 筆 test 全部保留並輸出「編號、總價元」',70,277,1138,74,32);tx(s,'程式入口',70,388,265,50,30,{bold:true});tx(s,'src/predict_v22.py',359,388,845,60,36,{mono:true,color:C.green});tx(s,'保存模型加原始 test 重新推論，47,073 筆及 CSV 完全一致',70,498,1138,67,29);bottom(s,'已驗證的是保存模型推論；全新環境安裝與從零重訓尚未驗證',{size:25,color:C.rust});}
// 20 Plain conclusion tied to method, outcome and uncertainty.
{const s=slide(20,true);tx(s,'用歷史交易學習房價規律',70,171,1138,80,43,{bold:true,color:C.white});tx(s,'清理資料、設計特徵，再用時間回測檢查誤差',70,301,1138,86,33,{color:C.pale});tx(s,'六模型形成三分支，修正後逐筆取中位數',70,421,1138,86,33,{color:C.pale});tx(s,'Public MAE 約 199.86 萬元，Private 分數仍未知',70,554,1138,79,33,{bold:true,color:C.white});}
// 21 Exact operations for code inspection.
{const s=slide(21);tx(s,'所有模型預測先還原成總價元',70,149,1138,48,30,{bold:true,color:C.green});tx(s,['B_unit = .75*C_unit + .25*L_unit','B_nn   = .75*(.5*N_area + .5*N_prior) + .25*L_unit','B_log  = .75*(.5*C_raw  + .5*C_log)   + .25*L_unit','','U = S(B_unit, R(B_unit))','N = S(B_nn,   R(B_nn))','H = S(B_log,  R(B_log))','','prediction = median(N, H, U)'].join('\n'),72,223,1136,331,27,{mono:true,color:C.ink});bottom(s,'R 代表成交參考與時間趨勢，S 代表固定空間修正',{size:27});}
// 22 Settings and real file paths.
{const s=slide(22);table(s,[['項目','保存設定'],['單價 CatBoost','MAE，2,999 棵樹，深度 8，學習率 0.065'],['單價目標與權重','目標 y / (A × 10,000)，權重 A / 100 × 日期權重'],['有效面積 A','正且有限：先建物面積，再車位面積，否則 1'],['v22 推論入口','deliverables/v22_github/src/predict_v22.py'],['主要訓練來源','src/round79_recency_unit.py']],70,183,1138,381,[306,832],24);bottom(s,'精確環境、依賴與完整指令見隨附報告及 GitHub 重現文件',{size:26});}
// 23 Beginner glossary for questions, not spoken time padding.
{const s=slide(23);table(s,[['名詞','白話意思'],['特徵與標籤','特徵是已知房屋條件，標籤是學習時的成交價答案'],['訓練與推論','訓練找規律，推論用保存的規律估計新交易'],['過擬合','過度貼近看過的資料，遇到新資料誤差可能變大'],['權重','某筆資料的誤差，在學習目標中占多少比重'],['先驗與 OOF','先建立參考價，並用其他組資料建立訓練特徵'],['集成與中位數','集成結合多個估計，中位數是排序後的中間值']],70,177,1138,418,[301,837],25);}
await fs.mkdir(BUILD,{recursive:true});
await fs.writeFile(BUILD+'/source_data.json',JSON.stringify({publicMAE,historicalMAE_TWD:historicalTWD,historicalMAE10kTWD_display:historical,medianExample:{illustrative:true,values10kTWD:[980,1020,1800],median:1020},narrativeSource:'../content/narrative.json',nativeTableSlides:[3,4,5,6,7,8,10,12,13,22,23],nativeChartSlides:[11,15,16]},null,2));
await (await PresentationFile.exportPptx(p)).save(BUILD+'/candidate.pptx');
await fs.writeFile(BUILD+'/presentation.json',JSON.stringify(p.toProto()));
console.log(JSON.stringify({candidate:BUILD+'/candidate.pptx',slides:p.slides.items.length}));

