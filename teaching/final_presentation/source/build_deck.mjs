import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const MODULES=process.env.RUNTIME_NODE_MODULES||'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(MODULES,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);

const ROOT=path.resolve(import.meta.dirname,'..');
const QA=path.join(ROOT,'build');
const OUT=path.join(ROOT,'build/deliverables');
const SKILL='C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const {resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const FONT=resolvePresentationFont({fontFamily:'Microsoft JhengHei',availableFonts:['Microsoft JhengHei']});
const old=JSON.parse(await fs.readFile(path.join(import.meta.dirname,'readable_ppt_content_20261012.json'),'utf8'));
const curveBytes=await fs.readFile(path.join(import.meta.dirname,'training_curve.json'));
const curve=JSON.parse(curveBytes.toString('utf8'));
const curveSHA=createHash('sha256').update(curveBytes).digest('hex');
const sha=b=>createHash('sha256').update(b).digest('hex');
if(curve.epochs.length!==30||curve.train_mae_twd.length!==30||curve.validation_mae_twd.length!==30||curve.metric!=='unweighted_total_price_MAE'||curve.epoch_selection!=='fixed_last_epoch')throw Error('Unexpected curve scope');
const C={bg:'#FFFFFF',ink:'#172C45',body:'#33485F',blue:'#2259B6',orange:'#BC5B2E',muted:'#617285',line:'#D9E1E9',light:'#F1F5F9',white:'#FFFFFF'};
const slides=old.slides.map(s=>({number:s.number,title:s.title,script:s.script,is_appendix:s.is_appendix,why:s.why,sources:s.sources}));
function set(n,title,script,why){Object.assign(slides[n-1],{title,script,why:why||slides[n-1].why});}
set(1,'台灣房屋銷售價格預測挑戰賽',`大家好，我是廖子頤，今天介紹台灣房屋銷售價格預測挑戰賽。想像我們拿到一筆交易，知道面積、地點和成交日期，卻不知道總價。這份專題要做的，就是利用以前有答案的成交資料，估計新交易的價格。

我會先說明怎麼判斷估價好不好，再介紹資料清理、文字轉換與模型學習，最後看完整方法的驗證結果和限制。每一步都會說明理由，讓沒有機器學習背景的人，也能跟著一筆交易理解整個流程。`);
set(5,'表頭排除與完全去重',`資料整理先處理兩個明確問題。原始訓練檔混入三列英文欄名，程式同時確認編號欄是英文的序號，總價欄是英文的總價名稱，才把它辨識為表頭。不是看到英文就刪。

再比較除了編號以外的全部原始欄位，成交總價也要一起比較。全部相同才算完全重複，只留第一次，共移除五百二十九列。相同建案或面積並不夠，日期或價格不同仍可能是另一筆交易。

最後合併有答案的驗證檔，再移除二十八筆跨檔重複，得到四十六萬七千二百二十一筆。這樣避免同一紀錄反覆影響模型。測試檔則完整保留。`);
set(6,'文字轉成一致的資料',`格式一致，模型才知道哪些內容可以比較。數字先去除逗號，轉成可運算的數值。民國一百一十二年八月十一日，換成西元二〇二三年八月十一日，再拆出年、月和季等時間線索。

樓層把十二層解析成十二，地下二層解析成負二，另保留地下室或整棟標記。地址將台字統一成臺字，再拆縣市、行政區與道路。交易筆棟數拆成土地、建物、車位數量。

空白和無法解析的內容，保留缺失或異常標記。需要從資料估計的填補值、尺度和類別編碼，只由訓練資料建立，再沿用到驗證與測試，避免後面的資料影響學習規則。`);
set(7,'輸入欄位與價格答案',`整理房屋線索叫特徵工程。模型要看的是交易條件，例如面積、用途、樓層、區位和時間。答案則是成交總價，必須另外保存，不能放進輸入。

欄位不是越多越好。編號只用來對齊答案。完整備註可能含價格文字，解約資訊可能在交易後才產生，所以不直接輸入。備註只保留有限的條件標記。建築完成年月不可用，不能硬算屋齡，管理組織全相同也排除。

基礎特徵共有八十七欄，部分模型再加入八欄不含價格的道路地理資訊，共九十五欄，其中二十三欄是類別。道路代表點只提供大致位置，不等於精確門牌。`);
set(8,'用較早交易預測下一年',`房價會隨時間改變，因此採時間回測。每個起點只用那天以前的交易學習，再預測後面一年的價格。主要檢查三個八月起點，另外補一個二月起點，看結果是否只適合某個日期。

這個界線不只管模型。文字編碼、參考單價、成交參考和空間狀態，凡是從資料建立的部分，都要用當期訓練資料重建。不能拿後期模型回頭測較早市場，否則後面的資訊可能流入，形成資料洩漏。

這些期間已反覆用來研究，而且彼此有重疊。它們能檢查歷史穩定性，卻不是四次獨立盲測。後面看到三期改善，也不能把它當成隱藏分數的勝率。`);
set(12,'參考單價與分組先驗',`參考單價，也叫價格先驗，是先整理這類地點與房屋大約多少錢一平方公尺，再讓神經網路學調整倍率。查表依據是地區、交易類型和建物類型，資料不足就逐層放寬，最後用整體參考。

建案用來分組，不是這張先驗表的查詢鍵。訓練交易按建案分五組，每組只用其他四組建立自己的參考價，避免把本建案的答案放回自己的線索。這叫分組折外處理，但它不是按日期分折。新的待預測交易，則使用當期全部訓練資料建立參考。

這支網路採日期加權的平滑絕對損失，差距在十萬元內時調整較平順。固定訓練三十輪，不依驗證最低點挑輪數。`);
set(16,'四個時間窗口的完整結果',`這張表評分的是完整三分支中位數方法。比較基準的參考單價網路使用日期加權絕對誤差，此方法改用日期加權平滑損失，其餘設定固定。

最早年度比基準多錯約八百八十四元，中間年度少錯約一千四百一十六元，最近年度少錯約一千五百二十六元。額外二月起點也少錯約三百零三元。負差表示改善，正差表示退步。

每列是不同市場與交易組成，不能把四個分數畫成同一模型越練越好的曲線。改善幅度都很小，而且基準使用保存預測，沒有同期配對重訓，所以還不能把差異全歸因於損失函數。接著還要看分群和不支持這個方法的證據。`);
set(17,'平均改善仍有反例',`最近年度整體平均稍微變好，不代表所有交易都變好。既有建案的平均誤差增加約兩百一十三元，新建案則減少約兩千八百四十二元。最早年度也退步，因此原本要求跨期一致改善的強推薦條件沒有通過。

另一個對照方法更提醒我們：它的主要離線窗口整體較好，公開分數卻是兩百零八點三〇萬元，比基準更差。歷史表現較好，不能直接推出未知資料也較好。

最近年度按建案重抽樣的差值區間，從改善約三千五百九十五元，到退步約五百二十三元，跨過零。這只衡量既有預測的不確定性，沒有包含重新訓練和選型偏差，更不是隱藏分數的勝率。`);
set(18,'Public 成績與證據限制',`目前方法的公開平均絕對誤差約一百九十九點五八萬元。精確值是畫面上的一百九十九萬五千八百三十六點二一三六元，來自提交者文字回報，沒有新截圖。

相對基準少約兩千七百三十二元，約降低百分之零點一三七，屬於小幅變化。公開部分只涵蓋約三成測試交易，其餘七成的隱藏分數仍然未知。公開與歷史驗證用的是不同交易，不能直接比較兩邊誰比較低。

窗口反覆使用、單一亂數種子和粗略道路位置，都限制我們能下的結論。多模型和中位數也可能一起估錯。因此這份專題保留不利證據，將目前成果理解為有限的改善，沒有宣稱未來一定更準。`);
set(19,'實際訓練與時間驗證曲線',`這張是固定設定另跑一次的單一神經網路診斷，不是原提交模型當時的軌跡，也不是完整融合。用二〇二三年八月以前資料訓練，後一年交易驗證。

藍線是訓練誤差，橘線是時間驗證誤差，越低越好。每輪關閉隨機丟棄，用完整資料計算不加權總價誤差。最後一輪約七十一點五與一百七十七點七一萬元，驗證後段波動趨平，仍不能證明沒有過擬合。

固定完成三十輪並封存後才讀驗證答案，沒有依最低點挑輪數。模型大小全程固定，輪數不是模型容量或ＶＣ維度。訓練先驗來自其他建案組，驗證先驗來自全部訓練資料，兩線差距也受到這個差異與時間組成影響。`);
set(20,'最終預測與交付檢查',`方法確認後，最終使用四十六萬七千二百二十一筆有答案資料，只新增一次參考單價神經網路訓練。其餘五個模型沿用保存的預測向量，沒有宣稱全部重新訓練或重新原生推論。

替換新的網路分量後，完成分支修正與中位數，最後才取整，輸出四萬七千零七十三筆。兩欄是編號與總價元，必須核對列數、順序、整數和檔案雜湊，避免預測對到錯誤房屋。

新網路已重新載入，全部待預測輸出核對一致。程式、模型與發行證據都保存，能追查這次價格如何形成。但乾淨環境從原始資料完成六模型全部重訓與推論，仍未驗證。`);
set(21,'結論',`回到最初問題，只有房屋條件怎麼估總價？先排除混入表頭與完全重複，把數字、日期、樓層和地址整理成一致線索。從以前有答案的交易學習，再用較晚交易檢查誤差。

不同模型分別學總價、單價與對數。面積權重對齊總價差距，日期權重提高近期成交影響，參考單價網路使用平滑損失。六模型形成三分支，各自修正後逐筆取中位數。

完整方法的歷史改善很小，也有退步與反例。公開平均誤差約一百九十九點五八萬元，隱藏分數仍未知。以上是這次資料整理、模型設計與驗證的完整流程，謝謝大家。`);
set(22,'附錄　融合公式',`這頁供問答核對。所有模型符號都代表已換回新臺幣元的總價。兩個神經網路先平均，直接總價和對數總價模型也先平均。每路主要部分占四分之三，共同的單價樹模型占四分之一。各路先做成交參考，再把原初估和成交修正結果一起交給空間修正。最後每筆交易取三路中位數，不再乘面積。`);
set(23,'附錄　固定訓練設定',`這頁提供參數核對。參考單價網路有三個隱藏層，固定三十輪，種子四十二，採日期加權平滑損失，十萬元為轉折。基準只改用日期加權絕對誤差，但本次沒有同期配對重訓基準，因此不能完全分離損失效果和訓練變異。其他模型仍使用各自的設定，沒有把六模型全改成同一種損失。`);
set(24,'附錄　名詞對照',`特徵是房屋條件，標籤是已知總價。訓練利用有答案的資料調整規則，推論則套用保存規則。先驗是訓練成交整理出的參考基準。分組折外處理，讓某組只使用其他組的標籤建立參考。中位數是排序後的中間值。過擬合表示太貼合練習資料的細節，換成新交易可能失準。`);
set(2,'回歸與模型學習',`預測連續的金額叫回歸。機器學習就是讓電腦從題目與答案找估價規則。面積、樓層、地點等條件叫特徵，已知總價叫標籤。

模型先估價格，和答案比較差距，再調整規則。反覆練習後保存模型，遇到未知交易時，只看條件估價。學規則叫訓練，套用保存規則叫推論。

房價同時受區位、面積和建物型態影響，很難用一條人工公式處理，所以讓模型從成交資料學習。但練習表現好還不夠，必須用沒參與學習的交易檢查。`);
set(4,'四份資料的用途',`四份資料用途不同。訓練檔約四十五萬六千列，有房屋條件與價格答案。驗證檔約一萬二千列，也有答案，最後完整訓練時才加入。

測試檔四萬七千零七十三筆，沒有價格答案，要全部估出來。提交範本只規定欄位與編號順序，示範價格不能拿來學。

平台把測試檔約三成用於公開分數，叫Public，其餘約七成算隱藏分數，叫Private。三比七分的是測試檔，和有答案的驗證檔用途不同。隱藏部分的答案仍不知道。`);
set(9,'樹模型與神經網路',`這次使用兩類模型。樹模型依條件逐步判斷，例如面積在哪個範圍、房子在哪個地區。條件門檻由資料學習，許多小樹再逐步修正誤差。工具叫CatBoost和LightGBM。

神經網路則把數值與分類資訊轉成可學習的數字表示，經過多層計算，學習不同條件一起出現時價格怎麼變。建案的數字表示也會隨訓練調整，不是人工指定等級。

兩種學法有機會互補，但模型更多不保證更準，必須看完整組合的驗證結果。`);
set(13,'六個模型與三條分支',`所有模型先換回總價元。第一路以單價CatBoost為主，第二路先平均兩個神經網路，第三路先平均直接總價與對數總價兩個樹模型。

每一路再加入同一個LightGBM的總價預測，主要部分占七成五，共同模型占兩成五。合起來共四個樹模型、兩個神經網路。

教學假設中，主要部分估一千萬，共同模型估一千兩百萬，加權後就是一千零五十萬。這只是初估，還要分別修正，再取中位數。固定比例沒有證明最佳，三路共用一個模型，也不是完全獨立的估價者。`);
set(15,'逐筆取中位數',`三個總價由小到大排序，取中間那個，叫中位數。教學例子中，三支修正後估一千萬、一千一百萬和一千六百萬元，就採用一千一百萬元。

平均容易被單一極端值往上拉，中位數能減少這種影響。但如果高估價剛好正確，中位數也可能捨掉有用資訊。三條一起低估時，取中間仍會低估。

所以它沒有每次更好的保證。順序是先換回總價，各支完成成交與空間修正，最後才取中位數。此時已是總價，不能再乘一次面積。`);
// Keep presentation language neutral. Exact immutable source identities stay in the private lineage file.
for(const s of slides){s.script=s.script.replace(/沒有去外部實價登錄找測試答案。/g,'').replace(/v25|v22|v24|R107/g,'').replace(/本研究/g,'此方法');}
slides[13].script=slides[13].script.replace('這些價格只來自當期訓練資料，','價格參考只使用當期訓練資料。');
const p=Presentation.create({slideSize:{width:1280,height:720}});
function tx(s,v,x,y,w,h,size=32,o={}){const sh=s.shapes.add({name:o.name||'body',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=v;sh.text.style={typeface:o.mono?'Consolas':FONT,fontSize:size,color:o.color||C.body,bold:!!o.bold,alignment:o.align||'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};return sh;}
function note(s,n){const a=slides[n-1];s.speakerNotes.textFrame.setText(a.script+'\n\n補充理由\n'+a.why+'\n\n資料依據\n課程交易資料、保存的程式與發行證據、方法與程式導覽。完整來源對照另存交付核對紀錄。');}
function slide(n){const s=p.slides.add();s.background.fill=C.bg;tx(s,slides[n-1].title,64,43,1152,86,46,{bold:true,color:C.ink,name:'slide-title'});tx(s,String(n).padStart(2,'0'),1155,676,60,28,20,{color:C.muted,align:'right'});note(s,n);return s;}
function foot(s,v,o={}){tx(s,v,66,o.y||614,1135,o.h||55,o.size||27,{color:o.color||C.blue,bold:!!o.bold});}
function table(s,values,x,y,w,h,widths,font=29){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.white,width:2,style:'solid'});for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?C.blue:(r%2?C.white:C.light);a.text.style={typeface:FONT,fontSize:font,color:r===0?C.white:C.body,bold:r===0,verticalAlignment:'middle',autoFit:'none',insets:{left:16,right:12,top:9,bottom:9}};}}return t;}
function pair(s,y,a,b){tx(s,a,66,y,310,75,35,{bold:true,color:C.blue});tx(s,b,412,y,795,132,31);}
function bar(s,cats,values,{x=72,y=205,w=1138,h=350,max,format='0',points}={}){const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cats,series:[{name:'數值',values,fill:C.blue,points,valuesFormatCode:format}],barOptions:{direction:'column',grouping:'clustered',gapWidth:130},hasLegend:false,chartFill:'none',plotAreaFill:'none',xAxis:{textStyle:{typeface:FONT,fontSize:29,fill:C.body},line:{fill:C.line,width:1}},yAxis:{min:0,max,numberFormatCode:format,textStyle:{typeface:FONT,fontSize:27,fill:C.muted},majorGridlines:{fill:C.line,width:1},line:{fill:'none',width:0}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{typeface:FONT,fontSize:32,fill:C.ink}}});applyPresentationChartFont(c,{fontFamily:FONT});return c;}

// A new layout uses the old deck solely as a content and evidence source.
{const s=p.slides.add();s.background.fill=C.bg;tx(s,'台灣房屋銷售價格\n預測挑戰賽',73,119,1134,205,70,{bold:true,color:C.ink,name:'slide-title'});tx(s,'利用歷史成交資料，估計新交易總價',78,379,1110,74,38,{color:C.blue});tx(s,'姓名：廖子頤\n學號：315581048',80,520,1070,108,31);note(s,1);}
{const s=slide(2);pair(s,173,'特徵','地點、面積、樓層、交易日期\n模型估價時能看到的房屋條件');pair(s,329,'標籤','已知成交總價\n訓練時用來比較的答案');foot(s,'訓練用已知答案調整規則，推論用保存規則估計新交易',{bold:true,size:30});}
{const s=slide(3);tx(s,'MAE = 每筆相差金額的平均，越低越好',66,155,1142,65,35,{bold:true,color:C.blue});table(s,[['教學假設','真實總價','預測總價','相差金額'],['交易 A','1,000 萬','1,100 萬','100 萬'],['交易 B','1,000 萬','950 萬','50 萬']],66,262,1144,251,[240,300,300,304],31);foot(s,'（100 萬 + 50 萬）÷ 2 = 75 萬元',{bold:true,size:36});}
{const s=slide(4);table(s,[['資料','原始列數','有價格答案','用途'],['train','455,787','有','學習與歷史回測'],['valid','11,994','有','最終訓練時合併'],['test','47,073','無','每筆都要估價'],['sample','47,073','僅格式示例','提供編號與欄位']],66,171,1144,379,[205,210,235,494],28);foot(s,'Public 約占 test 30%，Private 約占 70%，和 valid 用途不同',{size:27});}
{const s=slide(5);table(s,[['整理步驟','保留列數'],['原始訓練資料','455,787'],['移除 3 列混入表頭','455,784'],['移除 529 列完全重複','455,255'],['加入 valid，再排除 28 列跨檔重複','467,221']],66,164,1144,360,[834,310],29);tx(s,'表頭核對：serial number 與 total price NTD 同時符合',68,549,1142,54,27,{color:C.muted});foot(s,'去重比較全部原始欄位，包含總價、排除編號',{bold:true,size:29});}
{const s=slide(6);table(s,[['原始內容','實際轉換'],['數字 1,200.5','移除逗號，轉成數值 1200.5'],['112年8月11日','2023-08-11，再拆年、月、季'],['十二層／地下二層','12／−2，另保留地下室等標記'],['地址、交易筆棟數','拆地區道路、土地建物車位數量']],66,162,1144,377,[445,699],29);foot(s,'保留缺失標記；需要學習的填補、尺度與編碼只從訓練資料建立',{bold:true,size:26});}
{const s=slide(7);pair(s,166,'保留的線索','面積、用途、樓層、地區、交易時間\n部分模型補充不含價格的道路位置');pair(s,341,'排除的欄位','編號與總價不當特徵\n完整備註、解約情形、不可用或常數欄位');foot(s,'87 欄基礎特徵 + 8 欄地理資訊 = 95 欄，其中 23 欄為類別',{size:28});}
{const s=slide(8);tx(s,'每個時間起點各自建立完整方法',66,160,1144,65,35,{bold:true,color:C.blue});table(s,[['學習資料','評分交易'],['截至 2021-08-10','2021-08-11 至 2022-08-10'],['截至 2022-08-10','2022-08-11 至 2023-08-10'],['截至 2023-08-10','2023-08-11 至 2024-08-10'],['截至 2023-02-10','2023-02-11 至 2024-02-10']],66,256,1144,294,[464,680],27);foot(s,'模型、編碼、先驗與價格參考，都只使用當期學習資料',{bold:true,size:28});}
{const s=slide(9);pair(s,173,'樹模型','由資料學習條件與門檻\n許多小樹逐步修正誤差');tx(s,'CatBoost、LightGBM',412,305,790,55,28,{color:C.muted});pair(s,415,'神經網路','數值與類別轉成可學習的表示\n經多層計算，學習條件如何一起影響價格');foot(s,'不同學法有機會互補，是否有效仍要看完整回測',{size:29});}
{const s=slide(10);tx(s,'總價 = 單價 × 有效面積',66,156,1144,66,39,{bold:true,color:C.blue});table(s,[['教學假設','單價估錯','總價誤差'],['50 平方公尺','每平方公尺 1 萬元','50 萬元'],['100 平方公尺','每平方公尺 1 萬元','100 萬元']],66,270,1144,235,[350,444,350],30);tx(s,'面積權重讓單價誤差對齊總價差距',67,550,1138,57,31,{bold:true,color:C.blue});foot(s,'另外保留直接總價與對數總價學法，最後都換回元',{size:27,color:C.muted,y:626});}
{const s=slide(11);tx(s,'相對日期權重示意，尚未正規化',66,153,1144,55,30,{color:C.muted});bar(s,['最新日期','早一年','早兩年'],[1,0.5,0.25],{y:238,h:296,max:1.15,format:'0.00'});foot(s,'365 日半衰期：改變學習時的影響大小，價格本身不乘權重',{bold:true,size:29});}
{const s=slide(12);tx(s,'參考單價 × 有效面積 × 網路調整倍率',66,151,1144,67,35,{bold:true,color:C.blue});pair(s,255,'參考單價查表','地區 + 交易類型 + 建物類型\n資料不足時逐層放寬查詢');pair(s,423,'按建案分 5 組','訓練每一組只使用其他四組的標籤\n這是分組 OOF，並非時間分折');foot(s,'網路固定 30 輪，日期加權 SmoothL1 的轉折為 10 萬元',{size:27});}
{const s=slide(13);tx(s,'六個模型先換回總價元，再組成三條分支',66,152,1144,65,34,{bold:true,color:C.blue});table(s,[['分支','主要部分 75%','共同部分 25%'],['單價','單價 CatBoost','同一個單價 LightGBM'],['神經網路','面積 NN 與參考單價 NN 先平均','同一個單價 LightGBM'],['總價／對數','總價與對數 CatBoost 先平均','同一個單價 LightGBM']],66,259,1144,287,[212,528,404],27);foot(s,'教學例：1,000 萬 × 75% + 1,200 萬 × 25% = 1,050 萬',{size:28});}
{const s=slide(14);pair(s,174,'成交參考','同建案、地址或道路的可比成交\n結合物件條件與當期學習的時間趨勢');pair(s,367,'空間修正','參考較弱時，補充附近且相容的交易資訊\n同時使用原初估與成交修正結果，支持不足就回退');foot(s,'三條分支各自完成修正，價格參考只使用當期學習資料',{bold:true,size:28});}
{const s=slide(15);tx(s,'同一間房的三個修正後估價，單位：萬元（教學假設）',66,150,1144,61,28,{color:C.muted});bar(s,['分支 A','分支 B','分支 C'],[1000,1100,1600],{y:242,h:300,max:2000,points:[{idx:1,fill:C.orange}]});foot(s,'逐筆取中間值 1,100 萬元，減少單一極端輸出的影響',{bold:true,size:29});}
{const s=slide(16);tx(s,'完整方法的總價 MAE，單位：元，越低越好',66,150,1144,55,29,{color:C.muted});table(s,[['評分期間','基準方法','此方法','差值'],['2021–2022 八月起點','2,309,352.36','2,310,236.02','+883.66'],['2022–2023 八月起點','1,407,387.07','1,405,971.19','−1,415.88'],['2023–2024 八月起點','1,383,228.41','1,381,702.87','−1,525.54'],['2023–2024 二月起點','1,336,410.67','1,336,107.62','−303.06']],66,239,1144,315,[368,260,260,256],25);foot(s,'差值用未取整 MAE 計算。負值改善、正值退步，四窗非獨立盲測',{size:26});}
{const s=slide(17);tx(s,'最近年度按建案分群的 MAE 差值（元）',66,151,1144,55,29,{color:C.muted});table(s,[['群別','此方法 − 基準','判讀'],['既有建案','+213.10','小幅退步'],['新建案','−2,841.61','改善']],66,234,1144,213,[420,390,334],30);tx(s,'另一對照方法：離線較好，Public 卻退步至 208.30 萬元',68,488,1139,76,30,{bold:true,color:C.orange});foot(s,'最早年亦退步，原強推薦條件未通過；重抽樣區間跨 0',{size:27,color:C.muted});}
{const s=slide(18);tx(s,'Public MAE',66,153,1144,55,33,{color:C.blue});tx(s,'199.58 萬元',66,222,1137,108,79,{bold:true,color:C.blue});tx(s,'精確值 1,995,836.2136 元，依提交者文字回報',68,369,1140,58,28,{color:C.muted});tx(s,'相對基準降低約 0.1367%，幅度有限',68,458,1140,66,32);foot(s,'Public 約 30%；其餘約 70% 的 Private 成績仍未知',{bold:true,size:31});}
{const s=slide(19);tx(s,'固定設定另行診斷，只看參考單價神經網路',67,137,1141,49,28,{bold:true,color:C.blue});
  tx(s,'未\n加\n權\n總\n價\n誤\n差',35,258,48,285,27,{color:C.body,align:'center'});
  // Excel stores 15 significant decimal digits. Retain all 60 raw values in source_data,
  // and make only this explicit sub-micro-TWD conversion for the editable workbook.
  const excelValue=v=>Number(v.toPrecision(15));
  const c=s.charts.add('scatter',{position:{left:94,top:202,width:1115,height:367},series:[{name:'訓練資料',xValues:curve.epochs,values:curve.train_mae_twd.map(excelValue),valuesFormatCode:'0',line:{fill:C.blue,width:4,style:'solid'},marker:{symbol:'none'}},{name:'時間驗證',xValues:curve.epochs,values:curve.validation_mae_twd.map(excelValue),valuesFormatCode:'0',line:{fill:C.orange,width:4,style:'solid'},marker:{symbol:'none'}}],scatterOptions:{style:'line'},hasLegend:true,legend:{position:'top',overlay:false,textStyle:{typeface:FONT,fontSize:28,fill:C.body}},chartFill:'none',plotAreaFill:'none',xAxis:{title:{text:'訓練輪數 Epoch',textStyle:{typeface:FONT,fontSize:26,fill:C.body}},min:1,max:30,majorUnit:5,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:25,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:0,max:2500000,majorUnit:500000,numberFormatCode:'0.0,,',textStyle:{typeface:FONT,fontSize:27,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(c,{fontFamily:FONT});
  tx(s,'百萬元',88,189,158,42,24,{color:C.muted});
  tx(s,'固定 30 輪，未依最低點選輪數；輪數不代表模型容量或 VC 維度',66,588,1144,49,26,{bold:true,color:C.blue});
  tx(s,'兩線均關閉 dropout 計算完整資料 MAE；曲線不能證明沒有過擬合',66,642,1144,41,24,{color:C.muted});}
{const s=slide(20);pair(s,167,'467,221 筆','最終有答案資料\n只新增一次參考單價 NN 訓練');pair(s,341,'47,073 筆','每筆測試交易均有預測\n檢查編號、順序、整數價格及檔案雜湊');foot(s,'新 NN 重載輸出一致；乾淨環境全部六模型重現仍未驗證',{size:28,color:C.orange});}
{const s=slide(21);tx(s,'先整理資料，再學習估價規則',66,176,1144,83,43,{bold:true,color:C.ink});tx(s,'六模型形成三分支，各自修正後取中位數',68,322,1141,94,36,{color:C.blue});tx(s,'時間驗證看完整流程，逐輪曲線看單一網路',68,459,1141,90,33);foot(s,'Public 有小幅改善，Private 未知，保留退步與反例',{bold:true,size:31});}
{const s=slide(22);tx(s,'全部符號皆代表已還原成元的總價',66,151,1144,57,29,{bold:true,color:C.blue});tx(s,['B_unit = .75*C_unit + .25*L_unit','B_nn   = .75*(.5*N_area + .5*N_prior) + .25*L_unit','B_log  = .75*(.5*C_raw  + .5*C_log)   + .25*L_unit','','U = S(B_unit, R(B_unit))','N = S(B_nn,   R(B_nn))','H = S(B_log,  R(B_log))','','prediction = median(N, H, U)'].join('\n'),68,238,1140,324,27,{mono:true,color:C.ink});foot(s,'R 為成交參考與時間趨勢，S 同時讀取初估與 R 結果',{size:28});}
{const s=slide(23);table(s,[['參考單價 NN','固定設定'],['隱藏層','512、256、128，LayerNorm 與 ReLU'],['loss','日期加權 SmoothL1，beta 0.1 百萬元'],['epochs／batch／seed','30／2,048／42，使用最後一輪'],['dropout／category dropout','0.1／0.1'],['optimizer','AdamW，學習率 0.001，weight decay 0.0001']],66,171,1144,390,[455,689],25);foot(s,'基準 NN 使用日期加權 MAE；此批未同期配對重訓基準',{size:27,color:C.orange});}
{const s=slide(24);table(s,[['名詞','白話意思'],['特徵／標籤','房屋條件／已知成交總價'],['訓練／推論','從有答案的資料學習／用保存規則估價'],['先驗／分組 OOF','參考單價／只用其他組建立自己的先驗'],['權重','每筆誤差在學習時的影響大小'],['過擬合','太貼合看過的資料，新交易可能失準'],['中位數','數字由小到大排序後的中間值']],66,173,1144,420,[380,764],28);}

if(p.slides.items.length!==24)throw Error('Wrong count');
const han=s=>(s.match(/[\u4e00-\u9fff]/g)||[]).length;
const count=slides.filter(s=>!s.is_appendix).reduce((a,s)=>a+han(s.script),0);
const meta={mainSlideCount:21,appendixSlideCount:3,mainHan:count,estimatedMinutesAt250:count/250,estimatedMinutesAt275:count/275,estimatedMinutesAt300:count/300,humanRehearsalPerformed:false,countScope:'主講 script 的漢字，不含標題、英文、數字、標點、補充理由、附錄、問答'};
await fs.mkdir(QA,{recursive:true});await fs.mkdir(OUT,{recursive:true});
await fs.writeFile(QA+'/narrative.json',JSON.stringify({meta,slides},null,2));
await fs.writeFile(QA+'/presentation.json',JSON.stringify(p.toProto()));
await fs.writeFile(QA+'/source_data.json',JSON.stringify({curveSource:'outputs/v25_training_curve_20261010/curve.json',curveSHA256:curveSHA,curveStoredUnit:'TWD',curveDisplayUnit:'million_TWD',rawCurveValuesPreservedExactlyInThisFile:true,editableWorkbookPrecision:'15 significant digits required by Excel',maximumCurveWorkbookChangeTWD:Math.max(...[...curve.train_mae_twd,...curve.validation_mae_twd].map(v=>Math.abs(v-Number(v.toPrecision(15))))),curve,sourceContent:'src/readable_ppt_content_20261012.json',originalNarrative:'outputs/v25_presentation_20261010/source/narrative.json',sourceDeck:'deliverables/v25_簡報與訓練曲線_20261010/臺灣房價預測_v25_上台簡報.pptx',privateScore:null,completeCleanSixModelReproduction:false,nativeChartSlides:[11,15,19],nativeTableSlides:[3,4,5,6,8,10,13,16,17,23,24],meta},null,2));
const main=slides.filter(s=>!s.is_appendix);
const timing=main.map(s=>{const sec=han(s.script)/275*60;return {slide:s.number,title:s.title,han:han(s.script),estimatedSeconds:sec};});
const duration=n=>`${Math.floor(n)}分${Math.round((n%1)*60)}秒`;
let md=`# 完整口語講稿\n\n台灣房屋銷售價格預測挑戰賽\n\n姓名：廖子頤　學號：315581048\n\n主講為第1至21頁，第22至24頁供問答使用，不計入主講。正文共${count.toLocaleString('en-US')}個漢字，275字／分估計${duration(count/275)}；250至300字／分約${duration(count/300)}至${duration(count/250)}。這是字數配速估算，尚未本人計時排練。英文字詞及圖表停頓可能增加時間。\n\n朗讀各頁「口語正文」即可。標題、配速提示、補充理由與附錄不必在主講中朗讀。\n\n`;
let elapsed=0;
for(const s of slides){const sec=s.is_appendix?0:han(s.script)/275*60;const fmt=v=>`${String(Math.floor(v/60)).padStart(2,'0')}:${String(Math.round(v%60)).padStart(2,'0')}`;md+=`## ${String(s.number).padStart(2,'0')}　${s.title}\n\n`+(s.is_appendix?'附錄，不計主講時間。\n\n':`配速提示：約${Math.round(sec)}秒，累計${fmt(elapsed+sec)}。\n\n`)+`### 口語正文\n\n${s.script}\n\n### 補充理由\n\n${s.why}\n\n`;elapsed+=sec;}
md+=`## 主講配速表\n\n| 頁 | 主題 | 漢字數 | 估計秒數 |\n| --- | --- | ---: | ---: |\n`+timing.map(t=>`| ${t.slide} | ${t.title} | ${t.han} | ${Math.round(t.estimatedSeconds)} |`).join('\n')+'\n';
await fs.writeFile(OUT+'/完整口語講稿.md',md);
await (await PresentationFile.exportPptx(p)).save(QA+'/candidate.pptx');
console.log(JSON.stringify({candidate:QA+'/candidate.pptx',slides:24,meta}));
