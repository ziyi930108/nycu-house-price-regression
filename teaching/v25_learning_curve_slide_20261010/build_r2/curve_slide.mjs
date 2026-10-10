/** Native editable supplement. Source only until a reviewed real curve is supplied. */
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const need = (value, message) => { if (!value) throw new Error(message); };
export function validateCurveData(d) {
const contract = {
  format: 'v25_prior_nn_learning_curve_v1',
  component: 'R107_prior_NN_native',
  metric: 'unweighted_total_price_MAE',
  unit: 'TWD',
  evaluation_mode: 'eval',
  epoch_selection: 'fixed_last_epoch',
  query_labels_used_for_fitting: false,
};
for (const [key, value] of Object.entries(contract)) need(d[key] === value, 'Curve contract differs: ' + key);
for (const key of ['fit_period_label', 'query_period_label']) need(typeof d[key] === 'string' && d[key].length > 0 && d[key].length < 100, 'Expected short period label: ' + key);
for (const key of ['fit_rows', 'query_rows']) need(Number.isSafeInteger(d[key]) && d[key] > 0, 'Positive row count required: ' + key);
need(Array.isArray(d.epochs) && d.epochs.length === 30 && d.epochs.every((v, i) => v === i + 1), 'Exactly epochs 1 through 30 required.');
for (const key of ['train_mae_twd', 'validation_mae_twd']) {
  need(Array.isArray(d[key]) && d[key].length === 30 && d[key].every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0), 'Thirty finite nonnegative MAEs required: ' + key);
}
need(Array.isArray(d.sources) && d.sources.length > 0, 'At least one bound source required.');
for (const source of d.sources) need(typeof source.path === 'string' && /^[0-9a-f]{64}$/.test(source.sha256), 'Invalid curve source record.');
return contract;
}

export async function addLearningCurveSlide(p, d, options = {}) {
  validateCurveData(d);
  const SKILL = path.resolve(options.skillDir || 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
  const {resolvePresentationFont, applyPresentationChartFont} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const chosenFont = options.fontFamily || 'Microsoft JhengHei';
const FONT = resolvePresentationFont({fontFamily: chosenFont, availableFonts: [chosenFont]});
const C = {bg: '#FFFFFF', ink: '#122740', body: '#30435A', muted: '#64748B', train: '#2259CB', validation: '#AD4B2E', gray: '#BAC9DD', grid: '#DDE5F0'};
const s = p.slides.add();
s.background.fill = C.bg;
function tx(value, x, y, w, h, size, color = C.body, bold = false, name = 'body') {
  const t = s.shapes.add({name, geometry: 'textbox', position: {left: x, top: y, width: w, height: h}, fill: 'none', line: {fill: 'none', width: 0}});
  t.text = value;
  t.text.style = {typeface: FONT, fontSize: size, color, bold, alignment: 'left', verticalAlignment: 'top', autoFit: 'none', wrap: 'square', insets: {left: 0, right: 0, top: 0, bottom: 0}};
  return t;
}
tx('訓練輪數與時間驗證誤差', 68, 44, 1144, 77, 46, C.ink, true, 'slide-title');
tx('v25配方診斷重跑，先驗神經網路的原生分量', 70, 139, 1138, 40, 27, C.train, true);
tx('學習：' + d.fit_period_label + '（' + d.fit_rows.toLocaleString('en-US') + '筆）　時間驗證：' + d.query_period_label + '（' + d.query_rows.toLocaleString('en-US') + '筆）', 70, 184, 1138, 49, 21, C.muted);
const train = d.train_mae_twd.map(v => Number((v / 10000).toFixed(4)));
const validation = d.validation_mae_twd.map(v => Number((v / 10000).toFixed(4)));
const largest = Math.max(...train, ...validation);
const step = Math.max(10, 10 * Math.ceil(largest / 6 / 10));
const upper = Math.max(step, step * Math.ceil(largest / step));
const chart = s.charts.add('scatter', {
  position: {left: 70, top: 236, width: 1138, height: 338},
  series: [
    {name: '訓練資料', xValues: d.epochs, values: train, valuesFormatCode: '0.0', line: {fill: C.train, width: 3, style: 'solid'}, marker: {symbol: 'circle', size: 3, fill: C.train}},
    {name: '時間驗證', xValues: d.epochs, values: validation, valuesFormatCode: '0.0', line: {fill: C.validation, width: 3, style: 'solid'}, marker: {symbol: 'diamond', size: 3, fill: C.validation}},
  ],
  scatterOptions: {style: 'lineWithMarkers', varyColors: false},
  hasLegend: true,
  legend: {position: 'top', overlay: false, textStyle: {typeface: FONT, fontSize: 24, fill: C.body}},
  chartFill: 'none', plotAreaFill: 'none',
  xAxis: {title: {text: '訓練輪數 Epoch', textStyle: {typeface: FONT, fontSize: 23, fill: C.body}}, min: 1, max: 30, majorUnit: 5, numberFormatCode: '0', textStyle: {typeface: FONT, fontSize: 21, fill: C.muted}, line: {fill: C.gray, width: 1}, majorGridlines: null},
  yAxis: {title: {text: '總價 MAE（萬元）', textStyle: {typeface: FONT, fontSize: 23, fill: C.body}}, min: 0, max: upper, majorUnit: step, numberFormatCode: '0', textStyle: {typeface: FONT, fontSize: 22, fill: C.muted}, line: {fill: 'none', width: 0}, majorGridlines: {fill: C.grid, width: 1}},
  dataLabels: {showValue: false},
});
applyPresentationChartFont(chart, {fontFamily: FONT});
tx('兩線都用完整資料、關閉 dropout，計算不加權總價 MAE', 70, 587, 1138, 42, 26, C.train, true);
tx('固定30輪，原v25模型與CSV保持。訓練先驗為分組OOF，驗證先驗來自完整fit。', 70, 637, 1138, 35, 21, C.muted);
tx('尚未加入其他分支、中位數或成交參考', 70, 679, 1040, 27, 19, C.muted);
if (options.pageLabel) tx(String(options.pageLabel), 1150, 672, 60, 28, 19, C.muted);

const finalTrainWan = (d.train_mae_twd[29] / 10000).toFixed(2);
const finalValidationWan = (d.validation_mae_twd[29] / 10000).toFixed(2);
const spoken = '這一頁用v25配方另外做一次診斷重跑，補上老師要求的訓練與驗證曲線。原v25的模型和提交價格都保持，這不是它當時每一輪的原始軌跡。橫軸是同一個神經網路從第一輪到第三十輪的學習過程，模型大小一直相同。縱軸是平均總價誤差，單位是萬元，越低越好。藍線用模型學過的交易計算，橘線用時間較晚、沒有參與這期訓練的交易計算。每一輪都關閉 dropout，兩條線也都不用日期權重，讓誤差的單位一致。第三十輪時，訓練誤差約' + finalTrainWan + '萬元，時間驗證約' + finalValidationWan + '萬元。驗證後段有波動，沒有持續上升的典型U形，因此不能只憑這張圖斷言沒有過擬合。這張圖只看先驗神經網路的一個原生分量，還沒有加入其他模型、成交參考或中位數。另外，兩組的先驗建立方式不同，因此不能只看線與線的距離，就直接判定過擬合。原本的設定已固定三十輪，沒有看這張圖再挑最好的一輪。';
const sourceLines = d.sources.map(v => v.path + ' SHA256 ' + v.sha256);
s.speakerNotes.textFrame.setText(spoken + '\n\n資料與範圍\n' +
  'curve: ' + (options.curveSource?.path || 'caller-supplied reviewed curve') + ' SHA256 ' + (options.curveSource?.sha256 || 'see caller receipt') + '\n' +
  'fit_rows=' + d.fit_rows + '; query_rows=' + d.query_rows + '\n' +
  'v25配方的一次診斷重跑，非原v25每epoch實際軌跡，原選定模型與CSV不改。原始資料以TWD保存，圖表除以10000顯示萬元。固定30epoch，非模型容量/VC dimension曲線，非v25 median或Kaggle分數。\n' +
  '學習loss為日期加權SmoothL1，但兩條診斷線均為epoch末eval模式的全資料unweighted總價MAE。\n' + sourceLines.join('\n'));

  return {slide: s, chart, spoken, metricData: {epochs: d.epochs, train_mae_twd: d.train_mae_twd, validation_mae_twd: d.validation_mae_twd}};
}
