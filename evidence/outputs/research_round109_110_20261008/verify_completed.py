"""Post-evaluation CPU audit only; never fit or change frozen research artifacts.

Run --self-check before evaluation, then --evaluation-complete only after the
root has confirmed evaluation execution has exited successfully. The latter
mode refuses to deserialize query prices until every required completion
receipt, model seal and source/input/protected SHA has passed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import round100_contract as c

FOLDER = 'outputs/research_round109_110_20261008/'
DRIVER = 'src/round109_110_research.py'
TRAINER = 'src/round109_tail_parameter_average_train.py'
HELPER = 'src/round110_target_borders_unit.py'
PLAN = 'reports/round109_110_plan.md'
LAYERS = {'native', 'base', 'reference', 'spatial', 'median'}


def same(left, right, name):
    c.need(left == right, name)


def locked(path):
    record = c.record(path)
    sidecar = c.safe(path).with_suffix('.sha256')
    c.need(sidecar.read_text(encoding='ascii').strip() == record['sha256'],
           'Receipt sidecar differs: ' + str(path))
    return c.read(path), record


def artifact(record):
    same(c.record(record['path']), {k: record[k] for k in ('path', 'sha256')},
         'Artifact changed: ' + record['path'])


def manual_comparison(control, candidate, control_name):
    """Rebuild all comparison arithmetic and adverse lists without driver gates."""
    def delta(a, b):
        return {'control_mae': a, 'candidate_mae': b, 'delta': b - a,
                'relative_change_pct': 100 * (b - a) / a if a else None}

    result = {k: delta(control[k], candidate[k]) for k in ('overall_mae', 'weighted_mae')}
    adverse = []
    for section in ('strata', 'forecast_quarters', 'months', 'cities', 'relative_forecast_months'):
        same(set(control[section]), set(candidate[section]), 'Group names changed')
        result[section] = {}
        for name, a in control[section].items():
            b = candidate[section][name]
            same(a['rows'], b['rows'], 'Group rows changed')
            if a['rows']:
                item = {'rows': a['rows'], **delta(a['mae'], b['mae'])}
                result[section][name] = item
                if (section == 'forecast_quarters' or a['rows'] >= 500) and b['mae'] > 1.05 * a['mae']:
                    adverse.append({'section': section, 'group': name, **item})
    result['price_bands'] = []
    same(len(control['price_bands']), len(candidate['price_bands']), 'Price band count changed')
    for a, b in zip(control['price_bands'], candidate['price_bands']):
        same((a['band'], a['rows']), (b['band'], b['rows']), 'Price band support changed')
        item = {'band': a['band'], 'rows': a['rows'],
                **(delta(a['mae'], b['mae']) if a['rows'] else {})}
        result['price_bands'].append(item)
        if a['rows'] >= 500 and b['mae'] > 1.05 * a['mae']:
            adverse.append({'section': 'price_bands', **item})
    result['both_point_metrics_improve'] = all(candidate[k] < control[k] for k in ('overall_mae', 'weighted_mae'))
    result['quarter_or_large_group_deterioration_above_5pct'] = adverse
    if control_name == 'v22':
        result['stage_research_gate_passed'] = result['both_point_metrics_improve'] and not adverse
    result.update(control=control_name, candidate='T', used_for_research_gate=control_name == 'v22')
    return result


def manual_gates(stages):
    latest = stages['confirmation']['metrics']
    all_metrics = [v['metrics'] for v in stages.values()]
    no_adverse = all(not v['comparison']['quarter_or_large_group_deterioration_above_5pct']
                     for v in stages.values())
    fraction = (latest['v22']['overall_mae'] - latest['T']['overall_mae']) / latest['v22']['overall_mae']
    strong = {'all_three_overall_improve': all(m['T']['overall_mae'] < m['v22']['overall_mae'] for m in all_metrics),
              'all_three_weighted_improve': all(m['T']['weighted_mae'] < m['v22']['weighted_mae'] for m in all_metrics),
              'confirmation_at_least_half_percent': fraction >= .005,
              'no_quarter_or_large_group_deterioration_above_5pct': no_adverse}
    trial = {'latest_both_no_worse': all(latest['T'][k] <= latest['v22'][k] for k in ('overall_mae', 'weighted_mae')),
             'each_metric_each_window_within_quarter_percent': all(
                 m['T'][k] <= 1.0025 * m['v22'][k] for m in all_metrics for k in ('overall_mae', 'weighted_mae')),
             'at_least_two_windows_overall_improve': sum(m['T']['overall_mae'] < m['v22']['overall_mae'] for m in all_metrics) >= 2,
             'no_large_group_or_quarter_above_five_percent': no_adverse}
    return {'components': trial, 'strong_components': strong,
            'exploratory_submission_eligible': all(trial.values()),
            'strong_research_gate_passed': all(strong.values()),
            'confirmation_fractional_improvement': fraction}


def checkpoint_cpu(path):
    """CPU weights-only inspection in the existing Torch venv; no inference."""
    command = [str(ROOT / '.venv_torch/Scripts/python.exe'), '-B', '-X', 'utf8',
               str(Path(__file__).resolve()), '--nn-checkpoint', str(c.safe(path))]
    execution = subprocess.run(command, cwd=ROOT, text=True, encoding='utf8',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    c.need(execution.returncode == 0, 'CPU checkpoint inspection failed: ' + execution.stderr)
    return json.loads(execution.stdout)


def inspect_checkpoint(path):
    import torch
    import round63_recency_weighted_prior_train as baseline
    baseline.check_configuration()
    payload = torch.load(c.safe(path), weights_only=True, map_location='cpu')
    c.need(set(payload) == {'configuration', 'numeric_count', 'cardinalities',
                           'parameter_selection', 'parameter_state_epochs', 'state_dict'},
           'Unexpected checkpoint schema')
    same(payload['configuration'], baseline.CONFIG, 'Saved checkpoint versus actual R63 recipe')
    state = payload.pop('state_dict')
    c.need(isinstance(state, dict) and state, 'Missing saved state')
    parameters = {}
    for name, value in state.items():
        c.need(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
               and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
               'Invalid CPU checkpoint parameter')
        parameters[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                            'values_sha256': c.values_sha(value.numpy())}
    print(json.dumps({'payload': payload, 'named_parameter_order': list(state),
                      'parameters': parameters, 'parameter_count': sum(v.numel() for v in state.values()),
                      'CPU_only': True, 'model_inference_performed': False}, allow_nan=False))


def check_nn(s, st, meta, spec, arrays):
    # Read the hash-bound saved baseline config; the CPU subprocess additionally
    # compares each actual checkpoint with the imported guarded R63 recipe.
    parent = c.safe('src/round63_recency_weighted_prior_train.py')
    original_metadata = f'outputs/round63/{s}/model/training.json'
    original_sha = c.read('outputs/round63/results.json')['stages'][s]['artifact_sha256'][original_metadata]
    same(c.sha(original_metadata), original_sha, 'Original R63 metadata authority')
    expected = c.read(original_metadata)['configuration']
    same(meta['configuration'], expected, 'R109 training configuration differs')
    same(meta['source_sha256'], c.sha(TRAINER), 'R109 source differs')
    same(meta['parent_trainer_source_sha256'], c.sha(parent), 'R109 parent source differs')
    same(meta['architecture_source_sha256'], c.sha('src/round47_nn_train.py'), 'Architecture source differs')
    same(meta['input_sha256'], spec['prepared']['sha256'], 'R109 NPZ source differs')
    same(meta['fit_rows'], c.SPLITS[s][2], 'R109 fit rows')
    same(meta['query_rows'], c.SPLITS[s][3], 'R109 query rows')
    c.need(meta['complete_R63_training_configuration_unchanged'] and meta['changed_configuration_fields'] == []
           and meta['added_configuration_fields'] == [] and meta['training_objective_name'] == 'weighted MAE',
           'R109 changed baseline training')
    c.need(meta['saved_model_reload_exact'] and meta['saved_epoch30_model_reload_exact']
           and meta['reload_rows'] == c.SPLITS[s][3]
           and not meta['query_labels_used'] and not meta['query_labels_available'], 'R109 scope/reload')
    for key in ('train_sample_weight', 'train_price_factor_million', 'query_price_factor_million'):
        same(meta[key + '_sha256'], c.values_sha(arrays[key]), 'R109 factor/date weight ' + key)
    average = meta['tail_parameter_average']
    same(average['epochs'], list(range(21, 31)), 'Average epochs')
    same(average['collected_epochs'], list(range(21, 31)), 'Collected epochs')
    same((average['count'], average['collected_count'], average['module_buffer_count']), (10, 10, 0), 'Average count/buffers')
    same(average['weights'], [.1] * 10, 'Average weights')
    same((average['accumulator_device'], average['accumulator_dtype'], average['output_parameter_dtype']),
         ('cpu', 'float64', 'float32'), 'Average precision')
    c.need(average['parameters_averaged_before_inference'] and not average['predictions_averaged']
           and not average['optimizer_or_scheduler_changed'] and average['one_training_trajectory_for_candidate_and_control'],
           'Average semantics differ')
    same([e['epoch'] for e in meta['epochs']], list(range(1, 31)), 'Thirty complete epochs')
    for e in meta['epochs']:
        same(e['tail_parameter_state_collected'], e['epoch'] >= 21, 'Snapshot collection timing')
        same(e['tail_parameter_average_count'], max(0, e['epoch'] - 20), 'Snapshot count timeline')
    records = {}
    same({v['path'] for v in st['sealed_models']},
         {f'outputs/round109/{s}/model/model.pt', f'outputs/round109/{s}/model/model_epoch30.pt'},
         'Both expected NN checkpoints directly sealed')
    for key, epoch in (('averaged_candidate', list(range(21, 31))), ('epoch30_control', [30])):
        item = meta[key]
        prefix = f'outputs/round109/{s}/model/'
        model_path, pred_path = prefix + item['model_file'], prefix + item['predictions_file']
        same(c.sha(model_path), item['model_sha256'], 'NN checkpoint file SHA')
        same(c.sha(pred_path), item['predictions_file_sha256'], 'NN native file SHA')
        c.need(item['saved_model_reload_exact'] and item['reload_rows'] == c.SPLITS[s][3], 'NN complete reload')
        same(item['parameter_state_epochs'], epoch, 'NN selected epochs')
        saved = checkpoint_cpu(model_path)
        payload = saved['payload']
        same(payload['configuration'], expected, 'NN saved configuration')
        same(payload['parameter_state_epochs'], epoch, 'NN saved state epochs')
        same(payload['parameter_selection'], item['parameter_selection'], 'NN saved state selection')
        same((payload['numeric_count'], payload['cardinalities']), (meta['numeric_count'], meta['cardinalities']), 'NN saved shape metadata')
        same(saved['named_parameter_order'], average['named_parameter_order'], 'NN saved parameter order')
        same({k: v['shape'] for k, v in saved['parameters'].items()}, average['parameter_shapes'], 'NN saved parameter shapes')
        same(saved['parameter_count'], meta['parameter_count'], 'NN parameter count')
        layer = st['layers'] if key == 'averaged_candidate' else st['same_trajectory_epoch30_layers']
        c.exact(np.load(c.safe(pred_path), allow_pickle=False), c.load_array(layer['native']), 'NN native/layer identity')
        records[key] = {'model': c.record(model_path), 'native_predictions': c.record(pred_path), 'CPU_checkpoint': saved}
    return records


def check_ctr(s, st, meta, spec, ctx, weights):
    import pandas as pd
    from catboost import CatBoostRegressor
    import round110_target_borders_unit as helper
    prefix = f'outputs/round110/{s}/model/'
    state = c.read(prefix + 'feature_state.json')
    same([v['path'] for v in st['sealed_models']], [prefix + 'model.cbm'], 'Expected CBM directly sealed')
    helper._check_state(state, ctx['fit_X'], meta['categorical_columns'])
    helper._check_state(state, ctx['query_X'], meta['categorical_columns'])
    same(meta['parameters'], helper.parameters(), 'R110 declared parameters')
    same(meta['baseline_parameters'], helper.baseline.parameters(), 'R110 baseline recipe')
    same(meta['changed_parameter_names'], ['ctr_target_border_count'], 'R110 changed parameter')
    same(meta['added_parameter_names'], ['ctr_target_border_count'], 'R110 added parameter')
    same(meta['source_sha256'], helper.source_hashes(), 'R110 source identity')
    for filename, key in (('model.cbm', 'model_sha256'), ('feature_state.json', 'feature_state_sha256'),
                          ('raw.npy', 'raw_prediction_sha256'), ('native.npy', 'prediction_sha256'),
                          ('target_border_summary.json', 'target_border_summary_sha256')):
        same(c.sha(prefix + filename), meta[key], 'R110 saved artifact SHA ' + filename)
    c.need(meta['saved_model_reload_exact'] and meta['raw_and_TWD_reload_exact'] and meta['reload_rows'] == len(ctx['query_X'])
           and not meta['query_labels_used'] and not meta['fit_uses_eval_set'] and not meta['early_stopping_used'], 'R110 scope/reload')
    dates = helper.baseline._date_values(weights, ctx['fit_X'])
    same(meta['fit_date_weights_sha256'], c.values_sha(dates), 'R110 date weights')
    y = c.load_array(spec['fit_labels']['fit_y'])
    same(meta['fit_target_values_sha256'], helper.original.values_sha256(y), 'R110 fit target')
    same(meta['fit_transformed_target_values_sha256'], helper.original.values_sha256(helper.original.transform_target(y, ctx['fit_X'])), 'R110 transformed fit target')
    same(meta['fit_sample_weights_sha256'], helper.original.values_sha256(helper.sample_weights(ctx['fit_X'], weights)), 'R110 area/date weights')
    tree = pd.read_pickle(c.safe(spec['tree_geo']['path']))
    G = tree.iloc[ctx['indices']['fit_indices']].reset_index(drop=True)
    Gq = tree.iloc[ctx['indices']['query_indices']].reset_index(drop=True)
    frames = {'fit_X': ctx['fit_X'], 'query_X': ctx['query_X'], 'fit_geo': G, 'query_geo': Gq,
              'fit_augmented': helper.original.augment_features(ctx['fit_X'], G),
              'query_augmented': helper.original.augment_features(ctx['query_X'], Gq)}
    same({k: helper.base.frame_sha(v) for k, v in frames.items()}, meta['frame_sha256'], 'R110 fit/query frame identity')
    model = CatBoostRegressor()
    model.load_model(str(c.safe(prefix + 'model.cbm')))
    actual = helper.check_model(model, state)
    same(actual, meta['reloaded_native_diagnostics'], 'Loaded R110 native model diagnostics')
    same(helper.compare_reload_diagnostics(meta['fit_native_diagnostics'], actual), meta['reload_diagnostics_comparison'], 'R110 serialization rounding evidence')
    saved_summary = c.read(prefix + 'target_border_summary.json')
    same(saved_summary, meta['target_border_summary'], 'R110 target-index summary identity')
    with tempfile.TemporaryDirectory(prefix='r110_review_cpu_json_') as directory:
        path = Path(directory) / 'model.json'
        model.save_model(str(path), format='json')
        with path.open(encoding='utf8') as handle:
            reproduced = helper.summarize_target_borders(json.load(handle), int(model.tree_count_))
        same(reproduced, {k: saved_summary[k] for k in reproduced}, 'R110 actual native target-index counts')
        # Export metadata may be formatted differently; retain this identity without requiring timing equality.
        import hashlib
        json_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    c.exact(np.load(c.safe(prefix + 'native.npy'), allow_pickle=False), c.load_array(st['layers']['native']), 'R110 native/layer identity')
    return {'loaded_native_diagnostics': actual, 'reproduced_target_border_summary': reproduced,
            'review_CPU_export_json_sha256': json_sha, 'saved_target_border_summary': c.record(prefix + 'target_border_summary.json')}


def check_bootstrap(item, folder, point):
    checked = []
    pair = item['paired']
    c.need(pair['control'] == 'v22' and pair['candidate'] == 'T' and pair['replicates'] == 2000,
           'Paired baseline/replicates')
    c.need('not a matched retraining control' in pair['caution'] and 'not Private probability' in pair['caution'], 'Paired caution')
    for family in ('project', 'month'):
        path = folder + family + '_samples.npy'
        boot = np.load(c.safe(path), allow_pickle=False)
        c.need(boot.dtype == np.float64 and boot.shape == (2000, 4), 'Bootstrap sample schema')
        for metric, left, right in (('overall', 0, 1), ('weighted', 2, 3)):
            differences = boot[:, right] - boot[:, left]
            valid = differences[np.isfinite(differences)]
            bounds = np.quantile(valid, [.025, .975], method='linear') if len(valid) else [None, None]
            summary = pair['methods'][family]['summary'][metric]
            expect = {'valid_replicates': len(valid), 'invalid_replicates': 2000 - len(valid),
                      'control_mae': float(point[left]), 'candidate_mae': float(point[right]),
                      'point_delta': float(point[right] - point[left]),
                      'percentile_2_5': bounds[0], 'percentile_97_5': bounds[1],
                      'negative_delta_fraction': float(np.mean(valid < 0)) if len(valid) else None,
                      'bootstrap_mean_delta': float(valid.mean()) if len(valid) else None}
            for k, v in expect.items():
                same(summary[k], v, 'Bootstrap summary ' + k)
        checked.append(c.record(path))
    return checked


def run():
    start = time.monotonic()
    output = FOLDER + 'postfit_review.json'
    c.need(not c.safe(output).exists() and not c.safe(output).with_suffix('.sha256').exists(), 'Preserve previous review')
    request, rr = locked(FOLDER + 'request.json')
    seal, sr = locked(FOLDER + 'seal.json')
    training, tr = locked(FOLDER + 'training_execution.json')
    evaluation, er = locked(FOLDER + 'evaluation_execution.json')
    same(training['exit_code'], 0, 'Training not successful')
    same(training['failure'], None, 'Training failure exists')
    same(evaluation['exit_code'], 0, 'Evaluation not successful')
    c.need(evaluation['training_already_finished_before_scoring'] and evaluation['query_labels_used_for_scoring_only']
           and evaluation['protected_unchanged'], 'Evaluation scope receipt')
    same(seal['status'], 'six_models_all_query_predictions_sealed_unscored', 'All six models not sealed')
    same(seal['request'], rr, 'Seal request identity')
    same(seal['query_scores_computed'], False, 'Seal unexpectedly scored')
    same(request['status'], 'frozen_before_six_fits', 'Request not frozen')
    same(request['new_fits'], 6, 'New fit count')
    same(request['plan'], c.record(PLAN), 'Frozen plan changed')
    c.verify_hashes(request['source_sha256'])
    c.verify_hashes(request['input_sha256'])
    c.verify_hashes(request['protected_before'])
    same(c.source_hashes(DRIVER, TRAINER, HELPER), request['source_sha256'], 'Recursive source closure changed')
    review = c.read(FOLDER + 'preflight_review.json')
    c.need(review['status'] == 'passed', 'Preflight not passed')
    for name in ('R109_average_selfcheck_execution.json', 'R110_helper_selfcheck_execution.json'):
        c.need(c.read(FOLDER + name)['exit_code'] == 0, 'Formal CPU selfcheck failure')
    import round109_110_research as driver
    same(driver.protected(), request['protected_before'], 'Protected file set/content changed')
    results, result_records = {}, {}
    for n in (109, 110):
        results[str(n)], result_records[str(n)] = locked(f'outputs/round{n}/results.json')
        same(results[str(n)]['status'], 'three_reused_windows_evaluated', 'Evaluation incomplete')
        same(results[str(n)]['request'], rr, 'Results request identity')
        same(results[str(n)]['seal'], sr, 'Results seal identity')
        same(results[str(n)]['Private_score'], None, 'Unknown Private must remain unknown')
    stage_records, trainmeta = {}, {}
    for n in (109, 110):
        for s in c.STAGES:
            st_record = seal['records'][str(n)][s]
            artifact(st_record)
            st = c.read(st_record['path'])
            artifact(st['model_metadata'])
            same(set(st['layers']), LAYERS, 'Stage layer schema')
            for record in st['sealed_models']:
                artifact(record)
            for record in st['layers'].values():
                c.load_array(record)
            if n == 109:
                same(set(st['same_trajectory_epoch30_layers']), LAYERS, 'Epoch30 layer schema')
                for record in st['same_trajectory_epoch30_layers'].values():
                    c.load_array(record)
            same((st['fit_rows'], st['query_rows']), c.SPLITS[s][2:], 'Stage row counts')
            same(st['query_labels_available'], False, 'Training stage query labels')
            stage_records[n, s] = st
            trainmeta[f'{n}/{s}'] = c.read(st['model_metadata']['path'])
    # Completion and all model/array identities have passed before labels are read.
    import pandas as pd
    import round104_composition as comp
    import round100_temporal_prior_evaluate_v2 as metric
    raw = pd.read_pickle(c.safe(c.RAW))
    artifacts, summary, configs, bootstraps = {}, {'109': {}, '110': {}}, {}, []
    score_count = 0
    for s in c.STAGES:
        spec = request['stages'][s]
        ctx = comp.load_context(spec['composition'])
        y = metric.select_query_y(raw, ctx['indices']['query_source_cache_indices'])
        labels = metric.methods.group_labels(ctx['fit_X'], ctx['query_X'])
        with np.load(c.safe(spec['prepared']['path']), allow_pickle=False) as npz:
            same(set(npz.files), set(c.ARRAY_NAMES), 'Exactly nine no-query-y arrays')
            arrays = {k: npz[k].copy() for k in npz.files}
        c.exact(arrays['train_y_million'], np.asarray(c.load_array(spec['fit_labels']['fit_y']) / 1e6, dtype=np.float32), 'NN fit labels')
        same(c.values_sha(arrays['train_sample_weight']), spec['date_weight_sha256'], 'Original R79 date weight identity')
        original_unit = driver.unit_chain(ctx, np.load(c.safe(spec['native_unit']['path']), allow_pickle=False))
        c.exact(original_unit['median'], ctx['original']['median'], 'Original unit final replay')
        for n in (109, 110):
            st = stage_records[n, s]
            meta = trainmeta[f'{n}/{s}']
            layers = {k: c.load_array(v) for k, v in st['layers'].items()}
            # Full CPU R/S replay checks every saved layer, including epoch30 control.
            chain = comp.apply(ctx, layers['native'])[0] if n == 109 else driver.unit_chain(ctx, layers['native'])
            for k in LAYERS:
                c.exact(chain[k], layers[k], 'Complete candidate chain replay: ' + k)
            if n == 109:
                controls = {k: c.load_array(v) for k, v in st['same_trajectory_epoch30_layers'].items()}
                replay = comp.apply(ctx, controls['native'])[0]
                for k in LAYERS:
                    c.exact(replay[k], controls[k], 'Complete epoch30 chain replay: ' + k)
                configs[f'{n}/{s}'] = check_nn(s, st, meta, spec, arrays)
            else:
                configs[f'{n}/{s}'] = check_ctr(s, st, meta, spec, ctx, arrays['train_sample_weight'])
            path = f'outputs/round{n}/{s}/evaluation/results.json'
            item, evaluation_record = locked(path)
            same(item, results[str(n)]['stages'][s], 'Per-stage versus aggregate evaluation')
            predictions = {'v22': ctx['original']['median'], 'v24': c.load_array(spec['v24']) if 'dtype' in spec['v24']
                           else np.load(c.safe(spec['v24']['path']), allow_pickle=False), 'T': layers['median']}
            if n == 109:
                predictions['epoch30'] = controls['median']
            for name, prediction in predictions.items():
                target = item['metrics'][name] if name != 'epoch30' else item['same_trajectory_epoch30_metrics']
                error = np.abs(prediction - y)
                same(float(error.mean()), target['overall_mae'], 'Independent TWD overall MAE')
                weighted = 0.
                for group, weight in c.WEIGHTS.items():
                    take = labels == group
                    same(int(take.sum()), target['strata'][group]['rows'], 'Fixed stratum support')
                    if take.any():
                        same(float(error[take].mean()), target['strata'][group]['mae'], 'Fixed stratum MAE')
                    if weight > 0:
                        c.need(take.any(), 'Positive-weight group absent')
                        weighted += weight * float(error[take].mean())
                same(weighted, target['weighted_mae'], 'Independent fixed-weight MAE')
                same(metric.descriptive_metrics(y, prediction, ctx['fit_X'], ctx['query_X'], c.SPLITS[s][0]),
                     target, 'Full descriptive metric replay')
                score_count += 1
            primary = manual_comparison(item['metrics']['v22'], item['metrics']['T'], 'v22')
            secondary = manual_comparison(item['metrics']['v24'], item['metrics']['T'], 'v24')
            same(primary, item['comparison'], 'Independent primary group/price-band comparisons')
            same(secondary, item['comparison_vs_v24'], 'Independent v24 group/price-band comparisons')
            if n == 109:
                same(manual_comparison(item['same_trajectory_epoch30_metrics'], item['metrics']['T'], 'epoch30'),
                     item['comparison_vs_same_trajectory_epoch30'], 'Epoch30 matched-control label/arithmetic')
                same(float(np.max(np.abs(controls['median'] - ctx['original']['median']))),
                     item['epoch30_vs_saved_v22_max_abs_prediction_difference_TWD'], 'Matched versus saved control distinction')
            for name, prediction in layers.items():
                if name != 'median':
                    same(metric.descriptive_metrics(y, prediction, ctx['fit_X'], ctx['query_X'], c.SPLITS[s][0]),
                         item['branch_metrics'][name], 'Branch metrics ' + name)
            paired_errors = np.abs(y[:, None] - np.column_stack([predictions['v22'], predictions['T']]))
            point = metric.methods.point_values(paired_errors, labels, c.WEIGHTS)
            bootstraps.extend(check_bootstrap(item, f'outputs/round{n}/{s}/evaluation/', point))
            artifacts[f'{n}/{s}'] = {'stage': c.record(f'outputs/round{n}/{s}/stage.json'),
                                    'training_metadata': st['model_metadata'], 'evaluation': evaluation_record,
                                    'sealed_models': st['sealed_models']}
            summary[str(n)][s] = {'overall_mae': item['metrics']['T']['overall_mae'],
                'weighted_mae': item['metrics']['T']['weighted_mae'],
                'overall_delta_vs_v22': primary['overall_mae']['delta'], 'weighted_delta_vs_v22': primary['weighted_mae']['delta'],
                'overall_delta_vs_v24': secondary['overall_mae']['delta'], 'weighted_delta_vs_v24': secondary['weighted_mae']['delta'],
                'adverse_groups': primary['quarter_or_large_group_deterioration_above_5pct'],
                'v24_adverse_groups': secondary['quarter_or_large_group_deterioration_above_5pct'],
                'project_overall_95pct': item['paired']['methods']['project']['summary']['overall'],
                'fit_seconds': meta['fit_seconds']}
            comp.verify_unchanged(ctx)
    verdicts = {}
    for n in (109, 110):
        verdict = manual_gates(results[str(n)]['stages'])
        gates = results[str(n)]['gates']
        same(verdict['components'], gates['exploratory_submission']['components'], 'Manual exploratory gate components')
        same(verdict['exploratory_submission_eligible'], gates['exploratory_submission']['passed'], 'Manual exploratory gate verdict')
        same(verdict['strong_components'], gates['promotion']['components'], 'Manual strong gate components')
        same(verdict['strong_research_gate_passed'], gates['promotion']['adoption_research_gate_passed'], 'Manual strong gate verdict')
        same(verdict['confirmation_fractional_improvement'], gates['promotion']['confirmation_fractional_improvement'], 'Manual strong improvement fraction')
        c.need(not gates['automatic_selection'] and gates['selected_submission'] == 'v22'
               and gates['private_score'] is None, 'Selection/Private changed')
        verdicts[str(n)] = verdict
    c.verify_hashes(request['source_sha256'])
    c.verify_hashes(request['input_sha256'])
    same(driver.protected(), request['protected_before'], 'Protected originals changed during review')
    payload = {'status': 'passed', 'reviewer': 'research_preflight_20261008', 'created_at_utc': c.now(),
        'scope': 'Independent post-evaluation CPU identity, full composition replay, actual saved model metadata and preregistered-rule review; no new fit or GPU work.',
        'request': rr, 'seal': sr, 'training_execution': tr, 'evaluation_execution': er,
        'results': result_records, 'artifacts': artifacts, 'trainmeta': trainmeta,
        'manual_gate_verdicts': verdicts, 'stage_summary': summary, 'model_identity_diagnostics': configs,
        'source_sha256': request['source_sha256'], 'review_tool': c.record(__file__), 'plan': request['plan'],
        'identity_checks': {'frozen_sources': len(request['source_sha256']), 'frozen_inputs': len(request['input_sha256']),
                            'protected_files': len(request['protected_before']), 'sealed_stages': 6,
                            'candidate_layers_replayed': 30, 'epoch30_layers_replayed': 15,
                            'NN_checkpoints_CPU_inspected': 6, 'native_CBM_loaded_CPU': 3,
                            'actual_native_target_index_summaries_reexported_CPU': 3},
        'scoring_checks': {'overall_and_weighted_MAE_independently_recomputed': score_count,
                           'primary_and_v24_comparisons_independently_reconstructed': 12,
                           'matched_epoch30_controls_correctly_labeled': 3, 'bootstrap_files_summary_checked': len(bootstraps)},
        'bootstrap_artifacts': bootstraps, 'query_labels_used_for_postseal_review_only': True,
        'new_model_fits': 0, 'GPU_work': False, 'Private_score': None, 'blocking_findings_remaining': [],
        'limits': ['NN checkpoints were loaded on CPU and inspected, without independent checkpoint model inference; full-query native reload evidence comes from frozen training.',
                   'The ten individual epoch states were not retained, so this audit cannot independently re-sum the actual trained states; it checks frozen implementation, collection metadata and saved states.',
                   'Descriptive metric definitions and composition helpers are frozen project implementations; independent overall/weighted arithmetic and gate reconstruction supplement their replay.',
                   'Bootstrap summaries are recomputed from saved replicates without resampling or correcting cumulative selection, refit uncertainty or future shift; they are not Private probabilities.',
                   'Reviewer authored R110 helper; its separate root and CTR-agent acceptance is recorded before freeze, while this audit independently inspects actual saved native models.',
                   'This review does not certify a subsequent full-data fit, test CSV, Kaggle upload or replacement of selected v22.'],
        'review_elapsed_seconds': time.monotonic() - start}
    payload['conclusion'] = 'Exploratory eligibility is determined separately for each round by manual_gate_verdicts; eligible candidates still require a separate full-data plan/freeze. v22 remains selected and Private is unknown.'
    c.write(output, payload)
    print(json.dumps({'status': 'passed', 'review': c.record(output), 'manual_gate_verdicts': verdicts}, ensure_ascii=False))


def self_check():
    def stages(a, b):
        return {s: {'metrics': {'v22': {'overall_mae': 100., 'weighted_mae': 100.},
                                'T': {'overall_mae': a[i], 'weighted_mae': b[i]}},
                    'comparison': {'quarter_or_large_group_deterioration_above_5pct': []}}
                for i, s in enumerate(c.STAGES)}
    strong = manual_gates(stages([99., 99., 99.], [99., 99., 99.]))
    c.need(strong['exploratory_submission_eligible'] and strong['strong_research_gate_passed'], 'Strong pass case')
    trial = manual_gates(stages([99.9, 99.9, 100.2], [99.9, 99.9, 100.2]))
    c.need(trial['exploratory_submission_eligible'] and not trial['strong_research_gate_passed'], 'Trial-only case')
    c.need(not manual_gates(stages([99., 100.1, 99.], [99., 99., 99.]))['exploratory_submission_eligible'], 'Latest regression case')
    c.need(not manual_gates(stages([99., 99., 100.3], [99., 99., 99.]))['exploratory_submission_eligible'], 'Quarter percent case')
    bad = stages([99., 99., 99.], [99., 99., 99.])
    bad['auxiliary']['comparison']['quarter_or_large_group_deterioration_above_5pct'] = [{'synthetic': True}]
    c.need(not manual_gates(bad)['exploratory_submission_eligible'], 'Adverse group case')
    import copy
    control = {'overall_mae': 100., 'weighted_mae': 100.,
               **{section: {'synthetic': {'rows': 500, 'mae': 100.}} for section in
                  ('strata', 'forecast_quarters', 'months', 'cities', 'relative_forecast_months')},
               'price_bands': [{'band': 'synthetic', 'rows': 500, 'mae': 100.}]}
    candidate = copy.deepcopy(control)
    for section in ('strata', 'forecast_quarters', 'months', 'cities', 'relative_forecast_months'):
        candidate[section]['synthetic']['mae'] = 105.
    candidate['price_bands'][0]['mae'] = 105.
    for name in ('v22', 'v24', 'epoch30'):
        compare = manual_comparison(control, candidate, name)
        same(compare['control'], name, 'Control name')
        same(compare['used_for_research_gate'], name == 'v22', 'Only v22 is primary gate')
        c.need(not compare['quarter_or_large_group_deterioration_above_5pct'], 'Exactly five percent allowed')
    candidate['cities']['synthetic']['mae'] = 105.1
    compare = manual_comparison(control, candidate, 'epoch30')
    same([(a['section'], a['group']) for a in compare['quarter_or_large_group_deterioration_above_5pct']],
         [('cities', 'synthetic')], 'Above five percent rejected')
    print(json.dumps({'status': 'passed', 'synthetic_gate_cases': 5, 'comparison_label_and_threshold_cases': 4, 'real_data_read': False,
                      'query_prices_or_new_scores_read': False, 'fits': 0, 'GPU': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-check', action='store_true')
    group.add_argument('--nn-checkpoint')
    group.add_argument('--evaluation-complete', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.nn_checkpoint:
        inspect_checkpoint(args.nn_checkpoint)
    else:
        run()
