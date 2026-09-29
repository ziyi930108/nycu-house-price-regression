"""R106 isolated scorer; all six matched NN fits seal before query labels.

C retrains original independent category dropout; T changes only four-mask linkage.
Only mature metric/group/bootstrap functions are reused; no model or NPZ is loaded.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
import traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as metric

PLAN = 'reports/round106_plan.md'
PLAN_SHA = '2748f8f5cd76df02e99717a10a13f88186b1547d1196affb54bd603cb0351f51'
OLD = {'path': 'outputs/round102/sealed_manifest.json', 'sha256': 'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
OLD_RESULTS = {'path': 'outputs/round102/evaluation/results.json', 'sha256': 'e385ed630cdf3a2b2b0418867563a4064fbe08c1c979feb004787df2592f2347'}
R63_RESULTS = {'path': 'outputs/round63/results.json', 'sha256': '83779882a45379a955aa88197e03dd1d566262c1bc75bbeb0935d735bd387506'}
LAYERS = ('native', 'base', 'reference', 'spatial', 'median')
TRAINER = 'src/round106_joint_dropout_train.py'
TRAINER_SOURCES = (TRAINER, 'src/round63_recency_weighted_prior_train.py',
                   'src/round52_nn_train.py', 'src/round47_nn_train.py')
IDENTITY_COLUMNS = ['project_name', 'address_raw', 'building_and_unit', 'building_block']
REPLAY_FLAGS = ('original_v22_all_five_layers_replay_exact', 'R_S_state_and_frames_unchanged')
CAUTION = ('Adaptive single-seed research on reused windows, not blind or a causal guarantee. '
           'C retrains original independent category dropout; its deviation from saved v22 is disclosed. '
           'T changes only linkage of four category masks, not numerical prior or all identity proxies. '
           'Static 2023 geography is not fully historically as-of. Conditional bootstrap is not Private probability. '
           'Only v22 remains selected; no masking-rate, subgroup or blend search or automatic adoption.')


def parameters():
    p = c.evaluation_parameters(); p.pop('point_only_control')
    p.update(round=106, candidate_role='four_identity_categories_share_project_name_dropout_mask',
             mechanism_control_role='retrained_R63_independent_category_dropout_same_inputs',
             mechanism_required_metrics=['overall_mae', 'weighted_mae'], mechanism_required_layer='median',
             mechanism_no_quarter_or_large_group_deterioration_above_5pct=True,
             native_improvement_required=False, category_dropout_probability=.1,
             identity_columns=IDENTITY_COLUMNS, candidate_training_fits=6,
             C_retraining_bitwise_equality_to_saved_v22_required=False, fully_historically_as_of=False)
    return p


def source_hashes():
    return c.source_hashes(__file__, *TRAINER_SOURCES)


def research_gates(stages):
    primary = metric.research_gate(stages)
    parts = {'all_three_final_' + k + '_improve': all(
        stages[s]['metrics']['T'][k] < stages[s]['metrics']['C'][k] for s in c.STAGES)
        for k in ('overall_mae', 'weighted_mae')}
    parts['no_quarter_or_large_group_deterioration_above_5pct'] = not any(
        stages[s]['comparison_vs_C']['quarter_or_large_group_deterioration_above_5pct'] for s in c.STAGES)
    mechanism = {'components': parts, 'mechanism_gate_passed': all(parts.values()),
                 'control': 'retrained original R63 independent dropout',
                 'native_improvement_required': False, 'causal_identification_claimed': False}
    return {'primary': primary, 'mechanism': mechanism,
            'both_research_gates_passed': primary['adoption_research_gate_passed'] and all(parts.values()),
            'automatic_adoption': False, 'selected_submission': 'v22', 'Private_score': None, 'new_CSV': False}


def check_header(seal):
    c.need(seal['schema_version'] == 1 and seal['status'] == 'all_joint_dropout_predictions_sealed' and
           seal['stage_order'] == list(c.STAGES) and set(seal['stages']) == set(c.STAGES), 'Three complete stages required')
    c.need(seal['plan_sha256'] == PLAN_SHA and seal['new_NN_fits'] == 6 and
           seal['new_native_tree_fits'] == 0, 'Plan or six-NN counts differ')
    c.need(seal['prior_or_R_S_build_calls'] == seal['encoder_build_calls'] == 0, 'Learned input/state rebuild forbidden')
    c.need(seal['all_query_reload_exact'] is True and all(seal[k] is True for k in REPLAY_FLAGS), 'Replay/reload incomplete')
    c.need(all(seal[k] is False for k in ('query_labels_available', 'query_scores_computed', 'raw_price_pickle_opened')), 'Prediction label boundary differs')
    c.need(len(seal['protected_before']) == 9 and seal['protected_before'] == seal['protected_after'], 'Protected identities differ')


def check_training(stage, contract, training, launch):
    """Read certificates and hash payloads; never deserialize a model or training NPZ."""
    n, q = stage['outer_fit_rows'], stage['query_rows']; shared = stage['all_input_arrays']
    c.need(set(shared) == set(c.ARRAY_NAMES), 'Exactly original nine input arrays required')
    expected = {'train_num': ('float32', [n,144]), 'query_num': ('float32', [q,144]),
                'train_cat': ('int64', [n,23]), 'query_cat': ('int64', [q,23]), 'cardinalities': ('int64', [23]),
                'train_y_million': ('float32', [n]), 'train_price_factor_million': ('float32', [n]),
                'query_price_factor_million': ('float32', [q]), 'train_sample_weight': ('float32', [n])}
    for name, (dtype, shape) in expected.items():
        c.need(shared[name]['dtype'] == dtype and shared[name]['shape'] == shape, 'Input schema differs: ' + name)
    cats = contract['categorical_columns']; indices = contract['identity_indices']
    c.need(len(cats) == len(set(cats)) == 23 and contract['identity_columns'] == IDENTITY_COLUMNS and
           indices == [cats.index(k) for k in IDENTITY_COLUMNS], 'Named identity column positions differ')
    for arm in ('C','T'):
        ar = stage['arms'][arm]; tr = training[arm]
        c.need(tr['status'] == 'completed' and tr['mask_mode'] == arm and tr['plan_sha256'] == PLAN_SHA, 'Trainer mode/status differs')
        c.need(tr['configuration'] == launch['trainer_configurations'][arm] and
               tr['parameters'] == launch['trainer_parameters'], 'Training configuration not frozen')
        c.need(tr['source_sha256'] == c.sha(TRAINER) and tr['source_hashes'] == {k:c.sha(k) for k in TRAINER_SOURCES}, 'Trainer source differs')
        c.need(tr['input_array_identities'] == shared and tr['input_sha256'] == ar['prepared']['sha256'], 'Shared input certificate differs')
        c.need(ar['prepared'] == stage['source_R63_prepared'] == stage['arms']['C']['prepared'], 'C/T must use original same R63 file')
        c.need(tr['fit_rows'] == n and tr['query_rows'] == q and tr['numeric_count'] == 144 and
               tr['categorical_count'] == 23, 'Full outer fit and 144/23 schema required')
        c.need(tr['identity_contract'] == stage['identity_contract'] and tr['categorical_columns'] == cats and
               tr['identity_columns'] == IDENTITY_COLUMNS and tr['identity_indices'] == indices, 'Trainer identity contract differs')
        c.need(tr['saved_model_reload_exact'] is True and tr['reload_rows'] == q and
               tr['NPY_roundtrip_exact'] is True, 'Full model/vector reload required')
        for key in ('initial_head_bias','initial_training_median_ratio','initial_state_sha256',
                    'initial_torch_CPU_rng_sha256','initial_torch_CUDA_rng_sha256','cardinalities','parameter_count'):
            c.need(tr[key] == training['C'][key], 'C/T initialization or capacity differs: ' + key)
        for flag in ('loss_equals_R63','complete_original_category_mask_draw','other_19_masks_unchanged'):
            c.need(tr[flag] is True, 'Required trainer behavior: ' + flag)
        c.need([e['epoch'] for e in tr['epochs']] == list(range(1,31)), 'Thirty fixed epochs required')
        for flag in ('query_labels_available','query_labels_used','sample_weights_recomputed_in_trainer',
                     'sample_weights_batch_renormalized','price_or_ratio_clipping','eval_category_mask_applied',
                     'epoch_selection_uses_training_metrics','automatic_retry'):
            c.need(tr[flag] is False, 'Forbidden training behavior: ' + flag)
        for key in ('train_price_factor_million','query_price_factor_million','train_sample_weight'):
            c.need(tr[key+'_sha256'] == shared[key]['values_sha256'], 'Factor/date identity differs')


def preflight(sealed_path, request_path):
    checked = {}
    def bind(path, digest):
        path = c.rel(path)
        c.need(path not in checked or checked[path] == digest, 'Conflicting file identity')
        c.need(c.sha(path) == digest, 'Input SHA differs: ' + path); checked[path] = digest
        return c.safe(path)
    def document(rec): return c.read(bind(rec['path'], rec['sha256']))
    def array(rec, semantic=True):
        a = np.load(bind(rec['path'], rec['sha256']), allow_pickle=False)
        c.need(isinstance(a, np.ndarray) and not a.dtype.hasobject, 'Numeric array required')
        if semantic: c.need(c.identity(a) == {k: rec[k] for k in ('dtype', 'shape', 'values_sha256')}, 'Array identity differs')
        return a
    request = document(c.record(request_path))
    c.need(request['schema_version'] == 1 and request['plan'] == {'path': PLAN, 'sha256': PLAN_SHA} and
           request['parameters'] == parameters(), 'Evaluation request differs')
    bind(PLAN, PLAN_SHA); own = source_hashes()
    c.need(all(request['source_sha256'].get(k) == v for k, v in own.items()), 'Evaluator sources not frozen')
    c.verify_hashes(request['source_sha256'])
    launch = document(request['launch_request'])
    c.need(launch['plan_sha256'] == PLAN_SHA, 'Research request plan differs'); c.verify_hashes(launch['source_sha256'])
    c.need(request['protected_sha256'] == c.protected_hashes(), 'Protected files changed')
    c.need(c.rel(sealed_path) == 'outputs/round106/sealed_manifest.json', 'Unexpected root seal path')
    seal = document(c.record(sealed_path)); check_header(seal)
    c.need(seal['request'] == request['launch_request'] and seal['protected_after'] == request['protected_sha256'], 'Root binding differs')
    execution = c.safe(sealed_path).parent / 'execution.json'
    ex = document({'path': c.rel(execution), 'sha256': execution.with_suffix('.sha256').read_text(encoding='ascii').strip()})
    c.need(ex['exit_code'] == 0 and ex.get('failure') is None, 'Root process incomplete')
    c.verify_hashes(seal['source_sha256'])
    c.need(all(launch['source_sha256'].get(k) == v for k, v in seal['source_sha256'].items()), 'Driver source not frozen')
    oldmain = document(OLD); oldresults = document(OLD_RESULTS); r63 = document(R63_RESULTS)
    for k, h in own.items():
        if k in oldresults['source_sha256']: c.need(h == oldresults['source_sha256'][k], 'Reused metric source drift')
    for key in ('features', 'canonical_mapping', 'raw_train'):
        c.need(seal[key] == oldmain[key], 'Dataset identity changed'); bind(seal[key]['path'], seal[key]['sha256'])
    mapping = array(seal['canonical_mapping'], False)
    metric.index(mapping, 455784, 'canonical mapping'); c.need(len(mapping) == 455255, 'Canonical row count differs')
    Xraw = pd.read_pickle(c.safe(c.FEATURES))
    c.need(len(Xraw) == 455784 and Xraw.index.equals(pd.RangeIndex(len(Xraw))) and
           not {'總價元', 'target', 'y'}.intersection(Xraw.columns), 'Pure-X schema differs')
    days = Xraw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64); stages = {}
    for s in c.STAGES:
        sr = seal['stages'][s]; sm = document(sr['stage_manifest']); oldstage = document(oldmain['stages'][s]['stage_manifest'])
        ex = document(sr['execution']); c.need(ex['exit_code'] == 0 and ex.get('failure') is None, 'Stage process incomplete')
        c.need(sm['stage'] == s and sm['status'] == 'joint_dropout_stage_predictions_sealed' and sm['plan_sha256'] == PLAN_SHA, 'Stage contract differs')
        c.need(sm['request'] == seal['request'] and sm['protected_before'] == sm['protected_after'] == request['protected_sha256'], 'Stage provenance differs')
        c.need(sm['all_query_reload_exact'] is True and all(sm[k] is True for k in REPLAY_FLAGS), 'Stage endpoint replay/reload incomplete')
        c.need(sm['new_NN_fits'] == 2 and sm['new_native_tree_fits'] == 0 and all(sm[k] is False for k in
               ('query_labels_available', 'query_scores_computed', 'raw_price_pickle_opened')), 'Stage fit/label boundary differs')
        for name, digest in sm['artifact_sha256'].items(): bind(name, digest)
        start, end, nf, nq = c.SPLITS[s]
        c.need((sm['cutoff'], sm['query_end'], sm['outer_fit_rows'], sm['query_rows']) == (start, end, nf, nq), 'Fixed window differs')
        c.need(sm['source_R102_stage'] == oldmain['stages'][s]['stage_manifest'], 'Endpoint source differs')
        ids = {k: array(sm[k]) for k in ('fit_indices', 'query_indices', 'fit_source_cache_indices', 'query_source_cache_indices')}
        for k, a in ids.items(): c.exact(a, array(oldstage[k]), 'Original index differs: ' + k)
        fi, qi = ids['fit_indices'], ids['query_indices']
        metric.check_temporal(s, fi, qi, ids['query_source_cache_indices'], days, mapping)
        c.exact(mapping[fi], ids['fit_source_cache_indices'], 'Raw fit mapping differs')
        c.need(sm['source_R63_results'] == R63_RESULTS, 'Original R63 authority differs')
        for key in ('source_R63_prepared','source_R63_preparation','source_R63_training'):
            rec = sm[key]
            c.need(r63['stages'][s]['artifact_sha256'][rec['path']] == rec['sha256'], 'Original R63 artifact differs')
        prep = document(sm['source_R63_preparation'])
        c.need(prep == r63['stages'][s]['preparation'] and prep['prepared_path'] == sm['source_R63_prepared']['path'] and
               prep['prepared_sha256'] == sm['source_R63_prepared']['sha256'], 'Original R63 preparation differs')
        c.need(sm['source_R63_prior_preparation'] == {'path':prep['prior_preparation_path'], 'sha256':prep['prior_preparation_sha256']},
               'Original grouped-prior preparation differs')
        prior = document(sm['source_R63_prior_preparation']); oldtrain = document(sm['source_R63_training'])
        expected_values = {**prior['unchanged_six_array_sha256'],
                           **{k:prior[k+'_sha256'] for k in ('train_price_factor_million','query_price_factor_million')},
                           'train_sample_weight':prep['train_sample_weight_sha256']}
        c.need({k:v['values_sha256'] for k,v in sm['all_input_arrays'].items()} == expected_values,
               'Nine shared input semantic identities differ from original authority')
        contract = document(sm['identity_contract']); training = {}
        c.need(contract == launch['stage_specifications'][s]['identity_contract'], 'Frozen named-mask contract differs')
        c.need(contract['prepared_source'] == sm['source_R63_prepared'], 'Identity contract prepared source differs')
        encoder = document(contract['encoder_source'])
        c.need(contract['categorical_columns'] == encoder['categorical_columns'], 'Encoder column order differs')
        bind(sm['source_R63_prepared']['path'], sm['source_R63_prepared']['sha256'])
        for arm in ('C','T'):
            ar = sm['arms'][arm]
            for key in ('model','prepared'): bind(ar[key]['path'], ar[key]['sha256'])
            training[arm] = document(ar['training']); ax = document(ar['execution'])
            c.need(ax['exit_code'] == 0 and ax.get('failure') is None, 'NN subprocess incomplete')
        check_training(sm, contract, training, launch)
        for arm in ('C','T'):
            for key in ('initial_head_bias','initial_training_median_ratio','cardinalities','parameter_count'):
                c.need(training[arm][key] == oldtrain[key], 'Original initialization/capacity differs: ' + key)
            c.need(all(training[arm]['configuration'][k] == v for k,v in oldtrain['configuration'].items()),
                   'An original R63 hyperparameter changed')
        fixed = {k: array(rec) for k, rec in sm['fixed_branches'].items()}
        c.need(set(fixed) == {'area', 'lgb', 'H', 'U'}, 'Fixed branches differ')
        for k, a in fixed.items(): c.exact(a, array(oldstage['fixed_branches'][k]), 'Fixed branch changed')
        predictions = {k: array(rec) for k, rec in sm['predictions'].items()}
        layers = {arm: {k: array(rec) for k, rec in records.items()} for arm, records in sm['layers'].items()}
        c.need(set(predictions) == set(layers) == {'v22', 'C', 'T'} and sr['predictions'] == sm['predictions'], 'Arm identities differ')
        c.need(all(set(a) == set(LAYERS[:-1]) for a in layers.values()), 'Four premedian layers required')
        for a in [*fixed.values(), *predictions.values(), *[a for layer in layers.values() for a in layer.values()]]:
            metric.prices(a, (nq,), 'Saved query total')
        metric.check_formulas(predictions, layers, fixed)
        for arm, oldarm in (('v22', 'v22'),):
            c.exact(predictions[arm], array(oldstage['predictions'][oldarm]), 'Endpoint median differs')
            for k, a in layers[arm].items(): c.exact(a, array(oldstage['layers'][oldarm][k]), 'Endpoint layer differs: ' + arm + '/' + k)
        for arm in ('C','T'): c.exact(layers[arm]['native'], array(sm['arms'][arm]['native_prediction']), 'Trainer native vector differs')
        stages[s] = {'fit': Xraw.iloc[mapping[fi]].reset_index(drop=True), 'query': Xraw.iloc[mapping[qi]].reset_index(drop=True),
                     'query_indices': qi, 'query_source_cache_indices': ids['query_source_cache_indices'],
                     'predictions': predictions, 'layers': layers}
    c.need(request['protected_sha256'] == c.protected_hashes(), 'Protected changed during prelabel gate')
    return {'stages': stages, 'input_sha256': checked, 'source_sha256': own,
            'protected_before': request['protected_sha256'], 'raw_train': seal['raw_train'],
            'old_results': oldresults, 'sealed_manifest': c.record(sealed_path), 'evaluation_request': c.record(request_path)}


def run(sealed_path, request_path, outpath):
    out = c.safe(outpath); out.mkdir(parents=True, exist_ok=False); tick = time.monotonic(); failure = None; opened = False
    try:
        checked = preflight(sealed_path, request_path)
        c.write(out / 'prelabel_gate.json', {'status': 'all_six_matched_NN_fits_and_three_windows_sealed_before_scoring',
                'input_sha256': checked['input_sha256'], 'source_sha256': checked['source_sha256'],
                'sealed_manifest': checked['sealed_manifest'], 'native_models_NPZ_or_R_S_states_deserialized': False,
                'new_fits': 0, 'completed_at_utc': c.now()})
        c.need(c.sha(checked['raw_train']['path']) == checked['raw_train']['sha256'], 'Raw prices changed')
        raw = pd.read_pickle(c.safe(checked['raw_train']['path'])); opened = True; stages = {}
        for s in c.STAGES:
            a = checked['stages'][s]; y = metric.select_query_y(raw, a['query_source_cache_indices'])
            score = lambda v: metric.descriptive_metrics(y, v, a['fit'], a['query'], c.SPLITS[s][0])
            metrics = {k: score(v) for k, v in a['predictions'].items()}
            layers = {arm: {k: score(v) for k, v in layer.items()} for arm, layer in a['layers'].items()}
            for arm in layers: layers[arm]['median'] = metrics[arm]
            old = checked['old_results']['stages'][s]
            for arm, oldarm in (('v22', 'v22'),):
                c.need(metrics[arm] == old['metrics'][oldarm] and layers[arm] == old['layer_metrics'][oldarm], 'Endpoint full five-layer metrics replay differs')
            c.need(c.values_sha(y) == old['query_target_values_sha256'], 'Original query target mapping differs')
            folder = out / s; folder.mkdir(); boots = {}; replicates = {}
            for control in ('v22', 'C'):
                item, samples = metric.paired(y, a['predictions'], a['fit'], a['query'], a['query_indices'], s, control)
                item['caution'] = CAUTION
                for family in item['methods'].values():
                    for short, key in (('overall', 'overall_mae'), ('weighted', 'weighted_mae')):
                        z = family['summary'][short]
                        c.need(abs(z['control_mae'] - metrics[control][key]) <= 1e-6 and
                               abs(z['candidate_mae'] - metrics['T'][key]) <= 1e-6, 'Bootstrap point differs')
                boots[control] = item
                replicates[control] = {f: c.save_array(folder / f'T_vs_{control}_{f}_replicates.npy', v) for f, v in samples.items()}
            layer_comparisons = {control: {k: metric.comparisons({arm: layers[arm][k] for arm in layers}, control)
                                           for k in LAYERS} for control in ('v22', 'C')}
            item = {'stage': s, 'cutoff': c.SPLITS[s][0], 'query_end': c.SPLITS[s][1],
                    'outer_fit_rows': len(a['fit']), 'query_rows': len(y), 'metrics': metrics, 'layer_metrics': layers,
                    'comparison': metric.comparisons(metrics), 'comparison_vs_C': metric.comparisons(metrics, 'C'),
                    'C_vs_v22': metric.comparisons(metrics, 'v22', 'C'), 'layer_comparisons': layer_comparisons,
                    'C_retraining_layer_comparisons': {k:metric.comparisons({arm:layers[arm][k] for arm in layers}, 'v22', 'C') for k in LAYERS},
                    'layer_comparisons_are_descriptive_not_adoption_gates': True,
                    'paired': boots, 'replicate_artifacts': replicates, 'control_C_role': 'retrained original R63 independent dropout, same full inputs',
                    'query_target_values_sha256': c.values_sha(y), 'query_target_values_persisted': False,
                    'group_membership_uses_original_outer_fit': True, 'original_v22_five_layer_metrics_replay_exact': True,
                    'C_retraining_bitwise_equality_to_saved_v22_required': False}
            c.write(folder / 'results.json', item); stages[s] = item
            print(json.dumps({'stage': s, 'T_minus_v22': item['comparison']['overall_mae']['delta'],
                              'T_minus_C': item['comparison_vs_C']['overall_mae']['delta']}), flush=True)
        del raw
        c.verify_hashes(checked['input_sha256'])
        c.need(source_hashes() == checked['source_sha256'] and c.protected_hashes() == checked['protected_before'], 'Sources/protected changed')
        result = {'round': 106, 'status': 'completed_three_window_evaluation', 'created_at_utc': c.now(), 'parameters': parameters(),
                  'stages': stages, 'research_gates': research_gates(stages),
                  **{k: checked[k] for k in ('input_sha256', 'source_sha256', 'protected_before', 'sealed_manifest', 'evaluation_request')},
                  'protected_after': c.protected_hashes(), 'raw_pickle_whole_table_deserialized_after_all_seals': True,
                  'native_models_NPZ_or_R_S_states_deserialized': False, 'conditional_intervals_reported': 24,
                  'additional_bootstrap_calls': 0, 'new_fits': 0, 'model_load_predict_calls': 0, 'test_files_read': 0,
                  'new_CSV': False, 'selected_submission': 'v22', 'Private_score': None, 'fresh_blind_validation': False,
                  'single_seed_causal_identification_claimed': False, 'fully_historically_as_of': False, 'caution': CAUTION,
                  'elapsed_seconds': time.monotonic() - tick}
        c.write(out / 'results.json', result); return result
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(out / 'execution.json', {'exit_code': int(failure is not None), 'failure': failure,
                'elapsed_seconds': time.monotonic() - tick, 'source_sha256': c.sha(__file__),
                'raw_prices_deserialized': opened, 'new_fits': 0, 'model_load_predict_calls': 0, 'automatic_retry': False})


def self_check():
    tick = time.monotonic(); checks = []
    def mark(name, ok): c.need(ok, name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError(name)
    stages = {s: {'metrics': {a: {'overall_mae': v, 'weighted_mae': v} for a,v in [('v22',100.),('C',95.),('T',90.)]},
                  'comparison': {'quarter_or_large_group_deterioration_above_5pct': []},
                  'comparison_vs_C': {'quarter_or_large_group_deterioration_above_5pct': []}} for s in c.STAGES}
    mark('both_fixed_gates_pass', research_gates(stages)['both_research_gates_passed'])
    for name in ('overall_mae','weighted_mae'):
        z = copy.deepcopy(stages); z['auxiliary']['metrics']['T'][name] = 95.
        mark('equal_C_is_not_improvement_'+name, not research_gates(z)['mechanism']['mechanism_gate_passed'])
    for key, gate in (('comparison','primary'),('comparison_vs_C','mechanism')):
        z = copy.deepcopy(stages); z['development'][key]['quarter_or_large_group_deterioration_above_5pct'] = [{}]
        mark('group_risk_required_'+gate, not research_gates(z)[gate]['adoption_research_gate_passed' if gate=='primary' else 'mechanism_gate_passed'])
    z = copy.deepcopy(stages); z['confirmation']['metrics']['T']['overall_mae'] = 99.6
    mark('latest_half_percent_required', not research_gates(z)['primary']['adoption_research_gate_passed'])
    mark('native_not_gate', research_gates(stages)['mechanism']['native_improvement_required'] is False)
    mark('C_need_not_equal_v22', research_gates(stages)['both_research_gates_passed'] and
         not parameters()['C_retraining_bitwise_equality_to_saved_v22_required'])
    header = {'schema_version':1,'status':'all_joint_dropout_predictions_sealed','stage_order':list(c.STAGES),
              'stages':dict.fromkeys(c.STAGES),'plan_sha256':PLAN_SHA,'new_NN_fits':6,'new_native_tree_fits':0,
              'prior_or_R_S_build_calls':0,'encoder_build_calls':0,
              'all_query_reload_exact':True,'protected_before':dict.fromkeys(range(9),'x'),
              'protected_after':dict.fromkeys(range(9),'x'),**dict.fromkeys(REPLAY_FLAGS,True),
              **dict.fromkeys(('query_labels_available','query_scores_computed','raw_price_pickle_opened'),False)}
    check_header(header); checks.append('six_completed_fit_header')
    for key,val in [('new_NN_fits',5),('query_scores_computed',True),('all_query_reload_exact',False),('plan_sha256',c.PLAN_SHA)]:
        reject('reject_header_'+key, lambda key=key,val=val:check_header({**header,key:val}))
    mark('fixed_CI_contract', parameters()['replicates']==2000 and parameters()['seed']==20260925 and
         parameters()['stage_indices']=={'development':0,'confirmation':1,'auxiliary':2} and
         parameters()['family_indices']=={'project':0,'month':1})
    # Certificate-only training fixture: no true inputs, labels, model or Torch import.
    n,q=7,3; cats=['cat_'+str(i) for i in range(23)]
    for index,name in zip((10,13,11,20),IDENTITY_COLUMNS): cats[index]=name
    contract={'categorical_columns':cats,'identity_columns':IDENTITY_COLUMNS,'identity_indices':[10,13,11,20]}
    shared={}
    for name in c.ARRAY_NAMES:
        shape=([23] if name=='cardinalities' else
               [n if name.startswith('train') else q]+([144] if name.endswith('_num') else [23] if name.endswith('_cat') else []))
        shared[name]={'dtype':'int64' if name.endswith('_cat') or name=='cardinalities' else 'float32',
                      'shape':shape,'values_sha256':'a'*64}
    prepared={'path':'synthetic.npz','sha256':'b'*64}; ident={'path':'synthetic.json','sha256':'c'*64}
    sm={'outer_fit_rows':n,'query_rows':q,'all_input_arrays':shared,'source_R63_prepared':prepared,
        'identity_contract':ident,'arms':{a:{'prepared':prepared} for a in ('C','T')}}
    launch={'trainer_configurations':{a:{'mode':a} for a in ('C','T')},'trainer_parameters':{'synthetic':True}}
    base={'status':'completed','plan_sha256':PLAN_SHA,'parameters':launch['trainer_parameters'],
          'source_sha256':c.sha(TRAINER),'source_hashes':{k:c.sha(k) for k in TRAINER_SOURCES},
          'input_array_identities':shared,'input_sha256':prepared['sha256'],'fit_rows':n,'query_rows':q,
          'numeric_count':144,'categorical_count':23,'identity_contract':ident,**contract,
          'saved_model_reload_exact':True,'reload_rows':q,'NPY_roundtrip_exact':True,
          'initial_head_bias':1.,'initial_training_median_ratio':1.3,'initial_state_sha256':'d'*64,
          'initial_torch_CPU_rng_sha256':'e'*64,'initial_torch_CUDA_rng_sha256':'f'*64,
          'cardinalities':[3]*23,'parameter_count':100,'epochs':[{'epoch':i} for i in range(1,31)]}
    base.update(dict.fromkeys(('loss_equals_R63','complete_original_category_mask_draw','other_19_masks_unchanged'),True))
    base.update(dict.fromkeys(('query_labels_available','query_labels_used','sample_weights_recomputed_in_trainer',
                              'sample_weights_batch_renormalized','price_or_ratio_clipping','eval_category_mask_applied',
                              'epoch_selection_uses_training_metrics','automatic_retry'),False))
    base.update({k+'_sha256':'a'*64 for k in ('train_price_factor_million','query_price_factor_million','train_sample_weight')})
    training={a:{**copy.deepcopy(base),'mask_mode':a,'configuration':launch['trainer_configurations'][a]} for a in ('C','T')}
    check_training(sm,contract,training,launch); checks.append('full144_23_same_nine_certificate_accepts')
    for name,value in [('numeric_count',145),('initial_state_sha256','wrong'),('epochs',[{'epoch':1}]),
                       ('train_price_factor_million_sha256','wrong'),('eval_category_mask_applied',True)]:
        bad=copy.deepcopy(training); bad['T'][name]=value
        reject('reject_training_'+name,lambda bad=bad:check_training(sm,contract,bad,launch))
    bad=copy.deepcopy(contract); bad['identity_indices']=[13,10,11,20]
    reject('wrong_named_identity_positions_reject',lambda:check_training(sm,bad,training,launch))
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,
            'source_sha256':c.sha(__file__),'source_hashes':source_hashes(),'plan_sha256':PLAN_SHA,
            'parameters':parameters(),'real_data_read':False,'new_fit_score_calls':0,
            'elapsed_seconds':time.monotonic()-tick}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-check',action='store_true'); ap.add_argument('--receipt')
    ap.add_argument('--sealed-manifest'); ap.add_argument('--evaluation-request'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.self_check:
        result = self_check()
        if a.receipt: c.write(a.receipt,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    else:
        if not all((a.sealed_manifest,a.evaluation_request,a.out)): ap.error('sealed-manifest/evaluation-request/out required')
        run(a.sealed_manifest,a.evaluation_request,a.out)
