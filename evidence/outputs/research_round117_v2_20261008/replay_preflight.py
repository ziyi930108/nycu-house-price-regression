"""Root CPU replay of original contexts only; prohibit any new prior fit/score."""
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import round100_contract as c
import round117_research_v2 as driver


def deny(*args, **kwargs):
    raise AssertionError('New fit/prior build/scoring forbidden in context-only preflight')


def main():
    tick = time.monotonic()
    sources, protected = driver.sources(), driver.protected()
    spec = driver.specifications()
    c.verify_hashes(spec['input_sha256'])
    original_reader = pd.read_pickle
    reads = []
    def guarded_reader(path, *args, **kwargs):
        actual = Path(path).resolve()
        c.need(actual != c.safe(c.RAW), 'Raw price pickle forbidden in context preflight')
        reads.append(c.rel(actual))
        return original_reader(path, *args, **kwargs)
    rows = {}
    with patch.object(pd, 'read_pickle', side_effect=guarded_reader), \
         patch.object(driver.prior, 'fit_oof_prior', side_effect=deny), \
         patch.object(driver.prior.original, '_build', side_effect=deny), \
         patch.object(driver.metric, 'descriptive_metrics', side_effect=deny), \
         patch.object(driver.metric, 'paired', side_effect=deny):
        for origin in driver.ORIGINS:
            started = time.monotonic()
            loaded = driver.load_context(spec, origin)
            driver.baseline.verify_context(loaded, origin)
            prior = loaded['original_prior']
            old_state = c.read(prior['state']['path'])
            old_oof = np.load(c.safe(prior['oof']['path']), allow_pickle=False)
            old_query = np.load(c.safe(prior['query_prior']['path']), allow_pickle=False)
            c.exact(driver.prior.predict_prior(old_state, loaded['query_X']), old_query,
                    'Original saved full-fit query prior replay')
            for name, frame, units in (('train', loaded['fit_X'], old_oof),
                                       ('query', loaded['query_X'], old_query)):
                factor = np.asarray(driver.prior.effective_area(frame) * units / 1e6, dtype=np.float32)
                c.exact(factor, loaded['arrays'][name + '_price_factor_million'],
                        'Original factor from original unit-prior and area')
            old_initialization = driver.ratio_bias(loaded['arrays']['train_y_million'],
                                                  loaded['arrays'][driver.CHANGED_ARRAY])
            training = c.read(loaded['original_training']['path'])
            c.need(training['initial_training_median_ratio'] == old_initialization['initial_training_median_ratio']
                   and training['initial_head_bias'] == old_initialization['initial_head_bias_formula_float64'],
                   'Original saved ratio/bias authority')
            rows[origin] = {'fit_rows': len(loaded['fit_X']), 'query_rows': len(loaded['query_X']),
                'original_prepared_array_identities': loaded['array_identities'],
                'context': loaded['before'], 'original_initialization': old_initialization,
                'original_full_fit_query_prior_and_both_factors_exact': True,
                'original_v22_v24_and_prior_slot_replayed_exact': True,
                'elapsed_seconds': time.monotonic() - started}
            print(origin, round(rows[origin]['elapsed_seconds'], 3), flush=True)
            del loaded, old_state, old_oof, old_query
    c.need(sources == driver.sources() and protected == driver.protected(), 'Source/protected changed')
    c.verify_hashes(spec['input_sha256'])
    c.write('outputs/research_round117_v2_20261008/root_context_preflight.json', {
        'status': 'passed', 'source_sha256': {**sources, driver.PLAN: c.sha(driver.PLAN), driver.RECOVERY_PLAN: c.sha(driver.RECOVERY_PLAN)},
        'input_sha256': spec['input_sha256'], 'origin_order': list(driver.ORIGINS), 'origins': rows,
        'elapsed_seconds': time.monotonic() - tick, 'pickle_readset': sorted(set(reads)),
        'R115_postfit': spec['R115_postfit'], 'engineering_recovery': spec['engineering_recovery'], 'saved_fit_labels_read': True,
        'raw_price_pickle_deserialized': False, 'query_prices_read': False,
        'new_prior_builds': 0, 'new_fits': 0, 'GPU_calls': 0, 'new_prepared_arrays': 0,
        'native_model_inference_calls': 0, 'new_CSV': False, 'protected_unchanged': True,
        'review_tool': c.record(__file__), 'scope': 'Root context replay, separate independent preflight required.'})


if __name__ == '__main__':
    main()
