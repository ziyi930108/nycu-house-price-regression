"""Isolated R69 seeds 137/2026 for the unchanged R63 weighted prior-ratio trainer."""
from __future__ import annotations
import argparse
import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path
import round63_recency_weighted_prior_train as trainer

ALLOWED_SEEDS = (137, 2026)
ROOT = Path(__file__).resolve().parents[1]
TRAINER_SHA = '26ed590b94992dbcd7ca7e26db7c45462ad6eaabfcc91c5bcee36c97390e253a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def seeded_configuration(seed):
    if type(seed) is not int or seed not in ALLOWED_SEEDS:
        raise ValueError('Only the fixed new seeds 137 and 2026 are allowed')
    old63, oldbase, old47 = trainer.CONFIG, trainer.BASELINE_CONFIG, trainer.prior.CONFIG
    original_check = trainer.check_configuration
    if any(c['seed'] != 42 for c in (old63, oldbase, old47)):
        raise ValueError('Expected unchanged seed 42 in all three imported configurations')
    original_check()  # All original R63/R52 configuration and parent-source guards.
    new63, newbase, new47 = (copy.deepcopy(c) for c in (old63, oldbase, old47))
    for original, changed in ((old63, new63), (oldbase, newbase), (old47, new47)):
        changed['seed'] = seed
        assert changed == {**original, 'seed': seed}

    def check_seeded():
        # Same R63 guard, with only the fixed seed-42 constraint replaced.
        if trainer.CONFIG != {**trainer.BASELINE_CONFIG, **trainer.LOSS_CONFIG_OVERRIDES}:
            raise ValueError('Only the fixed weighted-MAE configuration may differ from R52')
        if not all(trainer.BASELINE_CONFIG[k] == v for k, v in trainer.prior.CONFIG.items()):
            raise ValueError('R47 setup/architecture configuration changed')
        if trainer.CONFIG['seed'] != seed or trainer.CONFIG['epochs'] != 30 or trainer.CONFIG['batch_size'] != 2048:
            raise ValueError('Fixed requested seed/epochs/batch differ')
        if sha(trainer.PARENT_TRAINER) != trainer.PARENT_TRAINER_SHA:
            raise ValueError('Parent R52 trainer source changed')
        if (trainer.CONFIG != {**old63, 'seed': seed}
            or trainer.BASELINE_CONFIG != {**oldbase, 'seed': seed}
            or trainer.prior.CONFIG != {**old47, 'seed': seed}):
            raise ValueError('A non-seed configuration field changed')

    trainer.CONFIG, trainer.BASELINE_CONFIG, trainer.prior.CONFIG = new63, newbase, new47
    trainer.check_configuration = check_seeded
    try:
        check_seeded()
        yield
    finally:
        trainer.CONFIG, trainer.BASELINE_CONFIG, trainer.prior.CONFIG = old63, oldbase, old47
        trainer.check_configuration = original_check


def run(data_path, out, seed):
    if sha(trainer.__file__) != TRAINER_SHA:
        raise ValueError('Underlying R63 trainer source changed')
    if sha(trainer.prior.__file__) != trainer.ARCHITECTURE_SHA:
        raise ValueError('Original R47 architecture/setup source changed')
    sources = {Path(p).resolve().relative_to(ROOT).as_posix(): sha(p)
               for p in (__file__, trainer.__file__, trainer.prior.__file__, trainer.PARENT_TRAINER)}
    original_setup, original_write = trainer.prior.setup, trainer.write
    witness = {}
    with seeded_configuration(seed):
        def setup():
            trainer.check_configuration()
            assert trainer.CONFIG['seed'] == trainer.BASELINE_CONFIG['seed'] == trainer.prior.CONFIG['seed'] == seed
            device = original_setup()  # Original Python/NumPy/Torch/CUDA seeding.
            actual = int(trainer.torch.initial_seed())
            if actual != seed:
                raise RuntimeError('Torch RNG seed differs from the fixed requested seed')
            witness.update(actual_seed=actual, seed_setup_completed=True)
            return device

        def write(path, value):
            assert witness.get('actual_seed') == seed and witness.get('seed_setup_completed') is True
            trainer.check_configuration()
            assert value['configuration'] == trainer.CONFIG
            assert value['source_sha256'] == sources[Path(trainer.__file__).resolve().relative_to(ROOT).as_posix()]
            assert value['architecture_source_sha256'] == sources[Path(trainer.prior.__file__).resolve().relative_to(ROOT).as_posix()]
            enriched = copy.deepcopy(value)
            # Preserve R63's original loss-change fields; record the seed delta separately.
            enriched.update(requested_seed=seed, actual_seed=witness['actual_seed'], effective_seed=witness['actual_seed'],
                seed_setup_completed=True, seed_setup_method='unchanged R47 setup; torch.initial_seed checked',
                seeded_configurations={'round63_recency_weighted_prior_train': seed,
                                       'round63_BASELINE_CONFIG': seed, 'round47_nn_train': seed},
                baseline_seed=42, seed_changed_configuration_fields=['seed'],
                seed_entrypoint_source_sha256=sources[Path(__file__).resolve().relative_to(ROOT).as_posix()],
                underlying_trainer_source_sha256=sources[Path(trainer.__file__).resolve().relative_to(ROOT).as_posix()],
                seed_source_sha256=sources)
            original_write(path, enriched)

        trainer.prior.setup, trainer.write = setup, write
        try:
            trainer.run(Path(data_path), Path(out))
            if any(sha(ROOT / name) != digest for name, digest in sources.items()):
                raise RuntimeError('Trainer source changed during execution')
        finally:
            trainer.prior.setup, trainer.write = original_setup, original_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out, args.seed)
