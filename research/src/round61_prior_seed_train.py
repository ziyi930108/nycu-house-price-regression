"""Isolated R61 seed entrypoint: unchanged R52 prior-ratio trainer, seeds 137/2026 only."""
from __future__ import annotations
import argparse
import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path
import round52_nn_train as trainer

ALLOWED_SEEDS=(137,2026)
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def seeded_configuration(seed):
    if type(seed) is not int or seed not in ALLOWED_SEEDS:
        raise ValueError('Only the fixed new seeds 137 and 2026 are allowed')
    old52,old47=trainer.CONFIG,trainer.prior.CONFIG
    if old52['seed']!=42 or old47['seed']!=42:
        raise ValueError('Expected unchanged baseline seed 42 in both imported configurations')
    new52,new47=copy.deepcopy(old52),copy.deepcopy(old47)
    new52['seed']=new47['seed']=seed
    assert all(new52[k]==v for k,v in old52.items() if k!='seed')
    assert all(new47[k]==v for k,v in old47.items() if k!='seed')
    assert all(new52[k]==v for k,v in new47.items())
    trainer.CONFIG,trainer.prior.CONFIG=new52,new47
    try:
        yield
    finally:
        trainer.CONFIG,trainer.prior.CONFIG=old52,old47


def run(data_path,out,seed):
    # Mutations exist only inside this isolated invocation; no old source is edited.
    sources={Path(p).resolve().relative_to(ROOT).as_posix():sha(p)
             for p in (__file__,trainer.__file__,trainer.prior.__file__)}
    original_setup,original_write=trainer.prior.setup,trainer.write
    witness={}
    with seeded_configuration(seed):
        def setup():
            assert trainer.CONFIG['seed']==trainer.prior.CONFIG['seed']==seed
            device=original_setup()  # Original Python/NumPy/Torch/CUDA seeding.
            actual=int(trainer.torch.initial_seed())
            if actual!=seed:
                raise RuntimeError('Torch RNG seed differs from the fixed requested seed')
            witness.update(actual_seed=actual,seed_setup_completed=True)
            return device
        def write(path,value):
            assert witness.get('actual_seed')==seed and witness.get('seed_setup_completed') is True
            assert value['configuration']==trainer.CONFIG
            assert value['source_sha256']==sources[Path(trainer.__file__).resolve().relative_to(ROOT).as_posix()]
            assert value['architecture_source_sha256']==sources[Path(trainer.prior.__file__).resolve().relative_to(ROOT).as_posix()]
            enriched=copy.deepcopy(value)
            enriched.update(requested_seed=seed,actual_seed=witness['actual_seed'],effective_seed=witness['actual_seed'],
                seed_setup_completed=True,seed_setup_method='unchanged R47 setup; torch.initial_seed checked',
                seeded_configurations={'round52_nn_train':seed,'round47_nn_train':seed},
                baseline_seed=42,changed_configuration_fields=['seed'],
                seed_entrypoint_source_sha256=sources[Path(__file__).resolve().relative_to(ROOT).as_posix()],
                underlying_trainer_source_sha256=sources[Path(trainer.__file__).resolve().relative_to(ROOT).as_posix()],
                seed_source_sha256=sources)
            original_write(path,enriched)
        trainer.prior.setup,trainer.write=setup,write
        try:
            trainer.run(Path(data_path),Path(out))
            if any(sha(ROOT/n)!=h for n,h in sources.items()):
                raise RuntimeError('Trainer source changed during execution')
        finally:
            trainer.prior.setup,trainer.write=original_setup,original_write


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=ALLOWED_SEEDS,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.data,args.out,args.seed)
