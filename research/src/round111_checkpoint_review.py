"""CPU-only R111 checkpoint structure review in the isolated Torch runtime.

No housing arrays, native inference, fitting, CSV writing or CUDA API calls.
The main final driver runs this tool because its normal runtime lacks Torch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import round111_residual_prior_train as trainer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inspect_payload(payload):
    model = trainer.restore_checkpoint(payload, torch.device('cpu'))
    trainer.need(not list(model.named_buffers()), 'R111 must have no BatchNorm/running-state buffers')
    state = model.state_dict()
    return {'actual_model_class': trainer.MODEL_CLASS, 'configuration': payload['configuration'],
            'baseline_configuration': payload['baseline_configuration'],
            'numeric_count': payload['numeric_count'], 'cardinalities': payload['cardinalities'],
            'embedding_dimensions': model.dimensions, 'residual_block_count': len(model.blocks),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'state_keys': list(state), 'state_count': len(state),
            'all_state_float32_finite': all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),
            'state_shape_dtype_sha256': hashlib.sha256(json.dumps([
                [key, list(value.shape), str(value.dtype)] for key, value in state.items()],
                separators=(',', ':')).encode('utf8')).hexdigest()}


def review(model_path, metadata_path):
    trainer.check_configuration()
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8-sig'))
    trainer.need(metadata['status'] == 'completed' and metadata['source_sha256'] == trainer.sha(trainer.__file__)
                 and metadata['model_sha256'] == sha(model_path), 'Completed actual trainer/checkpoint identity')
    actual = inspect_payload(torch.load(model_path, map_location='cpu', weights_only=True))
    for key in ('actual_model_class', 'configuration', 'baseline_configuration', 'numeric_count',
                'cardinalities', 'embedding_dimensions', 'parameter_count'):
        trainer.need(actual[key] == metadata[key], 'Actual CPU checkpoint differs from training metadata: ' + key)
    trainer.need(actual['all_state_float32_finite'] and actual['residual_block_count'] == 2,
                 'Actual R111 native structure differs')
    return {'status': 'passed', 'model': {'path': str(Path(model_path).resolve()), 'sha256': sha(model_path)},
            'training': {'path': str(Path(metadata_path).resolve()), 'sha256': sha(metadata_path)},
            'review_tool_sha256': sha(__file__), 'trainer_source_sha256': trainer.sha(trainer.__file__),
            'actual': actual, 'new_fits': 0, 'GPU_calls': 0, 'native_model_inference_calls': 0,
            'query_or_test_labels_loaded': False, 'structural_review_only': True,
            'full_query_reload_claim_is_trainer_evidence_not_repeated_here': True}


def self_check():
    trainer.check_configuration()
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(111)
    model = trainer.TabularResidualPrior(144, [5] * 23)
    payload = trainer.checkpoint_payload(model, 144, [5] * 23)
    actual = inspect_payload(payload)
    trainer.need(actual['numeric_count'] == 144 and len(actual['cardinalities']) == 23
                 and actual['residual_block_count'] == 2 and actual['all_state_float32_finite'],
                 'Synthetic CPU payload structure')
    return {'status': 'passed', 'actual': actual, 'new_fits': 0, 'GPU_calls': 0,
            'housing_data_read': False, 'native_model_inference_calls': 0, 'new_CSV': False,
            'source_sha256': sha(__file__), 'author_selfcheck_not_independent_review': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--training-meta', type=Path)
    p.add_argument('--self-check', action='store_true')
    args = p.parse_args()
    if args.self_check:
        if args.model is not None or args.training_meta is not None:
            p.error('Self-check cannot accept housing checkpoint paths')
        result = self_check()
    else:
        if args.model is None or args.training_meta is None:
            p.error('--model and --training-meta required')
        result = review(args.model, args.training_meta)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
