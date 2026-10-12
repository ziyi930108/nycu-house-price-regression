"""Fresh inference from all six native models for the fixed selected v25.

--describe reads only small source/config JSON. Real inference requires a
reviewed closure SHA, original course Excel/sample and all 17 private artifacts.
No saved prediction path or external prepared NPZ is accepted as an input.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_guard import (need, read, digest, verify_closure, verify_artifacts,
    require_new_output, check_query_schema, NATIVE_KEYS)

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def child_environment():
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith('PYTHON')}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    return env

def probe_nn(interpreter, device):
    interpreter = Path(interpreter).absolute()  # Preserve venv executable identity.
    need(interpreter.is_file(), 'NN Python executable missing')
    command = [str(interpreter), '-I', '-B', '-X', 'utf8', str(ROOT/'src/nn_predict.py'),
               '--check-environment', '--device', device]
    p = subprocess.run(command, check=True, capture_output=True, text=True,
                       encoding='utf-8', timeout=60, env=child_environment())
    rec = json.loads(p.stdout)
    need(rec.get('status') == 'ready' and rec.get('device') == device
         and rec.get('models_loaded') == 0 and rec.get('new_fits') == 0,
         'NN environment probe did not pass')
    return interpreter, rec, command

def describe():
    config, manifest, closure_sha = verify_closure(ROOT)
    return {'status':'source_and_schema_described_not_inference_verified',
        'version':'v25', 'rows':config['rows'], 'native_models':6,
        'private_artifacts':len(manifest['files']), 'private_bytes':sum(e['bytes'] for e in manifest['files']),
        'closure_sha256':closure_sha, 'expected_selected_csv_sha256':config['expected_csv_sha256'],
        'input_mode':'raw course test Excel + sample IDs + 17 frozen native/fitted/geographic artifacts',
        'saved_predictions_accepted_as_inputs':False, 'external_prepared_NPZ_accepted':False,
        'models_loaded':0, 'new_fits':0, 'prediction_vectors_read':0, 'Private_score':None}

def run(a):
    start = time.perf_counter()
    need(not sys.flags.optimize, 'Python -O is forbidden')
    config, manifest, closure_sha = verify_closure(ROOT, a.expected_closure_sha)
    out = require_new_output(a.output_dir, ROOT)
    for key in ['test','sample']:
        need(digest(getattr(a,key)) == config['inputs'][key]['sha256'],
             key+' bytes differ; original row order is mandatory')
    verified_artifacts = verify_artifacts(a.artifacts, manifest)
    # No native model, pickle, Excel or saved vector has been deserialized above.
    nn_python, nn_probe, probe_command = probe_nn(a.nn_python, a.device)
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT/'src/closure_modules'))
    from house_features import build_features
    from submission_io import ids, write_submission, rounded_prices
    from round6_geo import frame_sha
    from nn_prepare import prepare_inputs
    from tree_runtime import predict_trees, load_states, apply_branch
    raw = pd.read_excel(a.test, sheet_name='Sheet1', dtype=object).reset_index(drop=True)
    sample = pd.read_csv(a.sample, dtype={'編號':str})
    query_ids, sample_ids = ids(raw), ids(sample)
    check_query_schema(list(raw),len(raw),query_ids,list(sample),sample_ids)
    X = build_features(raw)
    need(frame_sha(X) == config['feature_frame_sha256'], 'Frozen 87-feature frame identity differs')
    geo = pd.read_pickle(a.artifacts/'outputs/round6/geography/final/test.pkl')
    geo_spatial = pd.read_pickle(a.artifacts/'outputs/round39/geography/final/query.pkl')
    need(len(geo_spatial) == len(X) and X.index.equals(geo_spatial.index), 'Spatial geography row axis differs')
    encoder = read(a.artifacts/'outputs/round48/final/encoder_state.json')
    numeric = read(a.artifacts/'outputs/round55/final/numeric_state.json')
    prior = read(a.artifacts/'outputs/round60/final/price_prior_state.json')
    prepared = prepare_inputs(X,geo,encoder,numeric,prior)
    def identity(array):
        return {'dtype':str(array.dtype),'shape':list(array.shape),
                'values_sha256':hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    name_map = {'query_price_factor_million':'query_factor_million'}
    for name, expected in config['query_prior_identities'].items():
        need(identity(prepared[1][name_map.get(name,name)]) == expected, 'R107 rebuilt query array differs: '+name)
    need(hashlib.sha256(np.ascontiguousarray(prepared[0]['query_factor_million']).tobytes()).hexdigest()
         == config['area_factor_values_sha256'], 'Rebuilt N_area factor differs')
    need(prepared[0]['query_num'].shape == (47073,603)
         and identity(prepared[0]['query_num'][:,:144]) == config['query_prior_identities']['query_num'],
         'N_area numeric ramp schema/prefix differs')
    # Exclusive creation occurs before any prediction or output write.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'preflight.json', {'source_closure_sha256':closure_sha,
        'artifacts':verified_artifacts,'nn_probe':nn_probe,'probe_command':probe_command,
        'feature_frame_sha256':frame_sha(X),'query_prior_identities_verified':True,
        'saved_prediction_inputs':False,'query_labels_used':False})
    print('Predicting four native trees.',flush=True)
    native = predict_trees(X,geo,a.artifacts,threads=config['tree_threads'])
    need(set(native) == {'C_unit','C_log','C_raw','L_unit'}, 'Four native tree roles missing')
    child_receipts = {}
    for key, inp in zip(['N_area','N_prior'],prepared):
        spec = config['neural_models'][key]
        inp_path, pred_path = out/(key+'_inputs.npz'), out/(key+'.npy')
        with inp_path.open('xb') as f:
            np.savez_compressed(f,**inp)
        command = [str(nn_python),'-I','-B','-X','utf8',str(ROOT/'src/nn_predict.py'),
            '--model',str(a.artifacts/spec['model_path']),'--inputs',str(inp_path),
            '--out',str(pred_path),'--device',a.device,'--spec',str(ROOT/'config/v25.json'),'--role',key]
        print('Predicting native neural head: '+key,flush=True)
        subprocess.run(command,check=True,env=child_environment())
        r = read(pred_path.with_suffix('.json'))
        need(r['model_sha256'] == spec['expected_model_sha256'] and r['inputs_sha256'] == digest(inp_path)
             and r['output_sha256'] == digest(pred_path) and r['rows'] == config['rows']
             and r['models_loaded'] == 1 and r['new_fits'] == 0 and not r['saved_prediction_vectors_read'],
             'Neural child receipt does not bind this fresh model/inputs/output')
        # Only this just-created, exit-zero child output is loaded, never archived predictions.
        native[key] = np.load(pred_path,allow_pickle=False)
        child_receipts[key] = {'command':command,'receipt':r}
    need(set(native) == NATIVE_KEYS, 'All six fresh native roles required')
    native_ids = {}
    for key, value in native.items():
        need(value.shape == (47073,) and value.dtype == np.float64
             and np.isfinite(value).all() and (value >= 0).all(), 'Invalid native output: '+key)
        native_ids[key] = identity(value)
        if key.startswith('C_') or key == 'L_unit':
            with (out/(key+'.npy')).open('xb') as f:
                np.save(f,value,allow_pickle=False)
    reference,state = load_states(a.artifacts)
    bases = {'unit':.75*native['C_unit']+.25*native['L_unit'],
        'nn':.75*(.5*native['N_area']+.5*native['N_prior'])+.25*native['L_unit'],
        'log':.75*(.5*native['C_raw']+.5*native['C_log'])+.25*native['L_unit']}
    branches = {}
    for key,base in bases.items():
        print('Applying complete reference/spatial branch: '+key,flush=True)
        branches[key] = apply_branch(X,geo_spatial,base,reference,state)
        need(branches[key].shape == (47073,) and np.isfinite(branches[key]).all(), 'Invalid corrected branch')
        for suffix,value in [('base',base),('spatial',branches[key])]:
            with (out/(key+'_'+suffix+'.npy')).open('xb') as f:
                np.save(f,value,allow_pickle=False)
    prediction = np.median(np.vstack([branches['nn'],branches['log'],branches['unit']]),axis=0)
    need(np.all(rounded_prices(prediction) <= 2**53-1), 'Rounded CSV price exceeds original authoring safe-integer bound')
    with (out/'predictions.npy').open('xb') as f:
        np.save(f,prediction,allow_pickle=False)
    csv_path = out/'submission_v25_reproduced.csv'
    actual_sha = write_submission(csv_path,query_ids,sample_ids,prediction)
    # Recheck every source/input/state after use; no selected CSV is read or changed.
    verify_closure(ROOT,a.expected_closure_sha)
    verify_artifacts(a.artifacts,manifest)
    for key in ['test','sample']:
        need(digest(getattr(a,key)) == config['inputs'][key]['sha256'], 'Input changed during inference')
    receipt = {'status':'verified' if actual_sha == config['expected_csv_sha256'] else 'exact_CSV_replay_failed',
        'version':'v25','rows':len(prediction),'models_loaded':len(native),'new_fits':0,
        'saved_prediction_vectors_used_as_input':False,'external_prepared_NPZ_used':False,
        'query_labels_used':False,'Private_score':None,'selected_csv_read_or_modified':False,
        'source_closure_sha256':closure_sha,'artifact_manifest_sha256':digest(ROOT/'artifacts/manifest.json'),
        'output_sha256':actual_sha,'expected_selected_csv_sha256':config['expected_csv_sha256'],
        'csv_bytes_equal_selected':actual_sha == config['expected_csv_sha256'],
        'prediction_identity':identity(prediction),'native_output_identities':native_ids,
        'branch_output_identities':{k:identity(v) for k,v in branches.items()},
        'child_receipts':child_receipts,'nn_environment_probe':nn_probe,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'nn_device':a.device,'threads':4,'nn_batch_size':4096,'matmul_precision':'high',
        'fresh_OS_or_hardware_verified':False,'full_retraining_verified':False,
        'elapsed_seconds':time.perf_counter()-start}
    write_json(out/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    need(receipt['csv_bytes_equal_selected'], 'Fresh inference completed but exact selected v25 CSV differs; preserve and investigate outputs')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--describe',action='store_true')
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--test',type=Path)
    p.add_argument('--sample',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--nn-python',type=Path)
    p.add_argument('--expected-closure-sha')
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    a = p.parse_args()
    if a.describe:
        need(not any(getattr(a,k) is not None for k in ['artifacts','test','sample','output_dir','nn_python','expected_closure_sha']),
             '--describe cannot be combined with inference inputs')
        print(json.dumps(describe(),ensure_ascii=False,indent=2))
        return
    for key in ['artifacts','test','sample','output_dir','nn_python','expected_closure_sha']:
        if getattr(a,key) is None:
            p.error('--'+key.replace('_','-')+' is required for real inference')
    run(a)

if __name__ == '__main__':
    main()
