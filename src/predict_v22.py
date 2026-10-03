"""Recompute v22 from original test Excel and frozen model/geography states."""
from __future__ import annotations
import argparse, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/"closure_modules"))
from house_features import build_features
from submission_io import digest, ids, write_submission
from round6_geo import frame_sha
from nn_prepare import prepare_inputs
from tree_runtime import predict_trees, load_states, apply_branch

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def check_nn_environment(interpreter,device):
    """Probe the actual NN interpreter before creating outputs or loading models."""
    # Keep a venv's symlink path: resolving it can select the base interpreter.
    interpreter=interpreter.absolute()
    if not interpreter.is_file():
        raise FileNotFoundError(f"--nn-python must name an existing Python executable: {interpreter}")
    try:
        probe=subprocess.run([str(interpreter),str(ROOT/"src/nn_predict.py"),
            "--check-environment","--device",device],check=True,capture_output=True,
            text=True,encoding="utf-8",errors="replace",timeout=60)
    except subprocess.CalledProcessError as error:
        detail=(error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"NN environment check failed for --nn-python {interpreter}:\n{detail}") from error
    except (OSError,subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot check --nn-python {interpreter}: {error}") from error
    try:
        result=json.loads(probe.stdout)
        if not isinstance(result,dict) or result.get("status")!="ready" or result.get("device")!=device:
            raise ValueError("Unexpected probe result")
    except ValueError as error:
        raise RuntimeError("NN environment check returned an invalid result") from error
    return interpreter,result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts",type=Path,default=ROOT/"private/artifacts")
    ap.add_argument("--test",type=Path,required=True)
    ap.add_argument("--sample",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--nn-python",type=Path,default=Path(sys.executable))
    ap.add_argument("--device",choices=["cpu","cuda:0"],default="cuda:0")
    a=ap.parse_args();start=time.perf_counter()
    if a.output_dir.exists():raise FileExistsError("Use a new output directory")
    a.nn_python,nn_environment=check_nn_environment(a.nn_python,a.device)
    config=read(ROOT/"config/v22.json"); manifest=read(ROOT/"artifacts/manifest.json")
    for name in ["test","sample"]:
        if digest(getattr(a,name)) != config["inputs"][name]["sha256"]:
            raise ValueError(f"{name} file differs: frozen geography requires the original row order")
    for e in manifest["files"]:
        p=(a.artifacts/e["path"]).resolve()
        if not p.is_relative_to(a.artifacts.resolve()) or digest(p)!=e["sha256"]:
            raise ValueError(f"Artifact hash differs: {e['path']}")
    a.output_dir.mkdir(parents=True)
    raw=pd.read_excel(a.test,sheet_name="Sheet1",dtype=object).reset_index(drop=True)
    if "總價元" in raw:raise ValueError("Query input must not contain targets")
    sample=pd.read_csv(a.sample,dtype={"編號":str})
    if list(sample)!=["編號","總價元"] or len(raw)!=47073 or ids(raw)!=ids(sample):
        raise ValueError("Unexpected sample schema, row count or ID order")
    X=build_features(raw)
    if frame_sha(X)!=config["feature_frame_sha256"]:
        raise ValueError("Raw Excel features differ from audited training schema")
    geo=pd.read_pickle(a.artifacts/"outputs/round6/geography/final/test.pkl")
    geo_spatial=pd.read_pickle(a.artifacts/"outputs/round39/geography/final/query.pkl")
    print("Raw Excel -> 87 features verified; predicting four trees.",flush=True)
    native=predict_trees(X,geo,a.artifacts,threads=4)
    encoder=read(a.artifacts/"outputs/round48/final/encoder_state.json")
    numeric=read(a.artifacts/"outputs/round55/final/numeric_state.json")
    prior=read(a.artifacts/"outputs/round60/final/price_prior_state.json")
    prepared=prepare_inputs(X,geo,encoder,numeric,prior)
    print("Frozen neural inputs rebuilt; predicting two neural heads.",flush=True)
    for key,inp,model_rel in zip(["N_area","N_prior"],prepared,
        ["outputs/round55/final/model/model.pt","outputs/round63/final/model/model.pt"]):
        inp_path=a.output_dir/(key+"_inputs.npz")
        np.savez_compressed(inp_path,**inp)
        pred_path=a.output_dir/(key+".npy")
        subprocess.run([str(a.nn_python),str(ROOT/"src/nn_predict.py"),
          "--model",str(a.artifacts/model_rel),"--inputs",str(inp_path),
          "--out",str(pred_path),"--device",a.device],check=True)
        native[key]=np.load(pred_path,allow_pickle=False)
    reference,state=load_states(a.artifacts)
    bases={
      "unit":.75*native["C_unit"]+.25*native["L_unit"],
      "nn":.75*(.5*native["N_area"]+.5*native["N_prior"])+.25*native["L_unit"],
      "log":.75*(.5*native["C_raw"]+.5*native["C_log"])+.25*native["L_unit"]}
    branches={}
    for key,base in bases.items():
        print(f"Applying reference and spatial correction: {key}",flush=True)
        branches[key]=apply_branch(X,geo_spatial,base,reference,state)
        np.save(a.output_dir/(key+"_spatial.npy"),branches[key],allow_pickle=False)
    prediction=np.median(np.vstack([branches["nn"],branches["log"],branches["unit"]]),axis=0)
    np.save(a.output_dir/"predictions.npy",prediction,allow_pickle=False)
    output=a.output_dir/"submission_v22_reproduced.csv"
    actual_sha=write_submission(output,ids(raw),ids(sample),prediction)
    receipt={
      "status":"verified" if actual_sha==config["expected_csv_sha256"] else "verification_failed","rows":len(prediction),"models_loaded":6,"new_fits":0,
      "input_mode":"raw_test_excel_plus_frozen_geography_and_fit_states",
      "saved_prediction_vectors_used_as_input":False,"test_prices_used":False,
      "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
      "nn_device":a.device,"artifact_manifest_sha256":digest(ROOT/"artifacts/manifest.json"),
      "nn_environment_check":nn_environment,
      "output_sha256":actual_sha,"expected_v22_sha256":config["expected_csv_sha256"],
      "csv_bytes_equal_original":actual_sha==config["expected_csv_sha256"],
      "prediction_sha256":digest(a.output_dir/"predictions.npy"),
      "runtime_source_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sorted((ROOT/"src").rglob("*.py"))},
      "elapsed_seconds":time.perf_counter()-start}
    (a.output_dir/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(receipt,ensure_ascii=False))
    if actual_sha!=config["expected_csv_sha256"]:
        raise RuntimeError("Inference finished, but exact v22 replay failed; inspect receipt and hardware differences")
if __name__=="__main__":main()
