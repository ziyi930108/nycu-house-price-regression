"""Fixed embedding MLP in the isolated Torch runtime; no validation labels enter this process."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn

CONFIG = {
    "seed": 42, "epochs": 30, "batch_size": 2048,
    "hidden": [512, 256, 128], "dropout": 0.1, "category_dropout": 0.1,
    "embedding_dimension": "min(32,max(4,ceil(sqrt(cardinality))))",
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR", "minimum_learning_rate": 0.00001,
    "loss": "MAE in million TWD", "positive_output": "Softplus",
    "epoch_selection": "fixed last epoch; no query labels or early stopping",
    "float_precision": "float32; matmul precision high", "thread_count": 4,
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class EmbeddingMLP(nn.Module):
    def __init__(self, numeric_count, cardinalities):
        super().__init__()
        self.dimensions = [min(32, max(4, math.ceil(math.sqrt(k)))) for k in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(k, d) for k, d in zip(cardinalities, self.dimensions)])
        layers = []
        width = numeric_count + sum(self.dimensions)
        for hidden in CONFIG["hidden"]:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(CONFIG["dropout"])])
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.positive = nn.Softplus()

    def forward(self, numeric, categorical):
        if self.training:
            drop = torch.rand(categorical.shape, device=categorical.device) < CONFIG["category_dropout"]
            categorical = categorical.masked_fill(drop, 0)
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        joined = torch.cat([numeric, *embedded], dim=1)
        return self.positive(self.head(self.body(joined))).squeeze(1)


def setup():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(CONFIG["thread_count"])
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded experiment expects a CUDA GPU; do not silently start a long CPU run")
    torch.cuda.manual_seed_all(CONFIG["seed"])
    return torch.device("cuda:0")


@torch.inference_mode()
def predict(model, numeric, categorical):
    model.eval()
    chunks = []
    for start in range(0, len(numeric), 4096):
        chunks.append(model(numeric[start:start + 4096], categorical[start:start + 4096]).cpu().numpy())
    values = np.concatenate(chunks).astype(np.float64) * 1e6
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Nonfinite/nonpositive NN predictions")
    return values


def smoke_test():
    device = setup()
    numeric = torch.randn(16, 8, device=device)
    categorical = torch.stack([torch.randint(0, 5, (16,), device=device), torch.randint(0, 9, (16,), device=device)], dim=1)
    target = torch.linspace(1, 16, 16, device=device)
    model = EmbeddingMLP(8, [5, 9]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    model.train()
    loss = (model(numeric, categorical) - target).abs().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item()
    assert predict(model, numeric, categorical).shape == (16,)
    print(json.dumps({"status": "synthetic_forward_backward_passed", "real_housing_fit": False,
                      "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}), flush=True)


def run(data_path, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError("Run already started; preserve evidence instead of overwriting it")
    arrays = np.load(data_path, allow_pickle=False)
    expected = {"train_num", "train_cat", "train_y_million", "query_num", "query_cat", "cardinalities"}
    if set(arrays.files) != expected:
        raise ValueError(f"Unexpected NPZ fields, expected only train labels and X: {arrays.files}")
    xn, xc, y, qn, qc, card = [arrays[k] for k in ["train_num", "train_cat", "train_y_million", "query_num", "query_cat", "cardinalities"]]
    card = [int(k) for k in card]
    assert xn.ndim == qn.ndim == xc.ndim == qc.ndim == 2
    assert xn.shape[1] == qn.shape[1] and xc.shape[1] == qc.shape[1] == len(card)
    assert len(xn) == len(xc) == len(y) and len(qn) == len(qc) and y.ndim == 1
    assert xn.dtype == qn.dtype == np.float32 and xc.dtype == qc.dtype == np.int64
    assert np.isfinite(xn).all() and np.isfinite(qn).all() and np.isfinite(y).all() and (y > 0).all()
    for i, k in enumerate(card):
        assert k >= 1 and ((xc[:, i] >= 0) & (xc[:, i] < k)).all() and ((qc[:, i] >= 0) & (qc[:, i] < k)).all()
    device = setup()
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": CONFIG,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "fit_rows": len(xn), "query_rows": len(qn), "numeric_count": xn.shape[1], "cardinalities": card,
            "query_labels_available": False, "prediction_units": "TWD", "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "bitwise_retraining_determinism_claimed": False}
    write(out / "run.json", meta)
    model = EmbeddingMLP(xn.shape[1], card).to(device)
    median = float(np.median(y))
    inverse_median = median + float(np.log(-np.expm1(-median)))
    with torch.no_grad():
        model.head.bias.fill_(inverse_median)
    tn = torch.as_tensor(xn, device=device)
    tc = torch.as_tensor(xc, device=device)
    ty = torch.as_tensor(y.astype(np.float32), device=device)
    qtn = torch.as_tensor(qn, device=device)
    qtc = torch.as_tensor(qc, device=device)
    del arrays, xn, xc, y, qn, qc
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["minimum_learning_rate"])
    epochs = []
    fit_started = time.monotonic()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        order = torch.randperm(len(tn), device=device)
        total = torch.zeros((), device=device)
        lr = optimizer.param_groups[0]["lr"]
        for first in range(0, len(tn), CONFIG["batch_size"]):
            index = order[first:first + CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            pred = model(tn[index], tc[index])
            loss = (pred - ty[index]).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
        value = float((total / len(tn)).item())
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite training loss; no adaptive retry")
        epochs.append({"epoch": epoch, "learning_rate": lr, "training_mae_million_with_dropout": value})
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"event": "epoch_complete", **epochs[-1], "elapsed_seconds": time.monotonic() - fit_started}), flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - fit_started
    predictions = predict(model, qtn, qtc)
    model_path = out / "model.pt"
    torch.save({"numeric_count": meta["numeric_count"], "cardinalities": card, "configuration": CONFIG,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, model_path)
    loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    restored = EmbeddingMLP(loaded["numeric_count"], loaded["cardinalities"]).to(device)
    restored.load_state_dict(loaded["state_dict"])
    reloaded = predict(restored, qtn, qtc)
    np.testing.assert_array_equal(predictions, reloaded)
    np.save(out / "predictions_twd.npy", predictions, allow_pickle=False)
    meta.update(status="completed", epochs=epochs, initial_training_median_million=median,
                parameter_count=sum(p.numel() for p in model.parameters()), fit_seconds=fit_seconds,
                saved_model_reload_exact=True, reload_rows=len(predictions),
                prediction_min_twd=float(predictions.min()), prediction_max_twd=float(predictions.max()),
                elapsed_seconds=time.monotonic() - start, maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out / "training.json", meta)
    print(json.dumps({k: meta[k] for k in ["status", "fit_rows", "query_rows", "fit_seconds", "elapsed_seconds", "saved_model_reload_exact"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        if args.data is None or args.out is None:
            parser.error("--data and --out are required for a housing fit")
        run(args.data, args.out)
