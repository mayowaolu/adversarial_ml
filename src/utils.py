import os
import torch
from datetime import datetime
import secrets 

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int = None,
    metrics: dict = None,
    metric_key: str = None,
    top_checkpoints: list = None,
    max_checkpoints: int = 5):
    """
    Atomically save a training checkpoint.

    Args:
      path      - full filepath to write (e.g. "checkpoints/last.ckpt" or ".../best_acc.ckpt")
      model     - your nn.Module
      optimizer - your optimizer
      epoch     - (optional) current epoch number
      metrics   - (optional) dict of scalars (e.g. {"robust_acc":0.50, "clean_acc":0.84})
    """
    
    state = {
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
    }

    if epoch is not None:
        state["epoch"] = epoch
    if metrics:
        state["metrics"] = metrics
    

    # 2) Write to a temp file then atomically rename
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    if top_checkpoints is not None and metric_key and metrics:
        top_checkpoints.append((metrics[metric_key], path))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(top_checkpoints) > max_checkpoints: 
            _, old_path = top_checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Removed checkpoint: {old_path}")

def make_run_dir(base_dir="checkpoints", random_hex=True):
    # 1) timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 2) optional random hex so two runs in the same second don’t collide
    suffix=""
    if random_hex:
        suffix = f"_{secrets.token_hex(3)}"
    run_name = f"run_{stamp}{suffix}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def get_model_norm(model):
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5