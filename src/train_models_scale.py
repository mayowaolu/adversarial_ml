import os, argparse, random, yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy, MetricCollection, MeanMetric

import numpy as np
from tqdm import tqdm

from attacks.pgd import pgd_attack
from models.resnet import ResNet18
from models.wide_resnet import WideResNet
from mart_loss import mart_loss
from utils import save_checkpoint, make_run_dir, get_model_norm
from trainer import train, validate
from data_utils import create_dataloaders

import wandb

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Speedups
torch.set_float32_matmul_precision('high')
scaler = torch.GradScaler()


# ----------- CONFIG ----------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='./config_wrn.yml', help="Path to config file")
parser.add_argument('--adversarial', action='store_true', help="Train adversarially (default: False). Pass this flag to enable.")
parser.add_argument('--small', action='store_true', help="Train with tiny subset (default: False). Pass this flag to enable.")
parser.add_argument('--compile', action='store_true', help="Use torch.compile (default: False). Pass this flag to enable.")
parser.add_argument('--resume', default=None, help="Path to resume checkpoint")

args = parser.parse_args()

# ----------- FLAGS -------------------------------
adv = args.adversarial
small_subset = args.small
compile_flag = args.compile
args.resume = args.resume

try:
    cfg = yaml.safe_load(open(args.config))
except FileNotFoundError:
    print(f"cfg file {args.config} not found.")
    exit(1)
except yaml.YAMLError:
    print(f"Error parsing the cfg file {args.config}.")
    exit(1)

# ----------- seed for reproducibility ----------------------
seed = cfg["environment"]["seed"]
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


#------------ parse config file -------------------
#data
data_path = cfg['data']['path']

# training
train_cfg = cfg["training"]
batch_size = train_cfg["batch_size"]
num_epochs = train_cfg["epochs"] if not small_subset else 10
_optimizer = train_cfg['optimizer']['name']
lr = train_cfg['optimizer']['params']["lr"]
weight_decay = train_cfg['optimizer']['params']["weight_decay"]
lr_scheduler = train_cfg['lr_scheduler']['name']

# val cfg
val_cfg = cfg["validation"]
test_batch_size = val_cfg["batch_size"]
val_freq = val_cfg["frequency"]
val_num_steps = val_cfg['eval_attack']["num_steps"]
val_step_size = val_cfg['eval_attack']['step_size']

# Adversarial Params
adv_cfg = cfg["adversarial"]
epsilon = adv_cfg["epsilon"]
num_steps = adv_cfg['train_attack']["num_steps"]
step_size = adv_cfg['train_attack']["step_size"]

# MART Specific
lambda_reg = cfg["mart"]["lambda_reg"]

# Model Specific
if cfg["model"]["name"] == "WideResNet":
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
elif cfg["model"]["name"] == "ResNet":
    depth = cfg["model"]["depth"]

# Logging
log_dir = cfg['logging']['log_dir']
base_dir = cfg['logging']['checkpoint_dir']

os.makedirs(base_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def main():
    checkpoint_dir = None
    wandb_run_id = None

    #---------------create dataloaders------------------------
    print("Creating Dataloaders....")
    train_loader, test_loader = create_dataloaders(data_path='./data', train_batch_size=256, tiny=small_subset)
    
    #--------------- create model, optimizer and scheduler--------
    if cfg["model"]["name"] == "ResNet":
        model = ResNet18().to(device)
        print("Training ResNet 18 model")
    else:
        model = WideResNet(depth=depth, widen_factor=width).to(device)
        print("Training WideResNet 34/10 model")

    if compile_flag:
        model = torch.compile(model)
        print("Using torch.compile")
    
    # optimizer
    if _optimizer.lower() == "sgd":
        momentum = train_cfg['optimizer']['params']["momentum"]
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif _optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    print(f"Using {_optimizer} as the optimizer.")

    # scheduler
    if lr_scheduler.lower() == "multisteplr":
        milestones = train_cfg["lr_scheduler"]["params"]["milestones"]
        gamma = train_cfg["lr_scheduler"]["params"]["gamma"]
        scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler.lower() == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=0)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {lr_scheduler}")
    
    print(f"Using {lr_scheduler} as the learning rate scheduler.")

    start_epoch = 1
    best_val_acc = float("-inf")

    # ------------------- Resume from checkpoint -------------------------------------
    if args.resume is not None:
        # Extract directory path from resume checkpoint
        checkpoint_path = os.path.abspath(args.resume)
        # Go up to find the checkpoint directory (typically 2 levels: /path/to/checkpoints/best/checkpoint.pth)
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))

        # Load the entire checkpoint dictionary, mapping tensors to the correct device
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            print(f"Loading checkpoint from: {args.resume}")

            # checkpoint keys
            # model_state, optimizer_state, scheduler, epoch, metrics

            # Load Model State
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                print("Warning: 'model_state' not found in checkpoint. Model weights not loaded.")

            # Load Optimizer State
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                print("Warning: 'optimizer_state' not found in checkpoint. Optimizer state not loaded.")

            # Load Scheduler State
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                print("Warning: 'scheduler' not found in checkpoint. Scheduler state not loaded.")

            # Load Epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            else:
                print("Warning: 'epoch' not found in checkpoint. Starting from epoch 1.")
                start_epoch = 1 # Default if epoch not found

            # Load Best Validation Accuracy (Optional but recommended)
            if 'metrics' in checkpoint and 'robust_acc' in checkpoint['metrics']: # Or clean_acc depending on goal
                # Initialize best_val_acc from checkpoint if resuming
                best_val_acc = checkpoint['metrics'].get('robust_acc', float("-inf")) if adv else checkpoint['metrics'].get('clean_acc', float("-inf"))
                print(f"Resuming with best recorded accuracy: {best_val_acc:.4f}")
            else:
                print("Warning: Could not load previous best accuracy from checkpoint.")
                best_val_acc = float("-inf") # Reset if not found
            
            if 'wandb_run_id' in checkpoint:
                wandb_run_id = checkpoint['wandb_run_id']
                print(f"Continuing wandb run: {wandb_run_id}")

            print(f"Resuming training from epoch {start_epoch}")

        except FileNotFoundError:
            print(f"Error: Resume checkpoint file not found at {args.resume}")
            exit(1)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            exit(1)

    # ----------- Metrics--------------------------------------------
    NUM_CLASSES = 10

    base_metrics = MetricCollection({
        "loss" : MeanMetric(),
        "acc"  : Accuracy(task="multiclass", num_classes=NUM_CLASSES)}).to(device)

    train_metrics = base_metrics.clone(prefix="train/")   # ➜ keys: train/loss, train/acc

    val_metrics   = MetricCollection({
        "clean_acc" : Accuracy(task="multiclass", num_classes=NUM_CLASSES),
        "adv_acc"   : Accuracy(task="multiclass", num_classes=NUM_CLASSES)}, prefix="val/").to(device)
   
    top_checkpoints = []

    if args.compile:
        for _ in range(3):
            dummy = torch.randn(1, 3, 32, 32, device=device)
            _ = model(dummy)

    # ------------------ wandb logging -------------------
    wandb.login()

    wandb_init_args = {
        "entity": "johnolusetire-george-mason-university",
        "project": cfg["project"]["name"],
        "config": cfg
    }

    if wandb_run_id:
        wandb_init_args["id"] = wandb_run_id
        wandb_init_args["resume"] = "must"

    wandb.init(**wandb_init_args)
    current_run_id = wandb.run.id  # Save the wandb run ID in checkpoints

    if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
        checkpoint_dir = make_run_dir(base_dir)
    else:
        print(f"Resuming with existing checkpoint directory: {checkpoint_dir}")

    print(f"Starting training for {num_epochs} epochs. Start epoch is {start_epoch}...")
    for epoch in range(start_epoch, num_epochs):

        train_stats, elapsed = train(model=model,
                                dataloader=train_loader,
                                metrics=train_metrics,
                                optimizer=optimizer,
                                device=device,
                                adv=adv,
                                epsilon=epsilon,
                                step_size=step_size,
                                num_steps=num_steps,
                                lambda_reg=lambda_reg,
                                scaler=scaler)
        
        val_stats = validate(model=model,
                            dataloader=test_loader,
                            metrics=val_metrics,
                            num_steps=val_num_steps,
                            epsilon=epsilon,
                            step_size=val_step_size,
                            device=device,
                            adv=adv)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_log = {
            "epoch": epoch,
            "lr": current_lr,
            "time_elapsed": elapsed,
            "grad_norm": get_model_norm(model),
            **train_stats,
            **val_stats }
        
        wandb.log(epoch_log,step=epoch)
        
        print(f"Epoch [{epoch:03d}/{num_epochs}] | "
                f"LR: {current_lr:.6g} | "
                f"Train Loss: {train_stats['train/loss']:.4f} | "
                f"Train Acc: {train_stats['train/acc']*100:.2f}% | "
                f"Val Clean Acc: {val_stats['val/clean_acc']*100:.2f}% | "
                f"Val Adv Acc:  {val_stats['val/adv_acc']*100:.2f}% | "
                f"Time: {elapsed:.2f}s")
        
        robust_acc, clean_acc = val_stats['val/adv_acc'], val_stats['val/clean_acc']

        # ------------- checkpoint -------------------
        # Save last checkpoint for easy training resumption

        save_checkpoint(
            path = f"{checkpoint_dir}/last/last.pth",
            model=model, optimizer=optimizer, scheduler=scheduler, # Add scheduler
            epoch=epoch,
            wandb_run_id=current_run_id,
            metrics={"robust_acc": robust_acc.item(), "clean_acc": clean_acc.item()}
        )

        current = robust_acc if adv else clean_acc
        # Save top 10 models
        if current > best_val_acc:
            best_val_acc = current.item()
            best_check_path = f"{checkpoint_dir}/best/best_valAcc={best_val_acc:.3f}.pth"
            save_checkpoint(
                path = best_check_path,
                top_checkpoints=top_checkpoints,
                model=model, optimizer=optimizer, scheduler=scheduler, # Add scheduler
                epoch=epoch,
                metric_key="robust_acc" if adv else "clean_acc",
                metrics={"robust_acc": robust_acc.item(), "clean_acc": clean_acc.item()},
                max_checkpoints=7
            )

            wandb.save(f"{checkpoint_dir}/best_valAcc={current:.3f}.pth")
        
        if epoch % 5 == 0:
            save_checkpoint(
                path = f"{checkpoint_dir}/gradual/epoch{epoch}.pth",
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                metrics={"robust_acc": robust_acc.item(), "clean_acc": clean_acc.item()})


    

    print("Training Done")
    print('================================================================')

    # clean_acc, robust_acc = validate(model=model,
    #                                 dataloader=test_loader,
    #                                 num_steps=val_num_steps,
    #                                 epsilon=epsilon,
    #                                 step_size=val_step_size,
    #                                 device=device)
    
    best_model = torch.load(best_check_path)
    model.load_state_dict(best_model["model_state"])
    val_stats = validate(model=model,
                        dataloader=test_loader,
                        metrics=val_metrics,
                        num_steps=val_num_steps,
                        epsilon=epsilon,
                        step_size=val_step_size,
                        device=device)
    robust_acc, clean_acc = val_stats['val/adv_acc'], val_stats['val/clean_acc']

    print(f"Best Model Loaded at epoch {best_model['epoch']}")
    print(f"Final Natural accuracy: {clean_acc*100:.2f}%, Robust Acc: {robust_acc*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()