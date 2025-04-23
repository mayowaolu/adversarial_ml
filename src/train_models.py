import os, time, argparse, random, yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
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
from utils import save_checkpoint

import wandb

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- CONFIG ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config_wrn.yml', help="Path to config file")
args = parser.parse_args()

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
torch.backends.cudnn.benchmark = False 


#------------parse config file--------------
#data
data_path = cfg['data']['path']

# training
train_cfg = cfg["training"]
batch_size = train_cfg["batch_size"]
num_epochs = train_cfg["epochs"]
lr = train_cfg['optimizer']['params']["lr"]
momentum = train_cfg['optimizer']['params']["momentum"]
weight_decay = float(train_cfg['optimizer']['params']["weight_decay"])

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
checkpoint_dir = cfg['logging']['checkpoint_dir']

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)



try:
    cpu_count = len(os.sched_getaffinity(0))
except AttributeError:
    cpu_count = 1

def create_dataloaders(tiny=False):
    # Create dataloaders
    try:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        # Attempt to load or download the datasets
        train_cifar = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)
        test_cifar = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transforms)

        if tiny:
            rng  = torch.Generator().manual_seed(0)
            train_idx = torch.randperm(len(train_cifar), generator=rng)[:5000]
            test_idx  = torch.randperm(len(test_cifar),  generator=rng)[:500]
            train_cifar = Subset(train_cifar, train_idx.tolist())
            test_cifar  = Subset(test_cifar , test_idx.tolist())

        # Create DataLoaders
        train_loader = DataLoader(dataset=train_cifar, batch_size=batch_size, shuffle=True, num_workers=min(cpu_count, 10), pin_memory=True)
        test_loader = DataLoader(dataset=test_cifar, batch_size=test_batch_size, shuffle=False, num_workers=min(cpu_count, 10), pin_memory=True)

        print("Dataloaders created successfully")

    except (OSError, RuntimeError, Exception) as e:
        print(f"Error creating dataloaders: {e}")
        print(f"Please check the data path '{data_path}', network connection, and dataset integrity.")
        exit(1)
    
    return train_loader, test_loader


def train(model, dataloader, metrics, optimizer, device):
    metrics.reset()
    model.train()
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # ---------------- get adv samples --------------
        x_adv = pgd_attack(model= model,
                           inputs=images, 
                           labels=labels,
                           epsilon=epsilon,
                           step_size=step_size,
                           num_steps=num_steps)
        model.train()
        # ---------------- mart loss and model optimization step --------------
        optimizer.zero_grad()
        logits_clean = model(images)
        logits_adv = model(x_adv)
        loss = mart_loss(logits_clean=logits_clean,
                         logits_adv=logits_adv,
                         labels=labels,
                         lambda_reg=lambda_reg)
        loss.backward()
        optimizer.step()

        metrics["train/loss"].update(loss.detach())
        metrics["train/acc"].update(logits_adv.argmax(dim=1), labels)
    
    end_evt.record(); end_evt.synchronize()
    epoch_time = start_evt.elapsed_time(end_evt) / 1e3
    
    return metrics.compute(), epoch_time



def validate(model, dataloader, metrics, num_steps=20, epsilon=0.031, step_size=0.003, device="cuda"):
    metrics.reset()
    model.eval()

    for images, labels in tqdm(dataloader, desc=f"Validation"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # ---------- clean accuracy ----------
        with torch.no_grad():
            logits_clean = model(images)

        metrics["val/clean_acc"].update(logits_clean.argmax(dim=-1), labels)

        # ---------- adversarial accuracy ----------
        x_adv = pgd_attack(model, inputs=images, labels=labels, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
        with torch.no_grad():
            logits_adv = model(x_adv)        
        metrics["val/adv_acc"].update(logits_adv.argmax(dim=-1), labels)

    return metrics.compute()


def main():
    wandb.login()

    wandb.init(
        entity="johnolusetire-george-mason-university",
        project="adversarial-ml-mart-main-backup",
        config = train_cfg
    )

    #---------------create dataloaders------------------------
    train_loader, test_loader = create_dataloaders()
    
    #--------------- create model, optimizer and scheduler--------
    if cfg["model"]["name"] == "ResNet":
        model = ResNet18().to(device)
        print("Training ResNet 18 model")
    else:
        model = WideResNet(depth=depth, widen_factor=width).to(device)
        print("Training WideResNet 34/10 model")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones= [74, 89, 99], gamma=0.1)

    #-----------metrics--------------------------------------------
    NUM_CLASSES = 10

    base_metrics = MetricCollection({
        "loss" : MeanMetric(),
        "acc"  : Accuracy(task="multiclass", num_classes=NUM_CLASSES)}).to(device)

    train_metrics = base_metrics.clone(prefix="train/")   # ➜ keys: train/loss, train/acc

    val_metrics   = MetricCollection({
        "clean_acc" : Accuracy(task="multiclass", num_classes=NUM_CLASSES),
        "adv_acc"   : Accuracy(task="multiclass", num_classes=NUM_CLASSES)}).to(device).clone(prefix="val/")


    best_val_acc = float("-inf")
    top_checkpoints = []    
   
    for epoch in range(1, num_epochs):

        train_stats, elapsed = train(model=model,
                                dataloader=train_loader,
                                metrics=train_metrics,
                                optimizer=optimizer,
                                device=device)
         
        
        val_stats = validate(model=model,
                            dataloader=test_loader,
                            metrics=val_metrics,
                            num_steps=val_num_steps,
                            epsilon=epsilon,
                            step_size=val_step_size,
                            device=device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_log = {
            "epoch": epoch,
            "lr": current_lr,
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
            path = f"{checkpoint_dir}/newlast.pth",
            model=model, optimizer=optimizer,
            epoch=epoch,
            metrics={"robust_acc": robust_acc, "clean_acc": clean_acc}
        )

        # Save top 10 models
        if robust_acc > best_val_acc:
            best_val_acc = robust_acc
            save_checkpoint(
                path = f"{checkpoint_dir}/best_valAcc={robust_acc:.3f}.pth",
                top_checkpoints=top_checkpoints,
                model=model, optimizer=optimizer,
                epoch=epoch,
                metric_key="robust_acc",
                metrics={"robust_acc": robust_acc, "clean_acc": clean_acc},
                max_checkpoints=7
                )

    
    

    print("Training Done")
    print('================================================================')

    # clean_acc, robust_acc = validate(model=model,
    #                                 dataloader=test_loader,
    #                                 num_steps=val_num_steps,
    #                                 epsilon=epsilon,
    #                                 step_size=val_step_size,
    #                                 device=device)
    
    print(f"Final Test accuracy: {clean_acc*100:.2f}%, Robust Acc: {robust_acc*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()






