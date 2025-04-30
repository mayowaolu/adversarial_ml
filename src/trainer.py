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
from utils import save_checkpoint, make_run_dir, get_model_norm

import wandb

# -------------- TRAIN FUNCTION --------------------------------------
def train(model, dataloader, metrics, optimizer, device, adv=True):
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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits_clean = model(images)

            if adv:
                logits_adv = model(x_adv)
                loss = mart_loss(logits_clean=logits_clean,
                                logits_adv=logits_adv,
                                labels=labels,
                                lambda_reg=lambda_reg)
            else:
                loss = F.cross_entropy(logits_clean, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #optimizer.step()

        metrics["train/loss"].update(loss.detach())
        if adv:
            metrics["train/acc"].update(logits_adv.argmax(dim=1), labels)
        else:
            metrics["train/acc"].update(logits_clean.argmax(dim=1), labels)
    
    end_evt.record(); end_evt.synchronize()
    epoch_time = start_evt.elapsed_time(end_evt) / 1e3
    
    return metrics.compute(), epoch_time


# ------------ VALIDATION -------------------------------------
def validate(model, dataloader, metrics, num_steps=20, epsilon=0.031, step_size=0.003, device="cuda", adv=True):
    metrics.reset()
    model.eval()

    for images, labels in tqdm(dataloader, desc=f"Validation"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # ---------- clean accuracy ----------
        with torch.no_grad():
            logits_clean = model(images)

        metrics["val/clean_acc"].update(logits_clean.argmax(dim=-1), labels)

        # ---------- adversarial accuracy ----------
        if adv:
            x_adv = pgd_attack(model, inputs=images, labels=labels, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
            with torch.no_grad():
                logits_adv = model(x_adv)       
            preds_adv = logits_adv.argmax(dim=-1)
        else:
            # shift every true label by +1 (modulo) → 0% correct on purpose
            preds_adv = (labels + 1) % metrics["val/adv_acc"].num_classes

        metrics["val/adv_acc"].update(preds_adv, labels)

    return metrics.compute()