import os
import torch
from tqdm import tqdm

import torchattacks

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

def attack_setup(model, attack_config: dict) -> list:
    """
    Sets up the adversarial attacks based on the provided configuration.

    Args:
        model: The model to be attacked.
        attack_config: Dictionary containing configurations for attacks.

    Returns:
        List of attack instances.
    """
    try:
        # verify keys of attack config
        attacks = []
        possible_keys = {"natural", "fgsm", "pgd", "cw_inf"}

        for key in attack_config.keys():
            if key.lower() not in possible_keys:
                raise ValueError(f"Invalid attack key: {key}. Valid keys are: {', '.join(possible_keys)}")
            
            if key == "natural":
                attacks.append("natural")
            elif key == "fgsm":
                attacks.append(torchattacks.FGSM(model, **attack_config[key]))
            elif key == "pgd":
                attacks.append(torchattacks.PGD(model, **attack_config[key]))
            elif key == "cw_inf":
                attacks.append(torchattacks.CW(model, **attack_config[key]))
        
        return attacks
    
    except Exception as e:
        print(f"Error in attack setup: {e}")
        return []

def evaluate_security(model, dataloader, attack_configs, device="cuda"):
    """
    Evaluates the security of the model against various adversarial attacks.
    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        attack_configs: Dictionary containing configurations for attacks.
        device: Device to run the evaluation on (default is "cuda").
    Returns:
        results: A dictionary containing accuracy for each attack.
    """
    model.eval()
    results = {}
    print("Evaluating security...")

    attacks = attack_setup(model, attack_configs)
    
    if not attacks:
        print("No attacks were set up. Exiting evaluation.")
        return {}

    for attack in attacks:
        correct = 0
        total   = 0
        name    = attack if attack=="natural" else attack.__class__.__name__
        print(f"\n→ Evaluating {name}")

        for images, labels in tqdm(dataloader, desc=f"Evaluating {name}"):
            images, labels = images.to(device), labels.to(device)

            if attack != "natural":
                images = attack(images)     # create adversarial data samples

            with torch.no_grad():
                predictions = model(images).argmax(1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        
        acc = correct / total if total > 0 else 0
        results[name] = acc
        print(f"Accuracy for {name}: {acc:.4f}")

    return results

