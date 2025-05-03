import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


#------------ DATALOADER -------------------
try:
    cpu_count = len(os.sched_getaffinity(0))
except AttributeError:
    cpu_count = 1

def create_dataloaders(data_path='', train_batch_size=128, test_batch_size=256, tiny=False, return_datasets=False):
    # Create dataloaders
    if os.path.exists(data_path):
        print("Data path exists.")
    else:
        print("Data path does not exist. Please check the path.")
        exit(1)

    try:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])


        if tiny:
            cifar_length = 50000

            fixed_train_count = 5000
            fixed_val_count = 2000
            # Use a fixed generator seed for consistent subset selection 
            rng  = torch.Generator().manual_seed(42)
            train_idx = torch.randperm(cifar_length, generator=rng)[:(fixed_train_count+fixed_val_count)]
            
            train_cifar = Subset(datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms), 
                                 train_idx[:fixed_train_count].tolist())
            val_cifar  = Subset(datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms), 
                                train_idx[fixed_train_count:].tolist())
            if return_datasets:
                return train_cifar, val_cifar

            train_loader = DataLoader(dataset=train_cifar, batch_size=train_batch_size, shuffle=True, num_workers=min(cpu_count, 10), pin_memory=True)
            val_loader = DataLoader(dataset=val_cifar, batch_size=test_batch_size, shuffle=False, num_workers=min(cpu_count, 10), pin_memory=True)
    


            return train_loader, val_loader


        train_cifar = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)
        test_cifar = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transforms)

        if return_datasets:
            return train_cifar, test_cifar

        # Create DataLoaders
        train_loader = DataLoader(dataset=train_cifar, batch_size=train_batch_size, shuffle=True, num_workers=min(cpu_count, 10), pin_memory=True)
        test_loader = DataLoader(dataset=test_cifar, batch_size=test_batch_size, shuffle=False, num_workers=min(cpu_count, 10), pin_memory=True)

        print("Dataloaders created successfully")

    except (OSError, RuntimeError, Exception) as e:
        print(f"Error creating dataloaders: {e}")
        print(f"Please check the data path '{data_path}', network connection, and dataset integrity.")
        exit(1)
    
    return train_loader, test_loader