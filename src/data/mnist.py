from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_mnist(data_dir="artifacts/data"):
    """(train, test) datasets, pixels in [0,1]."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return train, test


def split_train_val(train_dataset, val_fraction=0.1, seed=0):
    """Split train into (train, val) with fixed seed."""
    n = len(train_dataset)
    n_val = int(n * val_fraction)
    gen = torch.Generator().manual_seed(seed)
    return random_split(train_dataset, [n - n_val, n_val], generator=gen)


def get_dataloaders(cfg, data_dir="artifacts/data"):
    """(train, val, test) loaders."""
    train_full, test = get_mnist(data_dir)
    train_sub, val_sub = split_train_val(train_full, cfg.val_fraction, cfg.val_seed)

    pin = torch.cuda.is_available()
    kw = dict(batch_size=cfg.batch_size, num_workers=2, pin_memory=pin)
    train_loader = DataLoader(train_sub, shuffle=True, **kw)
    val_loader = DataLoader(val_sub, shuffle=False, **kw)
    test_loader = DataLoader(test, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def sample_anchor_source(dataset, N=256, seed=42):
    """N class-balanced images, in class order -> (N, 1, 28, 28)."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        sub_idx = torch.tensor(dataset.indices)
        targets = base.targets[sub_idx]
        def get_img(local_i):
            return base[dataset.indices[local_i]][0]
    else:
        targets = dataset.targets
        def get_img(i):
            return dataset[i][0]

    rng = torch.Generator().manual_seed(seed)
    n_classes = 10
    base_n = N // n_classes
    remainder = N % n_classes

    selected = []
    for c in range(n_classes):
        n_c = base_n + (1 if c < remainder else 0)
        mask = (targets == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(mask), generator=rng)[:n_c]
        selected.extend(mask[perm].tolist())

    return torch.stack([get_img(i) for i in selected])


def anchor_class_labels(N=256, n_classes=10):
    """Digit class per anchor index, matching sample_anchor_source layout -> (N,)."""
    base, rem = divmod(N, n_classes)
    labels = []
    for c in range(n_classes):
        labels += [c] * (base + (1 if c < rem else 0))
    return torch.tensor(labels)
