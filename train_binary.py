"""Binary pneumonia classification training with the MICCAI 2024 Hybrid model.

Uses the Kaggle Pediatric Pneumonia Dataset (Normal vs Pneumonia).
Adapts the multi-region architecture for binary classification.

Innovations (all optional, default OFF for backward compatibility):
  --val_split 0.2          Stratified train/val split (fixes test-set leakage)
  --loss focal             Focal Loss to target hard negatives (low specificity)
  --label_smoothing 0.1    Penalise overconfident predictions
  --warmup_epochs 3        Linear warmup + cosine annealing
  --mixup_alpha 0.2        MixUp regularisation
  --randaugment            RandAugment(2, 9) for diverse augmentations
  --dropout 0.3            Dropout rate in Hybrid model
  --tta                    Test-time augmentation (horizontal flip)
  --ema_decay 0.999        Exponential moving average of weights
"""

import os
import sys
import math
import copy
import argparse
import time
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from model import Hybrid, HybridMSGR, HybridGRP, HybridCRC, HybridGR, HybridACRA, HybridATACRA, HybridCDA, HybridStandardQKV, HybridATACRADual, HybridVariant
from data.dataset import PneumoniaBinaryDataset


# ---------------------------------------------------------------------------
# Focal Loss — down-weights easy samples, focuses on hard negatives
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher = more weight on hard samples.
    alpha : tensor or None
        Optional per-class weighting, e.g. torch.tensor([1.0, 1.5]).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# EMA — exponential moving average of model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains a moving average of model parameters for smoother inference."""

    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self._shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    new = (1 - self.decay) * p.data + self.decay * self._shadow[name]
                    self._shadow[name] = new.clone()
                    self.ema_model.state_dict()[name].copy_(new)

    def state_dict(self):
        return self.ema_model.state_dict()


# ---------------------------------------------------------------------------
# Training epoch (with optional MixUp)
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device,
                mixup_alpha=0.0, proto_lambda=0.0, contrast_lambda=0.0,
                region_aux=0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for img, label in dataloader:
        img, label = img.to(device), label.to(device)

        # MixUp not supported with contrastive/proto — skip
        use_aug = (mixup_alpha > 0 and proto_lambda == 0 and contrast_lambda == 0)
        if use_aug:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(img.size(0), device=device)
            img_mixed = lam * img + (1 - lam) * img[idx]
            label_a, label_b = label, label[idx]
        else:
            img_mixed = img

        optimizer.zero_grad()
        output_raw = model(img_mixed)      # (B, 6, num_classes)
        output = output_raw.mean(dim=1)    # (B, num_classes)

        if use_aug:
            loss = lam * criterion(output, label_a) + \
                   (1 - lam) * criterion(output, label_b)
        else:
            loss = criterion(output, label)

        # --- Prototype regularization ---
        if proto_lambda > 0:
            normal_idx = (label == 0)
            if normal_idx.any():
                with torch.no_grad():
                    rf_normal = model.get_region_features(img[normal_idx])
                proto_loss = F.mse_loss(
                    rf_normal,
                    model.proto.prototypes.unsqueeze(0).expand(rf_normal.shape[0], -1, -1)
                )
                loss = loss + proto_lambda * proto_loss
                model.proto.update(rf_normal)

        # --- MoCo contrastive loss ---
        if contrast_lambda > 0:
            with torch.no_grad():
                # Get region features via momentum encoder path
                # (features from main encoder are fine for the query)
                rf_all = model.get_region_features(img)
            contrast_loss = model.compute_contrastive_loss(rf_all, label)
            loss = loss + contrast_lambda * contrast_loss

        # --- Region auxiliary loss: deep supervision on per-region logits ---
        if region_aux > 0:
            # per-region CE on the pre-pooled logits (B, 6, C)
            region_loss = F.cross_entropy(
                output_raw.reshape(-1, output_raw.shape[-1]),
                label.unsqueeze(1).expand(-1, output_raw.shape[1]).reshape(-1))
            loss = loss + region_aux * region_loss

        loss.backward()
        optimizer.step()

        # --- Update momentum encoder after each step ---
        if contrast_lambda > 0:
            model.update_momentum_encoder()

        total_loss += loss.item()
        correct += output.argmax(1).eq(label).sum().item()
        total += label.size(0)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Evaluation (returns full metrics dict with AUC)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, tta=False):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for img, label in dataloader:
        img, label = img.to(device), label.to(device)

        output = model(img)                # (B, 6, num_classes)

        # Optional TTA: average with horizontally flipped version
        if tta:
            output = (output + model(torch.flip(img, dims=[3]))) / 2.0

        output = output.mean(dim=1)        # (B, num_classes)
        loss = criterion(output, label)
        total_loss += loss.item()
        all_logits.append(output)
        all_labels.append(label)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = F.softmax(all_logits, dim=1)
    preds = all_logits.argmax(dim=1)

    n = all_labels.size(0)
    num_classes = all_logits.size(1)
    labels_np = all_labels.cpu().numpy()
    probs_np = all_probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # Per-class one-vs-rest confusion entries
    per_class = {}
    for c in range(num_classes):
        pos = labels_np == c
        p = preds_np == c
        tp = int((pos & p).sum())
        tn = int((~pos & ~p).sum())
        fp = int((~pos & p).sum())
        fn = int((pos & ~p).sum())
        per_class[c] = dict(tp=tp, tn=tn, fp=fp, fn=fn,
                            sensitivity=100.0 * tp / (tp + fn + 1e-8),
                            specificity=100.0 * tn / (tn + fp + 1e-8),
                            precision=100.0 * tp / (tp + fp + 1e-8),
                            f1=100.0 * 2 * tp / (2 * tp + fp + fn + 1e-8))

    if num_classes == 2:
        # Binary semantics (class 1 = positive) — back-compat with prior runs
        c1 = per_class[1]
        metrics = dict(
            sensitivity=c1['sensitivity'], specificity=c1['specificity'],
            precision=c1['precision'], f1=c1['f1'],
            auc_roc=roc_auc_score(labels_np, probs_np[:, 1]),
            auc_pr=average_precision_score(labels_np, probs_np[:, 1]),
            tp=c1['tp'], tn=c1['tn'], fp=c1['fp'], fn=c1['fn'],
        )
    else:
        # Multi-class: macro-averaged metrics + one-vs-rest AUC
        macro = lambda key: float(np.mean([per_class[c][key] for c in range(num_classes)]))
        try:
            auc_roc = roc_auc_score(labels_np, probs_np, multi_class='ovr',
                                    average='macro', labels=list(range(num_classes)))
        except ValueError:
            auc_roc = float('nan')
        try:
            auc_pr = average_precision_score(labels_np, np.eye(num_classes)[labels_np],
                                             average='macro')
        except ValueError:
            auc_pr = float('nan')
        metrics = dict(
            sensitivity=macro('sensitivity'), specificity=macro('specificity'),
            precision=macro('precision'), f1=macro('f1'),
            auc_roc=auc_roc, auc_pr=auc_pr,
            tp=0, tn=0, fp=0, fn=0,
        )

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100.0 * (preds == all_labels).float().mean().item(),
        'per_class': per_class,
        'num_classes': num_classes,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Prototype helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_initial_prototypes(model, train_loader, device):
    """Pass all normal training images through the model to initialise prototypes."""
    model.eval()
    all_feats = []
    for img, label in train_loader:
        normal_idx = (label == 0)
        if normal_idx.any():
            img_n = img[normal_idx].to(device)
            rf = model.get_region_features(img_n)  # (B, 6, 768)
            all_feats.append(rf.cpu())
    if all_feats:
        stacked = torch.cat(all_feats, dim=0)       # (N_normal, 6, 768)
        model.proto.prototypes.copy_(stacked.mean(dim=0).to(device))
        model.proto._initialized.fill_(1)
        print(f"Prototypes initialized from {stacked.shape[0]} normal images "
              f"(proto_norm={model.proto.prototypes.norm(dim=1).mean():.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_path', type=str, default='./chest_xray')
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='Fraction of training data for validation (0 = off)')
    # Model
    parser.add_argument('--arch', default='hybrid',
                        choices=['hybrid', 'hybrid_msgr', 'hybrid_grp', 'hybrid_crc', 'hybrid_gr', 'hybrid_acra', 'hybrid_atacra', 'hybrid_cda', 'hybrid_stdqkv', 'hybrid_atacra_dual', 'hybrid_variant'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (2 = binary, 3 = multi-class)')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in Hybrid model')
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', default='./checkpoints/')
    # Loss
    parser.add_argument('--loss', default='ce', choices=['ce', 'focal'],
                        help='Loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor for CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma for Focal Loss')
    # LR schedule
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Linear warmup epochs (0 = ReduceLROnPlateau)')
    # Regularisation
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='MixUp alpha (0 = off, 0.2 recommended)')
    parser.add_argument('--randaugment', action='store_true',
                        help='Enable RandAugment(2, 9) in training')
    # Inference
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation (horizontal flip)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay (0 = off, 0.999 recommended)')
    # Prototype
    parser.add_argument('--proto_lambda', type=float, default=0.1,
                        help='Weight of prototype regularization loss')
    parser.add_argument('--proto_warmup', type=int, default=3,
                        help='Epochs before computing initial prototypes')
    parser.add_argument('--proto_momentum', type=float, default=0.99,
                        help='EMA momentum for prototype update')
    parser.add_argument('--contrast_lambda', type=float, default=0.1,
                        help='Weight of MoCo contrastive loss (hybrid_crc only)')
    parser.add_argument('--warmstart', type=str, default='',
                        help='Path to pretrained Hybrid checkpoint for ACRA warm-start')
    parser.add_argument('--acra_lr_mult', type=float, default=10.0,
                        help='LR multiplier for ACRA-specific params (alpha, temperature, classifier)')
    parser.add_argument('--alpha_init', type=float, default=0.0,
                        help='Initial alpha value for ACRA residual gate (0=Hybrid baseline)')
    parser.add_argument('--prior_ipsi', type=float, default=0.5,
                        help='ATACRA anatomy prior weight for ipsilateral adjacency')
    parser.add_argument('--prior_contra', type=float, default=0.3,
                        help='ATACRA anatomy prior weight for contralateral mirror')
    parser.add_argument('--freeze_topo', action='store_true',
                        help='Freeze topo_bias Delta (ablation: ATACRA w/o Delta)')
    parser.add_argument('--no_gate', action='store_true',
                        help='Attention without alpha residual gate (ablation: Zero-QKV only)')
    parser.add_argument('--fixed_alpha', action='store_true',
                        help='Fixed residual coefficient alpha=alpha_init (not learned)')
    parser.add_argument('--no_prior', action='store_true',
                        help='Remove anatomy prior A entirely (keep learnable Delta)')
    parser.add_argument('--random_prior', action='store_true',
                        help='Shuffle anatomy prior positions (control: Random Prior)')
    parser.add_argument('--shuffle_seed', type=int, default=1234,
                        help='Seed for shuffling prior topology')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Initial attention temperature tau (default: sqrt(768)~27.7)')
    parser.add_argument('--no_balance', action='store_true',
                        help='Use ALL training images (no pneumonia undersampling)')
    parser.add_argument('--class_weights', action='store_true',
                        help='Inverse-frequency class weights for CE loss')
    parser.add_argument('--region_aux', type=float, default=0.0,
                        help='Region auxiliary loss weight (deep supervision, 0=off)')
    parser.add_argument('--tca', action='store_true',
                        help='Topology-Consistent Augmentation (flip + small rotation)')
    parser.add_argument('--aug_type', default='none',
                        choices=['none', 'cj01', 'cj02', 'sharp', 'erase'],
                        help='Intensity-only augmentation (geometry-preserving)')
    parser.add_argument('--variant_attn', default='multihead',
                        choices=['multihead', 'trainableA', 'attnlogits', 'evidence'],
                        help='Attention variant for hybrid_variant arch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (multi-seed experiments)')
    parser.add_argument('--split_file', type=str, default='',
                        help='Fixed train/val split JSON (created on first run if absent)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Reproducibility: fix all random sources ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(args.seed)

    print(f"Device: {device} | Backbone: {args.backbone} | Seed: {args.seed} | "
          f"Image size: {args.image_size} | Batch: {args.batch_size}")

    # --- Gather image paths (auto-detect classes from subdirectories) ---
    data_dir = args.data_path

    def gather_split(subset):
        root = os.path.join(data_dir, subset)
        class_names = sorted(d for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d)))
        paths, labels = [], []
        for ci, cn in enumerate(class_names):
            files = []
            for ext in ('*.jpeg', '*.jpg', '*.png'):
                files = sorted(glob.glob(os.path.join(root, cn, ext)))
                if files:
                    break
            paths += files
            labels += [ci] * len(files)
        return paths, labels, class_names

    train_paths, train_labels, class_names = gather_split('train')
    test_paths, test_labels, test_class_names = gather_split('test')
    if test_class_names != class_names:
        raise ValueError(f"Train/test class mismatch: {class_names} vs {test_class_names}")
    counts = {cn: train_labels.count(ci) for ci, cn in enumerate(class_names)}
    print(f"Classes: {class_names} | Raw train counts: {counts}")

    # Balance training set (binary only; FIXED subset regardless of --seed)
    if args.num_classes == 2 and not args.no_balance:
        n0, n1 = train_labels.count(0), train_labels.count(1)
        if n1 > n0:
            rng = np.random.RandomState(42)
            idx1 = [i for i, l in enumerate(train_labels) if l == 1]
            drop = set(rng.choice(idx1, size=n1 - n0, replace=False).tolist())
            train_paths = [p for i, p in enumerate(train_paths) if i not in drop]
            train_labels = [l for i, l in enumerate(train_labels) if i not in drop]

    val_paths, val_labels = None, None

    # --- Fixed data split: reuse saved split across runs/seeds ---
    import json as _json
    split_file = args.split_file or f"data_splits/split_seed{args.seed}.json"

    if args.val_split > 0:
        if os.path.exists(split_file):
            # Load the fixed split — identical train/val across all experiments
            with open(split_file) as f:
                split = _json.load(f)
            train_paths = split["train_paths"]
            val_paths = split["val_paths"]
            train_labels = split["train_labels"]
            val_labels = split["val_labels"]
            # Patient-disjoint split: test set also comes from the JSON
            if "test_paths" in split:
                test_paths = split["test_paths"]
                test_labels = split["test_labels"]
                print(f"Loaded fixed split (patient-disjoint) from {split_file}")
            else:
                print(f"Loaded fixed split from {split_file}")
        else:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=args.val_split,
                random_state=42, stratify=train_labels
            )
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            with open(split_file, "w") as f:
                _json.dump({"train_paths": train_paths, "val_paths": val_paths,
                            "train_labels": train_labels, "val_labels": val_labels}, f)
            print(f"Created fixed split -> {split_file}")

        print(f"Train: {len(train_paths)} | Val: {len(val_paths)} "
              f"| Test: {len(test_paths)}")
    else:
        print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")
        print("WARNING: --val_split 0 means model selection uses test set (data leakage).")

    # --- Transforms ---
    img_size = (args.image_size, args.image_size)
    aug_ops = []
    if args.tca:
        aug_ops += [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5)]
    if args.aug_type == 'cj01':
        aug_ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    elif args.aug_type == 'cj02':
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    elif args.aug_type == 'sharp':
        aug_ops.append(transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5))
    elif args.aug_type == 'erase':
        aug_ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                                     antialias=True),
        *aug_ops,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Build PIL augmentation pipeline ---
    pil_aug = None
    if args.randaugment:
        pil_aug = transforms.Compose([
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.7),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))],
                p=0.5
            ),
        ])

    train_ds = PneumoniaBinaryDataset(train_paths, train_labels,
                                      transform=train_tf, train=True,
                                      pil_aug=pil_aug)
    test_ds = PneumoniaBinaryDataset(test_paths, test_labels,
                                     transform=test_tf, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              generator=g)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             generator=g)

    val_loader = None
    if val_paths is not None:
        val_ds = PneumoniaBinaryDataset(val_paths, val_labels,
                                        transform=test_tf, train=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                generator=g)

    # --- Model ---
    arch_map = {
        'hybrid': Hybrid,
        'hybrid_msgr': HybridMSGR,
        'hybrid_grp': HybridGRP,
        'hybrid_crc': HybridCRC,
        'hybrid_gr': HybridGR,
        'hybrid_acra': HybridACRA,
        'hybrid_atacra': HybridATACRA,
        'hybrid_cda': HybridCDA,
        'hybrid_stdqkv': HybridStandardQKV,
        'hybrid_atacra_dual': HybridATACRADual,
        'hybrid_variant': HybridVariant,
    }
    arch_labels = {
        'hybrid': 'Hybrid (CNN+Transformer PAFE)',
        'hybrid_msgr': 'HybridMSGR (Multi-Scale + Graph)',
        'hybrid_grp': 'HybridGRP (Graph + Prototype)',
        'hybrid_crc': 'HybridCRC (Contrast + Graph + Prototype)',
        'hybrid_gr': 'HybridGR (Graph Reasoning)',
        'hybrid_acra': 'HybridACRA (Cross-Region Attention)',
        'hybrid_atacra': 'HybridATACRA (Topology-Aware Attention)',
        'hybrid_cda': 'HybridCDA (Contralateral Differential)',
        'hybrid_stdqkv': 'HybridStandardQKV (Learned QKV Control)',
        'hybrid_atacra_dual': 'HybridATACRA-Dual (Dual-Branch Attention)',
        'hybrid_variant': 'HybridVariant (Final Ablations)',
    }
    model_cls = arch_map[args.arch]
    model_kwargs = dict(backbone=args.backbone, num_classes=args.num_classes, num_regions=6,
                       input_size=args.image_size, dropout=args.dropout)
    if args.arch in ('hybrid_grp', 'hybrid_crc'):
        model_kwargs['proto_momentum'] = args.proto_momentum
    if args.arch == 'hybrid_gr':
        model_kwargs.pop('proto_momentum', None)
    if args.arch in ('hybrid_acra', 'hybrid_atacra', 'hybrid_cda', 'hybrid_stdqkv', 'hybrid_atacra_dual'):
        model_kwargs['alpha_init'] = args.alpha_init
        if args.temperature is not None:
            model_kwargs['temperature'] = args.temperature
    if args.arch == 'hybrid_acra' and args.no_gate:
        model_kwargs['no_gate'] = True
    if args.arch == 'hybrid_acra' and args.fixed_alpha:
        model_kwargs['fixed_alpha'] = True
    if args.arch in ('hybrid_atacra', 'hybrid_atacra_dual'):
        model_kwargs['prior_ipsilateral'] = args.prior_ipsi
        model_kwargs['prior_contralateral'] = args.prior_contra
        model_kwargs['freeze_topo'] = args.freeze_topo
        model_kwargs['random_prior'] = args.random_prior
        model_kwargs['shuffle_seed'] = args.shuffle_seed
    if args.arch == 'hybrid_atacra' and args.no_prior:
        model_kwargs['no_prior'] = True
    if args.arch == 'hybrid_stdqkv' and (args.prior_ipsi != 0.5 or args.prior_contra != 0.3):
        model_kwargs['prior_ipsilateral'] = args.prior_ipsi
        model_kwargs['prior_contralateral'] = args.prior_contra
        model_kwargs['freeze_topo'] = args.freeze_topo
        model_kwargs['random_prior'] = args.random_prior
        model_kwargs['shuffle_seed'] = args.shuffle_seed
    model = model_cls(**model_kwargs)
    model = model.to(device)
    # --- Warm-start ACRA classifier from pre-trained Hybrid ---
    if args.arch in ('hybrid_acra', 'hybrid_atacra', 'hybrid_cda', 'hybrid_stdqkv', 'hybrid_atacra_dual', 'hybrid_variant') and args.warmstart:
        model.warmstart_from_hybrid(args.warmstart, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    use_proto = (args.arch in ('hybrid_grp', 'hybrid_crc') and args.proto_lambda > 0)
    use_contrast = (args.arch == 'hybrid_crc' and args.contrast_lambda > 0)
    if args.arch == 'hybrid_gr':
        use_proto = False  # HybridGR doesn't have prototype module
    print(f"Model: {arch_labels[args.arch]} | Params: {n_params:.1f}M")
    config_parts = [f"loss={args.loss}"]
    if args.label_smoothing > 0: config_parts.append(f"label_smoothing={args.label_smoothing}")
    if args.warmup_epochs > 0: config_parts.append(f"warmup={args.warmup_epochs}")
    if args.mixup_alpha > 0: config_parts.append(f"mixup={args.mixup_alpha}")
    if args.randaugment: config_parts.append("randaugment")
    if args.tta: config_parts.append("tta")
    if args.ema_decay > 0: config_parts.append(f"ema={args.ema_decay}")
    config_parts.append(f"dropout={args.dropout}")
    if use_proto: config_parts.append(f"proto_lambda={args.proto_lambda}")
    if use_proto: config_parts.append(f"proto_warmup={args.proto_warmup}")
    if use_contrast: config_parts.append(f"contrast_lambda={args.contrast_lambda}")
    print(f"Config: " + " ".join(config_parts))

    # --- Optimiser (differential LR for ACRA) ---
    if args.arch in ('hybrid_acra', 'hybrid_atacra', 'hybrid_cda', 'hybrid_stdqkv', 'hybrid_atacra_dual', 'hybrid_variant'):
        acra_params = []
        base_params = []
        for name, p in model.named_parameters():
            if name.startswith('acra.'):
                acra_params.append(p)
            else:
                base_params.append(p)
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': args.lr},
            {'params': acra_params, 'lr': args.lr * args.acra_lr_mult},
        ], weight_decay=args.weight_decay)
        print(f"Differential LR: backbone={args.lr:.1e}  ACRA={args.lr * args.acra_lr_mult:.1e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

    # --- Scheduler: warmup+cosine or ReduceLROnPlateau ---
    if args.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        use_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        use_plateau = True

    # --- Loss ---
    if args.class_weights:
        counts = np.bincount(train_labels)
        weights = torch.tensor(
            [sum(counts) / (len(counts) * c) for c in counts],
            dtype=torch.float32).to(device)
        print("Class weights: " + " ".join(
            f"{cn}={weights[ci]:.3f}" for ci, cn in enumerate(class_names)))
        if args.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing,
                                            weight=weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # --- EMA ---
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # --- Checkpoint dir ---
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    monitor_loader = val_loader if val_loader is not None else test_loader
    monitor_name = 'Val' if val_loader is not None else 'Test'

    header = f"{'Epoch':>5s} | {'Train':>7s} {'Loss':>8s} | {monitor_name:>7s} {'Loss':>8s}"
    extras = []
    if args.warmup_epochs <= 0:
        extras.append("{'LR':>8s}")
    extras.append("{'AUC':>7s} {'Spec':>7s}")
    extras.append("{'Time':>6s}")
    header += " | " + " | ".join(extras)
    print(f"\n=== Training ===\n")
    print(header)
    print("-" * (len(header) + 20))

    for epoch in range(args.epochs):
        t0 = time.time()

        # --- Prototype warmup: compute initial prototypes ---
        if use_proto and epoch == args.proto_warmup:
            compute_initial_prototypes(model, train_loader, device)

        # --- Determine proto/contrast lambdas for this epoch ---
        cur_proto_lambda = args.proto_lambda if (use_proto and epoch >= args.proto_warmup) else 0.0
        cur_contrast_lambda = args.contrast_lambda if (use_contrast and epoch >= args.proto_warmup) else 0.0

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            region_aux=args.region_aux,
            mixup_alpha=args.mixup_alpha,
            proto_lambda=cur_proto_lambda,
            contrast_lambda=cur_contrast_lambda,
        )

        if ema is not None:
            ema.update(model)

        # Evaluate on monitor set
        eval_model = ema.ema_model if ema is not None else model
        monitor_metrics = evaluate(eval_model, monitor_loader, criterion,
                                   device, tta=args.tta)

        if use_plateau:
            scheduler.step(monitor_metrics['accuracy'])
        else:
            scheduler.step()

        elapsed = time.time() - t0
        val_acc = monitor_metrics['accuracy']

        marker = " [BEST]" if val_acc > best_val_acc else ""
        lr_str = f" {scheduler.get_last_lr()[0]:.2e}" if use_plateau else ""

        print(f"Epoch {epoch + 1:2d}/{args.epochs} | "
              f"{train_acc:6.1f}% {train_loss:7.4f} | "
              f"{val_acc:6.1f}% {monitor_metrics['loss']:7.4f} |"
              f"{lr_str}"
              f" {monitor_metrics['auc_roc']:.4f}"
              f" {monitor_metrics['specificity']:6.1f}%"
              f" {elapsed:5.0f}s{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model = ema.ema_model if ema is not None else model
            torch.save(save_model.state_dict(),
                       os.path.join(args.save_dir, "binary_pneumonia_best.pth"))

    # --- Final test-set evaluation (load best checkpoint) ---
    print(f"\n=== Best {monitor_name} Accuracy: {best_val_acc:.2f}% ===\n")
    # Rebuild with the same kwargs as training so variant-specific layers
    # (fixed_alpha, no_prior, std-QKV topology, ...) match the checkpoint keys.
    best_model = arch_map[args.arch](**model_kwargs)
    best_model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "binary_pneumonia_best.pth"),
                   map_location='cpu')
    )
    best_model = best_model.to(device)

    final = evaluate(best_model, test_loader, criterion, device, tta=args.tta)
    print(f"{'='*50}")
    print(f"Final Test-Set Evaluation ({len(test_labels)} images, "
          f"{args.num_classes} classes)")
    print(f"{'='*50}")
    print(f"Accuracy:     {final['accuracy']:.2f}%")
    if args.num_classes == 2:
        print(f"Sensitivity:  {final['sensitivity']:.2f}%")
        print(f"Specificity:  {final['specificity']:.2f}%")
        print(f"Precision:    {final['precision']:.2f}%")
        print(f"F1 Score:     {final['f1']:.2f}%")
        print(f"AUC-ROC:      {final['auc_roc']:.4f}")
        print(f"AUC-PR:       {final['auc_pr']:.4f}")
        print(f"TP:{final['tp']}  TN:{final['tn']}  FP:{final['fp']}  FN:{final['fn']}")
    else:
        print(f"Macro Sens:   {final['sensitivity']:.2f}%")
        print(f"Macro Spec:   {final['specificity']:.2f}%")
        print(f"Macro Prec:   {final['precision']:.2f}%")
        print(f"Macro F1:     {final['f1']:.2f}%")
        print(f"AUC-ROC(ovr): {final['auc_roc']:.4f}")
        print(f"AUC-PR(macro):{final['auc_pr']:.4f}")
        for c in range(args.num_classes):
            m = final['per_class'][c]
            print(f"  Class {c} ({class_names[c]}): Acc={100.0*(m['tp']+m['tn'])/len(test_labels):.2f}% "
                  f"Sens={m['sensitivity']:.2f}% Spec={m['specificity']:.2f}% "
                  f"F1={m['f1']:.2f}% TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
