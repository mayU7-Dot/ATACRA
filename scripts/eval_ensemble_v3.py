"""Fair ensemble comparison: hybrid_v3 vs atacra_v3_warmup3 (10 seeds each).

Train-identical pipeline (ToTensor -> Resize(tensor) -> Normalize), TTA flip.
Reports Acc/Sens/Spec/F1/AUC at threshold 0.5 and val-optimized threshold.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '/home/raid-5/MYQ/pneumonia-miccai2024')
from model import Hybrid, HybridATACRA

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE = '/home/raid-5/MYQ/pneumonia-miccai2024'
SEEDS = [42, 3407, 2024, 2025, 2026, 1234, 777, 888, 999, 31415]

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_hybrid():
    return Hybrid(backbone='resnet50', num_classes=2, num_regions=6,
                  input_size=512, dropout=0.3)


def build_atacra():
    return HybridATACRA(backbone='resnet50', num_classes=2, num_regions=6,
                        input_size=512, dropout=0.3, alpha_init=0.1,
                        prior_ipsilateral=0.2, prior_contralateral=0.1)


def predict_probs(model, paths, batch=16):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = torch.stack([tf(Image.open(p).convert('RGB'))
                              for p in paths[i:i + batch]]).to(device)
            o = model(xs).mean(dim=1)
            o = (o + model(torch.flip(xs, dims=[3])).mean(dim=1)) / 2
            out.append(F.softmax(o, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(probs, labels, thr=0.5):
    preds = (probs >= thr).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    acc = 100.0 * (tp + tn) / len(labels)
    sens = 100.0 * tp / (tp + fn + 1e-8)
    spec = 100.0 * tn / (tn + fp + 1e-8)
    f1 = 100.0 * 2 * tp / (2 * tp + fp + fn + 1e-8)
    return acc, sens, spec, f1, int(tp), int(tn), int(fp), int(fn)


def main():
    split = json.load(open(f'{BASE}/data_splits/split_disjoint_70_15_15.json'))
    val_paths, val_labels = split['val_paths'], np.array(split['val_labels'])
    test_paths, test_labels = split['test_paths'], np.array(split['test_labels'])

    results = {}
    for name, builder, ckpt_dir in [('hybrid_v3', build_hybrid, 'hybrid_v3'),
                                    ('atacra_v3_warmup3', build_atacra, 'atacra_v3_warmup3')]:
        val_probs, test_probs = [], []
        for s in SEEDS:
            m = builder()
            m.load_state_dict(torch.load(
                f'{BASE}/results/{ckpt_dir}/ckpt_seed{s}/binary_pneumonia_best.pth',
                map_location='cpu'), strict=False)
            m = m.to(device)
            val_probs.append(predict_probs(m, val_paths))
            test_probs.append(predict_probs(m, test_paths))
            del m
            torch.cuda.empty_cache()
            print(f'{name} seed{s} done', flush=True)
        ens_val = np.stack(val_probs).mean(0)
        ens_test = np.stack(test_probs).mean(0)
        acc, sens, spec, f1, tp, tn, fp, fn = metrics(ens_test, test_labels)
        print(f'\n=== {name} ensemble ({len(SEEDS)} seeds), thr=0.5 ===')
        print(f'Acc={acc:.2f}  Sens={sens:.2f}  Spec={spec:.2f}  F1={f1:.2f}  '
              f'TP={tp} TN={tn} FP={fp} FN={fn}')
        best_t = max(((t, metrics(ens_val, val_labels, t)[0])
                      for t in np.arange(0.1, 0.91, 0.01)), key=lambda x: x[1])[0]
        acc2, sens2, spec2, f1_, tp2, tn2, fp2, fn2 = metrics(ens_test, test_labels, best_t)
        print(f'val-opt thr={best_t:.2f}: Acc={acc2:.2f}  Sens={sens2:.2f}  Spec={spec2:.2f}  FP={fp2}')
        print(f'AUC-ROC={roc_auc_score(test_labels, ens_test):.4f}  '
              f'AUC-PR={average_precision_score(test_labels, ens_test):.4f}')
        results[name] = dict(acc=acc, ens_test=ens_test)
    print(f'\nGap at thr=0.5: '
          f'{results["atacra_v3_warmup3"]["acc"] - results["hybrid_v3"]["acc"]:+.2f}pp')


if __name__ == '__main__':
    main()
