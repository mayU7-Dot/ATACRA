"""Re-run final test evaluation for training runs that crashed at final eval.

Training completed and the best checkpoint was saved; only the final-eval
reconstruction crashed (state_dict key mismatch). This rebuilds the model
with the correct variant kwargs, evaluates on the disjoint test set with TTA,
and appends a "Final Test-Set Evaluation" block to the seed log.
"""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '/home/raid-5/MYQ/pneumonia-miccai2024')
from model import HybridACRA, HybridStandardQKV

BASE = '/home/raid-5/MYQ/pneumonia-miccai2024'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize((512, 512), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build(model_dir):
    if model_dir.startswith('zeroqkv_fixed'):
        return HybridACRA(backbone='resnet50', num_classes=2, num_regions=6,
                          input_size=512, dropout=0.3, alpha_init=0.1,
                          fixed_alpha=True)
    if model_dir.startswith('stdqkv_atacra'):
        return HybridStandardQKV(backbone='resnet50', num_classes=2, num_regions=6,
                                 input_size=512, dropout=0.3, alpha_init=0.1,
                                 prior_ipsilateral=0.2, prior_contralateral=0.1)
    raise ValueError(f'unknown model dir {model_dir}')


def evaluate_ckpt(model, test_paths, test_labels):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_paths), 16):
            imgs = [tf(Image.open(p).convert('RGB')) for p in test_paths[i:i+16]]
            x = torch.stack(imgs).to(device)
            out = model(x).mean(dim=1)
            out_f = model(torch.flip(x, dims=[3])).mean(dim=1)
            out = (out + out_f) / 2.0
            probs.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.array(test_labels)
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return dict(
        accuracy=100.0 * (tp + tn) / len(labels),
        sensitivity=100.0 * tp / (tp + fn + 1e-8),
        specificity=100.0 * tn / (tn + fp + 1e-8),
        precision=100.0 * tp / (tp + fp + 1e-8),
        f1=100.0 * 2 * tp / (2 * tp + fp + fn + 1e-8),
        auc_roc=roc_auc_score(labels, probs),
        auc_pr=average_precision_score(labels, probs),
        tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    split = json.load(open(f'{BASE}/data_splits/split_disjoint_70_15_15.json'))
    test_paths, test_labels = split['test_paths'], split['test_labels']
    targets = sys.argv[1:]
    if not targets:
        print('usage: python eval_only_crashed.py MODEL_DIR [MODEL_DIR ...] (5 seeds each)')
        return
    for model_dir in targets:
        for seed in [42, 3407, 2024, 2025, 2026]:
            ckpt = f'{BASE}/results/{model_dir}/ckpt_seed{seed}/binary_pneumonia_best.pth'
            logfile = f'{BASE}/results/{model_dir}/seed{seed}.txt'
            if not os.path.exists(ckpt):
                print(f'[skip] {model_dir} seed{seed}: no checkpoint')
                continue
            if os.path.exists(logfile) and 'Final Test-Set' in open(logfile).read():
                print(f'[skip] {model_dir} seed{seed}: already has final eval')
                continue
            model = build(model_dir)
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            model = model.to(device)
            r = evaluate_ckpt(model, test_paths, test_labels)
            block = (f"\n{'='*50}\n"
                     f"Final Test-Set Evaluation ({len(test_labels)} images, 2 classes)\n"
                     f"{'='*50}\n"
                     f"Accuracy:     {r['accuracy']:.2f}%\n"
                     f"Sensitivity:  {r['sensitivity']:.2f}%\n"
                     f"Specificity:  {r['specificity']:.2f}%\n"
                     f"Precision:    {r['precision']:.2f}%\n"
                     f"F1 Score:     {r['f1']:.2f}%\n"
                     f"AUC-ROC:      {r['auc_roc']:.4f}\n"
                     f"AUC-PR:       {r['auc_pr']:.4f}\n"
                     f"{'='*50}\n"
                     f"TP:{r['tp']}  TN:{r['tn']}  FP:{r['fp']}  FN:{r['fn']}\n")
            with open(logfile, 'a') as f:
                f.write(block)
            print(f'[eval-done] {model_dir} seed{seed}: '
                  f"Acc={r['accuracy']:.2f} Sens={r['sensitivity']:.2f} "
                  f"Spec={r['specificity']:.2f} F1={r['f1']:.2f} AUC={r['auc_roc']:.4f}")
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
