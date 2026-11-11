"""Thesis figure set v3 (corrected 3x2 ROI, train-identical pipeline).

Figures (figures/paper_heatmaps_v3/):
  fig1_gradcam_compare.png  — Grad-CAM: Hybrid/ACRA/ATACRA (matched protocol)
                              + ATACRA(Warmup3); unified pneumonia target
  fig2_attention_M.png      — per-case M0/MA/M/M-MA matrices + CXR+ROI
  fig2b_attention_classmean.png — class-averaged matrices + aggregation bar chart
  fig3_delta_stability.png  — row-centered Delta mean/std/SNR (3 decimals)
  fig3b_delta_per_seed.png  — appendix: 5 seeds, shared colorbar
  fig4_confmat_binary.png   — binary confusion (1-decimal counts + row-%)
  fig5_confmat_3class.png   — 3-class confusion
Train pipeline: ToTensor -> Resize(tensor) -> Normalize (identical to training).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['AR PL UMing CN', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
from scipy.ndimage import zoom
import torch, torch.nn.functional as F
import sys, os, json
from PIL import Image
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, '/home/raid-5/MYQ/pneumonia-miccai2024')
from model import Hybrid, HybridACRA, HybridATACRA
from model.hybrid_model import _get_roi_boxes

BASE = '/home/raid-5/MYQ/pneumonia-miccai2024'
OUT = f'{BASE}/figures/paper_heatmaps_v3'
CACHE = f'{OUT}/cache'
os.makedirs(CACHE, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 3407, 2024, 2025, 2026]
rlab = ['A', 'B', 'C', 'D', 'E', 'F']

# train-identical pipeline: ToTensor -> Resize(tensor, antialias) -> Normalize
tf_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build(model_dir, num_classes=2):
    if model_dir.startswith('hybrid_v3') or model_dir.startswith('hybrid_3class'):
        return Hybrid(backbone='resnet50', num_classes=num_classes, num_regions=6,
                      input_size=512, dropout=0.3)
    if model_dir.startswith('acra_v3'):
        return HybridACRA(backbone='resnet50', num_classes=num_classes, num_regions=6,
                          input_size=512, dropout=0.3, alpha_init=0.1)
    if model_dir.startswith('atacra'):
        return HybridATACRA(backbone='resnet50', num_classes=num_classes, num_regions=6,
                            input_size=512, dropout=0.3, alpha_init=0.1,
                            prior_ipsilateral=0.2, prior_contralateral=0.1)
    raise ValueError(model_dir)


def load_ckpt(model_dir, seed):
    return torch.load(f'{BASE}/results/{model_dir}/ckpt_seed{seed}/binary_pneumonia_best.pth',
                      map_location='cpu')


def predict_probs(model, paths, batch=16, tta=True):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = torch.stack([tf_norm(Image.open(p).convert('RGB'))
                              for p in paths[i:i + batch]]).to(device)
            o = model(xs).mean(dim=1)
            if tta:
                o = (o + model(torch.flip(xs, dims=[3])).mean(dim=1)) / 2
            out.append(F.softmax(o, dim=1).cpu().numpy())
    return np.concatenate(out)


split_bin = json.load(open(f'{BASE}/data_splits/split_disjoint_70_15_15.json'))
bin_paths, bin_labels = split_bin['test_paths'], np.array(split_bin['test_labels'])
split_tri = json.load(open(f'{BASE}/data_splits/split_3class.json'))
tri_paths, tri_labels = split_tri['test_paths'], np.array(split_tri['test_labels'])


def cached_probs(name, model_dir, paths, n_classes):
    f = f'{CACHE}/{name}.npy'
    if os.path.exists(f):
        return np.load(f)
    probs = []
    for s in SEEDS:
        m = build(model_dir, n_classes)
        m.load_state_dict(load_ckpt(model_dir, s), strict=False)
        m = m.to(device)
        probs.append(predict_probs(m, paths))
        del m
        torch.cuda.empty_cache()
        print(f'  {name} seed{s} done', flush=True)
    arr = np.stack(probs)
    np.save(f, arr)
    return arr


print('Caching v3 predictions...', flush=True)
p_hyb_bin = cached_probs('v3_bin_hybrid', 'hybrid_v3', bin_paths, 2)
p_ata_bin = cached_probs('v3_bin_atacra_w3', 'atacra_v3_warmup3', bin_paths, 2)
p_hyb_tri = cached_probs('v3_tri_hybrid', 'hybrid_3class_v3', tri_paths, 3)
p_ata_tri = cached_probs('v3_tri_atacra_w3', 'atacra_3class_v3_warmup3', tri_paths, 3)

# ============================================================
# Figure 1: Grad-CAM comparison (unified pneumonia target)
# ============================================================
def grad_cam(model, img_t, class_idx=1):
    features, grads = [], []
    target = model.backbone[7][-1].conv3
    h1 = target.register_forward_hook(lambda m, i, o: features.append(o))
    h2 = target.register_full_backward_hook(lambda m, i, o: grads.append(o[0]))
    x = img_t.unsqueeze(0).to(device)
    x.requires_grad = True
    out = model(x)
    score = out.mean(dim=1)[0, class_idx]      # pre-softmax pneumonia logit
    model.zero_grad()
    score.backward()
    h1.remove(); h2.remove()
    fmap, grad = features[0][0], grads[0][0]
    w = grad.mean(dim=(1, 2))
    cam = (w.view(-1, 1, 1) * fmap).sum(dim=0)
    cam = F.relu(cam).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return zoom(cam, 512 / cam.shape[0], order=1)   # bilinear to input size


print('Figure 1: Grad-CAM...', flush=True)
gcam_dirs = [('Hybrid', 'hybrid_v3'), ('ACRA', 'acra_v3'),
             ('ATACRA', 'atacra_v3'), ('ATACRA(Warmup3)', 'atacra_v3_warmup3')]
gcam_nets = {}
for name, d in gcam_dirs:
    m = build(d)
    m.load_state_dict(load_ckpt(d, 42), strict=False)
    gcam_nets[name] = m.to(device).eval()

pred_h = p_hyb_bin[SEEDS.index(42)].argmax(1)
pred_a = p_ata_bin[SEEDS.index(42)].argmax(1)
conf_a = p_ata_bin[SEEDS.index(42)].max(1)
cases = []
idx = np.where((bin_labels == 0) & (pred_h == 0) & (pred_a == 0))[0]
cases.append(idx[np.argmax(conf_a[idx])])
idx = np.where((bin_labels == 1) & (pred_h == 1) & (pred_a == 1))[0]
cases.append(idx[np.argmax(conf_a[idx])])
idx = np.where((bin_labels != pred_h) & (pred_a == bin_labels))[0]
if len(idx):
    cases.append(idx[np.argmax(conf_a[idx])])
idx = np.where(bin_labels != pred_a)[0]
if len(idx):
    cases.append(idx[np.argmax(conf_a[idx])])
cases = list(dict.fromkeys(cases))
print(f'  cases: {[os.path.basename(bin_paths[i]) for i in cases]}', flush=True)

n_rows, n_cols = len(cases), 1 + len(gcam_nets)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4.5 * n_rows))
if n_rows == 1:
    axes = axes[None, :]
for r, ci in enumerate(cases):
    p = bin_paths[ci]
    img = np.array(Image.open(p).convert('RGB').resize((512, 512))) / 255.0
    gray = img.mean(axis=2)
    t = tf_norm(Image.open(p).convert('RGB'))
    axes[r, 0].imshow(gray, cmap='gray')
    axes[r, 0].set_title(f"Original CXR\nGT={'Normal' if bin_labels[ci]==0 else 'Pneumonia'}",
                         fontsize=10)
    axes[r, 0].axis('off')
    for c, (name, _) in enumerate(gcam_dirs, start=1):
        m = gcam_nets[name]
        with torch.no_grad():
            prob = F.softmax(m(t.unsqueeze(0).to(device)).mean(dim=1), dim=1)[0]
        cam = grad_cam(m, t, class_idx=1)
        axes[r, c].imshow(gray, cmap='gray')
        axes[r, c].imshow(cam, cmap='jet', alpha=0.4)
        axes[r, c].set_title(f"{name}\nPred={'Normal' if prob.argmax()==0 else 'Pneumonia'}, "
                             f"P(Pneumonia)={prob[1]:.3f}", fontsize=10)
        axes[r, c].axis('off')
        if c == 2 and r == 0:
            axes[r, c].text(512, -45, 'matched protocol', ha='center', fontsize=9,
                            color='gray')
        if c == 4 and r == 0:
            axes[r, c].text(512, -45, 'final (Warmup3)', ha='center', fontsize=9,
                            color='gray')
fig.suptitle('Grad-CAM comparison (target class = pneumonia logit for ALL models; '
             'per-image min-max, colors only comparable within each panel)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUT}/fig1_gradcam_compare.png', dpi=200)
plt.close(fig)

# ============================================================
# Figure 2: attention matrices M0 / MA / M / M-MA
# ============================================================
print('Figure 2: attention matrices...', flush=True)
m = build('atacra_v3_warmup3')
m.load_state_dict(load_ckpt('atacra_v3_warmup3', 42), strict=False)
m = m.to(device).eval()
A = m.acra.anatomy_prior.cpu().numpy()
D = m.acra.topo_bias.detach().cpu().numpy()
tau = m.acra.temperature.item()


def get_mats(img_t):
    with torch.no_grad():
        R = m(img_t.unsqueeze(0).to(device), embedding=True)[0]
        S = torch.einsum('id,jd->ij', R, R).cpu().numpy() / tau
    def rsoft(X):
        e = np.exp(X - X.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return rsoft(S), rsoft(S + A), rsoft(S + A + D)


def draw_mat(ax, mat, ttl, cm, vmin=None, vmax=None):
    im = ax.imshow(mat, cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(6)); ax.set_xticklabels(rlab, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(rlab, fontsize=8)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('Source region j')
    ax.set_ylabel('Receiving region i', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im


boxes_px = [(int(b[1] * 512), int(b[0] * 512), int(b[3] * 512), int(b[2] * 512))
            for b in _get_roi_boxes(6, torch.device('cpu'))]
idx_n = np.where((bin_labels == 0) & (pred_a == 0))[0]
i_n = idx_n[np.argmax(conf_a[idx_n])]
idx_p = np.where((bin_labels == 1) & (pred_a == 1))[0]
i_p = idx_p[np.argmax(conf_a[idx_p])]
cases2 = [('Normal', i_n), ('Pneumonia', i_p)]
fig, axes = plt.subplots(2, 5, figsize=(20, 8.6))
for r, (name, ci) in enumerate(cases2):
    gray = np.array(Image.open(bin_paths[ci]).convert('RGB').resize((512, 512))).mean(axis=2) / 255.0
    axes[r, 0].imshow(gray, cmap='gray')
    for (x1, y1, x2, y2), l in zip(boxes_px, rlab):
        axes[r, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor='cyan', lw=1))
        axes[r, 0].text((x1 + x2) / 2, y1 - 12, l, color='cyan', ha='center', fontsize=8)
    axes[r, 0].set_title(f'{name} case\n(ROI layout A-F)', fontsize=10)
    axes[r, 0].axis('off')
    M0, MA, M = get_mats(tf_norm(Image.open(bin_paths[ci]).convert('RGB')))
    diff = M - MA
    lim = np.abs(diff).max()
    draw_mat(axes[r, 1], M0, 'M0 = softmax(S)', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 2], MA, 'MA = softmax(S+A)', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 3], M, 'M = softmax(S+A+Δ)', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 4], diff, 'M − MA (effect of Δ)', 'RdBu_r', -lim, lim)
fig.suptitle('Cross-region attention (fixed R, τ from final model seed42; '
             'only bias terms toggled at inference — not a retraining ablation)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT}/fig2_attention_M.png', dpi=200)
plt.close(fig)

# class-averaged + aggregation bar chart
print('  class-averaged M...', flush=True)
M0s, MAs, Ms = [], [], []
with torch.no_grad():
    for ci in range(0, len(bin_paths), 16):
        xs = torch.stack([tf_norm(Image.open(p).convert('RGB'))
                          for p in bin_paths[ci:ci + 16]]).to(device)
        R = m(xs, embedding=True)
        S = torch.einsum('bid,bjd->bij', R, R).cpu().numpy() / tau
        for k in range(S.shape[0]):
            s = S[k]
            def rsoft(X):
                e = np.exp(X - X.max(1, keepdims=True))
                return e / e.sum(1, keepdims=True)
            M0s.append(rsoft(s)); MAs.append(rsoft(s + A)); Ms.append(rsoft(s + A + D))
M0s, MAs, Ms = np.stack(M0s), np.stack(MAs), np.stack(Ms)
fig, axes = plt.subplots(2, 5, figsize=(20, 8.6))
for r, (mask, ttl) in enumerate([(bin_labels == 0, 'Normal (class mean)'),
                                 (bin_labels == 1, 'Pneumonia (class mean)')]):
    m0, ma, mm = M0s[mask].mean(0), MAs[mask].mean(0), Ms[mask].mean(0)
    # bar chart: source-region aggregation weights q_j = mean_i M_ij
    q0 = m0.mean(axis=0)
    qa = ma.mean(axis=0)
    qm = mm.mean(axis=0)
    ax = axes[r, 0]
    x = np.arange(6)
    ax.bar(x - 0.25, q0, 0.22, label='M0', color='#FFC107')
    ax.bar(x, qa, 0.22, label='MA', color='#66BB6A')
    ax.bar(x + 0.25, qm, 0.22, label='M', color='#42A5F5')
    ax.set_xticks(x); ax.set_xticklabels(rlab)
    ax.set_ylim(0, 0.3)
    ax.set_title(f'{ttl}\nSource-region aggregation weights\nq_j = mean_i M_ij', fontsize=10)
    ax.legend(fontsize=8)
    diff = mm - ma
    lim = np.abs(diff).max()
    draw_mat(axes[r, 1], m0, 'M0', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 2], ma, 'MA', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 3], mm, 'M', 'YlOrRd', 0, 1)
    draw_mat(axes[r, 4], diff, 'M − MA', 'RdBu_r', -lim, lim)
fig.suptitle('Class-averaged cross-region attention (all 872 test images, final model seed42)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT}/fig2b_attention_classmean.png', dpi=200)
plt.close(fig)

# ============================================================
# Figure 3: Delta cross-seed stability (row-centered)
# ============================================================
print('Figure 3: Delta stability...', flush=True)
deltas = []
for s in SEEDS:
    mm = build('atacra_v3_warmup3')
    mm.load_state_dict(load_ckpt('atacra_v3_warmup3', s), strict=False)
    d = mm.get_topology_report().numpy() - mm.acra.anatomy_prior.cpu().numpy()
    d = d - d.mean(axis=1, keepdims=True)
    deltas.append(d)
deltas = np.stack(deltas)
d_mean, d_std = deltas.mean(0), deltas.std(0)
snr = np.abs(d_mean) / (d_std + 1e-8)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
lim = np.abs(d_mean).max()
for ax, mat, ttl, cm, dec in zip(
        axes, [d_mean, d_std, snr],
        ['Row-centered Δ mean (5 seeds)', 'Row-centered Δ std (5 seeds)', 'SNR = |mean|/std'],
        ['RdBu_r', 'YlOrRd', 'YlOrRd'], ['+.3f', '.3f', '.1f']):
    im = ax.imshow(mat, cmap=cm, vmin=-lim if cm == 'RdBu_r' else 0,
                   vmax=lim if cm == 'RdBu_r' else None)
    ax.set_xticks(range(6)); ax.set_xticklabels(rlab)
    ax.set_yticks(range(6)); ax.set_yticklabels(rlab)
    ax.set_title(ttl, fontsize=11)
    for i in range(6):
        for j in range(6):
            v = mat[i, j]
            if cm == 'RdBu_r':
                col = 'k' if abs(v) < lim * 0.6 else 'w'
                txt = format(v, '+.3f')
            else:
                col = 'k' if v < mat.max() * 0.6 else 'w'
                txt = format(v, '.3f' if cm == 'YlOrRd' and dec == '.3f' else '.1f')
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=col)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Δ cross-seed stability (row-centered; final model, Warmup3, 3×2 ROI)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{OUT}/fig3_delta_stability.png', dpi=200)
plt.close(fig)

# appendix: per-seed, shared colorbar
fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
glim = np.abs(deltas).max()
ims = []
for ax, s, d in zip(axes, SEEDS, deltas):
    ims.append(ax.imshow(d, cmap='RdBu_r', vmin=-glim, vmax=glim))
    ax.set_xticks(range(6)); ax.set_xticklabels(rlab, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(rlab, fontsize=8)
    ax.set_title(f'seed {s}', fontsize=10)
for im in ims:
    im.set_clim(-glim, glim)
fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.suptitle('Appendix: row-centered Δ per seed (shared color scale)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{OUT}/fig3b_delta_per_seed.png', dpi=200)
plt.close(fig)

# ============================================================
# Figure 4 & 5: confusion matrices (1-decimal counts + row-%)
# ============================================================
print('Figure 4/5: confusion matrices...', flush=True)


def confmat_figure(probs_hyb, probs_ata, labels, class_names, title, fname):
    n_c = len(class_names)
    mats = []
    for P in (probs_hyb, probs_ata):
        cm = np.zeros((n_c, n_c))
        for k in range(P.shape[0]):
            pred = P[k].argmax(1)
            for i in range(n_c):
                for j in range(n_c):
                    cm[i, j] += ((labels == i) & (pred == j)).sum()
        mats.append(cm / P.shape[0])
    vmax = max(cm.max() for cm in mats)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for r, (P, name) in enumerate(zip((probs_hyb, probs_ata), ('Hybrid', 'ATACRA'))):
        cm = mats[r]
        ax = axes[r, 0]
        im = ax.imshow(cm, cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_title(f'{name}: mean counts (5 seeds)', fontsize=11)
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, f'{cm[i,j]:.1f}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                        ha='center', va='center', fontsize=9)
        ax.set_xticks(range(n_c)); ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticks(range(n_c)); ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[r, 1]
        prop = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(prop, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{name}: row-normalized (5 seeds)', fontsize=11)
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, f'{prop[i,j]*100:.1f}%', ha='center', va='center', fontsize=10)
        ax.set_xticks(range(n_c)); ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticks(range(n_c)); ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.colorbar(im, ax=ax, fraction=0.046)
        # reconciliation check
        acc = np.trace(cm) / cm.sum() * 100
        print(f'  {name} ({fname}): matrix-derived Acc = {acc:.2f}%', flush=True)
    fig.suptitle(title + '\n（5-seed 平均，TTA 翻转平均，训练管线一致）', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/{fname}', dpi=200)
    plt.close(fig)


confmat_figure(p_hyb_bin, p_ata_bin, bin_labels, ['Normal', 'Pneumonia'],
               'Binary confusion matrices', 'fig4_confmat_binary.png')
confmat_figure(p_hyb_tri, p_ata_tri, tri_labels, ['COVID19', 'Normal', 'Pneumonia'],
               '3-class confusion matrices', 'fig5_confmat_3class.png')

print(f'\nAll v3 figures saved to {OUT}/', flush=True)
