# ATACRA: Anatomical Topology-Aware Zero-Parameter Cross-Region Attention

Official code for **"ATACRA: Anatomical Topology-Aware Zero-Parameter Cross-Region Attention for Pediatric Pneumonia Detection in Chest Radiographs"**.

ATACRA adds a cross-region attention module over the 6 anatomical lung regions of the PAFE architecture with only **38 learnable parameters**:

- **Zero-parameter QKV**: identity projection `Q=K=V=R`, `Attn = softmax(RRᵀ/τ)` (τ learnable scalar)
- **α residual gate**: `R' = R + α·Attn(R)·R` (α learnable, init 0.1)
- **Anatomical prior A** (fixed): diagonal 1.0 self-attention, 0.2 ipsilateral vertical adjacency (A-B, B-C, D-E, E-F), 0.1 contralateral mirror (A-D, B-E, C-F); strictly invariant under horizontal flip (permutation σ=[3,4,5,0,1,2], verified max|A−A_perm|=0)
- **Learnable topology deviation Δ** (36 params, zero-init): `Attn = softmax(RRᵀ/τ + A + Δ)`

ROI layout (3 rows × 2 columns, BrixIA convention):

```
+-------+-------+
|   A   |   D   |   upper band
+-------+-------+
|   B   |   E   |   middle band
+-------+-------+
|   C   |   F   |   lower band
+-------+-------+
```

## Main Results (patient-disjoint protocol, n=10 seeds, paired t-test)

| Metric | Hybrid | ATACRA (Warmup3) | Δ (p) |
|--------|--------|------------------|-------|
| Accuracy | 96.90±0.42 | **97.63±0.35** | **+0.72 (p=0.0035)** |
| Sensitivity | 96.65±0.76 | 98.09±0.47 | +1.44 |
| F1 | 97.81±0.30 | 98.33±0.25 | +0.52 |
| AUC-ROC | 0.9959 | 0.9966 | +0.0007 |

3-class (COVID19/NORMAL/PNEUMONIA): 96.48±0.27 → **99.02±0.17 (+2.55pp, p=0.0001)**.

Fair 10-seed ensemble: Hybrid 96.56% → ATACRA **97.71%** (+1.15pp); val-calibrated threshold: **98.05%**.

Parameter efficiency: standard QKV (+1,769,510 params) achieves 97.39±0.52 vs zero-param 97.13±0.35 (p=0.164, n.s.) — comparable accuracy at **1/46,566** of the parameters.

## Environment

```
python>=3.10
torch>=2.0
torchvision>=0.15
numpy
scipy
scikit-learn
matplotlib
pillow
pandas
```

## Data

1. **Binary**: Kaggle pediatric pneumonia dataset (`chest_xray/` with `train/{NORMAL,PNEUMONIA}` and `test/...`), place under `--data_path`.
2. **3-class**: COVID19/NORMAL/PNEUMONIA folders under `Data/{train,test}`.

Patient-disjoint splits are provided in `data_splits/`:
- `split_disjoint_70_15_15.json` (binary, zero patient overlap between train/val/test)
- `split_3class.json` (image-level stratified split)

## Training

All models: ResNet50 backbone, 512×512, batch 16, 30 epochs, lr 1e-5 (attention layers ×100), class weights, TTA horizontal flip, best-val selection, warmstart fork from a Hybrid checkpoint (`checkpoints/warmstart_clean_v2.pth`).

Main pair (n=10 seeds; base 5 + extra 5):

```bash
python scripts/run_retrain_v3.py          # base 5 seeds, all 12 model configs
python scripts/run_extra_seeds.py         # extra 5 seeds for the main pair
```

Ablation rows (E0–E9) are defined in `scripts/run_retrain_v3.py` (MODELS dict):
- E0 Hybrid (plateau + LS0.1), E1 fixed-α residual, E2 ACRA, E3 w/o Δ, E4 w/o A,
- E5 **ATACRA (Warmup3)**, E6 standard QKV + A + Δ, E7 random prior, E8 random prior w/o Δ, E9 ATACRA (plateau)

Single-run example (E5, one seed):

```bash
python train_binary.py --data_path /path/to/chest_xray \
  --arch hybrid_atacra --backbone resnet50 --image_size 512 \
  --batch_size 16 --epochs 30 --lr 1e-5 --weight_decay 1e-4 \
  --num_workers 4 --val_split 0.2 --warmup_epochs 3 --tta \
  --acra_lr_mult 100 --alpha_init 0.1 --prior_ipsi 0.2 --prior_contra 0.1 \
  --seed 42 --split_file data_splits/split_disjoint_70_15_15.json \
  --no_balance --class_weights --warmstart checkpoints/warmstart_clean_v2.pth \
  --save_dir results/atacra_v3_warmup3/ckpt_seed42/
```

## Evaluation

```bash
# Fair 10-seed ensemble (train-identical pipeline)
python scripts/eval_ensemble_v3.py

# Thesis figures (Grad-CAM, attention M matrices, Delta stability, confusion matrices)
python scripts/gen_paper_figures_v3.py
```

## Results summary

Full tables (ablation E0–E9 with paired t-tests, parameter efficiency, 3-class per-class breakdown, per-seed details): see `实验结果汇总.md`.

## Weights

Trained checkpoints (v3, corrected 3×2 ROI layout) are released separately (see release assets): 12 model configurations × 5–10 seeds. Each checkpoint `binary_pneumonia_best.pth` is a ResNet50+PAFE model (~130 MB).

## Important notes

- **v3 ROI correction**: earlier v2 runs had a coordinate-swap bug in `_get_roi_boxes` (executed 2×3 layout); v3 fixes this to the true 3×2 layout. All reported results are v3. Old v2 results are retained in `results/*_v2/` for history only.
- The disjoint split removes 214 patient overlaps present in the official Kaggle split (see paper for the leakage audit).
- Grad-CAM/attention figures are descriptive analyses; attention weights are not causal explanations of disease spread.
