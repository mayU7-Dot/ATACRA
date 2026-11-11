"""Complete hybrid model: CNN backbone + Vision Transformer + multi-region ROI.

Implements the MICCAI 2024 architecture:
  1. Segmentation (landmark-based lung/heart masks)
  2. Spatial Transformer Network for alignment
  3. CNN feature extraction
  4. Transformer Position-Aware Feature Encoding (PAFE)
  5. Multi-region ROI pooling & severity classification

Extended with deep architectural innovations:
  - MultiScaleFusion: FPN-style feature pyramid across ResNet layers 2/3/4
  - GraphRegionReasoning: GCN over 6 lung regions with learned importance
  - HybridMSGR: Combined multi-scale + graph reasoning model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbedding, TransformerEncoderBlock
from .backbone import model_dict
from .stn import SpatialTransformer


# ---------------------------------------------------------------------------
# Lightweight pure-CNN baseline (no transformer, no STN)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """CNN-only baseline with multi-region ROI pooling."""

    def __init__(self, backbone='densenet121', num_classes=4, num_regions=6):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions

        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(*list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, self.num_classes)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]

        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False):
        feat = self.backbone(x)
        rois = self.pool_rois(feat)

        outputs = []
        for i in range(self.num_regions):
            pred = self.score_GAP(rois[:, i])
            pred = pred.view(-1, self.fc.in_features)
            if not embedding:
                pred = self.fc(pred)
            outputs.append(pred)
        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Hybrid CNN-Transformer (no STN)
# ---------------------------------------------------------------------------

class Hybrid(nn.Module):
    """CNN backbone + Transformer PAFE + multi-region ROI pooling.

    Parameters
    ----------
    backbone : str
        One of 'resnet18', 'resnet34', 'resnet50', 'densenet121', etc.
    num_classes : int
        Number of severity classes (4 for Brixia, 2 for binary).
    num_regions : int
        Number of lung sub-regions (6 or 4).
    input_size : int
        Input image size (H==W). Used to compute the feature-map spatial
        size for the transformer positional embedding.
        ResNet/DenseNet downsample by 32x, so input_size must be divisible by 32.
    """

    def __init__(self, backbone='densenet121', num_classes=4, num_regions=6,
                 input_size=512, dropout=0.3):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2]
            )

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, self.num_classes)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)

        outputs = []
        for i in range(self.num_regions):
            pred = self.score_GAP(rois[:, i])
            pred = pred.view(-1, 768)
            if not embedding:
                pred = self.dropout(pred)
                pred = self.fc(pred)
            outputs.append(pred)
        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Full end-to-end Hybrid + STN + Segmentation
# ---------------------------------------------------------------------------

class HybridE2ESTN(nn.Module):
    """Complete MICCAI 2024 pipeline.

    Segmentation -> STN alignment -> CNN -> Transformer PAFE -> ROI pooling -> scores
    """

    def __init__(self, backbone='resnet50', num_classes=4, num_regions=6,
                 input_size=512, seg_model=None, stn_checkpoint=None):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        # --- Segmentation model (pre-trained, frozen) ---
        self.seg_model = seg_model
        if self.seg_model is not None:
            for p in self.seg_model.parameters():
                p.requires_grad = False

        # --- STN module ---
        self.stn = SpatialTransformer(
            in_shape=(1, input_size, input_size),
            mask_resize=input_size,
            dense_neurons=50,
            freeze_align_model=False,
        )
        if stn_checkpoint is not None:
            self.stn.load_state_dict(
                torch.load(stn_checkpoint, map_location='cpu')
            )

        # --- CNN backbone ---
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2]
            )

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()
        self.fc = nn.Linear(768, self.num_classes)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x):
        transformed_img = x
        transformed_mask = None
        grid = None

        if self.seg_model is not None:
            with torch.no_grad():
                x_seg = x
                if x.shape[-1] != 1024:
                    x_seg = F.interpolate(x, size=(1024, 1024),
                                          mode='bilinear', align_corners=False)
                if x_seg.shape[1] == 3:
                    x_seg = (0.299 * x_seg[:, 0:1] + 0.587 * x_seg[:, 1:2]
                             + 0.114 * x_seg[:, 2:3])

                landmarks = self.seg_model(x_seg)
                seg_mask = _landmarks_to_mask(landmarks, x.shape[2:], x.device)

            transformed_mask = self.stn(seg_mask)
            grid = self.stn.get_last_grid()
            transformed_img = F.grid_sample(x, grid, align_corners=False)

        feat = self.backbone(transformed_img)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)

        outputs = []
        for i in range(self.num_regions):
            pred = self.score_GAP(rois[:, i])
            pred = pred.view(-1, 768)
            pred = self.fc(pred)
            outputs.append(pred)

        return (torch.stack(outputs, dim=1),
                transformed_img,
                transformed_mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_roi_boxes(num_regions, device):
    """Return normalised bounding boxes for 6 or 4 lung sub-regions.

    6-region layout (matches original PAFE paper Fig.2, BrixIA convention):
        +-------+-------+
        |   A   |   D   |   upper band
        |-------+-------|
        |   B   |   E   |   middle band
        |-------+-------|
        |   C   |   F   |   lower band
        +-------+-------+
    A/B/C = right lung upper/middle/lower (image left, radiographic right)
    D/E/F = left lung upper/middle/lower (image right, radiographic left)
    """
    if num_regions == 6:
        # b = (y1, x1, y2, x2) as consumed by the crop statement
        return [
            torch.tensor([0.0, 0.0, 0.34, 0.5], device=device),   # A: right-upper
            torch.tensor([0.33, 0.0, 0.67, 0.5], device=device),  # B: right-middle
            torch.tensor([0.66, 0.0, 1.0, 0.5], device=device),   # C: right-lower
            torch.tensor([0.0, 0.5, 0.34, 1.0], device=device),   # D: left-upper
            torch.tensor([0.33, 0.5, 0.67, 1.0], device=device),  # E: left-middle
            torch.tensor([0.66, 0.5, 1.0, 1.0], device=device),   # F: left-lower
        ]
    elif num_regions == 4:
        return [
            torch.tensor([0.0, 0.0, 0.5, 0.5], device=device),
            torch.tensor([0.5, 0.0, 1.0, 0.5], device=device),
            torch.tensor([0.0, 0.5, 0.5, 1.0], device=device),
            torch.tensor([0.5, 0.5, 1.0, 1.0], device=device),
        ]
    else:
        raise ValueError(f"num_regions must be 4 or 6, got {num_regions}")


def _landmarks_to_mask(landmarks, out_size, device):
    """Convert landmark coordinates to a dense binary segmentation mask.

    landmarks: (B, N, 2) tensor of normalised [0, 1] coordinates
    """
    B = landmarks.shape[0]
    masks = []
    for i in range(B):
        mask = torch.zeros((1, out_size[0], out_size[1]), device=device)
        pts = (landmarks[i] * torch.tensor([out_size[1], out_size[0]],
                                           device=device)).long()
        pts = pts.clamp(0, max(out_size) - 1)

        for start, end in [(0, 44), (44, 94), (94, landmarks.shape[1])]:
            region_pts = pts[start:end]
            if region_pts.shape[0] < 3:
                continue
            for j in range(region_pts.shape[0] - 1):
                x0, y0 = region_pts[j]
                x1, y1 = region_pts[j + 1]
                mask[0, min(y0, y1):max(y0, y1) + 1,
                     min(x0, x1):max(x0, x1) + 1] = 1.0
        masks.append(mask)
    return torch.stack(masks)


# ---------------------------------------------------------------------------
# Innovation 1: Multi-Scale Feature Pyramid Fusion
# ---------------------------------------------------------------------------

class MultiScaleFusion(nn.Module):
    """Fuse feature maps from ResNet layers 2, 3, 4 into a single representation.

    Layer2 (64x64)  — fine details (nodules, interstitial patterns)
    Layer3 (32x32)  — mid-scale (segmental boundaries)
    Layer4 (16x16)  — coarse (lobar patterns, global appearance)

    All projected to a common dimension, upsampled, concatenated, and fused.
    """

    def __init__(self, in_channels=(512, 1024, 2048), proj_dim=256,
                 fused_dim=768):
        super().__init__()
        # 1x1 projections to unify channel count
        self.proj2 = nn.Conv2d(in_channels[0], proj_dim, 1)
        self.proj3 = nn.Conv2d(in_channels[1], proj_dim, 1)
        self.proj4 = nn.Conv2d(in_channels[2], proj_dim, 1)

        # Fusion: 3 * proj_dim → fused_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim * 3, fused_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat2, feat3, feat4):
        """feat2: (B, 512, 64, 64), feat3: (B, 1024, 32, 32), feat4: (B, 2048, 16, 16)"""
        p2 = self.proj2(feat2)                                    # (B, 256, 64, 64)
        p3 = F.interpolate(self.proj3(feat3), size=p2.shape[2:],
                           mode='bilinear', align_corners=False)  # (B, 256, 64, 64)
        p4 = F.interpolate(self.proj4(feat4), size=p2.shape[2:],
                           mode='bilinear', align_corners=False)  # (B, 256, 64, 64)

        fused = self.fuse(torch.cat([p2, p3, p4], dim=1))        # (B, 768, 64, 64)
        return fused


# ---------------------------------------------------------------------------
# Innovation 2: Cross-Region Graph Reasoning Network
# ---------------------------------------------------------------------------

class GraphRegionReasoning(nn.Module):
    """2-layer GCN over 6 lung regions with learned region importance weighting.

    Replaces naive mean pooling.  Adjacent regions exchange information
    through graph convolutions, then a learned attention mechanism decides
    how much each region contributes to the final prediction.
    """

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128,
                 num_regions=6, num_classes=2):
        super().__init__()
        self.num_regions = num_regions

        # Build normalised adjacency matrix from spatial layout
        A = self._build_adjacency(num_regions)
        self.register_buffer('A_hat', A)  # (6, 6), frozen

        self.gcn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gcn2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

        # Region importance: scalar weight per region
        self.importance = nn.Sequential(
            nn.Linear(out_dim, 1),
        )

        self.classifier = nn.Linear(out_dim, num_classes)

    @staticmethod
    def _build_adjacency(num_regions):
        """Build symmetric-normalised adjacency with self-loops.

        6-region spatial layout:
            A(0) -- B(1) -- C(2)
             |   /   |  /   |
            D(3) -- E(4) -- F(5)
        """
        edges = {
            0: [1, 3, 4],     # A → B, D, E
            1: [0, 2, 3, 4, 5],  # B → A, C, D, E, F
            2: [1, 4, 5],     # C → B, E, F
            3: [0, 1, 4],     # D → A, B, E
            4: [0, 1, 2, 3, 5],  # E → A, B, C, D, F
            5: [1, 2, 4],     # F → B, C, E
        }
        A = torch.zeros(num_regions, num_regions)
        for i, neighbours in edges.items():
            for j in neighbours:
                A[i, j] = 1.0

        # Symmetric normalisation: Â = D^(-1/2) * (A+I) * D^(-1/2)
        A_self = A + torch.eye(num_regions)
        D = A_self.sum(dim=1)
        D_inv_sqrt = torch.diag(D.pow(-0.5))
        A_hat = D_inv_sqrt @ A_self @ D_inv_sqrt
        return A_hat

    def _forward_gcn(self, region_features):
        """Graph convolution only — returns per-region features before pooling."""
        x = torch.einsum('ij,bjd->bid', self.A_hat, region_features)
        x = self.gcn1(x)
        x = torch.einsum('ij,bjd->bid', self.A_hat, x) + x
        x = self.gcn2(x)
        return x  # (B, num_regions, 128)

    def forward(self, region_features):
        """region_features: (B, num_regions, in_dim) → (B, num_classes)"""
        x = self._forward_gcn(region_features)                 # (B, 6, 128)

        # Learned region importance
        weights = self.importance(x)                           # (B, 6, 1)
        weights = torch.softmax(weights, dim=1)                # (B, 6, 1)

        # Weighted aggregation
        global_feat = (x * weights).sum(dim=1)                 # (B, 128)
        return self.classifier(global_feat)                    # (B, num_classes)


# ---------------------------------------------------------------------------
# Innovation 3: HybridMSGR — Multi-Scale + Graph Reasoning
# ---------------------------------------------------------------------------

class HybridMSGR(nn.Module):
    """Hybrid model with Multi-Scale feature fusion and Graph Region Reasoning.

    Key improvements over vanilla Hybrid:
    1. Multi-scale features from ResNet layers 2/3/4 via FPN-style fusion
    2. Cross-region graph reasoning replaces naive mean pooling
    3. Learned region importance weights

    Parameters match Hybrid for drop-in compatibility.
    """

    def __init__(self, backbone='resnet50', num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        # --- Split ResNet into named stages for multi-scale access ---
        full = self.backbone_fn()
        children = list(full.children())[:-2]  # strip avgpool + fc
        # children = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]

        if backbone == 'densenet121' or backbone == 'mobilenet_v3_small':
            # Non-ResNet backbones: keep as sequential trunk, use single-scale
            self._use_multiscale = False
            self.backbone = nn.Sequential(*children)
            # Project backbone channels to 768 for PAFE compatibility
            self._channel_proj = nn.Conv2d(num_features, 768, 1) \
                if num_features != 768 else nn.Identity()
        else:
            self._use_multiscale = True
            self.stem = nn.Sequential(*children[:4])    # conv1..maxpool
            self.layer1 = children[4]                    # out: 256
            self.layer2 = children[5]                    # out: 512
            self.layer3 = children[6]                    # out: 1024
            self.layer4 = children[7]                    # out: 2048
            ms_in = self._get_layer_channels(backbone)

        # --- Multi-scale fusion ---
        self.ms_fusion = MultiScaleFusion(
            in_channels=ms_in, proj_dim=256, fused_dim=768
        )

        # --- Pool fused features from 64x64 → 16x16 for PAFE ---
        self.fusion_pool = nn.AdaptiveMaxPool2d((feat_size, feat_size))

        # --- Transformer PAFE (reuse existing components) ---
        self.patch_emb = PatchEmbedding(
            in_channels=768, patch_size=1, emb_size=768,
            img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()

        # --- Shared components ---
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # --- Graph region reasoning replaces per-region FC + mean ---
        self.graph_reasoning = GraphRegionReasoning(
            in_dim=768, hidden_dim=256, out_dim=128,
            num_regions=num_regions, num_classes=num_classes,
        )

    @staticmethod
    def _get_layer_channels(backbone):
        """Return (layer2_ch, layer3_ch, layer4_ch) for the backbone."""
        ch_map = {
            'resnet18': (128, 256, 512),
            'resnet34': (128, 256, 512),
            'resnet50': (512, 1024, 2048),
            'mobilenet_v3_small': (24, 48, 576),
        }
        return ch_map.get(backbone, (512, 1024, 2048))

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False):
        # --- Feature extraction ---
        if self._use_multiscale:
            s = self.stem(x)                    # (B, 64, 128, 128)
            f1 = self.layer1(s)                 # (B, 256, 128, 128)
            f2 = self.layer2(f1)               # (B, 512, 64, 64)
            f3 = self.layer3(f2)               # (B, 1024, 32, 32)
            f4 = self.layer4(f3)               # (B, 2048, 16, 16)
            fused = self.ms_fusion(f2, f3, f4)  # (B, 768, 64, 64)
            fused = self.fusion_pool(fused)     # (B, 768, feat, feat)
        else:
            fused = self.backbone(x)            # fallback: single-scale
            # May need channel projection if backbone output != 768
            if fused.shape[1] != 768:
                fused = self._channel_proj(fused)

        # --- Transformer PAFE ---
        H, W = fused.shape[2], fused.shape[3]
        tok = self.patch_emb(fused)             # (B, N, 768)
        tok = self.transformer(tok)             # (B, N, 768)
        feat = tok.reshape(-1, H, W, 768).permute(0, 3, 1, 2)

        # --- ROI pooling ---
        rois = self.pool_rois(feat)             # (B, 6, 768, H, W)

        # --- Per-region GAP → graph reasoning ---
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])     # (B, 768, 1, 1)
            rf = rf.view(-1, 768)               # (B, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)  # (B, 6, 768)

        if embedding:
            return region_feats

        logits = self.graph_reasoning(region_feats)  # (B, num_classes)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)


# ---------------------------------------------------------------------------
# Innovation 4: Normal Anatomy Prototype Network
# ---------------------------------------------------------------------------

class NormalPrototypeModule(nn.Module):
    """Per-region feature prototypes representing "normal anatomy".

    During training: EMA-updated centroids of normal-image region features.
    During inference: per-region cosine distance from prototype = disease signal.

    Zero learnable parameters — prototypes are maintained as buffers.
    """

    def __init__(self, num_regions=6, feat_dim=768, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('prototypes', torch.zeros(num_regions, feat_dim))
        self.register_buffer('_initialized', torch.tensor(0))

    @torch.no_grad()
    def update(self, region_features):
        """EMA update prototypes with region features from normal images.

        Args:
            region_features: (B, 6, 768) from NORMAL images only
        """
        batch_mean = region_features.mean(dim=0)                  # (6, 768)
        if self._initialized.item() == 0:
            self.prototypes.copy_(batch_mean)
            self._initialized.fill_(1)
        else:
            self.prototypes.data = (self.momentum * self.prototypes +
                                    (1 - self.momentum) * batch_mean)

    def anomaly_scores(self, region_features):
        """Per-region cosine distance from prototype.

        Args:
            region_features: (B, 6, feat_dim)
        Returns:
            scores: (B, 6) in [0, 2], 0 = identical to normal, 2 = opposite
        """
        if self._initialized.item() == 0:
            return torch.zeros(region_features.shape[0], self.prototypes.shape[0],
                               device=region_features.device, dtype=region_features.dtype)
        r_norm = F.normalize(region_features, dim=2)              # (B, 6, D)
        p_norm = F.normalize(self.prototypes, dim=1)              # (6, D)
        cos_sim = (r_norm * p_norm.unsqueeze(0)).sum(dim=2)      # (B, 6)
        return 1.0 - cos_sim                                       # cosine distance


# ---------------------------------------------------------------------------
# HybridGRP: Hybrid + Graph Reasoning + Prototype
# ---------------------------------------------------------------------------

class HybridGRP(nn.Module):
    """Hybrid model with Graph Region Reasoning and Normal Anatomy Prototypes.

    Three innovations on top of the PAFE baseline:
    1. Multi-scale feature fusion (ResNet layers 2/3/4)
    2. Cross-region graph reasoning (GCN over 6 anatomical regions)
    3. Normal anatomy prototype network (cosine distance from normal centroids)

    Parameters match Hybrid for drop-in compatibility.
    """

    def __init__(self, backbone='resnet50', num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, proto_momentum=0.99):
        super().__init__()
        self.backbone_name = backbone
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        self._proto_enabled = True  # set False during warmup / eval-only

        # --- Split ResNet for multi-scale access ---
        full = self.backbone_fn()
        children = list(full.children())[:-2]

        if backbone in ('densenet121', 'mobilenet_v3_small'):
            self._use_multiscale = False
            self.backbone = nn.Sequential(*children)
            self._channel_proj = nn.Conv2d(num_features, 768, 1) \
                if num_features != 768 else nn.Identity()
        else:
            self._use_multiscale = True
            self.stem = nn.Sequential(*children[:4])
            self.layer1 = children[4]
            self.layer2 = children[5]
            self.layer3 = children[6]
            self.layer4 = children[7]
            ms_in = HybridMSGR._get_layer_channels(backbone)

        # --- Multi-scale fusion ---
        if self._use_multiscale:
            self.ms_fusion = MultiScaleFusion(
                in_channels=ms_in, proj_dim=256, fused_dim=768
            )
            feat_size = input_size // 32
            self.fusion_pool = nn.AdaptiveMaxPool2d((feat_size, feat_size))
        else:
            feat_size = input_size // 32

        # --- Transformer PAFE ---
        self.patch_emb = PatchEmbedding(
            in_channels=768, patch_size=1, emb_size=768, img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()

        # --- Shared ---
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # --- Graph region reasoning ---
        self.graph_reasoning = GraphRegionReasoning(
            in_dim=768, hidden_dim=256, out_dim=128,
            num_regions=num_regions, num_classes=num_classes,
        )

        # --- Normal prototype module ---
        self.proto = NormalPrototypeModule(
            num_regions=num_regions, feat_dim=768, momentum=proto_momentum,
        )

        # --- Prototype-augmented classifier ---
        # Combines graph-reasoned features (128) with anomaly scores (6)
        self.cls_fusion = nn.Sequential(
            nn.Linear(128 + num_regions, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, update_proto=False):
        # --- Feature extraction ---
        if self._use_multiscale:
            s = self.stem(x)
            f1 = self.layer1(s)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            fused = self.ms_fusion(f2, f3, f4)
            fused = self.fusion_pool(fused)
        else:
            fused = self.backbone(x)
            if fused.shape[1] != 768:
                fused = self._channel_proj(fused)

        # --- Transformer PAFE ---
        H, W = fused.shape[2], fused.shape[3]
        tok = self.patch_emb(fused)
        tok = self.transformer(tok)
        feat = tok.reshape(-1, H, W, 768).permute(0, 3, 1, 2)

        # --- ROI pooling ---
        rois = self.pool_rois(feat)

        # --- Per-region GAP ---
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)  # (B, 6, 768)

        if embedding:
            return region_feats

        # --- Graph reasoning ---
        gcn_feat = self.graph_reasoning._forward_gcn(region_feats)  # (B, 6, 128)

        # --- Prototype anomaly scores ---
        anomaly = self.proto.anomaly_scores(region_feats)           # (B, 6)

        # --- GAP over region dimension for the GCN features ---
        gcn_global = gcn_feat.mean(dim=1)                           # (B, 128)

        # --- Fuse graph features + anomaly scores ---
        combined = torch.cat([gcn_global, anomaly], dim=1)          # (B, 128+6)
        logits = self.cls_fusion(combined)                          # (B, num_classes)

        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        """Extract 6-region features without classification (for prototype update)."""
        return self.forward(x, embedding=True)

    def get_anomaly_scores(self, x):
        """Get per-region anomaly scores for interpretability."""
        region_feats = self.get_region_features(x)
        return self.proto.anomaly_scores(region_feats)  # (B, 6)


# ---------------------------------------------------------------------------
# Innovation 5: Region-Level MoCo Contrastive Learning
# ---------------------------------------------------------------------------

class RegionMoCoModule(nn.Module):
    """Momentum contrastive learning at the anatomical region level.

    Maintains 12 FIFO memory banks (6 regions x 2 classes) of L2-normalised
    feature vectors.  Uses a momentum encoder to produce stable keys.

    Key insight: same region + same label = positive; different label/different
    region = negative.  This converts 2,145 images into ~13K region-level
    training signals via contrastive self-supervision.
    """

    def __init__(self, num_regions=6, num_classes=2, feat_dim=768,
                 proj_dim=128, queue_size=4096, momentum=0.999, temperature=0.07):
        super().__init__()
        self.num_regions = num_regions
        self.num_classes = num_classes
        self.num_queues = num_regions * num_classes  # 12
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # --- Projection head: 768 -> 256 -> 128 ---
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

        # --- Memory banks (12 queues) ---
        self.register_buffer('queues',
            torch.zeros(self.num_queues, queue_size, proj_dim))
        self.register_buffer('queue_ptrs', torch.zeros(self.num_queues, dtype=torch.long))

        # --- Momentum projection head (for key encoder) ---
        self.projection_head_momentum = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        # Copy weights initially
        for p_m, p in zip(self.projection_head_momentum.parameters(),
                          self.projection_head.parameters()):
            p_m.data.copy_(p.data)
            p_m.requires_grad_(False)

    @torch.no_grad()
    def _momentum_update_projection(self):
        """EMA update momentum projection head."""
        for p_m, p in zip(self.projection_head_momentum.parameters(),
                          self.projection_head.parameters()):
            p_m.data = self.momentum * p_m.data + (1 - self.momentum) * p.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """FIFO enqueue keys into their respective memory banks.

        Args:
            keys: (B, 6, proj_dim) L2-normalised region features (momentum encoder)
            labels: (B,)  integer labels 0=normal, 1=pneumonia
        """
        B = keys.shape[0]
        for i in range(self.num_regions):
            for c in range(self.num_classes):
                # Select samples where label == c
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                batch_keys = keys[mask, i, :]  # (N_c, proj_dim)
                N = batch_keys.shape[0]
                q_idx = i * self.num_classes + c  # queue index

                ptr = self.queue_ptrs[q_idx].item()
                if ptr + N > self.queue_size:
                    # Wrap around
                    space = self.queue_size - ptr
                    self.queues[q_idx, ptr:] = batch_keys[:space]
                    self.queues[q_idx, :N - space] = batch_keys[space:]
                    self.queue_ptrs[q_idx] = (ptr + N) % self.queue_size
                else:
                    self.queues[q_idx, ptr:ptr + N] = batch_keys
                    self.queue_ptrs[q_idx] = ptr + N

    def forward(self, region_features, labels):
        """Compute region-level InfoNCE contrastive loss.

        Args:
            region_features: (B, 6, 768) — from main encoder
            labels: (B,) integer labels
        Returns:
            contrastive_loss: scalar
        """
        B = region_features.shape[0]
        device = region_features.device

        # Project and L2-normalise
        rf_flat = region_features.view(B * self.num_regions, -1)  # (B*6, 768)
        q = F.normalize(self.projection_head(rf_flat), dim=1)     # (B*6, 128)
        q = q.view(B, self.num_regions, -1)                       # (B, 6, 128)

        total_loss = 0.0
        count = 0

        for i in range(self.num_regions):
            for b in range(B):
                query = q[b, i, :]                        # (128,)
                label = labels[b].item()

                # Positive: from same region + same label queue
                pos_q = i * self.num_classes + label
                pos_ptr = self.queue_ptrs[pos_q].item()
                if pos_ptr == 0:
                    continue  # queue not yet filled
                # Sample a random positive from the queue
                pos_idx = torch.randint(0, min(pos_ptr, self.queue_size), (1,), device=device)
                positive = self.queues[pos_q, pos_idx[0], :]  # (128,)

                # Negatives: aggregate from other queues
                negatives = []
                for j in range(self.num_queues):
                    if j == pos_q:
                        continue
                    ptr_j = self.queue_ptrs[j].item()
                    if ptr_j == 0:
                        continue
                    # Sample negatives from each queue
                    n_sample = min(128, ptr_j)
                    neg_idx = torch.randint(0, ptr_j, (n_sample,), device=device)
                    negatives.append(self.queues[j, neg_idx, :])
                if not negatives:
                    continue
                negatives = torch.cat(negatives, dim=0)     # (N_neg, 128)

                # InfoNCE loss
                pos_sim = (query * positive).sum() / self.temperature
                neg_sim = (query.unsqueeze(0) @ negatives.T).squeeze(0) / self.temperature

                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0)
                target = torch.zeros(1, dtype=torch.long, device=device)
                total_loss += F.cross_entropy(logits.unsqueeze(0), target)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / count


# ---------------------------------------------------------------------------
# HybridCRC: Hybrid + Contrastive Region Classification
# ---------------------------------------------------------------------------

class HybridCRC(nn.Module):
    """Complete model: Multi-Scale + Graph Reasoning + Prototype + MoCo.

    Combines all innovations:
    1. Multi-scale feature fusion (ResNet layers 2/3/4)
    2. Cross-region graph reasoning (GCN over 6 regions)
    3. Normal anatomy prototypes (cosine distance from normal centroids)
    4. Region-level MoCo contrastive learning (InfoNCE over 12 memory banks)
    """

    def __init__(self, backbone='resnet50', num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, proto_momentum=0.99,
                 moco_queue_size=4096, moco_momentum=0.999, moco_temperature=0.07):
        super().__init__()
        self.backbone_name = backbone
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions

        # --- Split ResNet for multi-scale access ---
        full = self.backbone_fn()
        children = list(full.children())[:-2]

        if backbone in ('densenet121', 'mobilenet_v3_small'):
            self._use_multiscale = False
            self.backbone = nn.Sequential(*children)
            self._channel_proj = nn.Conv2d(num_features, 768, 1) \
                if num_features != 768 else nn.Identity()
        else:
            self._use_multiscale = True
            self.stem = nn.Sequential(*children[:4])
            self.layer1 = children[4]
            self.layer2 = children[5]
            self.layer3 = children[6]
            self.layer4 = children[7]
            ms_in = HybridMSGR._get_layer_channels(backbone)

        # --- Multi-scale fusion ---
        if self._use_multiscale:
            self.ms_fusion = MultiScaleFusion(
                in_channels=ms_in, proj_dim=256, fused_dim=768
            )
            feat_size = input_size // 32
            self.fusion_pool = nn.AdaptiveMaxPool2d((feat_size, feat_size))
        else:
            feat_size = input_size // 32

        # --- Transformer PAFE ---
        self.patch_emb = PatchEmbedding(
            in_channels=768, patch_size=1, emb_size=768, img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()

        # --- Shared ---
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # --- Graph region reasoning ---
        self.graph_reasoning = GraphRegionReasoning(
            in_dim=768, hidden_dim=256, out_dim=128,
            num_regions=num_regions, num_classes=num_classes,
        )

        # --- Normal prototype module ---
        self.proto = NormalPrototypeModule(
            num_regions=num_regions, feat_dim=768, momentum=proto_momentum,
        )

        # --- MoCo contrastive module ---
        self.moco = RegionMoCoModule(
            num_regions=num_regions, num_classes=num_classes,
            feat_dim=768, proj_dim=128, queue_size=moco_queue_size,
            momentum=moco_momentum, temperature=moco_temperature,
        )

        # --- Classifier fusion: graph features + anomaly scores ---
        self.cls_fusion = nn.Sequential(
            nn.Linear(128 + num_regions, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def _extract_features(self, x):
        """Shared feature extraction: image → region features."""
        if self._use_multiscale:
            s = self.stem(x)
            f1 = self.layer1(s)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            fused = self.ms_fusion(f2, f3, f4)
            fused = self.fusion_pool(fused)
        else:
            fused = self.backbone(x)
            if fused.shape[1] != 768:
                fused = self._channel_proj(fused)

        H, W = fused.shape[2], fused.shape[3]
        tok = self.patch_emb(fused)
        tok = self.transformer(tok)
        feat = tok.reshape(-1, H, W, 768).permute(0, 3, 1, 2)

        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            region_feats.append(rf)
        return torch.stack(region_feats, dim=1)  # (B, 6, 768)

    def forward(self, x, labels=None):
        """Full forward pass.

        Args:
            x: (B, 3, H, W)
            labels: (B,) optional, for contrastive loss computation
        Returns:
            logits: (B, 6, num_classes)
            If labels provided, also returns contrastive_loss
        """
        region_feats = self._extract_features(x)  # (B, 6, 768)

        # Dropout before GCN
        region_feats_drop = self.dropout(
            region_feats.view(-1, 768)
        ).view(-1, self.num_regions, 768)

        # Graph reasoning
        gcn_feat = self.graph_reasoning._forward_gcn(region_feats_drop)  # (B, 6, 128)

        # Prototype anomaly scores
        anomaly = self.proto.anomaly_scores(region_feats)  # (B, 6)

        # Fuse and classify
        gcn_global = gcn_feat.mean(dim=1)  # (B, 128)
        combined = torch.cat([gcn_global, anomaly], dim=1)
        logits = self.cls_fusion(combined)

        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        """Extract 6-region features (before dropout/FC) for prototype/MoCo."""
        return self._extract_features(x)

    def get_anomaly_scores(self, x):
        """Get per-region anomaly scores."""
        region_feats = self._extract_features(x)
        return self.proto.anomaly_scores(region_feats)

    def compute_contrastive_loss(self, region_feats, labels):
        """Compute MoCo contrastive loss and update memory banks."""
        return self.moco(region_feats, labels)

    @torch.no_grad()
    def update_momentum_encoder(self):
        """EMA-update the momentum projection head."""
        self.moco._momentum_update_projection()


# ---------------------------------------------------------------------------
# HybridGR — the simplest effective model: Hybrid + Graph Reasoning
# ---------------------------------------------------------------------------

class HybridGR(nn.Module):
    """Hybrid model with Graph Region Reasoning only.

    Minimalist design — keeps the proven Hybrid backbone and PAFE transformer,
    replaces the per-region FC + mean-pooling with learned graph reasoning
    over 6 anatomically-connected lung regions.

    No multi-scale FPN, no prototypes, no contrastive learning.
    ~0.1M more parameters than vanilla Hybrid (just the GCN weights).
    """

    def __init__(self, backbone='resnet50', num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        # Standard single-scale backbone
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2]
            )

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))

        # Handle backbone feature dim → PAFE 768-dim
        self._need_proj = (num_features != 768)
        patch_in = 768 if self._need_proj else num_features
        if self._need_proj:
            self._proj = nn.Conv2d(num_features, 768, 1)

        self.patch_emb = PatchEmbedding(
            in_channels=patch_in, patch_size=1, emb_size=768,
            img_size=feat_size,
        )
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        # Graph region reasoning replaces per-region FC + mean
        self.graph_reasoning = GraphRegionReasoning(
            in_dim=768, hidden_dim=256, out_dim=128,
            num_regions=num_regions, num_classes=num_classes,
        )

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode='bilinear', align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False):
        feat = self.backbone(x)
        if self._need_proj:
            feat = self._proj(feat)

        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)

        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)  # (B, 6, 768)

        if embedding:
            return region_feats

        logits = self.graph_reasoning(region_feats)  # (B, num_classes)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

# ---------------------------------------------------------------------------
# HybridACRA - Hybrid + Anatomical Cross-Region Attention (ACRA)
# ---------------------------------------------------------------------------

class HybridACRA(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, temperature=None, alpha_init=0.0,
                 no_gate=False, fixed_alpha=False):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        from .acra import CrossRegionAttention
        self.acra = CrossRegionAttention(
            feat_dim=768, num_regions=num_regions, num_classes=num_classes,
            temperature=temperature, alpha_init=alpha_init, no_gate=no_gate,
            fixed_alpha=fixed_alpha)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0]*x.shape[2]):int(b[2]*x.shape[2]),
                    int(b[1]*x.shape[3]):int(b[3]*x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        if return_attention:
            logits, attn = self.acra(region_feats, return_attention=True)
            return logits.unsqueeze(1).expand(-1, self.num_regions, -1), attn
        logits = self.acra(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Copy all compatible weights (backbone, patch_emb, transformer, etc.)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        # Map Hybrid fc -> ACRA classifier (key names differ)
        if "fc.weight" in ckpt:
            self.acra.classifier.weight.copy_(ckpt["fc.weight"])
            self.acra.classifier.bias.copy_(ckpt["fc.bias"])
            missing = [k for k in missing
                       if k not in ("acra.classifier.weight", "acra.classifier.bias")]
            unexpected = [k for k in unexpected
                          if k not in ("fc.weight", "fc.bias")]
        if missing:
            print(f"ACRA new layers (random init): {sorted(missing)}")
        if unexpected:
            print(f"Hybrid keys not loaded into ACRA: {sorted(unexpected)}")
        print(f"ACRA warm-started from {checkpoint_path}")


# ---------------------------------------------------------------------------
# HybridATACRA: Anatomy Topology-Aware variant
# ---------------------------------------------------------------------------

class HybridATACRA(nn.Module):
    """CNN backbone + Transformer PAFE + ATACRA (topology-aware attention).

    Same architecture as HybridACRA, but the cross-region attention injects
    a fixed anatomical topology prior (bilateral symmetry + ipsilateral adjacency)
    and learns a 6×6 topology deviation matrix Δ that reveals how pathology
    rewires the healthy attention graph.
    """

    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, temperature=None, alpha_init=0.0,
                 prior_ipsilateral=0.5, prior_contralateral=0.3,
                 freeze_topo=False, random_prior=False, shuffle_seed=1234,
                 no_prior=False):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        from .acra import TopologyAwareCrossRegionAttention
        self.acra = TopologyAwareCrossRegionAttention(
            feat_dim=768, num_regions=num_regions, num_classes=num_classes,
            temperature=temperature, alpha_init=alpha_init,
            prior_ipsilateral=prior_ipsilateral,
            prior_contralateral=prior_contralateral,
            freeze_topo=freeze_topo, random_prior=random_prior,
            shuffle_seed=shuffle_seed, no_prior=no_prior)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        if return_attention:
            logits, attn, topo = self.acra(region_feats, return_attention=True)
            return logits.unsqueeze(1).expand(-1, self.num_regions, -1), attn, topo
        logits = self.acra(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Copy all compatible weights (backbone, patch_emb, transformer, etc.)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        # Map Hybrid fc -> ATACRA classifier (key names differ)
        if "fc.weight" in ckpt:
            if ckpt["fc.weight"].shape == self.acra.classifier.weight.shape:
                self.acra.classifier.weight.copy_(ckpt["fc.weight"])
                self.acra.classifier.bias.copy_(ckpt["fc.bias"])
                missing = [k for k in missing
                           if k not in ("acra.classifier.weight", "acra.classifier.bias")]
                unexpected = [k for k in unexpected
                              if k not in ("fc.weight", "fc.bias")]
            else:
                print(f"Classifier shape mismatch "
                      f"{tuple(ckpt['fc.weight'].shape)} -> "
                      f"{tuple(self.acra.classifier.weight.shape)}: random init")
        if missing:
            print(f"ATACRA new layers (random init): {sorted(missing)}")
        if unexpected:
            print(f"Hybrid keys not loaded into ATACRA: {sorted(unexpected)}")
        print(f"ATACRA warm-started from {checkpoint_path}")

    @torch.no_grad()
    def get_topology_report(self):
        """Return effective topology A + Δ for interpretation."""
        return self.acra.get_topology_report()


# ---------------------------------------------------------------------------
# HybridCDA: Contralateral Differential Attention variant
# ---------------------------------------------------------------------------

class HybridCDA(nn.Module):
    """CNN backbone + Transformer PAFE + CDA (contralateral differential attention).

    PAFE-specific innovation: exploits the 3 bilateral mirror pairs unique to
    the 6-region lung architecture. Computes per-pair asymmetry gates from
    feature distances, then uses them to boost cross-contralateral attention
    when a pair is asymmetric — directly encoding the medical logic that
    unilateral pathology warrants contralateral comparison.
    """

    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, temperature=None, alpha_init=0.0):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        from .acra import ContralateralDifferentialAttention
        self.acra = ContralateralDifferentialAttention(
            feat_dim=768, num_regions=num_regions, num_classes=num_classes,
            temperature=temperature, alpha_init=alpha_init)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        if return_attention:
            logits, attn, gates = self.acra(region_feats, return_attention=True)
            return logits.unsqueeze(1).expand(-1, self.num_regions, -1), attn, gates
        logits = self.acra(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        if "fc.weight" in ckpt:
            self.acra.classifier.weight.copy_(ckpt["fc.weight"])
            self.acra.classifier.bias.copy_(ckpt["fc.bias"])
            missing = [k for k in missing
                       if k not in ("acra.classifier.weight", "acra.classifier.bias")]
            unexpected = [k for k in unexpected
                          if k not in ("fc.weight", "fc.bias")]
        if missing:
            print(f"CDA new layers (random init): {sorted(missing)}")
        if unexpected:
            print(f"Hybrid keys not loaded into CDA: {sorted(unexpected)}")
        print(f"CDA warm-started from {checkpoint_path}")


# ---------------------------------------------------------------------------
# HybridStandardQKV: control model with learned QKV projections
# ---------------------------------------------------------------------------

class HybridStandardQKV(nn.Module):
    """Same architecture as HybridACRA but with learned QKV projections
    (standard transformer attention, ~1.77M extra params). Control experiment."""

    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, temperature=None, alpha_init=0.1,
                 prior_ipsilateral=None, prior_contralateral=None,
                 random_prior=False, shuffle_seed=1234, freeze_topo=False):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        if prior_ipsilateral is not None:
            from .acra import StandardQKVTopologyAttention
            self.acra = StandardQKVTopologyAttention(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init,
                prior_ipsilateral=prior_ipsilateral,
                prior_contralateral=prior_contralateral,
                random_prior=random_prior, shuffle_seed=shuffle_seed,
                freeze_topo=freeze_topo)
        else:
            from .acra import StandardQKVAttention
            self.acra = StandardQKVAttention(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        logits = self.acra(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        if "fc.weight" in ckpt:
            self.acra.classifier.weight.copy_(ckpt["fc.weight"])
            self.acra.classifier.bias.copy_(ckpt["fc.bias"])
        print(f"StandardQKV warm-started from {checkpoint_path}")


# ---------------------------------------------------------------------------
# HybridATACRA_Dual: dual-branch (baseline + attention) variant
# ---------------------------------------------------------------------------

class HybridATACRADual(nn.Module):
    """Dual-branch ATACRA: baseline mean-pool path + attention context path.

    Path 1 is warmstarted from the Hybrid checkpoint (preserves baseline).
    Path 2 provides a complementary cross-region aggregated signal.
    """

    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1,
                 freeze_topo=False, random_prior=False, shuffle_seed=1234):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        from .acra import DualBranchTopologyAttention
        self.acra = DualBranchTopologyAttention(
            feat_dim=768, num_regions=num_regions, num_classes=num_classes,
            temperature=temperature, alpha_init=alpha_init,
            prior_ipsilateral=prior_ipsilateral,
            prior_contralateral=prior_contralateral,
            freeze_topo=freeze_topo, random_prior=random_prior,
            shuffle_seed=shuffle_seed)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        if return_attention:
            logits, attn = self.acra(region_feats, return_attention=True)
            return logits.unsqueeze(1).expand(-1, self.num_regions, -1), attn
        logits = self.acra(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        if "fc.weight" in ckpt:
            self.acra.fc1.weight.copy_(ckpt["fc.weight"])
            self.acra.fc1.bias.copy_(ckpt["fc.bias"])
            missing = [k for k in missing
                       if k not in ("acra.fc1.weight", "acra.fc1.bias")]
            unexpected = [k for k in unexpected
                          if k not in ("fc.weight", "fc.bias")]
        print(f"Dual-branch warm-started from {checkpoint_path}")
        if missing:
            print(f"  new layers (fresh init): {sorted(missing)[:6]}")


# ---------------------------------------------------------------------------
# Final variants wrappers (multi-head / trainable-A / attn-weighted logits)
# ---------------------------------------------------------------------------

class HybridVariant(nn.Module):
    """Generic wrapper: Hybrid backbone + configurable attention module."""

    def __init__(self, backbone="resnet50", num_classes=2, num_regions=6,
                 input_size=512, dropout=0.3, variant='multihead',
                 temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1, n_heads=4):
        super().__init__()
        self.backbone_fn, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        feat_size = input_size // 32

        if backbone == "densenet121":
            self.backbone = nn.Sequential(self.backbone_fn())
        else:
            self.backbone = nn.Sequential(
                *list(self.backbone_fn().children())[:-2])

        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(
            in_channels=num_features, patch_size=1, emb_size=768,
            img_size=feat_size)
        self.transformer = TransformerEncoderBlock()
        self.dropout = nn.Dropout(dropout)

        from .acra import (MultiHeadZeroAttention, TrainablePriorAttention,
                           AttnWeightedLogits, EvidenceAwarePoolingAttention)
        if variant == 'multihead':
            self.attn = MultiHeadZeroAttention(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init, n_heads=n_heads)
        elif variant == 'trainableA':
            self.attn = TrainablePriorAttention(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init,
                prior_ipsilateral=prior_ipsilateral,
                prior_contralateral=prior_contralateral)
        elif variant == 'attnlogits':
            self.attn = AttnWeightedLogits(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init)
        elif variant == 'evidence':
            self.attn = EvidenceAwarePoolingAttention(
                feat_dim=768, num_regions=num_regions, num_classes=num_classes,
                temperature=temperature, alpha_init=alpha_init,
                prior_ipsilateral=prior_ipsilateral,
                prior_contralateral=prior_contralateral)

    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]
        boxes = _get_roi_boxes(self.num_regions, x.device)
        out = []
        for b in boxes:
            roi = x[:, :,
                    int(b[0] * x.shape[2]):int(b[2] * x.shape[2]),
                    int(b[1] * x.shape[3]):int(b[3] * x.shape[3])]
            out.append(F.interpolate(roi, size=crop_size,
                                     mode="bilinear", align_corners=False))
        return torch.stack(out, dim=1)

    def forward(self, x, embedding=False, return_attention=False):
        feat = self.backbone(x)
        H, W = feat.shape[2], feat.shape[3]
        feat = self.patch_emb(feat)
        feat = self.transformer(feat)
        feat = feat.reshape(-1, H, W, 768).permute(0, 3, 1, 2)
        rois = self.pool_rois(feat)
        region_feats = []
        for i in range(self.num_regions):
            rf = self.score_GAP(rois[:, i])
            rf = rf.view(-1, 768)
            if not embedding:
                rf = self.dropout(rf)
            region_feats.append(rf)
        region_feats = torch.stack(region_feats, dim=1)
        if embedding:
            return region_feats
        if return_attention:
            logits, attn = self.attn(region_feats, return_attention=True)
            return logits.unsqueeze(1).expand(-1, self.num_regions, -1), attn
        logits = self.attn(region_feats)
        return logits.unsqueeze(1).expand(-1, self.num_regions, -1)

    def get_region_features(self, x):
        return self.forward(x, embedding=True)

    @torch.no_grad()
    def warmstart_from_hybrid(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        if "fc.weight" in ckpt:
            self.attn.classifier.weight.copy_(ckpt["fc.weight"])
            self.attn.classifier.bias.copy_(ckpt["fc.bias"])
        print(f"Variant warm-started from {checkpoint_path}")
