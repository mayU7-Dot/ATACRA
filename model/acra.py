"""Anatomical Cross-Region Attention (ACRA) — zero-parameter region interaction.

Key insight: the 6 region features R ∈ R^(6, 768) are already computed by the
pretrained PAFE model.  We can apply self-attention over these 6 vectors using
the raw features as Q, K, V — no learned projection matrices, no extra parameters.

Residual connection ensures that at initialization (uniform attention), the
behaviour is approximately equivalent to the working Hybrid baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRegionAttention(nn.Module):
    """Zero-parameter self-attention over anatomical lung regions.

    Given R ∈ R^(B, 6, 768): 6 region feature vectors from the PAFE model.
    Computes Q=K=V=R (identity projection), scaled dot-product attention
    with residual connection, then GAP-pools to a global classification vector.

    At initialisation: attention is roughly uniform (1/6 each), so
    R' ≈ R + mean(R).  The residual preserves each region's identity while
    the attention term injects global context.  After training, the attention
    matrix learns which regions should attend to which others for each sample.
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.0, no_gate=False,
                 fixed_alpha=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions
        self.no_gate = no_gate
        # Learnable temperature for softmax sharpness
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5)
        )
        # Residual gate α: controls how much attention contributes.
        # α=0 → pure mean-pooling (Hybrid baseline behaviour).
        # α>0 → attention-enhanced features.
        if fixed_alpha:
            self.alpha = float(alpha_init)  # fixed coefficient, not learned
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # Final classifier from region-aggregated features
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        """Cross-region self-attention with no learned QKV projections.

        Args:
            region_features: (B, 6, feat_dim)
            return_attention: if True, also return attention weights

        Returns:
            logits: (B, num_classes)
            attn_weights: (B, 6, 6) [only if return_attention=True]
        """
        R = region_features  # (B, 6, D)

        # Scaled dot-product self-attention over regions (Q=K=V=R)
        # attn_logits: (B, 6, 6) — how much region j attends to region i
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature

        # Causal-free softmax: each region attends to all 6 regions
        attn_weights = F.softmax(attn_logits, dim=2)  # (B, 6, 6)

        # Residual attention with learnable gate α
        # α=0 (init): R' = R → equivalent to Hybrid baseline (pure mean-pool)
        # α>0 (learned): R' = R + α * attention(R) → cross-region context
        context = torch.einsum('bij,bjd->bid', attn_weights, R)  # (B, 6, D)
        if self.no_gate:
            # Ablation: attention without the α residual gate
            R_enhanced = context
        else:
            R_enhanced = R + self.alpha * context  # α slowly unlocks attention

        # Mean-pool over regions → global feature → classify
        global_feat = R_enhanced.mean(dim=1)  # (B, D)
        logits = self.classifier(global_feat)

        if return_attention:
            return logits, attn_weights
        return logits

    @torch.no_grad()
    def init_from_pretrained_fc(self, fc_weight, fc_bias):
        """Initialise classifier from a pre-trained Hybrid model's FC layer.

        This warm-starts the classifier so the model begins from the working
        Hybrid baseline rather than random weights.

        Args:
            fc_weight: (num_classes, 768) from Hybrid model
            fc_bias: (num_classes,) from Hybrid model
        """
        self.classifier.weight.copy_(fc_weight)
        self.classifier.bias.copy_(fc_bias)
        print("ACRA classifier initialised from pretrained Hybrid FC.")


# ---------------------------------------------------------------------------
# ATACRA: Anatomical Topology-Aware Cross-Region Attention
# ---------------------------------------------------------------------------

class TopologyAwareCrossRegionAttention(nn.Module):
    """Cross-region attention with anatomical topology prior and learnable bias.

    Given R ∈ R^(B, 6, 768): 6 region feature vectors (A-F lung sub-regions).

    Core innovation: injects the known bilateral anatomy of the chest into
    the attention mechanism, then learns how pathology *deviates* from this
    healthy baseline topology.

    Anatomy prior A (fixed, zero-parameter):
      A[i,j] = 1.0   if i == j              (self-attention)
               0.5   if i,j ipsilateral-adjacent (A-B, B-C, D-E, E-F)
               0.3   if i,j contralateral-mirror (A-D, B-E, C-F)
               0.0   otherwise

    Topology bias Δ (learnable, 6×6=36 params):
      Initialised to zero → model starts from pure anatomy prior.
      After training, Δ[i,j] reveals how a specific disease *rewires*
      the anatomical attention graph.

    Final attention:
      attn = softmax(R @ R^T / τ  +  A  +  Δ)

    Parameters
    ----------
    feat_dim : int
        Dimension of each region feature (768).
    num_regions : int
        Number of lung sub-regions (6).
    num_classes : int
        Number of output classes.
    temperature : float, optional
        Initial temperature for softmax scaling.
    alpha_init : float
        Initial value for the residual gate α.
    prior_ipsilateral : float
        Prior weight for adjacent regions within the same lung.
    prior_contralateral : float
        Prior weight for mirrored regions across lungs.
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.0,
                 prior_ipsilateral=0.5, prior_contralateral=0.3,
                 freeze_topo=False, random_prior=False, shuffle_seed=1234,
                 no_prior=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions

        # Learnable temperature
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5)
        )
        # Residual gate
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # --- Anatomy topology prior A (fixed) ---
        # Actual ROI layout from _get_roi_boxes (2 cols x 3 rows, matches
        # original PAFE paper Fig.2 BrixIA convention):
        #     +-------+-------+
        #     |   A   |   D   |   upper band
        #     +-------+-------+
        #     |   B   |   E   |   middle band
        #     +-------+-------+
        #     |   C   |   F   |   lower band
        #     +-------+-------+
        # A/B/C = right lung (image-left), D/E/F = left lung (image-right).
        anatomy = torch.zeros(num_regions, num_regions)

        if not no_prior:
            # Self-attention: 1.0 on diagonal
            anatomy.fill_diagonal_(1.0)

            # Ipsilateral vertical adjacency (same lung, upper-middle-lower)
            ipsi_pairs = [(0, 1), (1, 0),   # A-B: right lung upper-middle
                          (1, 2), (2, 1),   # B-C: right lung middle-lower
                          (3, 4), (4, 3),   # D-E: left lung upper-middle
                          (4, 5), (5, 4)]    # E-F: left lung middle-lower
            for i, j in ipsi_pairs:
                anatomy[i, j] = prior_ipsilateral

            # Contralateral mirror (opposite lung, same vertical level)
            contra_pairs = [(0, 3), (3, 0),   # A-D: upper band mirror
                            (1, 4), (4, 1),   # B-E: middle band mirror
                            (2, 5), (5, 2)]    # C-F: lower band mirror
            for i, j in contra_pairs:
                anatomy[i, j] = prior_contralateral

            if random_prior:
                # Control: shuffle non-diagonal entries, keeping count/weight
                # distribution. Reproducible via fixed torch generator.
                gen = torch.Generator().manual_seed(shuffle_seed)
                nz_idx = torch.nonzero(anatomy - torch.eye(num_regions)).cpu()
                vals = anatomy[nz_idx[:, 0], nz_idx[:, 1]].clone()
                perm = torch.randperm(vals.numel(), generator=gen)
                vals = vals[perm]
                anatomy_rand = torch.eye(num_regions).clone()
                anatomy_rand[nz_idx[:, 0], nz_idx[:, 1]] = vals
                anatomy = anatomy_rand

        self.register_buffer('anatomy_prior', anatomy)

        # --- Learnable topology deviation Δ ---
        # Initialised to zero: at start, model trusts the anatomy prior.
        self.topo_bias = nn.Parameter(torch.zeros(num_regions, num_regions))
        if freeze_topo:
            self.topo_bias.requires_grad_(False)

        # Final classifier
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        """Cross-region attention with anatomical topology awareness.

        Args:
            region_features: (B, 6, feat_dim)
            return_attention: if True, return (logits, attn_weights, topo_matrix)

        Returns:
            logits: (B, num_classes)
            or (logits, attn_weights, topo_matrix) if return_attention=True
        """
        B = region_features.shape[0]
        R = region_features  # (B, 6, D)

        # Data-driven similarity (zero-parameter QKV)
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature  # (B, 6, 6)

        # Anatomical topology: prior + learned deviation
        # Broadcast across batch dimension
        topo = self.anatomy_prior + self.topo_bias  # (6, 6)
        topo = topo.unsqueeze(0).expand(B, -1, -1)   # (B, 6, 6)

        # Combined attention
        attn_weights = F.softmax(attn_logits + topo, dim=2)  # (B, 6, 6)

        # Context aggregation with residual gate
        context = torch.einsum('bij,bjd->bid', attn_weights, R)  # (B, 6, D)
        R_enhanced = R + self.alpha * context

        # Mean-pool → classify
        global_feat = R_enhanced.mean(dim=1)  # (B, D)
        logits = self.classifier(global_feat)

        if return_attention:
            return logits, attn_weights, topo
        return logits

    @torch.no_grad()
    def get_topology_report(self):
        """Return the learned effective topology A + Δ for analysis."""
        return (self.anatomy_prior + self.topo_bias).cpu().clone()


# ---------------------------------------------------------------------------
# CDA: Contralateral Differential Attention
# ---------------------------------------------------------------------------

class ContralateralDifferentialAttention(nn.Module):
    """Asymmetry-gated contralateral attention — PAFE-specific innovation.

    Core insight unique to the PAFE 6-region bilateral architecture:
    pneumonia is inherently asymmetric. A healthy lung has symmetric left-right
    pairs; pneumonia breaks this symmetry locally. The *degree* of broken
    symmetry between contralateral pairs is itself a diagnostic signal.

    Three mirror pairs (zero-parameter computation):
      upper:  A(0) ↔ D(3)
      middle: B(1) ↔ E(4)
      lower:  C(2) ↔ F(5)

    For each pair, we compute an asymmetry gate:
      g_k = σ(||R_left - R_right|| / τ_g  -  θ)
      g_k ≈ 0 → pair is symmetric → likely normal
      g_k ≈ 1 → pair is asymmetric → possible pathology

    When a pair is asymmetric, we boost mutual attention between them:
      attn[A,D] += γ × g_upper    (A looks harder at its mirror D)
      attn[D,A] += γ × g_upper    (D looks harder at its mirror A)

    This directly encodes the radiologist's workflow: "I see something in the
    right lower lobe — let me check the left lower lobe more carefully."

    Parameters (only 3 new scalars vs ACRA):
      τ_g : temperature for asymmetry distance scaling
      θ   : threshold offset (how much asymmetry to ignore as noise)
      γ   : boost multiplier (how strongly asymmetry amplifies contra-attention)
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions

        # Standard ACRA params (2 scalars)
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Contra-pair indices (fixed, zero-parameter)
        # A↔D(0,3), B↔E(1,4), C↔F(2,5)
        self.register_buffer('contra_pairs',
                             torch.tensor([[0, 3], [1, 4], [2, 5]]))

        # CDA-specific params (3 scalars)
        self.tau_g = nn.Parameter(torch.tensor(1.0))       # asymmetry temperature
        self.theta = nn.Parameter(torch.tensor(0.0))       # asymmetry threshold
        self.gamma = nn.Parameter(torch.tensor(0.0))       # boost multiplier

        # Final classifier
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        """Contralateral differential attention.

        Args:
            region_features: (B, 6, feat_dim)
            return_attention: if True, return (logits, attn_weights, asymmetry_gates)

        Returns:
            logits: (B, num_classes)
            or (logits, attn_weights, gates) if return_attention=True
        """
        B = region_features.shape[0]
        R = region_features  # (B, 6, D)

        # --- 1. Base attention (zero-parameter, same as ACRA) ---
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature  # (B, 6, 6)

        # --- 2. Compute 3 contra-pair asymmetry gates (zero-parameter) ---
        # For each pair (i,j), compute L2 distance between paired region features
        # g_k: how asymmetric is this pair? (B, 3)
        i_idx = self.contra_pairs[:, 0]  # [0, 1, 2]
        j_idx = self.contra_pairs[:, 1]  # [3, 4, 5]
        diff = R[:, i_idx] - R[:, j_idx]           # (B, 3, D)
        dist = torch.norm(diff, p=2, dim=2)         # (B, 3)  L2 distance
        gates = torch.sigmoid(dist / self.tau_g - self.theta)  # (B, 3)

        # --- 3. Boost contralateral attention based on asymmetry ---
        # When g_k is high (asymmetric pair), the two regions should look harder
        # at each other in both directions.
        for k, (i, j) in enumerate([(0, 3), (1, 4), (2, 5)]):
            boost = self.gamma * gates[:, k]           # (B,)
            attn_logits[:, i, j] += boost
            attn_logits[:, j, i] += boost

        # --- 4. Standard attention + residual + classify ---
        attn_weights = F.softmax(attn_logits, dim=2)  # (B, 6, 6)
        context = torch.einsum('bij,bjd->bid', attn_weights, R)  # (B, 6, D)
        R_enhanced = R + self.alpha * context

        global_feat = R_enhanced.mean(dim=1)  # (B, D)
        logits = self.classifier(global_feat)

        if return_attention:
            return logits, attn_weights, gates
        return logits


# ---------------------------------------------------------------------------
# StandardQKVAttention: learned QKV projection baseline (control experiment)
# ---------------------------------------------------------------------------

class StandardQKVAttention(nn.Module):
    """Standard transformer attention with learned Q, K, V projections.

    Control experiment for the zero-parameter QKV claim:
      Q = R @ W_Q, K = R @ W_K, V = R @ W_V   (3 x 768x768 = 1.77M params)
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # Learned projections
        self.W_Q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_K = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_V = nn.Linear(feat_dim, feat_dim, bias=False)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features  # (B, 6, D)
        Q = self.W_Q(R)
        K = self.W_K(R)
        V = self.W_V(R)
        attn_logits = torch.einsum('bid,bjd->bij', Q, K) / self.temperature
        attn_weights = F.softmax(attn_logits, dim=2)
        context = torch.einsum('bij,bjd->bid', attn_weights, V)
        R_enhanced = R + self.alpha * context
        global_feat = R_enhanced.mean(dim=1)
        logits = self.classifier(global_feat)
        if return_attention:
            return logits, attn_weights
        return logits


class StandardQKVTopologyAttention(nn.Module):
    """Standard transformer attention (learned QKV) + anatomical prior A + Δ.

    E6 control: identical to TopologyAwareCrossRegionAttention except that
    Q = R·W_Q, K = R·W_K, V = R·W_V (3 × 768×768 = 1.77M extra params).
    No output projection, FFN or layernorm — only the QKV generation differs.
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1,
                 random_prior=False, shuffle_seed=1234,
                 freeze_topo=False):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.W_Q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_K = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_V = nn.Linear(feat_dim, feat_dim, bias=False)

        # Anatomy prior A (same construction as TopologyAwareCrossRegionAttention)
        anatomy = torch.zeros(num_regions, num_regions)
        anatomy.fill_diagonal_(1.0)
        ipsi_pairs = [(0, 1), (1, 0), (1, 2), (2, 1),
                      (3, 4), (4, 3), (4, 5), (5, 4)]
        for i, j in ipsi_pairs:
            anatomy[i, j] = prior_ipsilateral
        contra_pairs = [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]
        for i, j in contra_pairs:
            anatomy[i, j] = prior_contralateral
        if random_prior:
            gen = torch.Generator().manual_seed(shuffle_seed)
            nz_idx = torch.nonzero(anatomy - torch.eye(num_regions)).cpu()
            vals = anatomy[nz_idx[:, 0], nz_idx[:, 1]].clone()
            perm = torch.randperm(vals.numel(), generator=gen)
            vals = vals[perm]
            anatomy_rand = torch.eye(num_regions).clone()
            anatomy_rand[nz_idx[:, 0], nz_idx[:, 1]] = vals
            anatomy = anatomy_rand
        self.register_buffer('anatomy_prior', anatomy)

        self.topo_bias = nn.Parameter(torch.zeros(num_regions, num_regions))
        if freeze_topo:
            self.topo_bias.requires_grad_(False)

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features
        B = R.shape[0]
        Q = self.W_Q(R)
        K = self.W_K(R)
        V = self.W_V(R)
        attn_logits = torch.einsum('bid,bjd->bij', Q, K) / self.temperature
        topo = (self.anatomy_prior + self.topo_bias).unsqueeze(0).expand(B, -1, -1)
        attn_weights = F.softmax(attn_logits + topo, dim=2)
        context = torch.einsum('bij,bjd->bid', attn_weights, V)
        R_enhanced = R + self.alpha * context
        global_feat = R_enhanced.mean(dim=1)
        logits = self.classifier(global_feat)
        if return_attention:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# DualBranchTopologyAttention: Hybrid path + attention path ensemble
# ---------------------------------------------------------------------------

class DualBranchTopologyAttention(nn.Module):
    """Dual-branch design: baseline mean-pool path + attention context path.

    Path 1 (baseline):  g1 = mean(R)                -> FC1  (warmstarted from Hybrid)
    Path 2 (attention): attn = softmax(RR^T/tau + A + Delta)
                        g2 = mean(attn @ R)         -> FC2  (fresh, complementary)
    final = logits1 + beta * logits2   (beta learnable)

    The attention path is an independent signal source (not a perturbation of
    the baseline), so it can only add complementary information.
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1,
                 freeze_topo=False, random_prior=False, shuffle_seed=1234):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions

        # Path 1: baseline classifier (warmstarted)
        self.fc1 = nn.Linear(feat_dim, num_classes)

        # Attention params
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Anatomy prior A (same as TopologyAwareCrossRegionAttention)
        anatomy = torch.zeros(num_regions, num_regions)
        anatomy.fill_diagonal_(1.0)
        ipsi_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4)]
        for i, j in ipsi_pairs:
            anatomy[i, j] = prior_ipsilateral
        contra_pairs = [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]
        for i, j in contra_pairs:
            anatomy[i, j] = prior_contralateral
        if random_prior:
            gen = torch.Generator().manual_seed(shuffle_seed)
            nz_idx = torch.nonzero(anatomy - torch.eye(num_regions)).cpu()
            vals = anatomy[nz_idx[:, 0], nz_idx[:, 1]].clone()
            perm = torch.randperm(vals.numel(), generator=gen)
            vals = vals[perm]
            anatomy_rand = torch.eye(num_regions).clone()
            anatomy_rand[nz_idx[:, 0], nz_idx[:, 1]] = vals
            anatomy = anatomy_rand
        self.register_buffer('anatomy_prior', anatomy)

        # Topology bias
        self.topo_bias = nn.Parameter(torch.zeros(num_regions, num_regions))
        if freeze_topo:
            self.topo_bias.requires_grad_(False)

        # Path 2: attention-context classifier (fresh)
        self.fc2 = nn.Linear(feat_dim, num_classes)

        # Learnable branch balance
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, region_features, return_attention=False):
        R = region_features  # (B, 6, D)

        # Attention
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature
        topo = self.anatomy_prior + self.topo_bias
        attn_weights = F.softmax(attn_logits + topo.unsqueeze(0), dim=2)

        # Path 1: baseline mean-pool
        g1 = R.mean(dim=1)
        logits1 = self.fc1(g1)

        # Path 2: attention-aggregated context
        context = torch.einsum('bij,bjd->bid', attn_weights, R)
        g2 = context.mean(dim=1)
        logits2 = self.fc2(g2)

        logits = logits1 + self.beta * logits2

        if return_attention:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# Final experiment variants (prescreen batch)
# ---------------------------------------------------------------------------

class MultiHeadZeroAttention(nn.Module):
    """Multi-head zero-param attention: 768 split into H heads, each with own tau."""

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1, n_heads=4):
        super().__init__()
        assert feat_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads
        base_t = temperature if temperature is not None else feat_dim ** 0.5
        self.temperatures = nn.Parameter(
            torch.full((n_heads,), base_t / (n_heads ** 0.5)))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features  # (B, 6, D)
        B, N, D = R.shape
        contexts = []
        attns = []
        for h in range(self.n_heads):
            Rh = R[:, :, h*self.head_dim:(h+1)*self.head_dim]  # (B, 6, d)
            logits = torch.einsum('bid,bjd->bij', Rh, Rh) / self.temperatures[h]
            attn = F.softmax(logits, dim=2)
            ctx = torch.einsum('bij,bjd->bid', attn, Rh)
            contexts.append(ctx)
            attns.append(attn)
        context = torch.cat(contexts, dim=2)  # (B, 6, D)
        R_enh = R + self.alpha * context
        global_feat = R_enh.mean(dim=1)
        logits = self.classifier(global_feat)
        if return_attention:
            return logits, attns[0]
        return logits


class TrainablePriorAttention(nn.Module):
    """Anatomy prior A as a trainable parameter (init from anatomy values)."""

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        anatomy = torch.zeros(num_regions, num_regions)
        anatomy.fill_diagonal_(1.0)
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4)]:
            anatomy[i, j] = prior_ipsilateral
        for i, j in [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]:
            anatomy[i, j] = prior_contralateral
        self.prior = nn.Parameter(anatomy)  # trainable A
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature
        attn = F.softmax(attn_logits + self.prior.unsqueeze(0), dim=2)
        context = torch.einsum('bij,bjd->bid', attn, R)
        R_enh = R + self.alpha * context
        global_feat = R_enh.mean(dim=1)
        logits = self.classifier(global_feat)
        if return_attention:
            return logits, attn
        return logits


class AttnWeightedLogits(nn.Module):
    """Attention column-sums weight the per-region logits before final combine."""

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features  # (B, 6, D)
        attn_logits = torch.einsum('bid,bjd->bij', R, R) / self.temperature
        attn = F.softmax(attn_logits, dim=2)
        # Per-region logits
        region_logits = self.classifier(R)  # (B, 6, C)
        # Attention column-sum as region importance
        importance = attn.sum(dim=1)  # (B, 6) how much each region is attended to
        weights = F.softmax(importance, dim=1)  # (B, 6)
        logits = (weights.unsqueeze(2) * region_logits).sum(dim=1)  # (B, C)
        if return_attention:
            return logits, attn
        return logits


# ---------------------------------------------------------------------------
# EvidenceAwarePoolingAttention: normalized zero-QKV + evidence-weighted pooling
# ---------------------------------------------------------------------------

class EvidenceAwarePoolingAttention(nn.Module):
    """Normalized zero-QKV attention + learnable region evidence pooling.

    R̂ = L2-normalize(R) per region
    M  = softmax(R̂R̂^T / tau + A + Delta)
    R' = R + alpha * M @ R
    beta = softmax(w^T R')        (per-sample region evidence weights)
    g = sum_i beta_i * r'_i        (evidence-weighted aggregation)
    """

    def __init__(self, feat_dim=768, num_regions=6, num_classes=2,
                 temperature=None, alpha_init=0.1,
                 prior_ipsilateral=0.2, prior_contralateral=0.1,
                 freeze_topo=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions
        self.temperature = nn.Parameter(
            torch.tensor(temperature if temperature is not None
                         else feat_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Anatomy prior A
        anatomy = torch.zeros(num_regions, num_regions)
        anatomy.fill_diagonal_(1.0)
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4)]:
            anatomy[i, j] = prior_ipsilateral
        for i, j in [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]:
            anatomy[i, j] = prior_contralateral
        self.register_buffer('anatomy_prior', anatomy)

        # Topology deviation
        self.topo_bias = nn.Parameter(torch.zeros(num_regions, num_regions))
        if freeze_topo:
            self.topo_bias.requires_grad_(False)

        # Region evidence scorer: per-sample per-region weight
        self.evidence_scorer = nn.Linear(feat_dim, 1, bias=False)

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, region_features, return_attention=False):
        R = region_features  # (B, 6, D)

        # L2-normalized zero-QKV
        R_norm = F.normalize(R, p=2, dim=2)  # (B, 6, D)
        attn_logits = torch.einsum('bid,bjd->bij', R_norm, R_norm) / self.temperature
        topo = self.anatomy_prior + self.topo_bias
        attn = F.softmax(attn_logits + topo.unsqueeze(0), dim=2)

        # Residual context
        context = torch.einsum('bij,bjd->bid', attn, R)
        R_enh = R + self.alpha * context

        # Evidence-aware pooling: learnable per-region weights
        scores = self.evidence_scorer(R_enh).squeeze(2)  # (B, 6)
        beta = F.softmax(scores, dim=1)                  # (B, 6)
        g = torch.einsum('bi,bid->bd', beta, R_enh)      # (B, D)

        logits = self.classifier(g)
        if return_attention:
            return logits, attn
        return logits
