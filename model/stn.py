"""Spatial Transformer Network (STN) for anatomical alignment.

Predicts an affine transformation matrix (6 parameters) to align chest X-rays
based on lung segmentation masks, enabling consistent multi-region analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """Lightweight STN that regresses 6 affine parameters from a mask/image.

    Parameters
    ----------
    in_shape : tuple
        Input shape as (C, H, W).
    mask_resize : int
        Divisor for adaptive max-pooling to reduce spatial size.
    dense_neurons : int
        Number of neurons in the hidden FC layer.
    freeze_align_model : bool
        If True, all parameters are frozen (inference-only).
    """

    def __init__(self, in_shape=(1, 512, 512), mask_resize=512,
                 dense_neurons=50, freeze_align_model=False):
        super().__init__()
        assert in_shape[1] % mask_resize == 0, \
            "STN mask_resize must divide input height evenly."

        trainable = not freeze_align_model
        self._last_grid = None

        self.pool1 = nn.MaxPool2d(
            kernel_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_shape[0], 20, kernel_size=5)
        if not trainable:
            self.conv1.requires_grad_(False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        if not trainable:
            self.conv2.requires_grad_(False)

        self.flatten = nn.Flatten()
        flat_size = self._calc_flatten_size(in_shape)

        self.fc1 = nn.Linear(flat_size, dense_neurons)
        if not trainable:
            self.fc1.requires_grad_(False)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_neurons, 6)
        if not trainable:
            self.fc2.requires_grad_(False)

        # Initialise to identity transform
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def _calc_flatten_size(self, in_shape):
        dummy = torch.randn(1, *in_shape)
        x = self.pool1(dummy)
        x = self.pool2(self.conv1(x))
        x = self.pool3(self.conv2(x))
        return x.numel()

    def get_last_grid(self):
        if self._last_grid is None:
            raise RuntimeError("No grid computed yet — run forward pass first.")
        return self._last_grid

    def forward(self, x):
        # Collapse multi-channel to single-channel if needed
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)

        xs = self.pool1(x)
        xs = self.pool2(self.conv1(xs))
        xs = self.pool3(self.conv2(xs))
        xs = self.flatten(xs)
        xs = self.relu(self.fc1(xs))
        theta = self.fc2(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        self._last_grid = grid
        return F.grid_sample(x, grid, align_corners=False)
