"""CNN backbone definitions for feature extraction.

Provides ResNet and DenseNet variants with ImageNet pre-trained weights.
The final pooling and classifier layers are stripped so we get feature maps.
Compatible with torchvision >= 0.12 (auto-detects weights= vs pretrained= API).
"""

import torch.nn as nn
import torchvision


def _get_weights(name):
    """Return the appropriate weights enum or string for the installed torchvision."""
    try:
        # torchvision >= 0.13
        return getattr(torchvision.models, name).DEFAULT
    except AttributeError:
        return 'DEFAULT'


class ModifiedDensenet121(nn.Module):
    """DenseNet-121 with classifier head removed."""

    def __init__(self):
        super().__init__()
        densenet = torchvision.models.densenet121(
            weights=_get_weights('DenseNet121_Weights')
        )
        self.features = nn.Sequential(*list(densenet.children())[:-1])

    def forward(self, x):
        return self.features(x)


def _resnet18():
    return torchvision.models.resnet18(weights=_get_weights('ResNet18_Weights'))


def _resnet34():
    return torchvision.models.resnet34(weights=_get_weights('ResNet34_Weights'))


def _resnet50():
    return torchvision.models.resnet50(weights=_get_weights('ResNet50_Weights'))


def _resnet101():
    return torchvision.models.resnet101(weights=_get_weights('ResNet101_Weights'))


def _densenet121():
    return ModifiedDensenet121()


def _mobilenet_v3_small():
    return torchvision.models.mobilenet_v3_small(
        weights=_get_weights('MobileNet_V3_Small_Weights')
    )


# (model_constructor, num_output_features)
model_dict = {
    'resnet18':          (_resnet18, 512),
    'resnet34':          (_resnet34, 512),
    'resnet50':          (_resnet50, 2048),
    'resnet101':         (_resnet101, 2048),
    'densenet121':       (_densenet121, 1024),
    'mobilenet_v3_small': (_mobilenet_v3_small, 576),
}
