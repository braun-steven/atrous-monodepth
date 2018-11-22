import torch
import torch.nn as nn

from monolab.networks.deeplab.resnet import ResNet
from monolab.networks.deeplab.submodules import Bottleneck
from monolab.networks.decoder import MonoDepthDecoder


class MonoDepthResNet50(nn.Module):
    def __init__(self, num_in_layers=3):
        super(MonoDepthResNet50, self).__init__()

        self.encoder = ResNet(
            in_channels=num_in_layers,
            block=Bottleneck,
            n_layer_blocks=[3, 4, 6, 3],
            output_stride=64,
        )

        self.decoder = MonoDepthDecoder()

    def forward(self, x):
        x1, x_pool1, x2, x3, x4, x_enc = self.encoder(x)

        disp1, disp2, disp3, disp4 = self.decoder(x1, x_pool1, x2, x3, x4, x_enc)

        return disp1, disp2, disp3, disp4
