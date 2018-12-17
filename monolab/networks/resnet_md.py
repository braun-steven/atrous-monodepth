import torch
import torch.nn as nn

from monolab.networks.backbones.resnet import ResNet50, ResNet18
from monolab.networks.decoder import MonoDepthDecoder


class MonoDepthResNet50(nn.Module):
    def __init__(self, num_in_layers=3, pretrained=False, BatchNorm=nn.BatchNorm2d):
        super(MonoDepthResNet50, self).__init__()

        self.encoder = ResNet50(
            num_in_layers=num_in_layers,
            output_stride=64,
            pretrained=pretrained,
            BatchNorm=BatchNorm,
        )

        self.decoder = MonoDepthDecoder()

    def forward(self, x):
        x1, x_pool1, x2, x3, x4, x_enc = self.encoder(x)

        disp1, disp2, disp3, disp4 = self.decoder(x1, x_pool1, x2, x3, x4, x_enc)

        return disp1, disp2, disp3, disp4


class MonoDepthResNet18(nn.Module):
    def __init__(self, num_in_layers=3, pretrained=False, BatchNorm=nn.BatchNorm2d):
        super(MonoDepthResNet18, self).__init__()

        self.encoder = ResNet18(
            num_in_layers=num_in_layers,
            output_stride=64,
            pretrained=pretrained,
            BatchNorm=BatchNorm,
        )

        self.decoder = MonoDepthDecoder()

    def forward(self, x):
        x1, x_pool1, x2, x3, x4, x_enc = self.encoder(x)

        disp1, disp2, disp3, disp4 = self.decoder(x1, x_pool1, x2, x3, x4, x_enc)

        return disp1, disp2, disp3, disp4


if __name__ == "__main__":

    x = torch.rand(1, 3, 256, 512)

    net = MonoDepthResNet50(num_in_layers=3, pretrained=True)

    d1, d2, d3, d4 = net.forward(x)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
