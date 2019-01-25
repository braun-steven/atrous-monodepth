import torch
import torch.nn as nn

from monolab.networks.resnet import Resnet50, Resnet18
from monolab.networks.decoder import MonodepthDecoder, MonodepthDecoderSkipless


class MonodepthResnet50(nn.Module):
    def __init__(
        self,
        num_in_layers,
        skip_connections,
        output_stride=64,
        resblock_dilations=[1, 1, 1, 1],
    ):
        super(MonodepthResnet50, self).__init__()

        self.encoder = Resnet50(
            num_in_layers=num_in_layers,
            output_stride=output_stride,
            dilations=resblock_dilations,
        )

        if skip_connections:
            self.decoder = MonodepthDecoder(output_stride=output_stride)
        else:
            self.decoder = MonodepthDecoderSkipless(output_stride=output_stride)

    def forward(self, x):
        x1, x_pool1, x2, x3, x4, x5 = self.encoder(x)

        disp1, disp2, disp3, disp4 = self.decoder(x1, x_pool1, x2, x3, x4, x5)

        return [disp1, disp2, disp3, disp4]


class MonodepthResnet18(nn.Module):
    def __init__(
        self,
        num_in_layers,
        skip_connections,
        output_stride=64,
        resblock_dilations=[1, 1, 1, 1],
    ):
        super(MonodepthResnet18, self).__init__()

        self.encoder = Resnet18(
            num_in_layers=num_in_layers,
            output_stride=output_stride,
            dilations=resblock_dilations,
        )

        if skip_connections:
            self.decoder = MonodepthDecoder(output_stride=output_stride)
        else:
            self.decoder = MonodepthDecoderSkipless(output_stride=output_stride)

    def forward(self, x):
        x1, x_pool1, x2, x3, x4, x5 = self.encoder(x)

        disp1, disp2, disp3, disp4 = self.decoder(x1, x_pool1, x2, x3, x4, x5)

        return [disp1, disp2, disp3, disp4]


if __name__ == "__main__":

    x = torch.rand(1, 3, 256, 512)

    net = MonodepthResnet50(num_in_layers=3, skip_connections=False, output_stride=64)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    d1, d2, d3, d4 = net.forward(x)

    print(d1.size())
    print(d2.size())
    print(d3.size())
    print(d4.size())
