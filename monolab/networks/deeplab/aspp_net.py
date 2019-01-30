from torch import nn

from monolab.networks.resnet import conv, maxpool, resblock
from monolab.networks.deeplab.aspp import ASPP
from monolab.networks.decoder import MonodepthDecoder, MonodepthDecoderSkipless


class ASPPNet(nn.Module):
    """ Resnet implementation with the quirk's of Godard et al. (2017), https://github.com/mrharicot/monodepth
        We made the resnet size variable by letting the user give a list of numbers of blocks for the three resblocks
    """

    def __init__(
        self,
        num_in_layers,
        blocks,
        output_stride=64,
        encoder_dilations=[1, 1, 1, 1],
        aspp_dilations=[1, 6, 12, 18],
    ):
        """
        Args:
            num_in_layers: number of input channels (3 for rgb)
            blocks: list of length 4, contains the numbers of blocks -> [3, 4, 6, 3] for Resnet50
        """
        super(ASPPNet, self).__init__()

        if output_stride == 64:
            strides = [2, 2, 2, 2]
        elif output_stride == 32:
            strides = [1, 2, 2, 2]
        # output_stride = 16 is the standard for deeplabv3+
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
        else:
            raise ValueError("Please specify a valid output stride")

        # encoder
        self.conv1 = conv(
            n_in=num_in_layers, n_out=64, kernel_size=7, stride=2
        )  # H/2 - 64D
        self.pool1 = maxpool(kernel_size=3)  # H/4 - 64D
        self.conv2 = resblock(
            n_in=64,
            num_layers=64,
            num_blocks=blocks[0],
            stride=strides[0],
            dilation=encoder_dilations[0],
        )  # H/8 - 256D
        self.aspp = ASPP(inplanes=256, dilations=aspp_dilations)
        self.conv3 = resblock(
            n_in=256,
            num_layers=128,
            num_blocks=blocks[1],
            stride=strides[1],
            dilation=encoder_dilations[1],
        )  # H/16 -  512D
        self.conv4 = resblock(
            n_in=512,
            num_layers=256,
            num_blocks=blocks[2],
            stride=strides[2],
            dilation=encoder_dilations[2],
        )  # H/32 - 1024D
        self.conv5 = resblock(
            n_in=1024,
            num_layers=512,
            num_blocks=blocks[3],
            stride=strides[3],
            dilation=encoder_dilations[3],
        )  # H/64 - 2048D

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)

        x2 = self.conv2(x_pool1)
        x2_aspp = self.aspp(x2)
        x3 = self.conv3(x2_aspp)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return x1, x_pool1, x2, x3, x4, x5


class MonodepthASPPNet(nn.Module):
    def __init__(
        self,
        num_in_layers,
        skip_connections,
        output_stride=64,
        resblock_dilations=[1, 1, 1, 1],
        aspp_dilations=[1, 6, 12, 18],
    ):
        super(MonodepthASPPNet, self).__init__()

        self.encoder = ASPPNet(
            num_in_layers=num_in_layers,
            blocks=[3, 4, 6, 3],
            output_stride=output_stride,
            encoder_dilations=resblock_dilations,
            aspp_dilations=aspp_dilations,
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
    import numpy as np
    import torch

    img = np.random.randn(1, 3, 256, 512)
    img = torch.Tensor(img)

    model = MonodepthASPPNet(
        3, skip_connections=True, output_stride=16, aspp_dilations=[1, 6, 12, 18]
    )

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.eval()
    y = model.forward(img)

    for yi in y:
        print(yi.size())
