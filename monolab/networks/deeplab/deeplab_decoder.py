import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from monolab.networks.resnet import upconv, conv, get_disp, upsample_nn


class Decoder(nn.Module):
    """
    Decoder module that has skip connections from the encoder:

    img-> (encoder: x_1 --> x_2 --> x_3 --> x_4 --> encout)
                     |       |       |       |        |
                     |       |       |       |        |
                     v       v       v       v        v
        (decout <-- ski <-- ski <-- ski <-- ski <-- decoder)

    """

    def __init__(
        self,
        x_low_inplanes_list: List[int],
        num_outplanes_list: List[int],
        backbone: str,
        BatchNorm=nn.BatchNorm2d,
    ):
        """
               Decoder module that gets the encoder output and some skip connections.
               Args:
                   x_low_inplanes_list: List of low level feature channels from the encoder
                   x_prev_inplanes_list: List of number of channels for each skip connection input
                   num_outplanes_list: List of number of out_channels for each skip connection
                   backbone: Backbone implementation
                   BatchNorm: Batchnorm implementation
                   num_skip_connections:
        """
        super(Decoder, self).__init__()

        if backbone == "resnet":
            encoder_out_planes = 256
        elif backbone == "xception":
            encoder_out_planes = 128
        else:
            raise NotImplementedError

        upsampling_scales = [1, 2, 2, 1]

        self.upconv5 = upconv(
            n_in=encoder_out_planes,
            n_out=256,
            kernel_size=3,
            scale=upsampling_scales[0],
        )
        self.iconv5 = conv(
            n_in=x_low_inplanes_list[0] + 256,
            n_out=256,
            kernel_size=3,
            stride=1,
        )

        self.upconv4 = upconv(
            n_in=256,
            n_out=128,
            kernel_size=3,
            scale=upsampling_scales[1],
        )
        self.iconv4 = conv(
            n_in=x_low_inplanes_list[1] + 128,
            n_out=128,
            kernel_size=3,
            stride=1,
        )
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(
            n_in=128,
            n_out=64,
            kernel_size=3,
            scale=upsampling_scales[2],
        )
        self.iconv3 = conv(
            n_in=x_low_inplanes_list[2] + 64 + 2,
            n_out=64,
            kernel_size=3,
            stride=1,
        )
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(
            n_in=64,
            n_out=32,
            kernel_size=3,
            scale=upsampling_scales[2],
        )
        self.iconv2 = conv(
            n_in=x_low_inplanes_list[3] + 32 + 2,
            n_out=32,
            kernel_size=3,
            stride=1,
        )
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(
            n_in=32, n_out=16, kernel_size=3, scale=2
        )
        self.iconv1 = conv(
            n_in=16 + 2, n_out=16, kernel_size=3, stride=1
        )
        self.disp1_layer = get_disp(16)

        self.upsample_nn = upsample_nn(scale=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self, x_aspp_out: Tensor, x_low1, x_low2, x_low3, x_low4
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass in decoder.
        Args:
            x_aspp_out (Tensor): Output from ASPP module
            x_low1 (Tensor): Low level features 1 from encoder
            x_low2 (Tensor): Low level features 2 from encoder
            x_low3 (Tensor): Low level features 3 from encoder
            x_low4 (Tensor): Low level features 4 from encoder
        Returns:
            Tuple of all four skip connection outputs
        """

        # skips
        skip1 = x_low1
        skip2 = x_low2
        skip3 = x_low3
        skip4 = x_low4

        # decoder
        upconv5 = self.upconv5(x_aspp_out)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = self.upsample_nn(self.disp4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = self.upsample_nn(self.disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = self.upsample_nn(self.disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)


        return self.disp1, self.disp2, self.disp3, self.disp4