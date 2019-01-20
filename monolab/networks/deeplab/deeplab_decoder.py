from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

from monolab.networks.resnet import conv, get_disp, upconv, upsample_nn

logger = logging.getLogger(__name__)

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
        decoder_type="deeplab",
    ):
        """
        Decoder module that gets the encoder output and some skip connections.
        Args:
            x_low_inplanes_list: List of low level feature channels from the encoder
            x_prev_inplanes_list: List of number of channels for each skip connection input
            num_outplanes_list: List of number of out_channels for each skip connection
            backbone: Backbone implementation
            BatchNorm: Batchnorm implementation
            decoder_type: Type of the decoder architecture, can be either "deeplab" or "godard"
        """
        super(Decoder, self).__init__()
        self.decoder_type = decoder_type

        if backbone == "resnet":
            encoder_out_planes = 256
        else:
            raise NotImplementedError

        if decoder_type == "deeplab":
            # Reduce input channels by a factor of 1/2 at each skip connection
            x_prev_inplanes_list = [encoder_out_planes] + [
                encoder_out_planes // 2 ** i for i in range(1, 4)
            ]

            # Create skip connections
            self.skip4 = SkipConnection(
                x_low_inplanes=x_low_inplanes_list[0],
                x_prev_inplanes=x_prev_inplanes_list[0],
                num_out_planes=num_outplanes_list[0],
                BatchNorm=BatchNorm,
            )
            self.skip3 = SkipConnection(
                x_low_inplanes=x_low_inplanes_list[1],
                x_prev_inplanes=x_prev_inplanes_list[1],
                num_out_planes=num_outplanes_list[1],
                BatchNorm=BatchNorm,
            )
            self.skip2 = SkipConnection(
                x_low_inplanes=x_low_inplanes_list[2],
                x_prev_inplanes=x_prev_inplanes_list[2],
                num_out_planes=num_outplanes_list[2],
                BatchNorm=BatchNorm,
            )
            self.skip1 = SkipConnection(
                x_low_inplanes=x_low_inplanes_list[3],
                x_prev_inplanes=x_prev_inplanes_list[3],
                num_out_planes=num_outplanes_list[3],
                BatchNorm=BatchNorm,
            )

            self.disp4 = get_disp(num_outplanes_list[0])
            self.disp3 = get_disp(num_outplanes_list[1])
            self.disp2 = get_disp(num_outplanes_list[2])
            self.disp1 = get_disp(num_outplanes_list[3])

        elif decoder_type == "godard":
            upsampling_scales = [1, 2, 2, 1]
            self.upconv5 = upconv(
                n_in=encoder_out_planes,
                n_out=256,
                kernel_size=3,
                scale=upsampling_scales[0],
            )
            self.iconv5 = conv(
                n_in=x_low_inplanes_list[0] + 256, n_out=256, kernel_size=3, stride=1
            )

            self.upconv4 = upconv(
                n_in=256, n_out=128, kernel_size=3, scale=upsampling_scales[1]
            )
            self.iconv4 = conv(
                n_in=x_low_inplanes_list[1] + 128, n_out=128, kernel_size=3, stride=1
            )
            self.disp4_layer = get_disp(128)

            self.upconv3 = upconv(
                n_in=128, n_out=64, kernel_size=3, scale=upsampling_scales[2]
            )
            self.iconv3 = conv(
                n_in=x_low_inplanes_list[2] + 64 + 2, n_out=64, kernel_size=3, stride=1
            )
            self.disp3_layer = get_disp(64)

            self.upconv2 = upconv(
                n_in=64, n_out=32, kernel_size=3, scale=upsampling_scales[2]
            )
            self.iconv2 = conv(
                n_in=x_low_inplanes_list[3] + 32 + 2, n_out=32, kernel_size=3, stride=1
            )
            self.disp2_layer = get_disp(32)

            self.upconv1 = upconv(n_in=32, n_out=16, kernel_size=3, scale=2)
            self.iconv1 = conv(n_in=16 + 2, n_out=16, kernel_size=3, stride=1)
            self.disp1_layer = get_disp(16)

            self.upsample_nn = upsample_nn(scale=2)

        self._init_weight()

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
        if self.decoder_type == "deeplab":
            x_skip4 = self.skip4(x_aspp_out, x_low4)
            x_skip3 = self.skip3(x_skip4, x_low3)
            x_skip2 = self.skip2(x_skip3, x_low2)
            x_skip1 = self.skip1(x_skip2, x_low1)

            # Compute disparity maps
            disp4 = self.disp4(x_skip4)
            disp3 = self.disp3(x_skip3)
            disp2 = self.disp2(x_skip2)
            disp1 = self.disp1(x_skip1)

            return [disp1, disp2, disp3, disp4]

        elif self.decoder_type == "godard":
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
            disp4 = self.disp4_layer(iconv4)
            udisp4 = self.upsample_nn(disp4)

            upconv3 = self.upconv3(iconv4)
            concat3 = torch.cat((upconv3, skip2, udisp4), 1)
            iconv3 = self.iconv3(concat3)
            disp3 = self.disp3_layer(iconv3)
            udisp3 = self.upsample_nn(disp3)

            upconv2 = self.upconv2(iconv3)
            concat2 = torch.cat((upconv2, skip1, udisp3), 1)
            iconv2 = self.iconv2(concat2)
            disp2 = self.disp2_layer(iconv2)
            udisp2 = self.upsample_nn(disp2)

            upconv1 = self.upconv1(iconv2)
            concat1 = torch.cat((upconv1, udisp2), 1)
            iconv1 = self.iconv1(concat1)
            disp1 = self.disp1_layer(iconv1)

            return disp1, disp2, disp3, disp4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SkipConnection(nn.Module):
    """
    Skip connection module that takes one input from the encoder and one input from
    the decoder pipeline and generates the next tensor in the decoder.
    """

    def __init__(
        self,
        x_prev_inplanes: int,
        x_low_inplanes: int,
        BatchNorm: nn.BatchNorm2d,
        num_out_planes: int,
    ):
        super(SkipConnection, self).__init__()

        # 1x1 conv2d block to reduce channel size
        # TODO: Is this necessary?
        num_out_conv2 = int(x_low_inplanes / 2)
        self.conv1 = nn.Conv2d(
            in_channels=x_low_inplanes,
            out_channels=num_out_conv2,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = BatchNorm(num_out_conv2)
        self.relu = nn.ELU()

        # Conv2d block as of deeplabv3+
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=x_prev_inplanes + num_out_conv2,
                out_channels=x_prev_inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(x_prev_inplanes),
            nn.ELU(),
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=x_prev_inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(x_prev_inplanes),
            nn.ELU(),
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=num_out_planes,
                kernel_size=1,
                stride=1,
            ),
        )

        self._init_weight()

    def forward(self, x_prev, x_low) -> Tensor:
        """
        Forward pass on skip connection.
        Args:
            x_prev: Previous output in decoder
            x_low: Low level features from encoder (dcnn)

        Returns:
            Skip connection output
        """

        # Apply 1x1 conv to x_low
        x_low = self.conv1(x_low)
        x_low = self.bn1(x_low)
        x_low = self.relu(x_low)

        x_prev = F.interpolate(
            x_prev, size=x_low.size()[2:], mode="bilinear", align_corners=True
        )

        # Concat
        x = torch.cat((x_prev, x_low), dim=1)
        # Reduce
        x = self.last_conv(x)

        # Interpolate
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderSkipless(nn.Module):
    """
    Decoder module that has no skip connections from the encoder:

    img-> (encoder: x_1 --> x_2 --> x_3 --> x_4 --> encout)
                                                      |
                                                      |
                                                      v
        (decout <-- ski <-- ski <-- ski <-- ski <-- decoder)

    """

    def __init__(
        self,
        x_low_inplanes_list: List[int],
        num_outplanes_list: List[int],
        backbone: str,
        BatchNorm=nn.BatchNorm2d,
        decoder_type="deeplab",
    ):
        """
        Decoder module that gets the encoder output and some skip connections.
        Args:
            x_low_inplanes_list: List of low level feature channels from the encoder
            x_prev_inplanes_list: List of number of channels for each skip connection input
            num_outplanes_list: List of number of out_channels for each skip connection
            backbone: Backbone implementation
            BatchNorm: Batchnorm implementation
            decoder_type: Type of the decoder architecture, can be either "deeplab" or "godard"
        """
        super(DecoderSkipless, self).__init__()
        self.decoder_type = decoder_type

        if backbone == "resnet":
            encoder_out_planes = 256
        else:
            raise NotImplementedError

        upsampling_scales = [1, 2, 2, 1]

        if decoder_type == "deeplab":
            x_prev_inplanes_list = [encoder_out_planes] + [
                encoder_out_planes // 2 ** i for i in range(1, 4)
            ]
            print("x_prev_inplanes_list: %s" % x_prev_inplanes_list)
            self.convblock4 = ConvBlock(
                x_prev_inplanes=x_prev_inplanes_list[0],
                num_out_planes=num_outplanes_list[0],
                BatchNorm=BatchNorm,
            )
            self.convblock3 = ConvBlock(
                x_prev_inplanes=x_prev_inplanes_list[1],
                num_out_planes=num_outplanes_list[1],
                BatchNorm=BatchNorm,
            )
            self.convblock2 = ConvBlock(
                x_prev_inplanes=x_prev_inplanes_list[2],
                num_out_planes=num_outplanes_list[2],
                BatchNorm=BatchNorm,
            )
            self.convblock1 = ConvBlock(
                x_prev_inplanes=x_prev_inplanes_list[3],
                num_out_planes=num_outplanes_list[3],
                BatchNorm=BatchNorm,
            )

            self.disp4 = get_disp(num_outplanes_list[0])
            self.disp3 = get_disp(num_outplanes_list[1])
            self.disp2 = get_disp(num_outplanes_list[2])
            self.disp1 = get_disp(num_outplanes_list[3])

        elif decoder_type == "godard":
            self.upconv5 = upconv(
                n_in=encoder_out_planes,
                n_out=256,
                kernel_size=3,
                scale=upsampling_scales[0],
            )
            self.iconv5 = conv(n_in=256, n_out=256, kernel_size=3, stride=1)

            self.upconv4 = upconv(
                n_in=256, n_out=128, kernel_size=3, scale=upsampling_scales[1]
            )
            self.iconv4 = conv(n_in=128, n_out=128, kernel_size=3, stride=1)
            self.disp4_layer = get_disp(128)

            self.upconv3 = upconv(
                n_in=128, n_out=64, kernel_size=3, scale=upsampling_scales[2]
            )
            self.iconv3 = conv(n_in=64 + 2, n_out=64, kernel_size=3, stride=1)
            self.disp3_layer = get_disp(64)

            self.upconv2 = upconv(
                n_in=64, n_out=32, kernel_size=3, scale=upsampling_scales[2]
            )
            self.iconv2 = conv(n_in=32 + 2, n_out=32, kernel_size=3, stride=1)
            self.disp2_layer = get_disp(32)

            self.upconv1 = upconv(n_in=32, n_out=16, kernel_size=3, scale=2)
            self.iconv1 = conv(n_in=16 + 2, n_out=16, kernel_size=3, stride=1)
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
            x_low1: Not used
            x_low2: Not used
            x_low3: Not used
            x_low4: Not used
        Returns:
            Tuple of all four skip connection outputs
        """

        if self.decoder_type == "deeplab":
            x4 = self.convblock4(x_aspp_out)
            x3 = self.convblock3(x4)
            x2 = self.convblock2(x3)
            x1 = self.convblock1(x2)

            # Compute disparity maps
            disp4 = self.disp4(x4)
            disp3 = self.disp3(x3)
            disp2 = self.disp2(x2)
            disp1 = self.disp1(x1)

            return [disp1, disp2, disp3, disp4]
        elif self.decoder_type == "godard":
            # decoder
            upconv5 = self.upconv5(x_aspp_out)
            iconv5 = self.iconv5(upconv5)

            upconv4 = self.upconv4(iconv5)
            iconv4 = self.iconv4(upconv4)
            disp4 = self.disp4_layer(iconv4)
            udisp4 = self.upsample_nn(disp4)

            upconv3 = self.upconv3(iconv4)
            concat3 = torch.cat((upconv3, udisp4), 1)
            iconv3 = self.iconv3(concat3)
            disp3 = self.disp3_layer(iconv3)
            udisp3 = self.upsample_nn(disp3)

            upconv2 = self.upconv2(iconv3)
            concat2 = torch.cat((upconv2, udisp3), 1)
            iconv2 = self.iconv2(concat2)
            disp2 = self.disp2_layer(iconv2)
            udisp2 = self.upsample_nn(disp2)

            upconv1 = self.upconv1(iconv2)
            concat1 = torch.cat((upconv1, udisp2), 1)
            iconv1 = self.iconv1(concat1)
            disp1 = self.disp1_layer(iconv1)

        return [disp1, disp2, disp3, disp4]


class ConvBlock(nn.Module):
    """
    Convolution block that behaves like the skip connection blocks in the normal Deeplab
    decoder but without skip connections.
    """

    def __init__(
        self, x_prev_inplanes: int, BatchNorm: nn.BatchNorm2d, num_out_planes: int
    ):
        super(ConvBlock, self).__init__()
        # Conv2d block as of deeplabv3+ but without skip connection from encoder
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=x_prev_inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(x_prev_inplanes),
            nn.ELU(),
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=x_prev_inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(x_prev_inplanes),
            nn.ELU(),
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=num_out_planes,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x):
        # x = F.interpolate(
        #     x, size=x.size()[2:], mode="bilinear", align_corners=True
        # )
        x = self.last_conv(x)
        # Interpolate
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        return x
