import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


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

        x_skip1 = self.skip4(x_aspp_out, x_low4)
        x_skip2 = self.skip3(x_skip1, x_low3)
        x_skip3 = self.skip2(x_skip2, x_low2)
        x_skip4 = self.skip1(x_skip3, x_low1)

        return x_skip1, x_skip2, x_skip3, x_skip4

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
