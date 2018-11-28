import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class Decoder(nn.Module):
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

        x_prev_inplanes_list = [encoder_out_planes] + [encoder_out_planes // 2**i for
                                                       i in range(1, 4)]

        # Create skip connections
        self.skip1 = SkipConnection(
            x_low_inplanes=x_low_inplanes_list[0],
            x_prev_inplanes=x_prev_inplanes_list[0],
            num_out_planes=num_outplanes_list[0],
            BatchNorm=BatchNorm,
        )
        self.skip2 = SkipConnection(
            x_low_inplanes=x_low_inplanes_list[1],
            x_prev_inplanes=x_prev_inplanes_list[1],
            num_out_planes=num_outplanes_list[1],
            BatchNorm=BatchNorm,
        )
        self.skip3 = SkipConnection(
            x_low_inplanes=x_low_inplanes_list[2],
            x_prev_inplanes=x_prev_inplanes_list[2],
            num_out_planes=num_outplanes_list[2],
            BatchNorm=BatchNorm,
        )
        self.skip4 = SkipConnection(
            x_low_inplanes=x_low_inplanes_list[3],
            x_prev_inplanes=x_prev_inplanes_list[3],
            num_out_planes=num_outplanes_list[3],
            BatchNorm=BatchNorm,
        )
        self._init_weight()

    def forward(
        self, x_aspp_out: Tensor, low_level_feats: List[Tensor]
    ) -> List[Tensor]:
        """
        Forward pass in decoder.
        Args:
            x_aspp_out (Tensor): Output from ASPP module
            low_level_feats (List[Tensor]): List of low level features from the encoder
            DCNN

        Returns:
            List of all skip connection outputs
        """
        result = []

        x = self.skip1(x_aspp_out, low_level_feats[0])
        result.append(x)
        x = self.skip2(x, low_level_feats[1])
        result.append(x)
        x = self.skip3(x, low_level_feats[2])
        result.append(x)
        x = self.skip4(x, low_level_feats[3])
        result.append(x)

        return result

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SkipConnection(nn.Module):
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
        self.relu = nn.ReLU()

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
            nn.ReLU(),
            nn.Conv2d(
                in_channels=x_prev_inplanes,
                out_channels=x_prev_inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(x_prev_inplanes),
            nn.ReLU(),
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
