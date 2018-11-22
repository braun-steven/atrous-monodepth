import math

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Tuple

from monolab.networks.deeplab.submodules import ASPPModule, Bottleneck
from monolab.networks.deeplab.resnet import ResNet
from monolab.networks.deeplab.xception import Xception
from monolab.networks.backbone import Backbone

logger = logging.getLogger(__name__)


class DCNNType(Enum):
    """DeepLabv3+ DCNN Type"""

    XCEPTION = 1
    RESNET = 2


class DeepLabv3Plus(Backbone):
    """
    DeepLabv3+ Model.
    """

    def __init__(
        self, dcnn_type: DCNNType, in_channels=3, output_stride=16, pretrained=False
    ):
        """
        Initialize the DeepLabev3+ Model.
        Args:
            dcnn_type: Type of the DCNN used in the encoder
            in_channels: Numper of input channels
            output_stride: Output stride
            pretrained: Flag if weights should be loaded from a pretrained model
        """
        logger.debug("Constructing DeepLabv3+ model...")
        logger.debug("Output stride: {}".format(output_stride))
        logger.debug("Number of Input Channels: {}".format(in_channels))
        super(DeepLabv3Plus, self).__init__(in_channels)

        self.dcnn_type = dcnn_type

        # Setup DCNN
        self.dcnn, dcnn_feature_size = self._create_dcnn(
            dcnn_type, in_channels, output_stride, pretrained
        )

        # Get ASPP rates
        rates = self._calculate_aspp_rates(output_stride)

        # ASPP Modules
        self.aspp1 = ASPPModule(inplanes=2048, planes=256, rate=rates[0])
        self.aspp2 = ASPPModule(inplanes=2048, planes=256, rate=rates[1])
        self.aspp3 = ASPPModule(inplanes=2048, planes=256, rate=rates[2])
        self.aspp4 = ASPPModule(inplanes=2048, planes=256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = self._create_global_avg_pooling()

        self.conv1 = nn.Conv2d(
            in_channels=1280, out_channels=256, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=256)

        # adopt [1x1, 2] for channel reduction.
        self.conv2 = nn.Conv2d(
            in_channels=dcnn_feature_size, out_channels=48, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=48)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=304,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1),
        )

        self.low_level_features_reduction = nn.Conv2d(
            in_channels=48, out_channels=2, kernel_size=3, stride=1, padding=1, bias=1
        )

        # TODO: missing init_weight(self) call?

    def _create_global_avg_pooling(self) -> nn.Module:
        """
        Create the global average pooling layer.
        Returns:
            Global average pooling layer
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(
                in_channels=2048, out_channels=256, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

    def _calculate_aspp_rates(self, output_stride: int) -> List[int]:
        """
        Calculates the ASPP rates based on the output stride.
        Args:
            output_stride: Output stride

        Returns:
            List of size four containing ASPP rates (dilation) for the four ASPP modules
        """
        if output_stride == 16:
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        return rates

    def _create_dcnn(
        self,
        dcnn_type: DCNNType,
        in_channels: int,
        output_stride: int,
        pretrained: bool,
    ) -> Tuple[nn.Module, int]:
        """
        Create the encoder DCNN.
        Args:
            dcnn_type: DCNN Type
            in_channels: Number of input channels
            output_stride: Output stride
            pretrained: Flag whether the DCNN should be loaded with pretrained weights

        Returns:
            Tuple with [0]=DCNN object, [1]=DCNN feature size
        """
        # DCNN Model
        if dcnn_type == DCNNType.XCEPTION:
            dcnn = Xception(
                inplanes=in_channels, output_stride=output_stride, pretrained=pretrained
            )
            dcnn_feature_size = 128
        elif dcnn_type == DCNNType.RESNET:
            dcnn = ResNet(
                in_channels=in_channels,
                block=Bottleneck,
                n_layer_blocks=[3, 4, 23, 3],
                output_stride=output_stride,
                pretrained=pretrained,
            )
            dcnn_feature_size = 256
        else:
            raise NotImplementedError
        return dcnn, dcnn_feature_size

    def forward(self, input):
        # Apply DCNN
        if self.dcnn_type == DCNNType.RESNET:
            _1, _pool1, low_level_features, _3, _4, x = self.dcnn(input)
        elif self.dcnn_type == DCNNType.XCEPTION:
            x, low_level_features = self.dcnn(input)

        # Get ASPP outputs
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        # Concat ASPP module outputs
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        size_ = (
            int(math.ceil(input.size()[-2] / 4)),
            int(math.ceil(input.size()[-1] / 4)),
        )
        x = F.upsample(x, size=size_, mode="bilinear", align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        # Concat DCNN output and ASPP output
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=x.size()[2:], mode="bilinear", align_corners=True)

        low_level_features = self.low_level_features_reduction(low_level_features)
        return x, low_level_features
