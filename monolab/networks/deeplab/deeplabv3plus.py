import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

from monolab.networks.deeplab.submodules import ASPPModule, Bottleneck
from monolab.networks.deeplab.resnet import ResNet
from monolab.networks.deeplab.xception import Xception
from monolab.networks.backbone import Backbone

class DCNNType(Enum):
    """DeepLabv3+ DCNN Type"""
    XCEPTION = 1
    RESNET = 2


class DeepLabv3Plus(Backbone):
    """
    DeepLabv3+ Model.
    """

    def __init__(
        self,
        dcnn_type: DCNNType,
        n_input_channels=3,
        n_classes=21,
        output_stride=16,
        pretrained=False,
        _print=True,
    ):
        """
        Initialize the DeepLabev3+ Model.
        Args:
            dcnn_type: Type of the DCNN used in the encoder
            n_input_channels: Numper of input channels
            n_classes: Number of classes
            output_stride: Output stride
            pretrained: Flag if weights should be loaded from a pretrained model
            _print: Print flag
        """
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(output_stride))
            print("Number of Input Channels: {}".format(n_input_channels))
        super(DeepLabv3Plus, self).__init__(n_input_channels)

        # DCNN Model
        if dcnn_type == DCNNType.XCEPTION:
            self.dcnn = Xception(n_input_channels, output_stride, pretrained)
            dcnn_feature_size = 128
        elif dcnn_type == DCNNType.RESNET:
            self.dcnn = ResNet(
                n_input_channels=n_input_channels,
                block=Bottleneck,
                layers=[3, 4, 23, 3],
                output_stride=output_stride,
                pretrained=pretrained,
            )
            dcnn_feature_size = 256
        else:
            raise NotImplementedError

        # ASPP
        if output_stride == 16:
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(2048, 256, rate=rates[0])
        self.aspp2 = ASPPModule(2048, 256, rate=rates[1])
        self.aspp3 = ASPPModule(2048, 256, rate=rates[2])
        self.aspp4 = ASPPModule(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(dcnn_feature_size, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1),
        )

        # TODO: missing init_weight(self) call?

    def forward(self, x):
        # Apply DCNN
        x, low_level_features = self.dcnn(x)

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
        x = F.upsample(
            x,
            size=(int(math.ceil(x.size()[-2] / 4)), int(math.ceil(x.size()[-1] / 4))),
            mode="bilinear",
            align_corners=True,
        )

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        # TODO: Shapes of x and low_level_feat do not fit
        # Concat DCNN output and ASPP output
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=x.size()[2:], mode="bilinear", align_corners=True)

        return x, low_level_features

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeepLabv3Plus(
        dcnn_type=DCNNType.XCEPTION,
        n_input_channels=3,
        n_classes=21,
        output_stride=16,
        pretrained=True,
        _print=True,
    )
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
