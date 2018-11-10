import logging
from torch import nn
from monolab.networks.utils import init_weights

logger = logging.getLogger(__name__)


class ASPPModule(nn.Module):
    """
    Single Atrous Spatial Pooling module used in Deeplabv3+ encoder.
    """

    def __init__(self, inplanes: int, planes: int, rate: int):
        """
        Args:
            inplanes: Number of incoming channels
            planes: Number of outgoing channels
            rate: Atrous rate (dilation)
        """
        super(ASPPModule, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=rate,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        init_weights(self)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)


class Bottleneck(nn.Module):
    """
    Bottleneck blocks of ResNet.
    """

    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, rate=1, downsample: nn.Module = None
    ):
        """
        Initialize the ResNet bottleneck module
        Args:
            inplanes: Number of incoming channels
            planes: Number of channels for the middle part
            stride: Stride
            rate: Atrous rate (dilation)
            downsample: Downsampling module
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            dilation=rate,
            padding=rate,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes, out_channels=planes * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
