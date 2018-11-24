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
