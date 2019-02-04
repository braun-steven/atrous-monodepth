import logging
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ASPPModule(nn.Module):
    """
    A single ASPP Module consisting of Conv2d > BN > ReLU
    """

    def __init__(
        self, inplanes, planes, kernel_size, padding, dilation, BatchNorm=nn.BatchNorm2d
    ):
        """
        Construct an ASPP Module.
        Args:
            inplanes: Number of inplanes
            planes: Number of outplanes
            kernel_size: Convolution kernel size
            padding: Convolution padding
            dilation: Convultion dilation
            BatchNorm: Batchnorm function
        """
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """
    ASPP contains a variable number of ASPPModules and a global average pooling module
    """

    def __init__(
        self,
        inplanes: int,
        dilations: List[int],
        BatchNorm=nn.BatchNorm2d,
        use_global_average_pooling=True,
    ):
        """
        Construct the ASPP global module that contains several ASPP submodules
        Args:
            backbone (str): Backbone name
            dilations: List of dilations (atrous rates)
            BatchNorm: Batchnorm function
            use_global_average_pooling: Flag to enable global average pooling layer 
        """
        super(ASPP, self).__init__()
        self.use_global_average_pooling = use_global_average_pooling

        # Create variable length module list of parallel ASPP modules with different dilations
        self.aspp_modules = nn.ModuleList()

        # First ASPP module with 1x1 conv
        self.aspp_modules.append(
            ASPPModule(
                inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm
            )
        )

        # Following ASPP modules with 3x3 conv
        for dil in dilations[1:]:
            self.aspp_modules.append(
                ASPPModule(
                    inplanes=inplanes,
                    planes=256,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    BatchNorm=BatchNorm,
                )
            )

        num_aspp_modules = len(dilations)

        # Global average pooling layer
        if self.use_global_average_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                BatchNorm(256),
                nn.ReLU(),
            )
            num_aspp_modules += 1

        self.conv1 = nn.Conv2d(256 * num_aspp_modules, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        # Collect results from ASPP modules
        aspp_results = [l(x) for l in self.aspp_modules]

        # Apply global average pooling if enabled
        if self.use_global_average_pooling:
            x_gl_avg_pool = self.global_avg_pool(x)
            x_gl_avg_pool = F.interpolate(
                x_gl_avg_pool,
                size=aspp_results[-1].size()[2:],
                mode="bilinear",
                align_corners=True,
            )
            x = aspp_results + [x_gl_avg_pool]
        else:
            x = aspp_results

        x = torch.cat(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
