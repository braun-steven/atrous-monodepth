import logging
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import List, Tuple

from monolab.networks.deeplab.submodules import Bottleneck
from monolab.networks.utils import init_weights

logger = logging.getLogger(__name__)


class ResNet(nn.Module):
    """
    ResNet implementation.
    """

    def __init__(
        self,
        in_channels: int,
        block: Bottleneck,
        n_layer_blocks: List[int],
        output_stride=16,
        pretrained=False,
    ):
        """
        Initialize ResNet module
        Args:
            in_channels: Number of incoming channels
            block: Block module
            n_layer_blocks: Number of blocks in each main layer
            output_stride: Output stride
            pretrained: Flag whether to use pretrained weights
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block=block,
            planes=64,
            blocks=n_layer_blocks[0],
            stride=strides[0],
            rate=rates[0],
        )

        self.layer2 = self._make_layer(
            block=block,
            planes=128,
            blocks=n_layer_blocks[1],
            stride=strides[1],
            rate=rates[1],
        )

        self.layer3 = self._make_layer(
            block=block,
            planes=256,
            blocks=n_layer_blocks[2],
            stride=strides[2],
            rate=rates[2],
        )

        self.layer4 = self._make_multigrid_unit(
            block=block, planes=512, blocks=blocks, stride=strides[3], rate=rates[3]
        )

        init_weights(self)

        if pretrained:
            self._load_pretrained_model_state()

    def _make_layer(
        self, block: Bottleneck, planes: int, blocks: int, stride=1, rate=1
    ) -> nn.Sequential:
        """
        Generate a layer block
        Args:
            block: Block type
            planes: Number of planes
            blocks: Number of blocks
            stride: Convolution stride for each convolution layer
            rate: Atrous rate
        :return: Sequential model
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = [
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                rate=rate,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def _make_multigrid_unit(
        self, block: Bottleneck, planes: int, blocks: List[int] = None, stride=1, rate=1
    ) -> nn.Sequential:
        """
        Generate a multigrid block
        Args:
            block: Block type
            planes: Number of planes
            blocks: List of blocks
            stride: Convolution stride for each convolution layer
            rate: Atrous rate
        Returns:
             Sequential model
        """
        if blocks is None:
            blocks = [1, 2, 4]

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = [
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                rate=blocks[0] * rate,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    rate=blocks[i] * rate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _load_pretrained_model_state(self):
        """Load the pretrained ResNet101 model state."""
        pretrain_dict = model_zoo.load_url(
            "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
