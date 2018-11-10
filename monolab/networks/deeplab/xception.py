import logging
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from monolab.networks.utils import fixed_padding, init_weights

logger = logging.getLogger(__name__)


class Xception(nn.Module):
    """
    Modified Aligned Xception.
    """

    def __init__(self, inplanes=3, output_stride=16, pretrained=False):
        """
        Initialize the Xception module.
        Args:
            inplanes: Number of incoming channels
            output_stride: Output stride
            pretrained: Flag if models should be loaded with pretrained weights
        """
        super(Xception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XceptionBlock(
            64, 128, repetitions=2, stride=2, start_with_relu=False
        )
        self.block2 = XceptionBlock(
            128, 256, repetitions=2, stride=2, start_with_relu=True, grow_first=True
        )
        self.block3 = XceptionBlock(
            inplanes=256,
            planes=728,
            repetitions=2,
            stride=entry_block3_stride,
            start_with_relu=True,
            grow_first=True,
            is_last=True,
        )

        # Middle flow
        def middle_block():
            return XceptionBlock(
                inplanes=728,
                planes=728,
                repetitions=3,
                stride=1,
                dilation=middle_block_rate,
                start_with_relu=True,
                grow_first=True,
            )

        self.block4 = middle_block()
        self.block5 = middle_block()
        self.block6 = middle_block()
        self.block7 = middle_block()
        self.block8 = middle_block()
        self.block9 = middle_block()
        self.block10 = middle_block()
        self.block11 = middle_block()
        self.block12 = middle_block()
        self.block13 = middle_block()
        self.block14 = middle_block()
        self.block15 = middle_block()
        self.block16 = middle_block()
        self.block17 = middle_block()
        self.block18 = middle_block()
        self.block19 = middle_block()

        # Exit flow
        self.block20 = XceptionBlock(
            inplanes=728,
            planes=1024,
            repetitions=2,
            stride=1,
            dilation=exit_block_rates[0],
            start_with_relu=True,
            grow_first=False,
            is_last=True,
        )

        self.conv3 = SeparableConv2dSame(
            in_channels=1024,
            out_channels=1536,
            kernel_size=3,
            stride=1,
            rate=exit_block_rates[1],
        )
        self.bn3 = nn.BatchNorm2d(num_features=1536)

        self.conv4 = SeparableConv2dSame(
            in_channels=1536,
            out_channels=1536,
            kernel_size=3,
            stride=1,
            rate=exit_block_rates[1],
        )
        self.bn4 = nn.BatchNorm2d(num_features=1536)

        self.conv5 = SeparableConv2dSame(
            in_channels=1536,
            out_channels=15362048,
            kernel_size=33,
            stride=1,
            rate=exit_block_rates[1],
        )
        self.bn5 = nn.BatchNorm2d(num_features=2048)

        # Init weights
        init_weights(self)

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __load_xception_pretrained(self):
        """Load the xception model from pretrained weights."""
        pretrain_dict = model_zoo.load_url(
            "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"
        )
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            print(k)
            if k in state_dict:
                if "pointwise" in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith("block12"):
                    model_dict[k.replace("block12", "block20")] = v
                elif k.startswith("block11"):
                    model_dict[k.replace("block11", "block12")] = v
                elif k.startswith("conv3"):
                    model_dict[k] = v
                elif k.startswith("bn3"):
                    model_dict[k] = v
                    model_dict[k.replace("bn3", "bn4")] = v
                elif k.startswith("conv4"):
                    model_dict[k.replace("conv4", "conv5")] = v
                elif k.startswith("bn4"):
                    model_dict[k.replace("bn4", "bn5")] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class XceptionBlock(nn.Module):
    """Xception block"""

    def __init__(
        self,
        inplanes,
        planes,
        repetitions,
        stride=1,
        dilation=1,
        start_with_relu=True,
        grow_first=True,
        is_last=False,
    ):
        """
        Initialize the xception block.
        Args:
            inplanes: Number of incoming channels
            planes: Number of planes used in the middle part
            repetitions: Number of repetitions of relu > sepconv2dsame > bn
            stride: Stride
            dilation: Dilation
            start_with_relu: Flag whether to start with a relu layer
            grow_first: Flag whether to grow first
            is_last: Flag if this block is the last one
        """
        super(XceptionBlock, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.skipbn = nn.BatchNorm2d(num_features=planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=1,
                    rate=dilation,
                )
            )
            rep.append(nn.BatchNorm2d(num_features=planes))
            filters = planes

        for i in range(repetitions - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=3,
                    stride=1,
                    rate=dilation,
                )
            )
            rep.append(nn.BatchNorm2d(num_features=filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=1,
                    rate=dilation,
                )
            )
            rep.append(nn.BatchNorm2d(num_features=planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(
                SeparableConv2dSame(
                    in_channels=planes, out_channels=planes, kernel_size=3, stride=2
                )
            )

        if stride == 1 and is_last:
            rep.append(
                SeparableConv2dSame(
                    in_channels=planes, out_channels=planes, kernel_size=3, stride=1
                )
            )

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class SeparableConv2dSame(nn.Module):
    """
    Separable 2D convolution with fixed padding.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, rate=1, bias=False
    ):
        """
        Initialize module.
        Args:
            in_channels: Number of incoming channels
            out_channels: Number of outgoing channels
            kernel_size: Kernel size
            stride: Stride
            rate: Dilation
            bias: Bias flag
        """
        super(SeparableConv2dSame, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=rate,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )

    def forward(self, x):
        x = fixed_padding(
            inputs=x, kernel_size=self.conv1.kernel_size[0], rate=self.conv1.dilation[0]
        )
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
