import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from monolab.networks.utils import fixed_padding


class Xception(nn.Module):
    """
    Modified Aligned Xception
    """

    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
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

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(
            128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True
        )
        self.block3 = Block(
            256,
            728,
            reps=2,
            stride=entry_block3_stride,
            start_with_relu=True,
            grow_first=True,
            is_last=True,
        )

        # Middle flow
        mblock = lambda: Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middle_block_rate,
            start_with_relu=True,
            grow_first=True,
        )
        self.block4 = mblock()
        self.block5 = mblock()
        self.block6 = mblock()
        self.block7 = mblock()
        self.block8 = mblock()
        self.block9 = mblock()
        self.block10 = mblock()
        self.block11 = mblock()
        self.block12 = mblock()
        self.block13 = mblock()
        self.block14 = mblock()
        self.block15 = mblock()
        self.block16 = mblock()
        self.block17 = mblock()
        self.block18 = mblock()
        self.block19 = mblock()

        # Exit flow
        self.block20 = Block(
            728,
            1024,
            reps=2,
            stride=1,
            dilation=exit_block_rates[0],
            start_with_relu=True,
            grow_first=False,
            is_last=True,
        )

        self.conv3 = SeparableConv2dSame(
            1024, 1536, 3, stride=1, dilation=exit_block_rates[1]
        )
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2dSame(
            1536, 1536, 3, stride=1, dilation=exit_block_rates[1]
        )
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2dSame(
            1536, 2048, 3, stride=1, dilation=exit_block_rates[1]
        )
        self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        self.__init_weight()

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

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
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


class Block(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        reps,
        stride=1,
        dilation=1,
        start_with_relu=True,
        grow_first=True,
        is_last=False,
    ):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(inplanes, planes, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(filters, filters, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2dSame(inplanes, planes, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2dSame(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2dSame(planes, planes, 3, stride=1))

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


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=inplanes,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SeparableConv2dSame(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False
    ):
        super(SeparableConv2dSame, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            0,
            dilation,
            groups=inplanes,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
