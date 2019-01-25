import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


class maxpool(nn.Module):
    """ Max pooling layer with padding (dependent on kernel size) and stride 2
    """

    def __init__(self, kernel_size):
        """
        Args:
            kernel_size: kernel size (determines padding)
        """
        super(maxpool, self).__init__()
        p = int(np.floor(kernel_size - 1) / 2)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=2, padding=(p, p))

    def forward(self, x):
        return self.maxpool(x)


class upsample_nn(nn.Module):
    """ Upsampling layer, nearest neighbor
    """

    def __init__(self, scale):
        """
        Args:
            scale: upsampling factor
        """
        super(upsample_nn, self).__init__()
        self.scale = scale

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class get_disp(nn.Module):
    """ Convolution followed by a sigmoid layer, multiplied with 0.3.
        Has two output channels (left and right disparity map)
    """

    def __init__(self, num_in_layers):
        """
        Args:
            num_in_layers: number of input channels
        """
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        return 0.3 * self.sigmoid(x)


class conv(nn.Module):
    """ Convolutional layer with padding
    """

    def __init__(self, n_in, n_out, kernel_size, stride, dilation=1):
        """
        Args:
            n_in: number of input channels
            n_out: number of output channels
            kernel_size: kernel size
            stride: stride of the convolution
            dilation: dilation (atrous rate) of the convolution
        """
        super(conv, self).__init__()
        self.p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        if dilation != 1:
            self.p = dilation
        self.conv = nn.Conv2d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.p, self.p),
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv(x)
        return F.elu(x)


class upconv(nn.Module):
    """ Upsampling and convolution
    """

    def __init__(self, n_in, n_out, kernel_size, scale):
        """
        Args:
            n_in: number of input channels
            n_out: number of output_channels
            kernel_size: kernel size
            scale: scale of the upsampling
        """
        super(upconv, self).__init__()
        self.upsample = upsample_nn(scale)
        self.conv = conv(n_in=n_in, n_out=n_out, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        conv = self.conv(x)
        return conv


class resconv(nn.Module):
    """ A resnet building block, consisting of 3 convolutional layers
    """

    def __init__(self, n_in, num_layers, stride, dilation=1):
        """
        Args:
            n_in: number of input channels
            num_layers: number of intermediate channels, output has 4 * num_layers
            stride: stride of the second conv layer
            dilation: dilation (atrous rate) of the second conv layer
        """
        super(resconv, self).__init__()
        self.num_layers = num_layers
        self.stride = stride

        self.conv1 = conv(n_in=n_in, n_out=num_layers, kernel_size=1, stride=1)
        self.conv2 = conv(
            n_in=num_layers,
            n_out=num_layers,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
        )
        self.conv3 = nn.Conv2d(num_layers, 4 * num_layers, kernel_size=1, stride=1)

        self.shortcut_conv = nn.Conv2d(
            n_in, 4 * num_layers, kernel_size=1, stride=stride
        )

    def forward(self, x):
        do_proj = x.shape[1] != 4 * self.num_layers or self.stride == 2
        shortcut = []

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if do_proj:
            shortcut = self.shortcut_conv(x)
        else:
            shortcut = x

        return F.elu(conv3 + shortcut)


class resblock(nn.Module):
    """ A Resnet block, that consists of num_blocks resconv building blocks.
    """

    def __init__(self, n_in, num_layers, num_blocks, stride=2, dilation=1):
        """
        Args:
            n_in: number of input channels
            num_layers: number of intermediate channels (for resconv units), output is 4 * num_layers
            num_blocks: number of resconv units
            stride: stride of the final resconv unit (all others are 1)
            dilation: dilation (atrous rate) of the final resconv unit (all others are 1 by default)
        """
        super(resblock, self).__init__()
        layers = []

        layers.append(resconv(n_in=n_in, num_layers=num_layers, stride=1))

        for i in range(1, num_blocks - 1):
            layers.append(resconv(n_in=4 * num_layers, num_layers=num_layers, stride=1))

        layers.append(
            resconv(
                n_in=4 * num_layers,
                num_layers=num_layers,
                stride=stride,
                dilation=dilation,
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Resnet(nn.Module):
    """ Resnet implementation with the quirk's of Godard et al. (2017), https://github.com/mrharicot/monodepth
        We made the resnet size variable by letting the user give a list of numbers of blocks for the three resblocks
    """

    def __init__(self, num_in_layers, blocks, output_stride=64, dilations=[1, 1, 1, 1]):
        """
        Args:
            num_in_layers: number of input channels (3 for rgb)
            blocks: list of length 4, contains the numbers of blocks -> [3, 4, 6, 3] for Resnet50
        """
        super(Resnet, self).__init__()

        if output_stride == 64:
            strides = [2, 2, 2, 2]
        elif output_stride == 32:
            strides = [1, 2, 2, 2]
        # output_stride = 16 is the standard for deeplabv3+
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
        else:
            raise ValueError("Please specify a valid output stride")

        # encoder
        self.conv1 = conv(
            n_in=num_in_layers, n_out=64, kernel_size=7, stride=2
        )  # H/2 - 64D
        self.pool1 = maxpool(kernel_size=3)  # H/4 - 64D
        self.conv2 = resblock(
            n_in=64,
            num_layers=64,
            num_blocks=blocks[0],
            stride=strides[0],
            dilation=dilations[0],
        )  # H/8 - 256D
        self.conv3 = resblock(
            n_in=256,
            num_layers=128,
            num_blocks=blocks[1],
            stride=strides[1],
            dilation=dilations[1],
        )  # H/16 -  512D
        self.conv4 = resblock(
            n_in=512,
            num_layers=256,
            num_blocks=blocks[2],
            stride=strides[2],
            dilation=dilations[2],
        )  # H/32 - 1024D
        self.conv5 = resblock(
            n_in=1024,
            num_layers=512,
            num_blocks=blocks[3],
            stride=strides[3],
            dilation=dilations[3],
        )  # H/64 - 2048D

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return x1, x_pool1, x2, x3, x4, x5


def Resnet50(num_in_layers, output_stride=64, dilations=[1, 1, 1, 1]):
    return Resnet(
        num_in_layers=num_in_layers,
        blocks=[3, 4, 6, 3],
        output_stride=output_stride,
        dilations=dilations,
    )


def Resnet18(num_in_layers, output_stride=64, dilations=[1, 1, 1, 1]):
    return Resnet(
        num_in_layers=num_in_layers,
        blocks=[2, 2, 2, 2],
        output_stride=output_stride,
        dilations=dilations,
    )


def Resnet101(num_in_layers, output_stride=64, dilations=[1, 1, 1, 1]):
    return Resnet(
        num_in_layers=num_in_layers,
        blocks=[3, 4, 23, 3],
        output_stride=output_stride,
        dilations=dilations,
    )


if __name__ == "__main__":

    img = np.random.randn(1, 3, 256, 512)
    img = torch.Tensor(img)

    model = Resnet50(3, output_stride=16)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.forward(img)
