import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        p = int(np.floor(kernel_size - 1) / 2)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=2, padding=(p, p))

    def forward(self, x):
        return self.maxpool(x)


class upsample_nn(nn.Module):
    def __init__(self, scale):
        super(upsample_nn, self).__init__()
        self.scale = scale

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        return 0.3 * self.sigmoid(x)


class conv(nn.Module):
    def __init__(self, n_in, n_out, k, s):
        super(conv, self).__init__()
        self.p = np.floor((k - 1) / 2).astype(np.int32)
        self.conv = nn.Conv2d(
            n_in, n_out, kernel_size=k, stride=s, padding=(self.p, self.p)
        )

    def forward(self, x):
        x = self.conv(x)
        return F.elu(x)


class upconv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, scale):
        super(upconv, self).__init__()
        self.upsample = upsample_nn(scale)
        self.conv = conv(n_in, n_out, kernel_size, 1)

    def forward(self, x):
        x = self.upsample(x)
        conv = self.conv(x)
        return conv


class resconv(nn.Module):
    def __init__(self, n_in, num_layers, stride):
        super(resconv, self).__init__()
        self.num_layers = num_layers
        self.stride = stride

        self.conv1 = conv(n_in, num_layers, 1, 1)
        self.conv2 = conv(num_layers, num_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_layers, 4 * num_layers, kernel_size=1, stride=1)

        self.shortcut_conv = nn.Conv2d(
            n_in, 4 * num_layers, kernel_size=1, stride=stride
        )

    def forward(self, x):
        do_proj = x.shape[3] != self.num_layers or self.stride == 2
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
    def __init__(self, n_in, num_layers, num_blocks):
        super(resblock, self).__init__()
        layers = []

        layers.append(resconv(n_in, num_layers, 1))

        for i in range(1, num_blocks - 1):
            layers.append(resconv(4 * num_layers, num_layers, 1))

        layers.append(resconv(4 * num_layers, num_layers, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Resnet50(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet50, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2 - 64D
        self.pool1 = maxpool(3)  # H/4 - 64D
        self.conv2 = resblock(64, 64, 3)  # H/8 - 256D
        self.conv3 = resblock(256, 128, 4)  # H/16 -  512D
        self.conv4 = resblock(512, 256, 6)  # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3)  # H/64 - 2048D

        # decoder
        self.upconv6 = upconv(2048, 512, 3, 2)  # H/32
        self.iconv6 = conv(512 + 1024, 512, 3, 1)  # upconv6 + conv4

        self.upconv5 = upconv(512, 256, 3, 2)  # H/16
        self.iconv5 = conv(256 + 512, 256, 3, 1)  # upconv5 + conv3

        self.upconv4 = upconv(256, 128, 3, 2)  # H/8
        self.iconv4 = conv(256 + 128, 128, 3, 1)  # upconv4 + conv2
        self.disp4 = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)  # H/4
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)  # upconv3 + pool1 + disp4
        self.disp3 = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)  # H/2
        self.iconv2 = conv(32 + 64 + 2, 32, 3, 1)  # upconv2 + conv1 + disp3
        self.disp2 = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)  # H
        self.iconv1 = conv(16 + 2, 16, 3, 1)  # upconv1 + disp2
        self.disp1 = get_disp(16)

        self.upsample_nn = upsample_nn(scale=2)

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

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat([upconv6, skip5], 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat([upconv5, skip4], 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat([upconv4, skip3], 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4(iconv4)
        udisp4 = self.upsample_nn(disp4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat([upconv3, skip2, udisp4], 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)
        udisp3 = self.upsample_nn(disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat([upconv2, skip1, udisp3], 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)
        udisp2 = self.upsample_nn(disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat([upconv1, udisp2], 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)

        return [disp1, disp2, disp3, disp4]


if __name__ == "__main__":

    img = np.random.randn(1, 3, 256, 512)
    img = torch.Tensor(img)

    model = Resnet50(3)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.forward(img)
