import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monolab.networks.utils import DisparityOut


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(
            num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride
        )
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv1(x)


class MonoDepthDecoder(nn.Module):
    def __init__(self):
        super(MonoDepthDecoder, self).__init__()

        # decoder
        self.upconv6 = UpConv(2048, 512, 3, 2)
        self.iconv6 = Conv(1024 + 512, 512, 3, 1)

        self.upconv5 = UpConv(512, 256, 3, 2)
        self.iconv5 = Conv(512 + 256, 256, 3, 1)

        self.upconv4 = UpConv(256, 128, 3, 2)
        self.iconv4 = Conv(256 + 128, 128, 3, 1)
        self.disp4_layer = DisparityOut(128)

        self.upconv3 = UpConv(128, 64, 3, 2)
        self.iconv3 = Conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = DisparityOut(64)

        self.upconv2 = UpConv(64, 32, 3, 2)
        self.iconv2 = Conv(32 + 64 + 2, 32, 3, 1)
        self.disp2_layer = DisparityOut(32)

        self.upconv1 = UpConv(32, 16, 3, 2)
        self.iconv1 = Conv(16 + 2, 16, 3, 1)
        self.disp1_layer = DisparityOut(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x_pool1, x2, x3, x4, x_enc):

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x_enc)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(
            self.disp4, scale_factor=2, mode="nearest"
        )

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(
            self.disp3, scale_factor=2, mode="nearest"
        )

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(
            self.disp2, scale_factor=2, mode="nearest"
        )

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
