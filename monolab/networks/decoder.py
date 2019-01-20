import torch
import torch.nn as nn

from monolab.networks.resnet import upconv, conv, get_disp, upsample_nn


class MonodepthDecoder(nn.Module):
    def __init__(self):
        super(MonodepthDecoder, self).__init__()

        # decoder
        self.upconv6 = upconv(n_in=2048, n_out=512, kernel_size=3, scale=2)  # H/32
        self.iconv6 = conv(
            n_in=512 + 1024, n_out=512, kernel_size=3, stride=1
        )  # upconv6 + conv4

        self.upconv5 = upconv(n_in=512, n_out=256, kernel_size=3, scale=2)  # H/16
        self.iconv5 = conv(
            n_in=256 + 512, n_out=256, kernel_size=3, stride=1
        )  # upconv5 + conv3

        self.upconv4 = upconv(n_in=256, n_out=128, kernel_size=3, scale=2)  # H/8
        self.iconv4 = conv(
            n_in=256 + 128, n_out=128, kernel_size=3, stride=1
        )  # upconv4 + conv2
        self.disp4 = get_disp(128)

        self.upconv3 = upconv(n_in=128, n_out=64, kernel_size=3, scale=2)  # H/4
        self.iconv3 = conv(
            n_in=64 + 64 + 2, n_out=64, kernel_size=3, stride=1
        )  # upconv3 + pool1 + disp4
        self.disp3 = get_disp(64)

        self.upconv2 = upconv(n_in=64, n_out=32, kernel_size=3, scale=2)  # H/2
        self.iconv2 = conv(
            n_in=32 + 64 + 2, n_out=32, kernel_size=3, stride=1
        )  # upconv2 + conv1 + disp3
        self.disp2 = get_disp(32)

        self.upconv1 = upconv(n_in=32, n_out=16, kernel_size=3, scale=2)  # H
        self.iconv1 = conv(
            n_in=16 + 2, n_out=16, kernel_size=3, stride=1
        )  # upconv1 + disp2
        self.disp1 = get_disp(16)

        self.upsample_nn = upsample_nn(scale=2)

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



class MonodepthDecoderSkipless(nn.Module):
    def __init__(self):
        super(MonodepthDecoderSkipless, self).__init__()

        # decoder
        self.upconv6 = upconv(n_in=2048, n_out=512, kernel_size=3, scale=2)  # H/32
        self.iconv6 = conv(
            n_in=512, n_out=512, kernel_size=3, stride=1
        )  # upconv6 + conv4

        self.upconv5 = upconv(n_in=512, n_out=256, kernel_size=3, scale=2)  # H/16
        self.iconv5 = conv(
            n_in=256, n_out=256, kernel_size=3, stride=1
        )  # upconv5 + conv3

        self.upconv4 = upconv(n_in=256, n_out=128, kernel_size=3, scale=2)  # H/8
        self.iconv4 = conv(
            n_in=128, n_out=128, kernel_size=3, stride=1
        )  # upconv4 + conv2
        self.disp4 = get_disp(128)

        self.upconv3 = upconv(n_in=128, n_out=64, kernel_size=3, scale=2)  # H/4
        self.iconv3 = conv(
            n_in=64 + 2, n_out=64, kernel_size=3, stride=1
        )  # upconv3 + pool1 + disp4
        self.disp3 = get_disp(64)

        self.upconv2 = upconv(n_in=64, n_out=32, kernel_size=3, scale=2)  # H/2
        self.iconv2 = conv(
            n_in=32 + 2, n_out=32, kernel_size=3, stride=1
        )  # upconv2 + conv1 + disp3
        self.disp2 = get_disp(32)

        self.upconv1 = upconv(n_in=32, n_out=16, kernel_size=3, scale=2)  # H
        self.iconv1 = conv(
            n_in=16 + 2, n_out=16, kernel_size=3, stride=1
        )  # upconv1 + disp2
        self.disp1 = get_disp(16)

        self.upsample_nn = upsample_nn(scale=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x_pool1, x2, x3, x4, x_enc):
        # decoder
        upconv6 = self.upconv6(x_enc)
        concat6 = torch.cat([upconv6], 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat([upconv5], 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat([upconv4], 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4(iconv4)
        udisp4 = self.upsample_nn(disp4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat([upconv3, udisp4], 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)
        udisp3 = self.upsample_nn(disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat([upconv2, udisp3], 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)
        udisp2 = self.upsample_nn(disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat([upconv1, udisp2], 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)

        return [disp1, disp2, disp3, disp4]
