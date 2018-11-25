#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from monolab.networks.decoder import MonoDepthDecoder


class VGGMonodepth(nn.Module):  # vgg version
    def __init__(self, num_in_layers=3):
        super(VGGMonodepth, self).__init__()

        self.output_nc = 2

        self.downconv_1 = self.conv_down_block(
            num_in_layers, 32, 7
        )
        self.downconv_2 = self.conv_down_block(32, 64, 5)
        self.downconv_3 = self.conv_down_block(64, 128, 3)
        self.downconv_4 = self.conv_down_block(128, 256, 3)
        self.downconv_5 = self.conv_down_block(256, 512, 3)
        self.downconv_6 = self.conv_down_block(512, 512, 3)
        self.downconv_7 = self.conv_down_block(512, 512, 3)

        self.decoder = MonoDepthDecoder()

    def conv_down_block(self, in_dim, out_dim, kernal):

        conv_down_block = []
        conv_down_block += [
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernal,
                stride=1,
                padding=int((kernal - 1) / 2),
            ),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]  # h,w -> h,w
        conv_down_block += [
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernal,
                stride=2,
                padding=int((kernal - 1) / 2),
            ),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]  # h,w -> h/2,w/2

        return nn.Sequential(*conv_down_block)

    def conv_up_block(self, in_dim, out_dim):

        conv_up_block = []
        conv_up_block += [
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]  # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def conv_block(self, in_dim, out_dim):

        conv_up_block = []
        conv_up_block += [
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]  # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def disp_block(self, in_dim):

        disp_block = []
        disp_block += [
            nn.Conv2d(
                in_dim,
                self.output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        ]  # h,w -> h,w

        return nn.Sequential(*disp_block)

    def upsample_(self, disp, ratio):

        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(
            disp, [nh, nw], mode="nearest"
        )

        return temp

    def forward(self, x):

        # Encoder
        # 3x256x512
        conv_1 = self.downconv_1(x)  # 32x128x256
        conv_2 = self.downconv_2(conv_1)  # 64x64x128 #x1
        conv_3 = self.downconv_3(conv_2)  # 128x32x64 #x2
        conv_4 = self.downconv_4(conv_3)  # 256x16x32 #x3
        conv_5 = self.downconv_5(conv_4)  # 512x8x16 #x4
        conv_6 = self.downconv_6(conv_5)  # 512x4x8 #x_enc

        # Decoder
        self.disp1, self.disp2, self.disp3, self.disp4 = self.decoder.forward(
            conv_1, conv_2, conv_3, conv_4, conv_5, conv_6
        )

        return [
            self.disp1,
            self.disp2,
            self.disp3,
            self.disp4,
        ]
