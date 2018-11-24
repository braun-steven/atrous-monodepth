import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        if backbone == "resnet":
            low_level_inplanes = 256
        elif backbone == "xception":
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        # code for intermediate disparity maps (from our previous dlv3+ implementation)
        # TODO: think about this and implement
        # adopt [1x1, 2] for channel reduction.
        # self.conv2 = nn.Conv2d(
        #    in_channels=low_level_inplanes, out_channels=48, kernel_size=1, bias=False
        # )
        # self.bn2 = nn.BatchNorm2d(num_features=48)

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True
        )

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        # code for intermediate disparity maps (from our previous dlv3+ implementation)
        # TODO: think about this and implement
        # low_level_features = self.conv2(low_level_features)
        # low_level_features = self.bn2(low_level_features)
        # low_level_features = self.relu(low_level_features)
        # low_level_features = self.low_level_features_reduction(low_level_features)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
