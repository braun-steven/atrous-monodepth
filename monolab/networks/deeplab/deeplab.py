import torch
import torch.nn as nn
import torch.nn.functional as F

from monolab.networks.backbones.resnet import ResNet50
from monolab.networks.backbones.xception import Xception
from monolab.networks.deeplab.deeplab_decoder import Decoder
from monolab.networks.deeplab.aspp import ASPP

from monolab.networks.utils import get_disp


class DeepLab(nn.Module):
    def __init__(
        self,
        num_in_layers=3,
        backbone="resnet",
        output_stride=16,
        freeze_bn=False,
        pretrained=False,
    ):
        super(DeepLab, self).__init__()

        self.dcnn_type = backbone

        if backbone == "resnet":
            self.backbone = ResNet50(
                output_stride=output_stride,
                num_in_layers=num_in_layers,
                pretrained=pretrained,
            )
        elif backbone == "xception":
            self.backbone = Xception(
                inplanes=num_in_layers,
                output_stride=output_stride,
                pretrained=pretrained,
            )
        else:
            raise NotImplementedError("Currently only resnet backbone is supported")

        self.aspp = ASPP(backbone, output_stride, BatchNorm=nn.BatchNorm2d)
        self.decoder = Decoder(
            num_classes=16, backbone=backbone, BatchNorm=nn.BatchNorm2d
        )

        self.disp_layer = get_disp(16)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        if self.dcnn_type == "resnet":
            _1, _pool1, low_level_feat, _3, _4, x = self.backbone(input)
        elif self.dcnn_type == "xception":
            x, low_level_feat = self.backbone(input)

        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        x = self.disp_layer(x)

        return [x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 512)

    net = DeepLab(num_in_layers=3, output_stride=16, backbone="resnet")

    net.eval()

    y = net.forward(x)

    for yi in y:
        print(yi.size())
