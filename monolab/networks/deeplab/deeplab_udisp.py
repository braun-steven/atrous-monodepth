import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from monolab.networks.backbones.resnet import ResNet50, Bottleneck
from monolab.networks.backbones.xception import Xception
from monolab.networks.deeplab.deeplab_udisp_decoder import Decoder
from monolab.networks.deeplab.aspp import ASPP


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

        # Define backbone DCNN in encoder
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
            raise NotImplementedError(f"Backbone {backbone} not found.")

        # ASPP module
        self.aspp = ASPP(backbone, output_stride, BatchNorm=nn.BatchNorm2d)

        # Decoder module
        if backbone == "resnet":
            # Output channel sizes of x4 x3 x2 x1 of resnet
            x_low_inplanes_list = [256 * 4, 128 * 4, 64 * 4, 64]
        else:
            raise NotImplementedError(f"Backbone {backbone} not found.")

        num_channels_out_list = [128, 64, 32, 16]
        self.decoder = Decoder(
            backbone=backbone,
            BatchNorm=nn.BatchNorm2d,
            x_low_inplanes_list=x_low_inplanes_list,
            num_outplanes_list=num_channels_out_list,
        )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x) -> List[Tensor]:
        if self.dcnn_type == "resnet":
            x1, x1_pool, x2, x3, x4, x_dcnn_out = self.backbone(x)
        elif self.dcnn_type == "xception":
            x_dcnn_out, x2 = self.backbone(input)
            raise NotImplementedError(
                "Multiple disparities for xception backend not " "yet implemented"
            )
        else:
            raise NotImplementedError(
                "Unknown DCNN backend type for the Deeplab backbone."
            )

        # Apply ASPP module
        x = self.aspp(x_dcnn_out)

        # List of decoder results of size num_disparity_maps
        disp1, disp2, disp3, disp4 = self.decoder(x, x1, x2, x3, x4)

        return [disp1, disp2, disp3, disp4]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    x = torch.rand(8, 3, 256, 512)

    net = DeepLab(num_in_layers=3, output_stride=16, backbone="resnet")
    # print(net)

    net.eval()

    y = net.forward(x)

    for yi in y:
        print(yi.size())
