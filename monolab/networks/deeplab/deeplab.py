import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from monolab.networks.deeplab.deeplab_decoder import Decoder
from monolab.networks.deeplab.aspp import ASPP
from monolab.networks.resnet import Resnet50


class DeepLab(nn.Module):
    def __init__(
        self,
        num_in_layers=3,
        backbone="resnet",
        output_stride=16,
        freeze_bn=False,
        aspp_dilations=None,
        decoder_type="deeplab",
    ):
        super(DeepLab, self).__init__()

        self.dcnn_type = backbone

        # Define backbone DCNN in encoder
        if backbone == "resnet":
            self.backbone = Resnet50(
                output_stride=output_stride, num_in_layers=num_in_layers
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} not found.")

        # ASPP module
        self.aspp = ASPP(
            backbone=backbone, dilations=aspp_dilations, BatchNorm=nn.BatchNorm2d
        )

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
            decoder_type=decoder_type,
        )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x) -> List[Tensor]:
        if self.dcnn_type == "resnet":
            x1, x1_pool, x2, x3, x4, x_dcnn_out = self.backbone(x)
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
    x = torch.rand(1, 3, 256, 512)

    net = DeepLab(
        num_in_layers=3,
        output_stride=16,
        backbone="resnet",
        aspp_dilations=[1, 2, 4, 8, 12, 18, 32],
        decoder_type="godard"
    )
    print(net)

    net.eval()

    y = net.forward(x)

    for yi in y:
        print(yi.size())
