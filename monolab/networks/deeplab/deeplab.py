import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import logging

from monolab.networks.decoder import MonodepthDecoder, MonodepthDecoderSkipless
from monolab.networks.deeplab.aspp import ASPP
from monolab.networks.resnet import Resnet50

logger = logging.getLogger(__name__)


class DeepLab(nn.Module):
    def __init__(
        self,
        num_in_layers=3,
        backbone="resnet",
        output_stride=16,
        freeze_bn=False,
        encoder_dilations=[1, 1, 1, 1],
        aspp_dilations=None,
        skip_connections=True,
        use_global_average_pooling_aspp=True,
    ):
        """
        DeepLab Module.

        Args:
            num_in_layers: Number of input layers.
            backbone: backbone type.
            output_stride: Factor by which the input image gets reduced in the encoder.
            freeze_bn: Flag to freeze batchnorm layer.
            aspp_dilations: List of aspp rates. Each rate will setup its own aspp layer with the given rate as dilation.
            decoder_type: Type of the decoder, can be one of ["godard", "deeplab"].
            skip_connections: Flag to use skip connections from the encoder to the decoder or not.
            use_global_average_pooling_aspp: Flag to enable global avg pooling in ASPP module
        """
        super(DeepLab, self).__init__()

        self.dcnn_type = backbone

        # Define backbone DCNN in encoder
        if backbone == "resnet":
            self.backbone = Resnet50(
                output_stride=output_stride,
                num_in_layers=num_in_layers,
                dilations=encoder_dilations,
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} not found.")

        # ASPP module
        self.aspp = ASPP(
            inplanes=2048,
            dilations=aspp_dilations,
            BatchNorm=nn.BatchNorm2d,
            use_global_average_pooling=use_global_average_pooling_aspp,
        )

        # Decoder module
        if backbone == "resnet":
            # Output channel sizes of x4 x3 x2 x1 of resnet
            x_low_inplanes_list = [256 * 4, 128 * 4, 64 * 4, 64]
        else:
            raise NotImplementedError(f"Backbone {backbone} not found.")

        num_channels_out_list = [256, 128, 64, 32, 16, 8]

        # Use Decoder if skip connections are enabled, else use Skipless Decoder
        if skip_connections:
            decoder_module = MonodepthDecoder
        else:
            decoder_module = MonodepthDecoderSkipless

        self.decoder = decoder_module(
            output_stride=output_stride,
            num_in_planes=256,
            num_out_planes=num_channels_out_list,
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
        disp1, disp2, disp3, disp4 = self.decoder(x1, x1_pool, x2, x3, x4, x)

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
        encoder_dilations=[1, 1, 1, 1],
        aspp_dilations=[1, 2, 6, 12],
        skip_connections=True,
        use_global_average_pooling_aspp=False,
    )

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.eval()
    print(net)

    y = net.forward(x)

    for yi in y:
        print(yi.size())
