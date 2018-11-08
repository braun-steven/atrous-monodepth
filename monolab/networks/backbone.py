import torch
import torch.nn as nn
from typing import List


class Backbone(nn.Module):
    """
    Backbone module that will be used in the monodepth architecture to construct
    disparity maps on different scales.
    """

    def __init__(self, n_input_channels: int):
        super(Backbone, self).__init__()
        self._n_input_channels = n_input_channels

    def forward(self, x) -> List[torch.Tensor]:
        """
        Forward input through backbone and generate disparity maps on different
        scales.
        :param x: Left input image
        :return: List of tensors of size [batch_size, 2, n_pixels_x, n_pixels_y]
        """
        pass
