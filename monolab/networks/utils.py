import torch
import torch.nn as nn
import torch.nn.functional as F


class DisparityOut(nn.Module):
    def __init__(self, num_in_layers):
        """
        Disparity output layer.
        Args:
            num_in_layers: Number of incoming layers
        """
        super(DisparityOut, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_in_layers, out_channels=2, kernel_size=3, stride=1
        )
        self.normalize = nn.BatchNorm2d(num_features=2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = F.pad(x, p2d)
        x = self.conv1(x)
        x = self.normalize(x)
        # Use scaled sigmoid non-linearity where d_max = 0.3x image width
        return 0.3 * self.sigmoid(x)
