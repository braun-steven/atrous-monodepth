from torch import nn

import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self, n_in_layers):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_in_layers, out_channels=32, kernel_size=3, padding=1
        )
        self.disp1 = get_disp(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1
        )
        self.disp2 = get_disp(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.disp3 = get_disp(8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.disp4 = get_disp(4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        x = self.conv1(input)
        disp1 = self.disp1(x)
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.conv2(x)
        disp2 = self.disp2(x)
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.conv3(x)
        disp3 = self.disp3(x)
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.conv4(x)
        disp4 = self.disp4(x)

        return disp1, disp2, disp3, disp4


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)
