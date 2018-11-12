import torch
from torch import nn

import torch.nn.functional as F


def init_weights(model: nn.Module):
    """
    Initialize all weights.
    Conv2d: He init
    BatchNorm2d: Weights: He init, Bias: zeros
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def get_1x_lr_params(model: nn.Module):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model: nn.Module):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [
        model.aspp1,
        model.aspp2,
        model.aspp3,
        model.aspp4,
        model.conv1,
        model.conv2,
        model.last_conv,
    ]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs
