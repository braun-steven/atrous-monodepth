import collections
import torch

from monolab.networks.backbone import Backbone
from monolab.networks.deeplab.deeplabv3plus import DCNNType, DeepLabv3Plus


def get_model(model: str, input_channels=3):
    """
    Get model via name
    Args:
        model: Model name
        input_channels: Number of input channels
    Returns:
        Instantiated model
    """
    if model == "backbone":
        out_model = Backbone(n_input_channels=input_channels)
    elif model == "deeplab":
        out_model = DeepLabv3Plus(
            DCNNType.XCEPTION, n_input_channels=3, output_stride=16
        )
    # elif and so on and so on
    else:
        raise ValueError("Please specify a valid model")
    return out_model


def to_device(input, device):
    """ Move a tensor or a collection of tensors to a device

    Args:
        input: tensor, dict of tensors or list of tensors
        device: e.g. 'cuda' or 'cpu'

    Returns:
        same structure as input, but on device
    """
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")
