from typing import Union, List, Dict
import collections
import torch

from monolab.networks.resnet_md import MonoDepthResNet50, MonoDepthResNet18
from monolab.networks.vgg_md import VGGMonodepth
from monolab.networks.deeplab.deeplabv3plus import DeepLabv3Plus, DCNNType
from monolab.networks.test_model import TestModel


import os
import logging
import sys

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


def to_device(
    x: Union[torch.Tensor, List[torch.tensor], Dict[str, torch.Tensor]], device: str
):
    """ Move a tensor or a collection of tensors to a device

    Args:
        x: tensor, dict of tensors or list of tensors
        device: e.g. 'cuda' or 'cpu'

    Returns:
        same structure as input, but on device
    """
    if torch.is_tensor(x):
        return x.to(device=device)
    elif isinstance(x, str):
        return x
    elif isinstance(x, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in x.items()}
    elif isinstance(x, collections.Sequence):
        return [to_device(sample, device=device) for sample in x]
    else:
        raise TypeError("Input must contain tensor, dict or list, found %s" % type(x))


def get_model(model: str, n_input_channels=3) -> Backbone:
    """
    Get model via name
    Args:
        model: Model name
        n_input_channels: Number of input channels
    Returns:
        Instantiated model
    """
    if model == "deeplab":
        out_model = DeepLabv3Plus(
            DCNNType.XCEPTION, in_channels=n_input_channels, output_stride=16
        )
    elif model == "resnet50_md":
        out_model = MonoDepthResNet50(num_in_layers=n_input_channels)
    elif model == "resnet18_md":
        out_model = MonoDepthResNet18(num_in_layers=n_input_channels)
    elif model == "vgg_md":
        out_model = VGGMonodepth(num_in_layers=n_input_channels)
    elif model == "testnet":
        out_model = TestModel(n_in_layers=n_input_channels)
    # elif and so on and so on
    else:
        raise NotImplementedError("Unknown model type")
    return out_model


def setup_logging(filename: str = "monolab.log", level: str = "INFO"):
    """
        Setup global loggers
        Args:
            filename: Log file destination
            level: Log level
        """

    # Check if previous log exists since logging.FileHandler only appends
    if os.path.exists(filename):
        os.remove(filename)

    logging.basicConfig(
        level=logging.getLevelName(level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=filename),
        ],
    )


def notify_mail(address, subject, message, filename=None):
    """
    Sends an E-Mail to the a specified address with a chosen message and subject

     Args:
        address (string): email-address to be sent to
        subject (string): subject of the e-mail
        message (string): message of the e-mail
        filename (string): path to the file that is going to be send via mail

    Returns:
        None
    """
    email = "dlcv2k18monolab@gmail.com"
    password = "hkR-KFa-ymB-gn2"

    msg = MIMEMultipart()

    msg["From"] = email
    msg["To"] = address
    msg["Subject"] = subject

    body = message

    msg.attach(MIMEText(body, "plain"))

    if filename is not None:
        attachment = open(filename, "rb")

        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename= %s" % filename)

        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, address, text)
    server.quit()
