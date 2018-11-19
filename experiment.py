import os
import sys

import collections
import numpy as np
import time

import torch

from evaluator import Evaluator
from monolab.data_loader import prepare_dataloader
from monolab.loss import MonodepthLoss
from monolab.networks.backbone import Backbone
from monolab.networks.resnet_models import Resnet18_md, Resnet50_md
from monolab.networks.deeplab.deeplabv3plus import DeepLabv3Plus, DCNNType
import logging

from monolab.networks.test_model import TestModel

logger = logging.getLogger(__name__)


class Experiment:
    """ A class for training and testing a model that contains the actual network as self.model
    The arguments are defined in the arg_parse()-function above.

    Important:
        - args.data_dir is the root directory
        - args.filenames_file should be different depending on whether you want to train or test the model
        - args.val_filenames_file is only used during training
    """

    def __init__(self, args):
        self.args = args

        self.loss_names = dict(
            full="monodepth-loss",
            images="image-loss",
            disp_gp="disparity-gradient-loss",
            lr_consistency="lr-consistency-loss",
        )

        # Setup Evaluator
        self.eval = Evaluator(metric_names=list(self.loss_names.values()), args=args)

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, n_input_channels=args.input_channels)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == "train":
            self.loss_function = MonodepthLoss(
                n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.learning_rate
            )
            self.val_n_img, self.val_loader = prepare_dataloader(
                args.data_dir,
                args.val_filenames_file,
                args.mode,
                args.augment_parameters,
                False,
                args.batch_size,
                (args.input_height, args.input_width),
                args.num_workers,
            )
            logging.info("Using a validation set with {} images".format(self.val_n_img))
        else:
            self.load(args.model_path)
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_dir = args.output_dir
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(
            args.data_dir,
            args.filenames_file,
            args.mode,
            args.augment_parameters,
            args.do_augmentation,
            args.batch_size,
            (args.input_height, args.input_width),
            args.num_workers,
        )
        logging.info(
            "Using a {}ing data set with {} images".format(args.mode, self.n_img)
        )

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def train(self):
        """ Train the model for self.args.epochs epochs

        Returns:
            None
        """
        # Store the best validation loss
        best_val_loss = float("Inf")

        # Start training
        for epoch in range(1, self.args.epochs + 1):
            # Adjust learning rate if flag is set
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch, self.args.learning_rate)

            c_time = time.time()

            # Init training loss
            running_loss = 0.0
            running_image_loss = 0.0
            running_disp_gradient_loss = 0.0
            running_lr_loss = 0.0

            self.model.train()
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data["left_image"]
                right = data["right_image"]

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss, image_loss, disp_gradient_loss, lr_loss = self.loss_function(
                    disps, [left, right]
                )
                loss.backward()
                self.optimizer.step()

                # Collect training loss
                running_loss += loss.item()
                running_image_loss += image_loss.item()
                running_disp_gradient_loss += disp_gradient_loss.item()
                running_lr_loss += lr_loss.item()

            # Init validation loss
            running_val_loss = 0.0
            running_val_image_loss = 0.0
            running_val_disp_gradient_loss = 0.0
            running_val_lr_loss = 0.0

            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data["left_image"]
                right = data["right_image"]
                disps = self.model(left)
                loss, image_loss, disp_gradient_loss, lr_loss = self.loss_function(
                    disps, [left, right]
                )

                # Collect validation loss
                running_val_loss += loss.item()
                running_val_image_loss += image_loss.item()
                running_val_disp_gradient_loss += disp_gradient_loss.item()
                running_val_lr_loss += lr_loss.item()

            # Estimate loss per image
            running_loss /= self.n_img
            running_image_loss /= self.n_img
            running_disp_gradient_loss /= self.n_img
            running_lr_loss /= self.n_img
            running_val_loss /= self.val_n_img
            running_image_loss /= self.val_n_img
            running_disp_gradient_loss /= self.val_n_img
            running_lr_loss /= self.val_n_img

            # Update best loss
            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss

            logger.info(
                "Epoch [{}/{}] time: {} s".format(
                    epoch, self.args.epochs, round(time.time() - c_time, 3)
                )
            )
            self.eval.add_epoch_metric(
                epoch=epoch,
                train_metric=running_loss,
                val_metric=running_val_loss,
                metric_name=self.loss_names["full"],
            )
            self.eval.add_epoch_metric(
                epoch=epoch,
                train_metric=running_image_loss,
                val_metric=running_val_image_loss,
                metric_name=self.loss_names["images"],
            )
            self.eval.add_epoch_metric(
                epoch=epoch,
                train_metric=running_disp_gradient_loss,
                val_metric=running_val_disp_gradient_loss,
                metric_name=self.loss_names["disp_gp"],
            )
            self.eval.add_epoch_metric(
                epoch=epoch,
                train_metric=running_lr_loss,
                val_metric=running_val_lr_loss,
                metric_name=self.loss_names["lr_consistency"],
            )

            self.eval.add_checkpoint(model=self.model, val_loss=running_val_loss)

        logging.info("Finished Training. Best loss: {}".format(best_val_loss))
        self.eval.save()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def test(self):
        """ Test the model.

        Returns:
            None
        """
        self.model.eval()
        disparities = np.zeros(
            (self.n_img, self.input_height, self.input_width), dtype=np.float32
        )
        disparities_pp = np.zeros(
            (self.n_img, self.input_height, self.input_width), dtype=np.float32
        )
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = post_process_disparity(
                    disps[0][:, 0, :, :].cpu().numpy()
                )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        np.save(os.path.join(self.output_dir, "disparities.npy"), disparities)
        np.save(os.path.join(self.output_dir, "disparities_pp.npy"), disparities_pp)

        logging.info("Finished Testing")


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
        raise TypeError(
            "Input must contain tensor, dict or list, found %s" % type(input)
        )


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """ Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epochs

    Args:
        optimizer: torch.optim type optimizer
        epoch: current epoch
        learning_rate: current learning rate

    """

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def post_process_disparity(disp):
    """ Apply the post-processing step described in the paper

    Args:
        disp: [2, h, w] np.array a disparity map

    Returns:
        post-processed disparity map
    """
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


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
    elif model == "resnet18_md":
        out_model = Resnet18_md(num_in_layers=n_input_channels)
    elif model == "resnet50_md":
        out_model = Resnet50_md(num_in_layers=n_input_channels)
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


if __name__ == "__main__":
    setup_logging("monolab.log", "info")
