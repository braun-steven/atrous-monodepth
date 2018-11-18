import datetime
import torch

import logging
import os
from shutil import copyfile

from typing import List
from torch import nn
from torch import Tensor
import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np

from argparse import Namespace

from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class Evaluator:
    """
    The evaluator stores all results and data that can be collected during an
    experiment.
    Main result locations are:
    - tensorboard: Tensorboard files
    - plots: Metric plots
    - checkpoints: Model checkpoints
    - args.txt: File containing all commandline arguments with which the
    experiment has been started
    """

    def __init__(self, metric_names: List[str], args: Namespace):
        """
        Initialize the Evaluator object.
        Args:
            metric_names: Names of different metrics
            args: Command line arguments
        """

        # Generate base path: ".../$(args.output_dir)/run-$(date)-$(tag)"
        _date_str = datetime.datetime.today().strftime("%y-%b-%d_%Hh:%Mm")
        tagstr = args.tag if args.tag == "" else "_" + args.tag

        self._base_dir = os.path.join(
            args.output_dir, "run_{0}{1}".format(_date_str, tagstr)
        )

        self._metric_names = metric_names
        self._metric_epochs_train = {name: [] for name in metric_names}
        self._metrics_epochs_val = {name: [] for name in metric_names}

        # Store best loss for model checkpoints
        self._best_val_loss = float("inf")
        self._best_cpt_path = os.path.join(self._base_dir, "best-model.pth")
        self._last_cpt_path = os.path.join(self._base_dir, "last-model.pth")

        # File/Directory names
        self._args_path = os.path.join(self._base_dir, "args.txt")
        self._tensorboard_dir = os.path.join(self._base_dir, "tensorboard/")
        self._checkpoints_dir = os.path.join(self._base_dir, "checkpoints/")
        self._plots_dir = os.path.join(self._base_dir, "plots/")
        self._create_dirs()

        # Tensorboard
        self._summary_writer = SummaryWriter(log_dir=self._tensorboard_dir)

        self._args = args

        # Store maxs
        self._max_epochs = args.epochs

        # Log template
        max_metric_name_len = max(map(len, metric_names))
        self._log_template = (
            "{progress: <10}{name: <10} ({metric_name: <"
            + str(max_metric_name_len)
            + "}): Train = {train_metric:10f}, Validation = {val_metric:10f}"
        )

    def _create_dirs(self):
        """Create necessary directories"""
        for d in [self._tensorboard_dir, self._checkpoints_dir, self._plots_dir]:
            self._ensure_dir(d)

    def _plot_metric(self, metric_dict: dict, xlabel: str, title: str, suffix: str):
        """
        Plot a specific metric
        Args:
            metric_dict (dict): Metric dictionary
            xlabel (str): X-Axis label
            title (str): Plot title
            suffix (str): File suffix
        """
        plt.figure()
        for name, metrics in metric_dict.items():
            data = np.array(metrics)
            plt.plot(data[:, 0], data[:, 1], label=name)
        plt.xlabel(xlabel)
        plt.ylabel("metric")
        plt.legend(loc="upper right")
        plt.title(title)
        plt.savefig(
            os.path.join(
                self._plots_dir, "{}-metric-{}.png".format(xlabel.lower(), suffix)
            )
        )

    def _plot_metric_epochs(self):
        """Plot metrics"""
        self._plot_metric(
            self._metric_epochs_train,
            xlabel="Epoch",
            title="Epochs: Train metric",
            suffix="train",
        )
        self._plot_metric(
            self._metrics_epochs_val,
            xlabel="Epoch",
            title="Epochs: Validation metric",
            suffix="val",
        )

    def _save_args(self):
        """Save arguments"""
        if self._args is None:
            return

        # Get maximum argument name length for formatting
        args = sorted(vars(self._args).items())
        length = max(map(lambda k: len(k[0]), args))

        # Save commandline args in a file
        line_template = "{0: <{2:}} = {1:}"
        with open(self._args_path, "w") as f:
            lines = [line_template.format(x, y, length) for x, y in args]
            header = "Command line arguments: \n"
            content = header + "\n".join(lines)
            f.write(content)

    def add_epoch_metric(
        self, epoch: int, train_metric: float, val_metric, metric_name: str
    ) -> None:
        """
        Add a specific metric for a single epoch.
        Args:
            epoch (int): Epoch index
            train_metric (float): Train metric value
            val_metric (float): Validation metric value
            metric_name (str): metric name
        """
        # Store metric for plots
        self._metric_epochs_train[metric_name].append([epoch, train_metric])
        self._metrics_epochs_val[metric_name].append([epoch, val_metric])

        # Tensorboard
        self._summary_writer.add_scalars(
            main_tag="epoch/" + metric_name,
            tag_scalar_dict={"train": train_metric, "val": val_metric},
            global_step=epoch,
        )

        # Log
        logging.info(
            self._log_template.format(
                name="Epoch",
                metric_name=metric_name,
                train_metric=train_metric,
                val_metric=val_metric,
                progress="[{}/{}]".format(epoch, self._max_epochs),
            )
        )

    def add_image(self, epoch: int, img: Tensor, tag: str):
        """
        Add an image to the evaluation results
        Args:
            epoch (int): Current epoch
            img (Tensor): Image
            tag (str): Tag as short description/identifier of the image
        """
        if isinstance(img, np.ndarray):
            img = Tensor(img)

        self._summary_writer.add_image(
            tag="image/" + tag, img_tensor=img, global_step=epoch
        )

    def add_checkpoint(self, model: nn.Module, val_loss: float) -> None:
        """
        Add a new checkpoint. Store latest model weights in checkpoints/last-model.pth
        and best model based on the current validation metric in
        checkpoints/best-model.pth.
        Args:
            model (nn.Module): PyTorch model
            val_loss (float): Latest validation loss
        """
        torch.save(model.state_dict(), f=self._last_cpt_path)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            torch.save(model.state_dict(), f=self._best_cpt_path)

    def save(self):
        """
        Save some results:
        - Log file
        - Arguments
        - Scalar values as JSON
        - Plots
        """
        # Copy log to the output-dir
        log_path = os.path.join(self._base_dir, "log.txt")
        copyfile(self._args.log_file, log_path)

        # Save arguments with which the current experiment has been started
        self._save_args()

        # Save all scalars to a json for future processing
        self._summary_writer.export_scalars_to_json(
            os.path.join(self._base_dir, "metric-results.json")
        )

        # Save plots
        self._plot_metric_epochs()

    def _ensure_dir(self, file: str) -> None:
        """
        Ensures that a given directory exists.

        Args:
            file: file
        """
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
