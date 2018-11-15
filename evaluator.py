import logging
import os

from typing import List

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiment import setup_logging
from argparse import Namespace

# SNS style setup
sns.set(palette="deep", color_codes=True)
sns.set_style("ticks")


logger = logging.getLogger(__name__)


class Evaluator:
    """
    The evaluator stores all results and data that can be collected during an
    experiment.
    Main result locations are:
    - tensorboard: Tensorboard files
    - plots:
    - checkpoints: Model checkpoints
    - args.txt: File containing all commandline arguments with which the
    experiment has been started
    """

    def __init__(self, base_dir: str, loss_names: List[str], args: Namespace):
        """
        Initialize the Evaluator object.
        Args:
            base_dir: Base directory in which all results will be stored
            loss_names: Naming of different losses
            args: Command line arguments
        """

        self._loss_names = loss_names
        self._base_dir = base_dir
        self._loss_epochs = {name: [] for name in loss_names}
        self._loss_iterations = {name: [] for name in loss_names}

        # File/Directory names
        self._args_path = os.path.join(base_dir, "args.txt")
        self._tensorboard_dir = os.path.join(base_dir, "tensorboard/")
        self._checkpoints_dir = os.path.join(base_dir, "checkpoints/")
        self._plots_dir = os.path.join(base_dir, "plots/")
        self._create_dirs()

        self._args = args

    def _create_dirs(self):
        """Create necessary directories"""
        for d in [self._tensorboard_dir, self._checkpoints_dir, self._plots_dir]:
            ensure_dir(d)

    def _plot_loss_epochs(self):
        plt.figure()
        for name, losses in self._loss_epochs.items():
            data = np.array(losses)
            plt.plot(data[:, 0], data[:, 1], label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self._plots_dir, "epoch-loss.png"))

    def _plot_loss_iterations(self):
        plt.figure()

        for name, losses in self._loss_iterations.items():
            data = np.array(losses)
            plt.plot(data[:, 0], data[:, 1], label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self._plots_dir, "iteration-loss.png"))

    def _save_args(self):
        """Save options"""
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

    def add_epoch_loss(self, epoch: int, value: float, loss_name: str) -> None:
        """
        Add a specific loss for a single epoch.
        Args:
            epoch: Epoch index
            value: Loss value
            loss_name: Loss name
        """
        self._loss_epochs[loss_name].append([epoch, value])
        # TODO: Add to tensorboard

    def add_iteration_loss(self, iteration: int, value: float, loss_name: str) -> None:
        """
        Add a specific loss for a single iteration.
        Args:
            iteration: Iteration index
            value: Loss value
            loss_name: Loss name
        """
        self._loss_iterations[loss_name].append([iteration, value])
        # TODO: Add to tensorboard

    def save_checkpoint(self, checkpoint, suffix: str):
        # TODO
        pass

    def save_sample_predictions(self, samples):
        # TODO
        pass

    def evaluate(self):
        """Evaluate"""
        self._plot_loss_epochs()
        self._plot_loss_iterations()
        self._save_args()


def ensure_dir(file: str) -> None:
    """
    Ensures that a given directory exists.

    Args:
        file: file
    """
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    setup_logging()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="hello")
    parser.add_argument("--test1", default="hello")
    parser.add_argument("--test12", default="hello")
    parser.add_argument("--test123", default="hello")
    args = parser.parse_args()
    # Test Evaluator
    ev = Evaluator(base_dir="/tmp/test", loss_names=["loss1", "loss2"], args=args)
    for i in range(100):
        if i % 10 == 0:
            ev.add_epoch_loss(epoch=np.floor(i / 10.0), loss_name="loss1", value=i / 2)
            ev.add_epoch_loss(epoch=np.floor(i / 10.0), loss_name="loss2", value=i / 4)

        ev.add_iteration_loss(iteration=i, loss_name="loss1", value=i / 2)
        ev.add_iteration_loss(iteration=i, loss_name="loss2", value=100 - i ** 0.98)

    ev.evaluate()
