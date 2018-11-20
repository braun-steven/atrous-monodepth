import time
from argparse import Namespace

import torch

from evaluator import Evaluator
from monolab.data_loader import prepare_train_loader
from monolab.loss import MonodepthLoss
from utils import get_model, to_device, setup_logging
import logging


from monolab.utils.utils import notify_mail

logger = logging.getLogger(__name__)


class Experiment:
    """ A class for training and testing a model that contains the actual network as self.model
    The arguments are defined in the arg_parse()-function above.

    Important:
        - args.data_dir is the root directory
        - args.filenames_file should be different depending on whether you want to train or test the model
        - args.val_filenames_file is only used during training
    """

    def __init__(self, args: Namespace):
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

        # Setup loss, optimizer and validation set
        self.loss_function = MonodepthLoss(
            n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )
        # the validation loader is a train loader but without data augmentation!
        self.val_n_img, self.val_loader = prepare_train_loader(
            root_dir=args.data_dir,
            filenames_file=args.val_filenames_file,
            augment_parameters=args.augment_parameters,
            do_augmentation=False,
            batch_size=args.batch_size,
            size=(args.input_height, args.input_width),
            num_workers=args.num_workers,
        )
        logging.info("Using a validation set with {} images".format(self.val_n_img))

        # Load data
        self.output_dir = args.output_dir
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_train_loader(
            root_dir=args.data_dir,
            filenames_file=args.filenames_file,
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            batch_size=args.batch_size,
            size=(args.input_height, args.input_width),
            num_workers=args.num_workers,
        )
        logging.info("Using a training data set with {} images".format(self.n_img))

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def train(self) -> None:
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

        if self.args.notify is not None:
            notify_mail(
                self.args.notify,
                "[MONOLAB] Training Finished",
                "Finished Training. Best loss: {}".format(best_val_loss),
            )

        self.eval.save()

    def save(self, path: str) -> None:
        """ Save a .pth state dict from self.model

        Args:
            path: path to .pth state dict file

        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """ Load a .pth state dict into self.model

        Args:
            path: path to .pth state dict file

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))




def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, learning_rate: float
):
    """ Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epochs

    Args:
        optimizer: torch.optim type optimizer
        epoch: current epoch
        learning_rate: current learning rate

    """

    if 30 <= epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    setup_logging("monolab.log", "info")
