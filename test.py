import argparse
import logging
import torch
import os
from os.path import dirname as path_up
import matplotlib.pyplot as plt
import numpy as np

from utils import get_model, to_device, setup_logging
from monolab.data_loader import prepare_dataloader
from eval.eval_eigensplit import EvaluateEigen
from eval.eval_kitti_gt import EvaluateKittiGT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch Monolab Testing: Monodepth x " "DeepLabv3+"
    )
    parser.add_argument(
        "--filenames-file",
        help="File that contains a list of filenames for testing. \
                            Each line should contain left and right image paths \
                            separated by a space.",
    )
    parser.add_argument(
        "--model",
        default="resnet18_md",
        help="encoder architecture: "
        + "resnet18_md or resnet50_md "
        + "(default: resnet18)"
        + "or torchvision version of any resnet model",
    )
    parser.add_argument("--model-path", help="Path to a trained model")
    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/kitti",
        help="path to the dataset folder. \
                            The filenames given in filenames_file \
                            are relative to this path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for all results generated during an experiment run",
    )
    parser.add_argument("--input-height", type=int, help="input height", default=256)
    parser.add_argument("--input-width", type=int, help="input width", default=512)
    parser.add_argument(
        "--input-channels",
        default=3,
        type=int,
        help="Number of channels in input tensor",
    )
    parser.add_argument(
        "--device", default="cuda:0", help='choose cpu or cuda:0 device"'
    )
    parser.add_argument(
        "--num-workers", default=4, type=int, help="Number of workers in dataloader"
    )
    parser.add_argument("--use-multiple-gpu", default=False)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["verbose", "info", "warning", "error", "debug"],
        help="Log level",
    )
    parser.add_argument(
        "--eval",
        default="none",
        type=str,
        help="Either evaluate on eigensplit or on kitti gt",
        choices=["kitti-gt", "eigen", "none"],
    )
    parser.add_argument("--log-file", default="monolab.log", help="Log file")
    args = parser.parse_args()
    return args


class TestRunner:
    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, n_input_channels=args.input_channels)

        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.load(args.model_path)
        self.model = self.model.to(self.device)

        args.augment_parameters = None
        args.do_augmentation = False

        # setup output directory
        if args.output_dir is not None:
            self.output_dir = args.output_dir
        else:
            self.output_dir = os.path.join(path_up(path_up(args.model_path)), "test")

        # Load data
        self.input_height = args.input_height
        self.input_width = args.input_width

        dataset = args.filenames_file.split("_")[0].split("/")[2]

        self.n_img, self.loader = prepare_dataloader(
            root_dir=args.data_dir,
            filenames_file=args.filenames_file,
            mode="test",
            augment_parameters=None,
            do_augmentation=False,
            shuffle=False,
            batch_size=1,
            size=(args.input_height, args.input_width),
            num_workers=args.num_workers,
            dataset=dataset,
        )

        logging.info("Using a testing data set with {} images".format(self.n_img))

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def test(self):
        """ Test the model.

        Returns:
            None
        """

        self.model.eval()
        self.disparities = np.zeros(
            (self.n_img, self.input_height, self.input_width), dtype=np.float32
        )
        self.disparities_pp = np.zeros(
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
                self.disparities[i] = disp[0].squeeze().cpu().numpy()
                self.disparities_pp[i] = post_process_disparity(
                    disps[0][:, 0, :, :].cpu().numpy()
                )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        np.save(os.path.join(self.output_dir, "disparities.npy"), self.disparities)
        np.save(
            os.path.join(self.output_dir, "disparities_pp.npy"), self.disparities_pp
        )

        for i in range(self.disparities.shape[0]):
            plt.imsave(
                os.path.join(self.output_dir, "pred_" + str(i) + ".png"),
                self.disparities[i],
                cmap="plasma",
            )

        logging.info("Finished Testing")

    def evaluate(self, postprocessing):
        """ Evaluates the model given either ground truth data or velodyne reprojected data

        Returns:
            None

        """

        if postprocessing:
            disparities = self.disparities_pp
        else:
            disparities = self.disparities

        # Evaluates on the 200 Kitti Stereo 2015 Test Files
        if self.args.eval == "kitti-gt":
            if "kitti_stereo_2015_test_files" not in self.args.filenames_file:
                raise ValueError(
                    "For KITTI GT evaluation, the test set should be 'kitti_stereo_2015_test_files.txt'"
                )
            abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = EvaluateKittiGT(
                predicted_disps=disparities,
                gt_path=self.args.data_dir,
                min_depth=0,
                max_depth=80,
            ).evaluate()

        # Evaluates on the 697 Eigen Test Files
        elif self.args.eval == "eigen":
            abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = EvaluateEigen(
                predicted_disps=disparities,
                test_file_path=self.args.filenames_file,
                gt_path=self.args.data_dir,
                min_depth=0,
                max_depth=80,
            ).evaluate()
        else:
            logging.warning(
                "{} is not a valid evaluation procedure.".format(self.args.eval)
            )


        logging.info(
            "{},{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
                "pp", "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"
            )
        )
        logging.info(
            "{},{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
                postprocessing,
                abs_rel.mean(),
                sq_rel.mean(),
                rms.mean(),
                log_rms.mean(),
                a1.mean(),
                a2.mean(),
                a3.mean(),
            )
        )

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


def post_process_disparity(disp: np.ndarray) -> np.ndarray:
    """ Apply the post-processing step described in the paper

    Args:
        disp: [2, h, w] array, a disparity map

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


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level, filename=args.log_file)

    model = TestRunner(args)
    model.test()

    if args.eval:
        model.evaluate(postprocessing=False)
        model.evaluate(postprocessing=True)
