import argparse
import logging
from typing import Tuple

import torch
import os
from os.path import dirname as path_up
import matplotlib.pyplot as plt
import numpy as np

from eval.eval_utils import Result, results_to_csv_str
from utils import get_model, to_device, setup_logging
from monolab.data_loader import prepare_dataloader
from eval.eval_eigensplit import EvaluateEigen
from eval.eval_kitti_gt import EvaluateKittiGT
from eval.eval_synthia import EvaluateSynthia
from torch import nn
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch Monolab Testing: Monodepth x " "DeepLabv3+"
    )
    parser.add_argument(
        "--test-filenames-file",
        help="File that contains a list of filenames for testing. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        metavar="FILE",
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
        metavar="DIR",
    )
    parser.add_argument(
        "--atrous-rates",
        nargs="+",
        type=int,
        default=[1, 6, 12, 18],
        help="Atrous rates for the ASPP Module.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for all results generated during an experiment run",
        metavar="DIR",
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


def run_test(model: nn.Module, device, result_dir, args):
    """
    Run the test procedure, that is: Generate disparity maps for all test files and
    evaluate the model on the given groun-truth data.
    Args:
        model: Model to be tested
        device: Execution device
        result_dir: Result directory (for predictions)
        args: Commandline arguments
    """
    output_dir = result_dir

    preds_dir = os.path.join(output_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Load data
    input_height = args.input_height
    input_width = args.input_width

    dataset = args.test_filenames_file.split("_")[0].split("/")[2]

    n_img, loader = prepare_dataloader(
        root_dir=args.data_dir,
        filenames_file=args.test_filenames_file,
        mode="test",
        augment_parameters=None,
        do_augmentation=False,
        shuffle=False,
        batch_size=1,
        size=(args.input_height, args.input_width),
        num_workers=args.num_workers,
        dataset=dataset,
        pin_memory=args.pin_memory,
    )

    logging.info("Using a testing data set with {} images".format(n_img))

    if "cuda" in device:
        torch.cuda.synchronize()

    model.eval()
    disparities = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    disparities_pp = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    with torch.no_grad():
        for (i, data) in enumerate(loader):
            # Get the inputs
            data = to_device(data, device)
            left = data.squeeze()
            # Do a forward pass
            disps = model(left)
            disp = disps[0][:, 0, :, :].unsqueeze(1)
            disparities[i] = disp[0].squeeze().cpu().numpy()
            disparities_pp[i] = post_process_disparity(
                disps[0][:, 0, :, :].cpu().numpy()
            )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, "disparities.npy"), disparities)
    np.save(os.path.join(output_dir, "disparities_pp.npy"), disparities_pp)

    for i in range(disparities.shape[0]):
        plt.imsave(
            os.path.join(preds_dir, str(i).zfill(3) + ".png"),
            disparities[i],
            cmap="plasma",
        )

    logging.info("Finished Testing")
    eval_result = _evaluate_scores(disparities, args)
    eval_result_pp = _evaluate_scores(disparities_pp, args)

    if eval_result is not None:
        # Save scores
        with open(os.path.join(output_dir, "scores.csv"), "w") as f:
            f.write(results_to_csv_str(eval_result, eval_result_pp))

    return eval_result, eval_result_pp


def _evaluate_scores(disparities, args):
    """ Evaluates the model given either ground truth data or velodyne reprojected data

    Returns:
        None

    """

    # Evaluates on the 200 Kitti Stereo 2015 Test Files
    if args.eval == "kitti-gt":
        if "kitti_stereo_2015_test_files" not in args.test_filenames_file:
            raise ValueError(
                "For KITTI GT evaluation, the test set should be 'kitti_stereo_2015_test_files.txt'"
            )
        result = EvaluateKittiGT(
            predicted_disps=disparities,
            gt_path=args.data_dir,
            min_depth=0,
            max_depth=80,
        ).evaluate()

    elif args.eval == "eigen":
        # Evaluates on the 697 Eigen Test Files
        result = EvaluateEigen(
            predicted_disps=disparities,
            test_file_path=args.test_filenames_file,
            gt_path=args.data_dir,
            min_depth=0,
            max_depth=80,
        ).evaluate()
    elif args.eval == "synthia":
        # Evaluates on a SYNTHIA test set
        result = EvaluateSynthia(
            predicted_disps=disparities,
            filenames_file=args.test_filenames_file,
            root_dir=args.data_dir,
            min_depth=0,
            max_depth=50,
        ).evaluate()
    elif args.eval == "none":
        return None
    else:
        logger.error("{} is not a valid evaluation procedure.".format(args.eval))
        raise Exception(f"Invalid evaluation procedure: {args.eval}")

    return result


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

    model = get_model(args.model, n_input_channels=args.input_channels, args=args)

    if args.use_multiple_gpu:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)

    run_test(
        model=model,
        args=args,
        device=args.device,
        result_dir=os.path.join(args.output_dir, "test"),
    )
