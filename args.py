import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch Monolab: Monodepth x " "DeepLabv3+"
    )

    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/kitti",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
    )
    parser.add_argument(
        "--filenames-file",
        help="File that contains a list of filenames for training. \
                        Each line should contain left and right image paths \
                        separated by a space.",
    )
    parser.add_argument(
        "--val-filenames-file",
        help="File that contains a list of filenames for validation. \
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
    parser.add_argument("--model-path", help="path to the trained model")
    parser.add_argument(
        "--output-dir",
        default="/visinf/projects_students/monolab/results/",
        help="Output directory for all results generated during an experiment run",
    )
    parser.add_argument("--input-height", type=int, help="input height", default=256)
    parser.add_argument("--input-width", type=int, help="input width", default=512)
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of total " "epochs to run"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="initial learning rate (" \
                                                      "default: 1e-4)"
    )
    parser.add_argument(
        "--adjust-lr",
        default=True,
        help="apply learning rate decay or not\
                            (default: True)",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, help="mini-batch size (default: 256)"
    )
    parser.add_argument(
        "--device", default="cuda:0", help='choose cpu or cuda:0 device"'
    )
    parser.add_argument(
        "--do-augmentation", default=True, help="do augmentation of images or not"
    )
    parser.add_argument(
        "--augment-parameters",
        default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
        help="lowest and highest values for gamma,\
                        brightness and color respectively",
    )
    parser.add_argument(
        "--input-channels",
        default=3,
        type=int,
        help="Number of channels in input tensor",
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
        "--notify",
        default=None,
        help="Put in email-address to notify when training has finished"
    )

    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Tag to identify runs in the result directory and tensorboard overviews",
    )
    parser.add_argument("--log-file", default="monolab.log", help="Log file")

    args = parser.parse_args()
    return args
