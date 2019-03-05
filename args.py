import argparse
import os


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
        metavar="DIR",
    )
    parser.add_argument(
        "--filenames-file",
        help="File that contains a list of filenames for training. \
                        Each line should contain left and right image paths \
                        separated by a space.",
        metavar="FILE",
    )
    parser.add_argument(
        "--val-filenames-file",
        help="File that contains a list of filenames for validation. \
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
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint of previously trained model to start from (used for pretraining on a different dataset)",
        metavar="FILE",
    )
    parser.add_argument(
        "--imagenet-pretrained",
        default=False,
        help="load imagenet-pretrained state dict for resnet backend",
    )
    parser.add_argument(
        "--output-dir",
        default="/visinf/projects_students/monolab/results/",
        help="Output directory for all results generated during an experiment run",
        metavar="DIR",
    )
    parser.add_argument("--input-height", type=int, help="input height", default=256)
    parser.add_argument("--input-width", type=int, help="input width", default=512)
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of total " "epochs to run"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="initial learning rate (" "default: 1e-4)",
    )
    parser.add_argument(
        "--adjust-lr",
        default=True,
        help="apply learning rate decay or not\
                            (default: True)",
    )
    parser.add_argument(
        "--batch-size", default=16, type=int, help="mini-batch size (default: 256)"
    )
    parser.add_argument(
        "--cuda-device-ids",
        nargs="+",
        type=int,
        default=[0],
        help="Cuda device ids. E.g. [0,1,2]. Use -1 for all GPUs available and -2 for cpu only.",
    )
    parser.add_argument(
        "--output-stride", type=int, default=64, help="Output stride after the encoder"
    )
    parser.add_argument(
        "--encoder-dilations",
        nargs="+",
        type=int,
        default=[1, 1, 1, 1],
        help="Atrous rates used in the encoder's resblocks",
    )
    parser.add_argument(
        "--atrous-rates",
        nargs="+",
        type=int,
        default=[1, 6, 12, 18],
        help="Atrous rates for the ASPP Module.",
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

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["verbose", "info", "warning", "error", "debug"],
        help="Log level",
    )

    parser.add_argument(
        "--notify",
        default=None,
        help="Put in email-address to notify when training has finished",
    )

    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Tag to identify runs in the result directory and tensorboard overviews",
    )
    parser.add_argument(
        "--weight-ssim", default=0.85, type=float, help="SSIM weight in Monodepth Loss"
    )
    parser.add_argument(
        "--weight-disp-gradient",
        default=0.1,
        type=float,
        help="Distparity gradient weight in Monodepth Loss",
    )
    parser.add_argument(
        "--weight-lr-consistency",
        default=1.0,
        type=float,
        help="Left-Right consistency weight in Monodepth Loss",
    )
    parser.add_argument(
        "--seed", default=7, type=int, help="Seed for random number generators"
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Whether to overfit on a small (10 batches) subset of the "
        "training "
        "images.",
    )

    kitti_name = "kitti"
    cityscapes_name = "cityscapes"
    synthia_name = "synthia"
    parser.add_argument(
        "--dataset-name-train",
        type=str,
        choices=[cityscapes_name, kitti_name, synthia_name],
        help="Define the train dataset name.",
    )
    parser.add_argument(
        "--dataset-name-val",
        type=str,
        choices=[cityscapes_name, kitti_name, synthia_name],
        help="Define the validation dataset name.",
    )

    parser.add_argument(
        "--eval",
        default="none",
        type=str,
        help="Either evaluate on eigensplit or on kitti gt, or on synthia",
        choices=["kitti-gt", "eigen", "synthia", "none"],
    )
    parser.add_argument(
        "--test-filenames-file",
        help="File that contains a list of filenames for testing. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        metavar="FILE",
    )
    parser.add_argument("--log-file", default="monolab.log", help="Log file")
    parser.add_argument(
        "--disable-skip-connections",
        default=False,
        action="store_true",
        help="Flag to add skip connections from the encoder to the decoder.",
    )
    parser.add_argument(
        "--disable-aspp-global-avg-pooling",
        default=False,
        action="store_true",
        help="Flag to disable global average pooling.",
    )
    parser.add_argument(
        "--pin-memory", default=True, help="pin_memory argument to all dataloaders"
    )
    args = parser.parse_args()

    # Detect training dataset name
    if args.dataset_name_train is None:
        if kitti_name in args.filenames_file.lower():
            args.dataset_name_train = kitti_name
        elif cityscapes_name in args.filenames_file.lower():
            args.dataset_name_train = cityscapes_name
        elif synthia_name in args.filenames_file.lower():
            args.dataset_name_train = synthia_name
        else:
            raise Exception(
                f"Usage:\n{parser.format_help()}"
                f"Could not detect dataset-name-train from "
                f"filenames file name and argument --dataset-name-train was not "
                f"explicitly set. "
            )
    # Detect validation dataset name
    if args.dataset_name_val is None:
        # Try to automatically detect the datasetname
        if kitti_name in args.val_filenames_file.lower():
            args.dataset_name_val = kitti_name
        elif cityscapes_name in args.val_filenames_file.lower():
            args.dataset_name_val = cityscapes_name
        elif synthia_name in args.val_filenames_file.lower():
            args.dataset_name_val = synthia_name
        else:
            raise Exception(
                f"Usage:\n{parser.format_help()}"
                f"Could not detect dataset-name-train from "
                f"filenames file name and argument --dataset-name-train was not "
                f"explicitly set. "
            )
    return args
