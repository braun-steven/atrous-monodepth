import argparse
import os
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate filenames file for SYNTHIA")

    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/vkitti",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
    )
    parser.add_argument(
        "--vkitti_version",
        default="vkitti_1.3.1",
        help="vkitti version name (including 'vkitti_')",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        type=str,
        default=["0001", "0002", "0006", "0018", "0020"],
        help="Sequence numbers (as str with leading zero) to be used",
    )
    parser.add_argument("--variations", nargs="+", type=str, default=["clone"])

    parser.add_argument(
        "--filenames-file-out",
        help="File that contains a list of filenames for training/testing. \
                        Each line should contain left and right image paths \
                        separated by a space.",
        default="resources/filenames/vkitti_test_files.txt",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    vkitti_name = args.vkitti_version + "_rgb"

    # generate all filenames
    filename_lines = []
    for nr in args.sequences:
        for variation in args.variations:
            seq = "{}/{}".format(nr, variation)

            directory = os.path.join(args.data_dir, vkitti_name, seq)
            files = [
                os.path.join(vkitti_name, seq, f)
                for f in sorted(os.listdir(directory))
                if os.path.isfile(os.path.join(directory, f))
            ]

            filename_lines += files

    n_files = len(filename_lines)
    print("Using a vkitti subset with {} files".format(n_files))

    with open(args.filenames_file_out, "w") as f:
        for item in filename_lines:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
