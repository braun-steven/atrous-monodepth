import argparse
import os

SYNTHIA_SEQS = [
    "SYNTHIA-SEQS-01-SPRING",
    "SYNTHIA-SEQS-01-SUMMER",
    "SYNTHIA-SEQS-02-SPRING",
    "SYNTHIA-SEQS-02-SUMMER",
    "SYNTHIA-SEQS-04-SPRING",
    "SYNTHIA-SEQS-04-SUMMER",
    "SYNTHIA-SEQS-05-SPRING",
    "SYNTHIA-SEQS-05-SUMMER",
    "SYNTHIA-SEQS-06-SPRING",
    "SYNTHIA-SEQS-06-SUMMER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate filenames file for SYNTHIA")

    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/synthia",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
    )

    parser.add_argument(
        "--filenames-file-out",
        help="File that contains a list of filenames for training/testing. \
                        Each line should contain left and right image paths \
                        separated by a space.",
        default="resources/filenames-synthia.txt",
    )
    parser.add_argument(
        "--val-filenames-file-out",
        help="File that contains a list of filenames for validation. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        default="val-filenames-synthia.txt",
    )
    parser.add_argument(
        "--test-filenames-file-out",
        help="File that contains a list of filenames for validation. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        default="test-filenames-synthia.txt",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[80, 10, 10],
        help="Size of train and val set",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    filename_lines = []
    for seq in SYNTHIA_SEQS:
        leftdir = os.path.join(args.data_dir, seq, "RGB", "Stereo_Left", "Omni_F")
        leftfiles = [
            os.path.join(seq, "RGB", "Stereo_Left", f)
            for f in sorted(os.listdir(leftdir))
            if os.path.isfile(os.path.join(leftdir, f))
        ]

        rightdir = os.path.join(args.data_dir, seq, "RGB", "Stereo_Left", "Omni_F")
        rightfiles = [
            os.path.join(seq, "RGB", "Stereo_Right", f)
            for f in sorted(os.listdir(rightdir))
            if os.path.isfile(os.path.join(rightdir, f))
        ]

        leftright = [left + " " + right for left, right in zip(leftfiles, rightfiles)]

        filename_lines += leftright

    print("Using a SYNTHIA subset with {} files".format(len(filename_lines)))

    with open(args.filenames_file_out, "w") as f:
        for item in filename_lines:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
