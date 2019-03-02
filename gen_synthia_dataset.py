import argparse
import os
import random

SYNTHIA_SEQS = [
    "SYNTHIA-SEQS-01-SPRING",
    "SYNTHIA-SEQS-01-SUMMER",
    "SYNTHIA-SEQS-01-FALL",
    "SYNTHIA-SEQS-02-SPRING",
    "SYNTHIA-SEQS-02-SUMMER",
    "SYNTHIA-SEQS-02-FALL",
    "SYNTHIA-SEQS-04-SPRING",
    "SYNTHIA-SEQS-04-SUMMER",
    "SYNTHIA-SEQS-04-FALL",
    "SYNTHIA-SEQS-05-SPRING",
    "SYNTHIA-SEQS-05-SUMMER",
    "SYNTHIA-SEQS-05-FALL",
    "SYNTHIA-SEQS-06-SPRING",
    "SYNTHIA-SEQS-06-SUMMER",
    "SYNTHIA-SEQS-01-FALL",
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
        default="resources/filenames/synthia_train_files.txt",
    )
    parser.add_argument(
        "--val-filenames-file-out",
        help="File that contains a list of filenames for validation. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        default="resources/filenames/synthia_val_files.txt",
    )
    parser.add_argument(
        "--test-filenames-file-out",
        help="File that contains a list of filenames for validation. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        default="resources/filenames/synthia_test_files.txt",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[80, 10, 10],
        help="Size of train and val set",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # generate all filenames (left and right pair, separated by space)
    filename_lines = []
    for seq in SYNTHIA_SEQS:
        # Stereo_Left directory
        leftdir = os.path.join(args.data_dir, seq, "RGB", "Stereo_Left", "Omni_F")
        leftfiles = [
            os.path.join(seq, "RGB", "Stereo_Left", "Omni_F", f)
            for f in sorted(os.listdir(leftdir))
            if os.path.isfile(os.path.join(leftdir, f))
        ]

        # Stereo_Right directory
        rightdir = os.path.join(args.data_dir, seq, "RGB", "Stereo_Left", "Omni_F")
        rightfiles = [
            os.path.join(seq, "RGB", "Stereo_Right", "Omni_F", f)
            for f in sorted(os.listdir(rightdir))
            if os.path.isfile(os.path.join(rightdir, f))
        ]

        leftright = [left + " " + right for left, right in zip(leftfiles, rightfiles)]

        filename_lines += leftright

    n_files = len(filename_lines)
    print("Using a SYNTHIA subset with {} files".format(n_files))

    randperm = list(range(n_files))
    random.shuffle(randperm)
    train_size, val_size, test_size = [int(size / 100 * n_files) for size in args.sizes]
    train_idx = randperm[:train_size]
    val_idx = randperm[train_size : (train_size + val_size)]
    test_idx = randperm[(train_size + val_size) : -1]

    print("Training set size: {}".format(len(train_idx)))
    print("Validation set size: {}".format(len(val_idx)))
    print("Test set size: {}".format(len(test_idx)))

    with open(args.filenames_file_out, "w") as f:
        for item in [filename_lines[i] for i in train_idx]:
            f.write("%s\n" % item)

    with open(args.val_filenames_file_out, "w") as f:
        for item in [filename_lines[i] for i in val_idx]:
            f.write("%s\n" % item)

    with open(args.test_filenames_file_out, "w") as f:
        for item in [filename_lines[i] for i in test_idx]:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
