import argparse
import glob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small sample dataset filenames file"
    )

    parser.add_argument(
        "--data-dir-left",
        default="/visinf/projects_students/monolab/data/kitti",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
    )
    parser.add_argument(
        "--data-dir-right",
        default="/visinf/projects_students/monolab/data/kitti",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
    )
    parser.add_argument(
        "--filenames-file-out",
        help="File that contains a list of filenames for training/testing. \
                        Each line should contain left and right image paths \
                        separated by a space.",
        default="filenames-sampled.txt",
    )
    parser.add_argument(
        "--val-filenames-file-out",
        help="File that contains a list of filenames for validation. \
                            Each line should contain left and right image paths \
                            separated by a space.",
        default="val-filenames-sampled.txt",
    )
    parser.add_argument(
        "--size", type=int, default=10, help="Size of train and val set"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Read files
    files_left = glob.glob(args.data_dir_left + "/**/*.jpg", recursive=True)
    files_right = glob.glob(args.data_dir_right + "/**/*.jpg", recursive=True)

    if len(files_right) == 0:
        print("No right files found.")

    if len(files_left) == 0:
        print("No left files found.")

    # Remove everything before "kitti"
    files_left = [f.split("kitti/")[1] for f in files_left]
    files_right = [f.split("kitti/")[1] for f in files_right]

    files_right = files_right[: args.size]
    files_left = files_left[: args.size]

    half = round(args.size / 2)

    files_left_train = files_left[:half]
    files_left_val = files_left[half:]
    files_right_train = files_right[:half]
    files_right_val = files_right[half:]

    def gen_filenames_file(left, right, fname):
        # Generate train and validation filenames file
        with open(fname, "w") as f:
            for l, r in zip(left, right):
                f.write("{} {}\n".format(l, r))

    gen_filenames_file(files_left_train, files_right_train, args.filenames_file_out)
    gen_filenames_file(files_left_val, files_right_val, args.val_filenames_file_out)


if __name__ == "__main__":
    main()
