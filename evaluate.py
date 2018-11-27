import argparse
from eval.eval_eigensplit import EvaluateEigen
from eval.eval_kitti_gt import EvaluateKittiGT
from eval.eval_utils import load_disparities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch Monolab Evaluating: Monodepth x " "DeepLabv3+"
    )


    parser.add_argument(
        "--disparities-file",
        help="File that contains the model output disparities as .npy file",
        type=str,
    )

    parser.add_argument(
        "--filenames-file",
        help="File that contains a list of filenames for testing. \
                                Each line should contain left and right image paths \
                                separated by a space.",
    )

    parser.add_argument(
        "--eval",
        default="none",
        type=str,
        help="Either evaluate on eigensplit or on kitti gt",
        choices=["kitti-gt", "eigen", "none"],
    )
    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/kitti",
        help="path to the dataset folder. \
                                The filenames given in filenames_file \
                                are relative to this path.",
    )

    parser.add_argument(
        "--min-depth",
        default=0,
        type=int,
        help="the minimum depth that is used for evaluation"
    )


    parser.add_argument(
        "--max-depth",
        default=80,
        type=int,
        help="the maximum depth that is used for evaluation"
    )

def evaluate():
    disparities = load_disparities(args.disparities_file)

    # Evaluates on the 200 Kitti Stereo 2015 Test Files
    if args.eval == "kitti-gt":
        if "kitti_stereo_2015_test_files" not in args.filenames_file:
            raise ValueError("For KITTI GT evaluation, the test set should be 'kitti_stereo_2015_test_files.txt'")
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = EvaluateKittiGT(
            predicted_disps=disparities,
            gt_path=args.data_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        ).evaluate()

    # Evaluates on the 697 Eigen Test Files
    elif args.eval == "eigen":
        if "eigen_test_files.txt" not in args.filenames_file:
            raise ValueError("For Eigen split evaluation, the test set should be 'eigen_test_files.txt'")
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = EvaluateEigen(
            predicted_disps=disparities,
            test_file_path=args.filenames_file,
            gt_path=args.data_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        ).evaluate()
    else:
        raise ValueError("{} is not a valid evaluation procedure.".format(args.eval))


    #TODO maybe do this with logging?

    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"
        )
    )

    print(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            abs_rel.mean(),
            sq_rel.mean(),
            rms.mean(),
            log_rms.mean(),
            a1.mean(),
            a2.mean(),
            a3.mean(),
        )
    )


if __name__ == "__main__":
    print("Starting to Evaluate")
    args = parse_args()
    evaluate()