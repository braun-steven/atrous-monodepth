import numpy as np
import cv2
import os

from eval.eval_utils import compute_errors, Result


class EvaluateVKittiGT:
    """
    Class that evaluates the VKITTI data set on the ground truth images

    How to use this class:
    Create an object of this class and then use the evaluate method to evaluate
    EvaluateVKittiGT(
    predicted_disps= np.array containing predicted disparities
    gt_path='../../data/vkitti/vkitti_1.3.1_depthgt/', filenames_file, min_depth=0, max_depth=80).evaluate()

    """

    def __init__(
        self, predicted_disps, root_dir, filenames_file, min_depth=0, max_depth=80
    ):
        """
        Args:
            predicted_disps: (np.ndarray) predicted disparities after training
            gt_path: path of the ground truth
            filenames_file: filenames of the test images
            min_depth: minimum depth used in predicted disparity map
            max_depth: maximim depth used in predicted disparity map
        """

        self.width_to_focal = dict()
        self.width_to_focal[1242] = 721.5377
        self.width_to_focal[1241] = 718.856
        self.width_to_focal[1224] = 707.0493
        self.width_to_focal[1238] = 718.3351

        self.predicted_disps = predicted_disps
        self.root_dir = root_dir

        # load image paths
        with open(filenames_file) as filenames:
            image_paths = sorted(
                os.path.join(root_dir, fname.split()[0]) for fname in filenames
            )

        # convert to depth paths
        self.depth_paths = [
            "/".join(
                [
                    path if path != "vkitti_1.3.1_rgb" else "vkitti_1.3.1_depthgt"
                    for path in str.split(image_path, "/")
                ]
            )
            for image_path in image_paths
        ]

        self.min_depth = min_depth
        self.max_depth = max_depth

    def evaluate(self):
        """
        Evaluates the predicted depth data for the KITI dataset on the 200 ground truth files

        Args:
            gt: numpy array (2D) of the ground depth map
            pred: numpy array (2D) of the predicted depth

        Returns:
            Evaluation result
        """

        pred_disparities = self.predicted_disps

        self.gt_depths = self.__load_gt_depth_vkitti()
        num_samples = len(self.gt_depths)

        self.pred_depths, pred_disparities_resized = self.__convert_disps_to_depths_kitti(
            pred_disparities, self.gt_depths
        )

        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        # d1_all = np.zeros(num_samples, np.float32)
        a1 = np.zeros(num_samples, np.float32)
        a2 = np.zeros(num_samples, np.float32)
        a3 = np.zeros(num_samples, np.float32)

        for i in range(num_samples):
            gt_depth = self.gt_depths[i]
            pred_depth = self.pred_depths[i]

            mask = gt_depth <= self.max_depth

            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
                i
            ] = compute_errors(gt_depth[mask], pred_depth[mask])

        return Result(
            abs_rel=abs_rel,
            sq_rel=sq_rel,
            rms=rms,
            log_rms=log_rms,
            a1=a1,
            a2=a2,
            a3=a3,
        )

    def __load_gt_depth_vkitti(self):
        """
        Loads in the ground truth files from the KITTI Stereo Dataset

        Args:

        Returns:
         -gt_depths (list): list of ground truth depth maps
         -gt_depths (list): list of ground truth depth maps
        """
        gt_depths = []
        for path in self.depth_paths:
            depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100

            gt_depths.append(depth)
        return gt_depths

    def __convert_disps_to_depths_kitti(self, pred_disparities, gt_depths):
        """
        Converts the predictions to depth values

        Args:
         -pred_disparities (list): predicted disparities
         -gt_depths (list): ground truth disparities (only needed for proper sizes)

        Returns:
         -pred_depths (list): list of predicted depths
         -pred_depths_resized (list): list of predicted depths, resized to ground truth disparity size

        """

        pred_depths = []
        pred_disparities_resized = []

        for i in range(len(pred_disparities)):
            gt_depth = gt_depths[i]
            height, width = gt_depth.shape

            pred_disp = pred_disparities[i]
            pred_disp = width * cv2.resize(
                pred_disp, (width, height), interpolation=cv2.INTER_LINEAR
            )

            pred_disparities_resized.append(pred_disp)

            pred_depth = self.width_to_focal[width] * 0.54 / pred_disp

            pred_depths.append(pred_depth)

        return pred_depths, pred_disparities_resized
