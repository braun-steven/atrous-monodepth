import numpy as np
import cv2
import os

from eval.eval_utils import compute_errors, Result


class EvaluateKittiGT:
    """
    Class that evaluates the KITTI data set on the 200 ground truth image

    How to use this class:
    Create an object of this class and then use the evaluate method to evaluate
    EvaluateKittiGT(
    predicted_disps= np.array containing predicted disparities
    gt_path='../../data/kitti/data_scene_flow/', min_depth=0, max_depth=80).evaluate()

    """

    def __init__(self, predicted_disps, gt_path, min_depth=0, max_depth=80):
        """
        Args:
            predicted_disps: (np.ndarray) predicted disparities after training
            gt_path: path of the ground truth
            min_depth: minimum depth used in predicted disparity map
            max_depth: maximim depth used in predicted disparity map
        """

        self.width_to_focal = dict()
        self.width_to_focal[1242] = 721.5377
        self.width_to_focal[1241] = 718.856
        self.width_to_focal[1224] = 707.0493
        self.width_to_focal[1238] = 718.3351

        self.predicted_disps = predicted_disps
        self.gt_path = gt_path
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

        num_samples = 200
        gt_disparities = self.__load_gt_disp_kitti(self.gt_path)
        gt_depths, pred_depths, pred_disparities_resized = self.__convert_disps_to_depths_kitti(
            gt_disparities, pred_disparities
        )

        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1_all = np.zeros(num_samples, np.float32)
        a1 = np.zeros(num_samples, np.float32)
        a2 = np.zeros(num_samples, np.float32)
        a3 = np.zeros(num_samples, np.float32)

        for i in range(num_samples):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth

            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]
            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(
                disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05
            )
            d1_all[i] = (
                100.0 * bad_pixels.sum() / mask.sum()
            )  # TODO return D1 all error

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

    def __load_gt_disp_kitti(self, path):
        """
        Loads in the ground truth files from the KITTI Stereo Dataset

        Args:
         -path (string): path to the the training files

        Returns:
         -gt_disparities (list): list of ground truth disparities
        """
        gt_disparities = []
        for i in range(200):
            disp = cv2.imread(
                path + "/training/disp_noc_0/" + str(i).zfill(6) + "_10.png", -1
            )
            disp = disp.astype(np.float32) / 256
            gt_disparities.append(disp)
        return gt_disparities

    def __convert_disps_to_depths_kitti(self, gt_disparities, pred_disparities):
        """
        Converts the ground truth disparities from the KITTI Stereo dataset and the predictions to depth values

        Args:
         -gt_disparities (list): ground truth disparities
         -pred_disparities (list): predicted disparities

        Returns:
         -gt_depths (list): list of ground truth depths
         -pred_depths (list): list of predicted depths
         -pred_depths_resized (list): list of predicted depths, resized to ground truth disparity size

        """
        gt_depths = []
        pred_depths = []
        pred_disparities_resized = []

        for i in range(len(gt_disparities)):
            gt_disp = gt_disparities[i]
            height, width = gt_disp.shape

            pred_disp = pred_disparities[i]
            pred_disp = width * cv2.resize(
                pred_disp, (width, height), interpolation=cv2.INTER_LINEAR
            )

            pred_disparities_resized.append(pred_disp)

            mask = gt_disp > 0

            gt_depth = self.width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
            pred_depth = self.width_to_focal[width] * 0.54 / pred_disp

            gt_depths.append(gt_depth)
            pred_depths.append(pred_depth)

        return gt_depths, pred_depths, pred_disparities_resized
