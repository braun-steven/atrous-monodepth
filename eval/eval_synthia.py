import numpy as np
import cv2
import os

from eval.eval_utils import compute_errors, Result


class EvaluateSynthia:
    """
    Class that evaluates on a SYNTHIA test set

    How to use this class:
    Create an object of this class and then use the evaluate method to evaluate
    EvaluateSynthia(...).evaluate()


    """

    def __init__(
        self,
        predicted_disps,
        filenames_file,
        min_depth=0,
        max_depth=80,
        root_dir="/visinf/projects_students/monolab/data/synthia",
        size=(1280, 760),
        baseline=0.8,
        focal=532.7403520000000,
    ):
        """
        Args:
            predicted_disps: (np.ndarray) predicted disparities after training
            filenames_file: paths of the test images
            min_depth: minimum depth used in predicted disparity map
            max_depth: maximim depth used in predicted disparity map
            root_dir: data directory
        """

        self.img_size = size
        self.predicted_disps = predicted_disps
        self.filenames_file = filenames_file

        # collect left image filepaths
        with open(filenames_file) as filenames:
            self.left_image_paths = sorted(
                os.path.join(root_dir, fname.split()[0]) for fname in filenames
            )

        self.num_images = len(self.left_image_paths)

        # load ground truth depth maps
        self.gt_depths = self._load_synthia_gt_depth(self.left_image_paths)

        self.min_depth = min_depth
        self.max_depth = max_depth

        # camera parameteres
        self.baseline = baseline
        self.focal = focal

    def _load_synthia_gt_depth(self, image_paths):
        """ Create a list of gt depth images given the original image paths

        Args:
            image_paths: list of strings containing left images

        Returns:
            list containing cv2 images
        """
        # create depth image paths from RGB image paths
        left_depth_paths = [
            "/".join(
                [
                    path if path != "RGB" else "Depth"
                    for path in str.split(image_path, "/")
                ]
            )
            for image_path in image_paths
        ]

        # read list of test images
        gt_depth = []
        for i in range(self.num_images):
            depth = cv2.imread(left_depth_paths[i], cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32)[:, :, 0] / 100
            gt_depth.append(depth)
        return gt_depth

    def _disp_to_depth(self, pred_disparities, size=(1280, 760)):
        """
        Converts the predicted disparities to depth values

        Args:
         -pred_disparities (list): predicted disparities
         -size (tuple of int): height, width of the gt disparity

        Returns:
         -pred_depths (list): list of predicted depths (resized to gt size)
         -pred_disp_resized (list): predicted disparities (resized to gt size)

        """
        width, height = size
        pred_depths = []
        pred_disp_resized = []

        for i in range(len(pred_disparities)):
            pred_disp = pred_disparities[i]
            pred_disp = width * cv2.resize(
                pred_disp, (width, height), interpolation=cv2.INTER_LINEAR
            )

            pred_disp_resized.append(pred_disp)

            mask = pred_disp > 0

            pred_depth = self.baseline * self.focal / (pred_disp + (1.0 - mask))

            pred_depths.append(pred_depth)

        return pred_depths, pred_disp_resized

    def evaluate(self):
        """
        Evaluates the predicted depth data for the SYNTHIA dataset

        Args:
            gt: numpy array (2D) of the ground depth map
            pred: numpy array (2D) of the predicted depth

        Returns:
            Evaluation result
        """

        pred_depths, pred_disparities_resized = self._disp_to_depth(
            self.predicted_disps, size=self.img_size
        )

        rms = np.zeros(self.num_images, np.float32)
        log_rms = np.zeros(self.num_images, np.float32)
        abs_rel = np.zeros(self.num_images, np.float32)
        sq_rel = np.zeros(self.num_images, np.float32)
        a1 = np.zeros(self.num_images, np.float32)
        a2 = np.zeros(self.num_images, np.float32)
        a3 = np.zeros(self.num_images, np.float32)

        for i in range(self.num_images):
            gt_depth = self.gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
                i
            ] = compute_errors(gt_depth, pred_depth)

        return Result(
            abs_rel=abs_rel,
            sq_rel=sq_rel,
            rms=rms,
            log_rms=log_rms,
            a1=a1,
            a2=a2,
            a3=a3,
        )


if __name__ == "__main__":
    filenames_file = "../resources/filenames/synthia_spring_05.txt"
    root_dir = "../data/synthia"

    predicted_disps = np.load(
        "../data/output/run_19-03-02_13h:39m_resnet50_md_synthia/test/disparities.npy"
    )
    predicted_disps = predicted_disps[:295]

    eval = EvaluateSynthia(
        predicted_disps=predicted_disps,
        filenames_file=filenames_file,
        root_dir=root_dir,
    )

    import matplotlib.pyplot as plt

    plt.imshow(eval.gt_depths[0] / eval.gt_depths[0].max(), cmap="gray")
    plt.show()
