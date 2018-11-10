import numpy as np
import os
import cv, cv2
from collections import Counter


def compute_errors(gt, pred):
    """ Evaluates the predicted depth data on ground truth

    Args:
        gt: numpy array (2D) of the ground depth map
        pred: numpy array (2D) of the predicted depth

    Returns:
        -abs_rel
        -sq_rel
        -rmse
        -rmse_log
        -a1
        -a2x
        -a3
    """


    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def read_file_data(files, data_root):
    """ Reads in the files for evaluation of ground truth data

        Args:
            files: a list of files
            pred: numpy array (2D) of the predicted depth

        Returns:
            -abs_rel
            -sq_rel
            -rmse
            -rmse_log
            -a1
            -a2x
            -a3
        """


    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0

    for filename in files:
        filename = filename.split()[0]
        splits = filename.split('/')
        camera_id = np.int32(splits[2][-1:])  # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        file_root = '{}/{}'

        im = filename
        vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

        if os.path.isfile(data_root + im):
            gt_files.append(data_root + vel)
            gt_calib.append(data_root + date + '/')
            im_sizes.append(cv2.imread(data_root + im).shape[:2])
            im_files.append(data_root + im)
            cams.append(2)
        else:
            num_probs += 1
            print('{} missing'.format(data_root + im))
            
    print num_probs, 'files missing'

    return gt_files, gt_calib, im_sizes, im_files, cams




