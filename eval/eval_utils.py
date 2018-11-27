import numpy as np

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


def load_disparities(path):
    disps = np.load(path)
    return disps