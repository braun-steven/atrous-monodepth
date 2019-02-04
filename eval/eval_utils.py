import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_disparities(path):
    disps = np.load(path)
    return disps


def results_to_csv_str(res, res_pp):
    """
    Convert the results and pp results to a csv string
    Args:
        res: Results
        res_pp: Postprocessed results

    Returns:
        CSV table of results
    """
    if res is None:
        return ""

    s = "{:>5}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(
        "pp", "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"
    )
    s += f"false, {res.as_csv_line()}\n"
    s += f"true , {res_pp.as_csv_line()}\n"
    return s


class Result:
    def __init__(
        self,
        abs_rel: np.ndarray,
        sq_rel: np.ndarray,
        rms: np.ndarray,
        log_rms: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        a3: np.ndarray,
    ):
        """
        Evaluation Results.
        """
        self.abs_rel = abs_rel
        self.sq_rel = sq_rel
        self.rms = rms
        self.log_rms = log_rms
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def log(self):
        logger.info(
            "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
                "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"
            )
        )
        logger.info(
            "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
                self.abs_rel.mean(),
                self.sq_rel.mean(),
                self.rms.mean(),
                self.log_rms.mean(),
                self.a1.mean(),
                self.a2.mean(),
                self.a3.mean(),
            )
        )

    def as_csv_line(self):
        return "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            self.abs_rel.mean(),
            self.sq_rel.mean(),
            self.rms.mean(),
            self.log_rms.mean(),
            self.a1.mean(),
            self.a2.mean(),
            self.a3.mean(),
        )

    def values(self):
        """ results as a list

        Returns: a list containing abs_rel, sq_rel, rms, log_rms, a1, a2, a3

        """
        return [
            self.abs_rel,
            self.sq_rel,
            self.rms,
            self.log_rms,
            self.a1,
            self.a2,
            self.a3,
        ]
