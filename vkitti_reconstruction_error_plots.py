import os
import cv2
import argparse

from PIL import Image, ImageFont, ImageDraw
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.interpolate import LinearNDInterpolator

toPILImage = transforms.ToPILImage()
toTensor = transforms.ToTensor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch Monolab: Monodepth x " "DeepLabv3+"
    )

    parser.add_argument(
        "--data-dir",
        default="/visinf/projects_students/monolab/data/vkitti",
        help="path to the dataset folder. \
                        The filenames given in filenames_file \
                        are relative to this path.",
        metavar="DIR",
    )
    parser.add_argument(
        "--results-dir", default="results", help="path to the results directory"
    )
    parser.add_argument(
        "--filenames-file",
        default="resources/filenames/vkitti_test_files.txt",
        help="Filenames file (png test images)",
    )
    return parser.parse_args()


def load_gt_depth_vkitti(filenames_file, root_dir):
    """
    Loads in the ground truth files from the KITTI Stereo Dataset

    Args:

    Returns:
     -gt_depths (list): list of ground truth depth maps
     -gt_depths (list): list of ground truth depth maps
    """
    # load image paths
    with open(filenames_file) as filenames:
        image_paths = sorted(
            os.path.join(root_dir, fname.split()[0]) for fname in filenames
        )

    # convert to depth paths
    depth_paths = [
        "/".join(
            [
                path if path != "vkitti_1.3.1_rgb" else "vkitti_1.3.1_depthgt"
                for path in str.split(image_path, "/")
            ]
        )
        for image_path in image_paths
    ]

    gt_depths = []
    for path in depth_paths:
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100

        gt_depths.append(depth)
    return gt_depths


def load_vkitti_image(path, index, image_paths, resize=True, asTensor=False):
    """
    Loads in the images from the KITTI Stereo Dataset

    Args:
        -path (string): path to the the training files
        -resize (string): left or right camera

    Returns:
        -images (list): list of images
    """

    size = (256, 512)

    left_image = Image.open(os.path.join(path, image_paths[index]))

    if resize:
        resize = transforms.Resize(size)
        left_image = resize(left_image)

    if asTensor:
        toTensor = transforms.ToTensor()
        left_image = toTensor(left_image)

    return left_image


def load_pred_disp(path, resize=False):
    """

    """

    disp = np.load(path)
    height, width = (375, 1242)

    if resize:
        resized_disp = []
        for i in range(disp.shape[0]):
            resized_disp.append(
                width
                * cv2.resize(disp[i], (width, height), interpolation=cv2.INTER_LINEAR)
            )
        disp = resized_disp

    return disp


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def lin_interp(image):
    """ Linearly interpolates the depth data to fill zero holes - adapted from https://github.com/hunse/kitti

    Args:
        -shape(tuple): size of the image
        -xyd (np array):

    Returns:
        -points (2D): points
    """

    shape = image.shape
    m, n = shape
    ij = np.matrix(np.nonzero(image)).T
    d = image[np.nonzero(image)]

    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity_interpolated = f(IJ).reshape(shape)

    return disparity_interpolated


def plot_disparity_map(disp, showFigure=True):
    ymax, xmax = disp.shape
    dpi = ymax
    # fig = plt.figure(figsize=(8, 4), frameon=False)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(disp, aspect="auto", cmap="plasma")

    if showFigure:
        plt.show()

    return fig


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def convert_disps_to_depths_kitti(pred_disparities, gt_depths):
    """
        Converts the predictions to depth values

        Args:
         -pred_disparities (list): predicted disparities
         -gt_depths (list): ground truth disparities (only needed for proper sizes)

        Returns:
         -pred_depths (list): list of predicted depths
         -pred_depths_resized (list): list of predicted depths, resized to ground truth disparity size

        """
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1238] = 718.3351

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

        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        pred_depths.append(pred_depth)

    return pred_depths, pred_disparities_resized


def apply_disparity(img, disp):
    """ Applies a disparity map to an image.

    Args:
        img: (n_batch, n_dim, nx, ny) input image
        disp: (n_batch, 1, nx, ny) disparity map to be applied

    Returns:
        the input image shifted by the disparity map (n_batch, n_dim, nx, ny)
    """

    width, height = img.size
    img = transforms.ToTensor()(img).unsqueeze(0)
    disp = torch.tensor(disp)

    device = torch.device("cpu")

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(1, height, 1).type_as(img).to(device)
    y_base = (
        torch.linspace(0, 1, height).repeat(1, width, 1).transpose(1, 2).type_as(img)
    ).to(device)

    # Apply shift in X direction
    x_shifts = disp[:, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(
        img, 2 * flow_field - 1, mode="bilinear", padding_mode="zeros"
    )

    return output.squeeze()


def evaluate_experiment(
    experiment_name, gt_depth, results_dir, data_dir, filenames_file
):
    experiment_folder = os.path.join(results_dir, "vkitti/{}/".format(experiment_name))
    files = [f for f in os.listdir(experiment_folder) if not f.startswith(".")]
    num_experiments = len(files)

    with open(filenames_file) as filenames:
        image_paths = sorted(
            os.path.join(data_dir, fname.split()[0]) for fname in filenames
        )

    experiments = {}
    for experiment in files:
        path = experiment_folder + "{}/test/disparities.npy".format(experiment)
        experiments[experiment] = load_pred_disp(path, resize=True)

    width = 1242
    total_height = 375 * (num_experiments + 1)
    output_dir = os.path.join("results", "report_disparity_error", experiment_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    try:
        fnt = ImageFont.truetype("/Library/Fonts/Arial.ttf", 15)
    except OSError:
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf", 15
        )

    for i in range(2126):
        # load image and ground truth
        img = load_vkitti_image(
            path=data_dir, index=i, resize=False, image_paths=image_paths
        )
        gt = gt_depth[i]

        new_im = Image.new("RGB", (width, total_height))
        new_im.paste(img, (0, 0))

        for j, experiment in enumerate(experiments):
            description = experiment_name + ": " + experiment
            pred_disp = experiments[experiment][i]

            pred = convert_disps_to_depths_kitti(pred_disp, gt)

            # skip those that are not of the same size as the prediction
            if gt.shape[0] != 375:
                print(gt.shape)
                print(pred.shape)
                continue

                # get the sparse pixels and calculate difference
            # mask = gt > 0
            # pred[np.logical_not(mask)] = 0
            diff = np.abs(gt - pred)
            diff = diff / np.max(diff)

            fig = Image.fromarray(np.uint8(cm.plasma(diff) * 255))
            # create figure and convert to PIL image
            # fig = plot_disparity_map(diff, showFigure=False)
            # fig = fig2img(fig)

            # label the image
            d = ImageDraw.Draw(fig)
            d.text((10, 350), description, font=fnt, fill=(255, 255, 255))

            new_im.paste(fig, (0, (j + 1) * 375))

        concat_outfile = "concat_{}.png".format(str(i).zfill(3))
        concat_outfile = os.path.join(output_dir, concat_outfile)
        new_im.save(concat_outfile)


if __name__ == "__main__":
    args = parse_args()
    gt_depth = load_gt_depth_vkitti(
        filenames_file=args.filenames_file, root_dir=args.data_dir
    )
    evaluate_experiment(
        "aspp-rates",
        gt_depth=gt_depth,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        filenames_file=args.filenames_file,
    )
