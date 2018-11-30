import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from .transforms import image_transforms, crop_cityscapes


class ImageLoader(Dataset):
    """ DataSet that reads a single Kitti sequence.
        Can be accessed like a list.
        If transform is specified, the transform is applied before returning an element.
        If mode='train', each element is a dict containing 'left_image' and 'right_image'
    """

    def __init__(self, root_dir, filenames_file, mode, transform=None, dataset = 'kitti'):
        """ Setup a Kitti sequence dataset.

        Args:
            root_dir: data directory
            filenames_file: file, where each line contains left and right image paths (separated by whitespace)
            mode: 'train' or 'test'
            transform: a torchvision.transforms type transform
        """

        with open(filenames_file) as filenames:
            self.left_paths = sorted(
                os.path.join(root_dir, fname.split()[0]) for fname in filenames
            )

        if mode == "train" or mode == "val":
            with open(filenames_file) as filenames:
                self.right_paths = sorted(
                    os.path.join(root_dir, fname.split()[1]) for fname in filenames
                )

        self.transform = transform
        self.mode = mode
        self.dataset = dataset


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):


        if self.dataset == 'kitti':
            left_image = Image.open(self.left_paths[idx])

        if self.dataset == 'cityscapes':
            left_image = Image.open('left/' + self.left_paths[idx])
            left_image = crop_cityscapes(left_image)


        if self.mode == "train" or self.mode == "val":

            if self.dataset == 'kitti':
                right_image = Image.open(self.right_paths[idx])

            if self.dataset == 'cityscapes':
                right_image = Image.open('right/' + self.right_paths[idx])
                right_image = crop_cityscapes(right_image)

            sample = {"left_image": left_image, "right_image": right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


def prepare_dataloader(
    root_dir,
    filenames_file,
    mode,
    augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
    do_augmentation=False,
    shuffle=False,
    batch_size=256,
    size=(256, 512),
    num_workers=1,
    dataset = 'kitti'
):
    """ Prepares a DataLoader that loads Kitti images from file names and performs transforms

    Args:

        root_dir: data directory
        filenames_file: file, where each line contains left and right image paths (separated by whitespace)
        mode: "train", "val" or "test"
                "train": do data augmentation and resize
                "val": resize
                "test": resize and stack the image with its flipped version (for post-processing)
        augment_parameters: list of parameters for the data augmentation (only needed when mode="train"
                and do_augmentation=True)
        do_augmentation: decides if data are augmented (only effective when mode="train")
        shuffle: (bool) shuffle the dataset?
        batch_size: number of images per batch
        num_workers: number of workers in the data loader

    Returns:
        n_img : int
            total number of images

        loader : torch.utils.data.DataLoader
            data loader
    """

    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size=size,
    )

    image_data_set = ImageLoader(root_dir, filenames_file, mode=mode, transform=data_transform, dataset = dataset)

    n_img = len(image_data_set)

    loader = DataLoader(
        image_data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return n_img, loader
