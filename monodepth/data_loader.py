import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .transforms import image_transforms


class KittiLoader(Dataset):
    """ DataSet that reads a single Kitti sequence.
        Can be accessed like a list.
        If transform is specified, the transform is applied before returning an element.
        If mode='train', each element is a dict containing 'left_image' and 'right_image'
    """

    def __init__(self, root_dir, filenames_file, mode, transform=None):
        """ Setup a Kitti sequence dataset.

        Args:
            root_dir: data directory. should contain 'image_02' (left) and 'image_03' (right) subfolders
            mode: 'train' or 'test'
            transform: a torchvision.transforms type transform
        """

        with open(filenames_file) as filenames:
            self.left_paths = sorted(os.path.join(root_dir,fname.split()[0]) for fname \
                                  in filenames)

        if mode == 'train':
              with open(filenames_file) as filenames:
                self.right_paths = sorted(os.path.join(root_dir,fname.split()[1]) for fname \
                                  in filenames)
            
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


def prepare_dataloader(root_dir,filenames_file, mode, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                       do_augmentation=True, batch_size=256, size=(256, 512), num_workers=1):
    """ Prepares a DataLoader that loads multiple Kitti sequences
    
    Args:
    
        root_dir: (str) folder that contains multiple subfolders with Kitti sequences
        mode: (str) 'test' or 'train'
        augment_parameters: list of parameters for the data augmentation
        do_augmentation: decides if data are augmented
        batch_size: number of images per batch
        num_workers: number of GPUs
        
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
        size=size)



    dataset = KittiLoader(root_dir, filenames_file, mode, transform=data_transform)

    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return n_img, loader
