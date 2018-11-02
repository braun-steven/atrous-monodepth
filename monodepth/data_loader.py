import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transforms import image_transforms


class KittiLoader(Dataset):
    """ DataSet that reads a single Kitti sequence
    """
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
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

        
def prepare_dataloader(data_directory, mode, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                       do_augmentation=True, batch_size=256, size=(256, 512), num_workers=1):
    """ Prepares a DataLoader that loads multiple Kitti sequences
    
    Parameters
    ----------
    
    data_directory : str
        folder that contains multiple subfolders with Kitti sequences
        
    mode : str
        'test' or 'train'
    
    augment_parameters : list
        parameters for the data augmentation
    
    do_augmentation : bool
        decides if data are augmented
    
    batch_size : int
        number of images per batch
        
    num_workers : int
        number of GPUs
        
    Returns
    _______
    
    n_img : int
        total number of images
        
    loader : torch.utils.data.DataLoader
        data loader
    """
    data_dirs = os.listdir(data_directory)
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size = size)
    datasets = [KittiLoader(os.path.join(data_directory,
                            data_dir), mode, transform=data_transform)
                            for data_dir in data_dirs]
    dataset = ConcatDataset(datasets)
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