import torch
import torchvision.transforms as transforms
import numpy as np


def image_transforms(
    mode="train",
    augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
    do_augmentation=True,
    transformations=None,
    size=(256, 512),
):
    """

    Args:
        mode: "train", "test" or "custom"
        augment_parameters: min and max parameters for the data augmentation (gamma, brightness color)
        do_augmentation: augmentation on or off
        transformations: torchvision.transform, only used if mode=="custom"
        size: image dimensions (nx, ny)

    Returns:
        torchvision.transforms transform
    """
    if mode == "train":
        data_transform = transforms.Compose(
            [
                ResizeImage(train=True, size=size),
                RandomFlip(do_augmentation),
                ToTensor(train=True),
                AugmentImagePair(augment_parameters, do_augmentation),
            ]
        )
        return data_transform
    elif mode == "test":
        data_transform = transforms.Compose(
            [ResizeImage(train=False, size=size), ToTensor(train=False), DoTest()]
        )
        return data_transform
    elif mode == "custom":
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print("Wrong mode")


class ResizeImage(object):
    """ Apply torchvision.transforms.Resize() to image (when train=False) or dict of left and right image (train=True)
    """

    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            left_image = sample["left_image"]
            right_image = sample["right_image"]
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {"left_image": new_left_image, "right_image": new_right_image}
        else:
            left_image = sample
            new_left_image = self.transform(left_image)
            sample = new_left_image
        return sample


class DoTest(object):
    """ For testing, we need the left image and the flipped left image (for post-processing).
        This transform stacks an image together with its flipped version.
    """

    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class ToTensor(object):
    """ Apply torchvision.transforms.ToTensor() to image (when train=False) or dict of left and right image (train=True)
    """
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            left_image = sample["left_image"]
            right_image = sample["right_image"]
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {"left_image": new_left_image, "right_image": new_right_image}
        else:
            left_image = sample
            sample = self.transform(left_image)
        return sample


class RandomFlip(object):
    """ Randomly flip an image pair
    """
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample["left_image"]
        right_image = sample["right_image"]
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(right_image)
                fliped_right = self.transform(left_image)
                sample = {"left_image": fliped_left, "right_image": fliped_right}
        else:
            sample = {"left_image": left_image, "right_image": right_image}
        return sample


class AugmentImagePair(object):
    """ Augment an image pair (left and right image) by applying gamma, brightness and color shifts
        with parameters given by augment_parameters.
    """
    def __init__(self, augment_parameters, do_augmentation):
        """

        Args:
            augment_parameters: list [gamma_low, gamma_high, brightness_low, brightness_high, color_low, color_high]
            do_augmentation: boolean, decides wether any augmentation is done
        """
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        left_image = sample["left_image"]
        right_image = sample["right_image"]
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_image ** random_gamma
                right_image_aug = right_image ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(
                    self.brightness_low, self.brightness_high
                )
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # saturate
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {"left_image": left_image_aug, "right_image": right_image_aug}

        else:
            sample = {"left_image": left_image, "right_image": right_image}
        return sample
