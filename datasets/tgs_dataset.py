# Script containing dataset classes for TGS challenge
import sys
sys.path.append('../')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T

import cv2
import os
import pdb

import numpy as np
import matplotlib.pyplot as plt

# Utils import
from utils.misc import get_paths


# Inspired by tutorial:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Train_Dataset(Dataset):
    """TGS Salt Training and Validation Dataset"""

    def __init__(self, img_path, msk_path, img_trans=None, msk_trans=None):
        """
        Inputs:
        - img_path (string): path to directory containing images
        - msk_path (string): path to directory containing masks
        - img_trans (callable, optional): transform to be applied
            on an image sample
        - msk_trans (callable, optional): transform to be applied
            on a mask sample
        """
        # Define paths
        self.img_path = img_path
        self.msk_path = msk_path
        # Define transforms
        self.img_trans = img_trans
        self.msk_trans = msk_trans
        # Get file names
        self.img_names = os.listdir(img_path)
        self.msk_names = os.listdir(msk_path)
        assert np.array_equal(self.img_names, self.msk_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Image name
        img_name = os.path.join(self.img_path,
                                self.img_names[idx])
        # Mask name
        msk_name = os.path.join(self.msk_path,
                                self.msk_names[idx])
        # Get image
        image = cv2.imread(img_name, 0)
        image = image[:, :, np.newaxis]
        # Get mask
        mask = cv2.imread(msk_name, 0)
        mask = mask[:, :, np.newaxis]
        mask /= 255  # Normalize
        # Apply transforms if necessary
        if self.img_trans:
            image = self.img_trans(image)
        if self.msk_trans:
            mask = self.msk_trans(mask)
        return image, mask, self.img_names[idx]


class Test_Dataset(Dataset):
    """TGS Salt Test Dataset"""

    def __init__(self, img_path, transforms=None):
        """
        Inputs:
        - img_path (string): path to directory containing images
        - transforms (callable, optional): transform to be applied
            on an image sample
        """
        self.img_path = img_path
        self.transforms = transforms
        self.fnames = os.listdir(img_path)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path,
                                self.fnames[idx])
        image = cv2.imread(img_name, 0)
        image = image[:, :, np.newaxis]
        if self.transforms:
            image = self.transforms(image)
        return image, self.fnames[idx]


# Custom class for padding preprocessing operation
class Padder(object):
    """Pads numpy array with zeros"""

    def __init__(self, output_size, mask=False):
        assert isinstance(output_size, int), 'Wrong input dtype!'
        assert output_size%2==0, 'Size must be even number!'
        self.output_size = output_size
        self.mask = mask

    def __call__(self, sample):
        h, w, d = sample.shape
        pad_x = 13
        pad_y = 14

        placeholder = np.zeros((self.output_size, self.output_size, d))
        placeholder[pad_y:pad_y+h, pad_x:pad_x+w] = sample
        if self.mask:
            return placeholder.squeeze()
        else:
            return placeholder


# Custom class for transforming images in numpy array to PyTorch tensor
class Numpy_to_Tensor(object):
    """Converts from numpy to pytorch tensor"""

    def __call__(self, sample):
        return torch.from_numpy(sample)


# Data loader and formatter
def data_formatter(paths, stats):
    """
    Function for loading and preprocessing datasets
    Inputs:
    - paths (tuple): tuple of paths
        - trn_path (string): path to train images directory
        - msk_path (string): path to mask images directory
        - tst_path (string): path to test images directory
    - stats (tuple): tuple of stats
        - NUM_TRAIN (int)
        - NUM_FULL (int)
        - batch_size (int)
    Outputs:
    - (trn_data, trn_load): PyTorch dataset and loader for training set
    - (val_data, val_load): PyTorch dataset and loader for validation set
    - (tst_data, tst_load): PyTorch dataset and loader for test set
    """
    # Unpack tuples
    trn_path, msk_path, tst_path = paths
    NUM_TRAIN, NUM_FULL, batch_size = stats
    # Defined mean and std vals from computation
    mean_img = (119.83 for i in range(3))
    std_img = (41.58 for i in range(3))
    # Image preprocessing modules
    img_trans = T.Compose([
        Padder(128, mask=False),
        T.ToTensor(),
        T.Normalize(mean_img, std_img)
        ])
    msk_trans = T.Compose([
        Padder(128, mask=True),
        Numpy_to_Tensor()
        ])
    # Training set
    trn_data = Train_Dataset(trn_path, msk_path, img_trans, msk_trans)
    trn_load = DataLoader(
        trn_data,
        batch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
    )
    # Validation set
    val_data = Train_Dataset(trn_path, msk_path, img_trans, msk_trans)
    val_load = DataLoader(
        val_data,
        batch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_FULL))
    )
    # Test set
    tst_data = Test_Dataset(tst_path, img_trans)
    tst_load = DataLoader(tst_data, batch_size)

    return (trn_data, trn_load), (val_data, val_load), (tst_data, tst_load)


# Unit tests
def check_preprocessing(trn, tst):
    """Unit test for visualizing preprocessing"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes = axes.ravel()
    # Plot train image and mask
    i=0
    axes[i].imshow(trn[0].numpy().squeeze(), cmap='gray')
    axes[i].set_title('trn')
    axes[i].set_axis_off()
    i+=1
    axes[i].imshow(trn[1].numpy().squeeze(), cmap='gray')
    axes[i].set_title('msk')
    axes[i].set_axis_off()
    i+=1
    axes[i].imshow(tst[0].numpy().squeeze(), cmap='gray')
    axes[i].set_title('tst')
    axes[i].set_axis_off()
    plt.show()
    return None

def check_split_sizes(trn, val, NUM_FULL):
    """Unit test for validating split sizes and exclusivity"""
    # Parse through each loader
    def parse_loader(loader):
        img_names = []
        for batch in loader:
            img_names += batch[2]
        return np.array(img_names)
    trn_names = parse_loader(trn)
    val_names = parse_loader(val)
    assert len(trn_names)+len(val_names)==NUM_FULL, "Error with split"
    assert len(np.intersect1d(trn_names, val_names))==0, "Non-exclusive"
    return None

def check_img_dims(trn):
    """Unit test for validating image and mask sizes along with max mask val"""
    img = trn[0]
    msk = trn[1]
    assert img.size()==(1, 128, 128)
    assert msk.size()==(128, 128)
    assert torch.max(msk).item()==1.


# Main function (unit tests)
def main():
    # Load data
    paths, stats = get_paths(debug=True)
    trn_path, msk_path, tst_path = paths
    NUM_TRAIN, NUM_FULL, batch_size = stats

    trn_set, val_set, tst_set = data_formatter(paths, stats)
    # Unpack sets
    trn_data, trn_load = trn_set
    val_data, val_load = val_set
    tst_data, tst_load = tst_set

    # Validate preprocessing
    trn_idx = np.random.randint(len(trn_data))
    tst_idx = np.random.randint(len(tst_data))
    # Unit test
    check_split_sizes(trn_load, val_load, NUM_FULL)
    # Unit test
    check_preprocessing(trn_data[trn_idx], tst_data[tst_idx])
    # Unit test
    check_img_dims(trn_data[trn_idx])


if __name__=='__main__':
    main()
