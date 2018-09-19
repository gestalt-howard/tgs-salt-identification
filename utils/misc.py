# Script containing utility functions and classes
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2

import numpy as np


def data_loader(path):
    """
    Function for loading images from given directory
    Input:
    - path (string): directory where images are contained
    Outputs:
    - images: loaded images
    - img_names: names of loaded images
    """
    h = 101; w = 101
    fnames = os.listdir(path)
    images = np.zeros((len(fnames), 1, h, w))
    img_names = []
    for i, name in enumerate(fnames):
        images[i, :, :, :] = cv2.imread(path+name, 0)
        img_names.append(name.split('.')[0])
    return images, img_names


def get_paths(debug):
    """
    Function for defining paths to training images, training masks, and test
    images
    Input:
    - debug (bool): flag for indicating whether or not debug sets are desired
    Outputs:
    - trn_path: path to training images
    - msk_path: path to training masks
    - tst_path: path to test images
    - trn_stat (tuple): (NUM_TRAIN, NUM_FULL, batch_size)
    """
    if debug:
        trn_path = '../data/debug_train/'
        msk_path = '../data/debug_masks/'
        tst_path = '../data/debug_test/'
        paths = (trn_path, msk_path, tst_path)
        trn_stat = (6, 9, 3)
        return paths, trn_stat
    else:
        trn_path = '../data/train_images/'
        msk_path = '../data/train_masks/'
        tst_path = '../data/test_images/'
        paths = (trn_path, msk_path, tst_path)
        trn_stat = (3600, 4000, 64)
        return paths, trn_stat


class AverageMeter(object):
    """Running average tracker"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


class Simple_Network(nn.Module):
    """Simple network module for debugging"""

    def __init__(self):
        super(Simple_Network, self).__init__()
        self.num_maps = 8
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=self.num_maps,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(self.num_maps)
        self.relu = nn.ReLU(inplace=True)
        self.trans_conv = nn.ConvTranspose2d(in_channels = self.num_maps,
                                             out_channels = 2,
                                             kernel_size = 3,
                                             stride = 2,
                                             padding = 1,
                                             output_padding = 1,
                                             bias = False)
        # Initialize weights
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_normal_(self.trans_conv.weight)

    def forward(self, x):
        """Forward structure for network"""

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.trans_conv(x)
        return x


# Custom cross-entropy loss class
class CrossEntropyLoss2d(nn.Module):
    """Custom cross-entropy loss function for 2D images"""

    def __init__(self, dim=1):
        super(CrossEntropyLoss2d, self).__init__()
        self.dim = dim
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=self.dim), targets)
