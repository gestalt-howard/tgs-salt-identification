# Script containing evaluation metric API and unit tests
import sys
sys.path.append('../')

import cv2
import pdb
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch

# Utils import
from misc import get_paths, crop_padding
# Datasets imports
from datasets.tgs_dataset import Padder


def evaluate(preds, masks):
    """
    Function for evaluating quality of predictions using an intersection over
    union metric (IoU)
    Input(s):
    - preds (numpy array): collection of predicted masks (shape of each image
        should be 128 x 128)
    - masks (numpy array): collection of ground-truth masks
    Output(s):
    -
    """
    N = preds.shape[0]
    preds.astype(np.uint8)
    masks.astype(np.uint8)
    assert preds.shape==masks.shape
    cropped_preds = crop_padding(preds)
    cropped_masks = crop_padding(masks)
    assert cropped_preds.shape[1]==cropped_preds.shape[2]

    def get_iou(p, m):
        """Helper function for getting iou vector"""
        intersection = np.logical_and(p, m)
        union = np.logical_or(p, m)
        iou = np.sum(intersection>0)/np.sum(union>0)
        vec = pd.Series(name='iou')
        for threshold in np.arange(0.5, 1, 0.05):
            vec[threshold] = iou>threshold
        return vec

    # Get iou series and calculate running sums
    running_iou = 0
    for i in range(N):
        iou_vec = get_iou(cropped_preds[i], cropped_masks[i])
        running_iou += iou_vec.mean()
    running_iou /= N

    return running_iou


# Unit tests
def check_evaluate(preds, masks):
    """Unit test for verifying that evaluate function works"""
    assert evaluate(preds, masks)!=None

def check_perfect_input(masks):
    """Unit test for verifying result of 100% accurate prediction"""
    assert evaluate(masks, masks)==1.


# Main function (unit tests)
def main():
    paths, stats = get_paths(debug=True)
    trn_path, msk_path, tst_path = paths
    NUM_TRAIN, NUM_FULL, batch_size = stats

    # Define unit test samples
    padder = Padder(128)
    idx_list = [3, 4]
    msk_list = []
    for idx in idx_list:
        msk_fname = os.listdir(msk_path)[idx]
        msk_sample = cv2.imread(msk_path+msk_fname, 0)/255
        msk_sample = padder(msk_sample[:, :, np.newaxis]).squeeze()
        msk_list.append(msk_sample)
    msk_samples = np.stack(msk_list, axis=0)

    # Define dummy predictions
    msk_dummies = np.ones_like(msk_samples, dtype=np.uint8)

    # Unit test
    check_evaluate(msk_dummies, msk_samples)
    # Unit test
    check_perfect_input(msk_samples)


if __name__=='__main__':
    main()
