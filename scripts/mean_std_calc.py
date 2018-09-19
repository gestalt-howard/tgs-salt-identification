# Script for calculating the standard deviation and mean of images
import sys
sys.path.append('../')


from utils.misc import data_loader, get_paths

import pdb
import gc
import numpy as np


def find_mean_std(data):
    """
    Function that calculates mean and standard deviation of data set
    Outputs:
    - mean_img: mean of all images in dataset
    - mean_std: standard deviation of all images in dataset
    """
    data_flat = data.reshape((data.shape[0], -1))
    mean_img = data_flat.mean()
    std_img = data_flat.std()
    return mean_img, std_img


def main():
    paths, _ = get_paths(debug=False)
    trn_path, msk_path, tst_path = paths
    # Load images
    trn_imgs, _ = data_loader(trn_path)
    tst_imgs, _ = data_loader(tst_path)
    all_imgs = np.concatenate([trn_imgs, tst_imgs], axis=0)
    mean, std = find_mean_std(all_imgs)
    print 'Mean of images:', mean
    print 'Standard deviation of images:', std
    # Clear memory
    del trn_imgs, tst_imgs, all_imgs
    gc.collect()


if __name__=='__main__':
    main()
