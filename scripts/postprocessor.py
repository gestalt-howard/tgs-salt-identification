# Script for postprocessing predictions
import sys
sys.path.append('../')

import argparse
import pdb

import numpy as np
import pandas as pd

# Utilities import
from utils.misc import crop_padding
from utils.misc import load_h5, save_h5
from utils.misc import load_pickle, save_pickle


def make_submission(images, names, uni):
    """
    Function for making submission from images and names
    Input(s):
    - images (numpy ndarray): numpy array containing images of mask predictions
    - names (list): list containing image names
    - uni (bool): flag indicating to turn on/off unit tests
    Output(s):
    - submit_df (pandas df): dataframe containing submission-ready data
    """

    def transform_to_enc(img):
        """Helper function to transform image into list encoding"""
        enc_mask = []
        if not uni: # Normal operation
            img = crop_padding(img)
            assert img.shape==(101, 101)

        assert len(np.unique(img))<=2

        # Look for activated pixels
        pixels = img.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        return ' '.join(str(x) for x in runs)

    # Iterate over each prediction image and get mask from image
    masks = []
    for img in images:
        e_mask = transform_to_enc(img)
        assert type(e_mask)==str
        masks.append(e_mask)

    save_names = [n.split('.png')[0] for n in names]
    submit_dict = {}
    submit_dict['id'] = save_names
    submit_dict['rle_mask'] = masks

    submit_df = pd.DataFrame.from_dict(submit_dict, orient='columns')

    return submit_df


def main():
    # Import settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--uni_flag', type=int, default=1, help='unit tst flg')
    parser.add_argument('--dat_path', type=str, default=None,
                        help='path to data directory')
    parser.add_argument('--img_name', type=str, default=None,
                        help='name of prediction file containing images')
    parser.add_argument('--nme_name', type=str, default=None,
                        help='name of name file corresponding to img preds')
    parser.add_argument('--sub_name', type=str, default=None,
                        help='name of submission file')
    parser.add_argument('--thres', type=float, default=0.5,
                        help='activation thresholding to transform SM vals')
    args = parser.parse_args()

    # Define some variables relative to parser inputs
    data_path = args.dat_path
    imgs_path = data_path + args.img_name
    name_path = data_path + args.nme_name
    subm_path = data_path + args.sub_name

    uni_flag = bool(args.uni_flag)

    # Load data
    if uni_flag: # Unit test
        names = ['sample_1', 'sample_2', 'sample_3', 'sample_4']
        sample_1 = np.array([[0, 1, 1, 0],
                             [0, 0, 1, 0],
                             [1, 1, 1, 1],
                             [0, 0, 0, 1]])
        sample_2 = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        sample_3 = np.array([[1, 1, 1, 1],
                             [1, 0, 1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0]])
        sample_4 = np.array([[1, 1, 1, 1],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0],
                             [1, 1, 1, 0]])
        images = np.stack((sample_1, sample_2, sample_3, sample_4), axis=0)
    else: # Normal operation
        images = load_h5(imgs_path)
        names = load_pickle(name_path)

    # Transform data
    thresholded_images = np.uint8(images>args.thres)
    assert len(thresholded_images.shape)==3

    # Make submissions
    df = make_submission(thresholded_images, names, uni_flag)

    if uni_flag: # Unit test
        assert np.array_equal(df['id'].values, names)
        assert df.loc[0]['rle_mask']=='3 1 5 1 7 1 9 3 15 2', 'Sample 1'
        assert df.loc[1]['rle_mask']=='', 'Sample 2'
        assert df.loc[2]['rle_mask']=='1 3 5 1 9 2 12 2', 'Sample 3'
        assert df.loc[3]['rle_mask']=='1 1 3 3 8 3 12 2', 'Sample 4'
    else:
        df.to_csv(subm_path, index=False)

    return None


if __name__=='__main__':
    main()
