# Script for postprocessing predictions
import sys
sys.path.append('../')

import argparse

import numpy as np

# Utilities import
from utils.misc import load_h5, save_h5
from utils.misc import load_pickle, save_pickle


def make_submission(images):



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat_path', type=str, help='path to data')
    parser.add_argument('--img_name', type=str, help='name of image file')
    parser.add_argument('--nme_name', type=str, help='name of name file')
    args = parser.parse_args()
    pass


if __name__=='__main__':
    main()
