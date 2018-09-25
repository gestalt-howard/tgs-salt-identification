# Script for loading trained model and making predictions
import argparse
import os
import pdb

import torch

# Models import
from models.res_seg_19 import ResSeg19, ResidualBlock
from models.res_seg_19 import ResSeg19_Reg, ResBlock_Reg
from models.res_seg_39 import ResSeg39
# Utilities import
from utils.misc import force_dir


def main():
    # Import settings (note that default debug settings are used)
    parser=argparse.ArgumentParser(description='TGS Challenge Test Script')
    parser.add_argument('--trn_path', type=str, default='./data/debug_train/',
                        help='path to training directory (default: debug)')
    parser.add_argument('--msk_path', type=str, default='./data/debug_masks',
                        help='path to mask directory (default: debug)')
    parser.add_argument('--tst_path', type=str, default='./data/debug_test/',
                        help='path to test directory (default: debug)')
    parser.add_argument('--mod_path', type=str, default='./weights/model_tmp/',
                        help='path to model weights directory (default: tmp)')
    parser.add_argument('--prd_path', type=str, default='preds/',
                        help='path to save test mask predictions')
    parser.add_argument('--val_path', type=str, default='validate/',
                        help='path to save validation visualizations')
    parser.add_argument('--mod_name', type=str,
                        help='name of epoch weights to load in given mod_path')
    args = parser.parse_args()

    # Define some variables relative to parser inputs
    trn_path = args.trn_path
    msk_path = args.msk_path
    tst_path = args.tst_path
    mod_path = args.mod_path
    prd_path = mod_path + args.prd_path
    val_path = mod_path + args.val_path
    mod_name = mod_path + args.mod_name

    # Aggressively check save paths
    force_dir(prd_path)
    force_dir(val_path)

    # Define model (comment out irrelvant models as necessary)
    net = ResSeg19(ResidualBlock)
    # net = ResSeg19_Reg(ResBlock_Reg)
    # net = ResSeg39(ResidualBlock, [3, 4, 6, 3])

    # Load model
    print 'Loading model:\n', mod_name



if __name__=='__main__':
    main()
