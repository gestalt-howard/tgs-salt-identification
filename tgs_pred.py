# Script for loading trained model and making predictions
import argparse
import os
import pdb

import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

# Models import
from models.res_seg_33 import ResSeg33, ResidualBlock
from models.res_seg_33 import ResSeg33_Reg, ResBlock_Reg
from models.res_seg_var import ResSegVar
# Utilities import
from utils.misc import force_dir
from utils.misc import save_pickle, load_pickle
from utils.misc import save_h5, load_h5
from utils.vis_validate import vis_validate
# Dataset import
from datasets.tgs_dataset import data_formatter


def main():
    # Import settings (note that default debug settings are used)
    parser=argparse.ArgumentParser(description='TGS Challenge Test Script')
    parser.add_argument('--trn_path', type=str, default='./data/debug_train/',
                        help='path to training directory (default: debug)')
    parser.add_argument('--msk_path', type=str, default='./data/debug_masks',
                        help='path to mask directory (default: debug)')
    parser.add_argument('--tst_path', type=str, default='./data/debug_test/',
                        help='path to test directory (default: debug)')
    parser.add_argument('--mod_path', type=str, default='./weights/model_tst/',
                        help='path to model weights directory (default: tst)')
    parser.add_argument('--prd_path', type=str, default='preds/',
                        help='path to save test mask predictions')
    parser.add_argument('--val_path', type=str, default='validate/',
                        help='path to save validation visualizations')
    parser.add_argument('--val_flag', type=int, default=1,
                        help='Flag for turning val visualization on/off')
    parser.add_argument('--prd_flag', type=int, default=0,
                        help='Flag for turning test prediction on/off')
    parser.add_argument('--mod_name', type=str, default='epoch_10.pth',
                        help='name of epoch weights to load in given mod_path')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='input batch size (default: 3)')
    parser.add_argument('--NUM_TRAIN', type=int, default=6,
                        help='num samples in split train set (default: 6)')
    parser.add_argument('--NUM_FULL', type=int, default=9,
                        help='num samples in full train set (default: 9)')
    args = parser.parse_args()

    # Define some variables relative to parser inputs
    trn_path = args.trn_path
    msk_path = args.msk_path
    tst_path = args.tst_path
    mod_path = args.mod_path
    prd_path = mod_path + args.prd_path
    val_path = mod_path + args.val_path
    mod_name = mod_path + args.mod_name
    NUM_TRAIN = args.NUM_TRAIN
    NUM_FULL = args.NUM_FULL

    val_flag = bool(args.val_flag)
    prd_flag = bool(args.prd_flag)

    # Define model (comment out irrelvant models as necessary)
    net = ResSeg33(ResidualBlock)
    # net = ResSeg33_Reg(ResBlock_Reg)
    # net = ResSegVar(ResidualBlock, [3, 4, 6, 3])

    # Define device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    # Parallelization init and set net to CUDA if possible
    if torch.cuda.is_available():
        net.cuda()
        net=torch.nn.DataParallel(net, device_ids=range(
            torch.cuda.device_count()))

    # Load model weights
    print 'Loading model:\n', mod_name
    if not torch.cuda.is_available():
        net.load_state_dict(torch.load(mod_name, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(mod_name))
    net.eval()

    # Load data
    paths = (trn_path, msk_path, tst_path)
    stats = (NUM_TRAIN, NUM_FULL, args.batch_size)
    trn_set, val_set, tst_set = data_formatter(paths, stats)
    # Unpack data
    trn_data, trn_load = trn_set
    val_data, val_load = val_set
    tst_data, tst_load = tst_set

    # Validation visualization option
    if val_flag:
        print 'Creating validation visualizations...'
        force_dir(val_path)
        vis_validate(val_load, net, device, dtype, val_path)

    # Test predictions
    if prd_flag:
        print 'Creating test predictions...'
        force_dir(prd_path)
        with torch.no_grad():
            preds_list = []
            names_list = []
            for i, (x, name) in enumerate(tst_load):
                # Get predictions
                x = x.to(device=device, dtype=dtype)
                scores = net(x)
                # Define predictions
                batch_preds = F.softmax(scores, dim=1
                    ).data.cpu().numpy()[:, 1, :, :]
                # Assemble save ingredients
                preds_list.append(batch_preds.squeeze())
                names_list += name
                # Print status
                print 'Batch %d / %d complete'%(i+1, len(tst_load))

            # Aggregate batch data
            preds = np.concatenate(preds_list)
            assert len(names_list)==preds.shape[0]

        # Save aggregated data
        save_h5(data=preds, path=prd_path+'pred_data.h5')
        save_pickle(data=names_list, path=prd_path+'pred_name.pickle')

    return None


if __name__=='__main__':
    main()
