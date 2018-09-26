# Script containing validation visualization API and unit tests
import cv2
import argparse

import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

# Utilities import
from misc import force_dir
from misc import save_pickle, load_pickle
from misc import save_h5, load_h5
# Models import
from models.res_seg_19 import ResSeg19_Reg, ResBlock_Reg
# Dataset import
from datasets.tgs_dataset import data_formatter


def vis_validate(loader, net, device, dtype, path):
    """
    Function for visualizing network performance on one batch of validation
    data
    Input(s):
    - loader (PyTorch loader object): loader for queueing minibatches
    - net (module object): Pytorch network module object (set to eval mode)
    - device (PyTorch device)
    - dtype (PyTorch datatype)
    Output(s):
    [None]
    """
    mask_path = path + 'val_masks/'
    pred_path = path + 'val_preds/'

    force_dir(mask_path)
    force_dir(pred_path)

    with torch.no_grad():
        for (x, y, name) in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            # Make predictions
            scores = net(x)

            # Define ground-truth and predictions
            batch_masks = y.data.cpu().numpy()
            batch_preds = F.softmax(scores, dim=1
                ).data.cpu().numpy()[:, 1, :, :]
            break

    # Save data
    save_h5(data=batch_masks, path=mask_path+'mask_data.h5')
    save_h5(data=batch_preds, path=pred_path+'pred_data.h5')
    save_pickle(data=name, path=pred_path+'pred_name.pickle')
    # Save images
    for i, n in enumerate(name):
        # Save ground-truth mask
        cv2.imwrite(mask_path + n, batch_masks[i])
        # Save mask prediction
        cv2.imwrite(pred_path + n, batch_preds[i])

    return None


# Unit tests
def check_visualization(params):
    """Unit test for verifying visualization validation"""
    vis_validate(**params)


# Main function (unit tests)
def main():
    # Import settings (note that default debug settings are used)
    parser=argparse.ArgumentParser(description='TGS Challenge Test Script')
    parser.add_argument('--trn_path', type=str, default='../data/debug_train/',
                        help='path to training directory (default: debug)')
    parser.add_argument('--msk_path', type=str, default='../data/debug_masks',
                        help='path to mask directory (default: debug)')
    parser.add_argument('--tst_path', type=str, default='../data/debug_test/',
                        help='path to test directory (default: debug)')
    parser.add_argument('--mod_path', type=str, default='../weights/model_tst/',
                        help='path to model weights directory (default: tst)')
    parser.add_argument('--prd_path', type=str, default='preds/',
                        help='path to save test mask predictions')
    parser.add_argument('--val_path', type=str, default='validate/',
                        help='path to save validation visualizations')
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

    # Aggressively check save paths
    force_dir(prd_path)
    force_dir(val_path)

    # Define model
    net = ResSeg19_Reg(ResBlock_Reg)

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

    # Unit test
    params = {
        'loader': val_load,
        'net': net,
        'device': device,
        'dtype': dtype,
        'path': val_path
    }
    check_visualization(params)


if __name__=='__main__':
    main()
