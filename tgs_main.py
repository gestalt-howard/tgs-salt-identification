# Main script for TGS Kaggle challenge
import argparse
import os
import pdb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T

# Utilities import

# Model import
from models.res_seg_19 import conv3x3, ResidualBlock, ResSeg19
# Dataset import
from datasets.tgs_dataset import data_formatter


# Main function
def main():
    # Import settings (note that default debug settings are used)
    parser=argparse.ArgumentParser(description='TGS Challenge Main Script')
    parser.add_argument('--trn_path', type=str, default='./data/debug_train/',
                        help='path to training directory (default: debug)')
    parser.add_argument('--msk_path', type=str, default='./data/debug_masks',
                        help='path to mask directory (default: debug)')
    parser.add_argument('--tst_path', type=str, default='./data/debug_test/',
                        help='path to test directory (default: debug)')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='input batch size (default: 3)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='num epochs to wait for LR reduce (default: 10)')
    parser.add_argument('--print_every', type=int, default=1,
                        help='num batches before printing (default: 1)')
    parser.add_argument('--NUM_TRAIN', type=int, default=6,
                        help='num samples in split train set (default: 6)')
    parser.add_argument('--NUM_FULL', type=int, default=9,
                        help='num samples in full train set (default: 9)')
    args = parser.parse_args()

    # Define some variables relative to parser inputs
    trn_path = args.trn_path
    msk_path = args.msk_path
    tst_path = args.tst_path
    NUM_TRAIN = args.NUM_TRAIN
    NUM_FULL = args.NUM_FULL

    # Define device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Current epoch information
    curr_epoch = 1

    # Define model
    net = ResSeg19(ResidualBlock)

    # Load data
    paths = (tnr_path, msk_path, tst_path)
    stats = (NUM_TRAIN, NUM_FULL, args.batch_size)
    trn_set, msk_set, tst_set = data_formatter(paths, stats)
    # Unpack data
    trn_data, trn_load = trn_set
    msk_data, msk_load = msk_set
    tst_data, tst_load = tst_set

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Define automatic LR reduction scheduler
    scheduler = optim.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=args.lr_patience,
        min_lr=1e-10
    )
    # Model API parameters
    param_dict = {
        'loader': loader,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'epoch': 1,
        'args': args,
        'device': device,
        'dtype': dtype
    }
    # Note: epoch starts from 1, not 0
    for epoch in range(curr_epoch, args.epochs+1):
        param_dict['epoch'] = epoch
        trn_log = train(**param_dict)
        val_loss = validate()
        scheduler.step(val_loss)


if __name__=='__main__':
    main()
