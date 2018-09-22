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
from utils.misc import load_pickle, save_pickle
from utils.misc import check_dir
from utils.misc import Simple_Network
from utils.train import train
from utils.validate import validate
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
    parser.add_argument('--mod_path', type=str, default='./weights/model_tmp/',
                        help='path to model weights directory (default: tmp)')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='input batch size (default: 3)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--starting_epoch', type=int, default=1,
                        help='index of starting epoch (default: 1)')
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
    mod_path = args.mod_path
    starting_epoch = args.starting_epoch
    NUM_TRAIN = args.NUM_TRAIN
    NUM_FULL = args.NUM_FULL

    record_name = 'best_record.pickle'
    history_name = 'training_history.pickle'

    # Validate specified model path
    restart_token = check_dir(mod_path)  # Returns None if path exists

    # Define model
    net = ResSeg19(ResidualBlock)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Define or load training history
    def format_epoch_fname(start_num):
        return mod_path + 'epoch_%s.pth'%start_num

    best_record = {}
    training_history = {}
    if restart_token:  # Starting from scratch
        curr_epoch = 1
        best_record['epoch'] = 0
        best_record['val_loss'] = 1e10
        best_record['mean_iou'] = 0
    else:
        print 'Resuming training from epoch:', starting_epoch
        curr_epoch = starting_epoch+1
        net.load_state_dict(torch.load(format_epoch_fname(curr_epoch)))
        best_record = load_pickle(mod_path + record_name)
        training_history = load_pickle(mod_path + history_name)

    # Define device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Load data
    paths = (trn_path, msk_path, tst_path)
    stats = (NUM_TRAIN, NUM_FULL, args.batch_size)
    trn_set, val_set, tst_set = data_formatter(paths, stats)
    # Unpack data
    trn_data, trn_load = trn_set
    val_data, val_load = val_set
    tst_data, tst_load = tst_set

    # Define automatic LR reduction scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=args.lr_patience,
        min_lr=1e-10
    )
    # Model API parameters
    param_dict = {
        'loader': None,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'epoch': 1,
        'args': args,
        'device': device,
        'dtype': dtype
    }
    # Note: epoch starts from 1, not 0
    for i, epoch in enumerate(range(curr_epoch, args.epochs+1)):
        # Update epoch
        param_dict['epoch'] = epoch
        # Train
        param_dict['loader'] = trn_load
        trn_log = train(**param_dict)
        # Validate
        param_dict['loader'] = val_load
        val_loss, mean_iou = validate(**param_dict)

        # Update logging files
        training_history['epoch_%s'%(i+1)] = trn_log
        # Save weights if avg_iou score improves
        if val_loss < best_record['val_loss']:
            best_record['epoch'] = epoch
            best_record['val_loss'] = val_loss
            best_record['mean_iou'] = mean_iou
            torch.save(net.state_dict(), format_epoch_fname(epoch))

        # Print best record information
        print '--------------------------------------'
        print 'best record: [epoch %d], [val_loss %.4f], [mean_iou %.4f]'%(
            best_record['epoch'],
            best_record['val_loss'],
            best_record['mean_iou']
        )
        print '--------------------------------------'
        print ''

        # Save logging information every epoch
        save_pickle(data=training_history, path=mod_path+history_name)
        save_pickle(data=best_record, path=mod_path+record_name)

        scheduler.step(val_loss)


if __name__=='__main__':
    main()
