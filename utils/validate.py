# Script containing validate API and unit tests
import sys
sys.path.append('../')

import pdb
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Utils import
from misc import get_paths, AverageMeter
from misc import Simple_Network
from misc import CrossEntropyLoss2d
from evaluate import evaluate
from train import train
# Dataset import
from datasets.tgs_dataset import data_formatter


def validate(loader, net, criterion, optimizer, epoch, args, device, dtype):
    """
    Function for validating a network's performance afer one epoch of training
    Input(s):
    - loader (PyTorch loader object): loader for queueing minibatches
    - net (module object): PyTorch network module object
    - criterion (loss object): PyTorch loss function
    - optimizer (optimizer object): PyTorch optimizer function
    - epoch (int): current training epoch
    - args (parser object): parser containing command-line inputs
    - device (PyTorch device)
    - dtype (PyTorch datatype)
    Output(s):
    - val_loss.avg (float): average of val_loss for all mini-batches in
        validation set
    - mean_iou (float) = average mean iou score over all ground-truth masks and
        respective predictions in the validation set
    """
    net.eval()

    val_loss = AverageMeter()

    with torch.no_grad():
        preds_list = []
        masks_list = []
        for v, (x, y, name) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = net(x)
            loss = criterion(scores, y)
            val_loss.update(loss.item())

            batch_masks = y.data.cpu().numpy()
            batch_preds = F.softmax(scores, dim=1
                ).data.cpu().numpy()[:, 1, :, :]
            # Assemble evaluation ingredients
            masks_list.append(batch_masks)
            preds_list.append(batch_preds.squeeze())

        # Evaluate performance
        preds = np.concatenate(preds_list)
        masks = np.concatenate(masks_list)
        mean_iou = evaluate(preds, masks)

    print '--------------------------------------'
    print '[epoch %d], [val_loss %.4f], [mean_iou %.4f]'%(
        epoch, val_loss.avg, mean_iou)
    print '--------------------------------------'

    net.train()
    return val_loss.avg, mean_iou


# Unit tests
def check_validate(param_dict, trn_load, val_load):
    """Unit test for checking the validate function's behavior"""
    num_epochs = 100
    # Train for certain number of epochs
    param_dict['loader'] = trn_load
    for epoch in range(num_epochs):
        param_dict['epoch'] = epoch+1
        trn_log = train(**param_dict)
    # Validate
    param_dict['loader'] = val_load
    val_loss, mean_iou = validate(**param_dict)
    assert val_loss!=None
    assert mean_iou!=None


# Main function (unit tests)
def main():
    parser = argparse.ArgumentParser(description='Model Training API')
    parser.add_argument('--print_every', type=int, default=1,
                        help='num batches before printing (default: 1)')
    args = parser.parse_args()

    # Define device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Load data
    paths, stats = get_paths(debug=True)
    trn_path, msk_path, tst_path = paths
    NUM_TRAIN, NUM_FULL, batch_size = stats

    trn_set, val_set, tst_set = data_formatter(paths, stats)
    # Unpack sets
    trn_data, trn_load = trn_set
    val_data, val_load = val_set
    tst_data, tst_load = tst_set

    # Define training modules
    net = Simple_Network()
    # criterion = CrossEntropyLoss2d(dim=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Define parameters
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

    # Unit test
    check_validate(param_dict, trn_load, val_load)


if __name__=='__main__':
    main()
