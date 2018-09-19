# Script containing validate API and unit tests
import sys
sys.path.append('../')

import pdb

import torch
import torch.nn as nn
import torch.optim as optim

# Utils import
from misc import get_paths, AverageMeter
from misc import Simple_Network
from misc import CrossEntropyLoss2d
# Dataset import
from datasets.tgs_dataset import data_formatter


def validate(loader, net, criterion, optimizer, epoch, args, device, dtype):
    """
    Function for validating a network's performance afer one epoch of training
    Inputs:
    - loader (PyTorch loader object): loader for queueing minibatches
    - net (module object): PyTorch network module object
    - criterion (loss object): PyTorch loss function
    - optimizer (optimizer object): PyTorch optimizer function
    - epoch (int): current training epoch
    - args (parser object): parser containing command-line inputs
    - device (PyTorch device)
    - dtype (PyTorch datatype)
    Output:
    - trn_log (list): list of training losses for epoch
    """
    net.eval()

    val_loss = AverageMeter()

    with torch.no_grad():
        for v, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)


# Main function (unit tests)
def main():
    pass


if __name__=='__main__':
    main()
