# Script containing training API and unit tests
import sys
sys.path.append('../')

import argparse
import pdb

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Utils import
from misc import get_paths, AverageMeter
from misc import Simple_Network
from misc import CrossEntropyLoss2d
# Dataset import
from datasets.tgs_dataset import data_formatter


def train(loader, net, criterion, optimizer, epoch, args, device, dtype):
    """
    Function for training a network through one epoch
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
    train_loss = AverageMeter()
    trn_log = []
    for t, (x, y, names) in enumerate(loader):
        net.train()
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        scores = net(x)
        loss = criterion(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item())
        trn_log.append(train_loss.val)

        if (t+1)%args.print_every==0:
            print '[epoch %d], [iter %d / %d], [train loss %.4f]' % (
            epoch, t+1, len(loader), train_loss.avg
            )

    return trn_log


# Unit tests
def check_train_log(loader, net, criterion, optimizer, args, device, dtype):
    """Unit test for verifying logging capability"""
    train_dict = {
        'loader': loader,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'epoch': 1,
        'args': args,
        'device': device,
        'dtype': dtype
    }
    trn_log = train(**train_dict)
    assert len(trn_log)==len(loader)

def check_loss(loader, net, criterion, optimizer, args, device, dtype):
    """Unit test for verifying loss behavior"""
    num_epochs=1000
    running_log = []
    train_dict = {
        'loader': loader,
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'epoch': 1,
        'args': args,
        'device': device,
        'dtype': dtype
    }
    for epoch in range(num_epochs):
        train_dict['epoch'] = epoch+1
        running_log += train(**train_dict)
    plt.figure(figsize=(10, 6))
    plt.plot(running_log, c='red')
    plt.xlabel('epochs'); plt.ylabel('loss'); plt.title('Loss wrt Epoch')
    plt.show()


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

    # Unit test
    check_train_log(trn_load, net, criterion, optimizer, args, device, dtype)
    # unit test
    check_loss(trn_load, net, criterion, optimizer, args, device, dtype)


if __name__=='__main__':
    main()
