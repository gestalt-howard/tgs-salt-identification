# Script containing validate API and unit tests
import sys
sys.path.append('../')

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Utils import
from misc import get_paths, AverageMeter
from misc import Simple_Network
from misc import CrossEntropyLoss2d
from evaluate import evaluate
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
        preds_list = []
        masks_list = []
        for v, (x, y, name) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = net(x)
            loss = criterion(scores, y)
            val_loss.update(loss.item())

            batch_masks = y.data.cpu.numpy()
            batch_preds = F.softmax(scores, dim=1).data.cpu.numpy()[:, 1, :, :]
            # Assemble evaluation ingredients
            masks_list.append(batch_masks)
            preds_list.append(batch_preds.squeeze())

        # Evaluate performance
        preds = np.concatenate(preds_list)
        masks = np.concatenate(masks_list)
        avg_iou = evaluate(preds, masks)

        # Logging


# Main function (unit tests)
def main():



if __name__=='__main__':
    main()
