# Script for loading trained model and making predictions
import argparse
import os
import pdb

import torch

# Models import
from models.res_seg_19 import ResSeg19, ResidualBlock


def main():
    # Import settings (note that default debug settings are used)



    net = ResSeg19(ResidualBlock)

    # Load model
    print


if __name__=='__main__':
    main()
