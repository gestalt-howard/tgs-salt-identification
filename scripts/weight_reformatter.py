# Script for reformatting PyTorch .pth weight files
import argparse
import pdb

from collections import OrderedDict

import torch


# Main function
def main():
    """
    Python script for reformatting PyTorch .pth weight files
    Inputs taken through shell
    """
    parser = argparse.ArgumentParser(description='Weight reformatting API')
    parser.add_argument('--mod_path', type=str,
                        help='path to model weights directory')
    parser.add_argument('--mod_name', type=str,
                        help='name of epoch weights to load in given mod_path')
    parser.add_argument('--sav_name', type=str,
                        help='name to save reformatted epoch weights as')
    args = parser.parse_args()

    # Define variables relative to parser inputs
    mod_path = args.mod_path
    mod_name = mod_path + args.mod_name
    sav_name = mod_path + args.sav_name

    # Load dictionary
    model_dict = torch.load(mod_name, map_location='cpu')
    dict_keys = model_dict.keys()
    new_keys = []
    for key in dict_keys:
        assert 'module.' in key
        new_keys.append(key.split('module.')[1])
    new_dict = OrderedDict((new_keys[i], v) for i, (_, v) in enumerate(
        model_dict.viewitems()))
    # Save new dictionary
    torch.save(new_dict, sav_name)

    return None


if __name__=='__main__':
    main()
