# Network for image segmentation
# Utilizes residual blocks with transpose convolutions
import torch
import torch.nn as nn

import numpy as np

import pdb


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Resdual block
class ResidualBlock(nn.Module):
    """Residual Block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Define downsample network
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Residual block with dropout
class ResBlock_Reg(nn.Module):
    """Residual Block with Dropout for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock_Reg, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        # Define downsample network
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


# Segmentation Resnet (33 Layers)
class ResSeg33(nn.Module):
    """ResNet for Segmentation Task"""

    def __init__(self, block):
        super(ResSeg33, self).__init__()

        # Define number of channels
        self.in_channels = 64

        self.relu = nn.ReLU(inplace=True)
        # 1st block
        self.conv1 = conv3x3(1, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = conv3x3(32, self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

        # ResNet blocks
        self.res1 = self.make_res_layer(block, self.in_channels, 3)
        # Downsample
        self.res2 = self.make_res_layer(block, 128, 3, stride=2)
        # Downsample
        self.res3 = self.make_res_layer(block, 256, 3, stride=2)
        # Downsample
        self.res4 = self.make_res_layer(block, 512, 3, stride=2)

        # Transpose convolutions
        self.trans_dict = {'in_channels': None,
                           'out_channels': None,
                           'kernel_size': 3,
                           'stride': 2,
                           'padding': 1,
                           'output_padding': 1,
                           'bias': False}
        # Upsample
        self.unres1 = self.make_trans_layer(self.trans_dict, 512, 256)
        # Upsample
        self.unres2 = self.make_trans_layer(self.trans_dict, 256, 128)
        # Upsample
        self.unres3 = self.make_trans_layer(self.trans_dict, 128, 2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)

    def make_res_layer(self, block, out_channels, num_blocks, stride=1):
        """Define residual block layer"""

        downsample = None
        if (stride!=1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # Include more blocks if desired (no downsampling)
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def make_trans_layer(self, params, in_channels, out_channels):
        """Define transpose conv layer"""

        params['in_channels'] = in_channels
        params['out_channels'] = out_channels
        layers = []
        layers.append(nn.ConvTranspose2d(**params))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward structure for network"""

        # 1st block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Res stacks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Transpose conv blocks
        x = self.unres1(x)
        x = self.unres2(x)
        x = self.unres3(x)
        return x


# Segmentation ResNet with MaxPool and Dropout (33 Layers)
class ResSeg33_Reg(ResSeg33):
    """ResNet with Regularization for Segmentation Task"""

    def __init__(self, block):
        ResSeg33.__init__(self, block)

        # Define number of channels
        self.in_channels = 64

        self.relu = nn.ReLU(inplace=True)
        # 1st block
        self.conv1 = conv3x3(1, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = conv3x3(32, self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # ResNet blocks
        self.res1 = self.make_res_layer(block, self.in_channels, 3)
        # Downsample
        self.res2 = self.make_res_layer(block, 128, 3, stride=2)
        # Downsample
        self.res3 = self.make_res_layer(block, 256, 3, stride=2)
        # Downsample
        self.res4 = self.make_res_layer(block, 512, 3, stride=2)

        # Transpose convolutions
        self.trans_dict = {'in_channels': None,
                           'out_channels': None,
                           'kernel_size': 3,
                           'stride': 2,
                           'padding': 1,
                           'output_padding': 1,
                           'bias': False}
        # Upsample
        self.unres1 = self.make_trans_layer(self.trans_dict, 512, 256)
        # Upsample
        self.unres2 = self.make_trans_layer(self.trans_dict, 256, 128)
        # Upsample
        self.unres3 = self.make_trans_layer(self.trans_dict, 128, 64)
        # Upsample
        self.unres4 = self.make_trans_layer(self.trans_dict, 64, 2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        """Forward structure for network"""

        # 1st block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # Res blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Transpose conv blocks
        x = self.unres1(x)
        x = self.unres2(x)
        x = self.unres3(x)
        x = self.unres4(x)
        return x


# Unit tests
def check_ResSeg33_size(dtype):
    """Unit test verifying output size for ResSeg33"""
    input_size = (8, 1, 128, 128)
    x = torch.zeros(input_size, dtype=dtype)
    model = ResSeg33(ResidualBlock)
    scores = model(x)
    print scores.size()
    print model
    print ''
    assert [i for i in scores.size()]==[8, 2, 128, 128]

def check_ResSeg33R_size(dtype):
    """Unit test verifying output size for ResSeg33_Reg"""
    input_size = (8, 1, 128, 128)
    x = torch.zeros(input_size, dtype=dtype)
    model = ResSeg33_Reg(ResBlock_Reg)
    scores = model(x)
    print scores.size()
    print model
    print ''
    assert [i for i in scores.size()]==[8, 2, 128, 128]


# Main function (unit tests)
def main():
    dtype = torch.float32
    check_ResSeg33_size(dtype)
    check_ResSeg33R_size(dtype)


if __name__=='__main__':
    main()
