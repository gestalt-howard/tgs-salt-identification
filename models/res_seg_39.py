# Script containing res_seg_19 network adapted to 39 layers
# Number of residual blocks in each residual "stack" is changed
# Additonal convlutional layers added after each transpose-conv layer
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


# Segmentation ResNet
class ResSeg39(nn.Module):
    """Network with Residual Blocks for Segmentation"""

    def __init__(self, block, layers):
        super(ResSeg39, self).__init__()

        # Define number of input channels
        self.in_channels = 64
        self.s1_ch = 64
        self.s2_ch = 128
        self.s3_ch = 256
        self.s4_ch = 512

        self.relu = nn.ReLU(inplace=True)
        # 1st block
        self.conv1 = conv3x3(1, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = conv3x3(32, self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

        # Upsampling convolutions
        self.conv4 = conv3x3(self.s4_ch, self.s3_ch)
        self.conv5 = conv3x3(self.s3_ch, self.s2_ch)
        self.conv6 = conv3x3(self.s2_ch, self.s1_ch)
        self.conv7 = conv3x3(self.s1_ch, 2)

        # ResNet blocks
        self.res1 = self.make_res_layer(block, self.s1_ch, layers[0])
        # Downsample
        self.res2 = self.make_res_layer(block, self.s2_ch, layers[1], stride=2)
        # Downsample
        self.res3 = self.make_res_layer(block, self.s3_ch, layers[2], stride=2)
        # Downsample
        self.res4 = self.make_res_layer(block, self.s4_ch, layers[3], stride=2)

        # Transpose convolutions
        self.params = {'in_channels': None,
                       'out_channels': None,
                       'kernel_size': 3,
                       'stride': 2,
                       'padding': 1,
                       'output_padding': 1,
                       'bias': False}
        # Upsample
        self.unres1 = self.make_trans_layer(self.params, self.s4_ch, self.s3_ch)
        # Upsample
        self.unres2 = self.make_trans_layer(self.params, self.s3_ch, self.s2_ch)
        # Upsample
        self.unres3 = self.make_trans_layer(self.params, self.s2_ch, self.s1_ch)

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
                nn.BatchNorm2d(out_channels)
            )
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

        # Res blocks
        x1d = self.res1(x)
        x2d = self.res2(x1d)
        x3d = self.res3(x2d)
        x4d = self.res4(x3d)

        # Transpose conv blocks
        x3u = self.unres1(x4d)
        x3u = torch.cat((x3u, x3d), dim=1)
        x3u = self.relu(self.conv4(x3u))
        x2u = self.unres2(x3u)
        x2u = torch.cat((x2u, x2d), dim=1)
        x2u = self.relu(self.conv5(x2u))
        x1u = self.unres3(x2u)
        x1u = torch.cat((x1u, x1d), dim=1)
        x1u = self.relu(self.conv6(x1u))
        x = self.relu(self.conv7(x1u))

        return x


# Unit tests
def check_ResSeg39_size(dtype):
    """Unit test verifying output size of ResSeg39"""
    input_size = (8, 1, 128, 128)
    x = torch.zeros(input_size, dtype=dtype)
    model = ResSeg39(ResidualBlock, [3, 4, 6, 3])
    scores = model(x)
    print scores.size()
    print model
    assert [i for i in scores.size()]==[8, 2, 128, 128]


# Main function (unit tests)
def main():
    dtype = torch.float32
    check_ResSeg39_size(dtype)


if __name__=='__main__':
    main()
