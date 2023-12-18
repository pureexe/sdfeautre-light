import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torchinfo import summary
import numpy as np

def conv3x3(c_in, c_out, k=3, s=2, p=0):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=s, padding=p),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )

def conv1x1(c_in, c_out, k=1, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=s, padding=p),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )

def block(c_in, c_out, p=0):
    return nn.Sequential(
        conv1x1(c_in, c_in, p=0),
        conv3x3(c_in, c_out, p=p),
        conv1x1(c_out, c_out, p=0),
    )

def linear(c_in, c_out):
    return nn.Sequential(
        nn.Linear(c_in, c_out),
        # nn.BatchNorm1d(c_out),
        nn.GELU(),
    )

class SimpleCNN(nn.Module):
    def __init__(self, input_channel=2240, output_channel=147, modifier=1.0):
        super(SimpleCNN, self).__init__()
        self.modifier = modifier
        initial_conv_channel = int(256 * modifier)
        initial_mlp_channel = int(2048 * modifier)
        head_channel = int(256 * modifier)

        self.conv_in = conv3x3(input_channel, initial_conv_channel, p=1)
        self.conv_blocks = nn.Sequential(
            *[block(initial_conv_channel // 2**i, initial_conv_channel // 2**(i+1), p=1) for i in range(3)]
        )
        # input = (32, 8, 8)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            *[linear(initial_mlp_channel // 2**i, initial_mlp_channel // 2**(i+1)) for i in range(3)]
        )
        self.head = nn.Linear(head_channel, output_channel)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_blocks(x)
        x = self.mlp(self.flatten(x))
        x = self.head(x)
        return x


class Dilated_CNN_61(nn.Module):
    def __init__(self, input_channel):
        super(Dilated_CNN_61, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = nn.Conv2d(128, 64, 3, stride = 1, padding = 2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride = 1, padding = 4, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 8, dilation=8)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1, dilation=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 2, dilation=2)
        self.conv7 = nn.Conv2d(64, 64, 3, stride = 1, padding = 4, dilation=4)
        self.conv8 = nn.Conv2d(64, 32, 3, stride = 1, padding = 8, dilation=8)
        self.conv9 = nn.Conv2d(32, 1, 3, stride = 1, padding = 8, dilation=8)
        self.out = nn.Linear(128 * 128, 27)

        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        # print(x.shape)
        x = self.out(x.flatten(start_dim=1))
        # print(x.shape)
        # y = self.softmax(x) #Log
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(in_channel, out_channel, dim, depth, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv2d(in_channel, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, out_channel)
    )

if __name__ == "__main__":
    # x = torch.randn((2240, 128, 128))
    model = SimpleCNN(2240, modifier=3.0)
    # model = ConvMixer(512, 8, 4, 4)
    # ConvMixer-768/32*	7	7	85MB
    summary(model, input_size=(1, 2240, 128, 128))
    
    # model = dilated_CNN_61(10, 3)
    # y = model(x)