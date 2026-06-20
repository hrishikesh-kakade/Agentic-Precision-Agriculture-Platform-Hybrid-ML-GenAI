import torch 
import torch.nn as torchnn
import torch.nn.functional as F


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [torchnn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             torchnn.BatchNorm2d(out_channels),
             torchnn.ReLU(inplace=True)]
    if pool:
        layers.append(torchnn.MaxPool2d(4))
    return torchnn.Sequential(*layers)


# Model Architecture
class ResNet9(torchnn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = torchnn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = torchnn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = torchnn.Sequential(torchnn.MaxPool2d(4),
                                       torchnn.Flatten(),
                                       torchnn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out