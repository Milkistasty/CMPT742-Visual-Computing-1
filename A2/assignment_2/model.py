import torch
import torch.nn as nn
import torch.nn.functional as F

"""
UNET model settings refers to https://github.com/milesial/Pytorch-UNet/tree/master
"""

#import any other libraries you need below this line

# Part 1: The Convolutional blocks
class twoConvBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock, self).__init__()
    #initialize the block
    self.double_conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    #implement the forward path
    return self.double_conv(x)

# Part 2: The contracting path
class downStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(downStep, self).__init__()
    #initialize the down path
    self.conv_block = twoConvBlock(input_channel, output_channel)
    self.pool = nn.MaxPool2d(kernel_size=2)

  def forward(self, x):
    #implement the forward path
    x_before_pool = self.conv_block(x)
    x = self.pool(x_before_pool)
    # return both the output after the convolutions and after the pooling
    return x, x_before_pool

# Before concatenating the skip connection with the upsampled tensor, 
# center-crop the skip connection to match the size of the upsampled tensor
def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    start_x = (w - target_size[1]) // 2
    start_y = (h - target_size[0]) // 2
    return tensor[:, :, start_y:start_y+target_size[0], start_x:start_x+target_size[1]]

# Part 3: The expansive path
class upStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(upStep, self).__init__()
    #initialize the up path
    # Upsampling using transposed convolution
    self.upsample = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        
    # twoConvBlock after concatenating with skip connection
    # Note: the input channels will be double because of concatenation
    self.conv_block = twoConvBlock(2*output_channel, output_channel)

  def forward(self, x, skip_connection):
    #implement the forward path
    x = self.upsample(x)
    
    # Center crop the skip connection to match the size of x
    skip_connection = center_crop(skip_connection, (x.size(2), x.size(3)))

    # Concatenate with skip connection
    x = torch.cat([x, skip_connection], dim=1)
        
    x = self.conv_block(x)
    return x

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    #initialize the complete model
    # Contracting Path
    self.down1 = downStep(n_channels, 64)
    self.down2 = downStep(64, 128)
    self.down3 = downStep(128, 256)
    self.down4 = downStep(256, 512)

    # Base Convolution Block without downsampling
    self.conv_block = twoConvBlock(512, 1024)

    # Expansive Path
    self.up4 = upStep(1024, 512)
    self.up3 = upStep(512, 256)
    self.up2 = upStep(256, 128)
    self.up1 = upStep(128, 64)

    # Final 1x1 convolution to map to desired output n classes (in this case, 2)
    self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

  def forward(self, x):
    #implement the forward path
    # Contracting Path
    # print(f"Input shape to UNet: {x.shape}")
    x1, x_before_pool1 = self.down1(x)
    x2, x_before_pool2 = self.down2(x1)
    x3, x_before_pool3 = self.down3(x2)
    x4, x_before_pool4 = self.down4(x3)

    # Base Convolution Block
    x_middle = self.conv_block(x4)

    # Expansive Path with Skip Connections
    x = self.up4(x_middle, x_before_pool4)
    x = self.up3(x, x_before_pool3)
    x = self.up2(x, x_before_pool2)
    x = self.up1(x, x_before_pool1)

    # Final Convolution
    x = self.final_conv(x)
    return x


