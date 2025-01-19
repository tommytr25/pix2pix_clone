import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """A simplified ResNet block with two convolutional layers and a skip connection."""

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return F.relu(out)

class ResnetGenerator(nn.Module):
    """A simplified ResNet-based generator with 9 ResNet blocks."""

    def __init__(self, input_nc, output_nc, ngf=64):
        super(ResnetGenerator, self).__init__()
        
        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        # ResNet blocks
        self.res_blocks = nn.Sequential(
            *[ResnetBlock(ngf * 4) for _ in range(9)]
        )

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x
