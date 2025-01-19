import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """Self attention block using nn.MultiheadAttention"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.channels = channels
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        x_flat = x_norm.reshape(B, C, H*W).transpose(1, 2)
        
        # Apply self attention
        out_attn, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = out_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        return out + x

class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True, use_attention=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn = None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
            
        # Add attention after downsampling if specified
        self.attention = None
        if use_attention:
            num_heads = max(1, outplanes // 64)  # Ensure channels divisible by num_heads
            self.attention = SelfAttention(outplanes, num_heads=num_heads)
        
    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        # Apply attention after downsampling if it exists
        if self.attention is not None:
            fx = self.attention(fx)
            
        return fx


class DecoderBlock(nn.Module):
    """Decoder block with optional attention before upsampling"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, 
                 dropout=False, use_attention=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Add attention before upsampling if specified
        self.attention = None
        if use_attention:
            num_heads = max(1, inplanes // 64)
            self.attention = SelfAttention(inplanes, num_heads=num_heads)
            
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)       
        
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.relu(x)
        
        # Apply attention before upsampling if it exists
        if self.attention is not None:
            fx = self.attention(fx)
            
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx



class CustomUnetGenerator(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128, use_attention=True)
        self.encoder3 = EncoderBlock(128, 256)  
        self.encoder4 = EncoderBlock(256, 512)  
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)  
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2*512, 512)
        self.decoder4 = DecoderBlock(2*512, 256)  
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # encoder forward
        e1 = self.encoder1(x)                    # [B x 64 x 128 x 128]
        e2 = self.encoder2(e1)                   # [B x 128 x 64 x 64]
        e3 = self.encoder3(e2)                   # [B x 256 x 32 x 32]
        e4 = self.encoder4(e3)                   # [B x 512 x 16 x 16]
        e5 = self.encoder5(e4)                   # [B x 512 x 8 x 8]
        e6 = self.encoder6(e5)                   # [B x 512 x 4 x 4]
        e7 = self.encoder7(e6)                   # [B x 512 x 2 x 2]
        e8 = self.encoder8(e7)                   # [B x 512 x 1 x 1]
        
        # decoder forward + skip connections
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)