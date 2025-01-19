import torch
import torch.nn as nn
import functools

class AttentionUnetGenerator(nn.Module):
    """
    Pix2Pix UNet Generator with attention mechanism
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, 
                 norm_layer=nn.BatchNorm2d, use_dropout=True, use_attention=True):
        """
        Parameters:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            num_downs (int): Number of downsamplings in UNet. For example,
                           if |num_downs| == 7, image of size 128x128 will become
                           of size 1x1 at the bottleneck
            ngf (int): Number of generator filters in first conv layer
            norm_layer: Normalization layer
            use_dropout (bool): Use dropout or not
            use_attention (bool): Use attention mechanism or not
        """
        super().__init__()
        
        # Normalize the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Construct the UNet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, 
            submodule=None, norm_layer=norm_layer, 
            innermost=True, use_attention=use_attention
        )

        # Add intermediate layers
        # Gradually decrease number of filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer, 
                use_dropout=use_dropout, use_attention=use_attention
            )

        # Gradually decrease filters further
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer,
            use_attention=use_attention
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer,
            use_attention=use_attention
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer,
            use_attention=use_attention
        )

        # Final layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc,
            submodule=unet_block, outermost=True, 
            norm_layer=norm_layer, use_attention=False
        )

    def forward(self, input):
        """Forward pass of the generator"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, use_attention=True):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        # Create attention block with correct channel count
        if use_attention and not outermost:
            self.attention = MultiHeadAttentionBlock(
                input_dim=inner_nc,  # Use inner_nc as it's the number of channels after downconv
                num_heads=8,
                head_dim=inner_nc // 8,
                dropout=0.0
            )
        else:
            self.attention = None

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout:
                up += [nn.Dropout(0.5)]
                
            model = down + [submodule] + up
            
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            out = self.model(x)
            if self.attention is not None:
                out = self.attention(out)
            return torch.cat([x, out], 1)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=64, dropout=0.0):
        """
        Multi-Head Attention Block with fixed dimension handling
        
        Args:
            input_dim (int): Number of input channels
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each attention head
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Store input dimension
        self.input_dim = input_dim
        
        # Adjust number of heads to be compatible with input dimension
        self.num_heads = min(num_heads, input_dim // 8)
        if self.num_heads == 0:
            self.num_heads = 1
            
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Use input_dim for normalization
        self.norm1 = nn.BatchNorm2d(input_dim)
        self.norm2 = nn.BatchNorm2d(input_dim)
        
        # FFN with correct channel dimensions
        self.ffn = nn.Sequential(
            nn.Conv2d(input_dim, input_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(input_dim * 4, input_dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Save original dimensions
        b, c, h, w = x.shape
        assert c == self.input_dim, f"Input channels {c} doesn't match expected {self.input_dim}"
        
        # First normalization
        normed = self.norm1(x)
        
        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        x_flat = normed.flatten(2).transpose(1, 2)
        
        # Self-attention
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)
        
        # Reshape back (B, H*W, C) -> (B, C, H, W)
        attn_output = attn_output.transpose(1, 2).reshape(b, c, h, w)
        
        # First residual connection
        x = x + attn_output
        
        # Second normalization and FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x