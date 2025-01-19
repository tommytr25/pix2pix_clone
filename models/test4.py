import torch
from networks import *

def test_attention_unet():
    # Define the necessary parameters
    input_nc = 3          # Number of input channels (RGB)
    output_nc = 3         # Number of output channels
    ngf = 64             # Number of generator filters
    norm = 'batch'       # Normalization layer type
    use_dropout = True   # Use dropout for better generalization
    init_type = 'normal' # Initialization method
    init_gain = 0.02     # Scaling factor for initialization
    gpu_ids = []         # Empty list means CPU only
    
    # Instantiate the Attention U-Net generator
    netG = define_G(input_nc, output_nc, ngf, netG='attention_unet', 
                   norm=norm, use_dropout=use_dropout, 
                   init_type=init_type, init_gain=init_gain, 
                   gpu_ids=gpu_ids)
    
    # Print model architecture
    print("=== Attention UNet Architecture ===")
    print(netG)
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in netG.parameters())
    trainable_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    # Create random input tensor (batch_size, channels, height, width)
    x = torch.randn(1, input_nc, 256, 256)
    print(f"Input shape: {x.shape}")
    
    # Perform forward pass
    try:
        with torch.no_grad():
            output = netG(x)
        print(f"Output shape: {output.shape}")
        print("\nForward pass successful!")
        
        # Check if output dimensions are correct
        expected_shape = (1, output_nc, 256, 256)
        assert output.shape == expected_shape, \
            f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
    
    # Print memory usage if running on GPU
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        print("\n=== GPU Memory Usage ===")
        print(f"Allocated: {torch.cuda.memory_allocated(0) // 1024 // 1024} MB")
        print(f"Cached: {torch.cuda.memory_cached(0) // 1024 // 1024} MB")

if __name__ == '__main__':
    test_attention_unet()