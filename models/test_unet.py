import networks
import torch

def test_custom_unet():
    """Test if the custom UNet can be created through networks.define_G"""
    print("Testing Custom UNet Configuration...")
    
    # Test parameters
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'unet_custom'  # Changed to match your setting
    norm = 'batch'
    use_dropout = False
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []
    
    print(f"\nTesting network creation through define_G with netG={netG}...")
    try:
        net = networks.define_G(input_nc=input_nc, 
                              output_nc=output_nc, 
                              ngf=ngf, 
                              netG=netG,
                              norm=norm,
                              use_dropout=use_dropout,
                              init_type=init_type,
                              init_gain=init_gain,
                              gpu_ids=gpu_ids)
        
        print("✓ Network created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Forward pass
        with torch.no_grad():
            output = net(dummy_input)
        
        # Verify output dimensions
        expected_shape = (1, 3, 256, 256)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
        print(f"✓ Output dimensions correct: {output.shape}")
        
        # Print network structure
        print("\nNetwork structure:")
        print(net)
        
        print("\nAll tests passed! The custom UNet generator is working correctly.")
        return net
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Print the available network options
        print("\nAvailable network options in define_G:")
        with open(networks.__file__, 'r') as f:
            for line in f:
                if "elif netG ==" in line:
                    print(line.strip())
        return None

if __name__ == '__main__':
    model = test_custom_unet()