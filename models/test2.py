from networks import *
# Define the necessary parameters
input_nc = 3       # Number of channels in input images (e.g., 3 for RGB)
output_nc = 3      # Number of channels in output images (e.g., 3 for RGB)
ngf = 64           # Number of filters in the last conv layer
norm = 'batch'     # Normalization layer type
use_dropout = False
init_type = 'normal'
init_gain = 0.02
gpu_ids = []       # List of GPU IDs to use (empty list means CPU)

# Instantiate the UnetGenerator model with 'unet_256'
netG = define_G(input_nc, output_nc, ngf, netG='unet_256', norm=norm,
                use_dropout=use_dropout, init_type=init_type,
                init_gain=init_gain, gpu_ids=gpu_ids)

# Print the model architecture
print(netG)
