import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from torchvision import transforms
from random import random

class DiffusionModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    # modified into a diffusion model JWW 20240911
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            
        parser.set_defaults(wandb_project_name="joint-denoising-diffuser")
        
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'noisy_A', 'recovered_A', 'real_B', 'noisy_B', 'recovered_B']
            
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc+opt.output_nc, opt.input_nc+opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        noise_A = torch.randn_like(self.real_A)         # generate noise
        alpha = random()                                # random float from 0 to 1
        self.noisy_A = alpha*self.real_A + (1-alpha)*noise_A # generate a mixture of the image and noise
        
        noise_B = torch.randn_like(self.real_A)         # generate noise
        beta = random()                                 # random float from 0 to 1
        self.noisy_B = beta*self.real_B + (1-beta)*noise_B   # generate a mixture of the image and noise
        
        noisy_AB = torch.cat((self.noisy_A, self.noisy_B), 1) # stack noisy A and B into a single multi-channel input to the generator
        recovered_AB = self.netG( noisy_AB )        # pass stacked noisy AB through denoiser
        self.recovered_A = recovered_AB[:,0:self.opt.input_nc]  # first few channels go to recovered_A
        self.recovered_B = recovered_AB[:,self.opt.input_nc:]   # last few channels go to recovered_B
      
    def backward_D(self):
        return None


    def backward_G(self):
        """Calculate L1 loss for the generator"""

        self.loss_G_L1 = (self.criterionL1(self.recovered_B, self.real_B) \
                            + self.criterionL1(self.recovered_A, self.real_A))

  
        if torch.is_grad_enabled():
            self.loss_G_L1.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
