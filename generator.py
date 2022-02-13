"""
Generator architecture and code taken from "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis" (arxiv.org/abs/2101.04775) and github.com/odegeasslbc/FastGAN-pytorch, respectively.
"""
import os
import argparse
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
from torch.nn.utils import spectral_norm
from utils import weights_init



def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        self.apply(weights_init)

    def forward(self, x):

        feat_4 = self.init(x)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        return self.to_big(feat_256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projected GAN Generator')
    parser.add_argument('--weights-path', type=str,metavar="Path", help='Path for Generator\'s weights',required=True )
    parser.add_argument('--latent-dim', type=int,metavar="N", help='Latent dimension for generator',required=True )
    parser.add_argument('--n-images', type=int,metavar="N", help='Number of Images to generate',required=True )
    parser.add_argument('--image-size', type=int,metavar="N",default=256, help='Size of Images to generate (N x N) (default: 256)')
    parser.add_argument('--mode', type=int,metavar="N", default=2, help='Wheter to generate images individually (0) ,in a grid (1), or both (2) (default: 2)')
    parser.add_argument('--grid-size', type=int,metavar="N", default=8, help='Size (N x N) of the images grid. default (8)')
    parser.add_argument('--out-dir', type=str,metavar="Path", default="./generated-images", help='Path of the output folder for generated images (default: ./generated-images)')
    parser.add_argument('--verbose', type=bool,metavar="Bool", default=True, help='Verbose. (default: True)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device : "+ device)

    # Loading state dict
    gen = Generator(im_size=args.image_size)
    gen.load_state_dict(torch.load(args.weights_path))
    if args.verbose:
        print("Successfully loaded generator's state dict at : " + args.weights_path)

    # Creating the output dir
    if not os.path.exists(args.out_dir):
        if args.verbose:
            print("Creating the output dir : ",args.out_dir)
        os.mkdir(args.out_dir)

    #Initializing list for grid
    gen_imgs_list=[]
    for i in tqdm(range(1,args.n_images + 1), position=0, leave=True):
        
        noise = torch.randn(args.latent_dim,1,1,device=device)
        #Generating single image
        gen_img = gen(noise.unsqueeze(0))
        #Saving the indivudual image
        if args.mode in [0,2]:
            img_path = os.path.join(args.out_dir,f"img-{i}.jpg")
            save_image(gen_img, img_path)
        
        # Saving grid of images (N x N)
        if args.mode in [1,2]:
            gen_imgs_list.append(gen_img.detach().cpu().squeeze(0))
            if len(gen_imgs_list) == args.grid_size ** 2:
                grid_img_path=os.path.join(args.out_dir,f"grid-imgs-{(i - args.grid_size ** 2 ) + 1}-{i}.jpg")
                save_image( gen_imgs_list, grid_img_path,nrow=args.grid_size)
                gen_imgs_list = []

    # If there are remaining images in the grid array
    if len(gen_imgs_list) and args.mode in [1,2]:
        save_image(gen_imgs_list ,os.path.join(args.out_dir,f"grid-imgs-{i - len(gen_imgs_list)}-{i}-remaining_batch.jpg"),nrow=args.grid_size)
    print("Done.")


