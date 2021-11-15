import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from torchvision import utils as vutils
from utils import kaiming_init, load_checkpoint
from efficient_net import build_efficientnet_lite
from generator import Generator
from differentiable_augmentation import DiffAugment
from dataset import load_data
import argparse
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 4, 2, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels, l):
        super(MultiScaleDiscriminator, self).__init__()
        self.head_conv = spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))

        # layers = []
        # if l == 1:
        #     layers.append(DownBlock(c_in, 64))
        #     layers.append(DownBlock(64, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 2:
        #     layers.append(DownBlock(c_in, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 3:
        #     layers.append(DownBlock(c_in, 256))
        #     layers.append(DownBlock(256, 512))
        # else:
        #     layers.append(DownBlock(c_in, 512))

        layers = [DownBlock(channels, 64 * [1, 2, 4, 8][l - 1])] + [DownBlock(64 * i, 64 * i * 2) for i in [1, 2, 4][l - 1:]]
        self.model = nn.Sequential(*layers)
        self.optim = Adam(self.model.parameters(), lr=0.0002, betas=(0, 0.99))

    def forward(self, x):
        x = self.model(x)
        return self.head_conv(x)


class CSM(nn.Module):
    """
    Implementation for the proposed Cross-Scale Mixing.
    """

    def __init__(self, channels, conv3_out_channels):
        super(CSM, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, conv3_out_channels, 3, 1, 1)

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        self.apply(kaiming_init)

    def forward(self, high_res, low_res=None):
        batch, channels, width, height = high_res.size()
        if low_res is None:
            # high_res_flatten = rearrange(high_res, "b c h w -> b c (h w)")
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res
        else:
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = torch.add(high_res, low_res)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res


class ProjectedGAN:
    def __init__(self, args):
        self.img_size = args.image_size

        self.gen = Generator(im_size=args.image_size)
        self.gen_optim = Adam(self.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        self.efficient_net = build_efficientnet_lite("efficientnet_lite1", 1000)
        self.efficient_net = nn.DataParallel(self.efficient_net)
        checkpoint = torch.load(args.checkpoint_efficient_net)
        load_checkpoint(self.efficient_net, checkpoint)
        self.efficient_net.eval()

        feature_sizes = self.get_feature_channels()
        self.csms = nn.ModuleList([
            CSM(feature_sizes[3], feature_sizes[2]),
            CSM(feature_sizes[2], feature_sizes[1]),
            CSM(feature_sizes[1], feature_sizes[0]),
            CSM(feature_sizes[0], feature_sizes[0]),
        ])

        self.discs = nn.ModuleList([
           MultiScaleDiscriminator(feature_sizes[0], 1),
           MultiScaleDiscriminator(feature_sizes[1], 2),
           MultiScaleDiscriminator(feature_sizes[2], 3),
           MultiScaleDiscriminator(feature_sizes[3], 4),
        ][::-1])

        self.latent_dim = args.latent_dim
        self.epochs = args.epochs

        augmentations = 'color,translation,cutout'
        self.DiffAug = DiffAugment(augmentations)
        self.diff_aug = args.diff_aug

        self.dataset = load_data(args.dataset_path, args.batch_size)
        self.log_every = args.log_every
        self.ckpt_path = args.checkpoint_path
        self.save_all = args.save_all

    def csm_forward(self, features):
        features = features[::-1]
        csm_features = []
        for i, csm in enumerate(self.csms):
            if i == 0:
                d = csm(features[i])
                csm_features.append(d)
            else:
                d = csm(features[i], d)
                csm_features.append(d)
        return features

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        self.gen.to(device)
        for disc in self.discs:
            disc.to(device)
        for csm in self.csms:
            csm.to(device)
        self.efficient_net.to(device)
        for epoch in range(self.epochs):
            logging.info(f"Starting epoch {epoch+1}")
            for i, (real_imgs, _) in enumerate(self.dataset):
                real_imgs = real_imgs.to(device)
                z = torch.randn(real_imgs.shape[0], self.latent_dim)
                z = z.to(device)

                gen_imgs_disc = self.gen(z).detach()
                if self.diff_aug:
                    gen_imgs_disc = self.DiffAug.forward(gen_imgs_disc)
                    real_imgs = self.DiffAug.forward(real_imgs)

                # get efficient net features
                _, features_fake = self.efficient_net(gen_imgs_disc)
                _, features_real = self.efficient_net(real_imgs)

                # feed efficient net features through CSM
                features_real = self.csm_forward(features_real)
                features_fake = self.csm_forward(features_fake)

                # Train Discriminators:
                disc_losses = []
                for feature_real, feature_fake, disc in zip(features_real, features_fake, self.discs):
                    disc.optim.zero_grad()
                    y_hat_real = disc(feature_real)  # Cx4x4
                    y_hat_fake = disc(feature_fake)  # Cx4x4
                    y_hat_real = y_hat_real.sum(1)  # sum along channels axis (is 1 anyways, however it still removes the unnecessary axis)
                    y_hat_fake = y_hat_fake.sum(1)
                    loss_real = torch.mean(F.relu(1. - y_hat_real))
                    loss_fake = torch.mean(F.relu(1. + y_hat_fake))
                    disc_loss = loss_real + loss_fake
                    disc_loss.backward(retain_graph=True)
                    disc.optim.step()
                    disc_losses.append(disc_loss.cpu().detach().numpy())

                # Train Generator:
                z = torch.randn(real_imgs.shape[0], self.latent_dim)
                z = z.to(device)
                # z = torch.Tensor(np.random.randn(real_imgs.shape[0], self.latent_dim))
                # while np.any(np.isnan(z.numpy())):
                #     logging.info("Recreating z because it has NaN values in it.")
                #     z = torch.Tensor(np.random.randn(real_imgs.shape[0], self.latent_dim))
                gen_imgs_gen = self.gen(z)

                if self.diff_aug:
                    gen_imgs_gen = self.DiffAug.forward(gen_imgs_gen)

                # get efficient net features
                _, features_fake = self.efficient_net(gen_imgs_gen)

                # feed efficient net features through CSM
                features_fake = self.csm_forward(features_fake)

                gen_loss = 0.
                self.gen_optim.zero_grad()
                for feature_fake, disc in zip(features_fake, self.discs):
                    y_hat = disc(feature_fake)
                    y_hat = y_hat.sum(1)
                    gen_loss = -torch.mean(y_hat)
                gen_loss.backward()
                self.gen_optim.step()

                if i % self.log_every == 0:
                    path = os.path.join(self.ckpt_path, str(epoch))
                    os.mkdir(path)
                    with torch.no_grad():
                        vutils.save_image(gen_imgs_gen.add(1).mul(0.5), os.path.join(path, f'/{epoch}_{i}.jpg'), nrow=4)
                    logging.info(f"Iteration {i}: Gen Loss = {gen_loss}, Disc Loss = {disc_losses}.")
                    torch.save(self.gen.state_dict(), os.path.join(path, "Generator"))
                    if self.save_all:
                        for j in range(len(self.discs)):
                            torch.save(self.discs[j].state_dict(), os.path.join(path, f"Discriminator_{j}"))
                            torch.save(self.csms[j].state_dict(), os.path.join(path, f"CSM_{j}"))

    def get_feature_channels(self):
        sample = torch.randn(1, 3, self.img_size, self.img_size)
        _, features = self.efficient_net(sample)
        return [f.shape[1] for f in features]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ProjectedGAN")
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.0, metavar='lambda', help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension for generator (default: 100)')
    parser.add_argument('--diff-aug', type=bool, default=True, help='Apply differentiable augmentation to both discriminator and generator (default: True)')
    parser.add_argument('--checkpoint-path', type=str, default="/checkpoints", metavar='Path', help='Path for checkpointing (default: /checkpoints)')
    parser.add_argument('--save-all', type=bool, default=False, help='Saves all discriminator, all CSMs and generator if True, only the generator otherwise (default: False)')
    parser.add_argument('--checkpoint-efficient-net', type=str, default="efficientnet_lite1.pth", metavar='Path', help='Path for EfficientNet checkpoint (default: efficientnet_lite1.pth)')
    parser.add_argument('--log-every', type=int, default=100, help='How often model will be saved, generated images will be saved etc. (default: 100)')
    parser.add_argument('--dataset-path', type=str, default='/data', metavar='Path', help='Path to data (default: /data)')
    parser.add_argument('--image-size', type=int, default=256, help='Size of images in dataset (default: 256)')
    args = parser.parse_args()
    Projected_GAN = ProjectedGAN(args)
    Projected_GAN.train()
