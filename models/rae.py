# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class ConvBlockDown(nn.Module):
    def __init__(self, fin, fout):
        super(ConvBlockDown, self).__init__()

        self.conv1 = nn.Conv2d(fin, fout, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(fout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(fout)
        self.downsample = nn.AvgPool2d(2)

        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.downsample(out)

        residual = self.conv_s(residual)
        residual = self.downsample(residual)

        out = out + residual
        out = self.relu(out)

        return out


class ConvBlockUp(nn.Module):
    def __init__(self, fin, fout):
        super(ConvBlockUp, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(fin, fout, kernel_size=3, stride=1, padding=1))
        self.bn1 = nn.BatchNorm2d(fout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1))
        self.bn2 = nn.BatchNorm2d(fout)

        self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.functional.interpolate(out, scale_factor=2)

        residual = self.conv_s(residual)
        residual = nn.functional.interpolate(residual, scale_factor=2)

        out = out + residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, num_classes, img_res=64, nef=32, latent_dim=64):
        super(Encoder, self).__init__()
        self.num_classes = num_classes
        self.img_res = img_res
        self.nef = nef
        self.latent_dim = latent_dim

        self.d = int(np.log2(self.img_res)) - 3

        layers = [ConvBlockDown(num_classes, nef)]

        for i in 2**np.arange(0, self.d):
            fin = nef * i
            fout = nef * (i * 2)
            fin = min(fin, 1024)  # cap the number of channels per layer at 1024
            fout = min(fout, 1024)

            layers.append(ConvBlockDown(fin, fout))

        self.conv = nn.Sequential(*layers)

        self.fc = nn.Linear(nef * (2**self.d) * 4 * 4, latent_dim)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.nef * (2**self.d) * 4 * 4)
        out = self.fc(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes, img_res=64, ndf=32, latent_dim=64):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.img_res = img_res
        self.ndf = ndf
        self.latent_dim = latent_dim

        self.d = int(np.log2(self.img_res)) - 3

        self.fc = spectral_norm(nn.Linear(latent_dim, ndf * (2**self.d) * 8 * 8))

        layers = [nn.BatchNorm2d(ndf * (2**self.d)),
                  nn.ReLU(True)]

        for i in 2**np.arange(0, self.d)[::-1]:
            fin = ndf * (i * 2)
            fout = ndf * i
            fin = min(fin, 1024)  # cap the number of channels per layer at 1024
            fout = min(fout, 1024)

            layers.append(ConvBlockUp(fin, fout))

        layers.append(spectral_norm(nn.Conv2d(ndf, num_classes, kernel_size=3, stride=1, padding=1)))
        layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, self.ndf * 2**self.d, 8, 8)
        out = self.conv(out)
        return out


# Based on architecture from https://arxiv.org/pdf/1903.12436.pdf, Appendix A
# Replaced convTranspose2d with conv2d + nearest neighbour upsample
# Replaced conv stride downsampling with average pool downsampling
class RAE_SN(nn.Module):
    def __init__(self, num_classes, img_res=64, nef=32, ndf=32, latent_dim=64):
        super(RAE_SN, self).__init__()
        self.num_classes = num_classes
        self.img_res = img_res
        self.nef = nef
        self.ndf = ndf
        self.latent_dim = latent_dim

        self.encoder = Encoder(num_classes, img_res, nef, latent_dim)
        self.decoder = Decoder(num_classes, img_res, ndf, latent_dim)

    def forward(self, x):
        # Assumes input is in range ([0, 1])
        z = self.encode(x)
        recon = self.decoder(z)
        recon = self.unnormalize(recon)
        return z, recon

    def encode(self, x):
        x = self.normalize(x)
        z = self.encoder(x)
        return z

    def normalize(self, x):
        return (x * 2.) - 1.

    def unnormalize(self, x):
        return (x + 1.) / 2.
