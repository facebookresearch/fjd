# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

from fjd_metric import get_embedding_statistics, calculate_alpha, calculate_fd
from embeddings import InceptionEmbedding, OneHotEmbedding, AutoencoderEmbedding
from datasets.dSprite_textures.dsprite_textures_dataset import DspriteTextures


parser = argparse.ArgumentParser(description='dSprite Experiments')
parser.add_argument('--datapath', help='path to dataset',
                    default='./datasets/dSprite_textures/')
parser.add_argument('--log_dir', help='path to log directory',
                    default='./logs')
parser.add_argument('--autoencoder_dir', help='path to autoencoder model weights',
                    default='./logs/label_encoder')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training')
parser.add_argument('--n_samples', default=10000, type=int,
                    help='number of samples from dataset to use')
parser.add_argument('--nef', default=32, type=int)
parser.add_argument('--ndf', default=32, type=int)
parser.add_argument('--latent_dim', default=2048, type=int)
parser.add_argument('--label_type', default='mask', type=str,
                    help='[class|bbox|mask]', choices=['class', 'bbox', 'mask'])
parser.add_argument('--mode', default='consistency', type=str,
                    help='[quality|consistency|diversity]',
                    choices=['quality', 'consistency', 'diversity'])


class Experiment(object):
    def __init__(self, args):
        self.args = args

        log_name = 'dsprite_textures_{}_experiment.csv'.format(args.mode)
        self.args.log_dir = os.path.join(self.args.log_dir, log_name)

        self.dataset = DspriteTextures(root_dir=self.args.datapath)
        self.attributes = ['shape', 'scale', 'orientation', 'posX', 'posY']
        args.n_classes = 3

        self.image_embedding = InceptionEmbedding(parallel=True)

        if self.args.label_type in ['bbox', 'mask']:
            weight_file_name = 'dsprite_{}_encoder_{}_{}_{}.pth.tar'\
                .format(args.label_type, args.nef, args.ndf, args.latent_dim)
            weights_path = os.path.join(args.autoencoder_dir, weight_file_name)
            self.label_embedding = AutoencoderEmbedding(args.n_classes,
                                                        args.nef,
                                                        args.ndf,
                                                        args.latent_dim,
                                                        weights_path)
        else:
            self.label_embedding = OneHotEmbedding(args.n_classes)

        if self.args.mode == 'quality':
            self.run_image_quality_experiment()
        elif self.args.mode == 'consistency':
            self.run_consistency_experiment()
        elif self.args.mode == 'diversity':
            self.run_diversity_experiment()

    def run_image_quality_experiment(self):
        print('Running image quality experiment')
        noise_magnitudes = np.logspace(1**0.5, 5.835**0.5, 10, 2) / 1000 - 0.01

        # The reference distribution is not changed for the rest of the experiment
        latents1 = self.get_latents_subset(seed=self.args.seed)
        img1_embed, label1_embed = self.get_data_from_latents(latents1,
                                                              noise_magnitude=0)
        alpha = calculate_alpha(img1_embed, label1_embed)
        label1_embed = alpha * label1_embed

        for noise_magnitude in noise_magnitudes:
            # For the "generated" distribution, draw the same samples as in the 
            # reference distribution, but add some noise to the images
            img2_embed, label2_embed = self.get_data_from_latents(latents1,
                                                                  noise_magnitude=noise_magnitude)
            label2_embed = alpha * label2_embed

            fid = self.get_fid(img1_embed, img2_embed)
            fjd = self.get_fjd(img1_embed, label1_embed, img2_embed, label2_embed)

            row = {'fid': fid,
                   'fjd': fjd,
                   'alpha': alpha,
                   'label_type': self.args.label_type,
                   'noise_magnitude': noise_magnitude
                   }
            print(row)
            self.write_to_log(row)

    def run_consistency_experiment(self):
        print('Running conditional consistency experiment')

        for attribute in self.attributes:
            # The reference and "generated" distribution are identical at the beginning
            latents1 = self.get_latents_subset(attribute, seed=self.args.seed)
            latents2 = np.copy(latents1)

            img1_embed, label1_embed = self.get_data_from_latents(latents1)
            img2_embed, label2_embed = self.get_data_from_latents(latents2)

            alpha = calculate_alpha(img1_embed, label1_embed)
            label1_embed = alpha * label1_embed

            # (Image distributions never change, so we only need to calculate FID once)
            fid = self.get_fid(img1_embed, img2_embed)

            idx = self.dataset.latents_names.index(attribute)
            size = (self.dataset.latents_sizes[idx] // 2) + 1
            for offset in range(0, size):
                # Swap the labels between samples for 30% of the data.
                # Swaps are selected such that two samples are identical in all 
                #   aspects except for a single attribute, which differs by some 
                #   desired offset.
                latents2_copy = latents2.copy()
                latents2_copy[:, idx] = swap_labels(latents2_copy[:, idx],
                                                    offset=offset,
                                                    n_swaps=0.3)

                latents2_indices = self.dataset.latent_to_index(latents2_copy)
                texture_indices = self.dataset.texture[latents2_indices]
                _, label2_embed = self.get_data_from_latents(latents=latents2_copy,
                                                             embed_images=False,
                                                             embed_labels=True,
                                                             texture_indices=texture_indices)
                label2_embed = alpha * label2_embed

                # Note that the original img2_embed is now paired with the new label2_embed
                # So now some image-label pairs in the "generated" distribution won't match
                fjd = self.get_fjd(img1_embed, label1_embed, img2_embed, label2_embed)

                row = {'attribute': attribute,
                       'fid': fid,
                       'fjd': fjd,
                       'alpha': alpha,
                       'label_type': self.args.label_type,
                       'offset': offset,
                       }
                print(row)
                self.write_to_log(row)

    def run_diversity_experiment(self):
        print('Running intra-conditioning diversity experiment')
        latents1 = self.get_latents_subset(seed=self.args.seed)
        textures1 = np.random.randint(0, 3, len(latents1))

        img1_embed, label1_embed = self.get_data_from_latents(latents1,
                                                              texture_indices=textures1)
        alpha = calculate_alpha(img1_embed, label1_embed)
        label1_embed = alpha * label1_embed

        attributes = ['shape', 'scale', 'orientation', 'x_pos']
        for attribute in attributes:
            for diversity_score in np.linspace(0, 1, 11):
                textures2 = np.copy(textures1)
                n_changed = int(len(textures2) * diversity_score)

                # Bias the texture distribution for a percentage of samples
                if attribute == 'shape':
                    textures2[:n_changed] = latents1[:n_changed, 1]  # set textures wrt shape
                elif attribute == 'scale':
                    textures2[:n_changed] = latents1[:n_changed, 2] // 2  # set textures wrt scale
                elif attribute == 'orientation':
                    textures2[:n_changed] = latents1[:n_changed, 3] // (42 / 3)  # set textures wrt orientation
                elif attribute == 'x_pos':
                    textures2[:n_changed] = latents1[:n_changed, 4] // (33 / 3)  # set textures wrt x position

                # All other attributes remain fixed, only texture is changed
                img2_embed, label2_embed = self.get_data_from_latents(latents1,
                                                                      texture_indices=textures2)
                label2_embed = alpha * label2_embed

                fid = self.get_fid(img1_embed, img2_embed)
                fjd = self.get_fjd(img1_embed, label1_embed, img2_embed, label2_embed)

                row = {'attribute': attribute,
                       'fid': fid,
                       'fjd': fjd,
                       'alpha': alpha,
                       'label_type': self.args.label_type,
                       'diversity_score': diversity_score
                       }
                print(row)
                self.write_to_log(row)

    def write_to_log(self, row):
        df = pd.DataFrame(row, index=[0])

        if self.args.log_dir:
            df.to_csv(self.args.log_dir,
                      mode='a',
                      sep=',',
                      encoding='utf-8',
                      index=False,
                      header=(not os.path.isfile(self.args.log_dir)))

    def get_fid(self, img1, img2):
        m1, s1 = get_embedding_statistics(img1)
        m2, s2 = get_embedding_statistics(img2)
        fid = calculate_fd(m1, s1, m2, s2)
        return fid

    def get_fjd(self, img1, label1, img2, label2):
        joint1 = np.concatenate([img1, label1], axis=1)
        joint2 = np.concatenate([img2, label2], axis=1)
        m1, s1 = get_embedding_statistics(joint1)
        m2, s2 = get_embedding_statistics(joint2)
        fjd = calculate_fd(m1, s1, m2, s2)
        return fjd

    def get_data_from_latents(self,
                              latents,
                              embed_images=True,
                              embed_labels=True,
                              texture_indices=None,
                              noise_magnitude=0):

        data = get_data_from_latents(self.dataset,
                                     latents=latents,
                                     texture_indices=texture_indices)
        img, label, bbox, mask = data

        if self.args.label_type == 'class':
            label = label
        elif self.args.label_type == 'bbox':
            label = bbox
        elif self.args.label_type == 'mask':
            label = mask

        if noise_magnitude > 0:
            img = self.noisify_image(img, noise_magnitude)

        if embed_images:
            img = batch_embed(embedding=self.image_embedding,
                              input=img,
                              desc='Image embedding')

        if embed_labels:
            if self.args.label_type in ['bbox', 'mask']:
                label = batch_embed(self.label_embedding,
                                    input=label,
                                    desc='Label embedding')

        return img, label

    def noisify_image(self, img, noise_magnitude):
        # When adding noise, multiply by 2 since image is in range [-1, 1]
        noise = torch.empty(img.shape).normal_(0, 1.) * noise_magnitude * 2.
        img = torch.clamp(img + noise, -1, 1.)
        return img

    def get_latents_subset(self, attribute=None, seed=0):
        '''
        If an attribute is specified, this function will creates a special 
        subset of the dataset where for each sample, all variants of that 
        sample for the selected attribute are also present. This property allows 
        us to swap conditions between samples in such a way that conditional 
        consistency decreases but the marginal distribution remains unchanged.

        If an attribute is not selected, then samples are selected randomly.
        '''
        np.random.seed(seed)
        sizes = self.dataset.latents_sizes

        if attribute is not None:
            attribute_idx = self.dataset.latents_names.index(attribute)
            attribute_dim = sizes[attribute_idx]
            n_samples = self.args.n_samples // attribute_dim
        else:
            n_samples = self.args.n_samples

        # The latent variable is a 6 element vector indicating properties of the sample
        # Elements represent colour, shape, scale, orientation, posX, and posY respectively
        # Attribute values are randomly selected for all samples
        latents = np.zeros((n_samples, len(sizes)))  # dimension = [n_samples, 6]
        for i, n_options in enumerate(sizes):
            latents[:, i] = np.random.choice(np.arange(n_options), size=n_samples)

        if attribute is not None:
            # Make copies of each sample, but each one has a unique value for the given attribute
            latents = np.repeat(latents, attribute_dim, axis=0)
            latents[:, attribute_idx] = np.tile(np.arange(attribute_dim), n_samples)

        return latents


def swap_labels(labels, offset, n_swaps=0.3):
    # Swap labels to achieve desired offset without changing marginal distributions.
    # This is basically a brute-force search that iterates through the list until 
    #   it finds a suitable candidate for swapping.
    swapped = np.zeros(len(labels))
    n_unique = len(np.unique(labels))

    if type(n_swaps) == float:
        n_swaps = int(n_swaps * len(labels))

    target_range = range(0, n_unique)
    swap_count = 0

    for i, label1 in enumerate(labels):
        if swap_count == n_swaps:
            break
        if (label1 in target_range) and (swapped[i] == 0):
            target_value = label1 + offset * np.random.choice([-1, 1])
            for j, label2 in enumerate(labels):
                if (label2 == target_value) and (swapped[j] == 0):
                    tmp = labels[i]
                    labels[i] = labels[j]
                    labels[j] = tmp
                    swapped[i] = swapped[j] = 1
                    swap_count += 1
                    break

    if swap_count < n_swaps:
        print('Not enough swaps, only reached {} out of {}'.format(swap_count, n_swaps))
    return labels


def get_data_from_latents(dataset, latents, texture_indices=None):
    imgs = []
    bboxes = []
    masks = []

    if texture_indices is not None:
        for latent, texture in zip(latents, texture_indices):
            img, bbox, mask = dataset.get_image_from_latents(latent, texture)
            imgs.append(img)
            bboxes.append(bbox)
            masks.append(mask)
    else:
        for latent in latents:
            img, bbox, mask = dataset.get_image_from_latents(latent)
            imgs.append(img)
            bboxes.append(bbox)
            masks.append(mask)

    imgs = torch.from_numpy(np.stack(imgs))
    bboxes = torch.from_numpy(np.stack(bboxes).astype(np.float32))
    masks = torch.from_numpy(np.stack(masks).astype(np.float32))

    imgs = imgs.float() / 255.  # normalize to [0, 1]
    imgs = (imgs * 2.) - 1.  # normalize to [-1, 1]
    imgs = imgs.permute(0, 3, 1, 2)

    labels = latents[:, 1]  # get shape attribute, which we will use as the class label
    labels = torch.eye(3)[labels]  # dimension = [batch_size, 3]

    bboxes = torch.unsqueeze(bboxes, dim=1) * labels.view(-1, 3, 1, 1)
    masks = torch.unsqueeze(masks, dim=1) * labels.view(-1, 3, 1, 1)

    return imgs, labels, bboxes, masks


def batch_embed(embedding, input, batch_size=256, desc=''):
    embeddings = []

    n_batches = int(np.ceil(len(input) / batch_size))
    for i in trange(n_batches, desc=desc):
        batch = input[i * batch_size: (i + 1) * batch_size]
        batch = batch.cuda().float()

        with torch.no_grad():
            feat = embedding(batch)
            feat = feat.cpu().detach().numpy()
        embeddings.append(feat)
    embeddings = np.concatenate(embeddings, 0)
    return embeddings


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    Experiment(args)


if __name__ == '__main__':
    main()
