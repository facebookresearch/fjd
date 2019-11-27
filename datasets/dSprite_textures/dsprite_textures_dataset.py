# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import h5py
import numpy as np
from torch.utils.data import Dataset

'''
This dataset is largely built on top of the dSprite dataset from Loic Matthey, 
Irina Higgins, Demis Hassabis, Alexander Lerchner. For a review of how to work 
with the dataset, it may be useful to review the following notebook:
https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
'''

class DspriteTextures(Dataset):
    def __init__(self, root_dir, n_samples=None, constraints={}, seed=0):

        self.root_dir = root_dir
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(self.seed)

        filename = 'dsprite_textures.h5'
        data_path = os.path.join(root_dir, filename)

        self.dsprites = h5py.File(data_path, 'r')

        self.masks = self.dsprites['masks']
        self.bboxes = self.dsprites['bboxes']
        self.imgs = self.dsprites['imgs']

        # Randomly assign each sprite a texture
        self.texture = np.random.choice(np.arange(self.imgs.shape[0]), size=self.imgs.shape[1], replace=True)

        self.latents_values = self.dsprites['latents_values'].value
        self.latents_classes = self.dsprites['latents_classes'].value

        self.metadata = self.dsprites['metadata']
        self.latents_names = list(self.metadata['latents_names'].value)
        self.latents_sizes = self.metadata['latents_sizes'].value

        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))

        self.set_constraints(constraints)

    def set_constraints(self, constraints={}):
        # reset latents_classes
        self.latents_classes = self.dsprites['latents_classes'].value

        # 'latents_names' = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
        for key, value in constraints.items():
            if key in self.latents_names:
                idx = self.latents_names.index(key)  # gets position of key in latent_names tuple

                n_options = self.latents_sizes[idx]
                assert (np.max(value) < n_options) and (np.min(value) >= 0), \
                    'Constraint value is out of range. {} must be below {:d} and above 0.'.format(key, n_options)

                self.latents_classes = self.latents_classes[np.isin(self.latents_classes[:, idx], value)]
            else:
                print('Key {} is not a valid key. Valid options are {}'.format(key, self.latents_names))

        if self.n_samples is not None:
            self.latents_classes = self.sample_latents(self.n_samples)

        self.indices = self.latent_to_index(self.latents_classes)

    def latent_to_index(self, latent_class):
        return np.dot(latent_class, self.latents_bases).astype(int)

    def sample_latents(self, n_samples):
        np.random.seed(self.seed + 1)
        n_samples = np.min([n_samples, len(self.latents_classes)])
        indices = np.random.choice(np.arange(len(self.latents_classes)), size=n_samples, replace=False)
        return self.latents_classes[indices]

    def bbox_coord_to_img(self, bbox, img_size):
        bbox = (img_size * bbox).astype(int)
        img = np.zeros((img_size, img_size), dtype=int)

        img[bbox[2]: bbox[3], bbox[0]: bbox[1]] = 1
        return img

    def get_image_from_idx(self, idx, texture_idx=None):
        if texture_idx is None:
            texture_idx = self.texture[idx]
        img = self.imgs[texture_idx, idx]
        bbox = self.bbox_coord_to_img(self.bboxes[idx], img_size=img.shape[0])
        mask = self.masks[idx]

        return img, bbox, mask

    def get_image_from_latents(self, latent_classes, texture_idx=None):
        idx = self.latent_to_index(latent_classes)
        return self.get_image_from_idx(idx, texture_idx)

    def apply_class_labels(self, class_id, mask):
        # one-hot encode class label
        class_id = np.eye(self.latents_sizes[1])[class_id]
        class_id = class_id[..., np.newaxis, np.newaxis]

        mask = np.expand_dims(mask, axis=0)
        mask = mask * class_id
        mask = mask.astype(np.float32)

        return mask

    def __len__(self):
        return len(self.latents_classes)

    def __getitem__(self, idx):
        latents = self.latents_classes[idx]
        class_id = latents[1]
        img, bbox, mask = self.get_image_from_latents(latents)

        bbox = self.apply_class_labels(class_id, bbox)
        mask = self.apply_class_labels(class_id, mask)

        return img, latents, bbox, mask
