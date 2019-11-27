# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import h5py
import skimage
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from dsprite_utils import batch_texturize, get_dSprite, get_texture_images

parser = argparse.ArgumentParser(description='Generate dSprite-textures')
parser.add_argument('--data_root', type=str, default='',
                    help='root directory for dataset')


def generate_dSprite_textures():
    args = parser.parse_args()
    data_root = args.data_root
    filepath = os.path.join(data_root, 'dsprite_textures.h5')

    if not os.path.isfile(filepath):
        get_texture_images(data_root, download=True)
        dataset_zip = get_dSprite(data_root, download=True)

        print('Generating dSprite-textures...')

        masks = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        latents_classes = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata'][()]

        img_path_list = ['crop_banded_0022.jpg',
                         'crop_grid_0079.jpg',
                         'crop_zigzagged_0024.jpg']

        n_textures = len(img_path_list)
        img_size = 64

        textured_sprites = np.zeros((n_textures, len(masks), img_size, img_size, 3),
                                    dtype=np.uint8)

        for i, path in tqdm(enumerate(img_path_list), total=len(img_path_list)):

            path2 = os.path.join('textures', path)
            texture = skimage.io.imread(path2)

            imgs, bboxes = batch_texturize(masks, metadata, latents_classes, texture)
            textured_sprites[i] = imgs

        print('Writing to H5 file...')
        with h5py.File('dsprite_textures.h5', 'w') as hf:
            hf.create_dataset('bboxes', data=bboxes)
            hf.create_dataset('latents_values', data=latents_values)
            hf.create_dataset('latents_classes', data=latents_classes)
            hf.create_dataset('masks', data=masks)
            hf.create_dataset('imgs', data=textured_sprites)

            hf.create_group('metadata')
            str_array = np.array(metadata['latents_names'], dtype=object)
            string_dt = h5py.special_dtype(vlen=str)
            hf['metadata'].create_dataset('latents_names', data=str_array, dtype=string_dt)
            hf['metadata'].create_dataset('latents_sizes', data=metadata['latents_sizes'])
        print('Done!')


if __name__ == "__main__":
    generate_dSprite_textures()
