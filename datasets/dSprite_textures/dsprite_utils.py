# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import math
import shutil
import skimage
import tarfile
import numpy as np
import urllib.request

from tqdm import tqdm
from torchvision.datasets.utils import gen_bar_updater

import warnings
warnings.filterwarnings("ignore")


def center_crop(img, crop_size):
    h, w = img.shape[0], img.shape[1]
    img_crop = img[h // 2 - crop_size // 2:h // 2 + crop_size // 2,
                   w // 2 - crop_size // 2:w // 2 + crop_size // 2]
    return img_crop


def resize(img, size):
    if type(size) == int:
        return skimage.transform.resize(img, (size, size), order=1)
    else:
        return skimage.transform.resize(img, size, order=1)


def rescale(img, scale):
    return skimage.transform.rescale(img, scale, order=1, multichannel=True)


def rotate(img, angle):
    angle = math.degrees(angle)
    img_rotate = skimage.transform.rotate(img, angle, order=1, resize=False)
    return img_rotate


def grey2rgb(img):
    img = np.tile(img, (3, 1, 1))
    img = np.moveaxis(img, 0, 2).astype(float)
    return img


def get_centroid_and_bbox(img):
    w = np.max(img, axis=0)
    h = np.max(img, axis=1)

    w_start = np.argmax(w)
    w_end = img.shape[0] - np.argmax(w[::-1])
    w_center = w_start + ((w_end - w_start) // 2)

    h_start = np.argmax(h)
    h_end = img.shape[1] - np.argmax(h[::-1])
    h_center = h_start + ((h_end - h_start) // 2)

    centroid = (h_center, w_center)
    bbox = np.array([w_start, w_end, h_start, h_end]) / img.shape[0]

    return centroid, bbox


def texturize_shape(img, texture, orientation, scale, return_bbox=True):
    if scale != 1:
        texture = rescale(texture, scale)

    if orientation != 0:
        texture = rotate(texture, orientation)

    d = (texture.shape[0] // 2)

    centroid, bbox = get_centroid_and_bbox(img)
    img = grey2rgb(img)

    h1, h2 = centroid[0] - d, centroid[0] + d
    w1, w2 = centroid[1] - d, centroid[1] + d

    d_h1, d_h2, d_w1, d_w2 = d, d, d, d

    s = img.shape[0]
    if h1 < 0:
        d_h1 = d + h1  # h1 will be negative
        h1 = np.clip(h1, 0, s)
    if h2 > s:
        d_h2 = d - (h2 - s)
        h2 = np.clip(h2, 0, s)
    if w1 < 0:
        d_w1 = d + w1  # w1 will be negative
        w1 = np.clip(w1, 0, s)
    if w2 > s:
        d_w2 = d - (w2 - s)
        w2 = np.clip(w2, 0, s)

    img[h1: h2, w1: w2] = img[h1: h2, w1: w2] * texture[d - d_h1: d + d_h2, d - d_w1: d + d_w2]
    img = (img * 255).astype(np.uint8)

    if return_bbox:
        return img, bbox
    else:
        return img


def latent_to_index(latents_classes, latents_sizes):
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
    return np.dot(latents_classes, latents_bases).astype(int)


def batch_texturize(imgs, metadata, latents_classes, texture):
    orientation_map = metadata['latents_possible_values']['orientation']
    scale_map = metadata['latents_possible_values']['scale']
    latents_sizes = metadata['latents_sizes']

    orientations = orientation_map[latents_classes[:, 3].astype(int)]
    scales = scale_map[latents_classes[:, 2].astype(int)]

    indices_sampled = latent_to_index(latents_classes, latents_sizes)
    shapes = imgs[indices_sampled]

    textured_shapes = []
    bboxes = []
    for shape, orientation, scale in tqdm(zip(shapes, orientations, scales), total=len(shapes)):
        img, bbox = texturize_shape(shape, texture, orientation, scale)
        textured_shapes.append(img)
        bboxes.append(bbox)
    return np.array(textured_shapes), np.array(bboxes)


def get_dSprite(data_dir='', download=True):
    dSprite_filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    file_path = os.path.join(data_dir, dSprite_filename)

    url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'

    if not os.path.isfile(file_path) and download:
        try:
            print('Downloading ' + url + ' to ' + file_path)
            urllib.request.urlretrieve(
                url, file_path,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + file_path)
                urllib.request.urlretrieve(
                    url, file_path,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e

    print('Loading {}'.format(file_path))
    dataset_zip = np.load(file_path, encoding='latin1', allow_pickle=True)
    return dataset_zip


def get_texture_images(data_dir='', download=True):
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
    tar_filename = url.split('/')[-1]
    file_path = os.path.join(data_dir, tar_filename)

    if not os.path.isdir('textures') and download:
        try:
            print('Downloading ' + url + ' to ' + data_dir)
            urllib.request.urlretrieve(
                url, file_path,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + file_path)
                urllib.request.urlretrieve(
                    url, file_path,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e

        print('Extracting images')
        tar = tarfile.open(tar_filename)
        tar.extractall()
        tar.close()

        print('Extracting texture patches')
        textures_dir = os.path.join(data_dir, 'textures')
        os.mkdir(textures_dir)

        img_filenames = ['banded_0022.jpg',
                         'grid_0079.jpg',
                         'zigzagged_0024.jpg']
        crop_list = [70, 200, 160]

        for i, (img_filename, crop) in tqdm(enumerate(zip(img_filenames, crop_list))):
            category_folder = img_filename.split('_')[0]
            path = os.path.join('dtd/images', category_folder, img_filename)
            texture = skimage.io.imread(path)
            texture = center_crop(texture, crop)
            texture = resize(texture, 28)
            texture = (texture * 255).astype(np.uint8)

            save_filename = 'crop_' + img_filename
            save_path = os.path.join(textures_dir, save_filename)
            skimage.io.imsave(save_path, texture)

        print('Cleaning up unnecessary files')
        os.remove(file_path)
        shutil.rmtree('dtd')
