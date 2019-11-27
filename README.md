# Evaluating Conditional GANs

This is the code repository for the paper [On the Evaluation of Conditional GANs](https://arxiv.org/pdf/1907.08175.pdf).  

Included is code for computing **Fréchet Joint Distance (FJD)**, our metric for evaluating conditional generative adversarial networks. Also included is code to generate the **dSprite-textures dataset**, and code to replicate our proof-of-concept experiments on dSprite-textures.

**Table of Contents**
1. [Fréchet Joint Distance (FJD)](#1-fréchet-joint-distance-fjd)
2. [dSprite-textures dataset](#2-dsprite-textures-dataset)
3. [dSprite-textures experiments](#3-proof-of-concept-experiments-on-dsprite-textures)  
  a. [Training label embeddings](#a-training-label-embeddings)  
  b. [Metric evaluation](#b-metric-evaluation-on-dsprite-textures)
4. [Pretrained label embeddings](#4-pretrained-label-embeddings)

## (0) Dependencies and Installation
- Clone this repo:
```bash
git clone https://github.com/facebookresearch/fjd  
cd fjd  
```

- Install the required dependencies:
```bash
pip install -r requirements.txt  
```

## (1) Fréchet Joint Distance (FJD)

In the paper we propose FJD as a single metric that can be used to evaluate conditional generative adversarial networks (cGANs). Our experimnets demonstrate that FJD captures several desirable properties of cGANs, including image quality, conditional consistency, and intra-conditioning diversity.  

For convenience, we provide an FJDMetric helper function for calculating FJD in `fjd_metric.py`. Aside from measuring FJD, this helper function handles tasks such as embedding images and conditionings, computing alpha, and calculating distribution statistics. _Please see_ `fjd_demo.ipynb` _or_ `fjd_demo.py` _for a demonstration of how to use FJDMetric._

## (2) dSprite-textures Dataset
dSprite-textures is a synthetic dataset adapted from the [dSprite](https://github.com/deepmind/dsprites-dataset) dataset by adding texture patterns to all shapes. Additionally, we add labels for shape class, bounding box, and mask. Textures are cropped from images in the [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html).  

- Generate the dSprite-textures dataset by running:
```bash
cd datasets/dSprite_textures
python generate_dsprite_textures.py
```
Warning: This generation script is not very memory efficient. It is recommended that you have at least 64GB of memory available on your machine before attempting to generate the dataset.

## (3) Proof-of-Concept Experiments on dSprite-textures

We provide code to replicate our experiments on the dSprite-dataset from Section 5 in the paper.

### (A) Training label embeddings  
First, create a label embedding by training an autoencoder on the bounding box and/or mask labels from the dSprite-textures dataset. If you have not yet generated the dataset, you will need to do so before this step. Train autoencoder with the following command (select either bbox or mask for mode):
```bash
python train_autoencoder.py --mode [bbox|mask]
```
You will need to train a separate autoencoder for each bounding boxes and masks. Training progress can be viewed with Tensorboard by running:
```bash
tensorboard --logdir logs/label_encoder
```
**Pretrained Models:** Alternatively, we provide pretrained autoencoders for bounding box and mask labels in [Section 4](#4-pretrained-label-embeddings). Download these files and place them in [logs/label_encoder](logs/label_encoder).  

### (B) Metric evaluation on dSprite-textures
To evaluate how FJD measures image quality, conditional consistency, and intra-conditioning diversity, run the following command (select only a single option for each arg):
```bash
python dsprite_experiments.py --mode [quality|consistency|diversity] --label_type [class|bbox|mask]
```
You will need to run the script once for each combination of arguments in order to replicate the entire experiment. Results will be saved in the logs folder in a csv file.

## (4) Pretrained Label Embeddings
We provide pretrained label embeddings for dSprite-textures and COCO-Stuff datasets. Please feel free to contribute your own pretrained label embeddings by submitting a pull request.

| Dataset       | Modality    | num_classes  | img_res | nef | ndf | latent_dim | File |
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| dSprite_textures  | bbox | 3 | 64 | 32 | 32 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/dsprite_bbox_encoder_32_32_2048.pth.tar)
| dSprite_textures  | mask | 3 | 64 | 32 | 32 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/dsprite_mask_encoder_32_32_2048.pth.tar)
| COCO-Stuff  | bbox | 183 | 64 | 64 | 64 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/ae_coco_bbox_64.pth.tar)
| COCO-Stuff  | bbox | 183 | 128 | 64 | 64 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/ae_coco_bbox_128.pth.tar)
| COCO-Stuff  | mask | 183 | 64 | 64 | 64 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/ae_coco_mask_64.pth.tar)
| COCO-Stuff  | mask | 183 | 128 | 64 | 64 | 2048 | [link](https://dl.fbaipublicfiles.com/fjd/ae_coco_mask_128.pth.tar)

## Ackowledgements
This repository builds upon code from several projects and individuals:  
[dSprite dataset](https://github.com/deepmind/dsprites-dataset) by Loic Matthey, Irina Higgins, Demis Hassabis, and Alexander Lerchner  
[Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html) by Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi  
[Numpy implementation of FID](https://github.com/bioinf-jku/TTUR/blob/master/fid.py) by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, and Marc Uecker  
[PyTorch implementation of FID](https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py) by Andrew Brock  
[PyTorch implementation of cov](https://discuss.PyTorch.org/t/covariance-and-gradient-support/16217/2) by Modar M. Alfadly  
[PyTorch implementation of matrix sqrt](https://github.com/msubhransu/matrix-sqrt) by Tsung-Yu Lin and Subhransu Maji  

## License
The FJD codebase is released under the [MIT license](LICENSE.md).