# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import time
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models.rae import RAE_SN
from utils.utils import AverageMeter, CSVLogger
from datasets.dSprite_textures.dsprite_textures_dataset import DspriteTextures

parser = argparse.ArgumentParser(description='Label Encoder')
parser.add_argument('--name', default='label_encoder', type=str, help='experiment name')
parser.add_argument('--data', help='path to dataset',
                    default='./datasets/dSprite_textures')
parser.add_argument('--log_dir', help='path to log directory',
                    default='./logs')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--nef', default=32, type=int)
parser.add_argument('--ndf', default=32, type=int)
parser.add_argument('--beta', default=0., type=float, help='latent space weight penalty')
parser.add_argument('--latent_dim', default=2048, type=int)
parser.add_argument('--n_train_samples', default=5000, type=int)
parser.add_argument('--n_val_samples', default=10000, type=int)
parser.add_argument('--eval_freq', default=10, type=int)
parser.add_argument('--mode', default='mask', type=str, choices=['bbox', 'mask'])
parser.add_argument('--patience', default=10, type=int,
                    help='number of epochs without improvement before stopping')


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.weight_file_name = 'dsprite_{}_encoder_{}_{}_{}.pth.tar'\
            .format(args.mode, args.nef, args.ndf, args.latent_dim)
        self.args.log_dir = os.path.join(self.args.log_dir, self.args.name)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)

        self.writer = SummaryWriter(self.args.log_dir)
        log_filepath = os.path.join(self.args.log_dir, 'log.csv')
        self.csv_logger = CSVLogger(args=self.args,
                                    fieldnames=['epoch', 'val_recon_loss'],
                                    filename=log_filepath)

        self.best_recon = 999999.
        self.n_iter = 0

        self.build_model()

        self.criterion = nn.BCELoss().cuda()
        self.optimizer = Adam(self.model.parameters(), self.args.learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        mode='min',
                                                        factor=0.5,
                                                        verbose=True,
                                                        patience=5)

        self.build_dataloaders()

        if self.args.resume:
            self.load_model()

        if self.args.evaluate:
            self.validate_step()
        else:
            self.train_loop()

        self.writer.close()
        self.csv_logger.close()

    def build_model(self):
        self.model = RAE_SN(num_classes=3,
                            img_res=64,
                            nef=self.args.nef,
                            ndf=self.args.ndf,
                            latent_dim=self.args.latent_dim)
        self.model.cuda()

    def load_model(self):
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.best_recon = checkpoint['best_recon']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

    def build_dataloaders(self):
        n_samples = self.args.n_train_samples + self.args.n_val_samples
        dataset = DspriteTextures(root_dir=self.args.data,
                                  n_samples=n_samples,
                                  seed=self.args.seed)
        train_set = torch.utils.data.Subset(dataset, range(0, self.args.n_train_samples))
        val_set = torch.utils.data.Subset(dataset, range(self.args.n_train_samples,
                                                         n_samples))

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=self.args.workers,
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(val_set,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=False,
                                                      drop_last=True,
                                                      num_workers=self.args.workers,
                                                      pin_memory=True)

    def preprocess_data(self, x):
        img, latents, bbox, mask = x

        if self.args.mode == 'bbox':
            out = bbox
        elif self.args.mode == 'mask':
            out = mask
        return out

    def train_step(self):
        recon_losses = AverageMeter('Loss', ':.4e')
        rae_kl_losses = AverageMeter('Loss', ':.4e')

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        pbar.set_description("Training epoch {}".format(self.epoch))

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, data in pbar:
            self.n_iter += 1
            label = self.preprocess_data(data)
            label = label.cuda()

            # measure data loading time
            data_time = time.time() - end

            # compute output
            z, recon = self.model(label)

            recon_loss = self.criterion(recon, label)
            rae_kl_loss = self.args.beta * torch.mean(0.5 * torch.sum(z**2, dim=1))
            loss = recon_loss + rae_kl_loss

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            recon_losses.update(recon_loss.item(), label.size(0))
            rae_kl_losses.update(rae_kl_loss.item(), label.size(0))

            pbar.set_postfix({'recon': recon_losses.avg, 'rae_kl': rae_kl_losses.avg})

            self.writer.add_scalar('train_loss/recon', recon_loss.item(), self.n_iter)
            self.writer.add_scalar('train_loss/rae_kl', rae_kl_loss.item(), self.n_iter)
            self.writer.add_scalar('train_loss/total_loss', loss.item(), self.n_iter)
            self.writer.add_scalar('time/batch', batch_time, self.n_iter)
            self.writer.add_scalar('time/data', data_time, self.n_iter)

    def validate_step(self):
        losses = AverageMeter('Loss', ':.4e')

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        pbar.set_description("Evaluating model")

        self.model.eval()

        with torch.no_grad():
            for i, data in pbar:
                label = self.preprocess_data(data)
                label = label.cuda()

                # compute output
                z, recon = self.model(label)
                recon_loss = self.criterion(recon, label)

                # measure accuracy and record loss
                losses.update(recon_loss.item(), label.size(0))

                pbar.set_postfix({'recon_loss': losses.avg})

        sample_label = make_grid(label)
        sample_recon = make_grid(recon)
        self.writer.add_image('samples/groundtruth', sample_label, global_step=self.epoch)
        self.writer.add_image('sample/recon', sample_recon, global_step=self.epoch)
        self.writer.add_scalar('val_loss/recon', losses.avg, self.epoch)

        row = {'epoch': str(self.epoch), 'val_recon_loss': str(losses.avg)}
        self.csv_logger.writerow(row)

        return losses.avg

    def train_loop(self):
        patience = self.args.patience
        patience_threshold = 1e-4

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            self.train_step()

            if self.epoch % self.args.eval_freq == 0:
                recon_loss = self.validate_step()

                self.writer.add_scalar('learning_rate', self.get_lr(), self.epoch)
                self.scheduler.step(recon_loss)

                if recon_loss < (self.best_recon - patience_threshold):
                    print('Current best val recon {:.6f}'.format(recon_loss))
                    self.best_recon = recon_loss
                    patience = self.args.patience  # reset patience if val performance improves

                    state = {'epoch': self.epoch + 1,
                             'state_dict': self.model.state_dict(),
                             'best_recon': self.best_recon,
                             'optimizer' : self.optimizer.state_dict()}
                    save_path = os.path.join(self.args.log_dir, self.weight_file_name)
                    torch.save(state, save_path)
                else:
                    patience -= 1
                    print('No improvement, current {:.6f} > best {:.6f}, patience {}'
                          .format(recon_loss, self.best_recon, patience))
                    if patience <= 0:
                        print('No improvement after {} epochs, stopping early'
                              .format(self.args.patience))
                        break  # stop early if model doesn't improve any more

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True

    trainer = Trainer(args)


if __name__ == '__main__':
    main()
