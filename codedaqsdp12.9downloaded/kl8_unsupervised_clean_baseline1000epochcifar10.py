#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 07:12:56 2023

@author: zhangruize
"""

import os
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from builder import build_optimizer, build_logger
from models import SimSiam, build_model
from losses import build_loss
from datasets import build_dataset, build_dataset_ccrop

from utils.utilnew import AverageMeter, format_time, set_seed, adjust_learning_rate1
from utils.config import Config, ConfigDict, DictAction

#my added code
import numpy as np
#from resnet_multibn_ensembleFC_kl6_base_unsupervised import resnet18 as ResNet18
#from models.resnet_add_normalizecifar100 import resnet18_NormalizeInput
from models.resnet_add_normalize import resnet34
import apex
import pickle
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image, ImageFilter
from typing import Sequence


torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--resume', type=str, help='path to resume checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='update the config; e.g., --cfg-options use_ema=True k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def load_weights(ckpt_path, train_set, model, optimizer, resume=True):
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        train_set.boxes = checkpoint['boxes'].cpu()
        model.load_state_dict(checkpoint['simclr_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise ValueError

    start_epoch = checkpoint['epoch'] + 1
    print("Loaded. (epoch {})".format(checkpoint['epoch']))
    return start_epoch

#my added freq code
def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask.cuda()

def generate_high(Images, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * (1.-mask)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

def generate_low(Images, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * mask
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

def generate_high_img(Image, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Image.shape[1], Image.shape[2]]), r)
    c, h, w = Image.shape
    x = Image.reshape([c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([c, 1, 1])
    fd = fd * (1.-mask)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([c, h, w])
    #fd=torch.clamp(fd,0,1)
    return fd

#my added adv code
# dict_name = 'cifar10_kmeans_10percent_5.pkl'
# f = open(dict_name, 'rb')  # Pickle file is newly created where foo1.py is
# chosen_inds_dict = pickle.load(f)  # dump data to f
# f.close()

# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#     ])
#cifar10_mean=
#cifar10_std=
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#     ])
#mean=(0.4914, 0.4822, 0.4465)
#std=(0.2470, 0.2435, 0.2616)
class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
    
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((32,32), scale=(0.08, 1.),interpolation=transforms.InterpolationMode.BICUBIC,),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur()],p=0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean,std),
    ])


train_transform_org = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

class TwoCropTransformAdv:
    """Create two crops of the same image"""
    def __init__(self, transform, transform_adv):
        self.transform = transform
        self.transform_adv = transform_adv

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.transform_adv(x)]
transform_train = TwoCropTransformAdv(transform_train, train_transform_org)

class CIFAR10Index1(Dataset):
    def __init__(self, root='/home/zhangruize/ContrastiveCrop-main/data/home/zhangruize/projects/ContrastiveCrop-main/data/cifar10/', transform=transform_train, download=False, train=True,\
                  ):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
        # self.cifar10_ind_l=list(chosen_inds_dict.keys())
        # self.cifar10_data_here=[]
        # self.cifar10_target_here=[]
        # for i in range(len(self.cifar10_ind_l)):
        #     img_i,target_i=self.cifar10[self.cifar10_ind_l[i]]
        #     self.cifar10_data_here.append(img_i)
        #     self.cifar10_target_here.append(target_i)
        #self.label_dict=label_dict
            
        #self.cifar10_data_here=np.vstack(self.cifar10_data_here)
        #self.cifar10_target_here=np.vstack(self.cifar10_target_here)

    def __getitem__(self, index):
        data,target = self.cifar10[index]
        #pseudo_label=self.label_dict[index]
        # label_p_002 = self.pseudo_label_002[index]
        # label_p_010 = self.pseudo_label_010[index]
        # label_p_050 = self.pseudo_label_050[index]
        # label_p_100 = self.pseudo_label_100[index]
        # label_p_500 = self.pseudo_label_500[index]

        # label_p = (label_p_002,
        #            label_p_010,
        #            label_p_050,
        #            label_p_100,
        #            label_p_500)
        return data, target
    
    def __len__(self):
        return len(self.cifar10)          


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


#my added adv code
class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, f1,f2,f3, f4, images_org, criterion):
        
        #x1 = images_t1.clone()
        #x2 = images_t2.clone()
        #x_LFC = generate_low(images_org, r=10)
        
        x_cl = images_org.clone()
        
        x_cl = x_cl + torch.zeros_like(x_cl).uniform_(-self.epsilon, self.epsilon)
        

        for i in range(self.num_steps):
            #print('attack iter ')
            #print(i)
            x_cl.requires_grad_()
            
            with torch.enable_grad():
                f_proj = self.model(x_cl, bn_name='pgd', contrast=True)
                #f1_proj, _ = self.model(x1, bn_name='normal', contrast=True)
                #f2_proj, _ = self.model(x2, bn_name='normal', contrast=True)
                #f3_proj, _ = self.model(x_LFC, bn_name='normal', contrast=True)
                
                features = torch.cat([f_proj.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1),f4.unsqueeze(1)], dim=1)
                loss_contrast = criterion(features)
                
            grad_x_cl = torch.autograd.grad(loss_contrast, x_cl)[0]
            x_cl = x_cl.detach() + self.step_size * torch.sign(grad_x_cl.detach())
            x_cl = torch.min(torch.max(x_cl, images_org - self.epsilon), images_org + self.epsilon)
            x_cl = torch.clamp(x_cl, 0, 1)
            
        return x_cl
    
class AttackPGD_ce(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD_ce, self).__init__()
        self.model = model
        
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        #self.num_steps = config['num_steps']
        self.num_steps = 10
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_org, ce_criterion, target):
        
        #x1 = images_t1.clone()
        #x2 = images_t2.clone()
        #x_LFC = generate_low(images_org, r=10)
        
        x_ce = images_org.clone()
        
        x_ce = x_ce + torch.zeros_like(x_ce).uniform_(-self.epsilon, self.epsilon)
        

        for i in range(self.num_steps):
            #print('attack iter ')
            #print(i)
            x_ce.requires_grad_()
            
            with torch.enable_grad():
                pred_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)
                #f1_proj, _ = self.model(x1, bn_name='normal', contrast=True)
                #f2_proj, _ = self.model(x2, bn_name='normal', contrast=True)
                #f3_proj, _ = self.model(x_LFC, bn_name='normal', contrast=True)
                
                #features = torch.cat([f_proj.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                loss_ce = F.cross_entropy(pred_ce,target,size_average=False)
                #print('loss ce')
                #print(loss_ce)
                
            grad_x_ce = torch.autograd.grad(loss_ce, x_ce)[0]
            x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            x_ce = torch.clamp(x_ce, 0, 1)
            
        return x_ce
    
class AttackPGD_kl(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD_kl, self).__init__()
        self.model = model
        
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_org, max_probs_idxs,second_probs_idxs, max_probs,second_probs):
        
        #x1 = images_t1.clone()
        #x2 = images_t2.clone()
        #x_LFC = generate_low(images_org, r=10)
        
        x_ce = images_org.clone()
        
        x_ce = x_ce + torch.zeros_like(x_ce).uniform_(-self.epsilon, self.epsilon)
        

        for i in range(self.num_steps):
            #print('attack iter ')
            #print(i)
            x_ce.requires_grad_()
            
            with torch.enable_grad():
                pred_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)
                softmax_fn=nn.Softmax(dim=1)
                pred_probs=softmax_fn(pred_ce)
                #f1_proj, _ = self.model(x1, bn_name='normal', contrast=True)
                #f2_proj, _ = self.model(x2, bn_name='normal', contrast=True)
                #f3_proj, _ = self.model(x_LFC, bn_name='normal', contrast=True)
                
                #features = torch.cat([f_proj.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                loss_kl = kl_d(max_probs_idxs,second_probs_idxs,max_probs,second_probs,pred_probs)
                
            grad_x_ce = torch.autograd.grad(loss_kl, x_ce)[0]
            x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            x_ce = torch.clamp(x_ce, 0, 1)
            
        return x_ce    


class AttackPGD_kl1(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD_kl1, self).__init__()
        self.model = model
        
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        #self.num_steps = config['num_steps']
        self.num_steps = 10
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_org, org_probs):
        
        #x1 = images_t1.clone()
        #x2 = images_t2.clone()
        #x_LFC = generate_low(images_org, r=10)
        
        x_ce = images_org.clone()
        
        x_ce = x_ce + torch.zeros_like(x_ce).uniform_(-self.epsilon, self.epsilon)
        

        for i in range(self.num_steps):
            #print('attack iter ')
            #print(i)
            x_ce.requires_grad_()
            
            with torch.enable_grad():
                pred_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)
                # softmax_fn=nn.Softmax(dim=1)
                # pred_probs=softmax_fn(pred_ce)
                # pred_probs=pred_probs+1e-6
                #f1_proj, _ = self.model(x1, bn_name='normal', contrast=True)
                #f2_proj, _ = self.model(x2, bn_name='normal', contrast=True)
                #f3_proj, _ = self.model(x_LFC, bn_name='normal', contrast=True)
                
                #features = torch.cat([f_proj.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                #loss_kl = kl_d(max_probs_idxs,second_probs_idxs,max_probs,second_probs,pred_probs)
                loss_kl=F.kl_div(F.log_softmax(pred_ce,dim=1),org_probs,reduction='batchmean')
            grad_x_ce = torch.autograd.grad(loss_kl, x_ce)[0]
            x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            x_ce = torch.clamp(x_ce, 0, 1)
            
        return x_ce    
    
def kl_d(largest_idxs, second_idxs, probs0_max,probs0_second, probs1):
    # probs0_copy=probs0.copy()
    # probs0_max,probs0_max_idxs=torch.max(probs0,dim=1)
    # probs1_copy=probs1.copy()
    # probs1_max,probs1_max_idxs=torch.max(probs1,dim=1)
    
    # for i in range(probs0.shape[0]):
    #     probs0_copy[i,probs0_max_idxs[i]]=0
    #     probs1_copy[i,probs1_max_idxs[i]]=0
    # probs0_second_max,probs0_second_max_idxs=torch.max(probs0_copy,dim=1)
    # probs1_second_max,probs1_second_max_idxs=torch.max(probs1_copy,dim=1)
    
    probs0_2=-1*(probs0_max+probs0_second)+1.0
    #probs1_2=-1*(probs1_max+probs1_second)+1.0
    probs1_2=torch.zeros(probs0_max.shape).cuda()
    probs1_max=torch.zeros(probs0_max.shape).cuda()
    probs1_second=torch.zeros(probs0_max.shape).cuda()
    for i in range(probs0_max.shape[0]):
        
        probs1_max[i]=probs1[i,largest_idxs[i]]
        probs1_second[i]=probs1[i,second_idxs[i]]
        probs1_2[i]=1-probs1[i,largest_idxs[i]]-probs1[i,second_idxs[i]]
    
    probs0_max=probs0_max+1e-6
    probs0_second=probs0_second+1e-6
    probs0_2=probs0_2+1e-6
    probs1_max=probs1_max+1e-6
    probs1_second=probs1_second+1e-6
    probs1_2=probs1_2+1e-6
    #print('probs1second: ')
    #print(probs1_second)
    #print('probs0second: ')
    #print(probs0_second)
    kl_d=(probs0_max*torch.log(probs0_max/probs1_max+1e-6)+ \
          probs0_second*torch.log(probs0_second/probs1_second+1e-6)+ \
          probs0_2* torch.log(probs0_2/probs1_2+1e-6)).mean()
    #print('kl_d: ')
    #print(kl_d)    
    return kl_d    

def update_box(eval_train_loader, model, len_ds, logger, t=0.05):
    if logger:
        logger.info(f'==> Start updating boxes...')
    model.eval()
    boxes = []
    t1 = time.time()
    for cur_iter, (images, _) in enumerate(eval_train_loader):  # drop_last=False
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feat_map = model(images, return_feat=True)  # (N, C, H, W)
        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)
        Hi, Wi = images.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_boxes, boxes)
    all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    all_boxes = all_boxes[:len_ds]
    if logger is not None:  # cfg.rank == 0
        t2 = time.time()
        epoch_time = format_time(t2 - t1)
        logger.info(f'Update box: {epoch_time}')
    return all_boxes


def train(train_loader, model, contrast_criterion2, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    #model.test()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_iter = len(train_loader)
    end = time.time()
    time1 = time.time()
    
    #my added code
    att_config = {
    'epsilon': 8.0 / 255.,
    'num_steps': 5,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',}

    model.module.train()
    #my added code for pseudo label
    #model_pre.module.eval()
    atnet=AttackPGD(model,att_config)
    
    at_ce=AttackPGD_ce(model,att_config)
    at_kl1=AttackPGD_kl1(model,att_config)
    criterion_ce=torch.nn.CrossEntropyLoss()
    #for idx, (images, _, labels) in enumerate(train_loader):
    for idx, (images, _) in enumerate(train_loader):    
        #model.module.eval()
        bsz = images[0].shape[0]
        #images = torch.cat([images[0], images[1], images[2]], dim=0)
        images[0] = images[0].cuda(cfg.local_rank, non_blocking=True)
        images[1] = images[1].cuda(cfg.local_rank, non_blocking=True)
        images[2] = images[2].cuda(cfg.local_rank, non_blocking=True)
        
        #images_high=generate_high(images[2].clone(),r=8)
        
        #labels=labels.cuda(cfg.local_rank, non_blocking=True)
        #images[3] = images[3].cuda(cfg.local_rank, non_blocking=True)
        # measure data time
        data_time.update(time.time() - end)

        # compute loss
        #my added code
        #x1, x2, x_cl = atnet(images[0],images[1],images[2],images[3],contrast_criterion)
        
        #model.module.train()
        #my added code
        #f_proj, _ = model(x_cl, bn_name='pgd', contrast=True)
        #fce_proj, fce_pred, logits_ce = model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
        f1_proj = model(images[0], return_feat=True)
        f2_proj = model(images[1], return_feat=True)
        
        #f1_proj=model(images[0])
        #f2_proj=model(images[1])
        #f_org_proj = model(images[2], bn_name='normal', contrast=True)
        #f_high_proj = model(images_high, bn_name='normal',contrast=True)
        
        #try pseudo label 
        #x_cl=atnet(f1_proj,f2_proj,f_org_proj,f_high_proj, images[2],contrast_criterion)
        #f_cl_proj = model(x_cl, bn_name='pgd', contrast=True)
        
        #features = torch.cat([f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        #features = torch.cat([f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_org_proj.unsqueeze(1)], dim=1)
        
        #loss_here = contrast_criterion(features)
        
        ####for unsupervised, changed to contrast_criterion1
        loss_here=contrast_criterion2(f1_proj,f2_proj)
        
        # #my added code
        # softmax_fn=nn.Softmax(dim=1)
        # with torch.no_grad():
        #   pre_prob=model_pre(images[2])
        #   #softmax_fn=nn.softmax(dim=1)
        #   pre_prob=softmax_fn(pre_prob)
        #   #print('pre_prob')
        #   #print(pre_prob)
          
        #   #change to efficient code
        #   largest2_pre_prob,largest2_prob_idxs=torch.topk(pre_prob,k=2,dim=1,largest=True,sorted=True)
        #   largest_pre_prob=largest2_pre_prob[:,0]
        #   largest_pre_prob_idxs=largest2_prob_idxs[:,0]
        #   second_largest_pre_prob=largest2_pre_prob[:,1]
        #   # second_largest_prob_idxs=largest2_prob_idxs[:,1]
        #   #change to efficient code
        #   #print('largest pre_prob: ')
        #   #print(largest_pre_prob)
        #    # pre_prob1=pre_prob.clone()
        #    # largest_pre_prob,largest_prob_idxs=torch.max(pre_prob,1)
        #    # for temp_i1 in range(pre_prob.shape[0]):
        #    #   pre_prob1[temp_i1][largest_prob_idxs[temp_i1]]=0
        #    # second_largest_pre_prob,second_largest_prob_idxs=torch.max(pre_prob1,1) 
        #   conf_ones=list(range(bsz))
        #   #labels1=labels.clone()
        #   #labels=labels.reshape(bsz)
        #   conf_ones=(largest_pre_prob-0.95)>second_largest_pre_prob
          
        #   #no_conf_ones=(largest_pre_prob-0.6)<=second_largest_pre_prob
        #   org_inputs_conf=images[2][conf_ones==1,:,:,:]
        #   org_inputs_no_conf=images[2][conf_ones==0,:,:,:]
          
        #   #org_conf_l=pre_prob[conf_ones==1,:]
        #   #org_no_conf_l=pre_prob[no_conf_ones,:]
          
        #   #comment some code
        #   #org_no_conf_largest_idxs=largest_prob_idxs[conf_ones==0]
        #   #org_no_conf_second_idxs=second_largest_prob_idxs[conf_ones==0]
        #   org_conf_l1=largest_pre_prob_idxs[conf_ones==1]
        #   #print('conf number: ')
        #   #print(org_conf_l1.shape[0])
          
        # #add some code for kl
        # pred_kl_model = model(org_inputs_no_conf, bn_name='normal', contrast=True, CF=True, return_logits=True,nonlinear=False)
        # pred_kl_model_prob=F.softmax(pred_kl_model,dim=1)
        # #print('no conf len: ')
        # #print(pred_kl_model_prob.shape[0])
        # #x_kl=at_kl(org_inputs_no_conf,pred_kl_model_prob+1e-6)
        
        # #the following five lines work with at_kl
        # # largest2_pre_prob_no_conf,largest2_prob_idxs_no_conf=torch.topk(pred_kl_model_prob,k=2,dim=1,largest=True,sorted=True)
        # # largest_pre_prob_no_conf=largest2_pre_prob_no_conf[:,0]
        # # largest_prob_idxs_no_conf=largest2_prob_idxs_no_conf[:,0]
        # # second_largest_pre_prob_no_conf=largest2_pre_prob_no_conf[:,1]
        # # second_largest_prob_idxs_no_conf=largest2_prob_idxs_no_conf[:,1]
        
        # if org_conf_l1.shape[0]!=0:
        #   x_conf_ce=at_ce(org_inputs_conf,criterion_ce,org_conf_l1)
        #   pred_x_ce_adv=model(x_conf_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)  
        # #pred_x_ce_adv_prob=softmax_fn(pred_x_ce_adv)
        #   loss_here=loss_here+0.2*criterion_ce(pred_x_ce_adv,org_conf_l1)    
        # #comment some code
        # # x_kl=at_kl(org_inputs_no_conf,org_no_conf_largest_idxs,org_no_conf_second_idxs, \
        # #              largest_pre_prob[conf_ones==0],second_largest_pre_prob[conf_ones==0])
        # #the following two lines useful with at_kl
        # #x_kl=at_kl(org_inputs_no_conf,largest_prob_idxs_no_conf,second_largest_prob_idxs_no_conf, \
        # #              largest_pre_prob_no_conf,second_largest_pre_prob_no_conf)
        # x_kl=at_kl1(org_inputs_no_conf,pred_kl_model_prob)    
            
        #   #_,org_conf_l1=torch.max(org_conf_l,dim=1)
        # #x_conf_ce=at_ce(org_inputs_conf,criterion_ce,org_conf_l1)
          
        # pred_x_kl=model(x_kl, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)  
        #   #org_conf_l1=torch.zeros(org_conf_l.shape).cuda()
        # pred_x_kl_adv_prob=F.log_softmax(pred_x_kl,dim=1)
        # #print('contrastive loss')
        # #print(loss_here)
        
        # #print('x_ce shape')
        # #print(x_conf_ce.shape)
        # #pred_x_ce_adv=model(x_conf_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True,nonlinear=False)  
        # #pred_x_ce_adv_prob=softmax_fn(pred_x_ce_adv)
        # #loss_here=loss_here+2*criterion_ce(pred_x_ce_adv,org_conf_l1)
        # #print('ce loss')
        # #print(criterion_ce(pred_x_ce_adv,org_conf_l1))
        
        # #comment some code
        # #the following line useful with at_kl
        # # loss_here=loss_here+2*kl_d(largest_prob_idxs_no_conf,second_largest_prob_idxs_no_conf, largest_pre_prob_no_conf \
        # #                          ,second_largest_pre_prob_no_conf,pred_x_kl_adv_prob)
        
        
        # # loss_here=loss_here+2*kl_d(org_no_conf_largest_idxs,org_no_conf_second_idxs,largest_pre_prob[conf_ones==0] \
        # #                          ,second_largest_pre_prob[conf_ones==0],pred_x_kl_adv_prob)
        # p_ratio=(bsz-org_conf_l1.shape[0])*1.0/bsz
        # loss_here=loss_here+0.2*p_ratio*F.kl_div(pred_x_kl_adv_prob,pred_kl_model_prob,reduction='batchmean')
        
        #try pseudo label  
        #print('kl loss')
        #print(kl_d(org_no_conf_largest_idxs,org_no_conf_second_idxs,largest_pre_prob[conf_ones==0] \
        #                         ,second_largest_pre_prob[conf_ones==0],pred_x_kl_adv_prob))
        #print(loss_here)
        #print('conf ones lenth')
        #print(conf_ones.sum())
        #print('no conf ones length')
        #print(bsz-conf_ones.sum())
        #f_ce_proj,_=model()
        #features = model(images)  # (2*bsz, C)
        # f1, f2 = torch.split(features, [bsz, bsz, bsz], dim=0)
        # loss = contrast_criterion(f1, f2)
        losses.update(loss_here.item(), bsz)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_here.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {loss_here:.3f}({losses.avg:.3f})')

    if logger is not None:  # cfg.rank == 0
        time2 = time.time()
        epoch_time = format_time(time2 - time1)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)

    world_size = torch.cuda.device_count()
    print('GPUs on this node:', world_size)
    cfg.world_size = world_size
    cfg.port=10001
    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # spawn
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))


def main_worker(rank, world_size, cfg):
    print('==> Start rank:', rank)

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # build logger, writer
    logger, writer = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'pretrain')

    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    train_set=CIFAR10Index1()
    #train_set = build_dataset_ccrop(cfg.data.train)
    len_ds = len(train_set)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    # eval_train_set = build_dataset(cfg.data.eval_train)
    # eval_train_sampler = torch.utils.data.distributed.DistributedSampler(eval_train_set, shuffle=False)
    # eval_train_loader = torch.utils.data.DataLoader(
    #     eval_train_set,
    #     batch_size=bsz_gpu,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True,
    #     sampler=eval_train_sampler,
    #     drop_last=False
    # )

    # build model, criterion; optimizer
    
    #my added code for pseudo label, need work
    # model_pre=build_model(dict(type='ResNet', depth=18, num_classes=10, maxpool=False))
    # model_pre=model_pre.cuda()
    # model_pre=torch.nn.parallel.DistributedDataParallel(model_pre,device_ids=[cfg.local_rank])
    # ckpt = torch.load('checkpoints/small/cifar10/simclr_rcrop/best_pretrain_finetuned_linear_cifar10_10percent_gt4_part3.pth', map_location='cuda')
    # state_dict=ckpt['model_state']
    # model_pre.load_state_dict(state_dict)
    
    
    #model_pre=model_pre.cuda()
    #my added code
    # model=build_model(cfg.model)
    # dim_mlp = model.fc.weight.shape[1]
    # model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    # model=model.cuda()
    #temporally comment
    #bn_names=['normal']
    model=resnet50()
    model.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
    model.maxpool = nn.Identity()
    
    model=model.cuda()
    #temporally comment
    
    #model = build_model(cfg.model)
    #dim_mlp = model.fc.weight.shape[1]
    #model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = apex.parallel.convert_syncbn_model(model)
    #model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    #model = torch.nn.DataParallel(model, device_ids=[cfg.local_rank])
    #model=model.cuda()
    #my added code
    #criterion = build_loss(cfg.loss).cuda()
    #contrast_criterion = SupConLoss(temperature=0.5)
    contrast_criterion2=build_loss(dict(type='NT_Xent_dist', temperature=0.2, base_temperature=0.07)).cuda()
    
    optimizer = build_optimizer(cfg.optimizer, model.parameters())

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, train_set, model, optimizer, resume=True)
    cudnn.benchmark = True

    # Start training
    print("==> Start training...")
    #for epoch in range(start_epoch, cfg.epochs + 1):
    for epoch in range(start_epoch,1001):    
        train_sampler.set_epoch(epoch)
        #adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)
        adjust_learning_rate1(cfg.lr_cfg, optimizer, epoch)
        # start ContrastiveCrop
        train_set.use_box = epoch >= cfg.warmup_epochs + 1

        # train; all processes
        train(train_loader, model, contrast_criterion2, optimizer, epoch, cfg, logger, writer)

        # # update boxes; all processes
        # if epoch >= cfg.warmup_epochs and epoch != cfg.epochs and epoch % cfg.loc_interval == 0:
        #     # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
        #     all_boxes = update_box(eval_train_loader, model.module, len_ds, logger,
        #                            t=cfg.box_thresh)  # on_cuda=True
        #     assert len(all_boxes) == len_ds
        #     train_set.boxes = all_boxes.cpu()

        # save ckpt; master process
        if rank == 0 and epoch % 100 == 0:
            #model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}_kl8try_unsupervised_clean_baseline_cifar10normalizedinputresnet34.2cifar10rebuiltresnet50.2.pth')
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}_kl8try_unsupervised_clean_baseline_cifar10normalizedinputresnet34.2cifar10rebuiltresnet50.2revisephnew.pth')
            
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'simclr_state': model.module.state_dict(),
                #'boxes': train_set.boxes,
                'epoch': epoch
            }
            torch.save(state_dict, model_path)

    # save the last model; master process
    if rank == 0:
        model_path = os.path.join(cfg.work_dir, 'last_kl8try_unsupervised_clean_baseline_cifar10normalizeinputresnet34.2cifar10rebuiltresnet50.2revisephnew.pth')
        state_dict = {
            'optimizer_state': optimizer.state_dict(),
            'simclr_state': model.module.state_dict(),
            #'boxes': train_set.boxes,
            'epoch': 1000
        }
        torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
