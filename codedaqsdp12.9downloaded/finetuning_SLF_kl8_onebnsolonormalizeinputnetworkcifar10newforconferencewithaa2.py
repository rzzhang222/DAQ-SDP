#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:15:40 2024

@author: zhangruize
"""

from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import os
import torch.backends.cudnn as cudnn

from utils1 import AverageMeter
from utils1 import accuracy
from torch import optim
from torchvision import transforms, datasets
import models
import torch.nn.functional as F
from torch import nn
#from models.resnet_cifar2 import ResNet18
from models.resnet_add_normalize import resnet34
#from models.resnet_add_normalizecifar100 import resnet34

from models.linear import LinearClassifier
import tensorboard_logger as tb_logger
from utils1 import load_BN_checkpoint
from autoattack import AutoAttack
def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                     transform=train_transform,
                                     download=False)
    val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                   train=False,
                                   transform=val_transform,download=False)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='weight decay')
    #parser.add_argument('--weight_decay', type=float, default=0,
    #                    help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'], help='dataset')
    # other setting
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--name', type=str, default='kl8_slf',
                        help='name of the exp')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '/home/zhangruize/ContrastiveCrop-main/data/home/zhangruize/projects/ContrastiveCrop-main/data/cifar10/'
    opt.n_cls =10
    
    return opt

class combinedmodel(nn.Module):
    def __init__(self, model, classifier):
        super(combinedmodel, self).__init__()
        self.model = model
        self.classifier=classifier
        # self.rand = config['random_start']
        # self.step_size = config['step_size']
        # self.epsilon = config['epsilon']
        # assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'
    def forward(self, inputs):
        return self.classifier(self.model(inputs,return_feat=True))
        # x = inputs.detach()
        # if self.rand:
        #     x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        # if train:
        #     num_step = 10
        # else:
        #     num_step = 20
        # for i in range(num_step):
        #     x.requires_grad_()
        #     with torch.enable_grad():
        #         features = self.model(x, return_feat=True)
        #         logits = self.classifier(features)
        #         loss = F.cross_entropy(logits, targets, size_average=False)
        #     grad = torch.autograd.grad(loss, [x])[0]
        #     x = x.detach() + self.step_size * torch.sign(grad.detach())
        #     x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
        #     x = torch.clamp(x, 0, 1)
        # features = self.model(x, return_feat=True)
        # return self.classifier(features), x
# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, classifier, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.classifier=classifier
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'
    def forward(self, inputs, targets, train=True):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        if train:
            num_step = 10
        else:
            num_step = 20
        for i in range(num_step):
            x.requires_grad_()
            with torch.enable_grad():
                features = self.model(x, return_feat=True)
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        features = self.model(x, return_feat=True)
        return self.classifier(features), x


def set_model(opt):
    model = resnet34()
    #model = resnet34()
    model.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
    model.maxpool = nn.Identity()
    
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.name, feat_dim=512, num_classes=opt.n_cls)
    if len(opt.ckpt) > 2:
        print('loading from {}'.format(opt.ckpt))
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        if 'state_dict' in ckpt.keys():
            state_dict = ckpt['state_dict']
        elif 'simclr_state' in ckpt.keys():
            state_dict = ckpt['simclr_state']
        elif 'simsiam_state' in ckpt.keys():
            state_dict=ckpt['simsiam_state']
        elif 'byol_state' in ckpt.keys():
            state_dict=ckpt['byol_state']
        print(torch.cuda.device_count())
        if torch.cuda.is_available():
            #state_dict, _ = load_BN_checkpoint(state_dict)
            #state_dict1={k.replace('encoder.', ''): v for k, v in state_dict.items()} #encoder for simsiam
            state_dict1={k.replace('encoder_q.', ''): v for k, v in state_dict.items()} #encoder for byol
           
            state_dict2={k.replace('backbone.', ''): v for k, v in state_dict1.items()}
            #state_dict2={k.replace('downsample', 'shortcut'): v for k, v in state_dict1.items()}
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict2.items()}, strict=False)
           
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            #state_dict, _ = load_BN_checkpoint(state_dict)
            model = model.cuda()
            classifier = classifier.cuda()
            criterion = criterion.cuda()
            config = {
                'epsilon': 8.0 / 255.,
                'num_steps': 20,
                'step_size': 2.0 / 255,
                'random_start': True,
                'loss_func': 'xent',
            }
            net = AttackPGD(model, classifier, config)
            net = net.cuda()
            cudnn.benchmark = True
            #model.load_state_dict(state_dict, strict=False)
        else:
            print("only GPU version supported")
            raise NotImplementedError

    else:
        print("please specify pretrained model")
        raise NotImplementedError
    return model, classifier, net, criterion


def train(train_loader, model, classifier, net, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
          features = model(images, return_feat=True)
        output = classifier(features.detach())
        #output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, net, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    top1_clean = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output, _ = net(images, labels, train=False)
            loss = criterion(output, labels)

            features_clean = model(images, return_feat=True)
            output_clean = classifier(features_clean)
            acc1_clean, acc5_clean = accuracy(output_clean, labels, topk=(1, 5))

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top1_clean.update(acc1_clean[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 Clean {top1_clean.val:.4f} ({top1_clean.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top1_clean=top1_clean))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Acc@1 Clean {top1_clean.avg:.3f}'.format(top1_clean=top1_clean))
    return losses.avg, top1.avg, top1_clean.avg


def adjust_lr(lr, optimizer, epoch):
    #print('lr is:')
    #print(lr)
    if epoch >= 15:
        lr /= 10
    if epoch >= 20:
        lr /= 10
    
    print('lr is:')
    print(lr)    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def runAA(combinedmodel,test_loader):
    combinedmodel.eval()
    #classifier.eval()
    acc = 0.
    # test_loader.batch_size = 8000 if args.dataset == 'stl10' else 10000
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=10000, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    adversary = AutoAttack(combinedmodel, norm='Linf', eps=8/255, version='standard', log_path='aa_logtemporary')
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        xadv = adversary.run_standard_evaluation(images, labels, bs=128)
        with torch.no_grad():
            #if advFlag is not None:
            #    output = model.eval()(xadv, advFlag)
            #else:
                output =combinedmodel(xadv)
        acc += (output.max(1)[1] == labels).float().sum()
    return acc.item() / len(test_loader.dataset)

def main():
    best_acc = 0
    best_acc_clean = 0
    opt = parse_option()

    log_ra = 0
    log_ta = 0

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, net, criterion = set_model(opt)

    # build optimizer
    #params0=list(model.module.parameters())
    params1 = list(classifier.parameters())
    params = params1
    optimizer = optim.SGD(params,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    logname = ('logger/' + '{}_linearnormal'.format(opt.name))
    logger = tb_logger.Logger(logdir=logname, flush_secs=2)

    # training routine
    #for epoch in range(1,2):
    for epoch in range(1, opt.epochs + 1):
        adjust_lr(opt.learning_rate, optimizer, epoch-1)
        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, net, criterion,
                          optimizer, epoch, opt)
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        # eval for one epoch
        loss, val_acc, val_acc_clean = validate(val_loader, model, classifier, net, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('val_acc_clean', val_acc_clean, epoch)

        logger.log_value('best_val_acc', log_ra, epoch)
        logger.log_value('best_val_acc_clean', log_ta, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_clean = val_acc_clean
            log_ra = val_acc
            log_ta = val_acc_clean
            best_state = {
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'rng_state': torch.get_rng_state()
            }
            if not os.path.exists('./checkpoint/{}/'.format(opt.name)):
                os.makedirs('./checkpoint/{}/'.format(opt.name))
            torch.save(best_state, './checkpoint/{}/finetunedtemporary.ckpt'.format(opt.name, epoch))

        if epoch % opt.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'rng_state': torch.get_rng_state()
            }
            if not os.path.exists('./checkpoint/{}/'.format(opt.name)):
                os.makedirs('./checkpoint/{}/'.format(opt.name))
            torch.save(state, './checkpoint/{}/ep_{}.ckpt'.format(opt.name, epoch))

    print('best accuracy: {:.2f}'.format(best_acc))
    print('best accuracy clean: {:.2f}'.format(best_acc_clean))
    #print(best_state['model'].keys())
    #print(model.state_dict().keys())
    model.load_state_dict(torch.load('./checkpoint/small/cifar10/simclr_rcrop_unsupervised/finetunedtemporary.ckpt')['model'],strict=True)
    classifier.load_state_dict(torch.load('./checkpoint/small/cifar10/simclr_rcrop_unsupervised/finetunedtemporary.ckpt')['classifier'],strict=True)
    #print('val again')
    #loss, val_acc, val_acc_clean = validate(val_loader, model, classifier, net, criterion, opt)
    thecombinedmodel=combinedmodel(model,classifier)
    ALF_AA_acc = runAA(thecombinedmodel,val_loader)
    print('SLF AA robust acc: {:.2f}'.format(ALF_AA_acc * 100))

if __name__ == '__main__':
    main()