#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 23:59:58 2022

@author: zhangruize
"""

# python DDP_simclr_ccrop.py path/to/this/config

# model
#dim = 128
#model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False)
#loss = dict(type='NT_Xent_dist', temperature=0.5, base_temperature=0.07)

# data
#root = '/path/to/your/dataset'
#root = '/data/home/zhangruize/projects/ContrastiveCrop-main/data/cifar10/'
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
#batch_size = 512
#batch_size=2048 #for no freq adv
#batch_size=512
#change for semi
batch_size=256 #changed to 512 from 256 only for taro
num_workers = 4
# data = dict(
#     train=dict(
#         ds_dict=dict(
#             type='CIFAR10_boxes',
#             root=root,
#             train=True,
#         ),
#         rcrop_dict=dict(
#             type='cifar_train_rcrop',
#             mean=mean, std=std
#         ),
#         ccrop_dict=dict(
#             type='cifar_train_ccrop',
#             alpha=0.1,
#             mean=mean, std=std
#         ),
#     ),
#     eval_train=dict(
#         ds_dict=dict(
#             type='CIFAR10',
#             root=root,
#             train=True,
#         ),
#         trans_dict=dict(
#             type='cifar_test',
#             mean=mean, std=std
#         ),
#     ),
# )

# boxes
#changed to 1000 epoch for unsupervised try
warmup_epochs =100
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs=100
lr = 0.5
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4)

#weight decay changed to 1e-5 for unsupervised, need to convert back for semi
##optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-5)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    warmup_steps=10,
    warmup_from=0.003
)


# log & save
log_interval = 20
save_interval = 250
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
