#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 02:00:33 2024

@author: zhangruize
"""

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
warmup_epochs =1000
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs=1000
lr = 1.0
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-5)

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
port = 10002
