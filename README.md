# Code for DAQ-SDP: Self-Supervised Adversarial Training via Diverse Augmented Queries and Self-Supervised Double Perturbation in NeurIPS 2024.

(https://neruips.cc/virtual/2024/poster/96153)

## To train the model (multi-GPUs):

First get the clean model:

$$CUDA \underline{} VISIBLE\underline{}DEVICES=indices\underline{} of \underline{}GPUs\qquad python3 \: kl8\underline{}unsupervised\underline{}clean\underline{}baseline1000epochcifar10.py \,configs/small/cifar10/simclr\underline{}rcrop\underline{}unsupervised.py (with epoch set to 1000 and weight_decay 1e-5)$$

Then get the robust model:

$$CUDA_VISIBLE_DEVICES=indices of GPUs python3 kl8_unsupervised_stage3try256dimtryotherthresanotherbaseline0solodeaclteststrongweakawpnewcifar10fourbnv2newawpscheme.py configs/small/cifar10/simclr_rcrop_unsupervised.py (with epoch set to 100 and weight_decay 5e-4, also remove the proj_head layer from the models)$$

## To finetune and test the model:

python3 finetuning_SLF_kl8_onebnsolonormalizeinputnetworkcifar10newforconferencewithaa2.py --ckpt checkpoints_of_models
