# Code for DAQ-SDP: Self-Supervised Adversarial Training via Diverse Augmented Queries and Self-Supervised Double Perturbation in NeurIPS 2024.

(https://neruips.cc/virtual/2024/poster/96153)

## To train the model (multi-GPUs):

First get the clean model:

$$CUDA \underline{} VISIBLE\underline{}DEVICES=indices\underline{} of \underline{}GPUs\quad python3 \quad kl8\underline{}unsupervised\underline{}clean\underline{}baseline1000epochcifar10.py \quad configs\text{/}small\text{/}cifar10\text{/}simclr\underline{}rcrop\underline{}unsupervised.py$$ 
(with epoch set to 1000 and weight_decay 1e-5)

Then get the robust model:

$$CUDA\underline{}VISIBLE\underline{}DEVICES=indices \underline{}of \underline{}GPUs \quad python3 \quad kl8\underline{}unsupervised\underline{}stage3try256dimtryotherthresanotherbaseline0solodeaclteststrongweakawpnewcifar10fourbnv2newawpscheme.py  \quad configs\text{/}small\text{/}cifar10\text{/}simclr\underline{}rcrop\underline{}unsupervised.py$$
(with epoch set to 100 and weight_decay 5e-4, also remove the proj_head layer from the models)

## To finetune and test the model:

$$python3\quad finetuning\underline{}SLF\underline{} kl8\underline{}onebnsolonormalizeinputnetworkcifar10newforconferencewithaa2.py \quad\text{--}ckpt \quad checkpoints\underline{}of\underline{}models$$
