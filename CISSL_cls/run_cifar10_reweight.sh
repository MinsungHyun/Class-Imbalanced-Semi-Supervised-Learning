#!/usr/bin/env bash

# cifar10 / nlabels 4000 / imbalance factor 100 / reweight focal loss
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --imb-factor 100 --reweight 'focal' --alg supervised
## cifar10 / imbalance factor 100 / unlabel imbalance 'same'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'focal' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'half'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'focal' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'uniform'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'focal' --alg MT --scl 0.5


# cifar10 / nlabels 4000 / imbalance factor 100 / reweight inverse normalization
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --imb-factor 100 --reweight 'inverse' --alg supervised
## cifar10 / imbalance factor 100 / unlabel imbalance 'same'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'inverse' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'half'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'inverse' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'uniform'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'inverse' --alg MT --scl 0.5


# cifar10 / nlabels 4000 / imbalance factor 100 / reweight cb loss
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --imb-factor 100 --reweight 'cls_bal' --alg supervised
## cifar10 / imbalance factor 100 / unlabel imbalance 'same'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'same' --reweight 'cls_bal' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'half'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'half' --reweight 'cls_bal' --alg MT --scl 0.5

## cifar10 / imbalance factor 100 / unlabel imbalance 'uniform'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset cifar10 --seed 0 --nlabels 4000 --nunlabels 7166 --imb-factor 100 --imb-unlabel 'uniform' --reweight 'cls_bal' --alg MT --scl 0.5