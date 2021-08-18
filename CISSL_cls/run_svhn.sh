#!/usr/bin/env bash

# svhn /  nlabels 1000 / imbalance factor 100
CUDA_VISIBLE_DEVICES=4 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --imb-factor 100 --alg supervised
## svhn /  nlabels 1000 / imbalance factor 100 / unlabel imbalance 'same'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg ICT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'same' --alg MT --scl 0.5

## svhn / nlabels 1000 /  imbalance factor 100 / unlabel imbalance 'half'
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg PI
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg MT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg PL
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg ICT
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=0 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'half' --alg MT --scl 0.5

## svhn /  nlabels 1000 / imbalance factor 100 / unlabel imbalance 'uniform'
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg PI
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg MT
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg PL
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg ICT
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg VAT --em 0.06
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg VAT --em 0.06 --sntg 0.4
CUDA_VISIBLE_DEVICES=2 python train_imbalance.py --dataset svhn --seed 0 --nlabels 1000 --nunlabels 15360 --imb-factor 100 --imb-unlabel 'uniform' --alg MT --scl 0.5

